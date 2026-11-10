#!/usr/bin/env python3
"""Fail closed on incomplete AMPA PyTorch Chrome traces.

Run this on an allocated compute node: a one-step Chrome trace can be large.
This gate checks exported events against independent callback call counts. It
does not turn CPU record-function ranges into GPU elapsed time.
"""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import re
import yaml

try:
    from analysis.aggregate_deep_trace import _collective_kernel, _device_scope_links, _resolve_forward_driver_metadata
except ModuleNotFoundError:
    from aggregate_deep_trace import _collective_kernel, _device_scope_links, _resolve_forward_driver_metadata


STEP_NAME = re.compile(r"^AMPA/STEP/(\d+)$")
LOSS_WARNING = re.compile(
    r"(?:lost|dropped|missing|incomplete) (?:\w+ )*(?:profiler |cuda |trace )?events|"
    r"Activity buffer request failed|Trace is incomplete",
    re.IGNORECASE,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_rank(run_dir: Path, rank: int, step: int, capture_steps: int = 1,
                 *, trace_path: Path | None = None, count_path: Path | None = None,
                 world_size: int = 8, cp: int = 2) -> dict:
    if world_size not in (8, 16) or cp not in (1, 2) or rank not in range(world_size):
        raise ValueError("unsupported all-rank EP trace topology")
    trace_path = trace_path or run_dir / "traces" / "torch_profile" / f"rank-{rank}.json.gz"
    count_path = count_path or run_dir / "raw" / f"deep_trace_rank-{rank}.json"
    if not trace_path.is_file() or trace_path.stat().st_size < 100_000:
        raise ValueError(f"missing or implausibly small Chrome trace: {trace_path}")
    if not count_path.is_file():
        raise ValueError(f"missing deep-trace call record: {count_path}")
    count_record = json.loads(count_path.read_text(encoding="utf-8"))
    expected_steps = list(range(step, step + capture_steps))
    records = count_record.get("steps", [])
    if count_record.get("rank") != rank or [s.get("step") for s in records] != expected_steps:
        raise ValueError(f"rank {rank}: call record does not describe exactly steps {expected_steps}")
    expected_counts: Counter[str] = Counter()
    for record in records:
        if record.get("layers") != list(range(48)):
            raise ValueError(f"rank {rank}: expected 48 observed MoE layers")
        expected_counts.update(record["counts"])

    with gzip.open(trace_path, "rt", encoding="utf-8") as stream:
        trace = json.load(stream)
    distributed = trace.get("distributedInfo", {})
    if distributed.get("rank") != rank or distributed.get("world_size") != world_size:
        raise ValueError(f"rank {rank}: Chrome trace has the wrong distributed rank/world size")
    groups = distributed.get("pg_config", [])
    if not any(group.get("pg_desc") == "EXPERT_MODEL_PARALLEL_GROUP"
               and group.get("pg_size") == world_size and group.get("ranks") == list(range(world_size))
               for group in groups):
        raise ValueError(f"rank {rank}: Chrome trace lacks the actual {world_size}-rank EP group")
    if not any(group.get("pg_desc") == "CONTEXT_PARALLEL_GROUP"
               and group.get("pg_size") == cp and len(group.get("ranks", [])) == cp
               and len(set(group.get("ranks", []))) == cp
               and rank in group.get("ranks", [])
               and set(group.get("ranks", [])).issubset(range(world_size))
               for group in groups):
        raise ValueError(f"rank {rank}: Chrome trace lacks the actual {cp}-rank CP group")
    events = trace.get("traceEvents")
    if not isinstance(events, list):
        raise ValueError(f"rank {rank}: no Chrome traceEvents array")
    events, driver_forward_recoveries = _resolve_forward_driver_metadata(
        events, distributed, tuple(expected_steps))
    # Kineto exports each record_function both as a host user_annotation and
    # often as a gpu_user_annotation. Count only the original host scope.
    names = Counter(event.get("name") for event in events
                    if event.get("ph") == "X" and event.get("cat") == "user_annotation")
    found_steps = {name: count for name, count in names.items() if isinstance(name, str) and STEP_NAME.fullmatch(name)}
    expected_scopes = {f"AMPA/STEP/{number}": 1 for number in expected_steps}
    if found_steps != expected_scopes:
        raise ValueError(f"rank {rank}: captured step scopes {found_steps}, expected {expected_scopes}")
    mismatches = {
        name: {"expected": expected, "observed": names.get(name, 0)}
        for name, expected in expected_counts.items() if names.get(name, 0) != expected
    }
    if mismatches:
        raise ValueError(f"rank {rank}: callback/Chrome label count mismatch: {str(mismatches)[:1000]}")
    if not any(name.endswith("/residual_bda") for name in expected_counts):
        raise ValueError(f"rank {rank}: missing end-of-MoE BDA labels")

    runtime_correlations = {
        event.get("args", {}).get("correlation")
        for event in events if event.get("cat") == "cuda_runtime"
    }
    runtime_correlations.discard(None)
    device_events = [
        event for event in events
        if event.get("cat") in {"kernel", "gpu_memcpy", "gpu_memset"}
        and event.get("ph") == "X" and event.get("dur", 0) > 0
    ]
    linked = sum(event.get("args", {}).get("correlation") in runtime_correlations for event in device_events)
    device_links, link_stats = _device_scope_links(
        events, tuple(expected_steps), driver_forward_recoveries=driver_forward_recoveries)
    kernels_by_correlation = Counter(event.get("args", {}).get("correlation") for event in device_events
                                     if event.get("cat") == "kernel")
    eligible_launches = [event for event in events if event.get("ph") == "X"
                        and event.get("cat") in {"cuda_runtime", "cuda_driver"}
                        and "LaunchKernel" in event.get("name", "")]
    missing_kernels = [event for event in eligible_launches
                       if event.get("args", {}).get("correlation") is None
                       or not kernels_by_correlation[event["args"]["correlation"]]]
    if missing_kernels:
        raise ValueError(f"rank {rank}: {len(missing_kernels)} eligible kernel launches lack positive device activity")
    expected_backward = sum(value for name, value in expected_counts.items() if name.endswith("_a2a"))
    if (link_stats.get("backward_ep_alltoall_cpu_nodes", 0) != expected_backward
            or link_stats.get("backward_ep_alltoall_nodes_with_correlated_nccl", 0) != expected_backward
            or link_stats.get("backward_ep_alltoall_kernel_count", 0) != expected_backward):
        raise ValueError(f"rank {rank}: backward EP AllToAll CPU-node/kernel coverage differs from {expected_backward} forward payload calls")
    if cp == 1 and (link_stats.get("untagged_cp_kv_p2p_by_owner") or any(
            event.get("ph") == "X" and (
                event.get("name") in {"AttnFuncWithCPAndKVP2P", "AttnFuncWithCPAndKVP2PBackward"}
                or (event.get("cat") == "kernel" and
                    event.get("args", {}).get("Process Group Description") == "CONTEXT_PARALLEL_GROUP"))
            for event in events)):
        raise ValueError(f"rank {rank}: CP1 trace contains CP KV peer communication")
    any_linked = link_stats.get("linked_by_any_route", 0)
    if not device_events or any_linked / len(device_events) < 0.95:
        raise ValueError(
            f"rank {rank}: only {any_linked}/{len(device_events)} positive CUDA activities "
            "link to a runtime launch, unique exact-ID CPU operation, or validated driver-forward EP chain"
        )
    total_activity = link_stats.get("positive_device_activity_sum_us", 0)
    unlinked_activity = link_stats.get("unlinked_device_activity_sum_us", 0)
    selected_activity = link_stats.get("selected_step_device_activity_sum_us", 0)
    if total_activity <= 0 or (total_activity - unlinked_activity) / total_activity < 0.95:
        raise ValueError(
            f"rank {rank}: only {total_activity - unlinked_activity}/{total_activity} us of "
            "positive CUDA activity links to a runtime launch, unique exact-ID CPU operation, or validated driver-forward EP chain"
        )
    selected_device = link_stats.get("device_events_launched_inside_selected_step", 0)
    staged_device = link_stats.get("selected_step_events_inside_labeled_stage", 0)
    unstaged_device = link_stats.get("selected_step_events_unassigned_to_stage", 0)
    if selected_device != staged_device + unstaged_device:
        raise ValueError(f"rank {rank}: selected-step stage/unknown activity accounting failed")
    if selected_device / len(device_events) < 0.95:
        raise ValueError(
            f"rank {rank}: only {selected_device}/{len(device_events)} device activities "
            "have a host launch inside selected optimizer step"
        )
    if selected_activity / total_activity < 0.95:
        raise ValueError(
            f"rank {rank}: only {selected_activity}/{total_activity} us of positive CUDA activity "
            "has a host launch inside the selected optimizer step"
        )
    step_events = [event for event in events if event.get("cat") == "user_annotation"
                   and event.get("name") in expected_scopes]
    if any(event.get("dur", 0) <= 0 for event in step_events):
        raise ValueError(f"rank {rank}: nonpositive optimizer compute/update scope")
    a2a_names = {name for name in expected_counts if name.endswith("_a2a")}
    a2a_instances = {
        id(event) for event in events
        if event.get("ph") == "X" and event.get("cat") == "user_annotation"
        and event.get("name") in a2a_names
    }
    a2a_nccl = Counter()
    for event in device_events:
        if not _collective_kernel(event):
            continue
        if (event.get("args", {}).get("Process Group Description") == "EXPERT_MODEL_PARALLEL_GROUP"
                and any(scope["name"] == "AMPA/TRAIN/microbatch_backward" for scope in device_links.get(id(event), []))):
            backward_ranks = event["args"].get("Process Group Ranks")
            if isinstance(backward_ranks, str):
                try:
                    backward_ranks = json.loads(backward_ranks)
                except json.JSONDecodeError:
                    pass
            if backward_ranks != list(range(world_size)):
                raise ValueError(f"rank {rank}: backward EP AllToAll rank membership differs")
        for scope in device_links.get(id(event), []):
            if scope["name"] in a2a_names:
                metadata = event.get("args", {})
                if metadata.get("Process Group Description") != "EXPERT_MODEL_PARALLEL_GROUP":
                    raise ValueError(
                        f"rank {rank}: NCCL activity in {scope['name']} lacks EP process-group metadata"
                    )
                ranks_value = metadata.get("Process Group Ranks")
                if isinstance(ranks_value, str):
                    try:
                        ranks_value = json.loads(ranks_value)
                    except json.JSONDecodeError:
                        pass
                if ranks_value != list(range(world_size)):
                    raise ValueError(f"rank {rank}: EP AllToAll NCCL rank membership differs from runtime EP group")
                collective = re.sub(r"[^a-z]", "", str(metadata.get("Collective name", "")).lower())
                if collective not in {"alltoall", "alltoallv"}:
                    raise ValueError(f"rank {rank}: NCCL activity in {scope['name']} is not AllToAll")
                a2a_nccl[id(scope)] += 1
    missing_a2a = sorted(id_ for id_ in a2a_instances if not a2a_nccl[id_])
    if missing_a2a:
        raise ValueError(f"rank {rank}: no correlated NCCL activity in {len(missing_a2a)} individual EP AllToAll calls")
    step_launches = set()
    for event in events:
        if event.get("cat") != "cuda_runtime" or event.get("ph") != "X":
            continue
        if any(
            event.get("pid") == scope.get("pid") and event.get("tid") == scope.get("tid")
            and scope["ts"] <= event["ts"] < scope["ts"] + scope["dur"]
            for scope in step_events
        ):
            step_launches.add(event.get("args", {}).get("correlation"))
    step_device_events = sum(event.get("args", {}).get("correlation") in step_launches for event in device_events)
    if not step_device_events:
        raise ValueError(f"rank {rank}: no CUDA activity linked to the selected optimizer-step scope")
    rank_log = run_dir / "logs" / f"rank-{rank}.out"
    if rank_log.is_file():
        warning = LOSS_WARNING.search(rank_log.read_text(encoding="utf-8", errors="replace"))
        if warning:
            raise ValueError(f"rank {rank}: trace-loss warning in rank log: {warning.group(0)}")
    return {
        "rank": rank,
        "trace_path": str(trace_path),
        "trace_sha256": _sha256(trace_path),
        "call_record_path": str(count_path),
        "call_record_sha256": _sha256(count_path),
        "compressed_bytes": trace_path.stat().st_size,
        "eligible_kernel_launches_with_positive_device_activity": len(eligible_launches),
        "steps": expected_steps,
        "step_scope_us": {event["name"]: event["dur"] for event in step_events},
        "labeled_forward_events": sum(expected_counts.values()),
        "positive_device_events": len(device_events),
        "device_events_linked_to_runtime": linked,
        "runtime_link_fraction": linked / len(device_events),
        "device_link_stats": link_stats,
        "device_link_fraction": any_linked / len(device_events),
        "selected_step_device_fraction": selected_device / len(device_events),
        "selected_step_device_activity_fraction": selected_activity / total_activity,
        "linked_device_activity_fraction": (total_activity - unlinked_activity) / total_activity,
        "selected_step_stage_assignment_fraction": staged_device / selected_device,
        "step_device_events": step_device_events,
        "ep_a2a_calls_with_correlated_nccl": len(a2a_nccl),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--ranks", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--step", type=int, default=5)
    parser.add_argument("--capture-steps", type=int, default=1)
    args = parser.parse_args()
    if args.capture_steps not in range(1, 4):
        parser.error("capture steps must be 1–3")
    ranks = tuple(int(item) for item in args.ranks.split(","))
    if not ranks or len(set(ranks)) != len(ranks) or any(rank not in range(8) for rank in ranks):
        parser.error("ranks must be a nonempty, unique subset of 0–7")
    observation = json.loads((args.run_dir / "config" / "observation_mode.json").read_text(encoding="utf-8"))
    if (observation.get("torch_trace") is not True
            or observation.get("lightweight_profiling") != "off"
            or observation.get("nsys_trace") is not False
            or observation.get("data_mode") != "indexed"):
        raise SystemExit("Run observation mode does not match a trace-only indexed training run")
    resolved = yaml.safe_load((args.run_dir / "config" / "resolved.yaml").read_text(encoding="utf-8"))
    profiler = resolved.get("profiling", {})
    if (profiler.get("use_pytorch_profiler") is not True
            or profiler.get("profile_step_start") != args.step + 1
            or profiler.get("profile_step_end") != args.step + 1 + args.capture_steps
            or profiler.get("profile_ranks") != list(ranks)):
        raise SystemExit("Resolved Bridge profiler schedule/ranks differ from requested trace")
    from compare_profile_overhead import _environment
    _environment(args.run_dir, cp=2)
    contract = json.loads((args.run_dir / "config" / "comparison_contract.json").read_text(encoding="utf-8"))
    if (contract.get("parallelism", {}).get("cp") != 2
            or contract.get("parallelism", {}).get("ep") != 8
            or contract.get("training", {}).get("warmup_steps") != 5
            or contract.get("training", {}).get("measured_steps") != 20):
        raise SystemExit("Trace run lacks the reviewed CP2/EP8 5+20 workload contract")
    results = []
    errors = []
    for rank in ranks:
        try:
            results.append(inspect_rank(args.run_dir, rank, args.step, args.capture_steps))
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
            errors.append({"rank": rank, "error": str(error)})
    expected_cp_peer = {"AttnFuncWithCPAndKVP2P": 48 * 16,
                        "AttnFuncWithCPAndKVP2PBackward": 48 * 16 * 2}
    for result in results:
        if result["device_link_stats"].get("untagged_cp_kv_p2p_by_owner") != expected_cp_peer:
            errors.append({"rank": result["rank"],
                           "error": "untagged CP KV peer NCCL calls differ from 48 layers × GA16"})
    validation = {
        "schema_version": 1,
        "valid_capture": not errors and len(results) == len(ranks),
        "requested_ranks": list(ranks),
        "claim_scope": "all_rank_diagnostic" if ranks == tuple(range(8)) else "rank_subset_acquisition_only",
        "step": args.step,
        "capture_steps": args.capture_steps,
        "ranks": results,
        "errors": errors,
        "analysis_source_sha256": {
            "validate_torch_trace.py": _sha256(Path(__file__)),
            "aggregate_deep_trace.py": _sha256(Path(__file__).with_name("aggregate_deep_trace.py")),
        },
        "claim_limit": "Chrome CPU scopes identify launches; they are not GPU elapsed times or complete wall-clock step timing.",
    }
    output = args.run_dir / "analysis" / "torch_trace_validation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(validation, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"valid_capture": validation["valid_capture"], "ranks": len(results), "errors": errors}))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
