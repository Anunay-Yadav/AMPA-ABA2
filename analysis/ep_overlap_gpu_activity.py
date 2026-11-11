"""Correlate overlap trace launches and measure recorded GPU intersections.

This validates the known UCCL payload set. Other activity is reported with
its exact-link coverage. No nearby timestamp is used to infer a launch owner.
Run on allocated compute, because each Chrome trace is large.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import json
import math
from pathlib import Path
import re

from analysis.inspect_uccl_trace import CALL_PHASES, PAIR, PAYLOAD, scope_routes, sha256, union_us
from profiling.ampa_profile.overlap_trace import CONTRACT, OverlapMethodLabels


NODE = re.compile(r"^AMPA/OVERLAP/(FWD|BWD|WGRAD)/microbatch_(\d{2})/(?:layer_(\d{2})/)?(.+)$")
COMPUTE = re.compile(r"gemm|(?:^|[_:])mma(?:[_:<]|$)|attention|flash|fmha|cudnn.*attn", re.IGNORECASE)
COMPUTE_SLOTS = {"pre_dispatch_computation", "mlp", "pre_process", "post_process"}


def require(value, message):
    if not value:
        raise ValueError(message)


def merged(intervals):
    result = []
    for start, end in sorted(intervals):
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(end, result[-1][1]))
        else:
            result.append((start, end))
    return result


def intersection_us(left, right):
    left, right = merged(left), merged(right)
    i = j = 0
    total = 0.0
    while i < len(left) and j < len(right):
        total += max(0, min(left[i][1], right[j][1]) - max(left[i][0], right[j][0]))
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return total


def inspect_events(trace, record):
    require(record.get("kind") == "ampa_ep_overlap_calls" and record.get("capture_mode") == "torch"
            and record.get("skipped_iter") is False, "not a completed overlap Torch capture")
    distributed = trace.get("distributedInfo", {})
    require(distributed.get("rank") == record["rank"] and distributed.get("world_size") == 16,
            "distributed trace identity differs")
    events = [row for row in trace["traceEvents"] if row.get("ph") == "X"]
    for row in events:
        require(all(type(row.get(key)) in (int, float) and math.isfinite(row[key]) for key in ("ts", "dur"))
                and row["dur"] >= 0, "invalid trace timestamp or duration")
    annotations = [row for row in events if row.get("cat") == "user_annotation"
                   and row.get("name", "").startswith("AMPA/")]
    backend = record["observation_contract"]["communication_backend"]
    require(backend in ("uccl_ep", "alltoall")
            and record["observation_contract"].get("contract") == CONTRACT,
            "unknown overlap observation contract")
    counts = record["counts"]
    expected = dict(OverlapMethodLabels(None, [], communication_backend=backend).expected_counts(4))
    require(counts == expected and record["observation_contract"]["expected_counts"] == expected,
            "observer differs from independently derived schedule counts")
    observed = Counter(row["name"] for row in annotations if row["name"] in counts)
    require(observed == Counter(counts), "trace differs from executed schedule counts")
    allowed_extra = {f"AMPA/STEP/{record['step']}", "AMPA/TRAIN/routing_audit"}
    require(all(row["name"] in counts or row["name"] in allowed_extra or PAIR.fullmatch(row["name"])
                for row in annotations), "trace contains unexpected AMPA observation scopes")
    step_scopes = [row for row in annotations if row["name"] == f"AMPA/STEP/{record['step']}"]
    require(len(step_scopes) == 1, "missing unique optimizer-step scope")
    for row in annotations:
        require(row.get("pid") == step_scopes[0].get("pid")
                and step_scopes[0]["ts"] <= row["ts"]
                and row["ts"] + row["dur"] <= step_scopes[0]["ts"] + step_scopes[0]["dur"],
                "observer scope lies outside its optimizer step")
    routes = scope_routes(events, annotations)
    runtime, driver, external = defaultdict(list), defaultdict(list), defaultdict(list)
    for row in events:
        args = row.get("args", {})
        if row.get("cat") == "cuda_runtime" and "correlation" in args:
            runtime[args["correlation"]].append(row)
        if row.get("cat") == "cuda_driver" and "LaunchKernel" in row.get("name", "") and "correlation" in args:
            driver[args["correlation"]].append(row)
        if row.get("cat") in ("cpu_op", "user_annotation") and "External id" in args:
            external[args["External id"]].append(row)
    pair_records = record["uccl_autograd_pairs"]
    pairs = {row["call_id"]: row for row in pair_records}
    pair_scopes = Counter((int(match[1]), match[2]) for row in annotations
                          if (match := PAIR.fullmatch(row["name"])))
    expected_tokens = {(call, direction) for call in pairs for direction in ("FWD", "BWD")}
    if backend == "uccl_ep":
        require(len(pairs) == len(pair_records) == 384
                and all(type(row.get(key)) is int for row in pair_records for key in ("call_id", "layer"))
                and [row.get("call_id") for row in pair_records] == list(range(384))
                and Counter((row.get("layer"), row.get("family")) for row in pair_records)
                == Counter({(layer, family): 4 for layer in range(48) for family in ("dispatch", "combine")})
                and all(type(row.get(key)) is int and row[key] > 0 for row in pair_records for key in ("epoch", "context_id"))
                and len({row["epoch"] for row in pair_records}) == 1
                and all(row.get("forward_returned") is True and row.get("backward_returned") is True for row in pair_records)
                and set(pair_scopes) == expected_tokens and all(value == 1 for value in pair_scopes.values()),
                "UCCL pair scopes are not bijective with all forward/adjoint calls")
        for scope in annotations:
            if match := PAIR.fullmatch(scope["name"]):
                call, direction = int(match[1]), match[2]
                pair = pairs[call]
                phase = f"AMPA/{direction}/layer_{pair['layer']:02d}/{CALL_PHASES[pair['family'], direction]}"
                require(phase in {parent["name"] for parent in routes.get(id(scope), [])},
                        "UCCL call has incorrect exact layer/phase ancestry")
    else:
        require(not pair_records and not pair_scopes, "native trace contains UCCL call identities")
    stats, payload_counts = Counter(), Counter()
    payload, compute, examples, selected_devices = [], [], [], set()
    native_forward_payload = []
    for row in events:
        if row.get("cat") != "kernel" or row["dur"] <= 0:
            continue
        stats["positive_kernels"] += 1
        args, candidates = row.get("args", {}), []
        for route, index, key in (("runtime", runtime, args.get("correlation")),
                                  ("driver", driver, args.get("correlation")),
                                  ("external", external, args.get("External id"))):
            if key is None:
                continue
            owners = index.get(key, [])
            require(len(owners) <= 1, f"ambiguous exact {route} identity")
            if owners:
                candidates.append(routes.get(id(owners[0]), []))
                stats[f"{route}_linked"] += 1
        if not candidates:
            stats["unlinked_kernels"] += 1
        scopes = max(candidates, key=len) if candidates else []
        tokens = [{(int(m[1]), m[2]) for scope in group if (m := PAIR.fullmatch(scope["name"]))}
                  for group in candidates]
        concrete = [token for token in tokens if token]
        require(not concrete or (all(len(token) == 1 for token in concrete)
                                 and all(token == concrete[0] for token in concrete)),
                "exact launch routes disagree on UCCL call identity")
        interval = (row["ts"], row["ts"] + row["dur"])
        match = PAYLOAD.search(row.get("name", ""))
        if match:
            require(backend == "uccl_ep" and concrete, "UCCL payload lacks its exact call owner")
            token = next(iter(concrete[0]))
            require(token in expected_tokens, "payload owner is outside the observed call grid")
            pair = pairs[token[0]]
            operation = "dispatch" if (pair["family"] == "dispatch") == (token[1] == "FWD") else "combine"
            require(match[1] == operation, "payload operation differs from its recorded adjoint")
            payload_counts[token] += 1
            payload.append(interval)
            require(type(args.get("device")) is int and args["device"] >= 0, "payload GPU identity unavailable")
            selected_devices.add(args["device"])
        elif "nccl" in row.get("name", "").lower() and backend == "alltoall":
            # Only exact forward EP payload scopes qualify here. Backward
            # process-group attribution requires its separate autograd proof.
            concrete_native = [{s["name"] for s in group if re.fullmatch(
                r"AMPA/FWD/layer_\d{2}/(?:dispatch_activations|dispatch_probabilities|combine_expert_output)_a2a", s["name"])}
                for group in candidates]
            concrete_native = [value for value in concrete_native if value]
            require(not concrete_native or (all(len(value) == 1 for value in concrete_native)
                                             and all(value == concrete_native[0] for value in concrete_native)),
                    "exact launch routes disagree on native EP payload owner")
            if concrete_native:
                native_forward_payload.append(interval)
                require(type(args.get("device")) is int and args["device"] >= 0, "payload GPU identity unavailable")
                selected_devices.add(args["device"])
        elif COMPUTE.search(row.get("name", "")):
            node_sets = [{scope["name"] for scope in group if NODE.fullmatch(scope["name"])} for group in candidates]
            concrete_nodes = [value for value in node_sets if value]
            require(not concrete_nodes or all(value == concrete_nodes[0] for value in concrete_nodes),
                    "exact launch routes disagree on computation node")
            if concrete_nodes and all(NODE.fullmatch(name)[4] in COMPUTE_SLOTS for name in concrete_nodes[0]):
                compute.append(interval)
                require(type(args.get("device")) is int and args["device"] >= 0, "compute GPU identity unavailable")
                selected_devices.add(args["device"])
                if len(examples) < 4:
                    examples.append({"name": row["name"], "scope": sorted(concrete_nodes[0]),
                                     "ts_us": row["ts"], "duration_us": row["dur"],
                                     "stream": args.get("stream")})
    if backend == "uccl_ep":
        require(all(payload_counts[token] >= 1 for token in expected_tokens),
                "some UCCL forward/adjoint calls lack positive correlated payload activity")
    else:
        payload = native_forward_payload
    require(payload and compute, "required correlated communication/computation activity is unavailable")
    require(len(selected_devices) == 1, "selected payload/compute kernels must share one GPU per rank")
    overlap = intersection_us(payload, compute)
    return {"schema_version": 1, "rank": record["rank"], "step": record["step"],
        "communication_backend": backend, "valid_known_payload_witness": backend == "uccl_ep",
        "native_forward_only": backend == "alltoall", "whole_session_event_completeness": None,
        "device": next(iter(selected_devices)),
        "known_payload_kernels": len(payload), "correlated_compute_kernels": len(compute),
        "device_link_stats": dict(stats), "payload_union_ms": union_us(payload) / 1000,
        "compute_union_ms": union_us(compute) / 1000, "intersection_ms": overlap / 1000,
        "observed_positive_gpu_concurrency": overlap > 0, "compute_examples": examples,
        "claim_limits": ["Intersections describe recorded GPU activity, not exposed latency or free SM capacity.",
                         "Computation selection covers named GEMM/attention kernels with exact node ancestry.",
                         "Autograd worker-thread computation without exact node ancestry remains outside this selection.",
                         "Other kernels and unrecorded work are excluded. No whole-session completeness claim.",
                         "Native activity covers exact forward payload scope attribution only."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bindings = {str(path): sha256(path) for path in (args.trace, args.record)}
    program_sha = sha256(Path(__file__))
    with gzip.open(args.trace, "rt", encoding="utf-8") as stream:
        trace = json.load(stream)
    result = inspect_events(trace, json.loads(args.record.read_text()))
    require(bindings == {str(path): sha256(path) for path in (args.trace, args.record)}
            and sha256(Path(__file__)) == program_sha, "analysis inputs or program changed")
    result["inputs"] = bindings
    result["program_inputs"] = {str(Path(__file__).resolve()): program_sha}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
