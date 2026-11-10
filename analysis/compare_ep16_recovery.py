#!/usr/bin/env python3
"""Report the explicit EP16 recovery without weakening the original comparator."""

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import yaml

from compare_deep_trace_overhead import _compare_consumed_batches, _mode, _training_resolved
from compare_profile_overhead import _environment, _physical_placement, _sha256
from extract_step_timing import select_timer_log, timing_rows
from summarize_training_evidence import summarize
from analyze_all_step_trace import _workload
from ampa_profile.provenance import validate_rank_placement, validate_runtime_groups, _ampa_runtime_sources
from tools.ep16_recovery_placement import require_same_placement


ENVIRONMENT_FIELDS = (
    "pytorch_version", "cuda_version", "nccl_version", "cuda_allocator_conf",
    "container_sqsh_sha256", "megatron_lm", "megatron_bridge", "ampa_runtime_sources",
    "checkpoint", "dataset_blend_per_split", "parallelism", "activation_recomputation",
)


def same_runtime(a, b, *, posthoc=None):
    """Allow recorded allocation changes only with identical actual runtime state."""
    for record in (a, b):
        if not record.get("slurm_job_id"):
            raise ValueError("recovery requires a recorded allocation ID")
        validate_rank_placement(record["rank_placement"], expected_world_size=16)
        validate_runtime_groups(record["rank_placement"], expected_world_size=16, cp=1)
    if _physical_placement(a) != _physical_placement(b):
        raise ValueError("recovery physical GPU placement differs")
    placements = [sorted(record["rank_placement"], key=lambda row: row["rank"]) for record in (a, b)]
    if placements[0] != placements[1]:
        raise ValueError("recovery actual ownership, process groups or device visibility differs")
    for field in ENVIRONMENT_FIELDS:
        if a.get(field) is None or a[field] != b.get(field):
            raise ValueError(f"recovery environment differs: {field}")
    if a.get("reviewed_inputs_proof") != b.get("reviewed_inputs_proof"):
        raise ValueError("reviewed input proof differs between matched runs")
    if posthoc is not None:
        if a["ampa_runtime_sources"] != posthoc["original_runtime_sources"]:
            raise ValueError("posthoc comparison substituted the original training source identity")
    else:
        if a["ampa_runtime_sources"] != _ampa_runtime_sources(Path(__file__).resolve().parent.parent):
            raise ValueError("frozen runtime source inventory changed or is incomplete")
        for relative, digest in a["ampa_runtime_sources"]["files"].items():
            if _sha256(Path(__file__).resolve().parent.parent / relative) != digest:
                raise ValueError(f"frozen runtime source changed: {relative}")


def checked_audit_binding(summary, audit_path):
    if (Path(summary.get("routing_audit_path", "")).resolve() != audit_path.resolve()
            or summary.get("routing_audit_sha256") != _sha256(audit_path)):
        raise ValueError("routing audit is missing its analysis binding or changed after validation")


def checked_audit(run, rank, environment, grouped):
    """Independently reconcile each complete route histogram and locality count."""
    record = _read(run / "raw" / f"routing_audit_rank-{rank}.json")
    if (record.get("valid") is not True or record.get("rank") != rank
            or record.get("num_groups") != (4 if grouped else None)
            or record.get("group_topk") != (2 if grouped else None)
            or [row.get("step") for row in record.get("steps", [])] != list(range(25))):
        raise ValueError("incomplete or incorrect routing audit")
    placements = {row["rank"]: row for row in environment["rank_placement"]}
    owners = {expert: row["rank"] for row in placements.values() for expert in row["local_expert_indices"]}
    group_hosts = []
    for group in range(4):
        ranks = {owners[expert] for expert in range(group * 32, (group + 1) * 32)}
        hosts = {placements[owner]["hostname"] for owner in ranks}
        if len(ranks) != 4 or len(hosts) != 1:
            raise ValueError("expert group does not cover exactly one four-GPU node")
        group_hosts.extend(hosts)
    if len(set(group_hosts)) != 4:
        raise ValueError("four expert groups do not span four distinct nodes")
    for step in record["steps"]:
        if [row.get("layer") for row in step.get("layers", [])] != list(range(48)):
            raise ValueError("routing audit lacks 48 layers for each step")
        for layer in step["layers"]:
            counts, traffic = layer.get("expert_counts"), layer.get("rank_traffic")
            if (not isinstance(counts, list) or len(counts) != 128
                    or not isinstance(traffic, list) or len(traffic) != 16
                    or any(type(value) is not int or value < 0 for value in counts + traffic)
                    or any(value > 16384 for value in counts)):
                raise ValueError("routing histogram has invalid expert/rank counts")
            rebuilt = [0] * 16
            for expert, count in enumerate(counts):
                rebuilt[owners[expert]] += count
            local = traffic[rank]
            intra = sum(value for owner, value in enumerate(traffic)
                        if owner != rank and placements[owner]["hostname"] == placements[rank]["hostname"])
            if (rebuilt != traffic or layer.get("routed_tokens") != 16384
                    or layer.get("assignments") != 131072 or sum(counts) != 131072
                    or layer.get("local") != local or layer.get("intra_node") != intra
                    or layer.get("inter_node") != sum(traffic) - local - intra):
                raise ValueError("routing assignment, owner traffic or locality invariant failed")
    return record


def route_difference(a, b):
    """Count histogram changes, without claiming per-token route equivalence."""
    differences = Counter()
    for left_step, right_step in zip(a["steps"], b["steps"]):
        for left, right in zip(left_step["layers"], right_step["layers"]):
            changed = left != right
            differences["compared_rank_step_layers"] += 1
            differences["changed_rank_step_layers"] += int(changed)
            for key in ("expert_counts", "rank_traffic"):
                differences[f"{key}_l1"] += sum(abs(x - y) for x, y in zip(left[key], right[key]))
            for key in ("local", "intra_node", "inter_node"):
                differences[f"trace_minus_control_{key}"] += right[key] - left[key]
    return differences


def checked_allocations(control, grouped_control, trace, grouped_trace, *, same_allocation):
    ids = [record.get("slurm_job_id") for record in (control, grouped_control, trace, grouped_trace)]
    if any(not isinstance(job, str) or not job for job in ids):
        raise ValueError("comparison requires four recorded allocation IDs")
    if ids[0] != ids[1] or ids[0] != ids[3]:
        raise ValueError("both controls and grouped capture must share one allocation")
    if (ids[0] == ids[2]) != same_allocation:
        raise ValueError("ordinary capture allocation differs from the declared comparison mode")


def _read(path):
    return json.loads(path.read_text())


def compare_pair(control, trace, *, analysis_dir=None, posthoc=None):
    if (analysis_dir is not None) != (posthoc is not None):
        raise ValueError("alternate analysis roots require an explicit validated posthoc proof")
    _mode(control, False)
    _mode(trace, True)
    for name in ("comparison_contract.json", "c0_manifest.json", "dataset_manifest.json", "model_manifest.json"):
        if _sha256(control / "config" / name) != _sha256(trace / "config" / name):
            raise ValueError(f"matched pair identity differs: {name}")
    contract = _read(control / "config" / "comparison_contract.json")
    world_size, cp, accumulation = _workload(contract)
    if contract["training"].get("completed_steps", 25) != 25:
        raise ValueError("a two-step memory pilot cannot be a full profiling comparison")
    a = _environment(control, cp=cp, world_size=world_size, ep=world_size)
    b = _environment(trace, cp=cp, world_size=world_size, ep=world_size)
    if (world_size, cp, accumulation) != (16, 1, 4):
        raise ValueError("this recovery is restricted to CP1/EP16")
    same_runtime(a, b, posthoc=posthoc)
    if _training_resolved(control) != _training_resolved(trace):
        raise ValueError("matched resolved training differs")
    batches = _compare_consumed_batches(control, trace, world_size=world_size, accumulation=accumulation)
    elapsed, health = {}, {}
    for name, run in (("control", control), ("trace", trace)):
        health[name] = summarize(run, steps=25, interval=1)
        rows = timing_rows(select_timer_log(run / "logs"), warmup_steps=5,
                           measured_steps=20, log_interval=1, tokens_per_step=64 * 4096)
        elapsed[name] = {row["first_step"]: row["mean_optimizer_step_ms"] for row in rows}
    phases = {}
    routing = Counter()
    export_seconds = []
    stop_seconds = []
    device_totals = Counter()
    device_link_counts = Counter()
    driver_recovery_provenance = []
    nccl_groups = Counter()
    cp_peer_activity = Counter()
    summaries = []
    parity = True
    differences = Counter()
    for rank in range(world_size):
        base = analysis_dir if analysis_dir is not None else trace / "analysis" / "all_steps"
        path = base / f"rank-{rank}" / "summary.json"
        record = _read(path)
        if posthoc is not None and record.get("posthoc_analysis") != posthoc["binding"]:
            raise ValueError("rank summary is not bound to this posthoc proof and allocation")
        if record["valid_capture"] is not True or record["rank"] != rank or record["steps"] != list(range(25)):
            raise ValueError("incomplete all-step trace coverage")
        if world_size == 16 and (
                record.get("world_size") != world_size or record.get("cp") != cp
                or record.get("gradient_accumulation") != accumulation):
            raise ValueError("all-step summary has the wrong runtime topology")
        if (len(record["validation"]) != 25 or any(
                row.get("rank") != rank or row.get("steps") != [step]
                for step, row in enumerate(record["validation"]))):
            raise ValueError("rank summary lacks one validation for each actual step")
        if set(record["analysis_source_sha256"]) != {
                "analyze_all_step_trace.py", "validate_torch_trace.py", "aggregate_deep_trace.py"}:
            raise ValueError("rank summary lacks required analysis source identities")
        for name, digest in record["analysis_source_sha256"].items():
            if _sha256(Path(__file__).with_name(name)) != digest:
                raise ValueError("analysis changed after validation")
        for validation in record["validation"]:
            step = validation["steps"][0]
            expected_paths = {
                "trace_path": trace / "traces" / "torch_profile" / f"step-{step:05d}" / f"rank-{rank}.json.gz",
                "call_record_path": trace / "raw" / "deep_trace" / f"step-{step:05d}" / f"rank-{rank}.json",
            }
            for key, digest_key in (("trace_path", "trace_sha256"), ("call_record_path", "call_record_sha256")):
                if Path(validation[key]).resolve() != expected_paths[key].resolve():
                    raise ValueError("validation references a different run, rank or step")
                if _sha256(Path(validation[key])) != validation[digest_key]:
                    raise ValueError("validated trace bytes changed")
                if posthoc is not None and posthoc["original_evidence_sha256"].get(
                        str(expected_paths[key].resolve())) != validation[digest_key]:
                    raise ValueError("validated trace/call bytes differ from the posthoc input snapshot")
            calls = _read(Path(validation["call_record_path"]))
            export_seconds.append(calls["steps"][0]["export_seconds"])
            stop_seconds.append(calls["steps"][0]["profiler_stop_seconds"])
            for key in ("positive_device_activity_sum_us", "selected_step_device_activity_sum_us",
                        "unlinked_device_activity_sum_us", "selected_step_unassigned_activity_sum_us",
                        "selected_step_nccl_activity_sum_us", "selected_step_unassigned_nccl_sum_us",
                        "driver_forward_recovered_activity_sum_us"):
                device_totals[key] += validation["device_link_stats"].get(key, 0)
            for key in ("positive_device_events", "runtime_linked", "external_cpu_op_linked",
                        "linked_by_either_route", "linked_by_any_route", "driver_forward_recovered_device_events"):
                device_link_counts[key] += validation["device_link_stats"].get(key, 0)
            driver_recovery_provenance.extend({"rank": rank, "step": step, "provenance": item}
                for item in validation["device_link_stats"].get("driver_forward_recovery_provenance", []))
            nccl_groups.update(validation["device_link_stats"].get("selected_step_nccl_by_group_us", {}))
            cp_peer_activity.update(validation["device_link_stats"].get("untagged_cp_kv_p2p_activity_us", {}))
        summaries.append({"rank": rank, "path": str(path), "sha256": _sha256(path)})
        rebuilt = {}
        if [row.get("step") for row in record["step_summaries"]] != list(range(25)):
            raise ValueError("rank summary lacks 25 step summary identities")
        for row in record["step_summaries"]:
            summary_path = Path(row["path"])
            expected_path = path.parent / f"step-{row['step']:05d}.json"
            if summary_path.resolve() != expected_path.resolve():
                raise ValueError("step aggregation references a different run, rank or step")
            if _sha256(summary_path) != row["sha256"]:
                raise ValueError("step aggregation changed after validation")
            summary = _read(summary_path)
            if summary["rank"] != rank or set(summary["step_scopes_ms"]) != {f"AMPA/STEP/{row['step']}"}:
                raise ValueError("step aggregation has wrong rank or step")
            for phase_row in summary["phase_rows"]:
                values = rebuilt.setdefault(phase_row["phase"], Counter())
                for key in ("calls", "cpu_scope_sum_ms", "cpu_exclusive_sum_ms",
                            "gpu_exclusive_activity_sum_ms", "gpu_exclusive_nccl_sum_ms"):
                    values[key] += phase_row[key]
        if rebuilt != record["phase_totals"]:
            raise ValueError("pooled phase totals differ from hashed step aggregations")
        for phase, values in rebuilt.items():
            phases.setdefault(phase, Counter()).update(values)
        grouped = contract["routing"]["num_groups"] == 4
        audit_path = trace / "raw" / f"routing_audit_rank-{rank}.json"
        checked_audit_binding(record, audit_path)
        audit = checked_audit(trace, rank, b, grouped)
        control_audit = checked_audit(control, rank, a, grouped)
        parity = parity and audit == control_audit
        differences.update(route_difference(control_audit, audit))
        for step in audit["steps"]:
            for layer in step["layers"]:
                routing.update({key: layer[key] for key in ("assignments", "local", "intra_node", "inter_node")})
    forward = {phase: values["gpu_exclusive_activity_sum_ms"] for phase, values in phases.items()
               if not phase.startswith("train_")}
    total = sum(forward.values())
    if total <= 0:
        raise ValueError("no forward CUDA activity")
    return {"control_run": str(control), "trace_run": str(trace), "valid_descriptive_comparison": True,
            "valid_strict_overhead_comparison": parity,
            "pure_profiling_overhead": {"available": False,
                "reason": "Sequential runs cannot isolate temporal system drift or prove per-token route equality."},
            "observed_profiling_overhead_estimate": {"available": parity,
                "observed_increase_percent": ((statistics.fmean(elapsed["trace"].values()) /
                    statistics.fmean(elapsed["control"].values()) - 1) * 100) if parity else None,
                "limit": "Exact histogram parity is necessary but does not prove per-token identical routes. Sequential drift remains."},
            "route_histograms_exactly_equal": parity, "route_histogram_difference": differences,
            "allocation_ids": {"control": a["slurm_job_id"], "trace": b["slurm_job_id"]},
            "same_physical_rank_placement": True, "training_health": health,
            "steps": list(range(25)), "ranks": list(range(world_size)), "consumed_batches": batches,
            "world_size": world_size, "cp": cp, "gradient_accumulation": accumulation,
            "phase_totals": phases, "forward_gpu_activity_sum_ms": total,
            "forward_activity_scope": "CUDA activity assigned exclusively to named MoE forward stages; broad forward and unknown activity excluded",
            "forward_percentages": {key: value * 100 / total for key, value in forward.items()},
            "routing": routing, "elapsed_steps_ms": elapsed,
            "postwarmup_control_mean_ms": statistics.fmean(elapsed["control"].values()),
            "postwarmup_trace_mean_ms": statistics.fmean(elapsed["trace"].values()),
            "mean_rank_export_seconds": statistics.fmean(export_seconds),
            "mean_rank_profiler_stop_seconds": statistics.fmean(stop_seconds),
            "device_activity_us": device_totals, "nccl_by_process_group_us": nccl_groups,
            "device_link_counts": device_link_counts,
            "driver_forward_recovery_provenance": driver_recovery_provenance,
            "cp_peer_activity_us": cp_peer_activity, "rank_summaries": summaries}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normal-control", type=Path, required=True)
    parser.add_argument("--normal-trace", type=Path, required=True)
    parser.add_argument("--grouped-control", type=Path, required=True)
    parser.add_argument("--grouped-trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recovery-proof", type=Path)
    parser.add_argument("--same-allocation", action="store_true")
    parser.add_argument("--reviewed-acquisition", action="store_true")
    parser.add_argument("--orchestration-sources", type=Path)
    parser.add_argument("--posthoc-proof", type=Path)
    parser.add_argument("--normal-analysis-dir", type=Path)
    parser.add_argument("--grouped-analysis-dir", type=Path)
    args = parser.parse_args()
    if args.reviewed_acquisition and not args.same_allocation:
        parser.error("--reviewed-acquisition requires --same-allocation")
    alternate_analysis = args.normal_analysis_dir is not None or args.grouped_analysis_dir is not None
    if args.posthoc_proof is not None:
        if (not args.same_allocation or not args.reviewed_acquisition or args.recovery_proof is not None
                or args.normal_analysis_dir is None or args.grouped_analysis_dir is None):
            parser.error("--posthoc-proof requires reviewed same-allocation training and both analysis directories")
    elif alternate_analysis:
        parser.error("alternate analysis directories require --posthoc-proof")
    grouped_name = "group4-top2"
    runs = {"normal": {"control": args.normal_control.resolve(), "torchprof-all": args.normal_trace.resolve()},
            grouped_name: {"control": args.grouped_control.resolve(), "torchprof-all": args.grouped_trace.resolve()}}
    if len({path for pair in runs.values() for path in pair.values()}) != 4:
        raise ValueError("recovery requires four distinct full runs")
    posthoc = None
    analysis_dirs = {}
    if args.posthoc_proof is not None:
        from tools.ep16_posthoc_proof import validate_posthoc_proof

        proof_runs = {route: {"control": str(pair["control"]), "trace": str(pair["torchprof-all"])}
                      for route, pair in runs.items()}
        analysis_dirs = {"normal": str(args.normal_analysis_dir.resolve()),
                         grouped_name: str(args.grouped_analysis_dir.resolve())}
        posthoc = validate_posthoc_proof(args.posthoc_proof, Path(__file__).resolve().parent.parent,
                                        runs=proof_runs, analysis_dirs=analysis_dirs)
        if args.orchestration_sources is None:
            args.orchestration_sources = Path(posthoc["original_orchestration_sources"]["path"])
    normal, grouped = runs["normal"], runs[grouped_name]
    a = _read(normal["control"] / "config" / "comparison_contract.json")
    b = _read(grouped["control"] / "config" / "comparison_contract.json")
    world_size, cp, accumulation = _workload(a)
    if (world_size, cp, accumulation) != (16, 1, 4) or _workload(b) != (16, 1, 4):
        raise ValueError("requested run names differ from the actual workload")
    _compare_consumed_batches(normal["control"], grouped["control"], world_size=world_size,
                              accumulation=accumulation)
    if a.pop("routing") != {"topk": 8, "num_groups": None, "group_topk": None}:
        raise ValueError("normal routing contract mismatch")
    if b.pop("routing") != {"topk": 8, "num_groups": 4,
                             "group_topk": 2} or a != b:
        raise ValueError("grouped comparison changed more than routing")
    environment_a = _environment(normal["control"], cp=cp, world_size=world_size, ep=world_size)
    environment_b = _environment(grouped["control"], cp=cp, world_size=world_size, ep=world_size)
    same_runtime(environment_a, environment_b, posthoc=posthoc)
    if environment_a["slurm_job_id"] != environment_b["slurm_job_id"]:
        raise ValueError("both fresh controls must share the recovery allocation")
    old_environment = _environment(normal["torchprof-all"], cp=1, world_size=16, ep=16)
    fresh_environment = _environment(grouped["torchprof-all"], cp=1, world_size=16, ep=16)
    checked_allocations(environment_a, environment_b, old_environment, fresh_environment,
                        same_allocation=args.same_allocation)
    proof = None
    if args.same_allocation:
        if (old_environment.get("slurm_job_id") != environment_a["slurm_job_id"]
                or fresh_environment.get("slurm_job_id") != environment_a["slurm_job_id"]):
            raise ValueError("fresh comparison requires all four runs in one recorded allocation")
        if args.orchestration_sources is None:
            parser.error("--same-allocation requires --orchestration-sources")
        identities = {}
        for line in args.orchestration_sources.read_text().splitlines():
            digest, source = line.split(maxsplit=1)
            path = Path(source).resolve()
            if path in identities or (posthoc is None and _sha256(path) != digest):
                raise ValueError("fresh orchestration or analysis changed after its source snapshot")
            identities[path] = digest
        wrapper = "ep16_all_steps_reviewed.sbatch" if args.reviewed_acquisition else "ep16_all_steps_fresh.sbatch"
        expected = {Path(__file__).resolve(), Path(__file__).resolve().parent.parent / "jobs/training" / wrapper}
        if set(identities) != expected:
            raise ValueError("fresh orchestration source snapshot is incomplete")
        if posthoc is not None and (
                str(args.orchestration_sources.resolve()) != posthoc["original_orchestration_sources"]["path"]
                or _sha256(args.orchestration_sources) != posthoc["original_orchestration_sources"]["sha256"]
                or {str(path): digest for path, digest in identities.items()}
                != posthoc["original_orchestration_source_sha256"]):
            raise ValueError("posthoc comparison replaced the original orchestration snapshot")
        if args.reviewed_acquisition:
            input_path = args.orchestration_sources.resolve().parent / "reviewed_inputs.json"
            proof_identity = environment_a.get("reviewed_inputs_proof")
            if (not isinstance(proof_identity, dict)
                    or Path(proof_identity.get("path", "")).resolve() != input_path
                    or proof_identity.get("sha256") != _sha256(input_path)):
                raise ValueError("reviewed input proof changed after training provenance recorded it")
            inputs = _read(input_path)
            if (inputs.get("valid_reviewed_inputs") is not True
                    or inputs.get("slurm_job_id") != environment_a["slurm_job_id"]
                    or inputs.get("ampa_runtime_sources") != environment_a["ampa_runtime_sources"]
                    or inputs.get("c0_identity_manifest_sha256") != a["c0_identity_manifest_sha256"]
                    or inputs.get("dataset_manifest_sha256") != a["dataset"]["manifest_sha256"]
                    or inputs.get("checkpoint_payload", {}).get("verified") is not True):
                raise ValueError("reviewed input proof has the wrong allocation, source or input identity")
            archived_sources = input_path.parent / "source/runtime"
            for relative, expected_digest in inputs["ampa_runtime_sources"]["files"].items():
                if _sha256(archived_sources / relative) != expected_digest:
                    raise ValueError("reviewed source archive differs from training identity")
    else:
        if (not old_environment.get("slurm_job_id")
                or old_environment["slurm_job_id"] == environment_a["slurm_job_id"]
                or fresh_environment.get("slurm_job_id") != environment_a["slurm_job_id"]):
            raise ValueError("recovery requires a distinct original allocation and one fresh allocation")
        if args.recovery_proof is None:
            parser.error("cross-allocation recovery requires --recovery-proof")
        proof = _read(args.recovery_proof)
    if proof is not None and (proof.get("valid_physical_placement") is not True
            or proof.get("slurm_job_id") != environment_a["slurm_job_id"]
            or proof.get("reference_slurm_job_id") != old_environment["slurm_job_id"]
            or Path(proof["reference_run"]).resolve() != normal["torchprof-all"]
            or proof.get("reference_environment_sha256") != _sha256(normal["torchprof-all"] / "environment.json")
            or proof["frozen_inputs"]["ampa_runtime_sources"] != old_environment["ampa_runtime_sources"]):
        raise ValueError("recovery placement proof does not bind to these allocations and reference")
    if proof is not None:
        require_same_placement(environment_a["rank_placement"], proof["rank_placement"])
    required_recovery_sources = {"jobs/training/ep16_all_steps_recovery.sbatch",
                                 "tools/ep16_recovery_placement.py", "analysis/compare_ep16_recovery.py"}
    if proof is not None and set(proof["recovery_source_sha256"]) != required_recovery_sources:
        raise ValueError("recovery proof lacks complete recovery code identity")
    for relative, digest in (proof["recovery_source_sha256"] if proof is not None else {}).items():
        if _sha256(Path(__file__).resolve().parent.parent / relative) != digest:
            raise ValueError(f"recovery code changed after placement proof: {relative}")
    for name in ("c0_manifest.json", "dataset_manifest.json", "model_manifest.json"):
        if _sha256(normal["control"] / "config" / name) != _sha256(grouped["control"] / "config" / name):
            raise ValueError(f"ordinary and grouped input identity differs: {name}")
    resolved = []
    for run, groups, selected in ((normal["control"], None, None),
                                  (grouped["control"], 4, 2)):
        configuration = yaml.safe_load(_training_resolved(run))
        model = configuration["model"]
        if model.pop("moe_router_num_groups") != groups or model.pop("moe_router_group_topk") != selected:
            raise ValueError("resolved grouping differs from requested counts")
        resolved.append(configuration)
    if resolved[0] != resolved[1]:
        raise ValueError("resolved models differ beyond the two grouped-routing fields")
    result = {routing: compare_pair(paths["control"], paths["torchprof-all"],
                                   analysis_dir=Path(analysis_dirs[routing]) if posthoc is not None else None,
                                   posthoc=posthoc) for routing, paths in runs.items()}
    result["recovery"] = {
        "mode": "posthoc_same_training_allocation" if posthoc is not None else "fresh_same_allocation" if args.same_allocation else "reused_ordinary_cross_allocation",
        "original_allocation_id": old_environment["slurm_job_id"],
        "recovery_allocation_id": environment_a["slurm_job_id"],
        "cross_allocation_pair": None if args.same_allocation else "normal", "same_actual_rank_gpu_and_runtime": True,
        "complete_captures_valid": True,
        "strict_overhead_available": all(result[name]["valid_strict_overhead_comparison"] for name in runs),
        "analysis_sha256": _sha256(Path(__file__)),
        "reviewed_inputs": {"path": str(input_path), "sha256": _sha256(input_path)} if args.reviewed_acquisition else None,
        "placement_proof": {"path": str(args.recovery_proof.resolve()), "sha256": _sha256(args.recovery_proof)} if proof is not None else None,
        "recovery_source_sha256": proof["recovery_source_sha256"] if proof is not None else {str(path): digest for path, digest in identities.items()},
        "orchestration_sources": {"path": str(args.orchestration_sources.resolve()), "sha256": _sha256(args.orchestration_sources)} if args.same_allocation else None,
        "original_comparator_sha256": _sha256(Path(__file__).with_name("compare_all_step_profiles.py")),
        "control_grouped_vs_normal_percent": (result[grouped_name]["postwarmup_control_mean_ms"] /
            result["normal"]["postwarmup_control_mean_ms"] - 1) * 100,
        "limit": "All four runs share one allocation. Sequential time, thermal or external traffic drift remains." if args.same_allocation else "Recovered sequential comparison across two allocations. Same physical GPUs do not remove time, thermal or external traffic drift.",
    }
    if posthoc is not None:
        result["recovery"]["posthoc_analysis"] = {"path": posthoc["binding"]["proof_path"],
                                                "sha256": posthoc["binding"]["proof_sha256"]}
        result["recovery"]["initial_acquisition_failure"] = posthoc["original_suite_failure"]
        result["recovery"]["strict_comparator_ran"] = False
        result["recovery"]["analysis_allocation_id"] = posthoc["posthoc_slurm_job_id"]
        result["recovery"]["posthoc_runtime_sources"] = posthoc["posthoc_runtime_sources"]
        result["recovery"]["analysis_dirs"] = analysis_dirs
        result["recovery"]["limit"] += (" The original suite stopped during grouped capture validation with status 143. "
                                        "Its strict comparator did not run. This separately bound posthoc analysis "
                                        "does not replace the original training proof or failed validation.")
    result["claim_limit"] = ("Phase sums measure overlapping CUDA activity, not elapsed step partitions. "
                             "Every profiled iteration includes stop/export overhead. Controls include identical "
                             "routing-audit kernels and CPU batch hashing. Sequential drift limits causal speed claims. "
                             "Broad backward uses exact autograd host ownership; ambiguous activity stays unknown. "
                             "Coverage checks reconcile recorded kernel launches and all expected forward/backward EP payload calls; "
                             "they do not prove capture of every memcpy, memset or jointly missing host/device math event. "
                             "Fine backward expert-stage, physical traffic and exposed communication costs remain unavailable. "
                             "Trace capture ends before audit readback and log/export overhead, so it does not span the full timer interval. "
                             "Unequal ON/OFF route histograms prevent a pure profiling-overhead claim. "
                             "Forward assignment totals do not measure backward traffic or node-deduplicated bytes.")
    destination = args.output.resolve()
    if args.output.is_symlink():
        raise ValueError("comparison output is a symlink")
    if posthoc is not None:
        root = Path(__file__).resolve().parent.parent
        if (root / "runs" not in destination.parents or destination.exists()
                or not Path(posthoc["binding"]["proof_path"]).parent in destination.parents):
            raise ValueError("posthoc comparison must use a new directory beneath its proof output")
        validate_posthoc_proof(args.posthoc_proof, root, runs=proof_runs, analysis_dirs=analysis_dirs)
    destination.mkdir(exist_ok=False)
    (destination / "comparison.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    lines = ["# Qwen ordinary and grouped routing across all 25 steps", "",
             f"Both captures cover {world_size} ranks and 25 complete optimizer steps, including five warmup steps. CP={cp}, EP={world_size}, gradient accumulation={accumulation}.", "",
             f"All four runs share allocation {environment_a['slurm_job_id']}. Actual rank-to-GPU placement and runtime identities are equal." if args.same_allocation else f"The ordinary capture uses allocation {old_environment['slurm_job_id']}. The controls and fresh grouped capture use allocation {environment_a['slurm_job_id']}. Actual rank-to-GPU placement and runtime identities are equal.", "",
             f"Strict profiling-overhead availability: ordinary={result['normal']['valid_strict_overhead_comparison']}, grouped={result[grouped_name]['valid_strict_overhead_comparison']}. Unequal route histograms leave both valid captures available for descriptive analysis.", "",
             "| Named MoE forward stage | Ordinary assigned activity (%) | Grouped assigned activity (%) |",
             "| --- | ---: | ---: |"]
    if posthoc is not None:
        lines[4:4] = ["Posthoc analysis uses allocation " + posthoc["posthoc_slurm_job_id"]
                      + ". The original acquisition suite stopped during grouped capture validation with status 143. "
                      "Its strict comparator did not run. Original training evidence and failed outputs remain preserved.", ""]
    for phase in sorted(set(result["normal"]["forward_percentages"]) | set(result[grouped_name]["forward_percentages"])):
        lines.append(f"| {phase} | {result['normal']['forward_percentages'].get(phase, 0):.3f} | {result[grouped_name]['forward_percentages'].get(phase, 0):.3f} |")
    lines.extend(["", result["claim_limit"], "", "| Measure | Ordinary | Grouped |", "| --- | ---: | ---: |"])
    for field in ("postwarmup_control_mean_ms", "postwarmup_trace_mean_ms", "mean_rank_export_seconds"):
        lines.append(f"| {field} | {result['normal'][field]:.3f} | {result[grouped_name][field]:.3f} |")
    for field in ("assignments", "local", "intra_node", "inter_node"):
        lines.append(f"| Trace forward logical {field} | {result['normal']['routing'][field]} | {result[grouped_name]['routing'][field]} |")
    lines.extend(["", "The following activity sums cover forward, backward, gradient finalization, optimizer update, and the routing audit.",
                  "", "| Exclusive launch bucket | Ordinary CUDA activity (s) | Grouped CUDA activity (s) |",
                  "| --- | ---: | ---: |"])
    for phase in sorted(set(result["normal"]["phase_totals"]) | set(result[grouped_name]["phase_totals"])):
        ordinary = result["normal"]["phase_totals"].get(phase, {}).get("gpu_exclusive_activity_sum_ms", 0) / 1000
        grouped_value = result[grouped_name]["phase_totals"].get(phase, {}).get("gpu_exclusive_activity_sum_ms", 0) / 1000
        lines.append(f"| {phase} | {ordinary:.3f} | {grouped_value:.3f} |")
    lines.extend(["", "CUDA activity outside the named buckets remains explicit in comparison.json. CPU scopes and overlapping activity sums are separate measurements."])
    (destination / "comparison.md").write_text("\n".join(lines) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(16, 8))
    for axis, routing in zip(axes, ("normal", grouped_name)):
        values = result[routing]["forward_percentages"]
        positive = {key: value for key, value in values.items() if value >= 0.5}
        remainder = sum(value for key, value in values.items() if key not in positive)
        if remainder:
            positive["Other named MoE forward stages"] = remainder
        axis.pie(list(positive.values()), labels=list(positive), autopct="%.1f%%", textprops={"fontsize": 8})
        axis.set_title(routing)
    figure.suptitle(f"CUDA activity assigned to named MoE forward stages, 25 steps and {world_size} ranks\nBroad forward and unknown activity excluded. Activity can overlap; percentages do not partition elapsed time.")
    figure.tight_layout()
    figure.savefig(destination / "forward_activity_comparison.png", dpi=180)
    figure.savefig(destination / "forward_activity_comparison.svg")
    plt.close(figure)
    phases = sorted(set(result["normal"]["phase_totals"]) | set(result[grouped_name]["phase_totals"]))
    phases.append("outside_named_buckets")
    figure, axis = plt.subplots(figsize=(14, 8))
    for offset, routing in ((-0.2, "normal"), (0.2, grouped_name)):
        values = [result[routing]["phase_totals"].get(phase, {}).get("gpu_exclusive_activity_sum_ms", 0)
                  for phase in phases[:-1]]
        values.append(result[routing]["device_activity_us"]["selected_step_unassigned_activity_sum_us"] / 1000)
        axis.barh([index + offset for index in range(len(phases))],
                  [value / (16 * 25) for value in values], height=0.4, label=routing)
    axis.set_yticks(range(len(phases)), phases)
    axis.set_xlabel("Mean exclusive CUDA activity per rank per optimizer step (ms)")
    axis.set_title("All 25 updates: forward, backward, gradient handling and optimizer\nActivity can overlap and does not partition elapsed time")
    axis.legend()
    figure.tight_layout()
    figure.savefig(destination / "whole_step_cuda_activity.png", dpi=180)
    figure.savefig(destination / "whole_step_cuda_activity.svg")
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(12, 6))
    for routing in ("normal", grouped_name):
        for leg in ("control", "trace"):
            timings = result[routing]["elapsed_steps_ms"][leg]
            axis.plot(list(timings), [value / 1000 for value in timings.values()], label=f"{routing} {leg}")
    axis.set_xlabel("Optimizer step (five warmup updates excluded)")
    axis.set_ylabel("Logged complete optimizer step time (seconds)")
    axis.set_title("Controls and per-step profiling runs on the same physical GPUs")
    axis.legend()
    figure.tight_layout()
    figure.savefig(destination / "optimizer_step_timing.png", dpi=180)
    figure.savefig(destination / "optimizer_step_timing.svg")
    plt.close(figure)
    if posthoc is not None:
        validate_posthoc_proof(args.posthoc_proof, Path(__file__).resolve().parent.parent,
                               runs=proof_runs, analysis_dirs=analysis_dirs)
    print(json.dumps({"complete_captures_valid": True,
                      "strict_overhead_available": result["recovery"]["strict_overhead_available"],
                      "output": str(destination)}))


if __name__ == "__main__":
    main()
