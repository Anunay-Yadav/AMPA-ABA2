#!/usr/bin/env python3
"""Compare matched ordinary/grouped experiments without merging time domains."""

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


def _read(path):
    return json.loads(path.read_text())


def compare_pair(control, trace):
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
    if not a["slurm_job_id"] or a["slurm_job_id"] != b["slurm_job_id"] or _physical_placement(a) != _physical_placement(b):
        raise ValueError("matched pair allocation or physical GPUs differ")
    for field in ("pytorch_version", "cuda_version", "nccl_version", "cuda_allocator_conf",
                  "container_sqsh_sha256", "megatron_lm", "megatron_bridge", "ampa_runtime_sources",
                  "checkpoint", "dataset_blend_per_split", "parallelism", "activation_recomputation"):
        if a.get(field) is None or a[field] != b.get(field):
            raise ValueError(f"matched pair environment differs: {field}")
    if _training_resolved(control) != _training_resolved(trace):
        raise ValueError("matched resolved training differs")
    batches = _compare_consumed_batches(control, trace, world_size=world_size, accumulation=accumulation)
    elapsed = {}
    for name, run in (("control", control), ("trace", trace)):
        summarize(run, steps=25, interval=1)
        rows = timing_rows(select_timer_log(run / "logs"), warmup_steps=5,
                           measured_steps=20, log_interval=1, tokens_per_step=64 * 4096)
        elapsed[name] = {row["first_step"]: row["mean_optimizer_step_ms"] for row in rows}
    phases = {}
    routing = Counter()
    export_seconds = []
    stop_seconds = []
    device_totals = Counter()
    nccl_groups = Counter()
    cp_peer_activity = Counter()
    summaries = []
    for rank in range(world_size):
        path = trace / "analysis" / "all_steps" / f"rank-{rank}" / "summary.json"
        record = _read(path)
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
            for key, digest_key in (("trace_path", "trace_sha256"), ("call_record_path", "call_record_sha256")):
                if _sha256(Path(validation[key])) != validation[digest_key]:
                    raise ValueError("validated trace bytes changed")
            calls = _read(Path(validation["call_record_path"]))
            export_seconds.append(calls["steps"][0]["export_seconds"])
            stop_seconds.append(calls["steps"][0]["profiler_stop_seconds"])
            for key in ("positive_device_activity_sum_us", "selected_step_device_activity_sum_us",
                        "unlinked_device_activity_sum_us", "selected_step_unassigned_activity_sum_us",
                        "selected_step_nccl_activity_sum_us", "selected_step_unassigned_nccl_sum_us"):
                device_totals[key] += validation["device_link_stats"].get(key, 0)
            nccl_groups.update(validation["device_link_stats"].get("selected_step_nccl_by_group_us", {}))
            cp_peer_activity.update(validation["device_link_stats"].get("untagged_cp_kv_p2p_activity_us", {}))
        summaries.append({"rank": rank, "path": str(path), "sha256": _sha256(path)})
        rebuilt = {}
        if [row.get("step") for row in record["step_summaries"]] != list(range(25)):
            raise ValueError("rank summary lacks 25 step summary identities")
        for row in record["step_summaries"]:
            summary_path = Path(row["path"])
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
        audit = _read(trace / "raw" / f"routing_audit_rank-{rank}.json")
        control_audit = _read(control / "raw" / f"routing_audit_rank-{rank}.json")
        if audit != control_audit:
            raise ValueError("matched pair actual route counts differ")
        for step in audit["steps"]:
            for layer in step["layers"]:
                routing.update({key: layer[key] for key in ("assignments", "local", "intra_node", "inter_node")})
    forward = {phase: values["gpu_exclusive_activity_sum_ms"] for phase, values in phases.items()
               if not phase.startswith("train_")}
    total = sum(forward.values())
    if total <= 0:
        raise ValueError("no forward CUDA activity")
    return {"control_run": str(control), "trace_run": str(trace), "valid_comparison": True,
            "steps": list(range(25)), "ranks": list(range(world_size)), "consumed_batches": batches,
            "world_size": world_size, "cp": cp, "gradient_accumulation": accumulation,
            "phase_totals": phases, "forward_gpu_activity_sum_ms": total,
            "forward_percentages": {key: value * 100 / total for key, value in forward.items()},
            "routing": routing, "elapsed_steps_ms": elapsed,
            "postwarmup_control_mean_ms": statistics.fmean(elapsed["control"].values()),
            "postwarmup_trace_mean_ms": statistics.fmean(elapsed["trace"].values()),
            "mean_rank_export_seconds": statistics.fmean(export_seconds),
            "mean_rank_profiler_stop_seconds": statistics.fmean(stop_seconds),
            "device_activity_us": device_totals, "nccl_by_process_group_us": nccl_groups,
            "cp_peer_activity_us": cp_peer_activity, "rank_summaries": summaries}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--pair-id", required=True)
    parser.add_argument("--cp", type=int, choices=(1, 2), default=2)
    parser.add_argument("--router-num-groups", type=int, default=8)
    parser.add_argument("--router-group-topk", type=int, default=2)
    args = parser.parse_args()
    if (args.router_num_groups <= 0 or 128 % args.router_num_groups
            or args.router_group_topk not in range(1, args.router_num_groups + 1)
            or 8 % args.router_group_topk
            or args.router_group_topk * (128 // args.router_num_groups) < 8):
        parser.error("invalid native grouped-routing counts")
    grouped_name = f"group{args.router_num_groups}-top{args.router_group_topk}"
    runs = {routing: {leg: args.runs_root / f"{args.pair_id}-cp{args.cp}-{routing}-{leg}"
                      for leg in ("control", "torchprof-all")} for routing in ("normal", grouped_name)}
    normal, grouped = runs["normal"], runs[grouped_name]
    a = _read(normal["control"] / "config" / "comparison_contract.json")
    b = _read(grouped["control"] / "config" / "comparison_contract.json")
    world_size, cp, accumulation = _workload(a)
    if cp != args.cp or _workload(b) != (world_size, cp, accumulation):
        raise ValueError("requested run names differ from the actual workload")
    _compare_consumed_batches(normal["control"], grouped["control"], world_size=world_size,
                              accumulation=accumulation)
    if a.pop("routing") != {"topk": 8, "num_groups": None, "group_topk": None}:
        raise ValueError("normal routing contract mismatch")
    if b.pop("routing") != {"topk": 8, "num_groups": args.router_num_groups,
                             "group_topk": args.router_group_topk} or a != b:
        raise ValueError("grouped comparison changed more than routing")
    environment_a = _environment(normal["control"], cp=cp, world_size=world_size, ep=world_size)
    environment_b = _environment(grouped["control"], cp=cp, world_size=world_size, ep=world_size)
    if (environment_a["slurm_job_id"] != environment_b["slurm_job_id"]
            or _physical_placement(environment_a) != _physical_placement(environment_b)):
        raise ValueError("ordinary and grouped routes used different allocations or GPUs")
    for field in ("pytorch_version", "cuda_version", "nccl_version", "cuda_allocator_conf",
                  "container_sqsh_sha256", "megatron_lm", "megatron_bridge", "ampa_runtime_sources",
                  "checkpoint", "dataset_blend_per_split", "parallelism", "activation_recomputation"):
        if environment_a.get(field) is None or environment_a[field] != environment_b.get(field):
            raise ValueError(f"ordinary and grouped environment differs: {field}")
    resolved = []
    for run, groups, selected in ((normal["control"], None, None),
                                  (grouped["control"], args.router_num_groups, args.router_group_topk)):
        configuration = yaml.safe_load(_training_resolved(run))
        model = configuration["model"]
        if model.pop("moe_router_num_groups") != groups or model.pop("moe_router_group_topk") != selected:
            raise ValueError("resolved grouping differs from requested counts")
        resolved.append(configuration)
    if resolved[0] != resolved[1]:
        raise ValueError("resolved models differ beyond the two grouped-routing fields")
    result = {routing: compare_pair(paths["control"], paths["torchprof-all"]) for routing, paths in runs.items()}
    result["claim_limit"] = ("Phase sums measure overlapping CUDA activity, not elapsed step partitions. "
                             "Every profiled iteration includes stop/export overhead. Controls include identical "
                             "routing-audit kernels and CPU batch hashing. Sequential drift limits causal speed claims. "
                             "Fine backward expert-stage and exposed communication costs remain unavailable.")
    destination = args.runs_root / f"{args.pair_id}-comparison"
    destination.mkdir(exist_ok=False)
    (destination / "comparison.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    lines = ["# Qwen ordinary and grouped routing across all 25 steps", "",
             f"Both captures cover {world_size} ranks and 25 complete optimizer steps, including five warmup steps. CP={cp}, EP={world_size}, gradient accumulation={accumulation}.", "",
             "| Stage | Ordinary forward activity (%) | Grouped forward activity (%) |",
             "| --- | ---: | ---: |"]
    for phase in sorted(set(result["normal"]["forward_percentages"]) | set(result[grouped_name]["forward_percentages"])):
        lines.append(f"| {phase} | {result['normal']['forward_percentages'].get(phase, 0):.3f} | {result[grouped_name]['forward_percentages'].get(phase, 0):.3f} |")
    lines.extend(["", result["claim_limit"], "", "| Measure | Ordinary | Grouped |", "| --- | ---: | ---: |"])
    for field in ("postwarmup_control_mean_ms", "postwarmup_trace_mean_ms", "mean_rank_export_seconds"):
        lines.append(f"| {field} | {result['normal'][field]:.3f} | {result[grouped_name][field]:.3f} |")
    for field in ("assignments", "local", "intra_node", "inter_node"):
        lines.append(f"| Logical {field} | {result['normal']['routing'][field]} | {result[grouped_name]['routing'][field]} |")
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
            positive["Other measured forward stages"] = remainder
        axis.pie(list(positive.values()), labels=list(positive), autopct="%.1f%%", textprops={"fontsize": 8})
        axis.set_title(routing)
    figure.suptitle(f"Forward CUDA activity pooled over 25 steps and {world_size} ranks\nActivity can overlap. Percentages do not partition elapsed step time.")
    figure.tight_layout()
    figure.savefig(destination / "forward_activity_comparison.png", dpi=180)
    figure.savefig(destination / "forward_activity_comparison.svg")
    plt.close(figure)
    print(json.dumps({"valid_comparison": True, "output": str(destination)}))


if __name__ == "__main__":
    main()
