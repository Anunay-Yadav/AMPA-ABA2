#!/usr/bin/env python3
"""Compare eight accepted EP16 runs without parsing or changing their traces.

Execute only in allocated compute. Complete-step timers describe performance.
Inclusive CUDA activity describes recorded work, never exposed step latency.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics

import yaml

from analysis import accept_uccl_pilot as evidence
from analysis.compare_deep_trace_overhead import _checked_fingerprint_batches
from analysis.extract_step_timing import select_timer_log, timing_rows
from analysis.inspect_uccl_trace import REQUIRED_MOE_PHASES
from analysis.summarize_training_evidence import summarize
from analysis.validate_ep16_received_routes import validate_run as validate_received_run
from profiling.ampa_profile.provenance import (
    _ampa_runtime_sources, _EP16_RUNTIME_TOPOLOGIES, validate_rank_placement, validate_runtime_groups,
)
from tools.uccl_ep_source_identity import verify_uccl_source


ROOT = Path(__file__).resolve().parents[1]
CASE_CONFIG = {
    "ordinary": {"label": "Native ordinary", "backend": "alltoall", "num_groups": None, "group_topk": None},
    "grouped": {"label": "Native grouped", "backend": "alltoall", "num_groups": 4, "group_topk": 2},
    "uccl_ordinary": {"label": "UCCL ordinary", "backend": "uccl_ep", "num_groups": None, "group_topk": None},
    "uccl": {"label": "UCCL grouped", "backend": "uccl_ep", "num_groups": 4, "group_topk": 2},
}
CASES = tuple(CASE_CONFIG)
BACKEND_PAIRS = (("ordinary", "uccl_ordinary"), ("grouped", "uccl"))
CONTRASTS = BACKEND_PAIRS + (("ordinary", "grouped"), ("uccl_ordinary", "uccl"))
LEGS = ("control", "profile")
STEPS = tuple(range(25))
MEASURED = tuple(range(5, 25))
TRAIN = {"microbatch_forward": 4, "microbatch_backward": 4, "gradient_finalize": 1, "optimizer_step": 1}
FORWARD = ("pre_mlp_norm", "moe_total", "router_total", "router_gate", "router_routing",
           "router_main_topk", "router_aux_scores", "router_aux_loss", "dispatch_preprocess",
           "dispatch_total", "local_expert_sort", "expert_total", "expert_fc1", "expert_fc2",
           "inverse_local_sort", "combine_total", "final_unpermute", "residual_bda")
COMPOSITES = {"dispatch_side": ("dispatch_preprocess", "dispatch_total", "local_expert_sort"),
              "combine_side": ("inverse_local_sort", "combine_total", "final_unpermute")}
FIELDS = ("gpu_inclusive_activity_ms", "cpu_scope_ms")
SOURCES = tuple(sorted(set(evidence.ANALYSIS_SOURCES) | {
    "analysis/compare_uccl_profiles.py", "tools/uccl_ep_validate_profiles.py",
    "profiling/ampa_profile/provenance.py", "profiling/ampa_profile/deep_trace.py",
    "profiling/ampa_profile/uccl_trace.py", "profiling/ampa_profile/routing_audit.py"}))
LIMITS = [
    "Complete-step performance uses unprofiled controls with batch hashing and routing/receive audits still enabled.",
    "Inclusive CPU scopes and concurrent GPU activity are not additive partitions of optimizer-step elapsed time.",
    "UCCL required membership/layout audit work differs from native audit work and remains in the measured setup.",
    "Native NCCL network provider/plugin selection was not recorded, so results describe the baseline as launched and do not establish performance against a tuned native transport.",
    "All eight runs are sequential. Time, thermal state, clocks and external contention can drift despite identical placement.",
    "Later routes and weights can differ. Equal marginal route histograms do not establish per-token route identity.",
    "ON/OFF differences do not isolate profiler overhead and cannot correct individual component durations.",
    "Trace capture ends before profiler stop/export, routing-audit readback and logging complete.",
    "Logical assignments and unique token destinations are not physical wire bytes or measured bandwidth.",
    "CPU proxy time, exposed communication latency, unrecorded persistent activity and fine backward expert costs are unavailable.",
    "Finite losses and gradients establish logged training health, not model-quality equivalence or every-parameter update proof.",
]
require, equal = evidence.require, evidence.equal


def number(value, label):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0, f"invalid nonnegative metric {label}")
    return float(value)


def describe(values):
    values = sorted(number(value, "observation") for value in values)
    require(bool(values), "empty observation population")
    position = (len(values) - 1) * .95
    low, high = math.floor(position), math.ceil(position)
    return {"count": len(values), "mean": statistics.fmean(values), "median": statistics.median(values),
            "min": values[0], "max": values[-1], "p95": values[low] + (values[high] - values[low]) * (position - low),
            "population_sd": statistics.pstdev(values)}


def change(base, case):
    return {"baseline": base, "case": case, "difference": case - base,
            "percent": 100 * (case / base - 1) if base else None}


def is_uccl(case):
    require(case in CASE_CONFIG, f"unknown comparison case {case}")
    return CASE_CONFIG[case]["backend"] == "uccl_ep"


def control_contrasts(timing):
    """Observed OFF-only ratios; profiled timings cannot enter these contrasts."""
    result = {}
    for base, case in CONTRASTS:
        first, second = timing[base]["control"], timing[case]["control"]
        result[f"{base}_to_{case}"] = {
            "baseline_case": base, "comparison_case": case, "leg": "control",
            "step_mean_ms": change(first["step_ms"]["mean"], second["step_ms"]["mean"]),
            "step_median_ms": change(first["step_ms"]["median"], second["step_ms"]["median"]),
            "observed_speed_ratio_mean": first["step_ms"]["mean"] / second["step_ms"]["mean"],
            "observed_speed_ratio_median": first["step_ms"]["median"] / second["step_ms"]["median"],
            "aggregate_tokens_per_second": change(first["aggregate_tokens_per_second"], second["aggregate_tokens_per_second"]),
        }
    return result


def observation_changes(timing, differences):
    return {case: {
        "raw_mean_step_change": change(timing[case]["control"]["step_ms"]["mean"], timing[case]["profile"]["step_ms"]["mean"]),
        "observed_overhead_estimate_percent": (100 * (timing[case]["profile"]["step_ms"]["mean"] / timing[case]["control"]["step_ms"]["mean"] - 1)
            if differences[f"{case}_on_off"]["route_histograms_exactly_equal"] else None),
        "pure_profiling_overhead_available": False,
    } for case in CASES}


def bound_text(path, inputs):
    digest = inputs.bind(path)
    payload = Path(path).read_bytes()
    equal(hashlib.sha256(payload).hexdigest(), digest, f"text read {path}")
    return payload.decode("utf-8")


def normalized_contract(contract, case):
    is_uccl(case)
    equal(contract["training"]["completed_steps"], 25, "full completed updates")
    equal(contract["launch_policy"], "torchrun_node4", "full launch policy")
    equal(contract["routing"]["topk"], 8, "top-eight routing")
    # Reuse the strict pilot workload checks. Only the explicitly checked
    # completed length differs; retain every other field for exact comparison.
    value = copy.deepcopy(contract)
    value["training"]["completed_steps"] = 2
    value = evidence._contract(value, case)
    value["training"]["completed_steps"] = 25
    return value


def normalized_resolved(configuration, case, run):
    is_uccl(case)
    equal(configuration["train"]["exit_interval"], None, "full exit interval")
    value = copy.deepcopy(configuration)
    value["train"]["exit_interval"] = 2
    value = evidence._resolved(value, case, run)
    value["train"]["exit_interval"] = None
    return value


def timing_summary(rows):
    equal([row["first_step"] for row in rows], list(MEASURED), "measured timer grid")
    values = [number(row["mean_optimizer_step_ms"], "complete step") for row in rows]
    require(min(values) > 0, "zero optimizer-step duration")
    equal([row["tokens_per_optimizer_step"] for row in rows], [262144] * 20, "timer token counts")
    return {"unit": "complete optimizer-step wall time (ms)", "step_ms": describe(values),
            "aggregate_tokens_per_second": 20 * 262144 / (sum(values) / 1000),
            "by_update": [{"step": row["first_step"], "optimizer_step_ms": value} for row, value in zip(rows, values)]}


def load_timing_health(run, inputs):
    for rank in range(16):
        inputs.bind(run / "logs" / f"rank-{rank}.out")
    health = inputs.read(run / "analysis/real_training_evidence.json")
    equal(health, summarize(run, steps=25, interval=1), "recomputed complete training health")
    equal(health["final_consumed_samples"], 1600, "completed consumed samples")
    equal(health["final_consumed_global_tokens"], 6553600, "completed training tokens")
    log = select_timer_log(run / "logs")
    rows = timing_rows(log, warmup_steps=5, measured_steps=20, log_interval=1, tokens_per_step=262144)
    result = timing_summary(rows)
    values = [row["mean_optimizer_step_ms"] for row in rows]
    expected = {"timing_unit": "complete optimizer-step wall-time averaged over each logged window", "window_count": 20,
                "mean_optimizer_step_ms": statistics.fmean(values), "median_optimizer_step_ms": statistics.median(values),
                "min_optimizer_step_ms": min(values), "max_optimizer_step_ms": max(values),
                "mean_tokens_per_second": statistics.fmean(row["tokens_per_second"] for row in rows),
                "note": "These are existing Megatron interval-timer windows; they are not isolated communication timings."}
    equal(inputs.read(run / "analysis/optimizer_step_breakdown.json"), expected, "recomputed timer summary")
    inputs.bind(run / "metrics/timing_windows.parquet")
    return result, health


def phase_rows(record, backend):
    """Normalize only common inclusive fields. Never use exclusive or union values."""
    rows = record["phases"] if backend == "uccl" else record["phase_rows"]
    by_name = {}
    for row in rows:
        name = row["phase"]
        require(isinstance(name, str) and name not in by_name, "duplicate phase identity")
        require(type(row["calls"]) is int and row["calls"] > 0, "invalid phase calls")
        gpu = row["gpu_inclusive_activity_sum_ms"] if backend == "uccl" else row["gpu_launched_activity_sum_ms"]
        by_name[name] = {"calls": row["calls"], "gpu_inclusive_activity_ms": number(gpu, name),
                         "cpu_scope_ms": number(row["cpu_scope_sum_ms"], name)}
    result = {}
    for phase in FORWARD:
        names = [f"AMPA/FWD/layer_{layer:02d}/{phase}" for layer in range(48)] if backend == "uccl" else [phase]
        require(all(name in by_name for name in names), f"missing common forward phase {phase}")
        for name in names:
            equal(by_name[name]["calls"], 4 if backend == "uccl" else 192, f"phase call grid {name}")
        result[phase] = {"calls": 192, **{field: sum(by_name[name][field] for name in names) for field in FIELDS}}
    for phase, calls in TRAIN.items():
        name = f"AMPA/TRAIN/{phase}" if backend == "uccl" else f"train_{phase}"
        require(name in by_name, f"missing common training phase {phase}")
        equal(by_name[name]["calls"], calls, f"training phase call grid {name}")
        result[f"train_{phase}"] = by_name[name]
    for phase, members in COMPOSITES.items():
        result[phase] = {"calls": 576, **{field: sum(result[name][field] for name in members) for field in FIELDS}}
    if backend == "uccl":
        for direction, phases in REQUIRED_MOE_PHASES.items():
            for layer in range(48):
                for phase in phases:
                    name = f"AMPA/{direction}/layer_{layer:02d}/{phase}"
                    require(name in by_name, f"missing UCCL phase grid {name}")
                    equal(by_name[name]["calls"], 4, f"UCCL phase grid {name}")
    else:
        layer_rows = record["layer_rows"]
        layers = {}
        for row in layer_rows:
            layer, phase = row["layer"], row["phase"]
            require(type(layer) is int and layer in range(48) and (layer, phase) not in layers, "duplicate/wrong native layer phase")
            equal(row["name"], f"AMPA/FWD/layer_{layer:02d}/{phase}", "native layer label")
            equal(row["calls"], 4, "native layer call grid")
            layers[layer, phase] = row
        for phase in FORWARD:
            require(all((layer, phase) in layers for layer in range(48)), f"missing native layer grid {phase}")
            for source, field in (("gpu_launched_activity_sum_ms", "gpu_inclusive_activity_ms"), ("cpu_scope_sum_ms", "cpu_scope_ms")):
                actual = sum(number(layers[layer, phase][source], source) for layer in range(48))
                require(math.isclose(actual, result[phase][field], rel_tol=1e-10, abs_tol=1e-6), "native layer-to-phase inclusive sums differ")
    return result


def profile_summary(records, costs):
    equal(sorted(records), [(rank, step) for rank in range(16) for step in STEPS], "complete profile rank-update grid")
    equal(sorted(costs), sorted(records), "profiler cost grid")
    names = set(records[0, 0])
    require(all(set(row) == names for row in records.values()), "common phase set changed within capture")
    phases = {}
    for name in sorted(names):
        per_update = [{"step": step,
                       "rank_mean_gpu_ms": statistics.fmean(records[rank, step][name][FIELDS[0]] for rank in range(16)),
                       "rank_max_gpu_ms": max(records[rank, step][name][FIELDS[0]] for rank in range(16))}
                      for step in MEASURED]
        gpu = describe(records[rank, step][name][FIELDS[0]] for rank in range(16) for step in MEASURED)
        cpu = describe(records[rank, step][name][FIELDS[1]] for rank in range(16) for step in MEASURED)
        calls = {records[rank, step][name]["calls"] for rank in range(16) for step in STEPS}
        require(len(calls) == 1, "phase call count changed within capture")
        phases[name] = {"scope_calls_per_rank_update": calls.pop(),
            "mean_gpu_inclusive_activity_ms_per_rank_update": gpu["mean"],
            "mean_cpu_scope_ms_per_rank_update": cpu["mean"], "rank_update_gpu_ms": gpu, "rank_update_cpu_ms": cpu,
            "update_rank_mean_gpu_ms": describe(row["rank_mean_gpu_ms"] for row in per_update),
            "update_rank_max_gpu_ms": describe(row["rank_max_gpu_ms"] for row in per_update), "per_update": per_update}
    return {"unit": "inclusive correlated GPU activity per rank optimizer update (ms)", "rank_update_observations": 320,
            "independent_experiment_repetitions": 1, "phases": phases,
            "profiler_cost": {field: {"unit": "per-rank host wall seconds", "rank_update": describe(costs[rank, step][field] for rank in range(16) for step in MEASURED),
                "update_rank_max": describe(max(costs[rank, step][field] for rank in range(16)) for step in MEASURED)}
                for field in ("profiler_stop_seconds", "export_seconds")}}


def load_profile(case, run, validation, inputs, root):
    records, costs = {}, {}
    uccl = is_uccl(case)
    native_sources = {name: inputs.bind(root / "analysis" / name) for name in (
        "analyze_all_step_trace.py", "validate_torch_trace.py", "aggregate_deep_trace.py")}
    for rank in range(16):
        summary = None
        if not uccl:
            summary = inputs.read(run / "analysis/all_steps" / f"rank-{rank}/summary.json")
            for key, value in {"valid_capture": True, "rank": rank, "steps": list(STEPS), "world_size": 16, "cp": 1, "gradient_accumulation": 4}.items():
                equal(summary.get(key), value, f"native capture {key}")
            equal(summary["analysis_source_sha256"], native_sources, "native analyzer sources")
            evidence._path(summary["routing_audit_path"], run / "raw" / f"routing_audit_rank-{rank}.json", "native audit path")
            inputs.bind(run / "raw" / f"routing_audit_rank-{rank}.json", summary["routing_audit_sha256"])
            equal(len(summary["validation"]), 25, "native validation grid")
            equal(len(summary["step_summaries"]), 25, "native summary grid")
        for step in STEPS:
            trace, call = evidence._trace_paths(run, rank, step)
            if uccl:
                record = inputs.read(validation / f"{case}-profile" / f"rank-{rank}-step-{step:05d}.json")
                for key, value in {"valid_uccl_trace_capture": True, "synthetic": False, "pair_count": 384, "paired_direction_count": 768}.items():
                    equal(record.get(key), value, f"UCCL capture {key}")
                equal(record["analysis_source_sha256"], inputs.bind(root / "analysis/inspect_uccl_trace.py"), "UCCL analyzer source")
                evidence._bound_map(record["inputs"], (trace, call), inputs, "UCCL actual raw capture")
            else:
                accepted = summary["validation"][step]
                equal(accepted["rank"], rank, "native validation rank")
                equal(accepted["steps"], [step], "native validation update")
                for path, prefix in ((trace, "trace"), (call, "call_record")):
                    evidence._path(accepted[f"{prefix}_path"], path, "native raw capture path")
                    inputs.bind(path, accepted[f"{prefix}_sha256"])
                binding = summary["step_summaries"][step]
                equal(binding["step"], step, "native aggregation update")
                path = run / "analysis/all_steps" / f"rank-{rank}" / f"step-{step:05d}.json"
                evidence._path(binding["path"], path, "native aggregation path")
                inputs.bind(path, binding["sha256"])
                record = inputs.read(path)
                equal(record["rank"], rank, "native aggregation rank")
                equal(sorted(record["step_scopes_ms"]), [f"AMPA/STEP/{step}"], "native aggregation step")
            journal = inputs.read(call)
            equal(journal["rank"], rank, "capture journal rank")
            equal(journal["schema_version"], 2 if uccl else 1, "capture journal schema")
            equal(len(journal["steps"]), 1, "capture journal coverage")
            row = journal["steps"][0]
            equal(row["step"], step, "capture journal update")
            equal(row["skipped_iter"], False, "capture journal skipped update")
            costs[rank, step] = {field: number(row[field], field) for field in ("profiler_stop_seconds", "export_seconds")}
            records[rank, step] = phase_rows(record, "uccl" if uccl else "native")
    return profile_summary(records, costs)


def load_spread(values):
    values = [number(value, "incoming assignments") for value in values]
    mean = statistics.fmean(values)
    require(mean > 0, "empty incoming workload")
    return {"cv": statistics.pstdev(values) / mean, "max_over_mean": max(values) / mean}


def routing_summary(audits, ownership, steps):
    hosts = ownership["rank_hosts"]
    nodes = sorted(set(hosts.values()))
    owners = ownership["expert_owners"]
    totals = {name: 0 for name in ("assignments", "local", "intra_node", "inter_node")}
    loads, ranks, node_copies, remote_copies, tokens = [], 0, 0, 0, 0
    dedup = audits[0]["communication_backend"] == "uccl_ep"
    matrix = {source: {target: 0 for target in nodes} for source in nodes}
    for step in steps:
        for layer in range(48):
            experts = [0] * 128
            incoming = [0] * 16
            for rank in range(16):
                row = audits[rank]["steps"][step]["layers"][layer]
                for name in totals:
                    totals[name] += row[name]
                tokens += row["routed_tokens"]
                for expert, value in enumerate(row["expert_counts"]):
                    experts[expert] += value
                for target, value in enumerate(row["rank_traffic"]):
                    incoming[target] += value
                    matrix[hosts[str(rank)]][hosts[str(target)]] += value
                if dedup:
                    ranks += sum(row["unique_token_rank_counts"])
                    node_copies += sum(row["unique_token_node_counts"])
                    remote_copies += sum(value for host, value in zip(row["unique_token_node_hosts"], row["unique_token_node_counts"])
                                         if host != hosts[str(rank)])
            expected = 16 * 131072
            equal(sum(experts), expected, "incoming layer assignments")
            equal(sum(incoming), expected, "incoming rank assignments")
            rebuilt = [0] * 16
            for expert, value in enumerate(experts):
                rebuilt[owners[str(layer)][expert]] += value
            equal(rebuilt, incoming, "incoming runtime ownership")
            incoming_nodes = [sum(incoming[rank] for rank in range(16) if hosts[str(rank)] == host) for host in nodes]
            loads.append({"step": step, "layer": layer, "expert": load_spread(experts),
                          "rank": load_spread(incoming), "node": load_spread(incoming_nodes)})
    equal(totals["assignments"], len(steps) * 16 * 48 * 131072, "routing window assignments")
    equal(totals["local"] + totals["intra_node"] + totals["inter_node"], totals["assignments"], "routing window locality")
    unique = {"available": False, "reason": "Native marginal counts do not recover per-token unique destinations."}
    if dedup:
        require(node_copies > 0 and ranks >= node_copies, "invalid deduplication window")
        unique = {"available": True, "unit": "forward logical token/destination deliveries, including local",
                  "unique_rank_deliveries": ranks, "unique_node_deliveries": node_copies,
                  "avg_unique_ranks_per_token": ranks / tokens, "avg_unique_nodes_per_token": node_copies / tokens,
                  "avg_remote_nodes_per_token": remote_copies / tokens,
                  "assignment_to_unique_rank_ratio": totals["assignments"] / ranks,
                  "assignment_to_unique_node_ratio": totals["assignments"] / node_copies}
    return {"steps": list(steps), "unit": "forward expert assignments", "totals": totals,
            "layer_update_observations": len(loads), "source_destination_node_assignments": matrix,
            "locality_percent": {name: 100 * totals[name] / totals["assignments"] for name in ("local", "intra_node", "inter_node")},
            "incoming_load_spread": {kind: {field: describe(row[kind][field] for row in loads)
                for field in ("cv", "max_over_mean")} for kind in ("expert", "rank", "node")}, "dedup": unique}


def route_difference(first, second):
    changed = expert_l1 = rank_l1 = 0
    by_update = []
    for step in STEPS:
        count = 0
        for rank in range(16):
            for layer in range(48):
                left = first[rank]["steps"][step]["layers"][layer]
                right = second[rank]["steps"][step]["layers"][layer]
                difference = left["expert_counts"] != right["expert_counts"] or left["rank_traffic"] != right["rank_traffic"]
                count += difference
                expert_l1 += sum(abs(a - b) for a, b in zip(left["expert_counts"], right["expert_counts"]))
                rank_l1 += sum(abs(a - b) for a, b in zip(left["rank_traffic"], right["rank_traffic"]))
        changed += count
        by_update.append({"step": step, "changed_rows": count})
    return {"route_histograms_exactly_equal": changed == 0, "changed_rows": changed,
            "compared_rank_update_layer_rows": 16 * 25 * 48, "expert_count_l1": expert_l1, "rank_traffic_l1": rank_l1,
            "per_token_route_identity_available": False, "by_update": by_update}


def coordinator_commands(validation, runs, root):
    commands = []
    for leg in LEGS:
        for case in CASES:
            run, label = runs[case][leg], f"{case}-{leg}"
            destination = validation / label
            commands.append((f"{label}-received", [str(root / "analysis/validate_ep16_received_routes.py"),
                "--run-dir", str(run), "--steps", "25", "--output", str(destination / "received_routes.json")]))
            if leg == "profile":
                for rank in range(16):
                    if is_uccl(case):
                        for step in STEPS:
                            trace, call = evidence._trace_paths(run, rank, step)
                            commands.append((f"{label}-rank-{rank}-step-{step:05d}", [str(root / "analysis/inspect_uccl_trace.py"),
                                "--trace", str(trace), "--record", str(call), "--output", str(destination / f"rank-{rank}-step-{step:05d}.json")]))
                    else:
                        commands.append((f"{label}-rank-{rank}", [str(root / "analysis/analyze_all_step_trace.py"),
                            "--run-dir", str(run), "--rank", str(rank), "--steps", "25"]))
    return commands


def check_coordinator(orchestration, validation, runs, inputs, root):
    commands = coordinator_commands(validation, runs, root)
    actual = inputs.read(validation / "commands.json")
    require(len(actual) == len(commands), "coordinator command coverage differs")
    for item, expected in zip(actual, commands):
        require(isinstance(item, list) and len(item) == 2 and isinstance(item[1], list)
                and len(item[1]) > 1 and isinstance(item[1][0], str) and Path(item[1][0]).is_absolute(), "invalid coordinator command")
        equal([item[0], item[1][1:]], [expected[0], expected[1]], "coordinator actual validator command")
    complete = inputs.read(validation / "COMPLETE.json")
    for key, value in {"all_invoked_validators_passed": True, "mode": "full", "steps": 25, "case_count": 8,
                       "commands": len(commands), "orchestration_dir": str(orchestration),
                       "coordinator_sha256": inputs.bind(root / "tools/uccl_ep_validate_profiles.py")}.items():
        equal(complete.get(key), value, f"coordinator completion {key}")
    require(type(complete.get("job")) is str and bool(complete["job"]), "missing validation allocation")
    return complete


def compare(orchestration_dir, validation_root, *, root=ROOT):
    root = Path(root).resolve()
    orchestration, validation = Path(orchestration_dir).resolve(), Path(validation_root).resolve()
    inputs = evidence.Inputs()
    rows = [line.split("\t") for line in bound_text(orchestration / "cases.tsv", inputs).splitlines()]
    expected = [(case, leg) for leg in LEGS for case in CASES]
    require(len(rows) == 8 and all(len(row) == 3 for row in rows), "exact eight-run acquisition required")
    runs = {case: {} for case in CASES}
    for row, (case, leg) in zip(rows, expected):
        equal(row[:2], [case, leg], "full acquisition order")
        run = Path(row[2])
        require(run.is_absolute() and run.is_dir() and not run.is_symlink() and run.parent == root / "runs", "case run root differs")
        runs[case][leg] = run.resolve()
    require(len({path for pair in runs.values() for path in pair.values()}) == 8, "case roots are not distinct")
    sources = _ampa_runtime_sources(root)
    for relative, digest in sources["files"].items():
        inputs.bind(root / relative, digest)
    for relative in SOURCES:
        inputs.bind(root / relative)
    acquisition = inputs.read(orchestration / "acquisition.json")
    for key, value in {"mode": "full", "topology": "cp1_ep16", "launch_policy": "torchrun_node4",
                       "runtime_source_sha256": sources["sha256"]}.items():
        equal(acquisition.get(key), value, f"full acquisition {key}")
    job = acquisition.get("job")
    require(type(job) is str and bool(job), "missing training allocation")
    completed = inputs.read(orchestration / "acquisition_completed.json")
    for key, value in {"success": True, "workers_srun_returned_zero": True, "mode": "full", "job": job,
        "runtime_source_sha256": sources["sha256"], "binary_sha256": acquisition["binary_sha256"],
        "cases_tsv_sha256": inputs.bind(orchestration / "cases.tsv")}.items():
        equal(completed.get(key), value, f"successful acquisition completion {key}")
    pilot_binding = acquisition["accepted_pilot_proof"]
    require(isinstance(pilot_binding, dict) and set(pilot_binding) == {"path", "sha256"}, "accepted pilot binding missing")
    require(Path(pilot_binding["path"]).is_absolute(), "pilot proof path must be absolute")
    inputs.bind(pilot_binding["path"], pilot_binding["sha256"])
    pilot = inputs.read(pilot_binding["path"])
    for key, value in {"valid_pilot": True, "cases": list(CASES), "completed_updates": 2, "topology": "cp1_ep16",
                       "valid_training_health": True, "valid_send_receive_audit": True, "valid_uccl_trace_capture": True,
                       "binary_sha256": acquisition["binary_sha256"], "runtime_source_sha256": sources["sha256"]}.items():
        equal(pilot.get(key), value, f"accepted pilot prerequisite {key}")
    coordinator = check_coordinator(orchestration, validation, runs, inputs, root)
    timing, health, activity, routing, audits, placements = ({case: {} for case in CASES} for _ in range(6))
    reference = None
    manifests, fingerprints = {}, {}
    reference_ownership = None
    for leg in LEGS:
        for case in CASES:
            run = runs[case][leg]
            contract = inputs.read(run / "config/comparison_contract.json")
            configuration = yaml.safe_load(bound_text(run / "config/resolved.yaml", inputs))
            environment = inputs.read(run / "environment.json")
            equal(environment["slurm_job_id"], job, "same training allocation")
            equal(environment["ampa_runtime_sources"], sources, "actual runtime sources")
            equal(environment["launch_policy"], "torchrun_node4", "environment launch policy")
            equal(environment["runtime_topology"], _EP16_RUNTIME_TOPOLOGIES["cp1_ep16"], "actual serialized runtime topology")
            placement = validate_rank_placement(environment["rank_placement"], expected_world_size=16, launch_policy="torchrun_node4")
            validate_runtime_groups(placement, expected_world_size=16, cp=1)
            placements[case][leg] = placement
            # The complete emitted environment has no output-path fields. Its
            # exact equality preserves every runtime setting, including affinity.
            common = (normalized_contract(contract, case), normalized_resolved(configuration, case, run), environment)
            if reference is None:
                reference = common
            equal(common[0], reference[0], "eight-run training contract identity")
            evidence.equal_resolved_configuration(common[1], reference[1], "eight-run resolved training identity")
            equal(common[2], reference[2], "eight-run runtime environment identity")
            for name in ("c0_manifest.json", "dataset_manifest.json", "model_manifest.json"):
                digest = inputs.bind(run / "config" / name)
                manifests.setdefault(name, digest)
                equal(digest, manifests[name], f"shared {name}")
            equal(manifests["c0_manifest.json"], contract["c0_identity_manifest_sha256"], "C0 contract binding")
            equal(manifests["dataset_manifest.json"], contract["dataset"]["manifest_sha256"], "dataset contract binding")
            mode = inputs.read(run / "config/observation_mode.json")
            for key, value in {"data_mode": "indexed", "lightweight_profiling": "off", "nsys_trace": False,
                "torch_trace": leg == "profile", "sample_fingerprints": True, "router_observer_attached": False,
                "route_wrapper_installed": False, "dispatcher_preprocess_wrapper_installed": False}.items():
                equal(mode.get(key), value, f"observation mode {key}")
            capture = inputs.read(run / "config/all_step_capture.json")
            for key, value in {"enabled": leg == "profile", "steps": list(STEPS),
                               "ranks": list(range(16)) if leg == "profile" else [0]}.items():
                equal(capture[key], value, f"capture declaration {key}")
            if leg == "control":
                require(not any((run / "traces/torch_profile").rglob("*.json*"))
                        and not any((run / "raw/deep_trace").rglob("*.json")), "control contains capture artifacts")
            for rank in range(16):
                path = run / "raw" / f"sample_fingerprint_rank-{rank}.json"
                batches = _checked_fingerprint_batches(inputs.read(path), rank=rank, accumulation=4, steps=25, token_shape=[1, 4096], path=path)
                fingerprints.setdefault(rank, batches)
                equal(batches, fingerprints[rank], "eight-run actual CPU batches")
            timing[case][leg], health[case][leg] = load_timing_health(run, inputs)
            received = inputs.read(validation / f"{case}-{leg}/received_routes.json")
            equal(received, validate_received_run(run, 25), "recomputed all-source receive proof")
            expected_paths = [run / "config/comparison_contract.json", run / "config/routing_ownership.json", run / "environment.json"] + [run / "raw" / f"routing_audit_rank-{rank}.json" for rank in range(16)]
            require(len(received["input_files"]) == 19, "receive binding grid")
            evidence._bound_map({row["path"]: row["sha256"] for row in received["input_files"]}, expected_paths, inputs, "receive actual inputs")
            ownership = inputs.read(run / "config/routing_ownership.json")
            common_ownership = {name: ownership[name] for name in ("expert_owners", "rank_hosts", "group_hosts")}
            if reference_ownership is None:
                reference_ownership = common_ownership
            equal(common_ownership, reference_ownership, "actual eight-run ownership/hosts")
            audits[case][leg] = [inputs.read(run / "raw" / f"routing_audit_rank-{rank}.json") for rank in range(16)]
            routing[case][leg] = {name: routing_summary(audits[case][leg], ownership, steps)
                                  for name, steps in (("all25", STEPS), ("postwarmup20", MEASURED))}
            if leg == "profile":
                activity[case] = load_profile(case, run, validation, inputs, root)
    reviewed_path = orchestration / "reviewed_inputs.json"
    reviewed = inputs.read(reviewed_path)
    for key, value in {"valid_reviewed_inputs": True, "slurm_job_id": job, "ampa_runtime_sources": sources,
        "c0_identity_manifest_sha256": manifests["c0_manifest.json"], "dataset_manifest_sha256": manifests["dataset_manifest.json"]}.items():
        equal(reviewed.get(key), value, f"reviewed immutable inputs {key}")
    equal(reviewed["checkpoint_payload"]["verified"], True, "verified immutable checkpoint payload")
    canonical_c0 = root / "checkpoint/megatron/qwen3-30b-a3b-base-c0"
    evidence._path(reviewed["checkpoint_payload"]["path"], canonical_c0, "immutable C0 path")
    equal(reference[0]["initialization"], {"mode": "pretrained", "checkpoint": str(canonical_c0)}, "declared canonical C0")
    equal(reference[2]["checkpoint"], str(canonical_c0), "actual canonical C0")
    equal(inputs.bind(canonical_c0 / "ampa_c0_manifest.json"), manifests["c0_manifest.json"], "actual canonical C0 manifest")
    equal(reviewed["native_checkouts"], {key: reference[2][key] for key in ("megatron_lm", "megatron_bridge")}, "reviewed native checkouts")
    equal(reference[2]["reviewed_inputs_proof"], {"path": str(reviewed_path), "sha256": inputs.bind(reviewed_path)}, "actual reviewed input binding")
    build = Path(inputs.read(runs["uccl"]["profile"] / "config/comparison_contract.json")["uccl_build_dir"])
    require(build.is_absolute() and build.is_dir() and not build.is_symlink(), "isolated UCCL build required")
    source_identity = verify_uccl_source(build)
    equal(inputs.read(build / "source_identity.json"), source_identity, "approved UCCL source manifest")
    equal(acquisition["uccl_source_identity"], source_identity, "actual approved acquisition source")
    equal(pilot["uccl_source_identity"], source_identity, "approved pilot source")
    inputs.bind(build / "source/uccl-borrowed-scratch.patch", source_identity["patch_sha256"])
    for relative, digest in source_identity["modified_files"].items():
        inputs.bind(build / "uccl" / relative, digest)
    for case in CASES:
        if not is_uccl(case):
            continue
        for leg in LEGS:
            equal(inputs.read(runs[case][leg] / "config/comparison_contract.json")["uccl_build_dir"], str(build), "same UCCL build for all UCCL runs")
            binary = evidence._uccl_identity(runs[case][leg], build, source_identity, placements[case][leg], job, root, inputs)
            equal(binary, acquisition["binary_sha256"], "actual UCCL binary")
    differences = {f"{case}_on_off": route_difference(audits[case]["control"], audits[case]["profile"]) for case in CASES}
    for base, case in BACKEND_PAIRS:
        for leg in LEGS:
            differences[f"{base}_vs_{case}_{leg}"] = route_difference(audits[base][leg], audits[case][leg])
    on_off = observation_changes(timing, differences)
    result = {"schema_version": 2, "kind": "ampa_uccl_ep_comparison", "valid_descriptive_comparison": True,
        "accepted_full_eight_case_evidence": True, "cases": list(CASES), "case_configuration": CASE_CONFIG,
        "completed_updates": 25, "measured_steps": list(MEASURED),
        "valid_training_health": True, "valid_send_receive_audit": True, "valid_all_profile_captures": True,
        "same_actual_rank_placement": True,
        "case_runs": {case: {leg: str(path) for leg, path in pair.items()} for case, pair in runs.items()},
        "training_slurm_job_id": job, "validation_slurm_job_id": coordinator["job"], "topology": "cp1_ep16",
        "rank_placement": placements["ordinary"]["control"], "ownership": reference_ownership,
        "runtime_source_sha256": sources["sha256"], "uccl_source_identity": source_identity, "binary_sha256": binary,
        "backend_bundle": {**{key: reference[2][key] for key in (
            "pytorch_version", "cuda_version", "nccl_version", "cuda_allocator_conf", "container_sqsh_sha256", "megatron_lm", "megatron_bridge")},
            "uccl_mode": "normal_high_throughput", "low_latency_dedup_api_exercised": False,
            "libfabric": inputs.read(runs["uccl"]["profile"] / "raw/uccl_backend_rank-0_startup.json")["libfabric"]},
        "matched_cpu_batches_per_case": 1600, "timing": timing, "health": health, "activity": activity, "routing": routing,
        "route_histogram_differences": differences, "on_off": on_off,
        "initial_backend_layer0_histograms_equal": {f"{base}_vs_{case}": {leg: all(
            audits[base][leg][rank]["steps"][0]["layers"][0]["expert_counts"] == audits[case][leg][rank]["steps"][0]["layers"][0]["expert_counts"]
            for rank in range(16)) for leg in LEGS} for base, case in BACKEND_PAIRS},
        "control_timing_changes": control_contrasts(timing),
        "activity_changes": {f"{base}_to_{case}": {phase: {
            metric: change(activity[base]["phases"][phase][metric], activity[case]["phases"][phase][metric])
            for metric in ("mean_gpu_inclusive_activity_ms_per_rank_update", "mean_cpu_scope_ms_per_rank_update")}
            for phase in activity[base]["phases"]}
            for base, case in CONTRASTS},
        "claims": {name: False for name in ("causal_speedup", "isolated_profiler_overhead", "physical_wire_bytes", "cpu_proxy_time", "exposed_communication_latency", "model_quality_equivalence")},
        "claim_limits": LIMITS,
        "analysis_source_sha256": {name: inputs.hashes[str((root / name).resolve())] for name in SOURCES}}
    inputs.recheck()
    result["input_files"] = [{"path": path, "sha256": digest} for path, digest in sorted(inputs.hashes.items())]
    return result


def report_text(data):
    lines = ["# Native and UCCL: ordinary and grouped routing", "",
        "All eight runs completed 25 updates and passed the bound input, health, routing, placement, backend and capture gates. "
        "The values below describe one sequential allocation. They do not establish causal speedup.", "",
        "The common setup is four nodes, EP16, CP1, BF16, 128 experts and top-eight routing. "
        "Grouped cases select two of four node-aligned groups. Each GPU owns eight experts.", "",
        "## Complete optimizer-step performance", "",
        "Controls retain batch hashing and routing/receive audits. The 20 measurement updates are 5 to 24. "
        "The token rate divides all measured global tokens by their total wall time.", "",
        "| Case | Control mean ms | Control median ms | Min–max ms | Aggregate tokens/s | Profile mean ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for case in CASES:
        control, profile = (data["timing"][case][leg] for leg in LEGS)
        value = control["step_ms"]
        lines.append(f"| {case} | {value['mean']:.3f} | {value['median']:.3f} | {value['min']:.3f}–{value['max']:.3f} | {control['aggregate_tokens_per_second']:.1f} | {profile['step_ms']['mean']:.3f} |")
    lines += ["", "| OFF-only contrast | Mean duration change % | Mean speed ratio | Median speed ratio |", "| --- | ---: | ---: | ---: |"]
    for base, case in CONTRASTS:
        row = data["control_timing_changes"][f"{base}_to_{case}"]
        lines.append(f"| {base} to {case} | {row['step_mean_ms']['percent']:.3f} | {row['observed_speed_ratio_mean']:.4f}x | {row['observed_speed_ratio_median']:.4f}x |")
    lines += ["", "A speed ratio above1 means the comparison run took less time. These ratios use complete unprofiled updates only. "
              "The profile activity below cannot establish a fraction of unprofiled critical-path time."]
    lines += ["", "## Common profile boundaries", "",
        "Each value is mean inclusive correlated GPU milliseconds per rank optimizer update. "
        "The 320 observations across ranks and updates come from 20 dependent optimizer updates. Nested rows overlap. "
        "These values cannot form a stacked partition of elapsed step time.", "",
        "| Boundary | Native ordinary ms | Native grouped ms | UCCL ordinary ms | UCCL grouped ms | UCCL/native ordinary change % | UCCL/native grouped change % |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    selected = ("router_total", "dispatch_side", "expert_total", "combine_side", "moe_total",
                "train_microbatch_forward", "train_microbatch_backward", "train_gradient_finalize", "train_optimizer_step")
    for phase in selected:
        values = [data["activity"][case]["phases"][phase]["mean_gpu_inclusive_activity_ms_per_rank_update"] for case in CASES]
        percentages = [change(values[0], values[2])["percent"], change(values[1], values[3])["percent"]]
        lines.append(f"| {phase} | " + " | ".join(f"{value:.3f}" for value in values) + " | "
                     + " | ".join(f"{value:.3f}" if value is not None else "unavailable" for value in percentages) + " |")
    lines += ["", "Dispatch side sums preparation, dispatch and local expert sorting. Combine side sums inverse sorting, combine and final restoration. "
        "These sibling composites account for work that UCCL moves into fused stages. They contain more than communication.", "",
        "## Routing, incoming load and health", "",
        "Locality counts forward expert assignments. Unique UCCL destinations count logical token deliveries, including local destinations. "
        "Native marginal counts cannot recover the corresponding unique sets.", "",
        "| Case/profile | Local % | Same-node other-rank % | Other-node % | Mean expert max/mean | Unique ranks/token | Unique nodes/token |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for case in CASES:
        row = data["routing"][case]["profile"]["postwarmup20"]
        locality, unique = row["locality_percent"], row["dedup"]
        rank_value = f"{unique['avg_unique_ranks_per_token']:.3f}" if unique["available"] else "unavailable"
        node_value = f"{unique['avg_unique_nodes_per_token']:.3f}" if unique["available"] else "unavailable"
        lines.append(f"| {case} | {locality['local']:.3f} | {locality['intra_node']:.3f} | {locality['inter_node']:.3f} | {row['incoming_load_spread']['expert']['max_over_mean']['mean']:.3f} | {rank_value} | {node_value} |")
    lines += ["", "All eight runs have finite logged optimization metrics, zero skipped/NaN updates and 1600 consumed samples per run. "
        "The JSON preserves every loss/gradient boundary and incoming expert/rank/node spread. These observations do not establish quality equivalence.", "",
        "## ON/OFF and trajectory limits", "",
        "| Case | Changed ON/OFF rank-update-layer histograms | Expert-count L1 | Raw mean duration change % |", "| --- | ---: | ---: | ---: |"]
    for case in CASES:
        diff = data["route_histogram_differences"][f"{case}_on_off"]
        lines.append(f"| {case} | {diff['changed_rows']}/{diff['compared_rank_update_layer_rows']} | {diff['expert_count_l1']} | {data['on_off'][case]['raw_mean_step_change']['percent']:.3f} |")
    lines += ["", "| Native/UCCL pair and leg | Changed rank-update-layer histograms | Expert-count L1 | Initial layer 0 histograms equal |",
              "| --- | ---: | ---: | --- |"]
    for base, case in BACKEND_PAIRS:
        pair = f"{base}_vs_{case}"
        for leg in LEGS:
            diff = data["route_histogram_differences"][f"{pair}_{leg}"]
            lines.append(f"| {pair}/{leg} | {diff['changed_rows']}/{diff['compared_rank_update_layer_rows']} | {diff['expert_count_l1']} | {data['initial_backend_layer0_histograms_equal'][pair][leg]} |")
    lines += ["", "All profiler stop/export costs appear separately in the JSON. "
        "A raw ON/OFF time difference includes observation cost, route changes and temporal drift. It does not isolate profiler overhead.", "",
        "## Evidence and limits", ""]
    lines += [f"- {limit}" for limit in data["claim_limits"]]
    lines += ["", "`comparison.json` binds every consumed input and analysis source. `phase_metrics.csv` contains the common inclusive rows. "
        "Native exclusive totals, interval unions and NCCL-only totals were not substituted for UCCL inclusive activity.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orchestration-dir", type=Path, required=True)
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    require(os.environ.get("SLURM_JOB_ID") and os.environ.get("SLURM_JOB_PARTITION") == "preemptable", "allocated preemptable compute required")
    output = args.output_dir
    require(not output.exists() and not output.is_symlink() and output.is_absolute() and output.parent.resolve() == ROOT / "runs", "output must be a new directory directly under runs")
    result = compare(args.orchestration_dir, args.validation_root)
    require(output.resolve() not in {Path(path) for pair in result["case_runs"].values() for path in pair.values()}, "output overlaps an immutable input run")
    output.mkdir()
    (output / "REPORT.md").write_text(report_text(result))
    with (output / "phase_metrics.csv").open("x", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("case", "phase", "mean_gpu_inclusive_activity_ms_per_rank_update", "mean_cpu_scope_ms_per_rank_update", "gpu_unit", "cpu_unit"))
        for case in CASES:
            for phase, row in result["activity"][case]["phases"].items():
                writer.writerow((case, phase, row["mean_gpu_inclusive_activity_ms_per_rank_update"], row["mean_cpu_scope_ms_per_rank_update"],
                                 "inclusive GPU activity ms per rank optimizer update", "inclusive CPU scope ms per rank optimizer update"))
    # Publish the acceptance-bearing JSON last, after report generation and a
    # second end-to-end read. A failed recheck leaves no valid comparison file.
    fresh = evidence.Inputs()
    for binding in result["input_files"]:
        fresh.bind(binding["path"], binding["sha256"])
    result["output_artifacts"] = {name: hashlib.sha256((output / name).read_bytes()).hexdigest()
                                  for name in ("REPORT.md", "phase_metrics.csv")}
    (output / "comparison.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"valid_descriptive_comparison": True, "output": str(output)}), flush=True)


if __name__ == "__main__":
    main()
