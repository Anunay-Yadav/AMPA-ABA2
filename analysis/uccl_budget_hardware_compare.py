#!/usr/bin/env python3
"""Compare four accepted, source-bound UCCL hardware metric analyses on CPU."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import sys
import tempfile

import yaml


ROOT = Path(__file__).resolve().parents[1]
ANALYZER = ROOT / "analysis/uccl_budget_hardware_metrics.py"
ANALYZER_SHA = "f43fb3fc18fb43cf278a61c8aba47eaab192525fe1c20ec3434316de1367b45c"
BUDGETS = (12, 20, 32, 40)
NAMES = ("SMs Active [Throughput %]", "SM Issue [Throughput %]", "Compute Warps in Flight [Throughput %]")
VIEWS = ("optimizer_step_window", "during_any_payload", "during_dispatch", "during_combine")
FIXED_IDENTITIES = ("c0_identity_manifest_sha256", "dataset_manifest_sha256", "sample_prefix_sha256",
                    "runtime_source_sha256", "binary_sha256", "adapter_source_sha256", "worker_source_sha256")
VM_MASK = 0xFFFF000000000000
PID_MASK = (1 << 24) - 1
LIMITS = [
    "Values are pooled arithmetic means of selected sample points, weighted by sample count, without time integration.",
    "Samples describe all device work during payload intervals, including concurrent work; communication-only activity is unavailable.",
    "Compute-warp throughput percentage is not a conversion to achieved occupancy or resident warps per SM.",
    "Payload block budgets do not denote reserved SMs or free SM capacity.",
    "Physical placement differs across independent allocations. These Nsight diagnostics establish no causal effect or speedup.",
    "Dispatch/combine selections overlap and are not additive partitions. Unsampled short intervals remain recorded.",
    "Whole-session event completeness remains unresolved; acceptance covers the known payload/call event set.",
]


def require(value, message):
    if not value:
        raise ValueError(message)


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def uuid_key(value):
    require(type(value) is str, "missing GPU UUID")
    result = value.lower().removeprefix("gpu-").replace("-", "")
    require(re.fullmatch(r"[0-9a-f]{32}", result), "invalid GPU UUID")
    return result


def bindings(rows):
    require(type(rows) is list and all(type(row) is dict and set(row) == {"path", "sha256"} for row in rows),
            "input binding schema differs")
    result = {}
    for row in rows:
        require(type(row["path"]) is str and Path(row["path"]).is_absolute()
                and re.fullmatch(r"[0-9a-f]{64}", row["sha256"] or ""), "invalid input binding")
        require(row["path"] not in result, "duplicate input binding")
        result[row["path"]] = row["sha256"]
    return result


class Inputs:
    def __init__(self):
        self.hashes = {}

    def bind(self, path, expected=None):
        path = Path(path)
        require(path.is_absolute() and path.is_file() and not path.is_symlink(), f"bad regular input: {path}")
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        require(expected is None or digest == expected, f"input SHA differs: {path}")
        require(str(path) not in self.hashes or self.hashes[str(path)] == digest, "input changed while reading")
        self.hashes[str(path)] = digest
        return digest

    def read(self, path, expected=None):
        return json.loads(self.text(path, expected))

    def text(self, path, expected=None):
        digest = self.bind(path, expected)
        data = Path(path).read_bytes()
        require(hashlib.sha256(data).hexdigest() == digest, "input changed during read")
        return data.decode("utf-8")

    def recheck(self):
        for path, digest in list(self.hashes.items()):
            self.bind(path, digest)


def check_record(record, budget):
    require(record.get("schema_version") == 1 and record.get("valid_bound_metric_analysis") is True,
            "hardware analysis is not accepted")
    require(type(record.get("communication_sms")) is int and record["communication_sms"] == budget,
            "metric record budget differs")
    require(bindings(record["input_files"]).get(str(ANALYZER)) == ANALYZER_SHA, "unreviewed metric analyzer identity")
    require(record.get("metric_unit") == "percent, source resolution one percentage point", "metric units differ")
    nodes = record["nodes"]
    require(len(nodes) == 4 and sorted(node["node_rank"] for node in nodes) == list(range(4)), "four distinct metric nodes required")
    slots = {}
    for node in nodes:
        require(len(node["metric_streams"]) == 4, "four metric streams per node required")
        for slot in node["metric_streams"]:
            rank = slot["rank"]
            require(type(rank) is int and rank in range(16) and rank not in slots, "duplicate/invalid metric rank")
            require(slot["node_rank"] == node["node_rank"] and slot["metric_names"] == list(NAMES), "metric stream identity differs")
            slots[rank] = (slot["node_rank"], uuid_key(slot["uuid"]), slot["type_id"])
    require(len({slot[1] for slot in slots.values()}) == 16, "sixteen distinct physical GPU UUIDs required")
    windows = record["rank_update_windows"]
    require(len(windows) == 80 and {(row["rank"], row["step_index"]) for row in windows}
            == {(rank, step) for rank in range(16) for step in range(5, 10)}, "eighty unique measured rank/update windows required")
    for row in windows:
        require(type(row["rank"]) is int and type(row["step_index"]) is int
                and row["human_update"] == row["step_index"] + 1, "window update identity differs")
        require((row["node_rank"], uuid_key(row["physical_uuid"]), row["type_id"]) == slots[row["rank"]], "window GPU identity differs")
        require(type(row["payload_kernels"]) is int and row["payload_kernels"] == 768
                and row["payloads_outside_host_step_bounds"] == 0, "window payload witness differs")
        cov = row["sample_coverage"]
        require(cov["valid_step_sample_coverage"] is True and cov["invalid_selected_metric_bundles"] == 0
                and cov["internal_gaps_over_two_periods"] == 0, "step sample coverage failed")
        require(type(cov["sample_bundles"]) is int and cov["sample_bundles"] >= 2, "insufficient step samples")
        period = cov["nominal_period_ns"]
        require(number(period) and period > 0 and all(number(cov[key]) and 0 <= cov[key] <= 2 * period
                for key in ("first_boundary_distance_ns", "last_boundary_distance_ns", "spacing_max_ns")), "numeric coverage limit failed")
        require(cov["spacing_max_ns"] > 0, "nonpositive sample spacing")
        require(type(row["step_start_ns"]) is int and type(row["step_end_ns"]) is int
                and row["step_end_ns"] > row["step_start_ns"]
                and (cov["verified_window_start_ns"], cov["verified_window_end_ns"])
                == (row["step_start_ns"], row["step_end_ns"]), "verified step bounds differ")
        for view in VIEWS:
            selected = row["views"][view]
            count = selected["selected_sample_bundles"]
            require(selected["metrics_available"] is True and selected["intervals_within_verified_sample_window"] is True
                    and selected["invalid_selected_metric_bundles"] == 0
                    and type(count) is int and 2 <= count <= cov["sample_bundles"], f"selected coverage failed: {view}")
            require(all(type(selected[key]) is int and selected[key] >= 0
                        for key in ("intervals_without_sample", "intervals_with_one_sample"))
                    and type(selected["interval_union_duration_ns"]) is int
                    and 0 < selected["interval_union_duration_ns"] <= row["step_end_ns"] - row["step_start_ns"],
                    "invalid selected interval summary")
            require(set(selected["metrics"]) == set(NAMES), "selected metric set differs")
            for metric in selected["metrics"].values():
                require(type(metric["count"]) is int and metric["count"] == count, "metric sample count differs")
                require(all(number(metric[key]) and 0 <= metric[key] <= 100 for key in
                        ("arithmetic_sample_mean_percent", "min_percent", "median_percent", "p95_percent", "max_percent")), "invalid percentage summary")
                require(metric["min_percent"] <= metric["arithmetic_sample_mean_percent"] <= metric["max_percent"]
                        and metric["min_percent"] <= metric["median_percent"] <= metric["p95_percent"] <= metric["max_percent"], "percentage bounds differ")
    require(sum(row["payload_kernels"] for row in windows) == 61440, "61,440 known payloads required")
    summaries = {}
    for view in VIEWS:
        pooled = record["pooled_sample_summaries"][view]
        require(pooled["all_80_windows_available"] is True and set(pooled["metrics"]) == set(NAMES), "pooled coverage failed")
        summaries[view] = {}
        for name in NAMES:
            parts = [row["views"][view]["metrics"][name] for row in windows]
            count = sum(part["count"] for part in parts)
            mean = math.fsum(part["count"] * part["arithmetic_sample_mean_percent"] for part in parts) / count
            reported = pooled["metrics"][name]
            require(type(reported["count"]) is int and reported["count"] == count
                    and number(reported["arithmetic_sample_mean_percent"])
                    and math.isclose(mean, reported["arithmetic_sample_mean_percent"], rel_tol=1e-12, abs_tol=1e-10), "pooled arithmetic sample aggregation differs")
            require(reported["min_percent"] == min(part["min_percent"] for part in parts)
                    and reported["max_percent"] == max(part["max_percent"] for part in parts), "pooled extrema differ")
            summaries[view][name] = {"sample_count": count, "mean_percent": mean}
    return {"communication_sms": budget, "training_slurm_job_id": record["training_slurm_job_id"],
        "rank_update_windows": 80, "physical_uuids": sorted(slot[1] for slot in slots.values()),
        "payload_kernels": 61440, "views": summaries,
        "coverage": {view: {"intervals_without_sample": sum(row["views"][view]["intervals_without_sample"] for row in windows),
            "intervals_with_one_sample": sum(row["views"][view]["intervals_with_one_sample"] for row in windows),
            "interval_union_duration_ns_sum_across_rank_windows": sum(row["views"][view]["interval_union_duration_ns"] for row in windows)} for view in VIEWS}}


def accepted_mapping(record, acceptance, inventory, metric_bindings, accepted_bindings):
    """Match every metric rank and database to its accepted runtime witness."""
    placement = acceptance["rank_placement"]
    require(len(placement) == 16 and sorted(row["rank"] for row in placement) == list(range(16)),
            "accepted runtime rank grid differs")
    by_rank = {row["rank"]: row for row in placement}
    nodes = {row["node_rank"]: row for row in inventory["nodes"]}
    for node in record["nodes"]:
        proof = nodes[node["node_rank"]]
        require(len({slot["type_id"] for slot in node["metric_streams"]}) == 4
                and len({slot["global_pid"] for slot in node["metric_streams"]}) == 4,
                "distinct metric streams/processes required per node")
        sqlite_path = node["sqlite_path"]
        require(sqlite_path == proof["path"] and sqlite_path in metric_bindings
                and metric_bindings[sqlite_path] == accepted_bindings.get(sqlite_path)
                == proof["export"]["sqlite_sha256"], "metric SQLite identity differs from accepted node")
        owners = proof["rank_by_global_pid"]
        require(len(owners) == 4 and all(type(pid) is str and pid.isdigit() and type(rank) is int
                for pid, rank in owners.items()) and len(set(owners.values())) == 4, "accepted node owners differ")
        by_owner_rank = {rank: int(pid) for pid, rank in owners.items()}
        require({slot["rank"] for slot in node["metric_streams"]} == set(by_owner_rank), "metric node rank membership differs")
        require(set(by_owner_rank) == {rank for rank, row in by_rank.items()
                if str(row["slurm_node_id"]) == str(node["node_rank"])}, "accepted owner/runtime node membership differs")
        for slot in node["metric_streams"]:
            owner, runtime = by_owner_rank[slot["rank"]], by_rank[slot["rank"]]
            require(type(slot["global_pid"]) is int and slot["global_pid"] == owner
                    and slot["os_pid"] == (owner >> 24) & PID_MASK, "metric process differs from accepted owner")
            require(uuid_key(slot["uuid"]) == uuid_key(runtime["physical_gpu_uuid"]), "metric rank/GPU permutation differs")
            require(type(slot["type_id"]) is int and type(slot["physical_gpu_id"]) is int
                    and slot["global_vm"] == owner & VM_MASK == slot["type_id"] & VM_MASK
                    and slot["physical_gpu_id"] == slot["type_id"] & 0xff, "metric VM/GPU target differs")


def normalized_config_identity(configuration, run, budget):
    value = copy.deepcopy(configuration)
    require(type(value["model"].pop("moe_flex_dispatcher_num_sms")) is int
            and configuration["model"]["moe_flex_dispatcher_num_sms"] == budget, "resolved native budget differs")
    for key, expected in {"tensorboard_dir": str(run / "logs/tensorboard"), "wandb_exp_name": run.name,
                          "wandb_save_dir": str(run / "logs/wandb")}.items():
        require(value["logger"].pop(key) == expected, f"resolved output differs: {key}")

    def checked(item, path=()):
        if type(item) is dict:
            require(all(type(key) is str for key in item), "non-string resolved configuration key")
            for key, child in item.items():
                checked(child, (*path, key))
        elif type(item) is list:
            for index, child in enumerate(item):
                checked(child, (*path, index))
        elif type(item) is float:
            require(math.isfinite(item) or (path == ("optimizer", "grad_norm_skip_threshold") and item == math.inf),
                    "unapproved nonfinite configuration value")
        else:
            require(type(item) in (type(None), bool, int, str), "unsupported resolved configuration value")

    checked(value)
    # The sole allowed Infinity is encoded inside the hashed canonical text.
    # JSON scalar spellings also distinguish bool/int/float without Python's aliases.
    encoded = json.dumps(value, sort_keys=True, allow_nan=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def fixed_identity(acceptance, accepted_bindings, inputs):
    run, orchestration = Path(acceptance["run_dir"]), Path(acceptance["orchestration_dir"])
    require(run.parent == orchestration.parent == ROOT / "runs", "accepted run roots differ")

    def bound_text(path):
        require(str(path) in accepted_bindings, f"accepted metadata lacks binding: {path}")
        return inputs.text(path, accepted_bindings[str(path)])

    environment = json.loads(bound_text(run / "environment.json"))
    require(environment["slurm_job_id"] == acceptance["training_slurm_job_id"], "accepted environment allocation differs")
    require(json.dumps(environment["rank_placement"], sort_keys=True) == json.dumps(acceptance["rank_placement"], sort_keys=True),
            "accepted environment placement differs")
    result = {}
    for key in FIXED_IDENTITIES:
        require(type(acceptance.get(key)) is str and re.fullmatch(r"[0-9a-f]{64}", acceptance[key]), "missing fixed source/input identity")
        result[key] = acceptance[key]
    native = {}
    for label in ("megatron_lm", "megatron_bridge"):
        identity = environment[label]
        require(type(identity) is dict and all(type(identity.get(key)) is str for key in ("revision", "status", "diff_sha256")),
                "incomplete native source identity")
        for kind in ("revision", "status", "diff"):
            text = bound_text(orchestration / "source" / f"{label}_{kind}.txt").strip()
            field = "diff_sha256" if kind == "diff" else kind
            observed = hashlib.sha256(text.encode()).hexdigest() if kind == "diff" else text
            require(observed == identity[field], f"runtime/archived native source differs: {label}/{kind}")
        native[label] = identity
    for key in ("pytorch_version", "cuda_version", "nccl_version", "container_sqsh_sha256", "hf_model_id",
                "hf_model_revision", "gpu_name", "cuda_allocator_conf"):
        require(key in environment, f"runtime identity missing {key}")
        native[key] = environment[key]
    model_path = run / "config/model_manifest.json"
    bound_text(model_path)
    native["model_manifest_sha256"] = inputs.hashes[str(model_path)]
    result["native_runtime_identity"] = native
    configuration = yaml.safe_load(bound_text(run / "config/resolved.yaml"))
    result["normalized_configuration_sha256"] = normalized_config_identity(configuration, run, acceptance["communication_sms"])
    return result


def check_comparability(cases):
    require(len(cases) == 4 and [case["communication_sms"] for case in cases] == list(BUDGETS), "four ordered cases required")
    require(len({case["training_slurm_job_id"] for case in cases}) == 4, "four independent training jobs required")
    first = json.dumps(cases[0]["fixed_workload_identity"], sort_keys=True, allow_nan=False)
    require(all(json.dumps(case["fixed_workload_identity"], sort_keys=True, allow_nan=False) == first for case in cases),
            "cross-budget workload/source/configuration differs")


def load_case(path, budget, inputs):
    record = inputs.read(path)
    summary = check_record(record, budget)
    bound = bindings(record["input_files"])
    inputs.bind(ANALYZER, ANALYZER_SHA)
    acceptance_path = Path(record["acceptance_path"])
    require(str(acceptance_path) in bound, "metric analysis does not bind its acceptance proof")
    acceptance = inputs.read(acceptance_path, bound[str(acceptance_path)])
    require(all(acceptance.get(key) is True for key in ("valid_budget_run", "valid_training", "valid_callback_capture",
            "valid_native_identity", "valid_nsight_timeline")), "training/timeline proof is not accepted")
    require(acceptance["communication_sms"] == budget and acceptance["training_slurm_job_id"] == record["training_slurm_job_id"], "accepted allocation differs")
    inventory_path = acceptance_path.parent / "nsight_inventory.json"
    accepted_bindings = bindings(acceptance["input_files"])
    require(str(inventory_path) in bound and bound[str(inventory_path)] == accepted_bindings.get(str(inventory_path)), "node proof binding differs")
    inventory = inputs.read(inventory_path, bound[str(inventory_path)])
    require(len(inventory["nodes"]) == 4 and sorted(node["node_rank"] for node in inventory["nodes"]) == list(range(4)), "accepted node grid differs")
    require(all(all(node.get(key) is True for key in ("valid_nsight_timeline", "valid_timeline_contents", "report_export_binding_valid"))
                and node["payload_kernel_count"] == 15360 and node["exact_paired_payload_directions"] == 15360 for node in inventory["nodes"]), "accepted payload scope differs")
    accepted_mapping(record, acceptance, inventory, bound, accepted_bindings)
    identity = fixed_identity(acceptance, accepted_bindings, inputs)
    payload = record["payload_interval_artifact"]
    inputs.bind(Path(payload["path"]), payload["sha256"])
    return {**summary, "metrics_path": str(path), "metrics_sha256": inputs.hashes[str(path)],
            "acceptance_path": str(acceptance_path), "acceptance_sha256": bound[str(acceptance_path)],
            "fixed_workload_identity": identity}


def parse_cases(values):
    result = {}
    for value in values:
        key, separator, raw = value.partition("=")
        require(separator and key.isdigit() and int(key) in BUDGETS and int(key) not in result, "invalid or duplicate budget argument")
        result[int(key)] = Path(raw).resolve()
    require(set(result) == set(BUDGETS), "exactly budgets 12,20,32,40 are required")
    require(len(set(result.values())) == 4, "four distinct metric JSON files are required")
    return result


def plot(cases, output, *, test_only=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.6))
    titles = ("SM activity", "Instruction issue", "Compute warps in flight")
    colors = ("#3578a6", "#238b7e", "#b87912", "#8056a0")
    for axis, name, title in zip(axes, NAMES, titles):
        values = [case["views"]["during_any_payload"][name]["mean_percent"] for case in cases]
        bars = axis.bar([str(budget) for budget in BUDGETS], values, color=colors, width=.62)
        axis.set(title=title, xlabel="Payload block budget", ylabel="Pooled sample mean (%)", ylim=(0, min(100, max(1, max(values) * 1.25))))
        axis.yaxis.set_major_formatter(PercentFormatter(xmax=100))
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=.18)
        axis.set_axisbelow(True)
        axis.bar_label(bars, labels=[f"{value:.2f}%" for value in values], padding=4, fontsize=9)
    prefix = "TEST ONLY: " if test_only else ""
    fig.suptitle(prefix + "Sampled device activity during UCCL payload intervals", fontsize=14)
    fig.text(.5, .025, "All device work, including concurrent work; 16 GPUs × 5 updates per case. Axes use different ranges.\n"
             "Nsight diagnostics with different physical placement. Not achieved occupancy. No reserved-SM, causal, or speedup claim.",
             ha="center", va="bottom", fontsize=8.5)
    fig.tight_layout(rect=(0, .14, 1, .93))
    for suffix in ("png", "svg"):
        fig.savefig(output / f"hardware_budget_comparison.{suffix}", dpi=180)
    plt.close(fig)


def write_outputs(cases, output, inputs):
    csv_path = output / "hardware_budget_comparison.csv"
    with csv_path.open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["payload_blocks", "training_job", "view", "metric", "pooled_sample_mean_percent", "sample_count", "rank_update_windows", "physical_uuids", "known_payloads", "intervals_without_sample", "intervals_with_one_sample"])
        for case in cases:
            for view in VIEWS:
                for name in NAMES:
                    metric = case["views"][view][name]
                    writer.writerow([case["communication_sms"], case["training_slurm_job_id"], view, name, metric["mean_percent"], metric["sample_count"], 80, 16, 61440,
                                     case["coverage"][view]["intervals_without_sample"], case["coverage"][view]["intervals_with_one_sample"]])
    lines = ["# Four-budget sampled hardware comparison", "", "These are sampled whole-device means during any UCCL payload interval. Every case passes all 80 selected coverage windows and its accepted 61,440-payload event witness.", "",
             "| Payload blocks | Job | SM activity (%) | Instruction issue (%) | Compute-warp activity (%) | Selected samples |", "| --- | --- | ---: | ---: | ---: | ---: |"]
    for case in cases:
        view = case["views"]["during_any_payload"]
        lines.append(f"| {case['communication_sms']} | {case['training_slurm_job_id']} | " + " | ".join(f"{view[name]['mean_percent']:.3f}" for name in NAMES) + f" | {view[NAMES[0]]['sample_count']} |")
    lines += ["", "Means pool samples across all rank/update windows. Longer selected windows contribute more samples. They are not an equal-weight average of the 80 window means.", "",
              "The CSV also includes dispatch, combine, and optimizer-step selections. Their intervals overlap. No pooled quantile is reconstructed from window quantiles.", ""]
    lines += [f"- {limit}" for limit in LIMITS]
    lines += ["", "Input records:", ""] + [f"- Budget {case['communication_sms']}: [{Path(case['metrics_path']).parent.name}]({case['metrics_path']}), SHA-256 `{case['metrics_sha256']}`." for case in cases]
    (output / "hardware_budget_comparison.md").write_text("\n".join(lines) + "\n")
    plot(cases, output)
    inputs.recheck()
    artifacts = []
    for path in sorted(output.iterdir()):
        with path.open("rb") as stream:
            artifacts.append({"path": str(path), "sha256": hashlib.file_digest(stream, "sha256").hexdigest()})
    result = {"valid_four_budget_hardware_comparison": True, "budgets": list(BUDGETS), "analyzer_sha256": ANALYZER_SHA,
        "execution": {"hostname": socket.gethostname(), "python": sys.executable, "gpu_execution": False, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")},
        "aggregation": "sample-count-weighted arithmetic means reconstructed from all 80 window summaries",
        "whole_session_event_completeness": "unresolved", "claim_limits": LIMITS, "cases": cases,
        "input_files": [{"path": path, "sha256": digest} for path, digest in sorted(inputs.hashes.items())], "artifacts": artifacts}
    (output / "comparison.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")


def self_check():
    fixture = {"schema_version": 1, "valid_bound_metric_analysis": True, "communication_sms": 12,
        "input_files": [{"path": str(ANALYZER), "sha256": ANALYZER_SHA}], "metric_unit": "percent, source resolution one percentage point",
        "training_slurm_job_id": "3999999", "nodes": [], "rank_update_windows": [], "pooled_sample_summaries": {}}
    for node in range(4):
        fixture["nodes"].append({"node_rank": node, "metric_streams": [{"rank": rank, "node_rank": node, "uuid": f"{rank + 1:032x}", "type_id": rank % 4, "metric_names": list(NAMES)} for rank in range(node * 4, node * 4 + 4)]})
    for rank in range(16):
        for step in range(5, 10):
            index = rank * 5 + step - 5
            row = {"rank": rank, "node_rank": rank // 4, "step_index": step, "human_update": step + 1, "physical_uuid": f"{rank + 1:032x}", "type_id": rank % 4,
                "payload_kernels": 768, "payloads_outside_host_step_bounds": 0, "step_start_ns": 0, "step_end_ns": 1000000,
                "sample_coverage": {"valid_step_sample_coverage": True, "invalid_selected_metric_bundles": 0, "internal_gaps_over_two_periods": 0,
                    "sample_bundles": 1000, "nominal_period_ns": 1000, "first_boundary_distance_ns": 500, "last_boundary_distance_ns": 500,
                    "spacing_max_ns": 1000, "verified_window_start_ns": 0, "verified_window_end_ns": 1000000}, "views": {}}
            for view in VIEWS:
                count = 1000 if view == "optimizer_step_window" else index + 10
                row["views"][view] = {"selected_sample_bundles": count, "metrics_available": True, "intervals_within_verified_sample_window": True,
                    "invalid_selected_metric_bundles": 0, "intervals_without_sample": 0, "intervals_with_one_sample": 0, "interval_union_duration_ns": 500000,
                    "metrics": {name: {"count": count, "arithmetic_sample_mean_percent": 10 + index / 10 + metric,
                        "min_percent": 0, "median_percent": 10, "p95_percent": 30, "max_percent": 50} for metric, name in enumerate(NAMES)}}
            fixture["rank_update_windows"].append(row)
    for view in VIEWS:
        fixture["pooled_sample_summaries"][view] = {"all_80_windows_available": True, "metrics": {}}
        for name in NAMES:
            parts = [row["views"][view]["metrics"][name] for row in fixture["rank_update_windows"]]
            count = sum(part["count"] for part in parts)
            fixture["pooled_sample_summaries"][view]["metrics"][name] = {"count": count,
                "arithmetic_sample_mean_percent": math.fsum(part["count"] * part["arithmetic_sample_mean_percent"] for part in parts) / count, "min_percent": 0, "max_percent": 50}
    checked = check_record(fixture, 12)
    changes = [lambda d: d.update(valid_bound_metric_analysis=False), lambda d: d.update(communication_sms=20),
        lambda d: d["input_files"][0].update(sha256="0" * 64), lambda d: d["rank_update_windows"].pop(),
        lambda d: d["rank_update_windows"].__setitem__(1, copy.deepcopy(d["rank_update_windows"][0])),
        lambda d: d["nodes"][0]["metric_streams"][1].update(uuid=d["nodes"][0]["metric_streams"][0]["uuid"]),
        lambda d: d["rank_update_windows"][0].update(payload_kernels=767),
        lambda d: d["rank_update_windows"][0]["sample_coverage"].update(valid_step_sample_coverage=False),
        lambda d: d["rank_update_windows"][0]["sample_coverage"].update(spacing_max_ns=3000),
        lambda d: d["rank_update_windows"][0]["views"]["during_any_payload"].update(metrics_available=False),
        lambda d: d["rank_update_windows"][0]["views"]["during_any_payload"]["metrics"][NAMES[0]].update(count=999),
        lambda d: d["rank_update_windows"][0]["views"]["during_any_payload"]["metrics"][NAMES[0]].update(arithmetic_sample_mean_percent=float("nan")),
        lambda d: d["pooled_sample_summaries"]["during_any_payload"]["metrics"][NAMES[0]].update(count=1),
        lambda d: d["pooled_sample_summaries"]["during_any_payload"]["metrics"][NAMES[0]].update(arithmetic_sample_mean_percent=13.95)]
    for change in changes:
        bad = copy.deepcopy(fixture)
        change(bad)
        try:
            check_record(bad, 12)
        except (ValueError, KeyError, TypeError):
            continue
        raise AssertionError("invalid hardware comparison record accepted")
    for arguments in (("12=/tmp/a", "20=/tmp/b"), ("12=/tmp/a",) * 4):
        try:
            parse_cases(arguments)
        except ValueError:
            continue
        raise AssertionError("incomplete or duplicate budget matrix accepted")
    accepted, inventory, linked = {"rank_placement": []}, {"nodes": []}, {}
    for node in fixture["nodes"]:
        node["sqlite_path"] = f"/synthetic/node-{node['node_rank']}.sqlite"
        linked[node["sqlite_path"]] = "2" * 64
        owners = {}
        for slot in node["metric_streams"]:
            owner = (1 << 48) + ((10000 + slot["rank"]) << 24)
            slot.update(global_pid=owner, os_pid=10000 + slot["rank"], global_vm=1 << 48,
                        physical_gpu_id=slot["rank"] % 4, type_id=(1 << 48) + slot["rank"] % 4)
            owners[str(owner)] = slot["rank"]
            accepted["rank_placement"].append({"rank": slot["rank"], "slurm_node_id": str(node["node_rank"]),
                "physical_gpu_uuid": slot["uuid"]})
        inventory["nodes"].append({"node_rank": node["node_rank"], "path": node["sqlite_path"],
            "export": {"sqlite_sha256": "2" * 64}, "rank_by_global_pid": owners})
    accepted_mapping(fixture, accepted, inventory, linked, linked)

    def swap_metric_uuids(record, _accepted, _inventory, _bindings):
        left, right = record["nodes"][0]["metric_streams"][:2]
        left["uuid"], right["uuid"] = right["uuid"], left["uuid"]
        for row in record["rank_update_windows"]:
            if row["rank"] in (0, 1):
                row["physical_uuid"] = (left, right)[row["rank"]]["uuid"]

    mapping_changes = [swap_metric_uuids,
        lambda r, a, i, b: r["nodes"][0]["metric_streams"][0].update(global_pid=123 << 24),
        lambda r, a, i, b: r["nodes"][0].update(sqlite_path="/synthetic/wrong.sqlite"),
        lambda r, a, i, b: b.update({r["nodes"][0]["sqlite_path"]: "3" * 64}),
        lambda r, a, i, b: i["nodes"][0]["export"].update(sqlite_sha256="4" * 64),
        lambda r, a, i, b: a["rank_placement"][0].update(slurm_node_id="1"),
        lambda r, a, i, b: r["nodes"][0]["metric_streams"][0].update(type_id=2 << 48),
        lambda r, a, i, b: r["nodes"][0]["metric_streams"][0].update(global_vm=2 << 48),
        lambda r, a, i, b: r["nodes"][0]["metric_streams"][1].update(type_id=r["nodes"][0]["metric_streams"][0]["type_id"]),
        lambda r, a, i, b: r["nodes"][0]["metric_streams"][1].update(global_pid=r["nodes"][0]["metric_streams"][0]["global_pid"])]
    for change in mapping_changes:
        bad, proof, nodes, wrong_bound = copy.deepcopy((fixture, accepted, inventory, linked))
        change(bad, proof, nodes, wrong_bound)
        try:
            accepted_mapping(bad, proof, nodes, wrong_bound, linked)
        except (ValueError, KeyError, TypeError):
            continue
        raise AssertionError("unbound rank/process/GPU/SQLite mapping accepted")

    run = ROOT / "runs/synthetic12"
    config = {"model": {"moe_flex_dispatcher_num_sms": 12, "seed": 1234},
        "logger": {"tensorboard_dir": str(run / "logs/tensorboard"), "wandb_exp_name": run.name,
                   "wandb_save_dir": str(run / "logs/wandb")}, "optimizer": {"grad_norm_skip_threshold": math.inf}}
    fixed_config = normalized_config_identity(config, run, 12)
    next_run = ROOT / "runs/synthetic20"
    next_config = copy.deepcopy(config)
    next_config["model"]["moe_flex_dispatcher_num_sms"] = 20
    next_config["logger"].update(tensorboard_dir=str(next_run / "logs/tensorboard"), wandb_exp_name=next_run.name,
                                  wandb_save_dir=str(next_run / "logs/wandb"))
    require(normalized_config_identity(next_config, next_run, 20) == fixed_config, "budget/output-only normalization failed")
    config_changes = [lambda c: c["model"].update(seed=5678),
        lambda c: c["logger"].update(wandb_save_dir="/wrong"), lambda c: c["model"].update(moe_flex_dispatcher_num_sms=True),
        lambda c: c["optimizer"].update(grad_norm_skip_threshold=-math.inf), lambda c: c["model"].update(seed=math.nan)]
    for change in config_changes:
        bad = copy.deepcopy(config)
        change(bad)
        try:
            require(normalized_config_identity(bad, run, 12) == fixed_config, "configuration drift")
        except (ValueError, KeyError, TypeError):
            continue
        raise AssertionError("configuration drift accepted")
    comparable = [{"communication_sms": budget, "training_slurm_job_id": str(3999000 + budget),
        "fixed_workload_identity": {**{key: "1" * 64 for key in FIXED_IDENTITIES},
                                    "native_runtime_identity": {"revision": "same"},
                                    "normalized_configuration_sha256": fixed_config}} for budget in BUDGETS]
    check_comparability(comparable)
    for key in FIXED_IDENTITIES + ("native_runtime_identity", "normalized_configuration_sha256"):
        bad = copy.deepcopy(comparable)
        bad[1]["fixed_workload_identity"][key] = "changed"
        try:
            check_comparability(bad)
        except ValueError:
            continue
        raise AssertionError("cross-case fixed identity drift accepted")
    cases = [{**checked, "communication_sms": budget} for budget in BUDGETS]
    with tempfile.TemporaryDirectory() as directory:
        plot(cases, Path(directory), test_only=True)
        require(all((Path(directory) / f"hardware_budget_comparison.{suffix}").stat().st_size > 0 for suffix in ("png", "svg")), "plot smoke failed")
    print(json.dumps({"self_check": "PASS", "record_rejections": len(changes), "matrix_rejections": 2,
        "mapping_rejections": len(mapping_changes), "configuration_rejections": len(config_changes),
        "cross_case_rejections": len(FIXED_IDENTITIES) + 2, "synthetic_plot": "PASS"}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", action="append", default=[], metavar="BUDGET=METRICS_JSON")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "mask CUDA for this CPU-only comparison")
    if args.self_check:
        self_check()
        return
    paths = parse_cases(args.metrics)
    require(args.output_dir is not None, "new output directory required")
    output = args.output_dir.resolve()
    require(output.parent == ROOT / "runs" and not output.exists(), "output must be a new runs child")
    inputs = Inputs()
    inputs.bind(Path(__file__).resolve())
    cases = [load_case(paths[budget], budget, inputs) for budget in BUDGETS]
    check_comparability(cases)
    inputs.recheck()
    output.mkdir()
    write_outputs(cases, output, inputs)
    print(json.dumps({"valid_four_budget_hardware_comparison": True, "output_dir": str(output)}))


if __name__ == "__main__":
    main()
