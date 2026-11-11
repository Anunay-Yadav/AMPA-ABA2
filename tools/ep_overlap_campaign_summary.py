#!/usr/bin/env python3
"""Summarize twelve accepted overlap cases on allocated compute only.

Keep optimizer timers, allocation elapsed time, posthoc cost, GPU activity,
logical routing, and sampled device percentages separate. Source acquisitions
remain read-only. The output is a new, self-contained evidence index.
"""
from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import socket
import statistics
import sys

import yaml

from analysis import accept_uccl_pilot as evidence
from analysis import uccl_budget_hardware_compare as hardware_plot
from analysis.extract_step_timing import select_timer_log, timing_rows
from analysis.summarize_training_evidence import summarize
from profiling import run_pretrain_ep_overlap as adapter
from tools import ep_overlap_validate as validator
from tools.ep16_posthoc_proof import import_closure


ROOT = Path(__file__).resolve().parents[1]
SOURCE_SHA = "ed74f4058652bb6f96dfa0496f70f7a888a150a2b2ee639df2d7edd499c8620a"
TOKENS_PER_STEP = 262144
EXPECTED = {
    "native-ordinary-off": ("ordinary", "off", 20),
    "native-ordinary-torch": ("ordinary", "torch", 20),
    "native-grouped-off": ("grouped", "off", 20),
    "native-grouped-torch": ("grouped", "torch", 20),
    "uccl-ordinary-off": ("uccl_ordinary", "off", 20),
    "uccl-ordinary-torch": ("uccl_ordinary", "torch", 20),
    "uccl-grouped-off": ("uccl", "off", 20),
    "uccl-grouped-torch": ("uccl", "torch", 20),
    **{f"uccl-ordinary-nsys{budget}": ("uccl_ordinary", "nsys", budget) for budget in (12, 20, 32, 40)},
}
METRICS = ("SMs Active [Throughput %]", "SM Issue [Throughput %]", "Compute Warps in Flight [Throughput %]")
VIEWS = ("optimizer_step_window", "during_any_payload", "during_dispatch", "during_combine")
NEW_SOURCES = ("tools/ep_overlap_campaign_summary.py", "jobs/analysis/ep_overlap_campaign_summary.sbatch",
               "tests/test_ep_overlap_campaign_summary.py",
               "ai/ep-overlap-summary-review/iteration-01/numerical_audit.py")
require, equal = evidence.require, evidence.equal


def write_new(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


class Inputs(evidence.Inputs):
    def __init__(self, output):
        super().__init__()
        self.output, self.archives = Path(output), []

    def archive(self, path, relative):
        path, target = Path(path), self.output / relative
        digest = self.bind(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        require(not target.exists() and not target.is_symlink(), "archive destination exists")
        with path.open("rb") as source, target.open("xb") as destination:
            shutil.copyfileobj(source, destination)
        equal(evidence.Inputs().bind(target), digest, "archived input bytes")
        self.archives.append({"original_path": str(path), "path": str(target), "sha256": digest})
        return digest


def tsv(path, columns):
    result = []
    for line in Path(path).read_text().splitlines():
        fields = line.split("\t")
        require(len(fields) == columns and all(fields), f"malformed {columns}-field ledger: {path}")
        result.append(fields)
    return result


def select_full_cases(accepted, acquisitions):
    """Select by accepted identity, never by the last attempt or directory glob."""
    acquired = {}
    for fields in acquisitions:
        require(len(fields) == 8, "acquisition ledger schema differs")
        label, attempt, case, capture, budget, steps, training, run_id = fields
        require(attempt.isdigit() and int(attempt) > 0 and training.isdigit()
                and budget.isdigit() and steps.isdigit(), "invalid acquisition identity")
        key = label, attempt, training, run_id
        require(key not in acquired, "duplicate acquisition identity")
        acquired[key] = {"label": label, "attempt": int(attempt), "case": case, "capture": capture,
                         "budget": int(budget), "steps": int(steps), "training_job": training, "id": run_id}
    selected, qualifications, seen_ids, seen_jobs = {}, [], set(), set()
    for fields in accepted:
        require(len(fields) == 5, "accepted ledger schema differs")
        label, attempt, training, posthoc, run_id = fields
        require(posthoc.isdigit() and training != posthoc and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_id),
                "invalid accepted run or job identity")
        require(run_id not in seen_ids and training not in seen_jobs and posthoc not in seen_jobs,
                "accepted runs and job IDs must be unique")
        seen_ids.add(run_id)
        seen_jobs.update((training, posthoc))
        key = label, attempt, training, run_id
        require(key in acquired, "accepted row lacks its exact acquisition attempt")
        row = {**acquired[key], "posthoc_job": posthoc}
        if label.startswith("pilot-"):
            equal(row["steps"], 2, "excluded qualification length")
            qualifications.append(row)
            continue
        require(label in EXPECTED, f"unexpected accepted full case: {label}")
        require(label not in selected, f"duplicate accepted full case: {label}")
        equal((row["case"], row["capture"], row["budget"]), EXPECTED[label], "accepted matrix selection")
        equal(row["steps"], 10, "full acquisition length")
        selected[label] = row
    equal(sorted(selected), sorted(EXPECTED), "exact twelve-case matrix; missing cases are not a partial summary")
    return [selected[label] for label in EXPECTED], qualifications


def descriptive(values):
    require(values and all(type(value) in (int, float) and math.isfinite(value) and value > 0 for value in values),
            "invalid optimizer timer")
    return {"count": len(values), "mean_ms": statistics.fmean(values), "median_ms": statistics.median(values),
            "sample_sd_ms": statistics.stdev(values) if len(values) > 1 else None,
            "min_ms": min(values), "max_ms": max(values),
            "aggregate_tokens_per_second": len(values) * TOKENS_PER_STEP / (sum(values) / 1000)}


def separate_timing(rows, training_accounting, posthoc_accounting):
    equal([(row["first_step"], row["last_step_exclusive"], row["optimizer_steps"]) for row in rows],
          [(step, step + 1, 1) for step in range(10)], "ten complete optimizer timer windows")
    values = [row["mean_optimizer_step_ms"] for row in rows]
    return {"optimizer_updates": rows, "optimizer_update_sum_seconds": sum(values) / 1000,
            "all_ten_statistics": descriptive(values), "measured_five_statistics": descriptive(values[5:]),
            "measured_human_updates": list(range(6, 11)),
            "training_allocation_elapsed_seconds": training_accounting["elapsed_seconds"],
            "posthoc_allocation_elapsed_seconds": posthoc_accounting["elapsed_seconds"],
            "claim_limit": "Optimizer timers, training allocation elapsed, and posthoc elapsed are separate quantities. Allocation elapsed includes startup and instrumentation work."}


def expand_nodes(value):
    pieces, first, depth = [], 0, 0
    for index, character in enumerate(value):
        depth += character == "["
        depth -= character == "]"
        require(depth in (0, 1), "unsupported Slurm node expression")
        if character == "," and depth == 0:
            pieces.append(value[first:index])
            first = index + 1
    require(depth == 0, "unclosed Slurm node expression")
    pieces.append(value[first:])
    result = []
    for piece in pieces:
        match = re.fullmatch(r"([A-Za-z0-9_.-]*)\[([0-9,-]+)\]([A-Za-z0-9_.-]*)", piece)
        if match:
            for item in match[2].split(","):
                bounds = item.split("-")
                require(len(bounds) in (1, 2) and all(part.isdigit() for part in bounds), "invalid node range")
                low, high = int(bounds[0]), int(bounds[-1])
                require(0 <= high - low <= 1024, "invalid or oversized node range")
                result.extend(f"{match[1]}{number:0{len(bounds[0])}d}{match[3]}" for number in range(low, high + 1))
        else:
            require(re.fullmatch(r"[A-Za-z0-9_.-]+", piece), "unsupported node name")
            result.append(piece)
    require(len(set(result)) == len(result), "duplicate accounting nodes")
    return sorted(result)


def read_accounting(path, expected_jobs):
    records = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        fields = line.split("|")
        if fields[-1] == "":
            fields.pop()
        require(len(fields) == 9, "Slurm accounting requires nine declared fields")
        job, state, exit_code, start, end, elapsed, nodes, partition, count = fields
        require(job in expected_jobs and job not in records, "unexpected or duplicate accounting job")
        equal((state, exit_code, partition), ("COMPLETED", "0:0", "debug"), "successful accepted debug allocation")
        require(elapsed.isdigit() and int(elapsed) > 0 and count.isdigit(), "missing accounting elapsed/node count")
        dates = []
        for value in (start, end):
            require(re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d", value), "accounting UTC timestamp unavailable")
            dates.append(datetime.fromisoformat(value).replace(tzinfo=timezone.utc))
        require(dates[1] >= dates[0], "accounting end precedes start")
        equal((dates[1] - dates[0]).total_seconds(), float(elapsed), "UTC accounting span versus elapsed seconds")
        expanded = expand_nodes(nodes)
        equal(len(expanded), int(count), "accounting allocation node count")
        records[job] = {"job_id": job, "state": state, "exit_code": exit_code, "partition": partition,
            "start_utc": dates[0].isoformat(), "end_utc": dates[1].isoformat(), "elapsed_seconds": int(elapsed),
            "utc_span_seconds": (dates[1] - dates[0]).total_seconds(), "node_list": nodes,
            "nodes": expanded, "allocated_nodes": int(count), "timezone": "UTC"}
    equal(sorted(records), sorted(expected_jobs), "all24 accepted training/posthoc accounting records")
    return records


def bind_declared(record, inputs, required=()):
    bindings = record.get("input_files")
    require(type(bindings) is list and bindings, "accepted proof lacks raw input bindings")
    mapped = {}
    for item in bindings:
        require(type(item) is dict and set(item) == {"path", "sha256"}, "invalid raw reference schema")
        path = Path(item["path"])
        require(path.is_absolute() and str(path) not in mapped, "duplicate or relative raw reference")
        inputs.bind(path, item["sha256"])
        mapped[str(path)] = item["sha256"]
    require(set(map(str, required)) <= set(mapped), "accepted proof omits required raw evidence")
    return mapped


def checked_torch(row, run, posthoc, execution, inputs, source_files, archive):
    summary_path = posthoc / "gpu_activity/summary.json"
    summary = inputs.read(summary_path)
    equal(summary.get("capture_count"), 160, "all-rank/all-update GPU analysis count")
    equal(summary.get("source_manifest_sha256"), SOURCE_SHA, "GPU summary source")
    grid, positives, scopes = set(), 0, set()
    for item in summary.get("captures", []):
        key = item.get("rank"), item.get("step")
        require(key not in grid, "duplicate GPU analysis capture")
        grid.add(key)
        path = posthoc / "gpu_activity" / f"step-{key[1]:05d}-rank-{key[0]}.json"
        record = inputs.read(path)
        equal(record.get("valid_known_payload_witness"), execution["communication_backend"] == "uccl_ep", "backend-specific known payload proof")
        equal(record.get("native_forward_only"), execution["communication_backend"] == "alltoall", "native forward-only attribution scope")
        require(type(record.get("known_payload_kernels")) is int and record["known_payload_kernels"] > 0
                and type(record.get("correlated_compute_kernels")) is int and record["correlated_compute_kernels"] > 0,
                "GPU attribution lacks recorded payload or selected computation")
        equal(record.get("source_manifest_sha256"), SOURCE_SHA, "GPU analyzer source")
        for name, digest in summary.get("program_inputs", {}).items():
            equal(source_files.get(str(Path(name).relative_to(ROOT))), digest, "frozen GPU analysis program")
            inputs.bind(Path(name), digest)
        equal(record.get("program_inputs"), summary.get("program_inputs"), "GPU analysis program agreement")
        for name in item:
            equal(record.get(name), item[name], "GPU summary agrees with actual rank/update result")
        expected = {str(run / "traces/torch_profile" / f"step-{key[1]:05d}" / f"rank-{key[0]}.json.gz"),
                    str(run / "raw/overlap_calls" / f"step-{key[1]:05d}" / f"rank-{key[0]}.json")}
        equal(sorted(record.get("inputs", {})), sorted(expected), "GPU proof trace/call scope")
        for name, digest in record["inputs"].items():
            inputs.bind(Path(name), digest)
        equal(record.get("communication_backend"), execution["communication_backend"], "GPU proof backend")
        require(type(record.get("observed_positive_gpu_concurrency")) is bool, "missing GPU concurrency observation")
        positives += record["observed_positive_gpu_concurrency"]
        scopes.add(record.get("native_forward_only"))
    equal(sorted(grid), sorted((rank, step) for rank in range(16) for step in range(10)), "GPU evidence rank/update grid")
    equal(summary.get("positive_concurrency_capture_count"), positives, "GPU positive-count summary")
    require(positives > 0, "campaign's reviewed positive-concurrency gate failed")
    inputs.archive(summary_path, archive / "gpu_activity_summary.json")
    return {"captures": 160, "positive_concurrency_captures": positives, "native_forward_only": sorted(scopes),
            "claim_limit": summary.get("claim_limit"), "whole_session_completeness_claimed": False}


def checked_nsight(row, root, placement, inputs, archive):
    directory = Path(str(root) + "-nsys-posthoc")
    acceptance = inputs.read(directory / "acceptance.json")
    for name in ("valid_overlap_nsys_run", "valid_training", "valid_callback_capture", "valid_nsight_timeline"):
        equal(acceptance.get(name), True, f"Nsight accepted {name}")
    for name, value in {"training_slurm_job_id": row["training_job"], "communication_sms": row["budget"],
                        "source_manifest_sha256": SOURCE_SHA, "run_dir": str(root) + "-training"}.items():
        equal(acceptance.get(name), value, f"Nsight actual identity {name}")
    equal(acceptance.get("rank_placement"), placement, "Nsight actual placement")
    equal(acceptance.get("execution", {}).get("slurm_job_id"), row["posthoc_job"], "Nsight posthoc job")
    equal(acceptance.get("whole_session_event_completeness"), "unresolved", "bounded Nsight claim")
    bind_declared(acceptance, inputs, (directory / "nsight_inventory.json",))
    inventory = inputs.read(directory / "nsight_inventory.json")
    equal(sorted(item["node_rank"] for item in inventory["nodes"]), list(range(4)), "four accepted Nsight nodes")
    for item in inventory["nodes"]:
        for name in ("valid_nsight_timeline", "valid_timeline_contents", "report_export_binding_valid"):
            equal(item.get(name), True, f"Nsight node {name}")
        equal(item.get("payload_kernel_count"), 4 * 5 * 768, "exact known payload count per node")
        equal(item.get("exact_paired_payload_directions"), 4 * 5 * 768, "exact paired payload directions")
    metrics_path = directory / "metrics.json"
    inputs.bind(metrics_path, acceptance["hardware_metrics"]["sha256"])
    metrics = inputs.read(metrics_path)
    equal(metrics.get("valid_bound_metric_analysis"), True, "bound hardware analysis")
    equal(metrics.get("communication_sms"), row["budget"], "hardware communication budget")
    equal(metrics.get("training_slurm_job_id"), row["training_job"], "hardware training job")
    equal(metrics.get("execution", {}).get("slurm_job_id"), row["posthoc_job"], "hardware posthoc job")
    bind_declared(metrics, inputs)
    windows = metrics.get("rank_update_windows", [])
    equal(len(windows), 80, "eighty hardware windows")
    equal(sorted((item["rank"], item["step_index"]) for item in windows),
          sorted((rank, step) for rank in range(16) for step in range(5, 10)), "hardware rank/update grid")
    by_rank = {item["rank"]: item for item in placement}
    from analysis.uccl_budget_hardware_metrics import uuid_key
    for item in windows:
        equal(uuid_key(item["physical_uuid"]), uuid_key(by_rank[item["rank"]]["physical_gpu_uuid"]), "hardware actual GPU UUID")
        equal(str(item["node_rank"]), by_rank[item["rank"]]["slurm_node_id"], "hardware actual node")
        equal(item["payload_kernels"], 768, "hardware known payload count")
    views = metrics.get("pooled_sample_summaries", {})
    equal(views.get("during_any_payload", {}).get("all_80_windows_available"), True, "reviewed payload metric availability gate")
    for view in VIEWS:
        available = views.get(view, {}).get("all_80_windows_available")
        require(type(available) is bool, "hardware view availability is missing")
        equal(available, all(item["views"][view]["metrics_available"] for item in windows), "actual per-window metric availability")
        for name in METRICS:
            value = views[view]["metrics"].get(name)
            if available:
                require(type(value) is dict and type(value.get("count")) is int and value["count"] > 0,
                        "available hardware view lacks samples")
                require(type(value.get("arithmetic_sample_mean_percent")) in (int, float)
                        and math.isfinite(value["arithmetic_sample_mean_percent"])
                        and 0 <= value["arithmetic_sample_mean_percent"] <= 100, "invalid hardware percentage")
            else:
                equal(value, None, "unavailable hardware metrics must remain null")
    for name in ("acceptance.json", "nsight_inventory.json", "metrics.json"):
        inputs.archive(directory / name, archive / name)
    return {"rank_update_windows": 80, "payload_directions": 16 * 5 * 768, "views": views,
            "claim_limits": metrics.get("claim_limits", []), "instantaneous_free_sms": None}


def checked_case(row, accounting, inputs):
    label, run_id = row["label"], row["id"]
    root = ROOT / "runs" / run_id
    run, posthoc = ROOT / "runs" / f"{run_id}-training", ROOT / "runs" / f"{run_id}-posthoc"
    archive = Path("evidence") / label
    proof_path = posthoc / "execution_accounting.json"
    proof = inputs.read(proof_path)
    for name in ("valid_execution_and_accounting", "valid_training_health", "valid_send_receive_audit",
                 "valid_all_rank_startup_and_completion"):
        equal(proof.get(name), True, f"accepted execution {name}")
    for name, value in {"kind": "ampa_ep_overlap_execution_accounting_validation", "qualification_only": False,
        "completed_updates": 10, "case": row["case"], "capture": row["capture"], "training_slurm_job_id": row["training_job"],
        "run_dir": str(run), "orchestration_dir": str(root), "source_manifest_sha256": SOURCE_SHA}.items():
        equal(proof.get(name), value, f"selected accepted identity {name}")
    equal(proof.get("execution", {}).get("slurm_job_id"), row["posthoc_job"], "actual separate posthoc job")
    equal(proof.get("execution", {}).get("partition"), "debug", "posthoc partition")
    critical = [root / name for name in ("acquisition.json", "acquisition_completed.json", "campaign-source.sha256")]
    critical += [run / "config" / name for name in ("comparison_contract.json", "resolved.yaml", "ep_overlap_contract.json", "routing_ownership.json")]
    critical += [run / "environment.json"]
    critical += [run / "raw" / f"{stem}{rank}{suffix}" for rank in range(16)
                 for stem, suffix in (("sample_fingerprint_rank-", ".json"), ("routing_audit_rank-", ".json"),
                                      ("ep_overlap_rank-", "_startup.json"), ("ep_overlap_rank-", "_capture.json"))]
    bind_declared(proof, inputs, critical)
    equal(proof.get("validator_sha256"), inputs.bind(Path(validator.__file__)), "frozen execution validator")
    source = adapter.verify_source_manifest(root / "campaign-source.sha256", SOURCE_SHA)
    source_files = source["files"]
    acquisition = inputs.read(root / "acquisition.json")
    for name, value in {"job": row["training_job"], "run_dir": str(run), "case": row["case"],
        "capture": row["capture"], "execution_steps": 10, "partition": "debug", "campaign_source_sha256": SOURCE_SHA}.items():
        equal(acquisition.get(name), value, f"actual acquisition {name}")
    equal(acquisition.get("source_manifest"), source, "actual full source inventory")
    completed = inputs.read(root / "acquisition_completed.json")
    equal(completed.get("success"), True, "completed acquisition")
    equal(completed.get("workers_srun_returned_zero"), True, "successful training workers")
    equal(completed.get("job"), row["training_job"], "completed acquisition training job")
    execution = validator.checked_execution_contract(inputs.read(run / "config/ep_overlap_contract.json"),
        case=row["case"], capture=row["capture"], steps=10, source_sha256=SOURCE_SHA)
    equal(execution.get("communication_sms"), row["budget"] if row["case"] in evidence.UCCL_CASES else None,
          "actual UCCL budget; native has no UCCL budget")
    equal(proof.get("execution_contract"), execution, "accepted actual execution contract")
    contract = validator.checked_training_contract(inputs.read(run / "config/comparison_contract.json"), execution, row["case"])
    resolved_path = run / "config/resolved.yaml"
    resolved = yaml.safe_load(resolved_path.read_text())
    validator.checked_resolved(resolved, execution, row["case"], run)
    # Keep selected overlap/DDP fields. Only checked routing/backend/budget,
    # logger destinations and the external NVTX switch differ across cells.
    historical = copy.deepcopy(resolved)
    historical["train"]["exit_interval"] = 2
    common_resolved = evidence._resolved(historical, row["case"], run)
    equal(common_resolved["model"].pop("moe_flex_dispatcher_num_sms"), execution["communication_sms"], "resolved case budget")
    equal(common_resolved["profiling"].pop("nvtx_ranges"), row["capture"] == "nsys", "resolved external NVTX switch")
    environment = inputs.read(run / "environment.json")
    equal(environment.get("slurm_job_id"), row["training_job"], "actual training runtime job")
    placement = validator.validate_rank_placement(environment["rank_placement"], expected_world_size=16, launch_policy="torchrun_node4")
    validator.validate_runtime_groups(placement, expected_world_size=16, cp=1)
    equal(proof.get("rank_placement"), placement, "accepted versus actual placement")
    validator.checked_worker_logs(run, placement, inputs)
    immutable = validator.checked_inputs(root, run, acquisition, environment, contract, inputs, root=ROOT)
    equal(immutable, proof.get("immutable_inputs"), "actual immutable metadata and historical proof policy")
    fingerprints = {}
    for rank in range(16):
        path = run / "raw" / f"sample_fingerprint_rank-{rank}.json"
        fingerprints[str(rank)] = validator._checked_fingerprint_batches(inputs.read(path), rank=rank,
            accumulation=4, steps=10, token_shape=[1, 4096], path=path)
    prefix = hashlib.sha256(json.dumps(fingerprints, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    equal(proof.get("consumed_input_fingerprint_sha256"), prefix, "all640 actual CPU batches")
    fingerprint_path = inputs.output / archive / "actual_batches.json"
    write_new(fingerprint_path, {"batches": fingerprints, "sha256": prefix, "cpu_batches": 640,
                                 "scope": "New full campaign only; no old/new equality inferred from qualifications."})
    health = summarize(run, steps=10, interval=1)
    equal(inputs.read(posthoc / "training_health.json"), health, "fresh authoritative optimizer health")
    equal(health["final_consumed_samples"], 640, "actual full-run samples")
    timer_log = select_timer_log(run / "logs")
    inputs.bind(timer_log)
    all_timers = timing_rows(timer_log, warmup_steps=0, measured_steps=10, log_interval=1, tokens_per_step=TOKENS_PER_STEP)
    cached_timing = inputs.read(posthoc / "timing.json")
    equal(cached_timing.get("all_updates"), all_timers, "all ten actual step timers")
    equal(cached_timing.get("by_update"), all_timers[5:], "five measured complete step timers")
    ownership = inputs.read(run / "config/routing_ownership.json")
    audits = [inputs.read(run / "raw" / f"routing_audit_rank-{rank}.json") for rank in range(16)]
    received = validator.checked_receives(contract, ownership, audits, environment, 10)
    equal(inputs.read(posthoc / "received_routes.json"), received, "fresh integer send/receive reconciliation")
    lifecycle = validator.checked_lifecycle(run, execution, placement, row["training_job"], environment, inputs)
    receipt_path = ROOT / "runs" / f"{run_id}-posthoc-job-{row['posthoc_job']}" / "completed.json"
    receipt = inputs.read(receipt_path)
    for name, value in {"success": True, "experiment_id": run_id, "training_slurm_job_id": row["training_job"],
        "posthoc_slurm_job_id": row["posthoc_job"], "source_manifest_sha256": SOURCE_SHA,
        "orchestration_dir": str(root), "partition": "debug", "requested_nodes": 1}.items():
        equal(receipt.get(name), value, f"separate allocation receipt {name}")
    train_account, post_account = accounting[row["training_job"]], accounting[row["posthoc_job"]]
    equal(train_account["allocated_nodes"], 4, "training accounting node count")
    equal(post_account["allocated_nodes"], 1, "posthoc accounting node count")
    equal(train_account["nodes"], sorted({slot["hostname"] for slot in placement}), "training accounting versus actual hosts")
    equal(post_account["nodes"], [proof["execution"]["hostname"]], "analysis accounting versus actual host")
    require(train_account["end_utc"] <= post_account["start_utc"], "analysis starts before training allocation ends")
    timing = separate_timing(all_timers, train_account, post_account)
    gpu = None
    hardware = None
    if row["capture"] == "torch":
        gpu = checked_torch(row, run, posthoc, execution, inputs, source_files, archive)
    elif row["capture"] == "nsys":
        hardware = checked_nsight(row, root, placement, inputs, archive)
    if row["case"] in evidence.UCCL_CASES:
        equal(proof.get("backend", {}).get("valid_backend_identity_and_cleanup"), True, "UCCL identity and cleanup")
        equal(proof["backend"]["binary_sha256"], acquisition["binary_sha256"], "UCCL native binary identity")
        for rank in range(16):
            equal(inputs.read(run / "raw" / f"uccl_backend_rank-{rank}_runtime.json").get("num_sms"), row["budget"], "actual runtime communication budget")
    else:
        equal(proof.get("backend"), None, "native case has no UCCL runtime")
    for name in ("execution_accounting.json", "timing.json", "training_health.json", "received_routes.json", "recorded_capture_lifecycle.json"):
        inputs.archive(posthoc / name, archive / name)
    for path, name in ((receipt_path, "posthoc_receipt.json"), (root / "acquisition.json", "acquisition.json"),
                       (run / "config/comparison_contract.json", "comparison_contract.json"),
                       (resolved_path, "resolved.yaml"), (run / "environment.json", "environment.json"), (timer_log, "timer.log")):
        inputs.archive(path, archive / name)
    common = {"source_manifest_sha256": SOURCE_SHA, "immutable_manifests": immutable["immutable_manifests"],
        "input_prefix_sha256": prefix, "model": contract["model"], "parallelism": contract["parallelism"],
        "dataset": contract["dataset"], "training": contract["training"], "logging": contract["logging"],
        "model_overlap_flags": execution["overlap_flags"], "ddp_delay_wgrad_compute": execution["ddp_delay_wgrad_compute"],
        "cuda_connections": execution["CUDA_DEVICE_MAX_CONNECTIONS"],
        "frameworks": {name: environment[name] for name in ("megatron_lm", "megatron_bridge", "pytorch_version", "cuda_version", "nccl_version", "container_sqsh_sha256")}}
    result = {**row, "run_dir": str(run), "orchestration_dir": str(root), "proof": str(proof_path),
        "timing": timing, "training_accounting": train_account, "posthoc_accounting": post_account,
        "rank_placement": placement, "cpu_batches": 640, "input_prefix_sha256": prefix,
        "logical_assignments": received["total_source_expert_assignments"], "gpu_activity": gpu, "hardware": hardware,
        "capture_lifecycle": {"captured_steps": lifecycle["captured_steps"], "nsight_boundaries": lifecycle["nsight_boundaries"]},
        "headline_timing_eligible": row["capture"] == "off", "whole_session_event_completeness": "unresolved"}
    return result, common, common_resolved


def write_csv(path, rows):
    require(rows, "empty output table")
    with Path(path).open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_campaign(args):
    out, coordinator = args.output_dir, args.coordinator_dir
    require(out.is_absolute() and out.parent == ROOT / "runs" and not out.exists() and not out.is_symlink(),
            "summary output must be a new direct project run directory")
    require(coordinator.is_absolute() and coordinator.parent == ROOT / "runs" and coordinator.is_dir()
            and not coordinator.is_symlink(), "invalid retry coordinator directory")
    out.mkdir()
    inputs = Inputs(out)
    source = adapter.verify_source_manifest(args.summary_source_manifest, args.summary_source_sha256)
    summary_sources = set(import_closure(ROOT, ("tools/ep_overlap_campaign_summary.py",))) | set(NEW_SOURCES)
    missing_sources = sorted(summary_sources - set(source["files"]))
    require(not missing_sources, "summary source inventory omits required files: " + ", ".join(missing_sources))
    inputs.archive(args.summary_source_manifest, "source/summary-source.sha256")
    for name in sorted(summary_sources):
        inputs.bind(ROOT / name, source["files"][name])
        inputs.archive(ROOT / name, Path("source") / name)
    for name in ("accepted.tsv", "acquisitions.tsv", "ACQUISITIONS_COMPLETE.txt", "posthoc-jobs.tsv", "qualifications.tsv",
                 "controller.sh", "controller.sha256"):
        inputs.archive(coordinator / name, Path("evidence/coordinator") / name)
    if (coordinator / "failed-attempts.tsv").exists():
        inputs.archive(coordinator / "failed-attempts.tsv", "evidence/coordinator/failed-attempts.tsv")
    selected, qualifications = select_full_cases(tsv(coordinator / "accepted.tsv", 5), tsv(coordinator / "acquisitions.tsv", 8))
    jobs = {row[name] for row in selected for name in ("training_job", "posthoc_job")}
    equal(len(jobs), 24, "twenty-four distinct accepted allocations")
    inputs.archive(args.accounting_file, "evidence/accounting_utc.psv")
    accounting = read_accounting(args.accounting_file, jobs)
    cases, reference, reference_resolved = [], None, None
    for row in selected:
        print(json.dumps({"stage": "case_started", "label": row["label"], "training_job": row["training_job"]}), flush=True)
        result, common, resolved = checked_case(row, accounting, inputs)
        if reference is None:
            reference, reference_resolved = common, resolved
        equal(common, reference, "common full workload, source, complete actual batches and runtime")
        evidence.equal_resolved_configuration(resolved, reference_resolved, "unchanged resolved configuration across twelve cases")
        cases.append(result)
    timers, eight, budgets = [], [], []
    for row in cases:
        for timer in row["timing"]["optimizer_updates"]:
            timers.append({"label": row["label"], "run_id": row["id"], "training_job": row["training_job"],
                "posthoc_job": row["posthoc_job"], "capture": row["capture"], "human_update": timer["first_step"] + 1,
                "measured": timer["first_step"] >= 5, "optimizer_step_ms": timer["mean_optimizer_step_ms"],
                "tokens_per_step": TOKENS_PER_STEP, "tokens_per_second": timer["tokens_per_second"]})
        stats = row["timing"]["measured_five_statistics"]
        flat = {"label": row["label"], "case": row["case"], "capture": row["capture"],
            "communication_budget": row["budget"] if row["case"] in evidence.UCCL_CASES else None,
            "training_job": row["training_job"], "posthoc_job": row["posthoc_job"],
            "measured_updates": 5, "mean_step_ms": stats["mean_ms"], "median_step_ms": stats["median_ms"],
            "sample_sd_ms": stats["sample_sd_ms"], "aggregate_tokens_per_second": stats["aggregate_tokens_per_second"],
            "all_ten_timer_seconds": row["timing"]["optimizer_update_sum_seconds"],
            "training_allocation_seconds": row["timing"]["training_allocation_elapsed_seconds"],
            "posthoc_allocation_seconds": row["timing"]["posthoc_allocation_elapsed_seconds"],
            "training_nodes": ",".join(row["training_accounting"]["nodes"]),
            "training_start_utc": row["training_accounting"]["start_utc"], "training_end_utc": row["training_accounting"]["end_utc"],
            "posthoc_start_utc": row["posthoc_accounting"]["start_utc"], "posthoc_end_utc": row["posthoc_accounting"]["end_utc"],
            "headline_timing_eligible": row["headline_timing_eligible"]}
        if row["capture"] == "nsys":
            for name in METRICS:
                flat[name + " during payload mean"] = row["hardware"]["views"]["during_any_payload"]["metrics"][name]["arithmetic_sample_mean_percent"]
            flat["whole_device_samples_include_concurrent_compute"] = True
            flat["instantaneous_free_sms"] = None
            budgets.append(flat)
        else:
            eight.append(flat)
    equal(len(timers), 120, "all120 accepted optimizer timers")
    equal((len(eight), len(budgets)), (8, 4), "separate timing and SM tables")
    write_csv(out / "optimizer_steps.csv", timers)
    write_csv(out / "eight_cases.csv", eight)
    write_csv(out / "sm_budgets.csv", budgets)
    # Reuse only the reviewed drawing function. Its old-schema main/load_case
    # gates are inapplicable to the independently validated overlap records.
    plot_cases = [{"communication_sms": row["budget"], "views": {"during_any_payload": {
        name: {"mean_percent": row["hardware"]["views"]["during_any_payload"]["metrics"][name]["arithmetic_sample_mean_percent"]}
        for name in METRICS}}} for row in sorted(cases, key=lambda item: item["budget"])
        if row["capture"] == "nsys"]
    equal([row["communication_sms"] for row in plot_cases], [12, 20, 32, 40], "four validated plot budgets")
    hardware_plot.plot(plot_cases, out)
    plots = [{"path": str(out / f"hardware_budget_comparison.{suffix}"),
              "sha256": evidence.Inputs().bind(out / f"hardware_budget_comparison.{suffix}")}
             for suffix in ("png", "svg")]
    result = {"schema_version": 1, "valid_twelve_case_summary": True, "full_cases": 12, "optimizer_steps": 120,
        "measured_optimizer_steps": 60, "actual_cpu_batches_per_case": 640, "training_source_sha256": SOURCE_SHA,
        "summary_source": source, "coordinator_dir": str(coordinator), "excluded_accepted_qualifications": qualifications,
        "common_workload": reference, "unchanged_resolved_configuration_verified": True,
        "allowed_resolved_differences": ["checked routing/backend fields", "UCCL payload budget", "logger destination paths", "external NVTX switch"],
        "cases": cases, "eight_case_table": eight, "sm_budget_table": budgets, "slurm_accounting": accounting,
        "plots": plots, "plotter_scope": "Only unchanged analysis.uccl_budget_hardware_compare.plot is reused after the new overlap schema passes its gates.",
        "research_conclusions_accepted": False, "old_campaign_comparison_performed": False,
        "same_physical_placement_claimed": False, "independent_replicates_per_cell": 1,
        "claim_limits": ["OFF update6-10 timers are primary. Torch and Nsight timers include their observation costs.",
            "Five sequential updates and sixteen ranks are not independent experimental replicates.",
            "Separate allocations retain placement, time, hardware and network confounds. No isolated causal speedup or optimal budget is claimed.",
            "All640 actual CPU batches are checked within each new full run and compared across the twelve new cases. Qualifications do not prove old/new full-prefix equality.",
            "GPU activity covers exact known payload and selected computation ancestry. Native attribution covers forward payloads only.",
            "CPU scopes, GPU interval unions/intersections, logical assignments and optimizer wall time are distinct and not additive partitions.",
            "Hardware percentages are device-wide during accepted payload intervals and include concurrent computation. Unavailable views stay unavailable.",
            "Hardware sampling periods can straddle kernel boundaries. No instantaneous free/active SM count, achieved occupancy, communication-only utilization, physical traffic or exposed latency is inferred.",
            "Nsight boundary waits/API costs are preserved separately and are not subtracted from optimizer timers.",
            "Whole-session GPU event completeness remains unresolved. Historical immutable input verification is reused without fresh bulk payload checks."],
        "execution": {"hostname": socket.gethostname(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                      "partition": os.environ.get("SLURM_JOB_PARTITION"), "python": sys.executable, "gpu_workload": False},
        "archive_manifest": inputs.archives}
    inputs.recheck()
    references = [{"path": name, "sha256": digest} for name, digest in sorted(inputs.hashes.items())]
    write_new(out / "input_references.json", {"input_files": references, "raw_artifacts_copied": False,
        "policy": "Raw artifact bytes remain in immutable original paths. Small proofs, logs, ledgers and program sources are archived here."})
    with (out / "input_references.sha256").open("x") as stream:
        for item in references:
            stream.write(f"{item['sha256']}  {item['path']}\n")
    result["input_reference_manifest"] = {"path": str(out / "input_references.json"), "sha256": evidence.Inputs().bind(out / "input_references.json")}
    result["finished_utc"] = datetime.now(timezone.utc).isoformat()
    write_new(out / "campaign_summary.json", result)
    with (out / "VALIDATED.txt").open("x") as stream:
        stream.write(f"valid_twelve_case_summary=true\nsummary_sha256={evidence.Inputs().bind(out / 'campaign_summary.json')}\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--coordinator-dir", type=Path, required=True)
    parser.add_argument("--accounting-file", type=Path, required=True, help="TZ=UTC sacct: JobIDRaw,State,ExitCode,Start,End,ElapsedRaw,NodeList,Partition,NNodes")
    parser.add_argument("--summary-source-manifest", type=Path, required=True)
    parser.add_argument("--summary-source-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    require(os.environ.get("SLURM_JOB_PARTITION") == "debug" and os.environ.get("SLURM_JOB_ID", "").isdigit()
            and Path(sys.prefix) == Path("/opt/venvs/megatron-bridge"), "allocated debug configured-container execution required")
    result = summarize_campaign(args)
    print(json.dumps({"valid_twelve_case_summary": True, "full_cases": result["full_cases"], "output_dir": str(args.output_dir)}))


if __name__ == "__main__":
    main()
