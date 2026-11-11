#!/usr/bin/env python3
"""Validate the new overlap schedule in four full-run Nsight reports.

The ordinary-schedule validators are never called. Their unchanged SQL reader
receives counts from the explicit overlap contract, including 384 autograd
pairs and 768 payload directions per rank/update. Native exports and analysis
run only in an allocated configured container.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import socket
import sqlite3
import subprocess
import sys
import time

from analysis import uccl_budget_hardware_metrics as hardware
from profiling.ampa_profile import overlap_trace
from tools import ep_overlap_validate as execution_validator
from tools import uccl_budget_sweep_posthoc as sql
from tools.ep16_posthoc_proof import import_closure


ROOT = Path(__file__).resolve().parents[1]
STEPS = tuple(range(5, 10))
require, equal = sql.require, sql.equal


def write_new(path, value):
    execution_validator.write_new(path, value)


def checked_journal(record, *, rank, step, execution):
    """Use the new schedule's strict gate, never the ordinary journal gate."""
    execution_validator.checked_call(record, rank=rank, step=step, execution=execution)
    equal(execution["capture"], "nsys", "sole Nsight capture owner")
    counts = Counter(record["counts"])
    pairs = record["uccl_autograd_pairs"]
    counts.update({f"AMPA/UCCL_CALL/{call}/{direction}": 1
                   for call in range(384) for direction in ("FWD", "BWD")})
    return {"nvtx_counts": counts, "call_families": {row["call_id"]: row["family"] for row in pairs}}


def checked_training(path, orchestration, inputs, *, revalidated_record=None):
    """Compare the prerequisite with a fresh strict validation of raw inputs."""
    record = inputs.read(path)
    for key in ("valid_execution_and_accounting", "valid_training_health", "valid_send_receive_audit",
                "valid_all_rank_startup_and_completion"):
        equal(record.get(key), True, f"prerequisite execution proof {key}")
    equal(record.get("kind"), "ampa_ep_overlap_execution_accounting_validation", "new execution proof kind")
    equal(record.get("orchestration_dir"), str(orchestration), "same acquisition directory")
    equal(record.get("qualification_only"), False, "full ten-update acquisition required")
    equal(record.get("case"), "uccl_ordinary", "ordinary UCCL case")
    equal(record.get("capture"), "nsys", "Nsight capture case")
    require(type(revalidated_record) is dict and revalidated_record.get("valid_execution_and_accounting") is True,
            "a fresh strict execution validation of the raw acquisition is required")
    # Rehashing an editable proof's chosen input list cannot establish coverage
    # or derive placement. Re-run the canonical new validator, then compare the
    # measurement identities instead of accepting copied derived fields.
    for key in ("kind", "orchestration_dir", "run_dir", "case", "capture", "qualification_only",
                "training_slurm_job_id", "execution_contract", "completed_updates", "rank_placement",
                "source_manifest_sha256", "training_worker_sha256", "consumed_input_fingerprint_sha256",
                "immutable_inputs", "backend"):
        equal(record.get(key), revalidated_record.get(key), f"fresh raw execution identity {key}")
    record = revalidated_record
    bindings = record.get("input_files")
    require(type(bindings) is list and bindings and all(type(row) is dict and set(row) == {"path", "sha256"}
            for row in bindings), "strict execution input bindings are missing")
    bound = {row["path"]: row["sha256"] for row in bindings}
    equal(len(bound), len(bindings), "unique strict execution input bindings")
    for name, digest in bound.items():
        inputs.bind(Path(name), digest)
    validator = Path(execution_validator.__file__).resolve()
    equal(record.get("validator_sha256"), inputs.bind(validator), "current compatible execution validator")
    run = Path(record.get("run_dir", ""))
    require(run.is_absolute() and run.parent == ROOT / "runs" and run.is_dir() and not run.is_symlink(), "invalid actual training run")
    execution = execution_validator.checked_execution_contract(record["execution_contract"],
        case="uccl_ordinary", capture="nsys", steps=10, source_sha256=record["source_manifest_sha256"])
    equal(execution["capture_step_indices"], list(STEPS), "full measured capture window")
    equal(execution["communication_backend"], "uccl_ep", "actual UCCL backend")
    equal(inputs.read(run / "config/ep_overlap_contract.json"), execution, "actual source-bound execution contract")
    comparison = inputs.read(run / "config/comparison_contract.json")
    acquisition = inputs.read(orchestration / "acquisition.json")
    equal(acquisition.get("source_manifest", {}).get("sha256"), record["source_manifest_sha256"], "current acquisition source identity")
    worker = ROOT / "jobs/training/ep_overlap_worker.sh"
    worker_sha = acquisition["source_manifest"]["files"][str(worker.relative_to(ROOT))]
    inputs.bind(worker, worker_sha)
    job = record.get("training_slurm_job_id")
    require(type(job) is str and job.isdigit(), "missing actual training job")
    return {"valid_training": True, "run_dir": str(run), "communication_sms": execution["communication_sms"],
        "training_slurm_job_id": job, "rank_placement": record["rank_placement"],
        "worker_source_path": str(worker), "worker_source_sha256": worker_sha,
        "execution_contract": execution, "source_manifest_sha256": record["source_manifest_sha256"],
        "sample_prefix_sha256": record["consumed_input_fingerprint_sha256"],
        "c0_identity_manifest_sha256": comparison["c0_identity_manifest_sha256"],
        "dataset_manifest_sha256": comparison["dataset"]["manifest_sha256"]}


def callback_expectations(training, inputs):
    run, execution = Path(training["run_dir"]), training["execution_contract"]
    expected, boundaries = {}, []
    for rank in range(16):
        expected[rank] = {step: checked_journal(inputs.read(run / "raw/overlap_calls" / f"step-{step:05d}" / f"rank-{rank}.json"),
            rank=rank, step=step, execution=execution) for step in STEPS}
        capture = inputs.read(run / "raw" / f"ep_overlap_rank-{rank}_capture.json")
        equal(capture.get("contract"), overlap_trace.CONTRACT, "new overlap lifecycle contract")
        equal(capture.get("valid_step_execution"), True, "completed overlap execution")
        equal(capture.get("capture_mode"), "nsys", "actual Nsight lifecycle")
        equal(capture.get("capture_steps"), list(STEPS), "actual Nsight measured updates")
        equal([(row.get("action"), row.get("step"), row.get("rank")) for row in capture["boundaries"]],
              [("start", 5, rank), ("stop", 9, rank)], "all-rank Nsight boundary grid")
        for row in capture["boundaries"]:
            required = {"cuda_synchronize_seconds", "pre_api_global_barrier_seconds", "profiler_api_seconds"}
            if row["action"] == "start":
                required.add("post_start_global_barrier_seconds")
            require(all(type(row.get(key)) in (int, float) and math.isfinite(row[key]) and row[key] >= 0 for key in required),
                    "invalid boundary duration")
            boundaries.append(row)
    summaries = []
    for action in ("start", "stop"):
        selected = [row for row in boundaries if row["action"] == action]
        fields = [key for key in selected[0] if key.endswith("_seconds")]
        summaries.append({"action": action, "ranks": 16,
            "per_field_seconds": {key: {"min": min(row[key] for row in selected), "max": max(row[key] for row in selected)} for key in fields}})
    return expected, {"valid_callback_capture": True, "observer_contract": overlap_trace.CONTRACT,
        "rank_update_journals": 80, "steps": list(STEPS), "boundaries": boundaries, "boundary_summary": summaries,
        "boundary_claim_limit": "Host waits and profiler APIs are outside the NVTX step kernel window. Do not subtract them from full-step timers or sum across ranks."}


def launch_success(db, proof):
    """Add API success/end evidence when the export records those fields."""
    schema = sql.sqlite_schema(db)
    strings = dict(db.execute("SELECT id,value FROM StringIds"))
    owners = set(proof["trainer_global_pids"])
    payloads = {}
    for start, end, owner, correlation in db.execute(
            "SELECT k.start,k.end,k.globalPid,k.correlationId FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.demangledName=s.id "
            "WHERE s.value LIKE '%uccl::internode::dispatch<%' OR s.value LIKE '%uccl::internode::combine<%'"):
        key = owner, correlation
        require(key not in payloads and owner in owners and end > start, "duplicate or malformed payload correlation")
        payloads[key] = (start, end)
    witnesses, successes, tables = defaultdict(list), set(), []
    for table in ("CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_DRIVER"):
        fields = schema.get(table, [])
        if not fields:
            continue
        required = {"start", "globalTid", "correlationId"}
        require(required <= set(fields), "API table lacks correlation ownership")
        selected = ["start", "globalTid", "correlationId"] + [key for key in ("end", "nameId", "returnValue") if key in fields]
        complete = {"end", "nameId", "returnValue"} <= set(fields)
        tables.append({"table": table, "success_and_end_fields_available": complete})
        seen = set()
        for raw in db.execute(f"SELECT {','.join(sql.quoted(key) for key in selected)} FROM {sql.quoted(table)}"):
            row = dict(zip(selected, raw))
            owner = (row["globalTid"] >> 24) << 24
            key = owner, row["correlationId"]
            if key not in payloads:
                continue
            require(payloads[key][0] >= row["start"], "payload activity predates its exact launch API")
            require(key not in seen, "duplicate API correlation in one table")
            seen.add(key)
            if "nameId" in row:
                require(re.search(r"LaunchKernel|LaunchCooperativeKernel", strings.get(row["nameId"], "")),
                        "payload correlation refers to a nonlaunch API")
            if "end" in row:
                require(type(row["end"]) is int and row["end"] > row["start"], "nonpositive API launch interval")
            if "returnValue" in row:
                equal(row["returnValue"], 0, "payload launch API return status")
            if complete:
                successes.add(key)
            witnesses[key].append(table)
    require(set(witnesses) == set(payloads), "all payloads have launch API witnesses")
    return {"payload_launches": len(payloads), "api_witness_rows": sum(map(len, witnesses.values())),
        "successful_launches_with_end_fields": len(successes),
        "all_payload_launch_success_verified": set(successes) == set(payloads), "tables": tables,
        "unavailable_reason": None if set(successes) == set(payloads) else "Some API tables lack launch name, end or return-value fields.",
        "correlation_policy": "One launch witness per API table and owner/correlation. Distinct runtime/driver witnesses remain separate."}


def cuda_identities(db, proof, placement, node):
    """Map accepted payload owners without requiring hardware sample tables."""
    mappings = db.execute("SELECT gpuId,cudaId,pid,uuid FROM TARGET_INFO_CUDA_DEVICE").fetchall()
    gpu = dict(db.execute("SELECT id,uuid FROM TARGET_INFO_GPU"))
    slots = {slot["rank"]: slot for slot in placement if str(slot["slurm_node_id"]) == str(node)}
    result = {}
    for raw_owner, rank in proof["rank_by_global_pid"].items():
        owner = int(raw_owner)
        rows = [row for row in mappings if row[2] == ((owner >> 24) & ((1 << 24) - 1))]
        devices = [row[0] for row in db.execute("SELECT DISTINCT k.deviceId FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.demangledName=s.id "
            "WHERE k.globalPid=? AND (s.value LIKE '%uccl::internode::dispatch<%' OR s.value LIKE '%uccl::internode::combine<%')", (owner,))]
        require(len(devices) == 1, "payload owner lacks one observed CUDA device")
        selected = [row for row in rows if row[1] == devices[0]]
        require(len(selected) == 1, "payload owner lacks one process-specific device mapping")
        device, _, _, uuid = selected[0]
        physical_uuid = hardware.uuid_key(uuid or gpu.get(device))
        equal(physical_uuid, hardware.uuid_key(slots[rank]["physical_gpu_uuid"]), "accepted physical rank UUID")
        result[owner] = {"rank": rank, "node_rank": node, "physical_gpu_id": device,
                         "uuid": physical_uuid, "cuda_mappings": rows}
    return result


def hardware_views(db, strings, proof, training, node, launch):
    """Reuse the reviewed percentage schema and sampling rules unchanged."""
    version = dict(db.execute("SELECT name,value FROM META_DATA_EXPORT"))
    equal(version.get("EXPORT_PRODUCT_VERSION"), "2025.6.3.343", "reviewed hardware export schema")
    frequencies = [arg.split("=", 1)[1] for arg in launch["collector_argv_prefix"] if arg.startswith("--gpu-metrics-frequency=")]
    require(len(frequencies) == 1 and frequencies[0].isdigit() and int(frequencies[0]) > 0, "recorded metric rate is missing")
    period = 1_000_000_000 / int(frequencies[0])
    identities, keys = hardware.mappings(db, strings, proof, training["rank_placement"], node)
    steps, payloads = hardware.payload_intervals(db, strings, proof, identities, training["communication_sms"])
    streams = hardware.metric_streams(db, keys)
    windows, pooled = [], {view: [Counter() for _ in hardware.NAMES] for view in hardware.VIEWS}
    for (owner, step), bounds in sorted(steps.items()):
        identity, kernels = identities[owner], payloads[owner, step]
        stream = streams[identity["type_id"]]
        coverage, _, _ = hardware.coverage(stream["timestamps"], stream["valid"], *bounds, period)
        selections = {"optimizer_step_window": [bounds], "during_any_payload": [(row["start"], row["end"]) for row in kernels],
            "during_dispatch": [(row["start"], row["end"]) for row in kernels if row["family"] == "dispatch"],
            "during_combine": [(row["start"], row["end"]) for row in kernels if row["family"] == "combine"]}
        views = {}
        for view, intervals in selections.items():
            views[view], histograms = hardware.summarize_view(stream, intervals, coverage)
            if views[view]["metrics_available"]:
                for index, histogram in enumerate(histograms):
                    pooled[view][index].update(histogram)
        windows.append({"rank": identity["rank"], "node_rank": node, "step_index": step,
            "human_update": step + 1, "physical_uuid": identity["uuid"], "type_id": identity["type_id"],
            "step_start_ns": bounds[0], "step_end_ns": bounds[1], "sample_coverage": coverage,
            "payload_kernels": len(kernels), "views": views})
    return windows, pooled


def analyze(args):
    require(os.environ.get("SLURM_JOB_PARTITION") == "debug" and os.environ.get("SLURM_JOB_ID", "").isdigit()
            and Path(sys.prefix) == Path("/opt/venvs/megatron-bridge"), "allocated debug configured-container execution required")
    out, orchestration = args.output_dir, args.orchestration_dir
    require(out.is_absolute() and out.parent == ROOT / "runs" and not out.exists() and not out.is_symlink(), "output must be a unique direct project run")
    require(orchestration.is_absolute() and orchestration.parent == ROOT / "runs" and orchestration.is_dir()
            and not orchestration.is_symlink(), "invalid source acquisition")
    require(args.export_timeout_seconds > 0 and args.max_seconds > 0, "positive export timeout and analysis deadline required")
    out.mkdir()
    inputs, nodes, windows, metrics = hardware.Inputs(), [], [], None
    deadline = time.monotonic() + args.max_seconds
    pooled = {view: [Counter() for _ in hardware.NAMES] for view in hardware.VIEWS}
    result = {"schema_version": 1, "kind": "ampa_ep_overlap_nsys_validation", "valid_overlap_nsys_run": False,
        "valid_training": False, "valid_callback_capture": False, "valid_nsight_timeline": False,
        "observer_contract": overlap_trace.CONTRACT, "orchestration_dir": str(orchestration),
        "whole_session_event_completeness": "unresolved", "comparison_accepted": False,
        "execution": {"hostname": socket.gethostname(), "slurm_job_id": os.environ["SLURM_JOB_ID"],
                      "partition": "debug", "python": sys.executable, "gpu_execution": False},
        "claim_limits": ["Known source-bound payload/call set only; whole-session completeness remains unresolved.",
            "No GPU process or kernel uses a timestamp-nearness ownership guess.", "No unprofiled throughput, exclusive communication activity, physical traffic, achieved occupancy or exposed-latency claim.",
            "Device clocks from independent nodes are not aligned by this analysis.",
            "Hardware samples can straddle payload boundaries and cannot count instantaneous busy or free SMs.",
            "Metric means and quantiles summarize sampled points without time integration or interpolation.",
            "The internal deadline is best effort for reused readers. The external launcher bounds total analysis time.",
            "Boundary waits and profiler APIs are reported separately and are not subtracted from step timers."]}
    try:
        for name in import_closure(ROOT, ("tools/ep_overlap_nsys_posthoc.py",)):
            inputs.bind(ROOT / name)
        fresh = execution_validator.validate(orchestration, out / "verified_execution")
        inputs.bind(out / "verified_execution/execution_accounting.json")
        training = checked_training(args.execution_accounting, orchestration, inputs, revalidated_record=fresh)
        result.update(training)
        expected, lifecycle = callback_expectations(training, inputs)
        result.update(lifecycle)
        write_new(out / "capture_boundaries.json", lifecycle)
        inputs.bind(out / "capture_boundaries.json")
        run = Path(training["run_dir"])
        capability_root = Path(inputs.read(run / "nsight/node-0-launch.json")["capability_root"])
        capability = inputs.read(capability_root / "capabilities.json")
        inputs.bind(capability_root / "commands.json")
        intervals_path = out / "payload_intervals.jsonl"
        with intervals_path.open("x") as intervals:
            for node in range(4):
                item = {"node_rank": node, "valid_nsight_timeline": False}
                try:
                    require(time.monotonic() < deadline, "analysis deadline exhausted")
                    path, bound, exported = sql.export_or_find(run, out, node, inputs, capability,
                        timeout=min(args.export_timeout_seconds, max(1, int(deadline - time.monotonic()))),
                        training_job=training["training_slurm_job_id"])
                    parent = sql.torchrun_parent(training, node, inputs)
                    item.update(sql.inspect_sqlite(path, node, training, expected, parent))
                    item.update(export=exported, report_export_binding_valid=bound)
                    require(item.get("valid_timeline_contents") is True and bound, "known payload/call SQL validation failed")
                    # JSON serializes integer object keys as strings. The helper
                    # consumes the same serialized inventory schema as before.
                    serial = json.loads(json.dumps(item))
                    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as db:
                        db.execute("PRAGMA query_only=ON")
                        db.set_progress_handler(lambda: int(time.monotonic() > deadline), 10000)
                        item["launch_success"] = launch_success(db, item)
                        strings = dict(db.execute("SELECT id,value FROM StringIds"))
                        identities = cuda_identities(db, serial, training["rank_placement"], node)
                        _, payloads = hardware.payload_intervals(db, strings, serial, identities, training["communication_sms"])
                        item["payload_interval_count"] = sum(map(len, payloads.values()))
                        equal(item["payload_interval_count"], 4 * 5 * 768, "all node payload intervals")
                        for (owner, _), kernels in sorted(payloads.items()):
                            for kernel in kernels:
                                intervals.write(json.dumps({**kernel, "node_rank": node, "physical_uuid": identities[owner]["uuid"]}, sort_keys=True) + "\n")
                        if args.hardware_metrics:
                            try:
                                launch = inputs.read(run / "nsight" / f"node-{node}-launch.json")
                                local_windows, histograms = hardware_views(db, strings, serial, training, node, launch)
                                windows.extend(local_windows)
                                for view in hardware.VIEWS:
                                    for index in range(len(hardware.NAMES)):
                                        pooled[view][index].update(histograms[view][index])
                                item["hardware_metric_analysis_valid"] = True
                            except (RuntimeError, ValueError, KeyError, TypeError, sqlite3.Error) as error:
                                item["hardware_metric_analysis_valid"] = False
                                item["hardware_metric_error"] = f"{type(error).__name__}: {error}"
                    item["valid_nsight_timeline"] = True
                except (RuntimeError, ValueError, KeyError, TypeError, OSError, sqlite3.Error, subprocess.SubprocessError) as error:
                    item["error"] = f"{type(error).__name__}: {error}"
                nodes.append(item)
        result["valid_nsight_timeline"] = len(nodes) == 4 and all(item["valid_nsight_timeline"] for item in nodes)
        result["valid_overlap_nsys_run"] = result["valid_training"] and result["valid_callback_capture"] and result["valid_nsight_timeline"]
        result["payload_interval_artifact"] = {"path": str(intervals_path), "sha256": inputs.bind(intervals_path),
            "eligible_for_hardware_analysis": result["valid_overlap_nsys_run"], "time_unit": "nanoseconds",
            "expected_intervals": 16 * 5 * 768, "schema": "analysis.uccl_budget_hardware_metrics payload_intervals.jsonl"}
        if args.hardware_metrics:
            valid = (result["valid_overlap_nsys_run"] and all(item.get("hardware_metric_analysis_valid") is True for item in nodes)
                     and len(windows) == 80 and {(row["rank"], row["step_index"]) for row in windows} == {(rank, step) for rank in range(16) for step in STEPS})
            metrics = {"schema_version": 1, "valid_bound_metric_analysis": valid, "communication_sms": training["communication_sms"],
                "training_slurm_job_id": training["training_slurm_job_id"], "observer_contract": overlap_trace.CONTRACT,
                "metric_unit": "percent, source resolution one percentage point", "rank_update_windows": windows,
                "execution": result["execution"], "pooled_sample_summaries": {}, "claim_limits": result["claim_limits"] + [
                    "Sampled device-wide percentages during payload intervals include concurrent compute.",
                    "Sample selections overlap, and individual short kernels can contain zero samples."]}
            for view in hardware.VIEWS:
                available = valid and all(row["views"][view]["metrics_available"] for row in windows)
                metrics["pooled_sample_summaries"][view] = {"all_80_windows_available": available,
                    "metrics": {name: hardware.histogram_summary(hist) if available else None for name, hist in zip(hardware.NAMES, pooled[view])}}
            result["hardware_metrics"] = {"requested": True, "valid_bound_metric_analysis": valid,
                "path": str(out / "metrics.json")}
        else:
            result["hardware_metrics"] = {"requested": False, "valid_bound_metric_analysis": False,
                "status": "not_requested; eligible intervals remain available if all timeline gates pass"}
    except (RuntimeError, ValueError, KeyError, TypeError, OSError, sqlite3.Error) as error:
        result["error"] = f"{type(error).__name__}: {error}"
    write_new(out / "nsight_inventory.json", {"observer_contract": overlap_trace.CONTRACT, "nodes": nodes})
    inputs.bind(out / "nsight_inventory.json")
    try:
        inputs.recheck()
    except (ValueError, RuntimeError, OSError) as error:
        result.update(valid_overlap_nsys_run=False, valid_nsight_timeline=False, input_recheck_error=str(error))
        if "payload_interval_artifact" in result:
            result["payload_interval_artifact"]["eligible_for_hardware_analysis"] = False
        if metrics is not None:
            metrics["valid_bound_metric_analysis"] = False
            result["hardware_metrics"]["valid_bound_metric_analysis"] = False
    if metrics is not None:
        metrics["payload_interval_artifact"] = result["payload_interval_artifact"]
        metrics["input_files"] = [{"path": name, "sha256": digest} for name, digest in sorted(inputs.hashes.items())]
        write_new(out / "metrics.json", metrics)
        result["hardware_metrics"]["sha256"] = inputs.bind(out / "metrics.json")
    result["input_files"] = [{"path": name, "sha256": digest} for name, digest in sorted(inputs.hashes.items())]
    result["finished_utc"] = datetime.now(timezone.utc).isoformat()
    write_new(out / "acceptance.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--orchestration-dir", type=Path, required=True)
    parser.add_argument("--execution-accounting", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--export-timeout-seconds", type=int, default=120)
    parser.add_argument("--max-seconds", type=int, default=1200)
    parser.add_argument("--hardware-metrics", action="store_true")
    args = parser.parse_args()
    result = analyze(args)
    print(json.dumps({"valid_overlap_nsys_run": result["valid_overlap_nsys_run"], "output_dir": str(args.output_dir)}, sort_keys=True))
    if not result["valid_overlap_nsys_run"] or (args.hardware_metrics and not result.get("hardware_metrics", {}).get("valid_bound_metric_analysis")):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
