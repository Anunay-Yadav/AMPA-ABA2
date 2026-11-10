#!/usr/bin/env python3
"""Validate one ordinary-UCCL budget diagnostic and inventory its Nsight reports.

No GPU work runs here. Native report export requires allocated compute or an
explicit authorized CPU-only invocation. Missing hardware metrics never become
occupancy estimates, and unbound SQLite files remain inventory-only evidence.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import socket
import sqlite3
import subprocess
import sys

import yaml

from tools import uccl_ep_debug_validate as single
from tools import uccl_debug_campaign_validate as campaign
from tools import check_uccl_budget_runtime as runtime_checks
from profiling import run_pretrain_uccl_budget as adapter
from profiling.ampa_profile.uccl_trace import UcclMethodLabels


ROOT = Path(__file__).resolve().parents[1]
require, equal = single.require, single.equal
STEPS = tuple(range(5, 10))
PAYLOAD = re.compile(r"uccl::internode::(dispatch|combine)<")
PAIR = re.compile(r"^AMPA/UCCL_CALL/(\d+)/(FWD|BWD)$")


def bind_list(inputs):
    return campaign.bind_list(inputs)


def safe(value):
    if isinstance(value, bytes):
        return {"blob_hex": value.hex()}
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite": str(value)}
    return value


def quoted(value):
    return '"' + value.replace('"', '""') + '"'


def metadata(orchestration, output, inputs, *, library_map=None):
    case, leg, run = single.one_case(orchestration, ROOT, inputs)
    equal((case, leg), ("uccl_ordinary", "control"), "legacy routing/input-reuse row")
    acquisition = inputs.read(orchestration / "acquisition.json")
    sms, job = acquisition.get("communication_sms"), acquisition.get("job")
    require(type(sms) is int and sms in adapter.BUDGETS and type(job) is str and job.isdigit(), "invalid budget/job")
    for record in (acquisition, inputs.read(orchestration / "acquisition_completed.json")):
        for key, value in {"experiment_protocol": adapter.PROTOCOL, "communication_sms": sms,
                "case": case, "job": job, "partition": "debug", "warmup_steps": 5, "measured_steps": 5,
                "completed_updates": 10, "train_iters": 25, "cache_budget_samples": 1600, "exit_interval": 10,
                "kineto_profiler_enabled": False, "nsight_systems_enabled": True,
                "capture_step_indices": list(STEPS), "capture_ranks": list(range(16)), "run_dir": str(run)}.items():
            equal(record.get(key), value, f"budget acquisition {key}")
    completed = inputs.read(orchestration / "acquisition_completed.json")
    for key in ("success", "workers_srun_returned_zero", "training_health_and_timer_summary_complete"):
        equal(completed.get(key), True, f"acquisition completion {key}")
    equal(inputs.read(run / "config/nsight_budget_contract.json"),
          inputs.read(orchestration / "nsight_budget_contract.json"), "launcher contract copies")
    contract = single.checked_contract(inputs.read(run / "config/comparison_contract.json"), case)
    equal(contract.get("communication_budget"), adapter.execution_contract(sms), "actual budget contract")
    equal(inputs.read(run / "config/uccl_budget_contract.json"), adapter.execution_contract(sms), "budget contract copy")
    equal(inputs.read(run / "config/debug_execution_contract.json"), single.execution_contract(), "cache/exit contract")
    resolved_path = run / "config/resolved.yaml"
    inputs.bind(resolved_path)
    resolved = yaml.safe_load(resolved_path.read_text())
    single.checked_resolved(resolved, case, run)
    equal(resolved["model"].get("moe_flex_dispatcher_num_sms"), sms, "native manager budget")
    equal(resolved["profiling"].get("nvtx_ranges"), True, "native NVTX ranges")
    environment = inputs.read(run / "environment.json")
    equal(environment.get("slurm_job_id"), job, "runtime allocation")
    sources = environment["ampa_runtime_sources"]
    equal(sources, single._ampa_runtime_sources(ROOT), "current compatible canonical sources")
    equal(acquisition.get("runtime_source_sha256"), sources["sha256"], "recorded source identity")
    runtime = campaign.recorded_manifest(orchestration / "runtime-source.sha256", inputs, orchestration, ROOT)
    equal(runtime, {str(ROOT / name): value for name, value in sources["files"].items()}, "runtime source coverage")
    launch = campaign.recorded_manifest(orchestration / "source.sha256", inputs, orchestration, ROOT)
    campaign.recorded_manifest(orchestration / "campaign-source.sha256", inputs, orchestration, ROOT)
    adapter_path = str(ROOT / "profiling/run_pretrain_uccl_budget.py")
    require(adapter_path in launch, "launch inventory lacks budget adapter")
    equal(launch[adapter_path], inputs.bind(adapter_path), "compatible budget adapter source")
    worker_path = str(ROOT / "jobs/training/uccl_budget_sweep_worker.sh")
    require(worker_path in launch, "launch inventory lacks executed torchrun worker")
    equal(launch[worker_path], inputs.bind(worker_path), "executed torchrun worker source")
    for label in ("megatron_lm", "megatron_bridge"):
        for kind in ("revision", "status", "diff"):
            inputs.bind(orchestration / "source" / f"{label}_{kind}.txt")
    reviewed_path = orchestration / "reviewed_inputs.json"
    reviewed = inputs.read(reviewed_path)
    verification = campaign.verification_evidence(orchestration, acquisition, reviewed, inputs)
    for key, value in {"valid_reviewed_inputs": True, "slurm_job_id": job, "ampa_runtime_sources": sources}.items():
        equal(reviewed.get(key), value, f"reviewed input {key}")
    equal(environment["reviewed_inputs_proof"], {"path": str(reviewed_path), "sha256": inputs.bind(reviewed_path)},
          "actual historical verification reference")
    manifests = {name: inputs.bind(run / "config" / name) for name in ("c0_manifest.json", "dataset_manifest.json", "model_manifest.json")}
    equal(manifests["c0_manifest.json"], reviewed["c0_identity_manifest_sha256"], "copied C0 manifest")
    equal(manifests["dataset_manifest.json"], reviewed["dataset_manifest_sha256"], "copied data manifest")
    equal(contract["c0_identity_manifest_sha256"], manifests["c0_manifest.json"], "actual C0 identity")
    equal(contract["dataset"]["manifest_sha256"], manifests["dataset_manifest.json"], "actual dataset identity")
    context = inputs.read(orchestration / "input_reuse_context.json")
    equal(context["run_dir"], str(run), "reuse run")
    equal(context["slurm_job_id"], job, "reuse allocation")
    equal(context["indexed_manifest"], inputs.read(run / "config/dataset_manifest.json"), "reused indexed manifest")
    c0 = str(ROOT / "checkpoint/megatron/qwen3-30b-a3b-base-c0")
    equal(environment.get("checkpoint"), c0, "actual C0 path")
    equal(contract["initialization"], {"mode": "pretrained", "checkpoint": c0}, "C0 initialization")
    equal(inputs.bind(Path(c0) / "ampa_c0_manifest.json"), manifests["c0_manifest.json"], "canonical C0 manifest")
    placement = single.validate_rank_placement(environment["rank_placement"], expected_world_size=16, launch_policy="torchrun_node4")
    single.validate_runtime_groups(placement, expected_world_size=16, cp=1)
    mode = inputs.read(run / "config/observation_mode.json")
    for key, value in {"nsys_trace": True, "torch_trace": False, "sample_fingerprints": True,
                       "lightweight_profiling": "off"}.items():
        equal(mode.get(key), value, f"observation {key}")
    require(not any((run / "traces/torch_profile").rglob("*.json*")), "unexpected Kineto trace artifacts")
    fingerprints = {}
    for rank in range(16):
        inputs.bind(run / "logs" / f"rank-{rank}.out")
        path = run / "raw" / f"sample_fingerprint_rank-{rank}.json"
        fingerprints[str(rank)] = single._checked_fingerprint_batches(inputs.read(path), rank=rank, accumulation=4,
            steps=10, token_shape=[1, 4096], path=path)
    prefix = hashlib.sha256(json.dumps(fingerprints, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    health, timing = single.checked_timing(run)
    audits = [inputs.read(run / "raw" / f"routing_audit_rank-{rank}.json") for rank in range(16)]
    receives = single.checked_receives(contract, inputs.read(run / "config/routing_ownership.json"), audits, environment)
    build = Path(contract["uccl_build_dir"])
    identity = single.verify_uccl_source(build)
    equal(inputs.read(build / "source_identity.json"), identity, "approved UCCL source identity")
    inputs.bind(build / "source/uccl-borrowed-scratch.patch", identity["patch_sha256"])
    for relative, digest in identity["modified_files"].items():
        inputs.bind(build / "uccl" / relative, digest)
    for relative in ("ep/include/ep_config.hpp", "ep/src/internode.cu"):
        inputs.bind(build / "uccl" / relative)
    mapping = campaign.container_library_evidence(library_map, inputs, build) if library_map else None
    with campaign.mapped_libfabric_manifest(mapping):
        binary = single.evidence._uccl_identity(run, build, identity, placement, job, ROOT,
                                               campaign.IdentityInputs(inputs, run, output))
    equal(binary, acquisition["binary_sha256"], "actual native binary")
    for rank in range(16):
        runtime = inputs.read(run / "raw" / f"uccl_backend_rank-{rank}_runtime.json")
        equal(runtime["num_sms"], sms, "actual exercised Buffer budget")
    for name, record in (("training_health.json", health), ("timing.json", timing), ("received_routes.json", receives)):
        single.write_new(output / name, record)
        inputs.bind(output / name)
    return {"valid_training": True, "valid_native_identity": True, "run_dir": str(run), "communication_sms": sms,
        "experiment_protocol": adapter.PROTOCOL, "warmup_steps": 5, "measured_steps": 5, "completed_updates": 10,
        "training_slurm_job_id": job, "rank_placement": placement, "sample_prefix_sha256": prefix,
        "consumed_cpu_batches": 640, "verification": verification, "timing": timing,
        "c0_identity_manifest_sha256": manifests["c0_manifest.json"], "dataset_manifest_sha256": manifests["dataset_manifest.json"],
        "runtime_source_sha256": sources["sha256"], "binary_sha256": binary, "container_library_mapping": mapping,
        "adapter_source_sha256": launch[adapter_path],
        "worker_source_path": worker_path, "worker_source_sha256": launch[worker_path]}


def checked_torchrun_parent(record, node, training):
    """Bind the launcher role to the exact recorded training command and source."""
    for key, value in {"schema_version": 1, "kind": "ampa_nsight_torchrun_parent",
            "job": training["training_slurm_job_id"], "node_rank": node, "role": "torchrun_launcher",
            "source_path": training["worker_source_path"], "source_sha256": training["worker_source_sha256"]}.items():
        equal(record.get(key), value, f"torchrun parent {key}")
    require(type(record.get("schema_version")) is int and type(record.get("node_rank")) is int,
            "torchrun parent schema/node must be integers")
    require(type(record.get("pid")) is int and 0 < record["pid"] < (1 << 24), "invalid torchrun parent PID")
    command = record.get("command")
    require(type(command) is list and len(command) == 15 and all(type(arg) is str for arg in command),
            "torchrun parent lacks exact command argv")
    require(re.fullmatch(r"--master_addr=[^\s=]+", command[6]) is not None, "invalid torchrun master address")
    require(re.fullmatch(r"--master_port=[0-9]+", command[7]) is not None
            and 0 < int(command[7].split("=", 1)[1]) < 65536, "invalid torchrun master port")
    equal(command, ["/opt/venvs/megatron-bridge/bin/python", "-m", "torch.distributed.run",
        "--nnodes=4", "--nproc_per_node=4", f"--node_rank={node}", command[6], command[7],
        "--max_restarts=0", f"--log_dir={Path(training['run_dir']) / 'logs' / f'torchrun-node-{node}'}",
        "--redirects=3", "--no-python", "bash", training["worker_source_path"], "--worker"],
        "recorded torchrun command and worker mode")
    return record


def torchrun_parent(training, node, inputs):
    path = Path(training["run_dir"]) / "nsight" / f"node-{node}-torchrun-parent.json"
    return checked_torchrun_parent(inputs.read(path), node, training)


def journal(record, rank, step):
    for key, value in {"schema_version": 1, "kind": "ampa_nsight_uccl_calls", "rank": rank, "step": step,
                       "skipped_iter": False, "layers": list(range(48))}.items():
        equal(record.get(key), value, f"NVTX call journal {key}")
    pairs = record["uccl_autograd_pairs"]
    equal([row.get("call_id") for row in pairs], list(range(384)), "call ID grid")
    require(all(type(row.get("layer")) is int and row["layer"] in range(48)
                and row.get("family") in ("dispatch", "combine") for row in pairs), "invalid call layer/family")
    require(Counter((row["layer"], row["family"]) for row in pairs)
            == Counter({(layer, family): 4 for layer in range(48) for family in ("dispatch", "combine")}),
            "call layer/family grid differs")
    epochs = set()
    for row in pairs:
        require(all(type(row.get(key)) is int and row[key] > 0 for key in ("epoch", "context_id")), "invalid call context")
        equal(row.get("forward_returned"), True, "native forward returned")
        equal(row.get("backward_returned"), True, "native backward returned")
        epochs.add(row["epoch"])
    require(len(epochs) == 1, "call journal crosses capture epochs")
    checker = object.__new__(UcclMethodLabels)
    checker.counts, checker.layers = Counter(record["counts"]), record["layers"]
    checker.pairs = {row["call_id"]: row for row in pairs}
    checker.validate_counts(4)
    require(all(type(v) is int and v > 0 for v in record["counts"].values()), "invalid NVTX count")
    expected = Counter(record["counts"])
    expected.update({f"AMPA/UCCL_CALL/{n}/{direction}": 1 for n in range(384) for direction in ("FWD", "BWD")})
    return {"nvtx_counts": expected, "call_families": {row["call_id"]: row["family"] for row in pairs}}


def pinned_buffer_hints(sms, rdma_receive):
    """Exact EP16/four-peer/BF16 hints from the bound pinned Config/SourceMeta."""
    channels = sms // 2
    token_bytes = 4096 + 8 + 128 * 8 + 128 * 4 + 128 * 4
    nvl = channels * 4 * ((2 * 4 + 3) * 4 + 288 * token_bytes)
    rdma = channels * 4 * ((4 * 2 + 2) * 2 * 4 + rdma_receive * (token_bytes + 16) * 2)
    return {"nvl_buffer_size_hint": (nvl + 127) // 128 * 128,
            "rdma_buffer_size_hint": (rdma + 127) // 128 * 128}


def callbacks(training, inputs):
    run, sms, job = Path(training["run_dir"]), training["communication_sms"], training["training_slurm_job_id"]
    reference, expected = None, {}
    for rank, slot in enumerate(training["rank_placement"]):
        start = inputs.read(run / "raw" / f"uccl_budget_rank-{rank}_startup.json")
        for key, value in {"valid_budget_startup": True, "rank": rank, "communication_sms": sms, "slurm_job_id": job,
                "hostname": slot["hostname"], "local_rank": str(slot["local_rank"]), "run_dir": str(run),
                "native_buffer_allocated": False, "all_rank_configuration_agreement": True,
                "configuration_ranks_checked": 16, "contract": adapter.execution_contract(sms)}.items():
            equal(start.get(key), value, f"budget startup {key}")
        equal(start["adapter_source"], {"path": str(ROOT / "profiling/run_pretrain_uccl_budget.py"),
              "sha256": training["adapter_source_sha256"]}, "executed budget adapter")
        equal(start["override_environment"], {key: None for key in adapter.OVERRIDES}, "no inherited config overrides")
        configs = start["effective_configs"]
        # Config's constructor rounds RDMA receive capacity to a multiple of
        # the send chunk: input128 becomes effective140 or132, respectively.
        for direction, values in (("dispatch", [sms, 36, 288, 20, 140]), ("combine", [sms, 4, 288, 12, 132])):
            equal(configs[direction]["config"], dict(zip(adapter.CONFIG_FIELDS, values)), f"effective {direction} Config")
            equal(configs[direction]["hidden_bytes"], 4096, "buffer hidden bytes")
            equal(configs[direction]["ep_ranks"], 16, "buffer EP ranks")
            for key, value in pinned_buffer_hints(sms, values[4]).items():
                equal(configs[direction].get(key), value, f"pinned native {direction} {key}")
        if reference is None:
            reference = configs
        equal(configs, reference, "all-rank effective configs/hints")
        runtime = inputs.read(run / "raw" / f"uccl_backend_rank-{rank}_runtime.json")
        for key, hint in (("num_nvl_bytes", "nvl_buffer_size_hint"), ("num_rdma_bytes", "rdma_buffer_size_hint")):
            require(runtime[key] >= max(configs[d][hint] for d in configs), "allocated native buffer smaller than required hint")
        capture = inputs.read(run / "raw" / f"uccl_budget_rank-{rank}_capture.json")
        for key, value in {"valid_nvtx_capture": True, "rank": rank, "communication_sms": sms,
                           "completed_updates": 10, "capture_steps": list(STEPS)}.items():
            equal(capture.get(key), value, f"capture proof {key}")
        boundaries = capture["boundaries"]
        equal([(b.get("action"), b.get("step"), b.get("rank")) for b in boundaries],
              [("start", 5, rank), ("stop", 9, rank)], "capture API boundaries")
        for boundary in boundaries:
            keys = ["cuda_synchronize_seconds", "pre_api_global_barrier_seconds", "profiler_api_seconds"]
            if boundary["action"] == "start":
                keys.append("post_start_global_barrier_seconds")
            require(all(type(boundary.get(key)) in (int, float) and math.isfinite(boundary[key]) and boundary[key] >= 0
                        for key in keys), "invalid capture boundary durations")
        expected[rank] = {step: journal(inputs.read(run / "raw/nsys_calls" / f"step-{step:05d}" / f"rank-{rank}.json"), rank, step)
                          for step in STEPS}
    equal(inputs.read(run / "uccl_budget_environment.json"),
          inputs.read(run / "raw/uccl_budget_rank-0_startup.json"), "budget environment record")
    return expected, {"valid_callback_capture": True, "ranks": 16, "steps": list(STEPS),
                      "rank_update_journals": 80, "effective_configs": reference}


def sqlite_schema(db):
    names = [row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
    return {name: [row[1] for row in db.execute(f"PRAGMA table_info({quoted(name)})")] for name in names}


def metric_inventory(db, schema):
    result = {"hardware_metric_samples_present": False, "tables": {}, "metric_names": []}
    metadata = {}
    for table, columns in schema.items():
        if "GPU_METRIC" not in table.upper():
            continue
        count = db.execute(f"SELECT COUNT(*) FROM {quoted(table)}").fetchone()[0]
        result["tables"][table] = {"columns": columns, "rows": count,
            "sample_rows": [{key: safe(value) for key, value in zip(columns, row)}
                            for row in db.execute(f"SELECT * FROM {quoted(table)} LIMIT 8")]}
        if table.upper() == "GPU_METRICS" and count > 0:
            result["hardware_metric_samples_present"] = True
        if "metricName" in columns:
            result["metric_names"] += [safe(row[0]) for row in db.execute(f"SELECT DISTINCT metricName FROM {quoted(table)}")]
            for row in db.execute(f"SELECT * FROM {quoted(table)}"):
                item = dict(zip(columns, row))
                metadata[item.get("typeId"), item.get("metricId")] = item
    columns = schema.get("GPU_METRICS", [])
    if {"metricId", "value"} <= set(columns):
        group = [key for key in ("deviceId", "gpuId", "typeId", "metricId") if key in columns]
        keys = ",".join(quoted(key) for key in group)
        timestamps = ",MIN(timestamp),MAX(timestamp)" if "timestamp" in columns else ",NULL,NULL"
        query = f"SELECT {keys},COUNT(*),COUNT(value),MIN(value),AVG(value),MAX(value){timestamps} FROM GPU_METRICS GROUP BY {keys}"
        stats = []
        for row in db.execute(query):
            item = dict(zip(group, row[:len(group)]))
            item.update(zip(("sample_rows", "nonnull_values", "min_value", "mean_value", "max_value",
                             "first_timestamp_ns", "last_timestamp_ns"), map(safe, row[len(group):])))
            name = metadata.get((item.get("typeId"), item.get("metricId")), {})
            item["metric_name"] = name.get("metricName")
            item["source_id"] = name.get("sourceId")
            item["metric_type_name"] = name.get("typeName")
            stats.append(item)
        result["sample_value_statistics"] = stats
        result["device_grouping"] = "Explicit deviceId/gpuId where exported; typeId is retained without assuming a GPU mapping."
    result["claim_limit"] = "Actual sampled metric names/rows only. Device-wide samples can mix communication and computation. No static occupancy substitution."
    return result


def uuid_key(value):
    return re.sub(r"[^0-9a-f]", "", str(value).lower().removeprefix("gpu-"))


def interval_index(rows):
    rows = sorted(rows)
    high, prefix = -1, []
    for row in rows:
        high = max(high, row[1])
        prefix.append(high)
    return rows, [row[0] for row in rows], prefix


def containing(index, timestamp):
    rows, starts, prefix = index
    pos, matches = bisect_right(starts, timestamp) - 1, []
    while pos >= 0 and prefix[pos] > timestamp:
        if rows[pos][0] <= timestamp < rows[pos][1]:
            matches.append(rows[pos][2:])
        pos -= 1
    return matches


def outside_event_clients(db, schema, strings, diagnostics, owners, parent_global_pid, ranges):
    """Classify observed clients outside the exact event set, without role claims."""
    allowed = runtime_checks.EXPECTED_PARENT_WARNINGS | {
        "Not all CUDA events might have been collected.",
    }
    vm = next(iter({pid >> 48 for pid in owners}))
    candidates = {row["global_pid"] for row in diagnostics
        if type(row.get("severity")) is int and row["severity"] == 2 and row.get("text") in allowed
        and type(row.get("global_pid")) is int and row["global_pid"] > 0
        and row["global_pid"] & ((1 << 24) - 1) == 0
        and (row["global_pid"] >> 24) & ((1 << 24) - 1) > 0
        and row["global_pid"] >> 48 == vm
        and row["global_pid"] not in owners and row["global_pid"] != parent_global_pid}
    identity = defaultdict(list)
    for table in ("PROCESSES", "TARGET_INFO_PROCESS"):
        columns = schema.get(table, [])
        if not {"globalPid", "pid"} <= set(columns):
            continue
        selected = ["globalPid", "pid"] + (["name"] if "name" in columns else [])
        for raw in db.execute(f"SELECT {','.join(quoted(key) for key in selected)} FROM {quoted(table)}"):
            item = dict(zip(selected, raw))
            pid = item["globalPid"]
            if pid in candidates and type(item["pid"]) is int and item["pid"] == ((pid >> 24) & ((1 << 24) - 1)):
                identity[pid].append({"table": table, **item})
    if "globalPid" in schema.get("ProcessStreams", []):
        selected = ["globalPid"] + (["filenameId"] if "filenameId" in schema["ProcessStreams"] else [])
        for raw in db.execute(f"SELECT {','.join(selected)} FROM ProcessStreams"):
            item = dict(zip(selected, raw))
            if item["globalPid"] in candidates:
                if "filenameId" in item:
                    item["filename"] = strings.get(item["filenameId"])
                identity[item["globalPid"]].append({"table": "ProcessStreams", **item})
    if {"severity", "globalPid", "source", "text"} <= set(schema.get("DIAGNOSTIC_EVENT", [])):
        initialized = {pid for (pid,) in db.execute(
            "SELECT DISTINCT globalPid FROM DIAGNOSTIC_EVENT WHERE severity=1 AND source=1 AND text=?",
            ("Common injection library initialized successfully.",)) if pid in candidates}
        if {"globalTid", "nameId"} <= set(schema.get("ThreadNames", [])):
            for tid, name_id in db.execute("SELECT globalTid,nameId FROM ThreadNames"):
                if type(tid) is not int or tid <= 0:
                    continue
                pid = (tid >> 24) << 24
                name = strings.get(name_id)
                if pid in initialized and type(name) is str and bool(name.strip()) and 0 < tid - pid < (1 << 24):
                    identity[pid].append({"table": "ThreadNames+DIAGNOSTIC_EVENT", "global_tid": tid,
                        "thread_name": name, "initialization_severity": 1, "initialization_source": 1,
                        "initialization_text": "Common injection library initialized successfully."})
    kernel_counts = dict(db.execute("SELECT globalPid,COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL GROUP BY globalPid"))
    evidence = []
    for pid in sorted(candidates):
        scopes = sum(0 <= tid - pid < (1 << 24) for _, _, tid, _ in ranges)
        if identity[pid] and kernel_counts.get(pid, 0) == 0 and scopes == 0:
            evidence.append({"global_pid": pid, "os_pid": (pid >> 24) & ((1 << 24) - 1),
                "classification": "outside_verified_event_set", "process_identity": identity[pid],
                "observed_client_role": "unknown", "activity_outside_verified_event_set": "unknown",
                "exported_cuda_kernel_count": 0, "exported_ampa_scope_count": 0,
                "claim_limit": "No claim of CPU-only execution, idle status, source role, or whole-session completeness."})
    classified = {row["global_pid"] for row in evidence}
    outside = [row for row in diagnostics if row["severity"] == 2 and row["text"] in allowed
               and row["global_pid"] in classified]
    return evidence, outside


def inspect_sqlite(path, node, training, expected, parent_proof=None):
    """Join closed NVTX ranges to exact payload launches, not GPU-time proximity."""
    result = {"node_rank": node, "path": str(path), "valid_timeline_contents": False}
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as db:
        schema = sqlite_schema(db)
        result["schema"] = schema
        result["hardware_metrics"] = metric_inventory(db, schema)
        try:
            for table, fields in {"StringIds": ("id", "value"), "NVTX_EVENTS": ("start", "end", "globalTid"),
                    "CUPTI_ACTIVITY_KIND_KERNEL": ("start", "end", "globalPid", "deviceId", "contextId", "correlationId", "demangledName",
                        "gridX", "gridY", "gridZ", "blockX", "blockY", "blockZ"),
                    "TARGET_INFO_CUDA_DEVICE": ("gpuId", "cudaId", "pid")}.items():
                require(table in schema and set(fields) <= set(schema[table]), f"unsupported Nsight schema: {table}")
            strings = dict(db.execute("SELECT id,value FROM StringIds"))
            text = "COALESCE(n.text,s.value)" if "text" in schema["NVTX_EVENTS"] and "textId" in schema["NVTX_EVENTS"] else "n.text"
            require("text" in schema["NVTX_EVENTS"], "unsupported NVTX text schema")
            join = "LEFT JOIN StringIds s ON n.textId=s.id" if "textId" in schema["NVTX_EVENTS"] else ""
            ranges = [(start, end, tid, name) for start, end, tid, name in db.execute(
                f"SELECT n.start,n.end,n.globalTid,{text} FROM NVTX_EVENTS n {join} WHERE {text} LIKE 'AMPA/%'")]
            require(ranges and all(type(s) is int and type(e) is int and e > s and type(t) is int for s, e, t, _ in ranges),
                    "missing/open/malformed NVTX ranges")
            columns = schema["CUPTI_ACTIVITY_KIND_KERNEL"]
            wanted = ["start", "end", "globalPid", "deviceId", "contextId", "correlationId", "demangledName",
                      "gridX", "gridY", "gridZ", "blockX", "blockY", "blockZ"]
            wanted += [name for name in ("registersPerThread", "staticSharedMemory", "dynamicSharedMemory") if name in columns]
            fields = ",".join("k." + quoted(name) for name in wanted)
            payloads = []
            for raw in db.execute(f"SELECT {fields} FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.demangledName=s.id "
                                  "WHERE s.value LIKE '%uccl::internode::dispatch<%' OR s.value LIKE '%uccl::internode::combine<%'"):
                row = dict(zip(wanted, raw))
                row["kernel_name"] = strings[row["demangledName"]]
                row["family"] = PAYLOAD.search(row["kernel_name"])[1]
                require(row["end"] > row["start"], "nonpositive or truncated payload activity")
                require(row["gridX"] * row["gridY"] * row["gridZ"] == training["communication_sms"], "payload grid differs from budget")
                payloads.append(row)
            owners = sorted({row["globalPid"] for row in payloads})
            require(len(owners) == 4, "Nsight must capture four CUDA payload processes per node")
            require(type(parent_proof) is dict, "source-bound torchrun parent proof is required")
            parent = checked_torchrun_parent(parent_proof, node, training)
            vm_ids = {pid >> 48 for pid in owners}
            require(len(vm_ids) == 1, "CUDA owners disagree on the serialized VM identity")
            parent_global_pid = (next(iter(vm_ids)) << 48) + (parent["pid"] << 24)
            require(parent_global_pid not in owners, "recorded torchrun parent is an actual CUDA payload owner")
            require(db.execute("SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE globalPid=?",
                               (parent_global_pid,)).fetchone()[0] == 0, "recorded torchrun parent executed CUDA kernels")
            result.update(known_torchrun_parent=parent, parent_serialized_global_pid=parent_global_pid,
                          parent_cuda_kernel_count=0)

            def owner(tid):
                matches = [pid for pid in owners if 0 <= tid - pid < (1 << 24)]
                require(len(matches) == 1, "NVTX/launch thread lacks one observed CUDA process owner")
                return matches[0]

            cuda_uuid = "uuid" if "uuid" in schema["TARGET_INFO_CUDA_DEVICE"] else "NULL"
            cuda_devices = db.execute(f"SELECT gpuId,cudaId,pid,{cuda_uuid} FROM TARGET_INFO_CUDA_DEVICE").fetchall()
            physical_uuids = (dict(db.execute("SELECT id,uuid FROM TARGET_INFO_GPU"))
                              if {"id", "uuid"} <= set(schema.get("TARGET_INFO_GPU", [])) else {})
            slots = [slot for slot in training["rank_placement"] if str(slot["slurm_node_id"]) == str(node)]
            equal(len(slots), 4, "runtime workers on node")
            by_uuid = {uuid_key(slot["physical_gpu_uuid"]): slot["rank"] for slot in slots}
            ranks = {}
            contexts = {}
            for pid in owners:
                devices = {row["deviceId"] for row in payloads if row["globalPid"] == pid}
                require(len(devices) == 1, "one payload process uses multiple devices")
                device = next(iter(devices))
                process_pid = (pid >> 24) & ((1 << 24) - 1)
                mapped = [(gpu_id, uuid) for gpu_id, cuda_id, os_pid, uuid in cuda_devices
                          if os_pid == process_pid and cuda_id == device]
                require(len(mapped) == 1, "CUDA activity lacks one process-specific device mapping")
                gpu_id, uuid = mapped[0]
                require(uuid or gpu_id in physical_uuids, "process-specific CUDA mapping lacks a physical GPU UUID")
                key = uuid_key(uuid or physical_uuids.get(gpu_id))
                require(key in by_uuid, "Nsight GPU UUID differs from actual runtime placement")
                ranks[pid] = by_uuid[key]
                contexts[pid] = sorted({row["contextId"] for row in payloads if row["globalPid"] == pid})
                require(len(contexts[pid]) == 1, "payload process uses multiple CUDA contexts")
            equal(sorted(ranks.values()), sorted(slot["rank"] for slot in slots), "four distinct GPU/rank owners")
            step_ranges = {}
            for start, end, tid, label in ranges:
                match = re.fullmatch(r"AMPA/STEP/(\d+)", label)
                if match:
                    key = owner(tid), int(match[1])
                    require(key not in step_ranges, "duplicate NVTX optimizer-step range")
                    step_ranges[key] = (start, end)
            equal(sorted(step_ranges), sorted((pid, step) for pid in owners for step in STEPS), "four-process five-step NVTX grid")
            pair_ranges = defaultdict(list)
            for (pid, step), (start, end) in step_ranges.items():
                local = [(s, e, tid, label) for s, e, tid, label in ranges if label != f"AMPA/STEP/{step}"
                         and owner(tid) == pid and start <= s and e <= end]
                actual = Counter(label for _, _, _, label in local)
                equal(dict(actual), dict(expected[ranks[pid]][step]["nvtx_counts"]), "NVTX scopes versus source-bound call journal")
                for s, e, tid, label in local:
                    match = PAIR.fullmatch(label)
                    if match:
                        pair_ranges[pid, step, tid].append((s, e, int(match[1]), match[2]))
            pair_ranges = {key: interval_index(value) for key, value in pair_ranges.items()}
            ids = {(row["globalPid"], row["correlationId"]) for row in payloads}
            routes = defaultdict(set)
            for table in ("CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_DRIVER"):
                if table not in schema:
                    continue
                require({"start", "globalTid", "correlationId"} <= set(schema[table]), "unsupported launch-correlation schema")
                for timestamp, tid, correlation in db.execute(f"SELECT start,globalTid,correlationId FROM {quoted(table)}"):
                    matches = [pid for pid in owners if 0 <= tid - pid < (1 << 24)]
                    if len(matches) != 1 or (matches[0], correlation) not in ids:
                        continue
                    pid = matches[0]
                    for step in STEPS:
                        start, end = step_ranges[pid, step]
                        if start <= timestamp < end:
                            matches = containing(pair_ranges.get((pid, step, tid), ([], [], [])), timestamp)
                            require(len(matches) == 1, "payload launch lacks one same-thread UCCL call scope")
                            call_id, direction = matches[0]
                            routes[pid, correlation].add((step, call_id, direction))
            counts, durations, geometry, paired = Counter(), Counter(), {}, Counter()
            for row in payloads:
                candidates = routes[row["globalPid"], row["correlationId"]]
                require(len(candidates) == 1, "payload lacks one exact in-step launch correlation")
                step, call_id, direction = next(iter(candidates))
                rank = ranks[row["globalPid"]]
                call_family = expected[rank][step]["call_families"][call_id]
                correct = "dispatch" if (call_family == "dispatch") == (direction == "FWD") else "combine"
                equal(row["family"], correct, "native payload versus forward/adjoint call identity")
                paired[rank, step, call_id, direction] += 1
                key = rank, step, row["family"]
                counts[key] += 1
                durations[key] += row["end"] - row["start"]
                shape = (row["family"], row["gridX"], row["gridY"], row["gridZ"], row["blockX"], row["blockY"], row["blockZ"],
                         row.get("registersPerThread"), row.get("staticSharedMemory"), row.get("dynamicSharedMemory"))
                geometry[shape] = geometry.get(shape, 0) + 1
            require(dict(counts) == {(rank, step, family): 384 for rank in ranks.values() for step in STEPS
                                    for family in ("dispatch", "combine")}, "every rank-step payload launch count differs")
            require(dict(paired) == {(rank, step, call_id, direction): 1 for rank in ranks.values() for step in STEPS
                                    for call_id in range(384) for direction in ("FWD", "BWD")},
                    "one or more exact forward/adjoint calls lack one payload kernel")
            diagnostics = []
            if "DIAGNOSTIC_EVENT" in schema:
                diagnostics = [dict(zip(("severity", "global_pid", "text"), row)) for row in db.execute(
                    "SELECT severity,globalPid,text FROM DIAGNOSTIC_EVENT WHERE severity IN (2,3)")]
            ignored = [row for row in diagnostics if runtime_checks.expected_parent_warning(row, parent["pid"], owners)]
            advisories = [row for row in diagnostics if runtime_checks.generic_possible_loss_advisory(row, owners)]
            clients, outside = outside_event_clients(db, schema, strings, diagnostics, owners, parent_global_pid, ranges)
            concerns = [row for row in diagnostics if not runtime_checks.expected_parent_warning(row, parent["pid"], owners)
                        and not runtime_checks.generic_possible_loss_advisory(row, owners) and row not in outside]
            result.update(diagnostics=diagnostics, ignored_exact_parent_control_warnings=ignored,
                generic_possible_loss_advisories=advisories,
                clients_outside_verified_event_set=clients, outside_verified_event_set_diagnostics=outside,
                whole_session_event_completeness="unresolved",
                quantitative_event_scope="Known UCCL dispatch/combine payload directions and exact source-bound AMPA call scopes for all four ranks and five measured updates on this node.",
                event_quality_concerns=concerns,
                parent_diagnostic_claim_limit="Only exact control warnings from the source-bound launcher without CUDA kernels are excluded; no whole-process event-completeness claim.")
            require(not concerns, f"Nsight capture has event-quality concerns: {concerns}")
            result.update(valid_timeline_contents=True, trainer_global_pids=owners, rank_by_global_pid=ranks,
                cuda_contexts=contexts, closed_step_ranges=20, payload_kernel_count=len(payloads),
                exact_paired_payload_directions=len(paired),
                payload_by_rank_step=[{"rank": rank, "step": step, "family": family, "count": count,
                    "gpu_activity_sum_ns": durations[rank, step, family]} for (rank, step, family), count in sorted(counts.items())],
                geometry=[{"family_grid_block_registers_shared": list(shape), "count": count} for shape, count in geometry.items()],
                diagnostics=diagnostics,
                claim_limit="Exact known payload/call coverage, not completeness of unenumerated GPU compute, physical traffic, exposed delay, useful work, or achieved occupancy.")
        except (ValueError, RuntimeError, KeyError, TypeError, sqlite3.Error) as error:
            result["timeline_error"] = f"{type(error).__name__}: {error}"
    return result


def cached_export(cache_root, run, node, report, report_sha, inputs, capability, training_job):
    """Reuse only exports bound by the preceding validation and exact tool proof."""
    cache_root = Path(cache_root).resolve()
    require(cache_root.is_relative_to(ROOT / "runs") and cache_root.name == "nsight" and cache_root.is_dir(),
            "SQLite cache must be a prior runs output's nsight directory")
    path = cache_root / f"node-{node}.sqlite"
    sidecar = path.with_suffix(".source.json")
    previous = inputs.read(cache_root.parent / "acceptance.json")
    for key, value in {"run_dir": str(run), "training_slurm_job_id": training_job,
                       "valid_training": True, "valid_callback_capture": True}.items():
        equal(previous.get(key), value, f"SQLite cache prior validation {key}")
    bindings = previous.get("input_files")
    require(type(bindings) is list and all(type(row) is dict and set(row) == {"path", "sha256"} for row in bindings),
            "SQLite cache prior input bindings unavailable")
    bound = {row["path"]: row["sha256"] for row in bindings}
    equal(len(bound), len(bindings), "unique prior cache bindings")
    for item in (path, sidecar, report):
        require(str(item) in bound, f"prior validation does not bind cache evidence: {item}")
        inputs.bind(item, bound[str(item)])
    equal(bound[str(report)], report_sha, "cached native report identity")
    proof = inputs.read(sidecar)
    equal(proof.get("report_path"), str(report), "cached export native source path")
    equal(proof.get("report_sha256"), report_sha, "cached export native source SHA")
    equal(proof.get("sqlite_sha256"), bound[str(path)], "cached export SQLite SHA")
    tool = Path((capability or {}).get("nsys_path", "")).resolve()
    require(tool.is_file() and str(tool) in bound, "cached exporter lacks the bound capability tool")
    equal(proof.get("nsys_path"), str(tool), "cached exporter path")
    equal(proof.get("nsys_sha256"), bound[str(tool)], "cached exporter SHA")
    inputs.bind(tool, proof["nsys_sha256"])
    equal(proof.get("command"), [str(tool), "export", "--type=sqlite", f"--output={path}", str(report)],
          "cached exact export command")
    require(type(proof.get("hostname")) is str and bool(proof["hostname"]), "cached export execution host unavailable")
    require((type(proof.get("slurm_job_id")) is str and proof["slurm_job_id"].isdigit())
            or (proof.get("slurm_job_id") is None and proof.get("user_authorized_cpu_export") is True),
            "cached export execution provenance unavailable")
    return path, True, {**proof, "reused_verified_sqlite": True,
        "source_sidecar": str(sidecar), "previous_validation": str(cache_root.parent / "acceptance.json")}


def export_or_find(run, output, node, inputs, capability, *, skip_export=False, allow_login_cpu_export=False,
                   timeout=120, sqlite_cache_root=None, training_job=None):
    report = run / "nsight" / f"node-{node}-native.nsys-rep"
    report_sha = inputs.bind(report)
    require(report.stat().st_size > 0, "empty native Nsight report")
    launch = inputs.read(run / "nsight" / f"node-{node}-launch.json")
    equal(launch.get("node_rank"), node, "collector node identity")
    exit_path = run / "nsight" / f"node-{node}-exit.txt"
    inputs.bind(exit_path)
    require("exit_code=0" in exit_path.read_text().splitlines(), "Nsight collector did not exit successfully")
    if sqlite_cache_root is not None:
        require(not skip_export, "SQLite cache reuse and acquisition --skip-export are separate modes")
        return cached_export(sqlite_cache_root, run, node, report, report_sha, inputs, capability, training_job)
    if skip_export:
        path = run / "nsight" / f"node-{node}-native.sqlite"
        if not path.is_file():
            return None, False, {"report": str(report), "report_sha256": report_sha, "error": "no existing SQLite export"}
        sidecar = path.with_suffix(".source.json")
        bound = False
        if sidecar.is_file():
            proof = inputs.read(sidecar)
            equal(proof.get("report_path"), str(report), "existing export source path")
            equal(proof.get("report_sha256"), report_sha, "existing export source SHA")
            require(type(proof.get("sqlite_sha256")) is str and re.fullmatch(r"[0-9a-f]{64}", proof["sqlite_sha256"]),
                    "existing export lacks an exact SQLite SHA")
            inputs.bind(path, proof.get("sqlite_sha256"))
            bound = True
        else:
            inputs.bind(path)
        return path, bound, {"report": str(report), "report_sha256": report_sha, "existing_export_bound": bound}
    compute = bool(os.environ.get("SLURM_JOB_ID") and os.environ.get("SLURM_JOB_PARTITION") in ("debug", "preemptable"))
    require(compute or (allow_login_cpu_export and Path(sys.prefix).resolve() == campaign.CPU_VENV),
            "Nsight export requires allocated compute or explicit isolated CPU-only export authorization")
    tool = (capability or {}).get("nsys_path") or shutil.which("nsys")
    require(tool and Path(tool).is_file() and os.access(tool, os.X_OK), "Nsight exporter unavailable")
    inputs.bind(Path(tool).resolve())
    path = output / "nsight" / f"node-{node}.sqlite"
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.exists(), "refuse to replace SQLite export")
    command = [str(tool), "export", "--type=sqlite", f"--output={path}", str(report)]
    log = output / "nsight" / f"node-{node}-export.log"
    with log.open("x") as stream:
        try:
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=timeout)
        finally:
            stream.flush()
            inputs.bind(log)
    sqlite_sha = inputs.bind(path)
    proof = {"report_path": str(report), "report_sha256": report_sha, "sqlite_sha256": sqlite_sha,
             "command": command, "nsys_path": str(tool), "nsys_sha256": inputs.bind(Path(tool).resolve()),
             "hostname": socket.gethostname(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
             "user_authorized_cpu_export": allow_login_cpu_export}
    single.write_new(path.with_suffix(".source.json"), proof)
    inputs.bind(path.with_suffix(".source.json"))
    return path, True, proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orchestration-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--capability-root", type=Path)
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--allow-login-cpu-export", action="store_true")
    parser.add_argument("--container-library-map", type=Path)
    parser.add_argument("--export-timeout-seconds", type=int, default=120)
    parser.add_argument("--sqlite-cache-root", type=Path,
                        help="Prior case output's nsight directory, with hash-bound SQLite exports and source sidecars")
    args = parser.parse_args()
    orchestration, output = args.orchestration_dir.resolve(), args.output_dir.resolve()
    direct = output.parent == ROOT / "runs"
    nested = (output.name in {f"sms{sms}" for sms in adapter.BUDGETS} and output.parent.name == "cases"
              and output.parent.parent.parent == ROOT / "runs")
    require((direct or nested) and not output.exists(), "output must be a new runs child or new delivery/cases/sms20|sms12|sms32|sms40 child")
    require(args.export_timeout_seconds > 0, "positive export timeout required")
    output.mkdir(parents=True)
    inputs, inventory, training = single.evidence.Inputs(), [], None
    inputs.bind(Path(__file__))
    for module in (single, campaign, adapter, runtime_checks):
        inputs.bind(Path(module.__file__))
    result = {"valid_budget_run": False, "valid_training": False, "valid_nsight_timeline": False,
              "orchestration_dir": str(orchestration), "validation_hostname": socket.gethostname(),
              "validation_slurm_job_id": os.environ.get("SLURM_JOB_ID"), "gpu_execution": False,
              "whole_session_event_completeness": "unresolved",
              "quantitative_event_scope": "Known UCCL payload directions and exact source-bound AMPA call scopes; all training and per-node event gates must pass."}
    result["parent_warning_policy_source"] = {"path": str(Path(runtime_checks.__file__).resolve()),
        "sha256": inputs.bind(Path(runtime_checks.__file__))}
    try:
        campaign.stage(output, "training_validation_started")
        training = metadata(orchestration, output, inputs, library_map=args.container_library_map)
        expected, callback_proof = callbacks(training, inputs)
        result.update(training, **callback_proof)
        capability = None
        caproot = args.capability_root
        recorded_caproot = Path(inputs.read(orchestration / "nsight_budget_contract.json")["nsight_capability_root"])
        if caproot is None:
            caproot = recorded_caproot
        equal(str(caproot.resolve()), str(recorded_caproot.resolve()), "recorded capability source")
        if caproot:
            capability = inputs.read(caproot / "capabilities.json")
            inputs.bind(caproot / "commands.json")
            require(capability.get("native_report_generated") is True, "capability probe did not produce a native report")
        campaign.stage(output, "training_valid_timeline_started")
        for node in range(4):
            item = {"node_rank": node, "valid_nsight_timeline": False}
            try:
                path, bound, proof = export_or_find(Path(training["run_dir"]), output, node, inputs, capability,
                    skip_export=args.skip_export, allow_login_cpu_export=args.allow_login_cpu_export,
                    timeout=args.export_timeout_seconds, sqlite_cache_root=args.sqlite_cache_root,
                    training_job=training["training_slurm_job_id"])
                item["export"] = proof
                if path is not None:
                    parent = torchrun_parent(training, node, inputs)
                    item.update(inspect_sqlite(path, node, training, expected, parent))
                    item["report_export_binding_valid"] = bound
                    item["valid_nsight_timeline"] = bound and item["valid_timeline_contents"]
            except (RuntimeError, ValueError, OSError, subprocess.SubprocessError, sqlite3.Error) as error:
                item["error"] = f"{type(error).__name__}: {error}"
            inventory.append(item)
        result["valid_nsight_timeline"] = all(item["valid_nsight_timeline"] for item in inventory)
        result["valid_budget_run"] = result["valid_training"] and result["valid_callback_capture"] and result["valid_nsight_timeline"]
        metrics = [item.get("hardware_metrics", {}) for item in inventory]
        result["hardware_metrics"] = {"sample_rows_present_all_nodes": all(item.get("hardware_metric_samples_present") is True for item in metrics),
            "availability_by_node": [item.get("hardware_metric_samples_present", False) for item in metrics],
            "claim_limit": "Metric names and sampled rows are inventoried separately; no measured occupancy is inferred from static launch metadata."}
    except (RuntimeError, ValueError, KeyError, OSError, TypeError, sqlite3.Error) as error:
        result["error"] = f"{type(error).__name__}: {error}"
    single.write_new(output / "nsight_inventory.json", {"nodes": inventory})
    inputs.bind(output / "nsight_inventory.json")
    result["input_files"] = bind_list(inputs)
    result["claim_limits"] = ["Nsight-instrumented diagnostic timing, not unprofiled throughput.",
        "Historical immutable input verification is reused explicitly.", "Physical placement is independently recorded per allocation.",
        "Hardware counter availability is separate from training and timeline acceptance.",
        "Exact source-bound non-CUDA launcher control warnings are classified separately.",
        "For verified CUDA payload owners, only the two exact generic possible-loss advisories with complete known-event witnesses permit scoped acceptance; all other owner warnings and all errors remain fatal.",
        "Exact empty/control advisories on observed clients with zero exported kernels and AMPA scopes are retained outside the verified event set, without CPU-only or idle claims.",
        "Unknown warnings and explicit dropped/lost/overflow diagnostics remain fatal for every client.",
        "Exact scoped payload/call coverage does not establish whole-process event completeness."]
    inputs.recheck()
    single.write_new(output / "acceptance.json", result)
    campaign.stage(output, "accepted" if result["valid_budget_run"] else "not_accepted", valid_budget_run=result["valid_budget_run"])
    print(json.dumps({"valid_budget_run": result["valid_budget_run"], "output": str(output)}), flush=True)
    raise SystemExit(0 if result["valid_budget_run"] else 1)


if __name__ == "__main__":
    main()
