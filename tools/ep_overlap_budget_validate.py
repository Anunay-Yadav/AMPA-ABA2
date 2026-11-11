#!/usr/bin/env python3
"""Validate truthful unprofiled UCCL budget legs and matched allocation pairs.

Run on allocated compute. Source acquisitions are read-only. This binds the
recorded configuration, all-rank lifecycle, real update logs, input batches,
runtime ownership, integer send/receive counts, and native backend identity.
Event correlation, hardware counters, and campaign comparisons remain pending.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import statistics
import sys
from types import SimpleNamespace

import yaml

from analysis import accept_uccl_pilot as evidence
from analysis.compare_deep_trace_overhead import _checked_fingerprint_batches
from analysis.extract_step_timing import select_timer_log, timing_rows
from analysis.summarize_training_evidence import summarize
from analysis.validate_ep16_received_routes import validate_records
from profiling import run_pretrain_ep_overlap_budget as adapter
from profiling.ampa_profile import overlap_trace
from profiling.ampa_profile.provenance import (
    _ampa_runtime_sources, _EP16_RUNTIME_TOPOLOGIES, validate_rank_placement, validate_runtime_groups,
)
from tools.bridge_indexed_consumer import expected_bridge_indexed_consumer
from tools.ep16_posthoc_proof import import_closure
from tools.uccl_debug_campaign_validate import IdentityInputs, recorded_manifest
from tools.uccl_ep_source_identity import verify_uccl_source
from tools.uccl_ep_verified_inputs_reuse import FIRST_ROOT, load_first_case


ROOT = Path(__file__).resolve().parents[1]
require, equal, write_new = evidence.require, evidence.equal, adapter.write_new
TRAINING_SOURCES = (
    "jobs/training/ep_overlap_budget_pair.sbatch", "jobs/training/ep_overlap_budget_worker.sh",
    "profiling/run_pretrain_ep_overlap_budget.py", "profiling/run_pretrain_ep_overlap.py",
    "profiling/ampa_profile/overlap_capture.py",
    "profiling/ampa_profile/overlap_trace.py",
)


def checked_execution_contract(record, *, case, capture, steps, source_sha256):
    require(case == "uccl_ordinary" and capture == "off" and type(steps) is int and steps in (2, 10),
            "unprofiled budget evidence requires ordinary UCCL and two qualification or ten full updates")
    backend = "uccl_ep"
    budget = record.get("communication_sms")
    require(adapter.allowed_budget(budget), "unsupported actual UCCL communication block budget")
    expected = adapter.execution_contract(SimpleNamespace(
        execution_steps=steps, overlap_capture=capture, case=case, communication_backend=backend,
        communication_sms=budget, overlap_source_sha256=source_sha256,
        warmup_steps=1 if steps == 2 else 5, measured_steps=1 if steps == 2 else 5))
    equal(record, expected, "actual unprofiled-budget execution contract")
    return record


def checked_training_contract(contract, execution, case):
    steps = execution["execution_steps"]
    for key in ("completed_steps", "warmup_steps", "measured_steps"):
        equal(contract.get("training", {}).get(key), execution[key], f"actual training {key}")
    equal(contract.get("ep_overlap"), execution, "comparison contract embeds the actual overlap selection")
    equal(contract.get("run_kind"), "ep16_overlap_experiment", "actual overlap run kind")
    equal(contract.get("model", {}).get("hf_model_id"), "Qwen/Qwen3-30B-A3B-Base", "actual model")
    equal(contract.get("model", {}).get("hf_revision"), "1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9", "model revision")
    equal(contract.get("training", {}).get("learning_rate"), 1e-6, "actual learning rate")
    equal(contract.get("dataset", {}).get("mode"), "indexed", "actual indexed data")
    equal(contract.get("dataset", {}).get("sampler_seed"), 1234, "actual sample order seed")
    for key, value in {"ep_a2a_overlap": True, "delay_expert_wgrad": True,
        "ep_overlap_early_attn_memory_release": True, "dispatch_backward_expert_wgrad_overlap": False,
        "bridge_comm_overlap_configured": False, "dense_dp_overlap_grad_reduce": True,
        "dense_dp_overlap_param_gather": True, "cuda_device_max_connections": "32",
        "capacity_drop": False, "router_padding": False, "cuda_graphs": False,
        "activation_recomputation": False}.items():
        equal(contract.get("baseline", {}).get(key), value, f"actual baseline {key}")
    # Adapt only the historical gate's run-window declarations. The overlap
    # flags and all routing/backend/input fields reach it unchanged.
    historical = copy.deepcopy(contract)
    historical["training"].update(completed_steps=2, warmup_steps=5, measured_steps=20)
    evidence._contract(historical, case)
    require(steps in (2, 10), "unsupported actual training length")
    return contract


def checked_resolved(configuration, execution, case, run):
    for key, value in adapter.OVERLAP_FLAGS.items():
        equal(configuration.get("model", {}).get(key), value, f"actual resolved {key}")
    equal(configuration.get("train", {}).get("exit_interval"), execution["execution_steps"], "actual exit interval")
    equal(configuration.get("train", {}).get("train_iters"), 25, "original cache training budget")
    equal(configuration.get("optimizer", {}).get("lr"), 1e-6, "actual resolved learning rate")
    equal(configuration.get("comm_overlap"), None, "Bridge aggregate overlap configuration")
    equal(configuration.get("ddp", {}).get("delay_wgrad_compute"), True, "actual resolved DDP delayed weight gradients")
    for key in ("overlap_grad_reduce", "overlap_param_gather"):
        equal(configuration.get("ddp", {}).get(key), True, f"actual dense DP {key}")
    if case in evidence.UCCL_CASES:
        equal(configuration["model"].get("moe_flex_dispatcher_num_sms"), execution["communication_sms"], "resolved UCCL budget")
    historical = copy.deepcopy(configuration)
    historical["train"]["exit_interval"] = 2
    evidence._resolved(historical, case, run)
    return configuration


def checked_receives(contract, ownership, audits, environment, steps):
    require(type(steps) is int and steps in (2, 10), "receive validation supports two or ten actual updates")
    equal(contract.get("training", {}).get("completed_steps"), steps, "original receive run length")
    expected_window = (5, 5) if steps == 10 else (1, 1)
    equal([contract["training"].get("warmup_steps"), contract["training"].get("measured_steps")],
          list(expected_window), "original receive measurement window")
    require(type(audits) is list and len(audits) == 16, "all sixteen original routing audits required")
    for audit in audits:
        equal([row.get("step") for row in audit.get("steps", [])], list(range(steps)), "original received-route step grid")
    historical = copy.deepcopy(contract)
    historical["training"].update(completed_steps=2, warmup_steps=5, measured_steps=20)
    chunks = []
    for first in range(0, steps, 2):
        rows = [{**audit, "steps": [{**row, "step": index}
                for index, row in enumerate(audit["steps"][first:first + 2])]} for audit in audits]
        result = validate_records(historical, ownership, rows, environment, 2)
        chunks.append({"original_steps": [first, first + 1], "validation": result})
    totals = ("total_source_expert_assignments", "total_received_expert_assignments", "checked_expert_equalities")
    if contract["communication_backend"] == "uccl_ep":
        totals += ("unique_token_rank_deliveries", "unique_token_node_deliveries")
    return {"valid_send_receive_audit": True, "steps": list(range(steps)),
            "communication_backend": contract["communication_backend"], "unchanged_integer_gate_chunks": chunks,
            **{key: sum(row["validation"][key] for row in chunks) for key in totals},
            "claim_limit": "Aggregate integer sends/receives and logical delivery bounds, not physical traffic or GPU event completeness."}


def checked_call(record, *, rank, step, execution):
    backend = execution["communication_backend"]
    for key, value in {"schema_version": 1, "kind": "ampa_ep_overlap_calls", "rank": rank,
        "step": step, "capture_mode": execution["capture"], "skipped_iter": False, "layers": list(range(48))}.items():
        equal(record.get(key), value, f"overlap call {key}")
    observer = overlap_trace.OverlapMethodLabels(None, [], communication_backend=backend)
    expected = observer.contract_record(4)
    equal(record.get("observation_contract"), expected, "independently reconstructed overlap observation contract")
    equal(record.get("counts"), expected["expected_counts"], "exact native schedule and phase counts")
    for key in ("payload_bytes", "payload_dtypes"):
        require(type(record.get(key)) is dict, f"missing overlap {key}")
    require(all(type(value) is int and value >= 0 for value in record["payload_bytes"].values()), "invalid logical payload byte counts")
    require(all(type(value) is str and value for value in record["payload_dtypes"].values()), "invalid logical payload dtype")
    pairs = record.get("uccl_autograd_pairs")
    require(type(pairs) is list, "missing UCCL autograd journal")
    if backend == "alltoall":
        equal(pairs, [], "native case contains UCCL autograd calls")
    else:
        equal([pair.get("call_id") for pair in pairs], list(range(384)), "all UCCL forward/backward call IDs")
        equal(dict(Counter(f"{pair.get('layer')}:{pair.get('family')}" for pair in pairs)),
              {f"{layer}:{family}": 4 for layer in range(48) for family in ("dispatch", "combine")}, "UCCL layer/family call grid")
        epochs = set()
        for pair in pairs:
            require(type(pair.get("layer")) is int and pair["layer"] in range(48), "invalid UCCL pair layer")
            for key in ("context_id", "epoch"):
                require(type(pair.get(key)) is int and pair[key] > 0, f"invalid UCCL pair {key}")
            epochs.add(pair["epoch"])
            equal(pair.get("forward_returned"), True, "UCCL forward returned")
            equal(pair.get("backward_returned"), True, "UCCL backward returned")
        require(len(epochs) == 1, "UCCL call journal spans observer epochs")
    if execution["capture"] == "torch":
        for key in ("profiler_stop_seconds", "export_seconds"):
            finite_nonnegative(record.get(key), key)
    else:
        require("profiler_stop_seconds" not in record and "export_seconds" not in record,
                "Nsight capture contains a PyTorch profiler lifecycle")
    return record


def finite_nonnegative(value, label):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0, f"invalid {label}")


def checked_lifecycle(run, execution, placement, job, environment, inputs):
    mode, steps = execution["capture"], execution["execution_steps"]
    captured = execution["capture_step_indices"]
    geometry = dict(pipeline_model_parallel_size=1, tensor_model_parallel_size=1,
                    context_parallel_size=1, expert_model_parallel_size=16,
                    expert_tensor_parallel_size=1, num_layers=48, cuda_graph_impl="none",
                    recompute_granularity=None, moe_router_topk=8)
    for suffix in ("startup", "capture"):
        equal(sorted(path.name for path in (run / "raw").glob(f"ep_overlap_rank-*_{suffix}.json")),
              sorted(f"ep_overlap_rank-{rank}_{suffix}.json" for rank in range(16)), f"exact all-rank {suffix} grid")
    starts, completions = [], []
    for rank, slot in enumerate(placement):
        start = inputs.read(run / "raw" / f"ep_overlap_rank-{rank}_startup.json")
        for key, value in {"schema_version": 1, "contract": overlap_trace.CONTRACT, "valid_overlap_startup": True,
            "rank": rank, "hostname": slot["hostname"], "slurm_job_id": job, "local_rank": str(slot["local_rank"]),
            "world_size": 16, "all_rank_configuration_agreement": True, "configuration_ranks_checked": 16,
            "selected_flags": overlap_trace.SELECTED_FLAGS, "geometry": geometry, "capture_mode": mode,
            "ddp_delay_wgrad_compute": True,
            "completed_steps": steps, "communication_backend": execution["communication_backend"],
            "communication_sms": execution["communication_sms"] if execution["communication_backend"] == "uccl_ep" else 20,
            "cuda_device_max_connections": "32"}.items():
            equal(start.get(key), value, f"rank {rank} finalized overlap {key}")
        versions = start.get("versions")
        require(type(versions) is dict and all(type(versions.get(key)) is str and versions[key]
                for key in ("torch", "transformer-engine", "megatron-core", "megatron-bridge")), "missing actual runtime package versions")
        equal(versions["torch"], environment["pytorch_version"], "startup/environment Torch version")
        if starts:
            equal(versions, starts[0]["versions"], "all-rank runtime versions")
        if execution["communication_backend"] == "uccl_ep":
            configs = start.get("effective_configs")
            require(type(configs) is dict and set(configs) == {"dispatch", "combine"}, "missing effective UCCL configurations")
            for family, config in configs.items():
                equal(config.get("config", {}).get("num_sms"), execution["communication_sms"], f"effective {family} budget")
                equal(config.get("hidden_bytes"), 4096, "UCCL hidden bytes")
                equal(config.get("ep_ranks"), 16, "UCCL effective EP ranks")
                for name in ("nvl_buffer_size_hint", "rdma_buffer_size_hint"):
                    require(type(config.get(name)) is int and config[name] > 0, f"invalid UCCL {name}")
            if starts:
                equal(configs, starts[0]["effective_configs"], "all-rank effective UCCL configuration")
        else:
            require("effective_configs" not in start, "native case contains an effective UCCL budget")
        done = inputs.read(run / "raw" / f"ep_overlap_rank-{rank}_capture.json")
        for key, value in {"schema_version": 1, "contract": overlap_trace.CONTRACT, "valid_step_execution": True,
            "rank": rank, "capture_mode": mode, "completed_updates": steps, "ended_steps": list(range(steps)),
            "capture_steps": captured, "qualification_only": steps == 2}.items():
            equal(done.get(key), value, f"rank {rank} completed overlap {key}")
        boundaries = done.get("boundaries")
        if mode == "nsys":
            require(type(boundaries) is list and len(boundaries) == 2, "missing Nsight start/stop boundaries")
            for boundary, action, step in zip(boundaries, ("start", "stop"), (captured[0], captured[-1])):
                for key, value in {"rank": rank, "action": action, "step": step}.items():
                    equal(boundary.get(key), value, f"Nsight boundary {key}")
                timing = {"cuda_synchronize_seconds", "pre_api_global_barrier_seconds", "profiler_api_seconds"}
                if action == "start":
                    timing.add("post_start_global_barrier_seconds")
                equal(sorted(boundary), sorted({"rank", "action", "step"} | timing), "Nsight boundary schema")
                for key in timing:
                    finite_nonnegative(boundary[key], key)
        else:
            equal(boundaries, [], "non-Nsight case contains profiler API boundaries")
        starts.append(start)
        completions.append(done)
    require(not list((run / "raw").glob("ep_overlap_rank-*_aborted.json"))
            and not list((run / "raw").glob("uccl_pretrain_aborted_rank-*.json")), "acquisition contains an aborted worker")
    calls = {run / "raw/overlap_calls" / f"step-{step:05d}" / f"rank-{rank}.json"
             for step in captured for rank in range(16)}
    equal(sorted(map(str, (run / "raw/overlap_calls").rglob("*.json"))), sorted(map(str, calls)), "exact call-record grid")
    for step in captured:
        for rank in range(16):
            checked_call(inputs.read(run / "raw/overlap_calls" / f"step-{step:05d}" / f"rank-{rank}.json"),
                         rank=rank, step=step, execution=execution)
    traces = {run / "traces/torch_profile" / f"step-{step:05d}" / f"rank-{rank}.json.gz"
              for step in captured for rank in range(16)} if mode == "torch" else set()
    equal(sorted(map(str, (run / "traces/torch_profile").rglob("*.json*"))), sorted(map(str, traces)), "exact Torch artifact grid")
    for path in traces:
        require(path.stat().st_size > 0, "empty Torch artifact")
        inputs.bind(path)
    return {"valid_all_rank_startup_and_completion": True,
            "host_call_observation_enabled": mode != "off", "valid_recorded_host_call_grid": True if mode != "off" else None,
            "ranks": 16, "completed_steps": list(range(steps)), "captured_steps": captured,
            "call_records": len(calls), "torch_artifacts_bound": len(traces),
            "versions": starts[0]["versions"], "nsight_boundaries": [row["boundaries"] for row in completions],
            "gpu_event_correlation_acceptance": "pending_separate_analysis",
            "hardware_metrics_acceptance": "pending_separate_analysis"}


def checked_nsight(run, execution, job, sources, inputs, *, root=ROOT):
    if execution["capture"] != "nsys":
        require(not list((run / "nsight").glob("*.nsys-rep")), "non-Nsight case contains native Nsight reports")
        return None
    reports = sorted((run / "nsight").glob("*.nsys-rep"))
    equal([path.name for path in reports], [f"node-{node}-native.nsys-rep" for node in range(4)], "four native Nsight reports")
    worker = root / "jobs/training/ep_overlap_worker.sh"
    cap_reference = None
    for node, path in enumerate(reports):
        require(path.stat().st_size > 0, "empty native Nsight report")
        inputs.bind(path)
        launch = inputs.read(run / "nsight" / f"node-{node}-launch.json")
        parent = inputs.read(run / "nsight" / f"node-{node}-torchrun-parent.json")
        for key, value in {"job": job, "node_rank": node, "cuda_device_max_connections": "32"}.items():
            equal(launch.get(key), value, f"Nsight launch {key}")
        for key, value in {"schema_version": 1, "kind": "ampa_nsight_torchrun_parent", "job": job,
            "node_rank": node, "role": "torchrun_launcher", "source_path": str(worker),
            "source_sha256": sources[str(worker)], "capture": "nsys", "cuda_device_max_connections": "32"}.items():
            equal(parent.get(key), value, f"Nsight parent {key}")
        require(type(parent.get("pid")) is int and parent["pid"] > 0, "invalid collector parent PID")
        command = parent.get("command")
        require(type(command) is list and all(type(arg) is str for arg in command), "missing torchrun command")
        for arg in ("torch.distributed.run", "--nnodes=4", "--nproc_per_node=4", f"--node_rank={node}"):
            require(arg in command, f"torchrun launch omits {arg}")
        cap = Path(launch.get("capability_root", ""))
        require(cap.is_absolute() and cap.parent == root / "runs" and not cap.is_symlink(), "invalid explicit capability root")
        if cap_reference is None:
            cap_reference = str(cap)
        equal(str(cap), cap_reference, "all-node Nsight capability identity")
        proof = inputs.read(cap / "capabilities.json")
        commands = inputs.read(cap / "commands.json")
        equal(proof.get("native_report_generated"), True, "historical Nsight tool capability")
        native = next(item for item in commands if item["name"] == "capture_native")["argv"]
        output_index = next(index for index, item in enumerate(native) if item.startswith("--output="))
        expected = [proof["nsys_path"], "profile", *native[2:output_index]]
        if proof.get("gpu_metrics_collected") is True:
            metrics = next(item for item in commands if item["name"] == "capture_gpu_metrics")["argv"]
            expected += [item for item in metrics if item.startswith("--gpu-metrics-")]
        expected.append(f"--output={run / 'nsight' / f'node-{node}-native'}")
        equal(launch.get("collector_argv_prefix"), expected, "capability-proven collector arguments")
        equal(launch.get("gpu_metrics_requested"), proof.get("gpu_metrics_collected") is True, "counter collection request")
        inputs.bind(Path(proof["nsys_path"]))
        exit_path = run / "nsight" / f"node-{node}-exit.txt"
        inputs.bind(exit_path)
        equal(dict(line.split("=", 1) for line in exit_path.read_text().splitlines()),
              {"exit_code": "0", "job": job, "node": str(node)}, "collector exit evidence")
        for stream in ("out", "err"):
            inputs.bind(run / "nsight" / f"node-{node}-collector.{stream}")
    return {"native_reports_bound": 4, "historical_capability_root": cap_reference,
            "actual_event_correlation_acceptance": "pending_separate_analysis",
            "actual_hardware_metric_acceptance": "pending_separate_analysis"}


def checked_inputs(orchestration, run, acquisition, environment, contract, inputs, *, root):
    reviewed = inputs.read(orchestration / "reviewed_inputs.json")
    context = inputs.read(orchestration / "input_reuse_context.json")
    first = load_first_case(FIRST_ROOT, root=root)
    reference = first["proof_reference"]
    inputs.bind(Path(reference["path"]), reference["sha256"])
    equal(inputs.read(Path(reference["path"])), first["proof"], "bound first proof contents")
    equal(inputs.read(FIRST_ROOT / "acquisition.json"), first["acquisition"], "bound first acquisition contents")
    first_completed = inputs.read(FIRST_ROOT / "acquisition_completed.json")
    for key, value in {"job": reference["slurm_job_id"], "success": True,
                       "workers_srun_returned_zero": True, "completed_updates": 10}.items():
        equal(first_completed.get(key), value, f"bound first completion {key}")
    inputs.bind(FIRST_ROOT / "cases.tsv")
    equal(acquisition.get("verification_mode"), "reused_first_case", "historical input verification mode")
    verification = acquisition.get("verification_reuse")
    require(type(verification) is dict, "missing explicit historical verification declarations")
    equal(verification.get("first_verified_proof"), reference, "pinned first verified proof")
    for name in ("payload_verified_in_current_allocation", "indexed_payload_verified_in_current_allocation",
                 "native_build_reverified_in_current_allocation"):
        equal(verification.get(name), False, f"historical verification {name}")
    for record in (reviewed, context):
        equal(record.get("verification_mode"), "reused_first_case", "consistent historical verification mode")
        equal(record.get("verification_reuse"), verification, "consistent historical verification proof")
        for key in ("experiment_protocol", "slurm_job_id", "partition", "case", "capture", "communication_sms",
                    "execution_steps", "qualification_only", "source_manifest"):
            equal(record.get(key), acquisition.get(key), f"prepared metadata {key}")
    equal(reviewed.get("valid_reviewed_inputs"), True, "reviewed input proof")
    payload = reviewed.get("checkpoint_payload", {})
    for key, value in {"verified": True, "verified_in_current_allocation": False,
        "verification_mode": "reused_first_case", "verification_slurm_job_id": reference["slurm_job_id"],
        "first_verified_proof": reference}.items():
        equal(payload.get(key), value, f"historical checkpoint {key}")
    for key in ("path", "file_count", "total_bytes", "inventory_sha256"):
        equal(payload.get(key), first["proof"]["checkpoint_payload"].get(key), f"historical payload identity {key}")
    manifests = {}
    for name in ("c0_manifest.json", "dataset_manifest.json", "model_manifest.json"):
        manifests[name] = inputs.bind(run / "config" / name)
        equal(manifests[name], inputs.bind(first["run"] / "config" / name), f"unchanged immutable {name}")
    for key, value in {"c0_identity_manifest_sha256": manifests["c0_manifest.json"],
                       "dataset_manifest_sha256": manifests["dataset_manifest.json"]}.items():
        equal(reviewed.get(key), value, f"reviewed {key}")
        equal(context.get(key), value, f"reuse context {key}")
        equal(first["proof"].get(key), value, f"first proof {key}")
    c0 = root / "checkpoint/megatron/qwen3-30b-a3b-base-c0"
    equal(payload["path"], str(c0), "immutable C0 payload path")
    equal(contract.get("initialization"), {"mode": "pretrained", "checkpoint": str(c0)}, "actual C0 initialization")
    equal(environment.get("checkpoint"), str(c0), "actual loaded checkpoint")
    equal(inputs.bind(c0 / "ampa_c0_manifest.json"), manifests["c0_manifest.json"], "canonical C0 manifest identity")
    equal(contract.get("c0_identity_manifest_sha256"), manifests["c0_manifest.json"], "training C0 binding")
    equal(contract["dataset"].get("manifest_sha256"), manifests["dataset_manifest.json"], "training dataset binding")
    first_contract = inputs.read(first["run"] / "config/comparison_contract.json")
    equal(contract["dataset"], first_contract["dataset"], "unchanged pinned mixture, tokenizer, and sample seed")
    dataset = inputs.read(run / "config/dataset_manifest.json")
    equal(context.get("indexed_manifest"), dataset, "prepared indexed dataset metadata")
    equal(environment.get("dataset_consumer"), expected_bridge_indexed_consumer(dataset), "actual pinned indexed consumer")
    for key, value in {"run_dir": str(run), "pretrained_checkpoint": str(c0),
                       "dataset_manifest_path": str(run / "config/dataset_manifest.json")}.items():
        equal(context.get(key), value, f"reuse context {key}")
    equal(environment.get("reviewed_inputs_proof"), {"path": str(orchestration / "reviewed_inputs.json"),
          "sha256": inputs.bind(orchestration / "reviewed_inputs.json")}, "actual input proof binding")
    equal(reviewed.get("native_checkouts"), {name: environment[name] for name in ("megatron_lm", "megatron_bridge")}, "actual native source identity")
    for key, value in reviewed.get("runtime_versions", {}).items():
        equal(environment.get(key), value, f"reviewed runtime {key}")
    for label, key in (("megatron_lm", "megatron_lm"), ("megatron_bridge", "megatron_bridge")):
        paths = {kind: orchestration / "source" / f"launch_{label}_{kind}.txt"
                 for kind in ("revision", "status", "diff")}
        for path in paths.values():
            inputs.bind(path)
        actual = {"revision": paths["revision"].read_text().strip(), "status": paths["status"].read_text().strip(),
                  "diff_sha256": hashlib.sha256(paths["diff"].read_text().strip().encode()).hexdigest()}
        equal(environment[key], actual, "archived actual native revision/status/diff")
    return {"verification_mode": "reused_first_case", "fresh_payload_verification_claimed": False,
            "first_verified_proof": reference, "first_training_run": str(first["run"]), "immutable_manifests": manifests}


def checked_worker_logs(run, placement, inputs):
    """Bind original streams for native and UCCL, using actual worker slots."""
    by_node = {}
    for slot in placement:
        rank, node, local = slot["rank"], slot["slurm_node_id"], slot["local_rank"]
        log_root = run / "logs" / f"torchrun-node-{node}"
        parts, records = [], []
        for stream in ("stdout", "stderr"):
            paths = list(log_root.glob(f"**/attempt_0/{local}/{stream}.log"))
            require(len(paths) == 1, "exactly one original worker stream is required")
            path = paths[0]
            records.append((str(path), inputs.bind(path)))
            parts.append(path.read_bytes())
        flat = run / "logs" / f"rank-{rank}.out"
        flat_sha = inputs.bind(flat)
        equal(hashlib.sha256(b"".join(parts)).hexdigest(), flat_sha, "flattened worker log matches original streams")
        records.append((str(flat), flat_sha))
        by_node.setdefault(node, []).extend(records)
    for node, expected in by_node.items():
        path = run / "logs" / f"log-map-node-{node}.sha256"
        inputs.bind(path)
        observed = []
        for line in path.read_text().splitlines():
            digest, name = line.split(maxsplit=1)
            observed.append((name.lstrip(" *"), digest))
        equal(sorted(observed), sorted(expected), "complete original/flattened log map")


def checked_budget_runtime_record(record, execution, slot, job):
    """Check the first real-update guard without interpreting block count as SM use."""
    budget = execution["communication_sms"]
    for name, value in {"schema_version": 1, "valid_actual_budget": True,
        "experiment_protocol": adapter.PROTOCOL, "rank": slot["rank"], "job": job,
        "first_completed_update": 1, "requested_budget": budget, "buffer_num_sms": budget,
        "buffer_class_num_sms": budget, "low_latency_mode": False, "group_size": 16}.items():
        equal(record.get(name), value, f"actual allocated Buffer guard {name}")
    limits = record.get("device_launch_limits")
    require(type(limits) is dict, "actual worker CUDA launch limits are missing")
    for name, value in {"ordinal": slot["cuda_current_device"], "major": 9,
                        "multi_processor_count": 132, "warp_size": 32}.items():
        equal(limits.get(name), value, f"selected-device guard {name}")
    for name in ("max_threads_per_block", "shared_memory_per_block_optin"):
        require(type(limits.get(name)) is int and limits[name] > 0, "invalid actual CUDA resource limit")
    supported = [value for value in adapter.BUDGETS
        if adapter.kernel_launch_requirements(value)["notify_threads"] <= limits["max_threads_per_block"]
        and adapter.kernel_launch_requirements(value)["notify_dynamic_shared_bytes"] <= limits["shared_memory_per_block_optin"]]
    equal(limits.get("supported_budgets"), supported, "supported budgets derive from actual CUDA limits")
    require(budget in supported, "requested budget exceeds selected-device launch limits")
    return record


def checked_budget_parent_records(run, placement, job, campaign, inputs, root):
    """Record the real OFF launcher producer without asserting a CUDA trace exists."""
    worker = root / "jobs/training/ep_overlap_budget_worker.sh"
    digest = campaign[str(worker)]
    by_node = {str(slot["slurm_node_id"]) for slot in placement}
    records = []
    for node in sorted(by_node):
        path = run / "nsight" / f"node-{node}-torchrun-parent.json"
        record = inputs.read(path)
        for name, value in {"schema_version": 1, "kind": "ampa_nsight_torchrun_parent",
            "job": job, "node_rank": int(node), "role": "torchrun_launcher",
            "source_path": str(worker), "source_sha256": digest, "capture": "off",
            "cuda_device_max_connections": "32"}.items():
            equal(record.get(name), value, f"actual unprofiled parent {name}")
        require(type(record.get("pid")) is int and record["pid"] > 0, "invalid launcher PID")
        command = record.get("command")
        require(type(command) is list and all(type(part) is str for part in command), "launcher command missing")
        equal(command[:3], ["/opt/venvs/megatron-bridge/bin/python", "-m", "torch.distributed.run"],
              "actual native torchrun launcher")
        equal(command[-3:], ["bash", str(worker), "--worker"], "actual budget worker entrypoint")
        for part in ("--nnodes=4", "--nproc_per_node=4", f"--node_rank={node}", "--max_restarts=0"):
            require(command.count(part) == 1, "actual launcher topology or restart policy differs")
        records.append(record)
    equal(len(records), 4, "four actual OFF torchrun parents")
    return records


def validate(orchestration, output, *, root=ROOT):
    root, orchestration, output = Path(root), Path(orchestration), Path(output)
    require(orchestration.is_absolute() and orchestration.parent == root / "runs" and orchestration.is_dir()
            and not orchestration.is_symlink(), "invalid orchestration directory")
    require(output.is_absolute() and root / "runs" in output.parents and ".." not in output.parts
            and not output.exists() and not output.is_symlink(), "validation output must be a new project run directory")
    output.mkdir(parents=True)
    inputs = evidence.Inputs()
    for name in import_closure(root, ("tools/ep_overlap_budget_validate.py",)):
        inputs.bind(root / name)
    inputs.bind(orchestration / "cases.tsv")
    rows = [line.split("\t") for line in (orchestration / "cases.tsv").read_text().splitlines()]
    require(len(rows) == 1 and len(rows[0]) == 3, "exactly one case/capture/run row required")
    case, capture, raw_run = rows[0]
    run = Path(raw_run)
    require(run.is_absolute() and run.parent == root / "runs" and run.is_dir() and not run.is_symlink()
            and run != orchestration, "invalid independent training directory")
    acquisition = inputs.read(orchestration / "acquisition.json")
    job, steps = acquisition.get("job"), acquisition.get("execution_steps")
    require(type(job) is str and job.isdigit(), "training job identity is missing")
    campaign = recorded_manifest(orchestration / "campaign-source.sha256", inputs, orchestration, root)
    source_sha = inputs.bind(orchestration / "campaign-source.sha256")
    for name in TRAINING_SOURCES:
        require(str(root / name) in campaign, f"actual execution source is missing: {name}")
        inputs.bind(root / name, campaign[str(root / name)])
    source = {"path": str(orchestration / "campaign-source.sha256"), "sha256": source_sha,
              "files": {str(Path(name).relative_to(root)): value for name, value in campaign.items()}}
    equal(acquisition.get("source_manifest"), source, "actual complete source manifest identity")
    equal(acquisition.get("campaign_source_sha256"), source_sha, "acquisition source manifest binding")
    execution = checked_execution_contract(inputs.read(run / "config/ep_overlap_contract.json"), case=case,
                                           capture=capture, steps=steps, source_sha256=source_sha)
    for key, value in {"experiment_protocol": adapter.PROTOCOL, "partition": "debug", "case": case,
        "capture": capture, "leg": capture, "slurm_job_id": job, "run_dir": str(run),
        "communication_backend": execution["communication_backend"], "communication_sms": execution["communication_sms"],
        "overlap_flags": adapter.OVERLAP_FLAGS, "cuda_device_max_connections": "32",
        "qualification_only": steps == 2, "completed_updates": steps, "cache_budget_samples": 1600,
        "cache_budget_updates": 25, "topology": "cp1_ep16", "launch_policy": "torchrun_node4",
        "warmup_steps": execution["warmup_steps"], "measured_steps": execution["measured_steps"]}.items():
        equal(acquisition.get(key), value, f"acquisition {key}")
    completed = inputs.read(orchestration / "acquisition_completed.json")
    for key, value in acquisition.items():
        equal(completed.get(key), value, f"completion retains actual acquisition {key}")
    for key, value in {"success": True, "workers_srun_returned_zero": True, "completed_updates": steps,
        "training_health_and_timer_summary_complete": True, "comparison_accepted": False,
        "fresh_checkpoint_payload_verified": False, "fresh_dataset_payload_verified": False}.items():
        equal(completed.get(key), value, f"completion {key}")
    contract = checked_training_contract(inputs.read(run / "config/comparison_contract.json"), execution, case)
    resolved = run / "config/resolved.yaml"
    inputs.bind(resolved)
    checked_resolved(yaml.safe_load(resolved.read_text()), execution, case, run)
    environment = inputs.read(run / "environment.json")
    equal(environment.get("slurm_job_id"), job, "actual training allocation")
    equal(environment.get("launch_policy"), "torchrun_node4", "actual launch policy")
    equal(environment.get("runtime_topology"), _EP16_RUNTIME_TOPOLOGIES["cp1_ep16"], "actual EP16 CP1 runtime topology")
    sources = environment.get("ampa_runtime_sources")
    equal(sources, _ampa_runtime_sources(root), "canonical runtime source identity")
    equal(acquisition.get("runtime_source_sha256"), sources["sha256"], "acquisition canonical source identity")
    runtime = recorded_manifest(orchestration / "runtime-source.sha256", inputs, orchestration, root)
    equal(runtime, {str(root / name): value for name, value in sources["files"].items()}, "canonical runtime manifest coverage")
    for name, digest in runtime.items():
        equal(campaign.get(name), digest, "complete campaign inventory covers actual runtime source")
    placement = validate_rank_placement(environment["rank_placement"], expected_world_size=16, launch_policy="torchrun_node4")
    validate_runtime_groups(placement, expected_world_size=16, cp=1)
    checked_worker_logs(run, placement, inputs)
    immutable = checked_inputs(orchestration, run, acquisition, environment, contract, inputs, root=root)
    equal(inputs.read(orchestration / "reviewed_inputs.json").get("ampa_runtime_sources"), sources, "reviewed current runtime source")
    mode = inputs.read(run / "config/observation_mode.json")
    for key, value in {"data_mode": "indexed", "lightweight_profiling": "off", "nsys_trace": capture == "nsys",
        "torch_trace": capture == "torch", "sample_fingerprints": True, "router_observer_attached": False,
        "route_wrapper_installed": False, "dispatcher_preprocess_wrapper_installed": False}.items():
        equal(mode.get(key), value, f"actual observation mode {key}")
    fingerprints = {}
    for rank in range(16):
        inputs.bind(run / "logs" / f"rank-{rank}.out")
        path = run / "raw" / f"sample_fingerprint_rank-{rank}.json"
        fingerprints[str(rank)] = _checked_fingerprint_batches(inputs.read(path), rank=rank, accumulation=4,
                                                               steps=steps, token_shape=[1, 4096], path=path)
        reference_path = Path(immutable["first_training_run"]) / "raw" / f"sample_fingerprint_rank-{rank}.json"
        reference_batches = _checked_fingerprint_batches(inputs.read(reference_path), rank=rank, accumulation=4,
                                                         steps=10, token_shape=[1, 4096], path=reference_path)
        equal(fingerprints[str(rank)], reference_batches[:steps], "actual consumed batches match the fixed original prefix")
    prefix = hashlib.sha256(json.dumps(fingerprints, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    health = summarize(run, steps=steps, interval=1)
    equal(health["final_consumed_samples"], steps * 64, "actual consumed global samples")
    equal(health["final_consumed_global_tokens"], steps * 262144, "actual consumed global tokens")
    equal(inputs.read(run / "analysis/real_training_evidence.json"), health, "recorded authoritative health summary")
    timers = timing_rows(select_timer_log(run / "logs"), warmup_steps=execution["warmup_steps"],
                        measured_steps=execution["measured_steps"], log_interval=1, tokens_per_step=262144)
    all_timers = timing_rows(select_timer_log(run / "logs"), warmup_steps=0,
                            measured_steps=steps, log_interval=1, tokens_per_step=262144)
    values = [row["mean_optimizer_step_ms"] for row in timers]
    require(all(type(value) in (int, float) and math.isfinite(value) and value > 0 for value in values), "invalid full-update timer")
    timing = {"unit": "complete optimizer-step milliseconds", "by_update": timers, "count": len(values),
              "all_updates": all_timers,
              "sum_all_optimizer_update_seconds": sum(row["mean_optimizer_step_ms"] for row in all_timers) / 1000,
              "whole_allocation_wall_time": "unavailable_from_optimizer_timers",
              "mean": statistics.fmean(values), "median": statistics.median(values),
              "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
              "qualification_only": steps == 2, "capture": capture}
    ownership = inputs.read(run / "config/routing_ownership.json")
    audits = [inputs.read(run / "raw" / f"routing_audit_rank-{rank}.json") for rank in range(16)]
    received = checked_receives(contract, ownership, audits, environment, steps)
    lifecycle = checked_lifecycle(run, execution, placement, job, environment, inputs)
    lifecycle["gpu_event_correlation_acceptance"] = "not_requested_profiler_off"
    lifecycle["hardware_metrics_acceptance"] = "not_requested_profiler_off"
    budget_guards = [checked_budget_runtime_record(
        inputs.read(run / "raw" / f"ep_overlap_budget_rank-{slot['rank']}_runtime.json"),
        execution, slot, job) for slot in placement]
    parents = checked_budget_parent_records(run, placement, job, campaign, inputs, root)
    backend = None
    if case in evidence.UCCL_CASES:
        build = Path(contract["uccl_build_dir"])
        equal(str(build), str(root / "runs/uccl-ep-owned-build-20261101T1713Z"), "pinned isolated UCCL build")
        identity = verify_uccl_source(build)
        equal(inputs.read(build / "source_identity.json"), identity, "approved UCCL source identity")
        equal(acquisition.get("uccl_source_identity"), identity, "recorded actual UCCL identity")
        inputs.bind(build / "source/uccl-borrowed-scratch.patch", identity["patch_sha256"])
        for name, digest in identity["modified_files"].items():
            inputs.bind(build / "uccl" / name, digest)
        binary = evidence._uccl_identity(run, build, identity, placement, job, root, IdentityInputs(inputs, run, output))
        equal(binary, acquisition.get("binary_sha256"), "actual pinned UCCL binary identity")
        for rank in range(16):
            runtime_backend = inputs.read(run / "raw" / f"uccl_backend_rank-{rank}_runtime.json")
            equal(runtime_backend.get("num_sms"), execution["communication_sms"], "actual native UCCL runtime block budget")
        backend = {"valid_backend_identity_and_cleanup": True, "source_identity": identity, "binary_sha256": binary}
    nsight = checked_nsight(run, execution, job, campaign, inputs, root=root)
    inputs.recheck()
    for name, record in (("training_health.json", health), ("timing.json", timing),
                         ("received_routes.json", received), ("recorded_capture_lifecycle.json", lifecycle)):
        write_new(output / name, record)
    result = {"schema_version": 1, "kind": "ampa_ep_overlap_unprofiled_budget_execution_accounting_validation",
        "valid_execution_and_accounting": True, "qualification_only": steps == 2,
        "comparison_accepted": False, "full_campaign_accepted": False,
        "gpu_event_correlation_acceptance": "not_requested_profiler_off",
        "hardware_metrics_acceptance": "not_requested_profiler_off", "orchestration_dir": str(orchestration),
        "run_dir": str(run), "case": case, "capture": capture, "training_slurm_job_id": job,
        "communication_backend": execution["communication_backend"], "communication_sms": execution["communication_sms"],
        "execution_contract": execution, "completed_updates": steps,
        "immutable_inputs": immutable, "consumed_input_fingerprint_sha256": prefix,
        "rank_placement": placement, "valid_training_health": True, "valid_send_receive_audit": True,
        "valid_all_rank_startup_and_completion": True, "backend": backend, "nsight": nsight,
        "valid_actual_budget_runtime_guards": True, "budget_runtime_guards": budget_guards,
        "unprofiled_launcher_producers": parents, "allocation_scope": "shared_pair",
        "per_case_slurm_elapsed_available": False,
        "source_manifest_sha256": source_sha,
        "training_worker_sha256": campaign[str(root / "jobs/training/ep_overlap_budget_worker.sh")],
        "validator_sha256": inputs.bind(Path(__file__)),
        "execution": {"hostname": socket.gethostname(), "python": sys.executable,
                      "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "partition": os.environ.get("SLURM_JOB_PARTITION")},
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "input_files": [{"path": path, "sha256": digest} for path, digest in sorted(inputs.hashes.items())],
        "claim_limit": "Source-bound real-update and logical-accounting evidence only. Host calls and nonempty traces do not establish correlated GPU completeness, physical traffic, useful occupancy, exposed latency, or a matched campaign comparison."}
    write_new(output / "execution_accounting.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--orchestration-dir", type=Path)
    selection.add_argument("--pair-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    require(os.environ.get("SLURM_JOB_PARTITION") == "debug" and os.environ.get("SLURM_JOB_ID", "").isdigit()
            and Path(sys.prefix) == Path("/opt/venvs/megatron-bridge"), "allocated debug configured-container execution required")
    if args.pair_dir is not None:
        from analysis.ep_overlap_budget_compare import compare_pair
        result = compare_pair(args.pair_dir, args.output_dir)
        print(json.dumps({"valid_paired_budget_comparison": result["valid_paired_budget_comparison"],
                          "output_dir": str(args.output_dir)}, sort_keys=True))
    else:
        result = validate(args.orchestration_dir, args.output_dir)
        print(json.dumps({"valid_execution_and_accounting": result["valid_execution_and_accounting"],
                          "comparison_accepted": False, "output_dir": str(args.output_dir)}, sort_keys=True))


if __name__ == "__main__":
    main()
