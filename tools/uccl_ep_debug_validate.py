#!/usr/bin/env python3
"""Validate one ten-update debug acquisition on allocated compute.

Existing event, routing, ownership, and UCCL lifetime checks remain authoritative.
Separate allocations are accepted individually and are never labelled matched
physical-placement comparisons by this tool.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys

import yaml

from analysis import accept_uccl_pilot as evidence
from analysis.compare_deep_trace_overhead import _checked_fingerprint_batches
from analysis.compare_uccl_profiles import phase_rows
from analysis.extract_step_timing import select_timer_log, timing_rows
from analysis.summarize_training_evidence import summarize
from analysis.validate_ep16_received_routes import validate_records
from profiling.ampa_profile.provenance import (
    _ampa_runtime_sources, _EP16_RUNTIME_TOPOLOGIES, validate_rank_placement, validate_runtime_groups,
)
from profiling.run_pretrain_debug import execution_contract
from tools.uccl_ep_source_identity import verify_uccl_source


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "debug_one_case_5_plus_5_v1"
STEPS, WARMUP, MEASURED = 10, 5, 5
DEBUG_TRANSPORT_POLICY = {
    "FI_MR_CACHE_MONITOR": "disabled",
    "scope": "both_uccl_routing_modes_and_profiler_settings_before_native_initialization",
    "claim_limit": "Disables the system monitor; registration overhead can change.",
}
require, equal = evidence.require, evidence.equal


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def manifest(path, inputs):
    """Bind the bytes listed by a source manifest without executing a shell."""
    inputs.bind(path)
    records = {}
    for line in Path(path).read_text().splitlines():
        fields = line.split(maxsplit=1)
        require(len(fields) == 2, "malformed source manifest")
        digest, raw_path = fields
        name = Path(raw_path.lstrip(" *"))
        require(len(digest) == 64 and all(c in "0123456789abcdef" for c in digest)
                and name.is_absolute() and str(name) not in records, "invalid or duplicate source binding")
        inputs.bind(name, digest)
        records[str(name)] = digest
    require(bool(records), "empty source manifest")
    return records


def one_case(orchestration, root, inputs):
    inputs.bind(orchestration / "cases.tsv")
    rows = [line.split("\t") for line in (orchestration / "cases.tsv").read_text().splitlines()]
    require(len(rows) == 1 and len(rows[0]) == 3, "debug acquisition requires exactly one case row")
    case, leg, raw_run = rows[0]
    require(case in evidence.CASES and leg in ("control", "profile"), "unknown debug case or leg")
    run = Path(raw_run)
    require(run.is_absolute() and run.is_dir() and not run.is_symlink()
            and run.parent == root / "runs" and run != orchestration, "invalid debug run directory")
    return case, leg, run


def checked_contract(contract, case):
    for name, value in {"completed_steps": STEPS, "warmup_steps": WARMUP,
                        "measured_steps": MEASURED, "learning_rate": 1e-6}.items():
        equal(contract["training"].get(name), value, f"debug training {name}")
    equal(contract["routing"].get("topk"), 8, "debug top-eight routing")
    equal(contract["model"].get("hf_model_id"), "Qwen/Qwen3-30B-A3B-Base", "debug model")
    equal(contract["model"].get("hf_revision"), "1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9", "debug model revision")
    equal(contract["dataset"].get("mode"), "indexed", "debug indexed data")
    equal(contract["dataset"].get("sampler_seed"), 1234, "debug data order seed")
    equal(contract.get("launch_policy"), "torchrun_node4", "debug launch policy")
    value = copy.deepcopy(contract)
    # Only the explicitly checked run length differs from the historical
    # workload gate. The original on-disk contract remains unchanged and bound.
    value["training"].update(completed_steps=2, measured_steps=20)
    evidence._contract(value, case)
    return contract


def checked_resolved(configuration, case, run):
    # Keep the original 25-update indexed-data/cache geometry. Exit after ten
    # real updates, just as the existing pilot exits after two.
    equal(configuration["train"].get("train_iters"), 25, "debug cache training budget")
    equal(configuration["train"].get("exit_interval"), STEPS, "debug actual exit interval")
    equal(configuration["optimizer"].get("lr"), 1e-6, "debug resolved learning rate")
    value = copy.deepcopy(configuration)
    value["train"]["exit_interval"] = 2
    evidence._resolved(value, case, run)
    return configuration


def checked_receives(contract, ownership, audits, environment):
    """Apply the unchanged two-update integer gate to five ordered chunks.

    The gate supports only historical two/25-update lengths. Validate the
    original ten-update contract and complete step grid first. Each copied
    chunk changes only its local step labels and declared historical length.
    Ownership, rank traffic, expert counts, received counts, delivery bounds,
    and online UCCL membership/weight evidence reach the existing gate intact.
    """
    equal(contract["training"].get("completed_steps"), STEPS, "receive original run length")
    equal(contract["training"].get("warmup_steps"), WARMUP, "receive original warmup")
    equal(contract["training"].get("measured_steps"), MEASURED, "receive original measurement length")
    require(len(audits) == 16, "all sixteen source audits required")
    for audit in audits:
        equal([row.get("step") for row in audit.get("steps", [])], list(range(STEPS)), "receive original step grid")
    normalized = copy.deepcopy(contract)
    normalized["training"].update(completed_steps=2, measured_steps=20)
    chunks = []
    for first in range(0, STEPS, 2):
        rows = []
        for audit in audits:
            row = dict(audit)
            row["steps"] = [{**step, "step": offset}
                            for offset, step in enumerate(audit["steps"][first:first + 2])]
            rows.append(row)
        result = validate_records(normalized, ownership, rows, environment, 2)
        chunks.append({"original_steps": [first, first + 1], "validation": result})
    fields = ["total_source_expert_assignments", "total_received_expert_assignments", "checked_expert_equalities"]
    if contract["communication_backend"] == "uccl_ep":
        fields += ["unique_token_rank_deliveries", "unique_token_node_deliveries"]
    totals = {field: sum(row["validation"][field] for row in chunks) for field in fields}
    return {"valid_send_receive_audit": True, "steps": list(range(STEPS)),
            "communication_backend": contract["communication_backend"], "unchanged_gate_chunks": chunks, **totals}


def checked_timing(run):
    health = summarize(run, steps=STEPS, interval=1)
    equal(health["final_consumed_samples"], 640, "debug consumed samples")
    equal(health["final_consumed_global_tokens"], 2621440, "debug consumed input tokens")
    rows = timing_rows(select_timer_log(run / "logs"), warmup_steps=WARMUP, measured_steps=MEASURED,
                       log_interval=1, tokens_per_step=262144)
    values = [row["mean_optimizer_step_ms"] for row in rows]
    require(all(type(value) in (int, float) and math.isfinite(value) and value > 0 for value in values),
            "debug nonfinite or nonpositive step timer")
    equal([row["first_step"] for row in rows], list(range(WARMUP, STEPS)), "debug measured step grid")
    return health, {"unit": "complete optimizer-step milliseconds", "count": MEASURED,
        "mean": statistics.fmean(values), "median": statistics.median(values),
        "sample_sd": statistics.stdev(values), "min": min(values), "max": max(values),
        "aggregate_tokens_per_second": MEASURED * 262144 / (sum(values) / 1000), "by_update": rows}


def native_rank(run, output, rank):
    from analysis.aggregate_deep_trace import aggregate_rank
    from analysis.validate_torch_trace import inspect_rank

    output.mkdir(parents=True, exist_ok=False)
    inputs, validations, aggregates = evidence.Inputs(), [], []
    audit = run / "raw" / f"routing_audit_rank-{rank}.json"
    inputs.bind(audit)
    equal([row["step"] for row in inputs.read(audit)["steps"]], list(range(STEPS)), "native routing step grid")
    for step in range(STEPS):
        trace, call = evidence._trace_paths(run, rank, step)
        inputs.bind(trace)
        inputs.bind(call)
        result = inspect_rank(run, rank, step, trace_path=trace, count_path=call, world_size=16, cp=1)
        equal(result["device_link_stats"].get("untagged_cp_kv_p2p_by_owner", {}), {}, "native CP1 activity")
        validations.append(result)
        gc.collect()
        aggregate = aggregate_rank(run, rank, (step,), trace_path=trace)
        phase_rows(aggregate, "native")
        destination = output / f"step-{step:05d}.json"
        write_new(destination, aggregate)
        aggregates.append({"step": step, "path": str(destination), "sha256": inputs.bind(destination)})
        del aggregate
        gc.collect()
    inputs.recheck()
    write_new(output / "summary.json", {"valid_capture": True, "rank": rank, "steps": list(range(STEPS)),
        "world_size": 16, "cp": 1, "gradient_accumulation": 4, "validation": validations,
        "step_summaries": aggregates, "routing_audit_path": str(audit), "routing_audit_sha256": inputs.bind(audit),
        "analysis_source_sha256": {name: inputs.bind(ROOT / "analysis" / name) for name in (
            "validate_torch_trace.py", "aggregate_deep_trace.py")}})


def profile_commands(case, run, output, root=ROOT):
    commands = []
    for rank in range(16):
        if case in evidence.UCCL_CASES:
            for step in range(STEPS):
                trace, call = evidence._trace_paths(run, rank, step)
                commands.append((f"rank-{rank}-step-{step:05d}", [sys.executable,
                    str(root / "analysis/inspect_uccl_trace.py"), "--trace", str(trace), "--record", str(call),
                    "--output", str(output / "profiles" / f"rank-{rank}-step-{step:05d}.json")]))
        else:
            commands.append((f"rank-{rank}", [sys.executable, str(root / "tools/uccl_ep_debug_validate.py"),
                "--native-run", str(run), "--native-rank", str(rank),
                "--native-output", str(output / "profiles" / f"rank-{rank}")]))
    return commands


def run_profile_validators(case, run, output, workers, root=ROOT):
    (output / "profiles").mkdir()
    (output / "logs").mkdir()
    commands = profile_commands(case, run, output, root)
    write_new(output / "commands.json", commands)

    def execute(row):
        name, command = row
        with (output / "logs" / f"{name}.log").open("x") as stream:
            subprocess.run(command, cwd=root, stdout=stream, stderr=subprocess.STDOUT, check=True)
        return name

    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = [pool.submit(execute, command) for command in commands]
        for future in as_completed(pending):
            print(json.dumps({"validator": future.result(), "completed": True}), flush=True)


def checked_profile(case, run, output, root, inputs):
    for rank in range(16):
        summary = None
        if case not in evidence.UCCL_CASES:
            summary = inputs.read(output / "profiles" / f"rank-{rank}" / "summary.json")
            for key, value in {"valid_capture": True, "rank": rank, "steps": list(range(STEPS)),
                               "world_size": 16, "cp": 1, "gradient_accumulation": 4}.items():
                equal(summary.get(key), value, f"debug native profile {key}")
            equal(len(summary["validation"]), STEPS, "debug native validation coverage")
            equal(len(summary["step_summaries"]), STEPS, "debug native aggregation coverage")
            equal(summary["analysis_source_sha256"], {name: inputs.bind(root / "analysis" / name) for name in (
                "validate_torch_trace.py", "aggregate_deep_trace.py")}, "debug native analyzer identity")
            evidence._path(summary["routing_audit_path"], run / "raw" / f"routing_audit_rank-{rank}.json", "native audit")
            inputs.bind(run / "raw" / f"routing_audit_rank-{rank}.json", summary["routing_audit_sha256"])
        for step in range(STEPS):
            trace, call = evidence._trace_paths(run, rank, step)
            if case in evidence.UCCL_CASES:
                record = inputs.read(output / "profiles" / f"rank-{rank}-step-{step:05d}.json")
                for key, value in {"valid_uccl_trace_capture": True, "synthetic": False,
                                   "pair_count": 384, "paired_direction_count": 768}.items():
                    equal(record.get(key), value, f"debug UCCL capture {key}")
                equal(record["analysis_source_sha256"], inputs.bind(root / "analysis/inspect_uccl_trace.py"), "debug UCCL analyzer")
                evidence._bound_map(record["inputs"], (trace, call), inputs, "debug UCCL raw capture")
                phase_rows(record, "uccl")
            else:
                validation = summary["validation"][step]
                equal(validation["rank"], rank, "debug native validation rank")
                equal(validation["steps"], [step], "debug native validation step")
                for path, prefix in ((trace, "trace"), (call, "call_record")):
                    evidence._path(validation[f"{prefix}_path"], path, "debug native capture path")
                    inputs.bind(path, validation[f"{prefix}_sha256"])
                binding = summary["step_summaries"][step]
                equal(binding["step"], step, "debug native aggregate step")
                aggregate = output / "profiles" / f"rank-{rank}" / f"step-{step:05d}.json"
                evidence._path(binding["path"], aggregate, "debug native aggregate path")
                inputs.bind(aggregate, binding["sha256"])
                record = inputs.read(aggregate)
                equal(record["rank"], rank, "debug native aggregate rank")
                equal(sorted(record["step_scopes_ms"]), [f"AMPA/STEP/{step}"], "debug native aggregate scope")
                phase_rows(record, "native")
            journal = inputs.read(call)
            equal(journal["rank"], rank, "debug call rank")
            equal(journal["schema_version"], 2 if case in evidence.UCCL_CASES else 1, "debug call schema")
            equal(len(journal["steps"]), 1, "debug call step count")
            equal(journal["steps"][0]["step"], step, "debug call update")
            equal(journal["steps"][0]["skipped_iter"], False, "debug call skipped update")
            for field in ("profiler_stop_seconds", "export_seconds"):
                value = journal["steps"][0].get(field)
                require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
                        f"invalid debug profiler lifecycle cost {field}")
    return {"valid_all_rank_all_update_capture": True, "ranks": 16, "steps": list(range(STEPS)),
            "rank_update_captures": 160, "claim_limit": "Correlated recorded activity is not exposed communication latency or physical traffic."}


def validate_case(orchestration, output, *, workers=2, root=ROOT):
    require(not Path(orchestration).is_symlink() and not Path(output).is_symlink(), "symlink output or orchestration is forbidden")
    root, orchestration, output = Path(root).resolve(), Path(orchestration).resolve(), Path(output).resolve()
    require(orchestration.is_dir() and orchestration.parent == root / "runs", "invalid orchestration directory")
    require(not output.exists() and output.parent == root / "runs" and output != orchestration, "new output under runs required")
    inputs = evidence.Inputs()
    case, leg, run = one_case(orchestration, root, inputs)
    acquisition = inputs.read(orchestration / "acquisition.json")
    for key, value in {"mode": "full", "experiment_protocol": PROTOCOL, "partition": "debug", "case": case,
        "leg": leg, "warmup_steps": WARMUP, "measured_steps": MEASURED, "completed_updates": STEPS,
        "cache_budget_updates": 25, "cache_budget_samples": 1600,
        "topology": "cp1_ep16", "launch_policy": "torchrun_node4", "accepted_pilot_proof": None,
        "uccl_transport_policy": DEBUG_TRANSPORT_POLICY}.items():
        equal(acquisition.get(key), value, f"debug acquisition {key}")
    job = acquisition.get("job")
    require(type(job) is str and bool(job), "debug training job identity missing")
    sources = _ampa_runtime_sources(root)
    equal(acquisition["runtime_source_sha256"], sources["sha256"], "debug runtime source")
    canonical = manifest(orchestration / "runtime-source.sha256", inputs)
    equal(canonical, {str(root / path): digest for path, digest in sources["files"].items()}, "debug runtime manifest coverage")
    campaign_sources = manifest(orchestration / "campaign-source.sha256", inputs)
    equal(acquisition.get("campaign_source_sha256"), inputs.bind(orchestration / "campaign-source.sha256"),
          "debug campaign manifest identity")
    launch_sources = manifest(orchestration / "source.sha256", inputs)
    for relative in ("jobs/training/uccl_ep_debug_case.sbatch", "jobs/training/qwen3_uccl_debug_worker.sh",
                     "profiling/run_pretrain_debug.py", "tools/uccl_ep_debug_validate.py"):
        require(str(root / relative) in launch_sources, f"debug launch source missing: {relative}")
        require(str(root / relative) in campaign_sources, f"debug campaign source missing: {relative}")
    require(str(orchestration / "campaign-source.sha256") in launch_sources, "debug campaign manifest is not source-bound")
    completed = inputs.read(orchestration / "acquisition_completed.json")
    for key, value in {"success": True, "workers_srun_returned_zero": True, "mode": "full", "job": job,
        "runtime_source_sha256": sources["sha256"], "binary_sha256": acquisition["binary_sha256"],
        "cases_tsv_sha256": inputs.bind(orchestration / "cases.tsv"), "warmup_steps": WARMUP,
        "measured_steps": MEASURED, "completed_updates": STEPS}.items():
        equal(completed.get(key), value, f"debug completion {key}")
    contract = checked_contract(inputs.read(run / "config/comparison_contract.json"), case)
    equal(inputs.read(run / "config/debug_execution_contract.json"), execution_contract(), "debug execution/cache contract")
    resolved_path = run / "config/resolved.yaml"
    inputs.bind(resolved_path)
    checked_resolved(yaml.safe_load(resolved_path.read_text()), case, run)
    environment = inputs.read(run / "environment.json")
    for key, value in {"slurm_job_id": job, "ampa_runtime_sources": sources, "launch_policy": "torchrun_node4",
                       "runtime_topology": _EP16_RUNTIME_TOPOLOGIES["cp1_ep16"]}.items():
        equal(environment.get(key), value, f"debug environment {key}")
    placement = validate_rank_placement(environment["rank_placement"], expected_world_size=16, launch_policy="torchrun_node4")
    validate_runtime_groups(placement, expected_world_size=16, cp=1)
    manifests = {name: inputs.bind(run / "config" / name) for name in ("c0_manifest.json", "dataset_manifest.json", "model_manifest.json")}
    equal(manifests["c0_manifest.json"], contract["c0_identity_manifest_sha256"], "debug C0 manifest")
    equal(manifests["dataset_manifest.json"], contract["dataset"]["manifest_sha256"], "debug dataset manifest")
    reviewed_path = orchestration / "reviewed_inputs.json"
    reviewed = inputs.read(reviewed_path)
    for key, value in {"valid_reviewed_inputs": True, "slurm_job_id": job, "ampa_runtime_sources": sources,
        "c0_identity_manifest_sha256": manifests["c0_manifest.json"], "dataset_manifest_sha256": manifests["dataset_manifest.json"]}.items():
        equal(reviewed.get(key), value, f"debug reviewed input {key}")
    equal(reviewed["checkpoint_payload"]["verified"], True, "debug checkpoint payload")
    c0 = root / "checkpoint/megatron/qwen3-30b-a3b-base-c0"
    evidence._path(reviewed["checkpoint_payload"]["path"], c0, "debug checkpoint payload path")
    equal(contract["initialization"], {"mode": "pretrained", "checkpoint": str(c0)}, "debug canonical initialization")
    equal(environment["checkpoint"], str(c0), "debug actual C0")
    equal(inputs.bind(c0 / "ampa_c0_manifest.json"), manifests["c0_manifest.json"], "debug canonical manifest")
    equal(reviewed["native_checkouts"], {name: environment[name] for name in ("megatron_lm", "megatron_bridge")}, "debug native source")
    equal(environment["reviewed_inputs_proof"], {"path": str(reviewed_path), "sha256": inputs.bind(reviewed_path)}, "debug actual reviewed inputs")
    mode = inputs.read(run / "config/observation_mode.json")
    for key, value in {"data_mode": "indexed", "lightweight_profiling": "off", "nsys_trace": False,
        "torch_trace": leg == "profile", "sample_fingerprints": True, "router_observer_attached": False,
        "route_wrapper_installed": False, "dispatcher_preprocess_wrapper_installed": False}.items():
        equal(mode.get(key), value, f"debug observation {key}")
    capture = inputs.read(run / "config/all_step_capture.json")
    for key, value in {"enabled": leg == "profile", "steps": list(range(STEPS)),
                       "ranks": list(range(16)) if leg == "profile" else [0]}.items():
        equal(capture.get(key), value, f"debug capture declaration {key}")
    if leg == "control":
        require(not any((run / "traces/torch_profile").rglob("*.json*"))
                and not any((run / "raw/deep_trace").rglob("*.json")), "debug control has profile artifacts")
    fingerprints = {}
    for rank in range(16):
        inputs.bind(run / "logs" / f"rank-{rank}.out")
        path = run / "raw" / f"sample_fingerprint_rank-{rank}.json"
        fingerprints[str(rank)] = _checked_fingerprint_batches(inputs.read(path), rank=rank, accumulation=4,
            steps=STEPS, token_shape=[1, 4096], path=path)
    prefix_hash = hashlib.sha256(json.dumps(fingerprints, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    health, timing = checked_timing(run)
    ownership = inputs.read(run / "config/routing_ownership.json")
    audits = [inputs.read(run / "raw" / f"routing_audit_rank-{rank}.json") for rank in range(16)]
    received = checked_receives(contract, ownership, audits, environment)
    if case in evidence.UCCL_CASES:
        build = Path(contract["uccl_build_dir"])
        require(build.is_absolute() and build.is_dir() and not build.is_symlink(), "debug isolated build required")
        identity = verify_uccl_source(build)
        equal(inputs.read(build / "source_identity.json"), identity, "debug approved source identity")
        equal(acquisition["uccl_source_identity"], identity, "debug acquisition source identity")
        inputs.bind(build / "source/uccl-borrowed-scratch.patch", identity["patch_sha256"])
        for relative, digest in identity["modified_files"].items():
            inputs.bind(build / "uccl" / relative, digest)
        binary = evidence._uccl_identity(run, build, identity, placement, job, root, inputs)
        equal(binary, acquisition["binary_sha256"], "debug actual UCCL binary")
    output.mkdir()
    write_new(output / "training_health.json", health)
    write_new(output / "timing.json", timing)
    write_new(output / "received_routes.json", received)
    profile = None
    if leg == "profile":
        run_profile_validators(case, run, output, workers, root)
        profile = checked_profile(case, run, output, root, inputs)
    inputs.recheck()
    result = {"valid_debug_case": True, "experiment_protocol": PROTOCOL, "case": case, "leg": leg,
        "training_slurm_job_id": job, "validation_slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "run_dir": str(run), "orchestration_dir": str(orchestration), "completed_updates": STEPS,
        "warmup_steps": WARMUP, "measured_steps": MEASURED, "valid_training_health": True,
        "valid_send_receive_audit": True, "valid_capture": True if profile else None,
        "sample_prefix_sha256": prefix_hash, "consumed_cpu_batches": 640,
        "c0_identity_manifest_sha256": manifests["c0_manifest.json"],
        "dataset_manifest_sha256": manifests["dataset_manifest.json"], "runtime_source_sha256": sources["sha256"],
        "campaign_source_sha256": inputs.bind(orchestration / "campaign-source.sha256"),
        "binary_sha256": acquisition["binary_sha256"], "rank_placement": placement,
        "same_actual_rank_placement_across_allocations_claimed": False, "pure_profiler_overhead_available": False,
        "timing": timing, "profile": profile,
        "validator_sha256": inputs.bind(root / "tools/uccl_ep_debug_validate.py"),
        "input_files": [{"path": path, "sha256": digest} for path, digest in sorted(inputs.hashes.items())],
        "claim_limits": ["One allocation is one experiment; consecutive updates and sixteen ranks are not independent repetitions.",
            "Separate allocations can change placement, hardware and network conditions.",
            "Unprofiled complete optimizer steps describe observed performance with audits enabled.",
            "Logical payload and inclusive correlated activity do not measure physical bytes or exposed communication latency."]}
    write_new(output / "acceptance.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orchestration-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--workers", type=int, default=2, choices=range(1, 9))
    parser.add_argument("--native-run", type=Path)
    parser.add_argument("--native-rank", type=int, choices=range(16))
    parser.add_argument("--native-output", type=Path)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOB_PARTITION") not in ("debug", "preemptable"):
        parser.error("fresh configured Slurm compute is required")
    native = (args.native_run, args.native_rank, args.native_output)
    if any(value is not None for value in native):
        if not all(value is not None for value in native) or args.orchestration_dir is not None or args.output_dir is not None:
            parser.error("native worker arguments must be complete and exclusive")
        native_rank(args.native_run.resolve(), args.native_output.resolve(), args.native_rank)
    else:
        if args.orchestration_dir is None or args.output_dir is None:
            parser.error("orchestration-dir and output-dir are required")
        result = validate_case(args.orchestration_dir, args.output_dir, workers=args.workers)
        print(json.dumps({key: result[key] for key in ("valid_debug_case", "case", "leg", "training_slurm_job_id")}), flush=True)


if __name__ == "__main__":
    main()
