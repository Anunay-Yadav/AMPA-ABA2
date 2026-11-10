#!/usr/bin/env python3
"""Validate and aggregate bounded per-step files on allocated compute."""

import argparse
from collections import Counter
import gc
import json
import os
from pathlib import Path
import re
import yaml

from aggregate_deep_trace import aggregate_rank, _sha256
from validate_torch_trace import inspect_rank


def _workload(contract):
    """Accept only the reviewed fixed-token all-step experiment topologies."""
    parallelism = contract.get("parallelism", {})
    training = contract.get("training", {})
    world_size = parallelism.get("world_size", parallelism.get("ep"))
    cp = parallelism.get("cp")
    accumulation = training.get("gradient_accumulation")
    dimensions = (world_size, cp, parallelism.get("dense_dp"), accumulation)
    if (dimensions not in ((8, 2, 4, 16), (16, 1, 16, 4), (16, 2, 8, 8))
            or parallelism.get("ep") != world_size
            or any(parallelism.get(key) != 1 for key in ("tp", "pp", "etp"))
            or training.get("global_batch_size") != 64
            or training.get("micro_batch_size") != 1
            or training.get("sequence_length") != 4096
            or training.get("warmup_steps") != 5
            or training.get("measured_steps") != 20
            or training.get("log_interval") != 1):
        raise ValueError("all-step trace lacks a reviewed EP8/EP16 fixed-token workload")
    return world_size, cp, accumulation


def _expected_cp_counts(cp, accumulation):
    return ({"AttnFuncWithCPAndKVP2P": 48 * accumulation,
             "AttnFuncWithCPAndKVP2PBackward": 48 * accumulation * 2}
            if cp == 2 else {})


def _posthoc_binding(args, root):
    """Check the worker's proof identity and output geometry without raw parsing."""
    from tools.ep16_posthoc_proof import validate_output_root

    proof_path = args.posthoc_proof.resolve()
    proof_sha256 = _sha256(proof_path)
    proof = json.loads(proof_path.read_text())
    training_job, posthoc_job = proof.get("training_slurm_job_id"), proof.get("posthoc_slurm_job_id")
    if (type(proof.get("schema_version")) is not int or proof["schema_version"] != 1
            or proof.get("valid_posthoc_analysis") is not True
            or not isinstance(training_job, str) or not training_job
            or not isinstance(posthoc_job, str) or not posthoc_job or training_job == posthoc_job
            or posthoc_job != os.environ.get("SLURM_JOB_ID")
            or os.environ.get("SLURM_JOB_PARTITION") != "preemptable" or args.steps != 25
            or args.posthoc_proof.is_symlink() or proof_path.name != "posthoc_proof.json"
            or not isinstance(proof.get("pair_id"), str)
            or re.fullmatch(r"[a-zA-Z0-9_-]+", proof["pair_id"]) is None):
        raise ValueError("posthoc proof does not authorize this allocation or capture")
    validate_output_root(proof_path.parent, root, proof["pair_id"], proof["runs"])
    expected_runs = {route: {"control": str(root / "runs" / f"{proof['pair_id']}-cp1-{route}-control"),
                             "trace": str(root / "runs" / f"{proof['pair_id']}-cp1-{route}-torchprof-all")}
                     for route in ("normal", "group4-top2")}
    expected_dirs = {route: str(proof_path.parent / "analysis" / route) for route in expected_runs}
    routes = [name for name, pair in expected_runs.items()
              if Path(pair["trace"]).resolve() == args.run_dir.resolve()]
    if (proof["runs"] != expected_runs or proof["analysis_dirs"] != expected_dirs or len(routes) != 1
            or args.output_dir.is_symlink()
            or args.output_dir.resolve() != Path(expected_dirs[routes[0]])
            or any(args.output_dir.resolve() == Path(pair[leg]).resolve()
                   or Path(pair[leg]).resolve() in args.output_dir.resolve().parents
                   for pair in proof["runs"].values() for leg in ("trace", "control"))):
        raise ValueError("posthoc proof does not authorize this capture or separate output path")
    for name in ("analyze_all_step_trace.py", "validate_torch_trace.py", "aggregate_deep_trace.py"):
        if _sha256(root / "analysis" / name) != proof["posthoc_runtime_sources"]["files"][f"analysis/{name}"]:
            raise ValueError("posthoc analyzer source differs from its proof")
    return {"proof_path": str(proof_path), "proof_sha256": proof_sha256,
            "training_slurm_job_id": training_job, "posthoc_slurm_job_id": posthoc_job}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--posthoc-proof", type=Path)
    args = parser.parse_args()
    if (args.output_dir is None) != (args.posthoc_proof is None):
        parser.error("a separate analysis directory requires an explicit posthoc proof")
    posthoc_identity = None
    if args.posthoc_proof is not None:
        posthoc_identity = _posthoc_binding(args, Path(__file__).resolve().parent.parent)
    contract = json.loads((args.run_dir / "config" / "comparison_contract.json").read_text())
    world_size, cp, accumulation = _workload(contract)
    if args.rank not in range(world_size) or args.steps not in (2, 25):
        parser.error(f"all-step acquisition requires rank 0–{world_size - 1} and 25 steps or a two-step pilot")
    if contract["training"].get("completed_steps", 25) != args.steps:
        raise ValueError("requested analysis length differs from the declared run length")
    capture = json.loads((args.run_dir / "config" / "all_step_capture.json").read_text())
    if (capture["enabled"] is not True or capture["steps"] != list(range(args.steps))
            or capture["ranks"] != list(range(world_size))):
        raise ValueError("all-step capture identity mismatch")
    resolved = yaml.safe_load((args.run_dir / "config" / "resolved.yaml").read_text())
    if (resolved["profiling"]["use_pytorch_profiler"] is not False
            or resolved["profiling"]["use_nsys_profiler"] is not False):
        raise ValueError("AMPA per-step capture cannot run with either Bridge profiler")
    audit_path = args.run_dir / "raw" / f"routing_audit_rank-{args.rank}.json"
    audit_sha256 = _sha256(audit_path)
    audit = json.loads(audit_path.read_text())
    if audit["valid"] is not True or [row["step"] for row in audit["steps"]] != list(range(args.steps)):
        raise ValueError("all-step routing audit incomplete")
    analysis_root = args.output_dir if args.output_dir is not None else args.run_dir / "analysis" / "all_steps"
    destination = analysis_root / f"rank-{args.rank}"
    destination.mkdir(parents=True, exist_ok=False)
    validations = []
    phase_totals = {}
    step_summaries = []
    expected_cp = _expected_cp_counts(cp, accumulation)
    for step in range(args.steps):
        trace = args.run_dir / "traces" / "torch_profile" / f"step-{step:05d}" / f"rank-{args.rank}.json.gz"
        counts = args.run_dir / "raw" / "deep_trace" / f"step-{step:05d}" / f"rank-{args.rank}.json"
        validation = inspect_rank(args.run_dir, args.rank, step, trace_path=trace, count_path=counts,
                                  world_size=world_size, cp=cp)
        if validation["device_link_stats"].get("untagged_cp_kv_p2p_by_owner", {}) != expected_cp:
            raise ValueError(f"rank {args.rank} step {step}: CP count mismatch")
        validations.append(validation)
        gc.collect()
        summary = aggregate_rank(args.run_dir, args.rank, (step,), trace_path=trace)
        if _sha256(trace) != validation["trace_sha256"] or _sha256(counts) != validation["call_record_sha256"]:
            raise ValueError("trace changed during analysis")
        summary_path = destination / f"step-{step:05d}.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
        step_summaries.append({"step": step, "path": str(summary_path), "sha256": _sha256(summary_path)})
        for row in summary["phase_rows"]:
            totals = phase_totals.setdefault(row["phase"], Counter())
            for key in ("calls", "cpu_scope_sum_ms", "cpu_exclusive_sum_ms",
                        "gpu_exclusive_activity_sum_ms", "gpu_exclusive_nccl_sum_ms"):
                totals[key] += row[key]
        del summary
        gc.collect()
        print(json.dumps({"rank": args.rank, "step": step, "valid": True}), flush=True)
    if _sha256(audit_path) != audit_sha256:
        raise ValueError("routing audit changed during analysis")
    if posthoc_identity is not None and _sha256(args.posthoc_proof) != posthoc_identity["proof_sha256"]:
        raise ValueError("posthoc proof changed during analysis")
    result = {"valid_capture": True, "rank": args.rank, "steps": list(range(args.steps)),
              "world_size": world_size, "cp": cp, "gradient_accumulation": accumulation,
              "routing_audit_path": str(audit_path), "routing_audit_sha256": audit_sha256,
              "validation": validations, "phase_totals": phase_totals,
              "step_summaries": step_summaries,
              "analysis_source_sha256": {name: _sha256(Path(__file__).with_name(name)) for name in (
                  "analyze_all_step_trace.py", "validate_torch_trace.py", "aggregate_deep_trace.py")},
              "claim_limit": "Summed CUDA activity can overlap. Export and routing audit perturb elapsed iterations."}
    if posthoc_identity is not None:
        result["posthoc_analysis"] = posthoc_identity
    (destination / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
