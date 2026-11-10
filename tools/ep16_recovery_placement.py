#!/usr/bin/env python3
"""Fail closed before reusing a completed EP16 capture across allocations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from profiling.ampa_profile.provenance import (
    _ampa_runtime_sources,
    _git_identity,
    _local_placement,
    validate_rank_placement,
    validate_runtime_groups,
)


PLACEMENT_FIELDS = (
    "rank", "hostname", "slurm_node_id", "slurm_local_id",
    "cuda_visible_devices", "cuda_visible_device_count", "cuda_current_device",
    "gpu_name", "physical_gpu_uuid", "physical_gpu_identity_source",
)
RUNTIME_FIELDS = ("pytorch_version", "cuda_version", "nccl_version")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_same_placement(reference: list[dict], actual: list[dict]) -> list[dict]:
    """Reject a changed GPU or visibility even when node names match."""
    old = validate_rank_placement(reference, expected_world_size=16)
    new = validate_rank_placement(actual, expected_world_size=16)
    for previous, current in zip(old, new):
        for field in PLACEMENT_FIELDS:
            if previous.get(field) != current.get(field):
                raise ValueError(f"rank {current['rank']} recovery placement differs: {field}")
    return new


def reference_records(reference_run: Path) -> tuple[dict, dict]:
    environment = json.loads((reference_run / "environment.json").read_text())
    contract = json.loads((reference_run / "config/comparison_contract.json").read_text())
    expected_parallelism = {"tp": 1, "pp": 1, "cp": 1, "ep": 16, "etp": 1, "dense_dp": 16}
    if (contract.get("parallelism") != expected_parallelism
            or contract.get("training", {}).get("completed_steps") != 25
            or contract.get("routing") != {"topk": 8, "num_groups": None, "group_topk": None}
            or not environment.get("slurm_job_id")):
        raise ValueError("recovery requires a complete ordinary CP1/EP16 reference contract")
    placement = validate_rank_placement(environment["rank_placement"], expected_world_size=16)
    validate_runtime_groups(placement, expected_world_size=16, cp=1)
    return environment, contract


def frozen_inputs(reference_run: Path, root: Path, manifest: Path) -> dict:
    """Bind the old source and inputs without changing its provenance collector."""
    environment, contract = reference_records(reference_run)
    sources = _ampa_runtime_sources(root)
    if sources != environment.get("ampa_runtime_sources"):
        raise ValueError("frozen AMPA runtime sources differ from the completed capture")
    for checkout, key in (("Megatron-LM", "megatron_lm"), ("Megatron-Bridge", "megatron_bridge")):
        if _git_identity(root / checkout) != environment.get(key):
            raise ValueError(f"frozen {checkout} checkout identity differs")
    if sha256(manifest) != contract["dataset"]["manifest_sha256"]:
        raise ValueError("recovery dataset manifest differs from the completed capture")
    checkpoint = Path(contract["initialization"]["checkpoint"])
    if sha256(checkpoint / "ampa_c0_manifest.json") != contract["c0_identity_manifest_sha256"]:
        raise ValueError("recovery C0 identity differs from the completed capture")
    if sha256(root / "configs/model/qwen3_30b_a3b_base.json") != sha256(reference_run / "config/model_manifest.json"):
        raise ValueError("recovery model manifest differs from the completed capture")
    return {"ampa_runtime_sources": sources,
            "dataset_manifest_sha256": sha256(manifest),
            "c0_identity_manifest_sha256": contract["c0_identity_manifest_sha256"]}


def write_new(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--task", action="store_true")
    mode.add_argument("--verify", action="store_true")
    mode.add_argument("--verify-sources", action="store_true")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    environment, _ = reference_records(args.reference_run)
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id or job_id == environment["slurm_job_id"]:
        raise ValueError("recovery must record a distinct, actual Slurm allocation")
    if args.verify_sources:
        result = frozen_inputs(args.reference_run, args.root, args.manifest)
        print(json.dumps({"valid_frozen_inputs": True, "slurm_job_id": job_id,
                          "source_sha256": result["ampa_runtime_sources"]["sha256"]}), flush=True)
        return
    if args.task:
        import torch
        rank = int(os.environ["SLURM_PROCID"])
        if int(os.environ["SLURM_NTASKS"]) != 16 or rank not in range(16):
            raise ValueError("recovery placement probe requires sixteen Slurm tasks")
        placement = _local_placement(torch, rank=rank)
        previous = next(row for row in environment["rank_placement"] if row["rank"] == rank)
        for field in PLACEMENT_FIELDS:
            if placement.get(field) != previous.get(field):
                raise ValueError(f"rank {rank} recovery placement differs: {field}")
        runtime = {"pytorch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
                   "nccl_version": list(torch.cuda.nccl.version())}
        for field in RUNTIME_FIELDS:
            if runtime[field] != environment.get(field):
                raise ValueError(f"rank {rank} recovery runtime differs: {field}")
        write_new(args.output_dir / f"rank-{rank}.json", {
            "slurm_job_id": job_id, "reference_slurm_job_id": environment["slurm_job_id"],
            "reference_environment_sha256": sha256(args.reference_run / "environment.json"),
            "placement": placement, "runtime": runtime,
        })
        print(json.dumps({"rank": rank, "same_physical_gpu": True}), flush=True)
        return
    records = [json.loads((args.output_dir / f"rank-{rank}.json").read_text()) for rank in range(16)]
    for record in records:
        if (record.get("slurm_job_id") != job_id
                or record.get("reference_slurm_job_id") != environment["slurm_job_id"]
                or record.get("reference_environment_sha256") != sha256(args.reference_run / "environment.json")):
            raise ValueError("recovery placement proof has an inconsistent allocation or reference")
        for field in RUNTIME_FIELDS:
            if record["runtime"][field] != environment.get(field):
                raise ValueError(f"recovery placement proof runtime differs: {field}")
    actual = require_same_placement(environment["rank_placement"], [row["placement"] for row in records])
    inputs = frozen_inputs(args.reference_run, args.root, args.manifest)
    added_sources = {relative: sha256(args.root / relative) for relative in (
        "jobs/training/ep16_all_steps_recovery.sbatch", "tools/ep16_recovery_placement.py",
        "analysis/compare_ep16_recovery.py",
    )}
    result = {"valid_physical_placement": True, "slurm_job_id": job_id,
              "reference_slurm_job_id": environment["slurm_job_id"],
              "reference_run": str(args.reference_run),
              "reference_environment_sha256": sha256(args.reference_run / "environment.json"),
              "rank_placement": actual, "frozen_inputs": inputs,
              "recovery_source_sha256": added_sources,
              "original_suite_sha256": inputs["ampa_runtime_sources"]["files"]["jobs/training/ep16_all_steps_group_routing.sbatch"],
              "claim_limit": "Runtime process groups, expert ownership, batches, health and all-step event quality still require final comparison gates. Identical GPUs do not remove variation between allocations."}
    write_new(args.output_dir / "placement_verified.json", result)
    print(json.dumps({"valid_physical_placement": True, "slurm_job_id": job_id,
                      "reference_slurm_job_id": environment["slurm_job_id"]}), flush=True)


if __name__ == "__main__":
    main()
