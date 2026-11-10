#!/usr/bin/env python3
"""Verify unchanged inputs and archive the newly reviewed AMPA source identity."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

from ampa_profile.provenance import _ampa_runtime_sources, _git_identity
from tools.c0_validation import canonical_json_sha256, verify_payload_inventory
from tools.ep16_recovery_placement import reference_records, sha256, write_new


def reviewed_inputs(reference_run, root, manifest):
    environment, contract = reference_records(reference_run)
    native = {}
    for checkout, key in (("Megatron-LM", "megatron_lm"), ("Megatron-Bridge", "megatron_bridge")):
        native[key] = _git_identity(root / checkout)
        if native[key] != environment.get(key):
            raise ValueError(f"native {checkout} checkout differs from the previous experiment")
    if sha256(manifest) != contract["dataset"]["manifest_sha256"]:
        raise ValueError("reviewed dataset manifest differs from the previous experiment")
    checkpoint = Path(contract["initialization"]["checkpoint"])
    if checkpoint.resolve() != (root / "checkpoint/megatron/qwen3-30b-a3b-base-c0").resolve():
        raise ValueError("reviewed experiment requires the same native C0")
    manifest_path = checkpoint / "ampa_c0_manifest.json"
    if sha256(manifest_path) != contract["c0_identity_manifest_sha256"]:
        raise ValueError("reviewed C0 identity differs from the previous experiment")
    c0_manifest = json.loads(manifest_path.read_text())
    payload = verify_payload_inventory(checkpoint, c0_manifest)
    if sha256(root / "configs/model/qwen3_30b_a3b_base.json") != sha256(reference_run / "config/model_manifest.json"):
        raise ValueError("reviewed model manifest differs from the previous experiment")
    import torch

    runtime = {"pytorch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
               "nccl_version": list(torch.cuda.nccl.version())}
    if any(value != environment.get(key) for key, value in runtime.items()):
        raise ValueError("reviewed PyTorch/CUDA/NCCL runtime differs")
    return {"valid_reviewed_inputs": True, "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "reference_run": str(reference_run.resolve()), "reference_environment_sha256": sha256(reference_run / "environment.json"),
            "reference_runtime_source_sha256": environment["ampa_runtime_sources"]["sha256"],
            "ampa_runtime_sources": _ampa_runtime_sources(root), "native_checkouts": native, "runtime_versions": runtime,
            "dataset_manifest_path": str(manifest.resolve()), "dataset_manifest_sha256": sha256(manifest),
            "c0_identity_manifest_sha256": contract["c0_identity_manifest_sha256"],
            "checkpoint_payload": {"verified": True, "path": str(checkpoint.resolve()), **payload,
                                   "inventory_sha256": canonical_json_sha256(c0_manifest["payload_inventory"])}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("SLURM_JOB_PARTITION") != "preemptable" or not os.environ.get("SLURM_JOB_ID"):
        parser.error("reviewed input verification requires a preemptable Slurm allocation")
    result = reviewed_inputs(args.reference_run, args.root, args.manifest)
    source_dir = args.output_dir / "source/runtime"
    source_dir.mkdir(parents=True, exist_ok=False)
    for relative, expected in result["ampa_runtime_sources"]["files"].items():
        destination = source_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.root / relative, destination)
        if sha256(destination) != expected or sha256(args.root / relative) != expected:
            raise ValueError(f"reviewed runtime source changed during snapshot: {relative}")
    write_new(args.output_dir / "reviewed_inputs.json", result)
    print(json.dumps({"valid_reviewed_inputs": True, "slurm_job_id": result["slurm_job_id"],
                      "checkpoint_payload": result["checkpoint_payload"],
                      "ampa_runtime_source_sha256": result["ampa_runtime_sources"]["sha256"]}), flush=True)


if __name__ == "__main__":
    main()
