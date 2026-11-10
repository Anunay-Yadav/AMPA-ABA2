#!/usr/bin/env python3
"""Prepare explicit reuse of the first debug case's immutable-input proof.

This records current source hashes but does not compare current checkpoint,
indexed-data, native-build, or source bytes against earlier inventories.
Training and all measurement recorders remain enabled in the launch wrapper.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil

from profiling.ampa_profile.provenance import _ampa_runtime_sources


ROOT = Path(__file__).resolve().parents[1]
FIRST_ROOT = ROOT / "runs/uccl-debug5x5-20261104T1415Z-grouped-profile"
CASES = ("ordinary", "grouped", "uccl_ordinary", "uccl")
REUSE_MODE = "reused_first_case"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    require(Path(path).is_file() and not Path(path).is_symlink(), f"expected regular metadata file: {path}")
    return json.loads(Path(path).read_text())


def write_new(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def metadata_hash(path):
    """Hash small metadata solely to record its identity, not validate payload."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_first_case(first_root, *, root=ROOT):
    first_root, root = Path(first_root), Path(root)
    require(first_root.is_absolute() and first_root.parent == root / "runs"
            and first_root.is_dir() and not first_root.is_symlink(), "invalid first verified root")
    proof_path = first_root / "reviewed_inputs.json"
    proof = read_json(proof_path)
    acquired = read_json(first_root / "acquisition.json")
    completed = read_json(first_root / "acquisition_completed.json")
    require(proof.get("valid_reviewed_inputs") is True and proof.get("checkpoint_payload", {}).get("verified") is True,
            "first case does not record successful immutable input verification")
    job = proof.get("slurm_job_id")
    require(type(job) is str and job.isdigit() and acquired.get("job") == completed.get("job") == job,
            "first input proof and acquisition job identities differ")
    require(completed.get("success") is True and completed.get("workers_srun_returned_zero") is True
            and completed.get("completed_updates") == 10, "first ten-update acquisition did not complete")
    require(acquired.get("experiment_protocol") == "debug_one_case_5_plus_5_v1"
            and acquired.get("case") == "grouped" and acquired.get("leg") == "profile",
            "unexpected first debug case")
    rows = [line.split("\t") for line in (first_root / "cases.tsv").read_text().splitlines()]
    require(len(rows) == 1 and len(rows[0]) == 3 and rows[0][:2] == ["grouped", "profile"],
            "first acquisition must name the one completed grouped profile")
    run = Path(rows[0][2])
    require(run.is_absolute() and run.parent == root / "runs" and run.is_dir() and not run.is_symlink(),
            "first case run is outside project runs")
    return {"root": first_root, "run": run, "proof": proof, "acquisition": acquired,
            "proof_reference": {"path": str(proof_path), "sha256": metadata_hash(proof_path), "slurm_job_id": job}}


def prepare(orchestration, run, case, leg, *, first_root=FIRST_ROOT, root=ROOT):
    root, orchestration, run = Path(root), Path(orchestration), Path(run)
    require(case in CASES and leg in ("control", "profile"), "invalid reuse case or leg")
    require(orchestration.is_absolute() and run.is_absolute() and orchestration.parent == run.parent == root / "runs"
            and orchestration != run and not orchestration.is_symlink() and not run.is_symlink(),
            "new reuse directories must be separate direct project runs")
    job = os.environ.get("SLURM_JOB_ID", "")
    require(job.isdigit() and os.environ.get("SLURM_JOB_PARTITION") == "debug"
            and os.environ.get("SLURM_JOB_NUM_NODES") == "4", "fresh four-node debug allocation required")
    first = load_first_case(first_root, root=root)
    require(orchestration != first["root"] and run != first["run"], "reuse must preserve first case evidence")
    orchestration.mkdir(parents=True, exist_ok=True)
    (run / "config").mkdir(parents=True, exist_ok=True)
    for name in ("c0_manifest.json", "dataset_manifest.json", "model_manifest.json"):
        destination = run / "config" / name
        require(not destination.exists() and not destination.is_symlink(), "reuse manifest destination already exists")
        shutil.copyfile(first["run"] / "config" / name, destination)
    c0 = read_json(run / "config/c0_manifest.json")
    dataset = read_json(run / "config/dataset_manifest.json")
    proof = copy.deepcopy(first["proof"])
    sources = _ampa_runtime_sources(root)
    reuse = {"verification_mode": REUSE_MODE, "first_verified_proof": first["proof_reference"],
        "payload_verified_in_current_allocation": False,
        "indexed_payload_verified_in_current_allocation": False,
        "native_build_reverified_in_current_allocation": False,
        "recording_policy": "All measurement records remain enabled; immutable file verification is reused and analysis is deferred."}
    proof.update(slurm_job_id=job, ampa_runtime_sources=sources, verification_mode=REUSE_MODE,
                 verification_reuse=reuse)
    proof["checkpoint_payload"].update(verification_mode=REUSE_MODE,
        verification_slurm_job_id=first["proof_reference"]["slurm_job_id"],
        verified_in_current_allocation=False, first_verified_proof=first["proof_reference"])
    write_new(orchestration / "reviewed_inputs.json", proof)
    context = {"schema_version": 1, "verification_mode": REUSE_MODE, "verification_reuse": reuse,
        "slurm_job_id": job, "run_dir": str(run), "pretrained_checkpoint": proof["checkpoint_payload"]["path"],
        "c0_identity_manifest_sha256": proof["c0_identity_manifest_sha256"],
        "c0_hf_snapshot_manifest_sha256": c0.get("hf_snapshot_manifest_sha256"),
        "hf_model_id": c0.get("hf_model"), "hf_revision": c0.get("hf_revision"),
        "dataset_manifest_path": str(run / "config/dataset_manifest.json"),
        "dataset_manifest_sha256": proof["dataset_manifest_sha256"], "indexed_manifest": dataset}
    write_new(orchestration / "input_reuse_context.json", context)
    with (orchestration / "runtime-source.sha256").open("x") as stream:
        for name, digest in sorted(sources["files"].items()):
            stream.write(f"{digest}  {root / name}\n")
    acquired = copy.deepcopy(first["acquisition"])
    campaign = orchestration / "campaign-source.sha256"
    require(campaign.is_file() and not campaign.is_symlink(), "new campaign source identity must be recorded before preparation")
    acquired.update(job=job, case=case, leg=leg, runtime_source_sha256=sources["sha256"],
        campaign_source_sha256=metadata_hash(campaign), accepted_pilot_proof=None, historical_pilot_proof=None,
        verification_mode=REUSE_MODE, verification_reuse=reuse, acquisition_analysis_deferred=True,
        first_verified_case={"root": str(first["root"]), "run_dir": str(first["run"]), **first["proof_reference"]})
    write_new(orchestration / "acquisition.json", acquired)
    return {"reviewed_inputs": proof, "context": context, "acquisition": acquired}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true", required=True)
    parser.add_argument("--orchestration-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--leg", choices=("control", "profile"), required=True)
    args = parser.parse_args()
    first = Path(os.environ.get("AMPA_UCCL_VERIFIED_FIRST_ROOT", str(FIRST_ROOT)))
    result = prepare(args.orchestration_dir, args.run_dir, args.case, args.leg, first_root=first)
    print(json.dumps({"prepared": True, "verification_mode": REUSE_MODE,
        "job": result["acquisition"]["job"], "first_verified_proof": result["context"]["verification_reuse"]["first_verified_proof"]}), flush=True)


if __name__ == "__main__":
    main()
