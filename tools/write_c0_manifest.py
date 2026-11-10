#!/usr/bin/env python3
"""Write immutable identity metadata next to a successfully imported AMPA C0."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--hf-model", required=True)
    parser.add_argument("--hf-revision", required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--hf-snapshot-manifest", type=Path, required=True)
    parser.add_argument("--dataset-independent", action="store_true")
    args = parser.parse_args()
    if not args.checkpoint.is_dir():
        parser.error(f"checkpoint directory does not exist: {args.checkpoint}")
    if not args.model_manifest.is_file():
        parser.error(f"model manifest does not exist: {args.model_manifest}")
    if not args.hf_snapshot_manifest.is_file():
        parser.error(f"HF snapshot manifest does not exist: {args.hf_snapshot_manifest}")
    try:
        model_lock = json.loads(args.model_manifest.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        parser.error(f"model manifest is not valid JSON: {error}")
    pinned_model = model_lock.get("hugging_face", {}).get("model_id")
    pinned_revision = model_lock.get("hugging_face", {}).get("revision")
    if (pinned_model, pinned_revision) != (args.hf_model, args.hf_revision):
        parser.error("--hf-model/--hf-revision do not match the pinned model manifest")
    try:
        snapshot = json.loads(args.hf_snapshot_manifest.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        parser.error(f"HF snapshot manifest is not valid JSON: {error}")
    if snapshot.get("lifecycle", {}).get("status") != "materialized_pinned_snapshot":
        parser.error("HF snapshot manifest is not a materialized pinned snapshot")
    if (snapshot.get("model_id"), snapshot.get("revision")) != (args.hf_model, args.hf_revision):
        parser.error("HF snapshot manifest does not match --hf-model/--hf-revision")
    destination = args.checkpoint / "ampa_c0_manifest.json"
    if destination.exists():
        parser.error(f"refusing to overwrite existing C0 manifest: {destination}")
    root = Path(__file__).resolve().parents[1]
    payload = {
        "schema_version": 1,
        "hf_model": args.hf_model,
        "hf_revision": args.hf_revision,
        "model_manifest_sha256": hashlib.sha256(args.model_manifest.read_bytes()).hexdigest(),
        "hf_snapshot_manifest_sha256": hashlib.sha256(args.hf_snapshot_manifest.read_bytes()).hexdigest(),
        "payload_inventory": _payload_inventory(args.checkpoint),
        "parallelism": {"tp": 1, "pp": 1, "cp": 1, "ep": 8, "etp": 1},
        "precision": "bfloat16",
        "dataset_independent": args.dataset_independent,
        "megatron_lm_revision": _git_revision(root / "Megatron-LM"),
        "megatron_bridge_revision": _git_revision(root / "Megatron-Bridge"),
        "lifecycle": {
            "status": "imported",
            "immutable_after_write": True,
            "validation_record": "checkpoint/validation/qwen3-30b-a3b-base-c0/validation.json",
        },
    }
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _git_revision(path: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _payload_inventory(checkpoint: Path) -> dict[str, object]:
    """Create the one-time C0 payload identity before its directory is sealed."""
    files: list[dict[str, object]] = []
    for path in sorted(item for item in checkpoint.rglob("*") if item.is_file()):
        if path.name == "ampa_c0_manifest.json":
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        files.append(
            {
                "path": str(path.relative_to(checkpoint)),
                "bytes": path.stat().st_size,
                "sha256": digest.hexdigest(),
            }
        )
    if not files:
        raise ValueError(f"C0 contains no payload files: {checkpoint}")
    return {
        "files": files,
        "file_count": len(files),
        "total_bytes": sum(int(item["bytes"]) for item in files),
    }


if __name__ == "__main__":
    main()
