#!/usr/bin/env python3
"""Materialize or verify the immutable local HF source for AMPA C0 conversion.

Bridge's lazy safetensor accessor resolves a remote model path without carrying
the wrapper revision.  C0 conversion must therefore consume this audited local
snapshot rather than a Hub identifier.  This program is intentionally run only
inside the checkpoint Slurm jobs; it neither loads model weights into a model
nor initializes CUDA.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


MANIFEST_NAME = "ampa_hf_snapshot_manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory(root: Path) -> list[dict[str, Any]]:
    """Hash every snapshot file except the manifest written by this program."""
    files: list[dict[str, Any]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.name == MANIFEST_NAME:
            continue
        files.append(
            {
                "path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    if not files:
        raise ValueError(f"snapshot contains no files: {root}")
    return files


def verify(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = args.output_dir / MANIFEST_NAME
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"snapshot manifest is missing: {manifest_path}") from error
    if manifest.get("lifecycle", {}).get("status") != "materialized_pinned_snapshot":
        raise ValueError("snapshot manifest is not materialized_pinned_snapshot")
    if manifest.get("model_id") != args.model_id or manifest.get("revision") != args.revision:
        raise ValueError("snapshot manifest model identity does not match requested model/revision")
    expected = manifest.get("files")
    actual = inventory(args.output_dir)
    if expected != actual:
        raise ValueError("snapshot file inventory differs from its immutable manifest")
    return manifest


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to reuse or overwrite snapshot directory: {args.output_dir}")
    from huggingface_hub import HfApi, snapshot_download

    model_info = HfApi().model_info(args.model_id, revision=args.revision)
    resolved_revision = getattr(model_info, "sha", None)
    if resolved_revision != args.revision:
        raise ValueError(
            f"Hub resolved {args.model_id}@{args.revision} to {resolved_revision}; "
            "AMPA requires the requested full commit SHA"
        )
    snapshot_download(
        repo_id=args.model_id,
        repo_type="model",
        revision=args.revision,
        local_dir=args.output_dir,
    )
    files = inventory(args.output_dir)
    manifest = {
        "schema_version": 1,
        "lifecycle": {"status": "materialized_pinned_snapshot", "immutable_after_write": True},
        "model_id": args.model_id,
        "revision": args.revision,
        "resolved_revision": resolved_revision,
        "files": files,
        "file_count": len(files),
        "total_bytes": sum(item["bytes"] for item in files),
    }
    destination = args.output_dir / MANIFEST_NAME
    destination.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def main() -> None:
    args = parse_args()
    result = verify(args) if args.verify_only else materialize(args)
    print(
        json.dumps(
            {
                "model_id": result["model_id"],
                "revision": result["revision"],
                "file_count": result["file_count"],
                "total_bytes": result["total_bytes"],
                "verified": args.verify_only,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
