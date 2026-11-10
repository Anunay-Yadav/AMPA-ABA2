#!/usr/bin/env python3
"""Shared, dependency-free contracts for AMPA's external C0 validation.

The native C0 directory is sealed after import.  This module therefore only
reads its identity manifest and writes records in a separate location.  The
GPU runner in :mod:`tools.validate_c0_compute` owns model loading and all
expensive measurements; the functions here deliberately remain CPU-only.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


IDENTITY_MANIFEST_NAME = "ampa_c0_manifest.json"
HF_SNAPSHOT_MANIFEST_NAME = "ampa_hf_snapshot_manifest.json"
MODEL_PARITY_SCHEMA_VERSION = 1
FINAL_PROFILE_ELIGIBILITY_SCHEMA_VERSION = 1
OPTIMIZER_SANITY_SCHEMA_VERSION = 1
REQUIRED_STAGES = ("identity", "architecture", "runtime_topology", "parameters", "router", "logits")
_STAGE_STATUSES = frozenset({"passed", "failed", "unsupported"})
BASELINE_RUNTIME_TOPOLOGY = {
    "world_size": 8,
    "tp": 1,
    "pp": 1,
    "cp": 1,
    "ep": 8,
    "etp": 1,
    "dense_data_parallel_size": 8,
    "expert_data_parallel_size": 1,
}
_MODEL_PARITY_KIND = "ampa_external_c0_model_parity"
_FINAL_PROFILE_ELIGIBILITY_KIND = "ampa_external_c0_profile_eligibility"
_OPTIMIZER_SANITY_KIND = "ampa_external_c0_optimizer_sanity"


class C0ValidationError(ValueError):
    """Raised when an immutable C0 identity or result contract is invalid."""


def sha256_bytes(contents: bytes) -> str:
    """Return the SHA256 digest of bytes as lowercase hexadecimal text."""
    return hashlib.sha256(contents).hexdigest()


def sha256_file(path: Path) -> str:
    """Return the SHA256 digest of a file without reading it all at once."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: object) -> str:
    """Hash a JSON value deterministically for compact result binding."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return sha256_bytes(encoded)


def load_json_object(path: Path, *, label: str) -> dict[str, object]:
    """Read one JSON object and provide a stable validation error on failure."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise C0ValidationError(f"{label} is missing: {path}") from error
    except json.JSONDecodeError as error:
        raise C0ValidationError(f"{label} is not valid JSON: {error}") from error
    if not isinstance(value, dict):
        raise C0ValidationError(f"{label} must contain a JSON object")
    return value


def validate_c0_identity_manifest(
    manifest: Mapping[str, object],
    *,
    model_manifest_sha256: str,
    model_id: str,
    revision: str,
) -> dict[str, object]:
    """Validate the immutable C0 identity fields before any model is loaded.

    This validates the manifest's internal accounting.  ``verify_payload_inventory``
    separately binds that accounting to the actual immutable checkpoint files.
    """
    if manifest.get("schema_version") != 1:
        raise C0ValidationError("C0 identity manifest schema_version must be 1")
    if manifest.get("hf_model") != model_id or manifest.get("hf_revision") != revision:
        raise C0ValidationError("C0 identity manifest does not match the pinned model and revision")
    if manifest.get("model_manifest_sha256") != model_manifest_sha256:
        raise C0ValidationError("C0 identity manifest does not bind the supplied model manifest")
    lifecycle = _mapping(manifest.get("lifecycle"), "C0 lifecycle")
    if lifecycle.get("status") != "imported" or lifecycle.get("immutable_after_write") is not True:
        raise C0ValidationError("C0 identity manifest is not an imported immutable C0")
    if manifest.get("precision") != "bfloat16":
        raise C0ValidationError("C0 identity manifest precision must be bfloat16")
    _sha256_text(manifest.get("hf_snapshot_manifest_sha256"), "HF snapshot manifest SHA256")
    for name in ("megatron_lm_revision", "megatron_bridge_revision"):
        _git_revision(manifest.get(name), name)

    parallelism = _mapping(manifest.get("parallelism"), "C0 parallelism")
    expected_parallelism = {"tp", "pp", "cp", "ep", "etp"}
    if set(parallelism) != expected_parallelism or any(
        not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in parallelism.values()
    ):
        raise C0ValidationError("C0 parallelism must contain positive tp, pp, cp, ep, and etp integers")

    inventory = _validate_inventory(_mapping(manifest.get("payload_inventory"), "C0 payload inventory"))
    return {
        "hf_model": model_id,
        "hf_revision": revision,
        "model_manifest_sha256": model_manifest_sha256,
        "parallelism": dict(parallelism),
        "payload_inventory_sha256": canonical_json_sha256(inventory),
        "payload_file_count": inventory["file_count"],
        "payload_total_bytes": inventory["total_bytes"],
    }


def validate_hf_snapshot_binding(
    snapshot_manifest: Mapping[str, object],
    *,
    snapshot_manifest_sha256: str,
    expected_snapshot_manifest_sha256: str,
    model_id: str,
    revision: str,
) -> None:
    """Ensure the local HF reference used by the runner is the C0-bound source."""
    if snapshot_manifest_sha256 != expected_snapshot_manifest_sha256:
        raise C0ValidationError("HF snapshot manifest does not match the C0-bound snapshot digest")
    lifecycle = _mapping(snapshot_manifest.get("lifecycle"), "HF snapshot lifecycle")
    if lifecycle.get("status") != "materialized_pinned_snapshot":
        raise C0ValidationError("HF snapshot is not a materialized pinned snapshot")
    if (snapshot_manifest.get("model_id"), snapshot_manifest.get("revision")) != (model_id, revision):
        raise C0ValidationError("HF snapshot does not match the C0 model identity")


def verify_hf_snapshot_inventory(snapshot_dir: Path, manifest: Mapping[str, object]) -> dict[str, object]:
    """Hash every immutable HF snapshot payload file against its manifest.

    The manifest digest alone does not prove that the current source files have
    not changed after C0 import.  Split-reference parity therefore verifies the
    actual snapshot independently in both isolated reference processes.
    """
    expected_files = _validate_hf_snapshot_inventory(manifest)
    expected = {item["path"]: item for item in expected_files}
    manifest_name = HF_SNAPSHOT_MANIFEST_NAME
    actual: set[str] = set()
    for path in snapshot_dir.rglob("*"):
        relative_path = path.relative_to(snapshot_dir).as_posix()
        if relative_path == manifest_name:
            continue
        if path.is_symlink():
            raise C0ValidationError(f"HF snapshot payload entry must not be a symlink: {path}")
        if path.is_file():
            actual.add(relative_path)
    if actual != set(expected):
        missing = sorted(set(expected) - actual)
        unexpected = sorted(actual - set(expected))
        raise C0ValidationError(
            "HF snapshot file set differs from identity inventory: "
            f"missing={missing[:4]}, unexpected={unexpected[:4]}"
        )
    for relative_path, entry in expected.items():
        path = snapshot_dir / relative_path
        if path.is_symlink() or not path.is_file():
            raise C0ValidationError(f"HF snapshot payload entry is unavailable: {relative_path}")
        if path.stat().st_size != entry["bytes"]:
            raise C0ValidationError(f"HF snapshot payload size differs from inventory: {relative_path}")
        if sha256_file(path) != entry["sha256"]:
            raise C0ValidationError(f"HF snapshot payload SHA256 differs from inventory: {relative_path}")
    return {
        "file_count": len(expected_files),
        "total_bytes": sum(int(item["bytes"]) for item in expected_files),
        "inventory_sha256": canonical_json_sha256(expected_files),
    }


def verify_payload_inventory(checkpoint_dir: Path, manifest: Mapping[str, object]) -> dict[str, int]:
    """Verify the sealed C0 payload matches its one-time identity inventory.

    This intentionally hashes every payload file.  It is run once in the C0
    validation allocation, not in training and not against the immutable C0
    directory itself.
    """
    inventory = _validate_inventory(_mapping(manifest.get("payload_inventory"), "C0 payload inventory"))
    expected = {item["path"]: item for item in inventory["files"]}
    actual = {
        path.relative_to(checkpoint_dir).as_posix()
        for path in checkpoint_dir.rglob("*")
        if path.is_file() and path.name != IDENTITY_MANIFEST_NAME
    }
    if actual != set(expected):
        missing = sorted(set(expected) - actual)
        unexpected = sorted(actual - set(expected))
        raise C0ValidationError(
            "C0 payload file set differs from identity inventory "
            f"(missing={missing[:4]}, unexpected={unexpected[:4]})"
        )

    for relative_path, entry in expected.items():
        path = checkpoint_dir / relative_path
        if path.is_symlink():
            raise C0ValidationError(f"C0 payload inventory entry must not be a symlink: {relative_path}")
        if path.stat().st_size != entry["bytes"]:
            raise C0ValidationError(f"C0 payload size differs from identity inventory: {relative_path}")
        if sha256_file(path) != entry["sha256"]:
            raise C0ValidationError(f"C0 payload SHA256 differs from identity inventory: {relative_path}")
    return {"file_count": int(inventory["file_count"]), "total_bytes": int(inventory["total_bytes"])}


def stage_result(
    status: str,
    *,
    summary: str,
    measurements: Mapping[str, object] | None = None,
    errors: list[str] | None = None,
) -> dict[str, object]:
    """Build one bounded stage result with an explicit support state."""
    if status not in _STAGE_STATUSES:
        raise C0ValidationError(f"invalid validation stage status: {status}")
    if not summary:
        raise C0ValidationError("validation stage summary must be non-empty")
    return {
        "status": status,
        "supported": status != "unsupported",
        "summary": summary,
        "measurements": dict(measurements or {}),
        "errors": list(errors or []),
    }


def build_model_parity_record(
    *,
    identity: Mapping[str, object],
    c0_identity_manifest_sha256: str,
    stages: Mapping[str, Mapping[str, object]],
    measurement: Mapping[str, object],
    runtime_topology: Mapping[str, object] | None,
) -> dict[str, object]:
    """Build a non-eligible external C0 model-parity record.

    Passing this record proves only conversion/model parity under the recorded
    topology.  A later finalizer must add the normal-trainer optimizer sanity
    result and write ``validation.json`` before profiling can become eligible.
    """
    _sha256_text(c0_identity_manifest_sha256, "C0 identity manifest SHA256")
    if set(stages) != set(REQUIRED_STAGES):
        raise C0ValidationError(f"validation record must contain exactly these stages: {REQUIRED_STAGES}")
    normalized_stages: dict[str, dict[str, object]] = {}
    for name in REQUIRED_STAGES:
        value = _mapping(stages[name], f"{name} stage")
        status = value.get("status")
        if status not in _STAGE_STATUSES:
            raise C0ValidationError(f"{name} stage has an invalid status")
        normalized_stages[name] = dict(value)

    passed = all(stage["status"] == "passed" for stage in normalized_stages.values())
    normalized_topology = _validated_baseline_runtime_topology(runtime_topology)
    if passed and normalized_topology is None:
        raise C0ValidationError("passed model parity must include exact runtime topology evidence")
    return {
        "schema_version": MODEL_PARITY_SCHEMA_VERSION,
        "kind": _MODEL_PARITY_KIND,
        "hf_model": identity["hf_model"],
        "hf_revision": identity["hf_revision"],
        "c0_identity_manifest_sha256": c0_identity_manifest_sha256,
        "model_manifest_sha256": identity["model_manifest_sha256"],
        "c0_payload_inventory_sha256": identity["payload_inventory_sha256"],
        "parallelism": dict(_mapping(identity["parallelism"], "validated parallelism")),
        "runtime_topology": normalized_topology,
        "measurement": dict(measurement),
        "checks": normalized_stages,
        "lifecycle": {
            "status": "model_parity_passed" if passed else "blocked",
            "eligible_for_profile": False,
            "immutable_c0_was_not_modified": True,
            "final_validation_required": "normal_trainer_optimizer_sanity",
            "final_validation_record": "validation.json",
        },
    }


def _validated_baseline_runtime_topology(value: Mapping[str, object] | None) -> dict[str, int] | None:
    """Normalize one stable runtime-topology summary for downstream gate readers."""
    if value is None:
        return None
    topology = _mapping(value, "runtime topology")
    if set(topology) != set(BASELINE_RUNTIME_TOPOLOGY):
        raise C0ValidationError("runtime topology must contain exactly the fixed baseline fields")
    if any(
        not isinstance(actual, int) or isinstance(actual, bool) or actual != expected
        for field, expected in BASELINE_RUNTIME_TOPOLOGY.items()
        for actual in (topology[field],)
    ):
        raise C0ValidationError("runtime topology does not match the fixed 8-GPU AMPA baseline")
    return {field: int(topology[field]) for field in BASELINE_RUNTIME_TOPOLOGY}


def validate_final_profile_eligibility(
    final_record: Mapping[str, object],
    *,
    final_record_path: Path,
    c0_identity_manifest: Mapping[str, object],
    c0_identity_manifest_sha256: str,
    model_id: str,
    revision: str,
) -> None:
    """Fail closed unless the final record binds C0, parity, and Bridge sanity evidence.

    This is CPU-only and intentionally validates raw-file digests of both
    sibling evidence records. It does not claim to rehash the sealed C0
    payload during every profiling preflight; that expensive audit belongs to
    the dedicated C0 validation allocation.
    """
    _sha256_text(c0_identity_manifest_sha256, "C0 identity manifest SHA256")
    c0_model_manifest_sha256 = c0_identity_manifest.get("model_manifest_sha256")
    _sha256_text(c0_model_manifest_sha256, "C0 model manifest SHA256")
    validated_c0_identity = validate_c0_identity_manifest(
        c0_identity_manifest,
        model_manifest_sha256=c0_model_manifest_sha256,
        model_id=model_id,
        revision=revision,
    )
    if validated_c0_identity["parallelism"] != {"tp": 1, "pp": 1, "cp": 1, "ep": 8, "etp": 1}:
        raise C0ValidationError("C0 identity manifest parallelism is not the fixed AMPA baseline")
    _require_exact_keys(
        final_record,
        {
            "schema_version",
            "kind",
            "hf_model",
            "hf_revision",
            "c0_identity_manifest_sha256",
            "model_manifest_sha256",
            "c0_payload_inventory_sha256",
            "topology",
            "evidence",
            "lifecycle",
        },
        "final C0 eligibility record",
    )
    if final_record.get("schema_version") != FINAL_PROFILE_ELIGIBILITY_SCHEMA_VERSION:
        raise C0ValidationError("final C0 eligibility record has an unsupported schema version")
    if final_record.get("kind") != _FINAL_PROFILE_ELIGIBILITY_KIND:
        raise C0ValidationError("final C0 eligibility record has an unsupported kind")
    if (final_record.get("hf_model"), final_record.get("hf_revision")) != (model_id, revision):
        raise C0ValidationError("final C0 eligibility record model identity differs from the requested model")
    _require_exact_mapping(
        _mapping(final_record.get("lifecycle"), "final C0 eligibility lifecycle"),
        {"status": "eligible_for_profile", "eligible_for_profile": True},
        "final C0 eligibility lifecycle",
    )
    expected_identity = _c0_identity_binding(c0_identity_manifest, c0_identity_manifest_sha256)
    for field, expected in expected_identity.items():
        if final_record.get(field) != expected:
            raise C0ValidationError(f"final C0 eligibility record {field} does not bind the imported C0")
    if _validated_baseline_runtime_topology(_mapping(final_record.get("topology"), "final topology")) != BASELINE_RUNTIME_TOPOLOGY:
        raise C0ValidationError("final C0 eligibility record topology is not the fixed AMPA baseline")

    evidence = _mapping(final_record.get("evidence"), "final C0 eligibility evidence")
    _require_exact_keys(evidence, {"model_parity", "optimizer_sanity"}, "final C0 eligibility evidence")
    parity_path, parity_sha256 = _verified_sibling_evidence(
        final_record_path, evidence.get("model_parity"), "model_parity.json"
    )
    sanity_path, sanity_sha256 = _verified_sibling_evidence(
        final_record_path, evidence.get("optimizer_sanity"), "optimizer_sanity.json"
    )
    parity = _load_regular_json(parity_path, "model parity evidence")
    sanity = _load_regular_json(sanity_path, "optimizer sanity evidence")
    _validate_model_parity_evidence(parity, expected_identity, model_id, revision)
    _validate_optimizer_sanity_evidence(sanity, expected_identity, parity_sha256, model_id, revision)
    if sha256_file(parity_path) != parity_sha256 or sha256_file(sanity_path) != sanity_sha256:
        raise C0ValidationError("C0 eligibility evidence changed while it was being validated")


def _c0_identity_binding(
    manifest: Mapping[str, object], c0_identity_manifest_sha256: str
) -> dict[str, str]:
    """Extract the C0 fields that every external record must bind exactly."""
    lifecycle = _mapping(manifest.get("lifecycle"), "C0 lifecycle")
    if lifecycle.get("status") != "imported" or lifecycle.get("immutable_after_write") is not True:
        raise C0ValidationError("C0 identity manifest is not an immutable imported checkpoint")
    model_manifest_sha256 = manifest.get("model_manifest_sha256")
    _sha256_text(model_manifest_sha256, "C0 model manifest SHA256")
    inventory = _validate_inventory(_mapping(manifest.get("payload_inventory"), "C0 payload inventory"))
    return {
        "c0_identity_manifest_sha256": c0_identity_manifest_sha256,
        "model_manifest_sha256": model_manifest_sha256,
        "c0_payload_inventory_sha256": canonical_json_sha256(inventory),
    }


def _verified_sibling_evidence(
    final_record_path: Path, value: object, expected_name: str
) -> tuple[Path, str]:
    """Return one regular sibling record after exact path and raw-digest checks."""
    evidence = _mapping(value, f"{expected_name} evidence")
    _require_exact_keys(evidence, {"path", "sha256"}, f"{expected_name} evidence")
    if evidence.get("path") != expected_name:
        raise C0ValidationError(f"{expected_name} evidence must use its canonical sibling path")
    digest = evidence.get("sha256")
    _sha256_text(digest, f"{expected_name} evidence SHA256")
    path = final_record_path.parent / expected_name
    if path.is_symlink() or not path.is_file():
        raise C0ValidationError(f"{expected_name} evidence must be a regular sibling file")
    if sha256_file(path) != digest:
        raise C0ValidationError(f"{expected_name} evidence SHA256 does not match")
    return path, digest


def _validate_model_parity_evidence(
    record: Mapping[str, object], expected_identity: Mapping[str, str], model_id: str, revision: str
) -> None:
    _require_exact_keys(
        record,
        {
            "schema_version",
            "kind",
            "hf_model",
            "hf_revision",
            "c0_identity_manifest_sha256",
            "model_manifest_sha256",
            "c0_payload_inventory_sha256",
            "parallelism",
            "runtime_topology",
            "measurement",
            "checks",
            "lifecycle",
        },
        "model parity evidence",
    )
    if record.get("schema_version") != MODEL_PARITY_SCHEMA_VERSION or record.get("kind") != _MODEL_PARITY_KIND:
        raise C0ValidationError("model parity evidence has an unsupported schema or kind")
    if (record.get("hf_model"), record.get("hf_revision")) != (model_id, revision):
        raise C0ValidationError("model parity evidence model identity differs from the requested model")
    for field, expected in expected_identity.items():
        if record.get(field) != expected:
            raise C0ValidationError(f"model parity evidence {field} does not bind the imported C0")
    _require_exact_mapping(
        _mapping(record.get("parallelism"), "model parity parallelism"),
        {"tp": 1, "pp": 1, "cp": 1, "ep": 8, "etp": 1},
        "model parity parallelism",
    )
    if _validated_baseline_runtime_topology(_mapping(record.get("runtime_topology"), "model parity topology")) != BASELINE_RUNTIME_TOPOLOGY:
        raise C0ValidationError("model parity evidence topology is not the fixed AMPA baseline")
    if not isinstance(record.get("measurement"), Mapping):
        raise C0ValidationError("model parity evidence has no measurement record")
    checks = _mapping(record.get("checks"), "model parity checks")
    if set(checks) != set(REQUIRED_STAGES):
        raise C0ValidationError("model parity evidence does not contain exactly the required stages")
    for name in REQUIRED_STAGES:
        stage = _mapping(checks[name], f"model parity {name} stage")
        if stage.get("status") != "passed" or stage.get("supported") is not True:
            raise C0ValidationError(f"model parity {name} stage did not pass")
    _require_exact_mapping(
        _mapping(record.get("lifecycle"), "model parity lifecycle"),
        {
            "status": "model_parity_passed",
            "eligible_for_profile": False,
            "immutable_c0_was_not_modified": True,
            "final_validation_required": "normal_trainer_optimizer_sanity",
            "final_validation_record": "validation.json",
        },
        "model parity lifecycle",
    )


def _validate_optimizer_sanity_evidence(
    record: Mapping[str, object],
    expected_identity: Mapping[str, str],
    model_parity_sha256: str,
    model_id: str,
    revision: str,
) -> None:
    _require_exact_keys(
        record,
        {
            "schema_version",
            "kind",
            "hf_model",
            "hf_revision",
            "c0_identity_manifest_sha256",
            "model_manifest_sha256",
            "c0_payload_inventory_sha256",
            "model_parity_sha256",
            "topology",
            "trainer",
            "checks",
            "lifecycle",
        },
        "optimizer sanity evidence",
    )
    if record.get("schema_version") != OPTIMIZER_SANITY_SCHEMA_VERSION or record.get("kind") != _OPTIMIZER_SANITY_KIND:
        raise C0ValidationError("optimizer sanity evidence has an unsupported schema or kind")
    if (record.get("hf_model"), record.get("hf_revision")) != (model_id, revision):
        raise C0ValidationError("optimizer sanity evidence model identity differs from the requested model")
    for field, expected in expected_identity.items():
        if record.get(field) != expected:
            raise C0ValidationError(f"optimizer sanity evidence {field} does not bind the imported C0")
    if record.get("model_parity_sha256") != model_parity_sha256:
        raise C0ValidationError("optimizer sanity evidence does not bind the accepted model parity record")
    if _validated_baseline_runtime_topology(_mapping(record.get("topology"), "optimizer sanity topology")) != BASELINE_RUNTIME_TOPOLOGY:
        raise C0ValidationError("optimizer sanity evidence topology is not the fixed AMPA baseline")
    _require_exact_mapping(
        _mapping(record.get("trainer"), "optimizer sanity trainer"),
        {"kind": "bridge_pretrain", "version": 1},
        "optimizer sanity trainer",
    )
    _require_exact_mapping(
        _mapping(record.get("checks"), "optimizer sanity checks"),
        {
            "complete_optimizer_steps": 1,
            "skipped_or_failed_optimizer_steps": 0,
            "finite_loss": True,
            "finite_grad_norm": True,
            "parameter_update_observed": True,
            "finite_router_gradients": True,
            "finite_expert_gradients": True,
            "finite_dense_gradients": True,
        },
        "optimizer sanity checks",
        allow_minimum={"complete_optimizer_steps"},
    )
    _require_exact_mapping(
        _mapping(record.get("lifecycle"), "optimizer sanity lifecycle"),
        {"status": "optimizer_sanity_passed", "eligible_for_profile": False},
        "optimizer sanity lifecycle",
    )


def _load_regular_json(path: Path, label: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise C0ValidationError(f"{label} must be a regular file")
    return load_json_object(path, label=label)


def _require_exact_keys(value: Mapping[str, object], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise C0ValidationError(f"{label} has an unexpected field set")


def _require_exact_mapping(
    value: Mapping[str, object], expected: Mapping[str, object], label: str, *, allow_minimum: set[str] | None = None
) -> None:
    _require_exact_keys(value, set(expected), label)
    for field, expected_value in expected.items():
        actual = value.get(field)
        if allow_minimum and field in allow_minimum:
            if not isinstance(actual, int) or isinstance(actual, bool) or actual < expected_value:
                raise C0ValidationError(f"{label} {field} is below the required minimum")
        elif actual != expected_value:
            raise C0ValidationError(f"{label} {field} differs from the required value")


def write_json_once(path: Path, value: Mapping[str, object]) -> None:
    """Publish a new external JSON record without ever replacing a prior result."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise C0ValidationError(f"refusing to overwrite existing validation record: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise C0ValidationError(f"refusing to overwrite existing validation record: {path}") from error
    finally:
        if temporary.exists():
            temporary.unlink()


def _validate_inventory(inventory: Mapping[str, object]) -> dict[str, object]:
    files = inventory.get("files")
    if not isinstance(files, list) or not files:
        raise C0ValidationError("C0 payload inventory must contain at least one file")
    normalized: list[dict[str, object]] = []
    paths: set[str] = set()
    for item in files:
        entry = _mapping(item, "C0 payload inventory entry")
        relative_path = entry.get("path")
        if not isinstance(relative_path, str) or not _safe_relative_path(relative_path):
            raise C0ValidationError("C0 payload inventory contains an unsafe relative path")
        if relative_path in paths:
            raise C0ValidationError(f"C0 payload inventory contains duplicate path: {relative_path}")
        paths.add(relative_path)
        byte_count = entry.get("bytes")
        if not isinstance(byte_count, int) or isinstance(byte_count, bool) or byte_count < 0:
            raise C0ValidationError(f"C0 payload inventory has invalid byte count: {relative_path}")
        digest = entry.get("sha256")
        _sha256_text(digest, f"C0 payload SHA256 for {relative_path}")
        normalized.append({"path": relative_path, "bytes": byte_count, "sha256": digest})
    if inventory.get("file_count") != len(normalized):
        raise C0ValidationError("C0 payload inventory file_count does not match files")
    total_bytes = sum(int(item["bytes"]) for item in normalized)
    if inventory.get("total_bytes") != total_bytes:
        raise C0ValidationError("C0 payload inventory total_bytes does not match files")
    return {"files": normalized, "file_count": len(normalized), "total_bytes": total_bytes}


def _validate_hf_snapshot_inventory(manifest: Mapping[str, object]) -> list[dict[str, object]]:
    """Validate the snapshot manifest's flat immutable-file inventory."""
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise C0ValidationError("HF snapshot manifest files must be a non-empty list")
    normalized: list[dict[str, object]] = []
    paths: set[str] = set()
    for index, raw_entry in enumerate(files):
        entry = _mapping(raw_entry, f"HF snapshot manifest files[{index}]")
        if set(entry) != {"path", "bytes", "sha256"}:
            raise C0ValidationError(f"HF snapshot manifest files[{index}] has an invalid field set")
        path = entry.get("path")
        byte_count = entry.get("bytes")
        digest = entry.get("sha256")
        if not isinstance(path, str) or not _safe_relative_path(path):
            raise C0ValidationError(f"HF snapshot manifest files[{index}] path is unsafe")
        if path == HF_SNAPSHOT_MANIFEST_NAME:
            raise C0ValidationError("HF snapshot manifest must not inventory itself")
        if not isinstance(byte_count, int) or isinstance(byte_count, bool) or byte_count < 0:
            raise C0ValidationError(f"HF snapshot manifest files[{index}] bytes is invalid")
        _sha256_text(digest, f"HF snapshot manifest files[{index}] SHA256")
        if path in paths:
            raise C0ValidationError(f"HF snapshot manifest inventories duplicate file: {path}")
        paths.add(path)
        normalized.append({"path": path, "bytes": byte_count, "sha256": digest})
    if manifest.get("file_count") != len(normalized):
        raise C0ValidationError("HF snapshot manifest file_count does not match files")
    total_bytes = sum(int(item["bytes"]) for item in normalized)
    if manifest.get("total_bytes") != total_bytes:
        raise C0ValidationError("HF snapshot manifest total_bytes does not match files")
    return normalized


def _safe_relative_path(value: str) -> bool:
    path = PurePosixPath(value)
    return bool(value) and not path.is_absolute() and ".." not in path.parts and path.as_posix() == value


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise C0ValidationError(f"{label} must be an object")
    return value


def _sha256_text(value: object, label: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise C0ValidationError(f"{label} must be a lowercase SHA256 digest")


def _git_revision(value: object, label: str) -> None:
    if not isinstance(value, str) or len(value) != 40 or any(char not in "0123456789abcdef" for char in value):
        raise C0ValidationError(f"{label} must be a lowercase 40-character git revision")
