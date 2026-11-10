#!/usr/bin/env python3
"""Offline contracts for AMPA's normal-GPT dataset cache.

This module does not create cache files or advance a dataset manifest. Its
inventory validator reads indexed headers with the installed Megatron Core
reader; run that validator on an allocated compute node. The checks bind a
cache to immutable indexed components and reject a cache that could cause a
timed training job to silently build or use a different index.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping


CACHE_CONTRACT_SCHEMA_VERSION = 1
_EXPECTED_SOURCES = (
    ("web", Decimal("0.55")),
    ("multilingual", Decimal("0.15")),
    ("code", Decimal("0.15")),
    ("math", Decimal("0.15")),
)
_WORKLOAD = {
    "micro_batch_size": 1,
    "dense_data_parallel_size": 8,
    "gradient_accumulation": 8,
    "global_batch_size": 64,
    "sequence_length": 4096,
    "warmup_steps": 5,
    "measured_steps": 20,
    "sampler_seed": 1234,
}
_CP2_VARIANT = "cp2_ep8"
_CP2_WORKLOAD = {**_WORKLOAD, "dense_data_parallel_size": 4, "gradient_accumulation": 16}
_GPT_CACHE = re.compile(
    r"^(?P<hash>[0-9a-f]{32})-GPTDataset-train-"
    r"(?P<suffix>description\.txt|document_index\.npy|sample_index\.npy|shuffle_index\.npy)$"
)
_BLENDED_CACHE = re.compile(
    r"^(?P<hash>[0-9a-f]{32})-BlendedDataset-train-"
    r"(?P<suffix>description\.txt|dataset_index\.npy|dataset_sample_index\.npy)$"
)


class CacheContractError(ValueError):
    """Raised when an indexed input or its offline cache is not canonical."""


def sha256_file(path: Path) -> str:
    """Hash one regular file without loading its full payload into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: object) -> str:
    """Hash a JSON value deterministically for compact contract binding."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_cache_contract_from_manifest(
    indexed_manifest_path: Path, *, workspace_root: Path, variant: str | None = None
) -> dict[str, object]:
    """Load an indexed-validation candidate and bind its exact file bytes.

    ``indexed_manifest_path`` must be the independent indexed-data validation
    result, before cache preparation.  A future cache-preparation job can copy
    the returned object into its separate candidate record; this helper never
    writes or changes the source manifest.
    """
    try:
        raw = indexed_manifest_path.read_bytes()
        manifest = json.loads(raw)
    except FileNotFoundError as error:
        raise CacheContractError(f"indexed validation manifest is missing: {indexed_manifest_path}") from error
    except json.JSONDecodeError as error:
        raise CacheContractError(f"indexed validation manifest is not valid JSON: {error}") from error
    if not isinstance(manifest, dict):
        raise CacheContractError("indexed validation manifest must contain a JSON object")
    return build_cache_contract(
        manifest,
        workspace_root=workspace_root,
        indexed_manifest_path=indexed_manifest_path,
        indexed_manifest_sha256=hashlib.sha256(raw).hexdigest(),
        variant=variant,
    )


def build_cache_contract(
    indexed_manifest: Mapping[str, object],
    *,
    workspace_root: Path,
    indexed_manifest_path: Path | None = None,
    indexed_manifest_sha256: str | None = None,
    variant: str | None = None,
) -> dict[str, object]:
    """Return the CP1 baseline, or the opt-in CP2/EP8 normal-GPT contract.

    The mapping form exists for CPU tests.  Production preparation must use
    :func:`build_cache_contract_from_manifest` so the input manifest's file
    hash is explicitly retained.
    """
    if variant not in (None, _CP2_VARIANT):
        raise CacheContractError(f"unsupported cache contract variant: {variant!r}")
    workload = _CP2_WORKLOAD if variant == _CP2_VARIANT else _WORKLOAD
    workspace = _workspace_root(workspace_root)
    lifecycle = _mapping(indexed_manifest.get("lifecycle"), "indexed manifest lifecycle")
    if lifecycle.get("status") != "indexed_validated_pending_cache":
        raise CacheContractError("cache preparation requires indexed_validated_pending_cache input")
    if lifecycle.get("canonical_training_allowed") is not False:
        raise CacheContractError("indexed cache input must still forbid canonical training")

    manifest_id = _manifest_id(indexed_manifest.get("manifest_id"))
    sources = indexed_manifest.get("sources")
    artifacts = indexed_manifest.get("materialized_artifacts")
    if not isinstance(sources, list) or not isinstance(artifacts, list):
        raise CacheContractError("indexed cache input must contain sources and materialized_artifacts lists")
    if len(sources) != len(_EXPECTED_SOURCES) or len(artifacts) != len(_EXPECTED_SOURCES):
        raise CacheContractError("AMPA cache preparation requires exactly four indexed source artifacts")

    artifacts_by_name: dict[str, Mapping[str, object]] = {}
    for artifact in artifacts:
        if not isinstance(artifact, Mapping) or not isinstance(artifact.get("name"), str):
            raise CacheContractError("materialized artifact entries require unique string names")
        if artifact["name"] in artifacts_by_name:
            raise CacheContractError("materialized artifact names must be unique")
        artifacts_by_name[artifact["name"]] = artifact

    entries: list[dict[str, object]] = []
    flat_tokens: list[str] = []
    expected_indexed_root = workspace / "dataset" / "indexed" / manifest_id
    for source, (expected_name, expected_weight) in zip(sources, _EXPECTED_SOURCES):
        if not isinstance(source, Mapping):
            raise CacheContractError("source entries must be objects")
        if source.get("name") != expected_name:
            raise CacheContractError("source order must be web, multilingual, code, math")
        weight = _decimal(source.get("weight"), f"{expected_name}.weight")
        if weight != expected_weight:
            raise CacheContractError(f"{expected_name} weight differs from the fixed realistic mixture")
        artifact = artifacts_by_name.get(expected_name)
        if artifact is None:
            raise CacheContractError(f"indexed artifact is missing for {expected_name}")
        expected_prefix = _relative_workspace_path(
            workspace, source.get("indexed_prefix"), f"{expected_name}.indexed_prefix"
        )
        _require_under(expected_prefix, expected_indexed_root, f"{expected_name}.indexed_prefix")
        _require_no_symlink_ancestors(expected_prefix, workspace, f"{expected_name}.indexed_prefix")
        entry = _verify_source_artifact(artifact, expected_name, expected_prefix, workspace)
        entry["weight"] = _decimal_text(weight)
        entries.append(entry)
        flat_tokens.extend((entry["weight"], entry["indexed_prefix"]))

    if set(artifacts_by_name) != {name for name, _ in _EXPECTED_SOURCES}:
        raise CacheContractError("materialized artifacts differ from the fixed four-source mixture")
    flat_data_path = " ".join(flat_tokens)
    if indexed_manifest.get("materialized_training_data_path") != flat_data_path:
        raise CacheContractError("indexed manifest flat data-path differs from its pinned source artifacts")

    indexed_binding: dict[str, object] = {
        "content_sha256": canonical_json_sha256(dict(indexed_manifest)),
    }
    if indexed_manifest_path is not None or indexed_manifest_sha256 is not None:
        if indexed_manifest_path is None or indexed_manifest_sha256 is None:
            raise CacheContractError("indexed manifest path and SHA256 must be supplied together")
        manifest_path = _regular_path(indexed_manifest_path, "indexed validation manifest")
        _require_under(manifest_path, workspace / "dataset" / "manifests", "indexed validation manifest")
        indexed_binding.update({"path": str(manifest_path), "sha256": _sha256_text(indexed_manifest_sha256)})

    train_tokens = [token for entry in entries for token in (entry["weight"], entry["indexed_prefix"])]
    contract = {
        "schema_version": CACHE_CONTRACT_SCHEMA_VERSION,
        "kind": "ampa_megatron_gpt_cache",
        "indexed_manifest": indexed_binding,
        "manifest_id": manifest_id,
        "workload_identity": dict(workload),
        "prepare_cache": {
            "train_iters": workload["warmup_steps"] + workload["measured_steps"],
            "train_samples": workload["global_batch_size"]
            * (workload["warmup_steps"] + workload["measured_steps"]),
            "eval_iters": 0,
            "eval_interval": 1,
            "prepare_cache_world_size": 8,
            "parallelism": {"tp": 1, "pp": 1, "cp": 2 if variant == _CP2_VARIANT else 1, "ep": 8, "etp": 1},
            "reset_attention_mask": False,
            "reset_position_ids": False,
            "fast_cache_load_during_preparation": False,
            "defer_npy_index_mmap_during_preparation": False,
        },
        "data": {
            "flat_data_path": flat_data_path,
            "per_split_data_args": {"train": " ".join(train_tokens), "valid": None, "test": None},
            "blend_per_split": {
                "train": {
                    "prefixes": [entry["indexed_prefix"] for entry in entries],
                    "weights": [entry["weight"] for entry in entries],
                },
                "valid": None,
                "test": None,
            },
            "source_artifacts": entries,
        },
    }
    if variant == _CP2_VARIANT:
        contract["variant_id"] = _CP2_VARIANT
    return contract


def cache_contract_sha256(contract: Mapping[str, object]) -> str:
    """Return the stable cache identity after validating the contract schema."""
    _validate_contract_shape(contract)
    return canonical_json_sha256(dict(contract))


def cache_directory_for_contract(workspace_root: Path, contract: Mapping[str, object]) -> Path:
    """Return the sole immutable cache destination for a cache contract."""
    workspace = _workspace_root(workspace_root)
    _validate_contract_shape(contract)
    return workspace / "dataset" / "metadata" / "cache" / str(contract["manifest_id"]) / cache_contract_sha256(contract)


def verify_source_inventory(contract: Mapping[str, object]) -> list[dict[str, object]]:
    """Re-hash all four input components and reject post-contract mutation."""
    _validate_contract_shape(contract)
    sources = _mapping(contract.get("data"), "cache contract data").get("source_artifacts")
    if not isinstance(sources, list) or len(sources) != len(_EXPECTED_SOURCES):
        raise CacheContractError("cache contract source artifact inventory is invalid")
    observed: list[dict[str, object]] = []
    for source, (expected_name, _) in zip(sources, _EXPECTED_SOURCES):
        if not isinstance(source, Mapping) or source.get("name") != expected_name:
            raise CacheContractError("cache contract source order is invalid")
        for suffix in ("idx", "bin"):
            path = _regular_path(Path(str(source.get(f"{suffix}_path", ""))), f"{expected_name}.{suffix}")
            expected_size = source.get(f"{suffix}_bytes")
            expected_hash = source.get(f"{suffix}_sha256")
            if not isinstance(expected_size, int) or isinstance(expected_size, bool) or expected_size <= 0:
                raise CacheContractError(f"{expected_name}.{suffix} has an invalid expected byte size")
            if path.stat().st_size != expected_size or sha256_file(path) != _sha256_text(expected_hash):
                raise CacheContractError(f"indexed source artifact changed after cache contract creation: {path}")
        observed.append(
            {
                "name": expected_name,
                "idx_sha256": source["idx_sha256"],
                "bin_sha256": source["bin_sha256"],
                "idx_bytes": source["idx_bytes"],
                "bin_bytes": source["bin_bytes"],
            }
        )
    return observed


def validate_cache_inventory(
    contract: Mapping[str, object], *, workspace_root: Path, cache_dir: Path
) -> dict[str, object]:
    """Validate the sealed MCore GPT/Blended cache layout and hashes.

    This checks the current MCore cache layout for four train GPT datasets plus
    one train blended dataset.  It intentionally does *not* claim that a
    dataloader has consumed the cache; that remains an allocated Slurm probe.
    """
    workspace = _workspace_root(workspace_root)
    expected_dir = cache_directory_for_contract(workspace, contract)
    _require_no_symlink_ancestors(expected_dir, workspace, "cache destination")
    actual_dir = _regular_directory(cache_dir, "cache directory")
    if actual_dir != expected_dir.resolve():
        raise CacheContractError("cache directory does not equal this contract's confined immutable destination")
    _require_under(actual_dir, workspace, "cache directory")
    source_inventory = verify_source_inventory(contract)
    files = _cache_regular_files(actual_dir)
    sequence_path = actual_dir / "per_dataset_sequences.json"
    if sequence_path not in files:
        raise CacheContractError("cache is missing per_dataset_sequences.json")
    sequences = _validate_sequences(sequence_path, contract)
    _validate_sequences_against_indexed_components(sequences, contract)

    gpt_groups: dict[str, dict[str, Path]] = {}
    blended_groups: dict[str, dict[str, Path]] = {}
    for path in files:
        if path == sequence_path:
            continue
        relative = path.relative_to(actual_dir).as_posix()
        match = _GPT_CACHE.fullmatch(relative)
        target = gpt_groups
        if match is None:
            match = _BLENDED_CACHE.fullmatch(relative)
            target = blended_groups
        if match is None:
            raise CacheContractError(f"unexpected cache artifact: {relative}")
        target.setdefault(match["hash"], {})[match["suffix"]] = path

    expected_gpt_suffixes = {"description.txt", "document_index.npy", "sample_index.npy", "shuffle_index.npy"}
    if len(gpt_groups) != len(_EXPECTED_SOURCES) or any(set(group) != expected_gpt_suffixes for group in gpt_groups.values()):
        raise CacheContractError("cache does not contain exactly four complete GPTDataset train bundles")
    expected_blended_suffixes = {"description.txt", "dataset_index.npy", "dataset_sample_index.npy"}
    if len(blended_groups) != 1 or any(
        set(group) != expected_blended_suffixes for group in blended_groups.values()
    ):
        raise CacheContractError("cache does not contain exactly one complete BlendedDataset train bundle")

    expected_prefixes = [entry["indexed_prefix"] for entry in _source_entries(contract)]
    seen_prefixes: list[str] = []
    for digest, group in gpt_groups.items():
        description = _description(group["description.txt"], digest, "GPTDataset")
        if description.get("index_split") != "train" or description.get("random_seed") != _WORKLOAD["sampler_seed"]:
            raise CacheContractError("GPTDataset cache description has the wrong train split or sampler seed")
        if description.get("sequence_length") != _WORKLOAD["sequence_length"]:
            raise CacheContractError("GPTDataset cache description has the wrong sequence length")
        prefix = description.get("dataset_path")
        if prefix not in expected_prefixes or prefix in seen_prefixes:
            raise CacheContractError("GPTDataset cache descriptions do not bind exactly the four source prefixes")
        seen_prefixes.append(prefix)

    blended_digest, blended_group = next(iter(blended_groups.items()))
    blended = _description(blended_group["description.txt"], blended_digest, "BlendedDataset")
    if blended.get("split") != "train" or blended.get("size") != expected_blended_train_size(contract):
        raise CacheContractError("BlendedDataset cache description has the wrong split or train sample count")
    weights = blended.get("weights")
    if not isinstance(weights, list) or [_decimal_text(_decimal(value, "cache blend weight")) for value in weights] != [
        item["weight"] for item in _source_entries(contract)
    ]:
        raise CacheContractError("BlendedDataset cache weights differ from the pinned four-source blend")
    datasets = blended.get("datasets")
    if not isinstance(datasets, list) or [item.get("dataset_path") if isinstance(item, dict) else None for item in datasets] != expected_prefixes:
        raise CacheContractError("BlendedDataset cache source order differs from the pinned four-source blend")

    inventory = [
        {
            "path": str(path),
            "relative_path": path.relative_to(actual_dir).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in files
    ]
    return {
        "cache_path": str(actual_dir),
        "cache_contract_sha256": cache_contract_sha256(contract),
        "source_inventory": source_inventory,
        "per_dataset_sequences_path": str(sequence_path),
        "per_dataset_sequences_sha256": sha256_file(sequence_path),
        "per_dataset_sequences": sequences,
        "artifacts": inventory,
        "artifact_inventory_sha256": canonical_json_sha256(inventory),
    }


def _workspace_root(path: Path) -> Path:
    root = Path(path)
    if root.is_symlink() or not root.is_dir():
        raise CacheContractError(f"workspace root must be an existing non-symlink directory: {root}")
    return root.resolve()


def _manifest_id(value: object) -> str:
    if not isinstance(value, str) or not value or value in {".", ".."} or Path(value).name != value:
        raise CacheContractError("manifest_id must be a non-path non-empty string")
    return value


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise CacheContractError(f"{label} must be an object")
    return value


def _decimal(value: object, label: str) -> Decimal:
    if isinstance(value, bool):
        raise CacheContractError(f"{label} must be a finite decimal")
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise CacheContractError(f"{label} must be a finite decimal") from error
    if not parsed.is_finite() or parsed <= 0:
        raise CacheContractError(f"{label} must be a positive finite decimal")
    return parsed


def _decimal_text(value: Decimal) -> str:
    return format(value.normalize(), "f")


def _sha256_text(value: object) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise CacheContractError("SHA256 identity must be 64 lowercase hexadecimal characters")
    return value


def _relative_workspace_path(workspace: Path, value: object, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise CacheContractError(f"{label} must be a workspace-relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise CacheContractError(f"{label} must be a confined workspace-relative path")
    return (workspace / path).resolve()


def _require_under(path: Path, root: Path, label: str) -> None:
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise CacheContractError(f"{label} escapes its required root: {path}") from error


def _require_no_symlink_ancestors(path: Path, root: Path, label: str) -> None:
    """Reject an existing symlink anywhere below the trusted workspace root."""
    try:
        relative = path.absolute().relative_to(root.absolute())
    except ValueError as error:
        raise CacheContractError(f"{label} is outside the trusted workspace root") from error
    current = root
    for part in relative.parts:
        current /= part
        if current.exists() and current.is_symlink():
            raise CacheContractError(f"{label} uses a symlinked path component: {current}")


def _regular_path(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise CacheContractError(f"{label} must be an existing non-symlink regular file: {path}")
    return path.resolve()


def _regular_directory(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_dir():
        raise CacheContractError(f"{label} must be an existing non-symlink directory: {path}")
    return path.resolve()


def _verify_source_artifact(
    artifact: Mapping[str, object], name: str, prefix: Path, workspace: Path
) -> dict[str, object]:
    if artifact.get("indexed_prefix") != str(prefix):
        raise CacheContractError(f"{name} artifact indexed_prefix differs from the immutable source prefix")
    result: dict[str, object] = {"name": name, "indexed_prefix": str(prefix)}
    for suffix in ("idx", "bin"):
        path = _regular_path(Path(str(artifact.get(f"{suffix}_path", ""))), f"{name}.{suffix}")
        expected_path = Path(f"{prefix}.{suffix}").resolve()
        if path != expected_path:
            raise CacheContractError(f"{name}.{suffix}_path differs from its indexed prefix")
        _require_under(path, workspace / "dataset" / "indexed", f"{name}.{suffix}_path")
        recorded_bytes = artifact.get(f"{suffix}_bytes")
        recorded_hash = _sha256_text(artifact.get(f"{suffix}_sha256"))
        if not isinstance(recorded_bytes, int) or isinstance(recorded_bytes, bool) or recorded_bytes <= 0:
            raise CacheContractError(f"{name}.{suffix}_bytes must be a positive integer")
        if path.stat().st_size != recorded_bytes or sha256_file(path) != recorded_hash:
            raise CacheContractError(f"{name} indexed {suffix} artifact does not match recorded bytes/SHA256")
        result[f"{suffix}_path"] = str(path)
        result[f"{suffix}_bytes"] = recorded_bytes
        result[f"{suffix}_sha256"] = recorded_hash
    return result


def _validate_contract_shape(contract: Mapping[str, object]) -> None:
    if contract.get("schema_version") != CACHE_CONTRACT_SCHEMA_VERSION or contract.get("kind") != "ampa_megatron_gpt_cache":
        raise CacheContractError("cache contract schema or kind is invalid")
    _manifest_id(contract.get("manifest_id"))
    variant = contract.get("variant_id") if "variant_id" in contract else None
    if variant not in (None, _CP2_VARIANT) or ("variant_id" in contract and variant is None):
        raise CacheContractError("cache contract variant is unsupported")
    workload = _CP2_WORKLOAD if variant == _CP2_VARIANT else _WORKLOAD
    if contract.get("workload_identity") != workload:
        raise CacheContractError("cache contract workload differs from its declared variant")
    prepare = _mapping(contract.get("prepare_cache"), "cache contract preparation")
    expected_prepare = {
        "train_iters": 25,
        "train_samples": 1600,
        "eval_iters": 0,
        "eval_interval": 1,
        "prepare_cache_world_size": 8,
        "parallelism": {"tp": 1, "pp": 1, "cp": 2 if variant == _CP2_VARIANT else 1, "ep": 8, "etp": 1},
        "reset_attention_mask": False,
        "reset_position_ids": False,
        "fast_cache_load_during_preparation": False,
        "defer_npy_index_mmap_during_preparation": False,
    }
    if dict(prepare) != expected_prepare:
        raise CacheContractError("cache contract preparation fields differ from the fixed baseline")
    data = _mapping(contract.get("data"), "cache contract data")
    per_split = _mapping(data.get("per_split_data_args"), "cache contract per_split_data_args")
    if set(per_split) != {"train", "valid", "test"} or per_split.get("valid") is not None or per_split.get("test") is not None:
        raise CacheContractError("cache contract must use a train-only per-split blend")
    if not isinstance(per_split.get("train"), str) or not per_split["train"]:
        raise CacheContractError("cache contract train blend is missing")
    blend_per_split = _mapping(data.get("blend_per_split"), "cache contract blend_per_split")
    expected_entries = _source_entries(contract)
    expected_prefixes = [entry.get("indexed_prefix") for entry in expected_entries]
    expected_weights = [entry.get("weight") for entry in expected_entries]
    if blend_per_split != {
        "train": {"prefixes": expected_prefixes, "weights": expected_weights},
        "valid": None,
        "test": None,
    }:
        raise CacheContractError("cache contract normalized per-split blend is invalid")
    expected_train = " ".join(
        token for entry in expected_entries for token in (str(entry.get("weight")), str(entry.get("indexed_prefix")))
    )
    if per_split["train"] != expected_train or data.get("flat_data_path") != expected_train:
        raise CacheContractError("cache contract flat and per-split train blends disagree with source artifacts")


def _source_entries(contract: Mapping[str, object]) -> list[Mapping[str, object]]:
    sources = _mapping(contract.get("data"), "cache contract data").get("source_artifacts")
    if not isinstance(sources, list):
        raise CacheContractError("cache contract source artifacts are invalid")
    return [_mapping(source, "cache contract source artifact") for source in sources]


def expected_blended_train_size(contract: Mapping[str, object]) -> int:
    """Mirror pinned MCore's per-source ceil before summing the train blend."""
    prepare = _mapping(contract.get("prepare_cache"), "cache contract prepare_cache")
    target = prepare.get("train_samples")
    if type(target) is not int or target <= 0:
        raise CacheContractError("cache contract train_samples must be a positive integer")
    weights = [float(source["weight"]) for source in _source_entries(contract)]
    total = sum(weights)
    if not math.isfinite(total) or total <= 0 or any(not math.isfinite(weight) or weight <= 0 for weight in weights):
        raise CacheContractError("cache contract train weights must be positive finite numbers")
    return sum(math.ceil(target * (weight / total)) for weight in weights)


def _cache_regular_files(cache_dir: Path) -> list[Path]:
    files: list[Path] = []
    for path in sorted(cache_dir.rglob("*")):
        relative = path.relative_to(cache_dir)
        if any(part.startswith(".partial-") for part in relative.parts) or any(part.endswith(".partial") for part in relative.parts):
            raise CacheContractError(f"cache contains an unsealed partial artifact: {relative}")
        if path.is_symlink():
            raise CacheContractError(f"cache must not contain symlinks: {relative}")
        if path.is_file():
            if path.stat().st_size <= 0:
                raise CacheContractError(f"cache contains an empty artifact: {relative}")
            files.append(path)
        elif not path.is_dir():
            raise CacheContractError(f"cache contains a non-regular artifact: {relative}")
    if not files:
        raise CacheContractError("cache contains no files")
    return files


def _validate_sequences(path: Path, contract: Mapping[str, object]) -> dict[str, list[int]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise CacheContractError(f"per-dataset sequence metadata is not JSON: {error}") from error
    if not isinstance(value, dict):
        raise CacheContractError("per-dataset sequence metadata must be an object")
    expected = [entry["indexed_prefix"] for entry in _source_entries(contract)]
    if list(value) != expected:
        raise CacheContractError("per-dataset sequence metadata keys/order differ from the pinned source prefixes")
    result: dict[str, list[int]] = {}
    for prefix in expected:
        counts = value[prefix]
        if (
            not isinstance(counts, list)
            or len(counts) != 2
            or any(not isinstance(count, int) or isinstance(count, bool) or count <= 0 for count in counts)
        ):
            raise CacheContractError("per-dataset sequence metadata must hold positive [sequences, documents] pairs")
        result[prefix] = counts
    return result


def _validate_sequences_against_indexed_components(
    sequences: Mapping[str, list[int]], contract: Mapping[str, object]
) -> None:
    """Compare cache metadata with the authoritative MCore ``.idx`` header.

    ``build_sequences_per_dataset.py`` records these two values to avoid an
    index-file read in the fast cache path.  They are therefore performance
    metadata, not self-authenticating evidence: independently re-read the
    source index before accepting the cache.
    """
    for entry in _source_entries(contract):
        prefix = Path(str(entry["indexed_prefix"]))
        observed = _mcore_sequence_document_counts(prefix)
        expected = sequences[str(prefix)]
        if observed != (expected[0], expected[1]):
            raise CacheContractError(
                "per-dataset sequence metadata disagrees with the authoritative indexed .idx header: "
                f"{prefix} expected {expected}, observed {list(observed)}"
            )


def _mcore_sequence_document_counts(prefix: Path) -> tuple[int, int]:
    """Read exact counts through the installed MCore indexed-dataset reader."""
    try:
        from megatron.core.datasets.indexed_dataset import _IndexReader
    except ImportError as error:
        raise CacheContractError("Megatron Core _IndexReader is required for cache sequence validation") from error
    reader = _IndexReader(f"{prefix}.idx", False)
    sequence_count = getattr(reader, "sequence_count", None)
    document_count = getattr(reader, "document_count", None)
    if (
        not isinstance(sequence_count, int)
        or isinstance(sequence_count, bool)
        or sequence_count <= 0
        or not isinstance(document_count, int)
        or isinstance(document_count, bool)
        or document_count <= 0
    ):
        raise CacheContractError(f"MCore _IndexReader returned invalid counts for {prefix}.idx")
    return sequence_count, document_count


def _description(path: Path, digest: str, expected_class: str) -> dict[str, object]:
    raw = path.read_bytes()
    if hashlib.md5(raw, usedforsecurity=False).hexdigest() != digest:
        raise CacheContractError(f"cache descriptor filename hash does not bind its contents: {path.name}")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise CacheContractError(f"cache descriptor is not JSON: {path.name}") from error
    if not isinstance(value, dict) or value.get("class") != expected_class:
        raise CacheContractError(f"cache descriptor has the wrong dataset class: {path.name}")
    return value
