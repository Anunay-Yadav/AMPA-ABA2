#!/usr/bin/env python3
"""Prepare, validate, and probe AMPA's fixed normal-GPT dataset cache.

This is deliberately a thin orchestration layer around Megatron Core's native
``build_sequences_per_dataset.py`` and ``prepare_cache.py`` utilities.  It
does not implement a dataset builder, training loop, or alternate cache
format.  Each subcommand writes a new, non-canonical dataset-manifest
candidate; none advances the dataset to canonical-training approval.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace
from typing import Any, Mapping, Sequence


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.megatron_cache_contract import (  # noqa: E402
    CacheContractError,
    build_cache_contract_from_manifest,
    cache_contract_sha256,
    cache_directory_for_contract,
    canonical_json_sha256,
    expected_blended_train_size,
    sha256_file,
    validate_cache_inventory,
    verify_source_inventory,
)
from tools.bridge_indexed_consumer import (  # noqa: E402
    IndexedConsumerError,
    apply_bridge_indexed_consumer,
    assert_bridge_indexed_consumer,
)


_LIFECYCLE_VERSION = 2
_TRANSITION_SEQUENCE = (
    "planned_not_materialized",
    "materialized_pending_data_review",
    "indexed_validated_pending_cache",
    "cache_prepared_pending_data_review",
    "cache_validated_pending_independent_review",
    "validated_for_canonical_run",
)
_CACHE_STAGE_SCHEMA_VERSION = 1
_MID_LEVEL_DATASET_SURPLUS = 0.005
_CACHE_SEQUENCE_NAME = "per_dataset_sequences.json"
_BUILD_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_GPT_DESCRIPTOR = re.compile(r"^[0-9a-f]{32}-GPTDataset-train-description\.txt\Z")


class CacheStageError(ValueError):
    """Raised when a cache-stage candidate is unsafe or internally inconsistent."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", allow_abbrev=False)
    prepare.add_argument("--indexed-manifest", type=Path, required=True)
    prepare.add_argument("--output-manifest", type=Path, required=True)
    prepare.add_argument("--workspace-root", type=Path, required=True)
    prepare.add_argument("--build-id", required=True)
    prepare.add_argument("--python", type=Path, default=Path(sys.executable))
    prepare.add_argument("--mcore-root", type=Path)
    prepare.add_argument("--variant", choices=("cp2_ep8",), help="opt-in CP2/EP8 cache contract")

    validate = subparsers.add_parser("validate", allow_abbrev=False)
    validate.add_argument("--cache-prepared-manifest", type=Path, required=True)
    validate.add_argument("--output-manifest", type=Path, required=True)
    validate.add_argument("--workspace-root", type=Path, required=True)

    probe = subparsers.add_parser("bridge-cache-hit-probe", allow_abbrev=False)
    probe.add_argument("--cache-validated-manifest", type=Path, required=True)
    probe.add_argument("--output-manifest", type=Path, required=True)
    probe.add_argument("--workspace-root", type=Path, required=True)

    return parser.parse_args()


def _workspace_root(value: Path) -> Path:
    if value.is_symlink() or not value.is_dir():
        raise CacheStageError(f"workspace root must be an existing non-symlink directory: {value}")
    return value.resolve()


def _regular_file(value: Path, label: str) -> Path:
    if value.is_symlink() or not value.is_file():
        raise CacheStageError(f"{label} must be an existing non-symlink regular file: {value}")
    return value.resolve()


def _python_interpreter(value: Path) -> Path:
    """Allow a normal venv launcher symlink without resolving away its venv."""
    if not value.is_absolute() or not value.is_file() or not os.access(value, os.X_OK):
        raise CacheStageError(f"Python interpreter must resolve to an existing executable file: {value}")
    return value


def _require_under(path: Path, root: Path, label: str) -> None:
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise CacheStageError(f"{label} escapes its required root: {path}") from error


def _read_json(path: Path, label: str) -> dict[str, Any]:
    path = _regular_file(path, label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise CacheStageError(f"{label} is not valid JSON: {error}") from error
    if not isinstance(value, dict):
        raise CacheStageError(f"{label} must contain a JSON object")
    return value


def _write_json_staged(path: Path, payload: Mapping[str, object], discriminator: str) -> Path:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite candidate manifest: {path}")
    if not _BUILD_ID.fullmatch(discriminator):
        raise CacheStageError("candidate-manifest discriminator must be a safe non-empty identifier")
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(f"{path}.partial-{discriminator}")
    if staged.exists() or staged.is_symlink():
        raise FileExistsError(f"staged candidate manifest already exists: {staged}")
    with staged.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return staged


def _lifecycle(status: str, **evidence: object) -> dict[str, object]:
    if status not in _TRANSITION_SEQUENCE:
        raise CacheStageError(f"unsupported AMPA dataset lifecycle status: {status}")
    return {
        "status": status,
        "state_machine_version": _LIFECYCLE_VERSION,
        "transition_sequence": list(_TRANSITION_SEQUENCE),
        "canonical_training_allowed": False,
        **evidence,
    }


def _require_lifecycle(candidate: Mapping[str, object], status: str, label: str) -> None:
    lifecycle = candidate.get("lifecycle")
    if not isinstance(lifecycle, Mapping) or lifecycle.get("status") != status:
        raise CacheStageError(f"{label} is not a {status} candidate")
    if lifecycle.get("state_machine_version") != _LIFECYCLE_VERSION:
        raise CacheStageError(f"{label} does not use the AMPA dataset lifecycle version")
    if lifecycle.get("transition_sequence") != list(_TRANSITION_SEQUENCE):
        raise CacheStageError(f"{label} does not use the AMPA dataset lifecycle state machine")
    if lifecycle.get("canonical_training_allowed") is not False:
        raise CacheStageError(f"{label} must not permit canonical training")


def _model_identity(manifest: Mapping[str, object]) -> dict[str, str]:
    model = manifest.get("model_and_tokenizer")
    if not isinstance(model, Mapping):
        raise CacheStageError("dataset manifest lacks model_and_tokenizer")
    required = ("model_id", "revision", "tokenizer_id", "tokenizer_revision")
    if any(not isinstance(model.get(key), str) or not model[key] for key in required):
        raise CacheStageError("dataset manifest has an incomplete model/tokenizer identity")
    return {key: str(model[key]) for key in required}


def _sha256_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise CacheStageError(f"{label} must be a lowercase SHA256")
    return value


def resolve_pinned_hf_snapshot(workspace_root: Path, model: Mapping[str, object]) -> dict[str, object]:
    """Find exactly one local HF snapshot bound to the input data manifest.

    The native MCore cache utility has no Hugging Face ``revision`` argument.
    Passing the local, immutable snapshot makes its tokenizer cache key stable
    and exactly matches the C0 tokenizer path recorded by Bridge.
    """
    workspace = _workspace_root(workspace_root)
    identity = _model_identity({"model_and_tokenizer": model})
    hf_root = workspace / "checkpoint" / "hf"
    if not hf_root.is_dir() or hf_root.is_symlink():
        raise CacheStageError(f"pinned HF snapshot root is missing: {hf_root}")
    matches: list[Path] = []
    for manifest_path in hf_root.glob(f"*/{identity['tokenizer_revision']}/ampa_hf_snapshot_manifest.json"):
        if manifest_path.is_symlink() or not manifest_path.is_file():
            continue
        snapshot = manifest_path.parent.resolve()
        try:
            _require_under(snapshot, hf_root, "pinned HF snapshot")
            record = _read_json(manifest_path, "pinned HF snapshot manifest")
        except CacheStageError:
            continue
        if (
            record.get("model_id") == identity["model_id"]
            and record.get("revision") == identity["revision"]
            and record.get("resolved_revision") == identity["tokenizer_revision"]
        ):
            matches.append(snapshot)
    if len(matches) != 1:
        raise CacheStageError(
            "expected exactly one local pinned HF snapshot matching the model/tokenizer revision; "
            f"found {len(matches)}"
        )
    snapshot = matches[0]
    snapshot_manifest = _regular_file(snapshot / "ampa_hf_snapshot_manifest.json", "pinned HF snapshot manifest")
    hf_config = _regular_file(snapshot / "config.json", "pinned HF model config")
    tokenizer_config = _regular_file(snapshot / "tokenizer_config.json", "pinned HF tokenizer config")
    return {
        "path": str(snapshot),
        "manifest_path": str(snapshot_manifest),
        "manifest_sha256": sha256_file(snapshot_manifest),
        "hf_config_path": str(hf_config),
        "hf_config_sha256": sha256_file(hf_config),
        "tokenizer_config_path": str(tokenizer_config),
        "tokenizer_config_sha256": sha256_file(tokenizer_config),
        "model_id": identity["model_id"],
        "revision": identity["revision"],
        "tokenizer_id": identity["tokenizer_id"],
        "tokenizer_revision": identity["tokenizer_revision"],
    }


def native_model_args_from_snapshot(snapshot: Mapping[str, object]) -> dict[str, int]:
    """Extract MCore validation fields from the immutable HF model config."""
    config_path = _regular_file(Path(str(snapshot.get("hf_config_path", ""))), "pinned HF model config")
    if sha256_file(config_path) != _sha256_text(snapshot.get("hf_config_sha256"), "pinned HF model config SHA256"):
        raise CacheStageError("pinned HF model config changed after snapshot resolution")
    config = _read_json(config_path, "pinned HF model config")
    fields = {
        "num_layers": "num_hidden_layers",
        "hidden_size": "hidden_size",
        "num_attention_heads": "num_attention_heads",
        "max_position_embeddings": "max_position_embeddings",
        "num_experts": "num_experts",
    }
    result: dict[str, int] = {}
    for output, source in fields.items():
        value = config.get(source)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise CacheStageError(f"pinned HF model config has no positive integer {source}")
        result[output] = value
    return result


def bridge_consumer_spec(contract: Mapping[str, object], *, cache_dir: Path) -> dict[str, object]:
    """Return the fixed Bridge/MCore data fields used by the cache-hit probe.

    This deliberately has no flat ``data_path`` field: MCore rejects the flat
    blend with ``fast_cache_load=True``.  Both native preparation and Bridge
    consumption instead use the same train-only per-split blend.
    """
    try:
        data = contract["data"]
        workload = contract["workload_identity"]
        prepare = contract["prepare_cache"]
    except KeyError as error:
        raise CacheStageError("cache contract is missing a required section") from error
    if not isinstance(data, Mapping) or not isinstance(workload, Mapping) or not isinstance(prepare, Mapping):
        raise CacheStageError("cache contract sections must be objects")
    blend = data.get("blend_per_split")
    if not isinstance(blend, Mapping):
        raise CacheStageError("cache contract lacks normalized blend_per_split")
    train = blend.get("train")
    if not isinstance(train, Mapping) or blend.get("valid") is not None or blend.get("test") is not None:
        raise CacheStageError("cache consumer must use only the explicit train per-split blend")
    prefixes = train.get("prefixes")
    weights = train.get("weights")
    if (
        not isinstance(prefixes, list)
        or not isinstance(weights, list)
        or len(prefixes) != 4
        or len(weights) != 4
        or any(not isinstance(item, str) or not item for item in prefixes + weights)
    ):
        raise CacheStageError("cache contract train per-split blend is invalid")
    try:
        numeric_weights = [float(item) for item in weights]
    except ValueError as error:
        raise CacheStageError("cache contract train weights are not numeric") from error
    if any(weight <= 0.0 for weight in numeric_weights):
        raise CacheStageError("cache contract train weights must be positive")
    cache_path = cache_dir.resolve()
    return {
        "data_path": None,
        "blend": None,
        "blend_per_split": [[list(prefixes), numeric_weights], None, None],
        "split": None,
        "path_to_cache": str(cache_path),
        "per_dataset_sequences_path": str(cache_path / _CACHE_SEQUENCE_NAME),
        "fast_cache_load": True,
        "defer_npy_index_mmap": False,
        "random_seed": workload["sampler_seed"],
        "sequence_length": workload["sequence_length"],
        "data_parallel_size": workload["dense_data_parallel_size"],
        "train_samples": prepare["train_samples"],
        "reset_attention_mask": False,
        "reset_position_ids": False,
        "eod_mask_loss": False,
        "add_extra_token_to_sequence": True,
        "mid_level_dataset_surplus": _MID_LEVEL_DATASET_SURPLUS,
    }


def _context_parallel_size(contract: Mapping[str, object]) -> int:
    prepare = contract.get("prepare_cache")
    if not isinstance(prepare, Mapping):
        raise CacheStageError("cache contract lacks preparation parallelism")
    parallelism = prepare.get("parallelism")
    if not isinstance(parallelism, Mapping) or type(parallelism.get("cp")) is not int or parallelism["cp"] not in (1, 2):
        raise CacheStageError("cache contract has no supported context-parallel size")
    return parallelism["cp"]


def native_prepare_command(
    *,
    python: Path,
    mcore_root: Path,
    contract: Mapping[str, object],
    per_split_args_path: Path,
    sequences_path: Path,
    cache_dir: Path,
    snapshot: Mapping[str, object],
) -> list[str]:
    """Build the exact native MCore command for the selected cache contract."""
    model_args = native_model_args_from_snapshot(snapshot)
    try:
        workload = contract["workload_identity"]
        prepare = contract["prepare_cache"]
    except KeyError as error:
        raise CacheStageError("cache contract is missing preparation fields") from error
    if not isinstance(workload, Mapping) or not isinstance(prepare, Mapping):
        raise CacheStageError("cache contract preparation fields must be objects")
    context_parallel_size = _context_parallel_size(contract)
    script = _regular_file(mcore_root / "tools" / "prepare_cache.py", "native MCore prepare_cache.py")
    return [
        str(_python_interpreter(python)),
        str(script),
        "--per-split-data-args-path",
        str(per_split_args_path),
        "--per-dataset-sequences-path",
        str(sequences_path),
        "--data-cache-path",
        str(cache_dir),
        "--tokenizer-type",
        "HuggingFaceTokenizer",
        "--tokenizer-model",
        str(snapshot["path"]),
        "--num-layers",
        str(model_args["num_layers"]),
        "--hidden-size",
        str(model_args["hidden_size"]),
        "--num-attention-heads",
        str(model_args["num_attention_heads"]),
        "--max-position-embeddings",
        str(model_args["max_position_embeddings"]),
        "--seq-length",
        str(workload["sequence_length"]),
        "--micro-batch-size",
        str(workload["micro_batch_size"]),
        "--global-batch-size",
        str(workload["global_batch_size"]),
        "--train-iters",
        str(prepare["train_iters"]),
        "--eval-iters",
        str(prepare["eval_iters"]),
        "--eval-interval",
        str(prepare["eval_interval"]),
        "--seed",
        str(workload["sampler_seed"]),
        "--mid-level-dataset-surplus",
        str(_MID_LEVEL_DATASET_SURPLUS),
        "--prepare-cache-world-size",
        str(prepare["prepare_cache_world_size"]),
        "--tensor-model-parallel-size",
        "1",
        "--pipeline-model-parallel-size",
        "1",
        "--context-parallel-size",
        str(context_parallel_size),
        # MCore validates EP against num_experts even though this utility only
        # builds data indices.  Bind both from the pinned HF config so the
        # offline job has the same EP=8 topology as the consumer.
        "--num-experts",
        str(model_args["num_experts"]),
        "--expert-model-parallel-size",
        "8",
        "--expert-tensor-parallel-size",
        "1",
    ]


def native_sequence_command(
    *, python: Path, mcore_root: Path, per_split_args_path: Path, sequences_path: Path
) -> list[str]:
    script = _regular_file(
        mcore_root / "tools" / "build_sequences_per_dataset.py", "native MCore build_sequences_per_dataset.py"
    )
    return [
        str(_python_interpreter(python)),
        str(script),
        "--per-split-data-args-path",
        str(per_split_args_path),
        "--per-dataset-sequences-path",
        str(sequences_path),
    ]


def _safe_build_id(value: str) -> str:
    if not isinstance(value, str) or not _BUILD_ID.fullmatch(value) or value in {".", ".."}:
        raise CacheStageError("build-id must be a safe non-empty identifier")
    return value


def _write_per_split_args(path: Path, contract: Mapping[str, object]) -> None:
    data = contract.get("data")
    if not isinstance(data, Mapping) or not isinstance(data.get("per_split_data_args"), Mapping):
        raise CacheStageError("cache contract lacks per_split_data_args")
    value = dict(data["per_split_data_args"])
    if set(value) != {"train", "valid", "test"} or value.get("valid") is not None or value.get("test") is not None:
        raise CacheStageError("native cache preparation requires a train-only per-split blend")
    if not isinstance(value.get("train"), str) or not value["train"]:
        raise CacheStageError("native cache preparation train blend is empty")
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def _cache_preflight(cache_dir: Path) -> list[dict[str, object]]:
    """Reject a failed native build before its cache can be atomically published.

    This is intentionally structural only.  The separate validation stage
    rehashes sources and validates MCore descriptors/indices independently.
    """
    if cache_dir.is_symlink() or not cache_dir.is_dir():
        raise CacheStageError("native cache preparation did not create a regular cache directory")
    files: list[Path] = []
    for path in sorted(cache_dir.rglob("*")):
        relative = path.relative_to(cache_dir)
        if path.is_symlink():
            raise CacheStageError(f"native cache preparation produced a symlink: {relative}")
        if any(part.startswith(".partial-") for part in relative.parts) or any(
            part.endswith(".partial") for part in relative.parts
        ):
            raise CacheStageError(f"native cache preparation left a partial artifact: {relative}")
        if path.is_file():
            if path.stat().st_size <= 0:
                raise CacheStageError(f"native cache preparation produced an empty artifact: {relative}")
            files.append(path)
        elif not path.is_dir():
            raise CacheStageError(f"native cache preparation produced a non-regular artifact: {relative}")
    # Four GPT bundles (4 files each), one blended bundle (3 files), and the
    # precomputed sequence map are the native fixed-blend layout.
    if len(files) != 20 or not (cache_dir / _CACHE_SEQUENCE_NAME).is_file():
        raise CacheStageError("native cache preparation did not produce the complete fixed-blend cache layout")
    return [
        {
            "relative_path": path.relative_to(cache_dir).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in files
    ]


def _gpt_descriptors(cache_dir: Path) -> list[dict[str, object]]:
    descriptors: list[dict[str, object]] = []
    for path in sorted(cache_dir.iterdir()):
        if not _GPT_DESCRIPTOR.fullmatch(path.name):
            continue
        descriptor = _read_json(path, "GPTDataset cache descriptor")
        descriptors.append(descriptor)
    if len(descriptors) != 4:
        raise CacheStageError("native cache does not contain four GPTDataset descriptors")
    return descriptors


def _descriptor_tokenizer_identifiers(cache_dir: Path) -> dict[str, object]:
    values: list[dict[str, object]] = []
    for descriptor in _gpt_descriptors(cache_dir):
        tokenizer = descriptor.get("tokenizer")
        if not isinstance(tokenizer, Mapping):
            raise CacheStageError("GPTDataset descriptor has no tokenizer cache key")
        values.append(dict(tokenizer))
    if any(item != values[0] for item in values[1:]):
        raise CacheStageError("native cache GPTDataset descriptors disagree on tokenizer cache identity")
    return values[0]


def _cache_preparation_record(
    *,
    contract: Mapping[str, object],
    cache_dir: Path,
    staged_cache_dir: Path,
    source_inventory: Sequence[Mapping[str, object]],
    preflight: Sequence[Mapping[str, object]],
    snapshot: Mapping[str, object],
    commands: Mapping[str, Sequence[str]],
) -> dict[str, object]:
    return {
        "schema_version": _CACHE_STAGE_SCHEMA_VERSION,
        "cache_contract": dict(contract),
        "cache_contract_sha256": cache_contract_sha256(contract),
        "cache_path": str(cache_dir),
        "source_inventory_before_native_preparation": [dict(item) for item in source_inventory],
        "native_artifact_preflight": [dict(item) for item in preflight],
        "native_commands": {name: list(command) for name, command in commands.items()},
        "pinned_hf_snapshot": dict(snapshot),
        "mcore_tokenizer_unique_identifiers": _descriptor_tokenizer_identifiers(staged_cache_dir),
        "bridge_consumer": bridge_consumer_spec(contract, cache_dir=cache_dir),
        "bridge_cache_hit_probe": {
            "required": True,
            "status": "required_not_run",
            "reason": "Native cache descriptors are not evidence that Bridge consumed the fast-cache path.",
        },
    }


def _contract_from_prepared_candidate(
    candidate: Mapping[str, object], *, workspace_root: Path
) -> tuple[dict[str, object], Mapping[str, object]]:
    preparation = candidate.get("cache_preparation")
    if not isinstance(preparation, Mapping):
        raise CacheStageError("cache-prepared candidate lacks cache_preparation evidence")
    contract = preparation.get("cache_contract")
    if not isinstance(contract, Mapping):
        raise CacheStageError("cache-prepared candidate lacks its immutable cache contract")
    contract_copy = dict(contract)
    expected_hash = _sha256_text(preparation.get("cache_contract_sha256"), "cache contract SHA256")
    if cache_contract_sha256(contract_copy) != expected_hash:
        raise CacheStageError("cache-prepared candidate cache contract SHA256 differs from its contents")
    expected_cache = cache_directory_for_contract(workspace_root, contract_copy)
    cache_path = preparation.get("cache_path")
    if not isinstance(cache_path, str) or Path(cache_path).resolve() != expected_cache.resolve():
        raise CacheStageError("cache-prepared candidate cache path differs from its immutable contract")
    indexed = contract_copy.get("indexed_manifest")
    if not isinstance(indexed, Mapping):
        raise CacheStageError("cache contract lacks its indexed-manifest binding")
    path_value = indexed.get("path")
    if not isinstance(path_value, str):
        raise CacheStageError("cache contract indexed-manifest path is missing")
    indexed_path = _regular_file(Path(path_value), "cache contract indexed-validation manifest")
    _require_under(indexed_path, workspace_root / "dataset" / "manifests", "cache contract indexed-validation manifest")
    if sha256_file(indexed_path) != _sha256_text(indexed.get("sha256"), "indexed manifest SHA256"):
        raise CacheStageError("indexed validation manifest changed after the cache contract was created")
    indexed_candidate = _read_json(indexed_path, "cache contract indexed-validation manifest")
    if _model_identity(indexed_candidate) != _model_identity(candidate):
        raise CacheStageError("cache candidate model/tokenizer identity differs from its indexed-validation input")
    lifecycle = candidate.get("lifecycle")
    if not isinstance(lifecycle, Mapping):
        raise CacheStageError("cache candidate has no lifecycle evidence")
    if lifecycle.get("status") == "cache_prepared_pending_data_review" and lifecycle.get(
        "indexed_validation_manifest_sha256"
    ) != indexed.get("sha256"):
        raise CacheStageError("cache-prepared lifecycle is not bound to its indexed-validation input")
    return contract_copy, preparation


def _validate_preparation_tokenizer(
    *, preparation: Mapping[str, object], cache_dir: Path, snapshot: Mapping[str, object]
) -> dict[str, object]:
    recorded_snapshot = preparation.get("pinned_hf_snapshot")
    if not isinstance(recorded_snapshot, Mapping) or dict(recorded_snapshot) != dict(snapshot):
        raise CacheStageError("pinned HF snapshot identity changed after native cache preparation")
    recorded = preparation.get("mcore_tokenizer_unique_identifiers")
    if not isinstance(recorded, Mapping):
        raise CacheStageError("cache-prepared candidate has no recorded MCore tokenizer identity")
    observed = _descriptor_tokenizer_identifiers(cache_dir)
    if dict(recorded) != observed:
        raise CacheStageError("cache descriptors no longer match recorded MCore tokenizer identity")
    expected_path = str(snapshot["path"])
    if observed.get("tokenizer_path") != expected_path:
        raise CacheStageError("cache descriptors do not use the pinned local HF tokenizer snapshot")
    return observed


def _validate_all_gpt_cache_key_fields(contract: Mapping[str, object], cache_dir: Path) -> None:
    """Independently check every GPTDataset cache-key field fixed by AMPA.

    ``num_samples`` is easy to overlook: MCore derives it from the train blend,
    target sample count, and ``mid_level_dataset_surplus``.  It participates in
    each GPTDataset cache key, so accept neither an arbitrary surplus nor a
    cache prepared for a different number of optimizer steps.
    """
    data = contract.get("data")
    workload = contract.get("workload_identity")
    prepare = contract.get("prepare_cache")
    if not isinstance(data, Mapping) or not isinstance(workload, Mapping) or not isinstance(prepare, Mapping):
        raise CacheStageError("cache contract has malformed GPT cache-key fields")
    sources = data.get("source_artifacts")
    if not isinstance(sources, list) or len(sources) != 4:
        raise CacheStageError("cache contract has malformed source artifacts")
    target_samples = prepare.get("train_samples")
    if not isinstance(target_samples, int) or isinstance(target_samples, bool) or target_samples <= 0:
        raise CacheStageError("cache contract has an invalid train sample count")
    expected: dict[str, int] = {}
    for source in sources:
        if not isinstance(source, Mapping):
            raise CacheStageError("cache contract source artifact is malformed")
        prefix = source.get("indexed_prefix")
        weight = source.get("weight")
        if not isinstance(prefix, str) or not prefix or not isinstance(weight, str):
            raise CacheStageError("cache contract source path or weight is malformed")
        try:
            numeric_weight = float(weight)
        except ValueError as error:
            raise CacheStageError("cache contract source weight is not numeric") from error
        expected[prefix] = int(math.ceil(math.ceil(target_samples * numeric_weight) * (1 + _MID_LEVEL_DATASET_SURPLUS)))
    seen: set[str] = set()
    for descriptor in _gpt_descriptors(cache_dir):
        prefix = descriptor.get("dataset_path")
        if not isinstance(prefix, str) or prefix not in expected or prefix in seen:
            raise CacheStageError("GPTDataset descriptors do not bind exactly the pinned source paths")
        seen.add(prefix)
        if descriptor.get("num_samples") != expected[prefix]:
            raise CacheStageError(
                "GPTDataset cache num_samples differs from the fixed train blend, 25-step target, "
                "or mid_level_dataset_surplus=0.005"
            )
        if descriptor.get("index_split") != "train":
            raise CacheStageError("GPTDataset cache index_split is not train")
        if descriptor.get("random_seed") != workload.get("sampler_seed"):
            raise CacheStageError("GPTDataset cache random_seed differs from the pinned sampler seed")
        if descriptor.get("sequence_length") != workload.get("sequence_length"):
            raise CacheStageError("GPTDataset cache sequence_length differs from L4096")
        if descriptor.get("split") is not None or descriptor.get("split_matrix") is not None:
            raise CacheStageError("GPTDataset cache did not use the required train-only per-split blend")
    if set(expected) != seen:
        raise CacheStageError("GPTDataset descriptors are missing a pinned source path")


def prepare_cache(args: argparse.Namespace) -> dict[str, object]:
    workspace = _workspace_root(args.workspace_root)
    if args.output_manifest.is_symlink():
        raise CacheStageError(f"cache-prepared output manifest must not be a symlink: {args.output_manifest}")
    args.output_manifest = args.output_manifest.resolve()
    _require_under(args.output_manifest, workspace / "dataset" / "manifests", "cache-prepared output manifest")
    indexed_manifest = _regular_file(args.indexed_manifest, "indexed-validation manifest")
    _require_under(indexed_manifest, workspace / "dataset" / "manifests", "indexed-validation manifest")
    if args.output_manifest.exists() or args.output_manifest.is_symlink():
        raise FileExistsError(f"refusing to overwrite cache-prepared manifest: {args.output_manifest}")
    build_id = _safe_build_id(args.build_id)
    contract = build_cache_contract_from_manifest(
        indexed_manifest, workspace_root=workspace, variant=getattr(args, "variant", None)
    )
    indexed = _read_json(indexed_manifest, "indexed-validation manifest")
    _require_lifecycle(indexed, "indexed_validated_pending_cache", "indexed-validation manifest")
    snapshot = resolve_pinned_hf_snapshot(workspace, _model_identity(indexed))
    target = cache_directory_for_contract(workspace, contract)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"refusing to overwrite immutable cache destination: {target}")
    mcore_root = (args.mcore_root if args.mcore_root is not None else workspace / "Megatron-LM").resolve()
    if mcore_root.is_symlink() or not mcore_root.is_dir():
        raise CacheStageError(f"MCore root must be an existing non-symlink directory: {mcore_root}")
    python = _python_interpreter(args.python)

    stage_root = target.parent / f".partial-{target.name}-{build_id}"
    if stage_root.exists() or stage_root.is_symlink():
        raise FileExistsError(f"cache preparation staging directory already exists: {stage_root}")
    stage_root.mkdir(parents=True)
    stage_cache = stage_root / "cache"
    stage_cache.mkdir()
    per_split_args = stage_root / "per-split-data-args.json"
    sequences = stage_cache / _CACHE_SEQUENCE_NAME
    _write_per_split_args(per_split_args, contract)
    source_inventory = verify_source_inventory(contract)
    sequence_command = native_sequence_command(
        python=python, mcore_root=mcore_root, per_split_args_path=per_split_args, sequences_path=sequences
    )
    prepare_command = native_prepare_command(
        python=python,
        mcore_root=mcore_root,
        contract=contract,
        per_split_args_path=per_split_args,
        sequences_path=sequences,
        cache_dir=stage_cache,
        snapshot=snapshot,
    )
    subprocess.run(sequence_command, check=True)
    subprocess.run(prepare_command, check=True)
    preflight = _cache_preflight(stage_cache)
    preparation = _cache_preparation_record(
        contract=contract,
        cache_dir=target,
        staged_cache_dir=stage_cache,
        source_inventory=source_inventory,
        preflight=preflight,
        snapshot=snapshot,
        commands={"build_sequences_per_dataset": sequence_command, "prepare_cache": prepare_command},
    )
    candidate = copy.deepcopy(indexed)
    candidate["lifecycle"] = _lifecycle(
        "cache_prepared_pending_data_review",
        indexed_validation_manifest_sha256=sha256_file(indexed_manifest),
        cache_contract_sha256=preparation["cache_contract_sha256"],
    )
    candidate["cache_preparation"] = preparation
    staged_manifest = _write_json_staged(args.output_manifest, candidate, build_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"immutable cache destination appeared during preparation: {target}")
    os.replace(stage_cache, target)
    os.replace(staged_manifest, args.output_manifest)
    return candidate


def validate_cache(args: argparse.Namespace) -> dict[str, object]:
    workspace = _workspace_root(args.workspace_root)
    if args.output_manifest.is_symlink():
        raise CacheStageError(f"cache-validated output manifest must not be a symlink: {args.output_manifest}")
    args.output_manifest = args.output_manifest.resolve()
    _require_under(args.output_manifest, workspace / "dataset" / "manifests", "cache-validated output manifest")
    input_manifest = _regular_file(args.cache_prepared_manifest, "cache-prepared manifest")
    _require_under(input_manifest, workspace / "dataset" / "manifests", "cache-prepared manifest")
    if args.output_manifest.exists() or args.output_manifest.is_symlink():
        raise FileExistsError(f"refusing to overwrite cache-validated manifest: {args.output_manifest}")
    prepared = _read_json(input_manifest, "cache-prepared manifest")
    _require_lifecycle(prepared, "cache_prepared_pending_data_review", "cache-prepared manifest")
    contract, preparation = _contract_from_prepared_candidate(prepared, workspace_root=workspace)
    snapshot = resolve_pinned_hf_snapshot(workspace, _model_identity(prepared))
    cache_dir = cache_directory_for_contract(workspace, contract)
    inventory = validate_cache_inventory(contract, workspace_root=workspace, cache_dir=cache_dir)
    _validate_all_gpt_cache_key_fields(contract, cache_dir)
    tokenizer_ids = _validate_preparation_tokenizer(
        preparation=preparation, cache_dir=cache_dir, snapshot=snapshot
    )
    bridge_probe = preparation.get("bridge_cache_hit_probe")
    if not isinstance(bridge_probe, Mapping) or bridge_probe.get("status") != "required_not_run":
        raise CacheStageError("cache validation requires a cache-prepared candidate awaiting a Bridge cache-hit probe")
    candidate = copy.deepcopy(prepared)
    candidate["lifecycle"] = _lifecycle(
        "cache_validated_pending_independent_review",
        cache_prepared_manifest_sha256=sha256_file(input_manifest),
        cache_contract_sha256=cache_contract_sha256(contract),
    )
    candidate["cache_validation"] = {
        "schema_version": _CACHE_STAGE_SCHEMA_VERSION,
        "inventory": inventory,
        "mcore_tokenizer_unique_identifiers": tokenizer_ids,
        "pinned_hf_snapshot": snapshot,
        "bridge_cache_hit_probe": dict(bridge_probe),
    }
    staged = _write_json_staged(args.output_manifest, candidate, str(os.getpid()))
    os.replace(staged, args.output_manifest)
    return candidate


def _cache_file_fingerprint(cache_dir: Path) -> list[dict[str, object]]:
    return _cache_preflight(cache_dir)


def _bridge_tokenizer(tokenizer_config: Any):
    from megatron.bridge.training.tokenizers.tokenizer import build_tokenizer

    return build_tokenizer(tokenizer_config)


def _probe_consumer_manifest(
    contract: Mapping[str, object], *, cache_dir: Path, snapshot: Mapping[str, object]
) -> dict[str, object]:
    """Build the finalized-consumer subset required by the shared adapter."""
    spec = bridge_consumer_spec(contract, cache_dir=cache_dir)
    sequences_path = Path(str(spec["per_dataset_sequences_path"]))
    if sequences_path.is_symlink() or not sequences_path.is_file():
        raise CacheStageError("Bridge cache-hit probe cannot find the pinned sequence metadata")
    model = {
        field: snapshot.get(field)
        for field in ("model_id", "revision", "tokenizer_id", "tokenizer_revision")
    }
    if any(not isinstance(value, str) or not value for value in model.values()):
        raise CacheStageError("pinned HF snapshot has incomplete model/tokenizer identity")
    return {
        "model_and_tokenizer": model,
        "megatron_cache": {
            "status": "validated",
            "path": str(spec["path_to_cache"]),
            "per_dataset_sequences_path": str(sequences_path),
            "per_dataset_sequences_sha256": sha256_file(sequences_path),
            "bridge_consumer": spec,
            "pinned_hf_snapshot": dict(snapshot),
        },
    }


def _set_probe_token_dtype(dataset: Any, tokenizer: Any, validated_tokenizer: Mapping[str, object]) -> None:
    """Mirror Bridge setup's tokenizer-derived dtype for this isolated probe."""
    vocab_size = getattr(tokenizer, "vocab_size", None)
    if type(vocab_size) is not int or vocab_size <= 0:
        raise CacheStageError("Bridge probe tokenizer has no positive integer vocabulary size")
    # 65536 is numpy.iinfo(numpy.uint16).max + 1 in Bridge training/setup.py.
    dtype_code = 4 if vocab_size > 65536 else 8
    dtype_name = {4: "int32", 8: "uint16"}[dtype_code]
    if validated_tokenizer.get("vocab_size") != vocab_size or validated_tokenizer.get("indexed_dtype") != dtype_name:
        raise CacheStageError("Bridge probe tokenizer dtype differs from independently validated indexed data")
    if getattr(dataset, "token_dtype_code", None) is not None:
        raise CacheStageError("Bridge probe dataset unexpectedly preconfigured a token dtype")
    dataset.token_dtype_code = dtype_code


def _bridge_fast_cache_dataset(
    contract: Mapping[str, object], *, cache_dir: Path, snapshot: Mapping[str, object],
    validated_tokenizer: Mapping[str, object],
):
    """Instantiate Bridge's actual dataset config and consume the MCore cache."""
    from megatron.bridge.training.config import GPTDatasetConfig
    from megatron.bridge.training.tokenizers.config import TokenizerConfig
    from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
    from megatron.core.datasets.gpt_dataset import GPTDataset

    spec = bridge_consumer_spec(contract, cache_dir=cache_dir)
    context_parallel_size = _context_parallel_size(contract)
    consumer_manifest = _probe_consumer_manifest(contract, cache_dir=cache_dir, snapshot=snapshot)
    consumer = SimpleNamespace(
        dataset=GPTDatasetConfig(
            seq_length=1,
            random_seed=1,
            sequence_parallel_size=0,
            context_parallel_size=context_parallel_size,
            num_dataset_builder_threads=1,
        ),
        tokenizer=TokenizerConfig(
            tokenizer_type="HuggingFaceTokenizer",
            tokenizer_model=str(snapshot["path"]),
            hf_tokenizer_kwargs={},
        ),
    )
    try:
        apply_bridge_indexed_consumer(consumer, consumer_manifest)
        tokenizer = _bridge_tokenizer(consumer.tokenizer)
        consumer.dataset.tokenizer = tokenizer
        consumer.dataset.finalize()
        _set_probe_token_dtype(consumer.dataset, tokenizer, validated_tokenizer)
        actual_consumer = assert_bridge_indexed_consumer(consumer, consumer_manifest)
    except IndexedConsumerError as error:
        raise CacheStageError(f"Bridge cache-hit consumer differs from the reviewed contract: {error}") from error
    config = consumer.dataset
    if config.blend is not None or config.split is not None or config.blend_per_split is None:
        raise CacheStageError("Bridge cache consumer did not retain the explicit per-split fast-cache configuration")
    datasets = BlendedMegatronDatasetBuilder(
        GPTDataset,
        [int(spec["train_samples"]), 0, 0],
        lambda: True,
        config,
    ).build()
    train = datasets[0]
    if train is None or len(train) != expected_blended_train_size(contract):
        raise CacheStageError("Bridge fast-cache consumer did not build the pinned rounded train blend size")
    sample = train[0]
    if not isinstance(sample, Mapping) or not sample:
        raise CacheStageError("Bridge fast-cache consumer did not return a real indexed training sample")
    return config, train, sample, actual_consumer


def bridge_cache_hit_probe(args: argparse.Namespace) -> dict[str, object]:
    workspace = _workspace_root(args.workspace_root)
    if args.output_manifest.is_symlink():
        raise CacheStageError(f"Bridge cache-hit output manifest must not be a symlink: {args.output_manifest}")
    args.output_manifest = args.output_manifest.resolve()
    _require_under(args.output_manifest, workspace / "dataset" / "manifests", "Bridge cache-hit output manifest")
    input_manifest = _regular_file(args.cache_validated_manifest, "cache-validated manifest")
    _require_under(input_manifest, workspace / "dataset" / "manifests", "cache-validated manifest")
    if args.output_manifest.exists() or args.output_manifest.is_symlink():
        raise FileExistsError(f"refusing to overwrite Bridge cache-hit probe manifest: {args.output_manifest}")
    validated = _read_json(input_manifest, "cache-validated manifest")
    _require_lifecycle(validated, "cache_validated_pending_independent_review", "cache-validated manifest")
    contract, preparation = _contract_from_prepared_candidate(validated, workspace_root=workspace)
    cache_dir = cache_directory_for_contract(workspace, contract)
    # Revalidate before the probe, then require that the fast path does not
    # add, rebuild, or alter a single cache artifact.
    inventory = validate_cache_inventory(contract, workspace_root=workspace, cache_dir=cache_dir)
    previous_validation = validated.get("cache_validation")
    if not isinstance(previous_validation, Mapping):
        raise CacheStageError("Bridge cache-hit probe requires the separate cache-validation evidence")
    previous_inventory = previous_validation.get("inventory")
    if (
        not isinstance(previous_inventory, Mapping)
        or previous_inventory.get("artifact_inventory_sha256") != inventory["artifact_inventory_sha256"]
    ):
        raise CacheStageError("Bridge cache-hit probe input does not match its independently validated cache inventory")
    snapshot = resolve_pinned_hf_snapshot(workspace, _model_identity(validated))
    _validate_all_gpt_cache_key_fields(contract, cache_dir)
    descriptor_ids = _validate_preparation_tokenizer(
        preparation=preparation, cache_dir=cache_dir, snapshot=snapshot
    )
    indexed_validation = validated.get("indexed_validation")
    if not isinstance(indexed_validation, Mapping):
        raise CacheStageError("Bridge probe lacks independently validated indexed-data evidence")
    validated_tokenizer = indexed_validation.get("tokenizer_independently_validated")
    if not isinstance(validated_tokenizer, Mapping):
        raise CacheStageError("Bridge probe lacks independently validated tokenizer/dtype evidence")
    before = _cache_file_fingerprint(cache_dir)
    config, train, sample, actual_consumer = _bridge_fast_cache_dataset(
        contract, cache_dir=cache_dir, snapshot=snapshot, validated_tokenizer=validated_tokenizer
    )
    tokenizer_ids = dict(config.tokenizer.unique_identifiers)
    if tokenizer_ids != descriptor_ids:
        raise CacheStageError("Bridge tokenizer cache identity differs from the native MCore cache descriptors")
    after = _cache_file_fingerprint(cache_dir)
    if before != after:
        raise CacheStageError("Bridge fast-cache probe modified or rebuilt a cache artifact")
    candidate = copy.deepcopy(validated)
    candidate["lifecycle"] = _lifecycle(
        "cache_validated_pending_independent_review",
        cache_validated_manifest_sha256=sha256_file(input_manifest),
        cache_contract_sha256=cache_contract_sha256(contract),
        bridge_cache_hit_probe_passed=True,
    )
    candidate["cache_validation"] = {
        **dict(candidate.get("cache_validation", {})),
        "bridge_cache_hit_probe": {
            "required": True,
            "status": "passed",
            "cache_inventory_sha256_before": canonical_json_sha256(before),
            "cache_inventory_sha256_after": canonical_json_sha256(after),
            "validated_cache_inventory_sha256": inventory["artifact_inventory_sha256"],
            "mcore_tokenizer_unique_identifiers": tokenizer_ids,
            "pinned_hf_snapshot": dict(snapshot),
            "bridge_consumer": actual_consumer,
            "token_dtype_code": config.token_dtype_code,
            "train_dataset_length": len(train),
            "sample_keys": sorted(str(key) for key in sample),
        },
    }
    staged = _write_json_staged(args.output_manifest, candidate, str(os.getpid()))
    os.replace(staged, args.output_manifest)
    return candidate


def main() -> None:
    args = parse_args()
    try:
        if args.command == "prepare":
            result = prepare_cache(args)
        elif args.command == "validate":
            result = validate_cache(args)
        elif args.command == "bridge-cache-hit-probe":
            result = bridge_cache_hit_probe(args)
        else:  # pragma: no cover - argparse guarantees this branch is unreachable.
            raise CacheStageError(f"unknown cache-stage command: {args.command}")
    except (CacheContractError, CacheStageError, FileNotFoundError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"cache stage failed: {error}") from error
    print(
        json.dumps(
            {
                "manifest_id": result["manifest_id"],
                "status": result["lifecycle"]["status"],
                "canonical_training_allowed": result["lifecycle"]["canonical_training_allowed"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
