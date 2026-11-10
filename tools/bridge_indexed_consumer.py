"""Apply and verify AMPA's reviewed indexed-data contract on Bridge config.

This module has no Megatron imports.  It works on Bridge's normal top-level
config (``config.dataset`` and ``config.tokenizer``) and is shared by the
offline cache-hit probe and the trainer.  The cache builder, rather than this
adapter, remains authoritative for sample-count and cache-artifact validation.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


class IndexedConsumerError(ValueError):
    """The configured Bridge consumer differs from the reviewed manifest."""


_DATA_FIELDS = (
    "data_path",
    "blend",
    "blend_per_split",
    "split",
    "path_to_cache",
    "per_dataset_sequences_path",
    "fast_cache_load",
    "defer_npy_index_mmap",
    "random_seed",
    "sequence_length",
    "data_parallel_size",
    "reset_attention_mask",
    "reset_position_ids",
    "eod_mask_loss",
    "add_extra_token_to_sequence",
    "mid_level_dataset_surplus",
)
_SPEC_FIELDS = frozenset((*_DATA_FIELDS, "train_samples"))
_TOKENIZER_KWARGS = {
    "use_fast": True,
    "include_special_tokens": True,
    "trust_remote_code": False,
}


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise IndexedConsumerError(f"{label} must be an object")
    return value


def _json_value(value: object) -> Any:
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_regular_file(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise IndexedConsumerError(f"{label} is not a regular file: {path}")


def _contract(manifest: Mapping[str, object]) -> tuple[dict[str, Any], str]:
    cache = _mapping(manifest.get("megatron_cache"), "megatron_cache")
    if cache.get("status") != "validated":
        raise IndexedConsumerError("megatron_cache is not validated")
    spec = dict(_mapping(cache.get("bridge_consumer"), "megatron_cache.bridge_consumer"))
    if set(spec) != _SPEC_FIELDS:
        raise IndexedConsumerError(
            f"bridge_consumer fields differ: missing={sorted(_SPEC_FIELDS - set(spec))}, "
            f"unexpected={sorted(set(spec) - _SPEC_FIELDS)}"
        )
    if spec["data_path"] is not None or spec["blend"] is not None or spec["split"] is not None:
        raise IndexedConsumerError("fast cache requires null data_path, blend, and split")
    blend = spec["blend_per_split"]
    if (
        not isinstance(blend, list)
        or len(blend) != 3
        or blend[1:] != [None, None]
        or not isinstance(blend[0], list)
        or len(blend[0]) != 2
        or not isinstance(blend[0][0], list)
        or not isinstance(blend[0][1], list)
        or len(blend[0][0]) != 4
        or len(blend[0][1]) != 4
        or any(not isinstance(item, str) or not item for item in blend[0][0])
        or any(type(item) not in (float, int) or not math.isfinite(item) or item <= 0 for item in blend[0][1])
    ):
        raise IndexedConsumerError("bridge_consumer must specify four positive train-only weighted prefixes")
    if not math.isclose(sum(blend[0][1]), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise IndexedConsumerError("bridge_consumer train weights must sum to one")
    for field in (
        "fast_cache_load",
        "defer_npy_index_mmap",
        "reset_attention_mask",
        "reset_position_ids",
        "eod_mask_loss",
        "add_extra_token_to_sequence",
    ):
        expected = field in ("fast_cache_load", "add_extra_token_to_sequence")
        if spec[field] is not expected:
            raise IndexedConsumerError(f"unsupported bridge_consumer.{field}={spec[field]!r}")
    for field in ("random_seed", "sequence_length", "data_parallel_size", "train_samples"):
        if type(spec[field]) is not int or spec[field] <= 0:
            raise IndexedConsumerError(f"bridge_consumer.{field} must be a positive integer")
    if type(spec["mid_level_dataset_surplus"]) not in (int, float) or spec["mid_level_dataset_surplus"] != 0.005:
        raise IndexedConsumerError("bridge_consumer.mid_level_dataset_surplus must be 0.005")
    for field in ("path_to_cache", "per_dataset_sequences_path"):
        if not isinstance(spec[field], str) or not Path(spec[field]).is_absolute():
            raise IndexedConsumerError(f"bridge_consumer.{field} must be an absolute path")
    if Path(spec["path_to_cache"]) != Path(str(cache.get("path"))):
        raise IndexedConsumerError("bridge_consumer cache path differs from manifest cache path")
    if Path(spec["per_dataset_sequences_path"]) != Path(str(cache.get("per_dataset_sequences_path"))):
        raise IndexedConsumerError("bridge_consumer sequences path differs from manifest cache path")
    if Path(spec["per_dataset_sequences_path"]).parent != Path(spec["path_to_cache"]):
        raise IndexedConsumerError("per-dataset sequences file is not in the selected cache directory")
    snapshot = _mapping(cache.get("pinned_hf_snapshot"), "megatron_cache.pinned_hf_snapshot")
    snapshot_path = snapshot.get("path")
    if not isinstance(snapshot_path, str) or not Path(snapshot_path).is_absolute():
        raise IndexedConsumerError("pinned HF tokenizer snapshot must be an absolute local path")
    if Path(snapshot_path).is_symlink() or not Path(snapshot_path).is_dir():
        raise IndexedConsumerError("pinned HF tokenizer snapshot is not a local non-symlink directory")
    manifest_path = Path(str(snapshot.get("manifest_path", "")))
    if manifest_path != Path(snapshot_path) / "ampa_hf_snapshot_manifest.json":
        raise IndexedConsumerError("pinned HF snapshot manifest path differs from snapshot directory")
    _require_regular_file(manifest_path, "pinned HF snapshot manifest")
    if _sha256_file(manifest_path) != snapshot.get("manifest_sha256"):
        raise IndexedConsumerError("pinned HF snapshot manifest SHA256 changed")
    model = _mapping(manifest.get("model_and_tokenizer"), "model_and_tokenizer")
    for field in ("model_id", "revision", "tokenizer_id", "tokenizer_revision"):
        if snapshot.get(field) != model.get(field) or not isinstance(snapshot.get(field), str):
            raise IndexedConsumerError("pinned HF tokenizer identity differs from dataset manifest")
    if model.get("model_id") != model.get("tokenizer_id") or model.get("revision") != model.get("tokenizer_revision"):
        raise IndexedConsumerError("pinned HF tokenizer identity differs from dataset manifest")
    return spec, snapshot_path


def expected_bridge_indexed_consumer(manifest: Mapping[str, object]) -> dict[str, Any]:
    """Return the normalized, manifest-bound Bridge configuration to verify."""
    spec, snapshot_path = _contract(manifest)
    return {
        **{field: _json_value(spec[field]) for field in _DATA_FIELDS},
        "tokenizer_type": "HuggingFaceTokenizer",
        "tokenizer_model": snapshot_path,
        "hf_tokenizer_kwargs": dict(_TOKENIZER_KWARGS),
    }


def apply_bridge_indexed_consumer(config: Any, manifest: Mapping[str, object]) -> dict[str, Any]:
    """Configure Bridge's normal indexed-data path; call before config finalize."""
    expected = expected_bridge_indexed_consumer(manifest)
    try:
        dataset = config.dataset
        tokenizer = config.tokenizer
    except AttributeError as error:
        raise IndexedConsumerError("config requires dataset and tokenizer attributes") from error
    for field in _DATA_FIELDS:
        value = expected[field]
        if field == "blend_per_split":
            value = [(list(value[0][0]), list(value[0][1])), None, None]
        setattr(dataset, field, value)
    dataset.seq_length = expected["sequence_length"]
    # Finalize must load the pinned file, not inherit stale precomputed values.
    dataset.sequences_per_dataset = None
    tokenizer.tokenizer_type = expected["tokenizer_type"]
    tokenizer.tokenizer_model = expected["tokenizer_model"]
    tokenizer.hf_tokenizer_kwargs = dict(expected["hf_tokenizer_kwargs"])
    tokenizer.tokenizer_hf_no_use_fast = False
    tokenizer.tokenizer_hf_no_include_special_tokens = False
    tokenizer.trust_remote_code = False
    return expected


def bridge_indexed_consumer_view(config: Any) -> dict[str, Any]:
    """Read back the actual Bridge config without trusting the input manifest."""
    try:
        dataset = config.dataset
        tokenizer = config.tokenizer
        view = {field: _json_value(getattr(dataset, field)) for field in _DATA_FIELDS}
        view.update(
            tokenizer_type=tokenizer.tokenizer_type,
            tokenizer_model=tokenizer.tokenizer_model,
            hf_tokenizer_kwargs=_json_value(tokenizer.hf_tokenizer_kwargs),
        )
        return view
    except AttributeError as error:
        raise IndexedConsumerError(f"Bridge indexed consumer field missing: {error}") from error


def assert_bridge_indexed_consumer(config: Any, manifest: Mapping[str, object]) -> dict[str, Any]:
    """Fail closed if Bridge's finalized dataset/tokenizer differs from C0 data contract."""
    expected = expected_bridge_indexed_consumer(manifest)
    actual = bridge_indexed_consumer_view(config)
    for field, value in expected.items():
        if actual[field] != value:
            raise IndexedConsumerError(f"Bridge finalized {field} differs from reviewed indexed consumer")
    dataset = config.dataset
    if dataset.seq_length != expected["sequence_length"]:
        raise IndexedConsumerError("Bridge seq_length differs from reviewed sequence_length")
    sequences_path = Path(expected["per_dataset_sequences_path"])
    _require_regular_file(sequences_path, "per-dataset sequences file")
    cache = _mapping(manifest["megatron_cache"], "megatron_cache")
    if _sha256_file(sequences_path) != cache.get("per_dataset_sequences_sha256"):
        raise IndexedConsumerError("per-dataset sequences file SHA256 changed")
    try:
        sequences = json.loads(sequences_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise IndexedConsumerError("per-dataset sequences file is unreadable JSON") from error
    if _json_value(getattr(dataset, "sequences_per_dataset", None)) != sequences:
        raise IndexedConsumerError("Bridge finalized sequences_per_dataset differs from reviewed cache file")
    if (
        getattr(config.tokenizer, "tokenizer_hf_no_use_fast", None) is not False
        or getattr(config.tokenizer, "tokenizer_hf_no_include_special_tokens", None) is not False
        or getattr(config.tokenizer, "trust_remote_code", None) is not False
    ):
        raise IndexedConsumerError("Bridge effective tokenizer flags differ from reviewed HF kwargs")
    return actual
