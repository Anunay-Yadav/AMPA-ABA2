#!/usr/bin/env python3
"""Independently validate AMPA indexed components without re-reading reservoir text."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


DATASET_SCHEMA_VERSION = 2
LIFECYCLE_VERSION = 2
TRANSITION_SEQUENCE = (
    "planned_not_materialized",
    "materialized_pending_data_review",
    "indexed_validated_pending_cache",
    "cache_prepared_pending_data_review",
    "cache_validated_pending_independent_review",
    "validated_for_canonical_run",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--materialized-manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _positive_decimal(value: Any, field: str) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a positive finite number")
    try:
        decimal = Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise ValueError(f"{field} must be a positive finite number") from error
    if not decimal.is_finite() or decimal <= 0:
        raise ValueError(f"{field} must be a positive finite number")
    return decimal


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _finite_scalar(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (str, int, float))
        and (not isinstance(value, str) or bool(value))
        and (not isinstance(value, float) or math.isfinite(value))
    )


def _required_row_filters(source: dict[str, Any]) -> dict[str, str | int | float]:
    provenance_fields = source.get("source_provenance_fields", [])
    if not isinstance(provenance_fields, list) or any(
        not isinstance(field, str) or not field for field in provenance_fields
    ):
        raise ValueError(f"{source['name']} source_provenance_fields must be non-empty strings")
    if len(set(provenance_fields)) != len(provenance_fields):
        raise ValueError(f"{source['name']} source_provenance_fields must be unique")
    filters = source.get("required_row_filters", {})
    if not isinstance(filters, dict) or any(
        not isinstance(field, str) or not field or not _finite_scalar(value) for field, value in filters.items()
    ):
        raise ValueError(f"{source['name']} required_row_filters must be a finite-scalar dictionary")
    if not set(filters).issubset(provenance_fields):
        raise ValueError(f"{source['name']} required_row_filters must be declared source provenance fields")
    return filters


def _validate_pinned_code_parquet_source(source: dict[str, Any]) -> None:
    """Independently check the immutable script-free CodeParrot subset."""
    expected = {
        "name": "code",
        "upstream_dataset": "codeparrot/github-code-clean",
        "upstream_revision": "c48d40f9e70f0196f8236901ee35807f7d6c44c0",
        "config": "Python-apache-2.0",
        "split": "train",
        "text_field": "code",
        "loader": "pinned_parquet",
        "parquet_glob": "data/train-*.parquet",
        "parquet_shard_count": 880,
        "required_path_suffix": ".py",
        "language_derivation": "path_suffix",
        "derived_language": "Python",
        "required_row_filters": {"license": "apache-2.0"},
    }
    for field, value in expected.items():
        if source.get(field) != value:
            raise ValueError(f"indexed plan pinned CodeParrot Parquet source differs at {field}")
    if not {"repo_name", "path", "language", "license"}.issubset(source.get("source_provenance_fields", [])):
        raise ValueError("indexed plan pinned CodeParrot Parquet source lacks provenance fields")


def _weighted_token_target(total_tokens: int, weight: Decimal, field: str) -> int:
    target = Decimal(total_tokens) * weight
    if target != target.to_integral_value():
        raise ValueError(
            f"{field} produces non-integral token target {target}; choose a planned token budget compatible with weights"
        )
    return int(target)


def _source_config_targets(source: dict[str, Any]) -> list[tuple[str | None, int]]:
    planned_tokens = _positive_int(source.get("planned_tokens"), f"{source.get('name', 'source')}.planned_tokens")
    if "configs" not in source:
        if "config_weights" in source or "planned_tokens_per_config" in source:
            raise ValueError(f"{source['name']} declares configuration weights without configs")
        config = source.get("config")
        if config is not None and (not isinstance(config, str) or not config):
            raise ValueError(f"{source['name']} config must be a non-empty string or null")
        return [(config, planned_tokens)]

    configs = source["configs"]
    if not isinstance(configs, list) or not configs or any(not isinstance(config, str) or not config for config in configs):
        raise ValueError(f"{source['name']} configs must be a non-empty list of non-empty strings")
    if len(set(configs)) != len(configs):
        raise ValueError(f"{source['name']} configs must be unique")
    weights = source.get("config_weights")
    if not isinstance(weights, dict) or set(weights) != set(configs):
        raise ValueError(f"{source['name']} config_weights must name exactly the pinned configs")
    if "planned_tokens_per_config" in source:
        raise ValueError(
            f"{source['name']} uses deprecated planned_tokens_per_config; targets must derive from config_weights"
        )
    decimal_weights = {
        config: _positive_decimal(weights[config], f"{source['name']}.config_weights.{config}") for config in configs
    }
    if sum(decimal_weights.values(), Decimal(0)) != Decimal(1):
        raise ValueError(f"{source['name']} config_weights must sum exactly to one")
    targets = [
        (config, _weighted_token_target(planned_tokens, decimal_weights[config], f"{source['name']}.{config}"))
        for config in configs
    ]
    if sum(target for _, target in targets) != planned_tokens:
        raise ValueError(f"{source['name']} configuration targets do not sum to planned_tokens")
    return targets


def _validate_plan(plan: dict[str, Any]) -> None:
    if plan.get("schema_version") != DATASET_SCHEMA_VERSION:
        raise ValueError(f"plan schema_version must be {DATASET_SCHEMA_VERSION}")
    manifest_id = plan.get("manifest_id")
    if (
        not isinstance(manifest_id, str)
        or not manifest_id
        or manifest_id in {".", ".."}
        or Path(manifest_id).name != manifest_id
    ):
        raise ValueError("plan manifest_id must be a non-path non-empty string")
    lifecycle = plan.get("lifecycle")
    if not isinstance(lifecycle, dict) or lifecycle.get("status") != TRANSITION_SEQUENCE[0]:
        raise ValueError(f"source plan is not {TRANSITION_SEQUENCE[0]}")
    if lifecycle.get("state_machine_version") != LIFECYCLE_VERSION:
        raise ValueError(f"plan lifecycle state_machine_version must be {LIFECYCLE_VERSION}")
    if lifecycle.get("transition_sequence") != list(TRANSITION_SEQUENCE):
        raise ValueError("plan lifecycle transition_sequence is not the AMPA dataset state machine")
    if lifecycle.get("canonical_training_allowed") is not False:
        raise ValueError("planning manifest must forbid canonical training")
    if not plan.get("sequence_construction", {}).get("append_eod"):
        raise ValueError("AMPA baseline requires append_eod=true")
    model = plan.get("model_and_tokenizer")
    if not isinstance(model, dict) or not all(
        isinstance(model.get(field), str) and model[field]
        for field in ("model_id", "revision", "tokenizer_id", "tokenizer_revision", "model_manifest")
    ):
        raise ValueError("plan must bind model, tokenizer, and model-manifest identity")
    if not _is_sha256(model.get("model_manifest_sha256")):
        raise ValueError("plan model_manifest_sha256 must be a SHA256")
    workload = plan.get("initial_workload_plan")
    if not isinstance(workload, dict):
        raise ValueError("plan lacks initial_workload_plan")
    planned_total = _positive_int(workload.get("planned_total_tokens"), "initial_workload_plan.planned_total_tokens")
    sources = plan.get("sources")
    if not isinstance(sources, list) or len(sources) != 4:
        raise ValueError("AMPA realistic mixture must declare exactly four sources")
    source_names: set[str] = set()
    source_weight_total = Decimal(0)
    for source in sources:
        if not isinstance(source, dict):
            raise ValueError("source entry must be an object")
        name = source.get("name")
        if not isinstance(name, str) or not name or name in source_names:
            raise ValueError("source names must be unique non-empty strings")
        source_names.add(name)
        source_weight = _positive_decimal(source.get("weight"), f"{name}.weight")
        source_weight_total += source_weight
        planned_tokens = _positive_int(source.get("planned_tokens"), f"{name}.planned_tokens")
        if planned_tokens != _weighted_token_target(planned_total, source_weight, f"{name}.weight"):
            raise ValueError(f"{name}.planned_tokens must derive exactly from its mixture weight")
        prefix = source.get("indexed_prefix")
        expected_prefix = Path("dataset") / "indexed" / manifest_id
        prefix_path = Path(prefix) if isinstance(prefix, str) else None
        if (
            prefix_path is None
            or prefix_path.is_absolute()
            or ".." in prefix_path.parts
            or not prefix_path.is_relative_to(expected_prefix)
        ):
            raise ValueError(f"{name}.indexed_prefix must be beneath {expected_prefix}")
        _source_config_targets(source)
        _required_row_filters(source)
        if source.get("loader") is not None:
            if source["loader"] != "pinned_parquet":
                raise ValueError(f"{name} has unsupported source loader")
            _validate_pinned_code_parquet_source(source)
    if source_weight_total != Decimal(1):
        raise ValueError("source weights must sum exactly to one")
    expected_data_path = " ".join(
        part for source in sources for part in (str(source["weight"]), source["indexed_prefix"])
    )
    if plan.get("planned_training_data_path") != expected_data_path:
        raise ValueError("planned_training_data_path must derive exactly from source weights and indexed prefixes")


def _load_bound_plan(materialized: dict[str, Any]) -> tuple[Path, dict[str, Any], str]:
    """Load the immutable plan named by the candidate and verify its digest."""
    source_plan = materialized.get("source_plan")
    if not isinstance(source_plan, dict) or set(source_plan) != {"path", "sha256", "manifest_id", "schema_version"}:
        raise ValueError("materialized candidate lacks a strict source_plan binding")
    plan_path_value = source_plan["path"]
    if not isinstance(plan_path_value, str) or not Path(plan_path_value).is_absolute():
        raise ValueError("source_plan.path must be an absolute immutable-plan path")
    plan_path = Path(plan_path_value)
    if not plan_path.is_file():
        raise ValueError(f"source plan is missing: {plan_path}")
    plan_sha256 = sha256(plan_path)
    if source_plan.get("sha256") != plan_sha256:
        raise ValueError("source plan SHA256 does not match the materialized candidate binding")
    lifecycle = materialized.get("lifecycle")
    if not isinstance(lifecycle, dict) or lifecycle.get("source_plan_sha256") != plan_sha256:
        raise ValueError("materialized lifecycle is not bound to the immutable source plan")
    try:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"source plan is invalid JSON: {error}") from error
    if not isinstance(plan, dict):
        raise ValueError("source plan must contain an object")
    _validate_plan(plan)
    if source_plan["manifest_id"] != plan["manifest_id"] or source_plan["schema_version"] != plan["schema_version"]:
        raise ValueError("source_plan identity fields do not match the immutable plan")
    immutable_fields = (
        "schema_version",
        "manifest_id",
        "model_and_tokenizer",
        "sampling",
        "sequence_construction",
        "initial_workload_plan",
        "sources",
        "planned_training_data_path",
        "materialization_requirements",
    )
    for field in immutable_fields:
        if materialized.get(field) != plan.get(field):
            raise ValueError(f"materialized candidate mutates immutable source-plan field {field!r}")
    return plan_path, plan, plan_sha256


def _model_manifest_vocab_size(plan_path: Path, plan: dict[str, Any]) -> int:
    """Verify the model manifest hash and return its pinned vocabulary assertion."""
    if plan_path.parent.name != "manifests" or plan_path.parent.parent.name != "dataset":
        raise ValueError("source plan must reside beneath dataset/manifests")
    workspace_root = plan_path.parents[2]
    model = plan["model_and_tokenizer"]
    model_reference = Path(model["model_manifest"])
    if model_reference.is_absolute() or ".." in model_reference.parts:
        raise ValueError("model_manifest must be a workspace-relative path")
    model_path = workspace_root / model_reference
    if not model_path.is_file() or sha256(model_path) != model["model_manifest_sha256"]:
        raise ValueError("pinned model manifest is missing or its SHA256 differs from the source plan")
    try:
        model_manifest = json.loads(model_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"pinned model manifest is invalid JSON: {error}") from error
    if (
        not isinstance(model_manifest, dict)
        or not isinstance(model_manifest.get("lifecycle"), dict)
        or model_manifest["lifecycle"].get("status") != "pinned"
    ):
        raise ValueError("model manifest is not pinned")
    hf = model_manifest.get("hugging_face")
    if not isinstance(hf, dict) or any(
        hf.get(model_key) != model[plan_key]
        for model_key, plan_key in (
            ("model_id", "model_id"),
            ("revision", "revision"),
            ("tokenizer_id", "tokenizer_id"),
            ("tokenizer_revision", "tokenizer_revision"),
        )
    ):
        raise ValueError("model manifest Hugging Face identity does not match the immutable source plan")
    vocab_size = model_manifest.get("architecture_assertions", {}).get("vocab_size")
    return _positive_int(vocab_size, "model manifest architecture_assertions.vocab_size")


def _build_tokenizer(plan: dict[str, Any]):
    """Construct the pinned Bridge tokenizer independently of materialization."""
    from megatron.bridge.training.tokenizers.config import TokenizerConfig
    from megatron.bridge.training.tokenizers.tokenizer import build_tokenizer

    model = plan["model_and_tokenizer"]
    return build_tokenizer(
        TokenizerConfig(
            tokenizer_type="HuggingFaceTokenizer",
            tokenizer_model=model["tokenizer_id"],
            hf_tokenizer_kwargs={
                "revision": model["tokenizer_revision"],
                "use_fast": True,
                "include_special_tokens": True,
            },
        )
    )


def _indexed_dtype_name(vocab_size: int) -> str:
    from megatron.core.datasets.indexed_dataset import DType

    return DType.optimal_dtype(vocab_size).__name__


def _validate_tokenizer_binding(plan_path: Path, plan: dict[str, Any], materialized: dict[str, Any]) -> dict[str, Any]:
    model_vocab_size = _model_manifest_vocab_size(plan_path, plan)
    tokenizer = _build_tokenizer(plan)
    eod = getattr(tokenizer, "eod", None)
    tokenizer_vocab_size = getattr(tokenizer, "vocab_size", None)
    if isinstance(eod, bool) or not isinstance(eod, int) or isinstance(tokenizer_vocab_size, bool) or not isinstance(
        tokenizer_vocab_size, int
    ):
        raise ValueError("independently rebuilt Bridge tokenizer does not expose integer eod and vocab_size")
    if not 0 <= eod < tokenizer_vocab_size <= model_vocab_size:
        raise ValueError("Bridge tokenizer EOD/vocabulary is outside the pinned model embedding capacity")
    expected = {
        "eod": eod,
        "vocab_size": tokenizer_vocab_size,
        "indexed_dtype": _indexed_dtype_name(tokenizer_vocab_size),
        "include_special_tokens": True,
        "model_manifest_sha256": plan["model_and_tokenizer"]["model_manifest_sha256"],
    }
    observed = materialized.get("tokenizer_observed")
    if not isinstance(observed, dict) or observed != expected:
        raise ValueError("materialized tokenizer/EOD/dtype evidence disagrees with the independently rebuilt tokenizer")
    return expected


def _config_key(config: str | None) -> str:
    return config if config is not None else "default"


def _ledger_summary(path: Path, source: dict[str, Any]) -> dict[str, Any]:
    """Stream and validate the selected-document ledger against its source schema."""
    allowed = {
        "source_name",
        "upstream_dataset",
        "upstream_revision",
        "config",
        "split",
        "stable_document_identity",
        "source_provenance",
        "text_sha256",
        "token_count_including_eod",
    }
    expected_targets = _source_config_targets(source)
    expected_configs = {config for config, _ in expected_targets}
    per_config = {
        _config_key(config): {"documents": 0, "tokens_including_eod": 0} for config, _ in expected_targets
    }
    identity_fields = source.get("stable_document_identity")
    provenance_fields = source.get("source_provenance_fields", [])
    required_filters = _required_row_filters(source)
    if not isinstance(identity_fields, list) or not identity_fields or any(
        not isinstance(field, str) or not field for field in identity_fields
    ):
        raise ValueError(f"{source['name']} has invalid stable_document_identity plan fields")
    if not isinstance(provenance_fields, list) or any(not isinstance(field, str) or not field for field in provenance_fields):
        raise ValueError(f"{source['name']} has invalid source_provenance_fields plan fields")
    documents = 0
    tokens = 0
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ValueError(f"ledger has a blank record at {path}:{line_number}")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"ledger has invalid JSON at {path}:{line_number}: {error}") from error
            if not isinstance(record, dict) or set(record) != allowed:
                raise ValueError(f"ledger schema differs from the selected-document contract at {path}:{line_number}")
            if (
                record["source_name"] != source["name"]
                or record["upstream_dataset"] != source["upstream_dataset"]
                or record["upstream_revision"] != source["upstream_revision"]
                or record["split"] != source["split"]
                or record["config"] not in expected_configs
            ):
                raise ValueError(f"ledger record disagrees with pinned source provenance at {path}:{line_number}")
            identity = record["stable_document_identity"]
            if not isinstance(identity, dict) or set(identity) != set(identity_fields) or any(
                not isinstance(value, str) or not value for value in identity.values()
            ):
                raise ValueError(f"ledger stable_document_identity schema mismatch at {path}:{line_number}")
            text_sha256 = record["text_sha256"]
            if not _is_sha256(text_sha256):
                raise ValueError(f"ledger record lacks a lower-case text SHA256 at {path}:{line_number}")
            if identity.get("sha256(text)") not in (None, text_sha256):
                raise ValueError(f"ledger text identity/hash mismatch at {path}:{line_number}")
            provenance = record["source_provenance"]
            if not isinstance(provenance, dict) or set(provenance) != set(provenance_fields) or any(
                isinstance(value, bool)
                or not isinstance(value, (str, int, float))
                or (isinstance(value, str) and not value)
                or (isinstance(value, float) and not math.isfinite(value))
                for value in provenance.values()
            ):
                raise ValueError(f"ledger source_provenance schema mismatch at {path}:{line_number}")
            if any(provenance[field] != expected for field, expected in required_filters.items()):
                raise ValueError(f"ledger record violates required_row_filters at {path}:{line_number}")
            if source.get("loader") == "pinned_parquet":
                suffix = source["required_path_suffix"]
                if (
                    not identity["path"].endswith(suffix)
                    or provenance["path"] != identity["path"]
                    or provenance["repo_name"] != identity["repo_name"]
                    or provenance["language"] != source["derived_language"]
                ):
                    raise ValueError(f"ledger CodeParrot path-derived language disagrees with source plan at {path}:{line_number}")
            length = record["token_count_including_eod"]
            if isinstance(length, bool) or not isinstance(length, int) or length <= 1:
                raise ValueError(f"ledger record has invalid token count at {path}:{line_number}")
            key = _config_key(record["config"])
            per_config[key]["documents"] += 1
            per_config[key]["tokens_including_eod"] += length
            documents += 1
            tokens += length
    return {"documents": documents, "tokens_including_eod": tokens, "per_config": per_config}


def _validate_artifact_schema(
    artifact: dict[str, Any], source: dict[str, Any], expected_prefix: Path, expected_ledger: Path
) -> None:
    required = {
        "name",
        "indexed_prefix",
        "idx_path",
        "bin_path",
        "idx_bytes",
        "bin_bytes",
        "idx_sha256",
        "bin_sha256",
        "ledger_path",
        "ledger_sha256",
        "documents",
        "tokens_including_eod",
        "per_config",
        "streaming_summary",
    }
    if source.get("loader") == "pinned_parquet":
        required.add("source_parquet_shards")
    if set(artifact) != required:
        raise ValueError(f"artifact schema differs from the materialization contract: {source['name']}")
    if source.get("loader") == "pinned_parquet":
        shards = artifact["source_parquet_shards"]
        if not isinstance(shards, list) or not shards or len(shards) > source["parquet_shard_count"]:
            raise ValueError("pinned code artifact has no valid consumed-shard list")
        for index, shard in enumerate(shards):
            expected_name = f"data/train-{index:05d}-of-{source['parquet_shard_count']:05d}.parquet"
            if (
                not isinstance(shard, dict)
                or set(shard) != {"filename", "bytes", "sha256"}
                or shard["filename"] != expected_name
                or type(shard["bytes"]) is not int
                or shard["bytes"] <= 0
                or not _is_sha256(shard["sha256"])
            ):
                raise ValueError(f"pinned code artifact consumed-shard evidence is invalid at {index}")
    if artifact["name"] != source["name"] or artifact["indexed_prefix"] != str(expected_prefix):
        raise ValueError(f"artifact identity/prefix differs from the immutable plan: {source['name']}")
    if artifact["idx_path"] != f"{expected_prefix}.idx" or artifact["bin_path"] != f"{expected_prefix}.bin":
        raise ValueError(f"artifact path fields differ from its indexed prefix: {source['name']}")
    if not all(_is_sha256(artifact[field]) for field in ("idx_sha256", "bin_sha256", "ledger_sha256")):
        raise ValueError(f"artifact has invalid SHA256 evidence: {source['name']}")
    for field in ("idx_bytes", "bin_bytes", "documents", "tokens_including_eod"):
        _positive_int(artifact[field], f"{source['name']}.{field}")
    if artifact["ledger_path"] != str(expected_ledger):
        raise ValueError(f"artifact ledger path is outside its published indexed build: {source['name']}")


def _validate_per_config_accounting(artifact: dict[str, Any], source: dict[str, Any], ledger: dict[str, Any]) -> None:
    expected = {_config_key(config): target for config, target in _source_config_targets(source)}
    per_config = artifact["per_config"]
    if not isinstance(per_config, dict) or set(per_config) != set(expected):
        raise ValueError(f"artifact per_config keys differ from the pinned source configs: {source['name']}")
    if ledger["per_config"].keys() != expected.keys():
        raise ValueError(f"ledger configuration keys differ from the pinned source configs: {source['name']}")
    documents = 0
    tokens = 0
    records_seen = 0
    records_rejected = 0
    records_skipped_empty = 0
    required = {
        "target_tokens",
        "documents",
        "tokens_including_eod",
        "records_seen",
        "records_rejected_by_filter",
        "records_skipped_empty",
    }
    for config, target in expected.items():
        observed = per_config[config]
        if not isinstance(observed, dict) or set(observed) != required:
            raise ValueError(f"artifact per_config schema mismatch: {source['name']}/{config}")
        if _positive_int(observed["target_tokens"], f"{source['name']}/{config}.target_tokens") != target:
            raise ValueError(f"artifact per_config target ignores pinned configuration weights: {source['name']}/{config}")
        selected_documents = _positive_int(observed["documents"], f"{source['name']}/{config}.documents")
        selected_tokens = _positive_int(observed["tokens_including_eod"], f"{source['name']}/{config}.tokens")
        seen = _positive_int(observed["records_seen"], f"{source['name']}/{config}.records_seen")
        rejected = _nonnegative_int(observed["records_rejected_by_filter"], f"{source['name']}/{config}.rejected")
        skipped_empty = _nonnegative_int(observed["records_skipped_empty"], f"{source['name']}/{config}.empty")
        if selected_tokens < target or seen != selected_documents + rejected + skipped_empty:
            raise ValueError(f"artifact per_config selection accounting is invalid: {source['name']}/{config}")
        if ledger["per_config"][config] != {
            "documents": selected_documents,
            "tokens_including_eod": selected_tokens,
        }:
            raise ValueError(f"artifact per_config disagrees with streaming ledger: {source['name']}/{config}")
        documents += selected_documents
        tokens += selected_tokens
        records_seen += seen
        records_rejected += rejected
        records_skipped_empty += skipped_empty
    if (documents, tokens) != (artifact["documents"], artifact["tokens_including_eod"]):
        raise ValueError(f"artifact source totals disagree with per_config totals: {source['name']}")
    summary = artifact["streaming_summary"]
    if not isinstance(summary, dict) or summary != {
        "records_seen": records_seen,
        "records_rejected_by_filter": records_rejected,
        "records_skipped_empty": records_skipped_empty,
    }:
        raise ValueError(f"artifact filtering summary disagrees with per_config accounting: {source['name']}")


def _validate_component(artifact: dict[str, Any], source: dict[str, Any], eod: int, indexed_dtype: str) -> dict[str, Any]:
    from megatron.core.datasets.indexed_dataset import IndexedDataset

    prefix = Path(artifact["indexed_prefix"])
    idx_path = Path(f"{prefix}.idx")
    bin_path = Path(f"{prefix}.bin")
    ledger_path = Path(artifact["ledger_path"])
    if not idx_path.is_file() or not bin_path.is_file() or not ledger_path.is_file():
        raise ValueError(f"indexed component or selection ledger is missing: {prefix}")
    if (idx_path.stat().st_size, bin_path.stat().st_size) != (artifact["idx_bytes"], artifact["bin_bytes"]):
        raise ValueError(f"indexed artifact byte size mismatch: {prefix}")
    if (sha256(idx_path), sha256(bin_path)) != (artifact["idx_sha256"], artifact["bin_sha256"]):
        raise ValueError(f"indexed artifact hash mismatch: {prefix}")
    if sha256(ledger_path) != artifact["ledger_sha256"]:
        raise ValueError(f"selection ledger hash mismatch: {ledger_path}")
    ledger = _ledger_summary(ledger_path, source)
    _validate_per_config_accounting(artifact, source, ledger)
    dataset = IndexedDataset(str(prefix))
    if getattr(dataset.index.dtype, "__name__", None) != indexed_dtype:
        raise ValueError(f"indexed dtype differs from independently rebuilt tokenizer expectation: {prefix}")
    document_indices = [int(index) for index in dataset.document_indices]
    expected_document_indices = list(range(len(dataset) + 1))
    if document_indices != expected_document_indices:
        raise ValueError(f"indexed document boundaries are not one sequence per selected document: {prefix}")
    if len(dataset) != ledger["documents"] or len(document_indices) != ledger["documents"] + 1:
        raise ValueError(f"indexed document count differs from the ledger: {prefix}")
    observed_tokens = 0
    for sequence_index in range(len(dataset)):
        sequence = dataset[sequence_index]
        if len(sequence) <= 1 or int(sequence[-1]) != eod:
            raise ValueError(f"indexed document lacks terminal EOD: {prefix} item {sequence_index}")
        observed_tokens += len(sequence)
    if observed_tokens != sum(int(length) for length in dataset.sequence_lengths):
        raise ValueError(f"indexed sequence lengths disagree with loaded token count: {prefix}")
    if (ledger["tokens_including_eod"], observed_tokens) != (
        artifact["tokens_including_eod"],
        artifact["tokens_including_eod"],
    ):
        raise ValueError(f"ledger/indexed token count mismatch: {prefix}")
    return {
        "name": artifact["name"],
        "documents": ledger["documents"],
        "tokens_including_eod": observed_tokens,
        "sequence_count": len(dataset),
        "document_index_boundaries_validated": True,
        "terminal_eod_validated": True,
        "indexed_dtype": indexed_dtype,
    }


def _lifecycle(status: str, **evidence: Any) -> dict[str, Any]:
    return {
        "status": status,
        "state_machine_version": LIFECYCLE_VERSION,
        "transition_sequence": list(TRANSITION_SEQUENCE),
        "canonical_training_allowed": False,
        **evidence,
    }


def _write_json_staged(path: Path, payload: dict[str, Any]) -> Path:
    staged = Path(f"{path}.partial-{os.getpid()}")
    if staged.exists():
        raise FileExistsError(f"staged indexed validation manifest already exists: {staged}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with staged.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return staged


def validate(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_manifest.exists():
        raise FileExistsError(f"refusing to overwrite indexed validation manifest: {args.output_manifest}")
    materialized = json.loads(args.materialized_manifest.read_text(encoding="utf-8"))
    if not isinstance(materialized, dict):
        raise ValueError("materialized manifest must contain an object")
    lifecycle = materialized.get("lifecycle")
    if not isinstance(lifecycle, dict) or lifecycle.get("status") != TRANSITION_SEQUENCE[1]:
        raise ValueError("input is not a materialized data-review candidate")
    if lifecycle.get("state_machine_version") != LIFECYCLE_VERSION or lifecycle.get("transition_sequence") != list(
        TRANSITION_SEQUENCE
    ):
        raise ValueError("materialized candidate does not use the AMPA dataset lifecycle state machine")
    plan_path, plan, plan_sha256 = _load_bound_plan(materialized)
    tokenizer = _validate_tokenizer_binding(plan_path, plan, materialized)
    workspace_root = plan_path.parents[2]
    artifacts = materialized.get("materialized_artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != len(plan["sources"]):
        raise ValueError("materialized artifact count differs from the immutable source plan")
    artifacts_by_name: dict[str, dict[str, Any]] = {}
    for artifact in artifacts:
        if not isinstance(artifact, dict) or not isinstance(artifact.get("name"), str) or artifact["name"] in artifacts_by_name:
            raise ValueError("materialized artifacts must have unique source names")
        artifacts_by_name[artifact["name"]] = artifact
    if set(artifacts_by_name) != {source["name"] for source in plan["sources"]}:
        raise ValueError("materialized artifacts do not match immutable source names")
    results = []
    data_path_parts: list[str] = []
    for source in plan["sources"]:
        artifact = artifacts_by_name[source["name"]]
        expected_prefix = workspace_root / source["indexed_prefix"]
        expected_ledger = (
            workspace_root / "dataset" / "indexed" / plan["manifest_id"] / "ledgers" / f"{source['name']}.jsonl"
        )
        _validate_artifact_schema(artifact, source, expected_prefix, expected_ledger)
        if artifact["tokens_including_eod"] < source["planned_tokens"]:
            raise ValueError(f"{artifact['name']} is below its source-specific planned token target")
        results.append(_validate_component(artifact, source, tokenizer["eod"], tokenizer["indexed_dtype"]))
        data_path_parts.extend((str(source["weight"]), artifact["indexed_prefix"]))
    expected_data_path = " ".join(data_path_parts)
    if materialized.get("materialized_training_data_path") != expected_data_path:
        raise ValueError("materialized training data path differs from the immutable source plan and artifact prefixes")
    candidate = copy.deepcopy(materialized)
    candidate["lifecycle"] = _lifecycle(
        "indexed_validated_pending_cache",
        source_plan_sha256=plan_sha256,
        materialized_manifest_sha256=sha256(args.materialized_manifest),
    )
    candidate["indexed_validation"] = {
        "components": results,
        "total_tokens": sum(row["tokens_including_eod"] for row in results),
        "immutable_plan_sha256": plan_sha256,
        "tokenizer_independently_validated": tokenizer,
    }
    staged = _write_json_staged(args.output_manifest, candidate)
    os.replace(staged, args.output_manifest)
    return candidate


def main() -> None:
    result = validate(parse_args())
    print(json.dumps(result["indexed_validation"], sort_keys=True))


if __name__ == "__main__":
    main()
