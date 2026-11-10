#!/usr/bin/env python3
"""Stream the pinned AMPA reservoirs into normal Megatron indexed components.

This is a one-time Slurm preprocessing program.  It retains no raw text: the
only durable selection evidence is a source-specific ledger of stable document
identity/provenance, text hash, and token count.  It never modifies the
planning lock; it emits a separate candidate manifest for independent review.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import tempfile
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterator


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
    parser.add_argument("--plan-manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--build-id", required=True, help="Unique Slurm job id used only in candidate paths.")
    return parser.parse_args()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _required_string(row: dict[str, Any], field: str, source_name: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{source_name} record lacks required non-empty {field!r}")
    return value


def _identity(row: dict[str, Any], fields: list[str], text: str, source_name: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for field in fields:
        if field == "sha256(text)":
            result[field] = sha256_bytes(text.encode("utf-8"))
        else:
            result[field] = _required_string(row, field, source_name)
    return result


def _provenance(row: dict[str, Any], fields: list[str], source_name: str) -> dict[str, str | int | float]:
    """Capture exactly the pinned provenance fields, never a permissive row dump."""
    result: dict[str, str | int | float] = {}
    for field in fields:
        value = row.get(field)
        if isinstance(value, bool) or not isinstance(value, (str, int, float)):
            raise ValueError(f"{source_name} provenance field {field!r} is not a finite JSON scalar")
        if isinstance(value, str) and not value:
            raise ValueError(f"{source_name} provenance field {field!r} is empty")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"{source_name} provenance field {field!r} is not finite")
        result[field] = value
    return result


def _finite_scalar(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (str, int, float))
        and (not isinstance(value, str) or bool(value))
        and (not isinstance(value, float) or math.isfinite(value))
    )


def _required_row_filters(source: dict[str, Any]) -> dict[str, str | int | float]:
    """Validate the immutable filtering rule before any reservoir rows are used."""
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


def _row_matches_required_filters(
    row: dict[str, Any], source: dict[str, Any], filters: dict[str, str | int | float] | None = None
) -> bool:
    """Return whether a streamed row belongs to this source's pinned subset."""
    rules = _required_row_filters(source) if filters is None else filters
    return all(row.get(field) == expected for field, expected in rules.items())


def _pinned_code_parquet_uri(source: dict[str, Any]) -> str:
    """Validate the one script-free view of the pinned CodeParrot reservoir."""
    expected = {
        "name": "code",
        "upstream_dataset": "codeparrot/github-code-clean",
        "upstream_revision": "c48d40f9e70f0196f8236901ee35807f7d6c44c0",
        "config": "Python-apache-2.0",  # Logical subset label, not a Parquet builder config.
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
            raise ValueError(f"pinned code Parquet source differs at {field}")
    if not {"repo_name", "path", "language", "license"}.issubset(
        source.get("source_provenance_fields", [])
    ):
        raise ValueError("pinned code Parquet source lacks required provenance fields")
    return (
        f"hf://datasets/{source['upstream_dataset']}@{source['upstream_revision']}/"
        f"{source['parquet_glob']}"
    )


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


def _weighted_token_target(total_tokens: int, weight: Decimal, field: str) -> int:
    target = Decimal(total_tokens) * weight
    if target != target.to_integral_value():
        raise ValueError(
            f"{field} produces non-integral token target {target}; choose a planned token budget compatible with weights"
        )
    return int(target)


def _source_configs(source: dict[str, Any]) -> list[tuple[str | None, int]]:
    """Derive configuration targets from the immutable source weights."""
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
    """Validate only immutable planning fields used by materialization."""
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
        raise ValueError(f"plan manifest is not {TRANSITION_SEQUENCE[0]}")
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
        for field in ("model_id", "revision", "tokenizer_id", "tokenizer_revision", "model_manifest", "model_manifest_sha256")
    ):
        raise ValueError("plan must bind model, tokenizer, and model-manifest identity")
    if len(model["model_manifest_sha256"]) != 64:
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
        expected_tokens = _weighted_token_target(planned_total, source_weight, f"{name}.weight")
        if planned_tokens != expected_tokens:
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
        _source_configs(source)
        _required_row_filters(source)
        if source.get("loader") is not None:
            if source["loader"] != "pinned_parquet":
                raise ValueError(f"{name} has unsupported source loader {source['loader']!r}")
            _pinned_code_parquet_uri(source)
    if source_weight_total != Decimal(1):
        raise ValueError("source weights must sum exactly to one")
    expected_data_path = " ".join(
        part for source in sources for part in (str(source["weight"]), source["indexed_prefix"])
    )
    if plan.get("planned_training_data_path") != expected_data_path:
        raise ValueError("planned_training_data_path must derive exactly from source weights and indexed prefixes")


def _validate_model_manifest_binding(plan: dict[str, Any], workspace_root: Path) -> int:
    """Fail before streaming if the tokenizer/model assertion lock is no longer intact."""
    model = plan["model_and_tokenizer"]
    reference = Path(model["model_manifest"])
    if reference.is_absolute() or ".." in reference.parts:
        raise ValueError("model_manifest must be a workspace-relative path")
    model_path = workspace_root / reference
    if not model_path.is_file() or sha256_file(model_path) != model["model_manifest_sha256"]:
        raise ValueError("pinned model manifest is missing or its SHA256 differs from the data plan")
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
        raise ValueError("model manifest Hugging Face identity does not match the data plan")
    return _positive_int(
        model_manifest.get("architecture_assertions", {}).get("vocab_size"),
        "model manifest architecture_assertions.vocab_size",
    )


def _stream_rows(
    source: dict[str, Any], config: str | None, shard_evidence: list[dict[str, Any]] | None = None
) -> Iterator[dict[str, Any]]:
    if source.get("loader") == "pinned_parquet":
        # The legacy CodeParrot loader derives language from filename suffix,
        # not the physical Parquet language column. Read one completed pinned
        # shard at a time; remote Parquet streaming aborted at Python shutdown.
        if config != source["config"]:
            raise ValueError("pinned CodeParrot logical config differs from its plan")
        _pinned_code_parquet_uri(source)
        from huggingface_hub import hf_hub_download
        from pyarrow.parquet import ParquetFile

        # Do not trust inherited TMPDIR here: on this cluster it may point to
        # shared storage. Shard downloads must be bounded and node-local.
        temporary_parent = Path(os.environ.get("SLURM_TMPDIR") or "/tmp")
        if not temporary_parent.is_absolute() or not temporary_parent.is_dir():
            raise ValueError("pinned CodeParrot shards require a node-local temporary directory")
        for index in range(source["parquet_shard_count"]):
            filename = f"data/train-{index:05d}-of-{source['parquet_shard_count']:05d}.parquet"
            with tempfile.TemporaryDirectory(prefix="ampa-codeparrot-", dir=temporary_parent) as cache_dir:
                local_path = Path(
                    hf_hub_download(
                        repo_id=source["upstream_dataset"],
                        repo_type="dataset",
                        revision=source["upstream_revision"],
                        filename=filename,
                        cache_dir=cache_dir,
                    )
                )
                if shard_evidence is not None:
                    shard_evidence.append(
                        {"filename": filename, "bytes": local_path.stat().st_size, "sha256": sha256_file(local_path)}
                    )
                parquet_file = ParquetFile(local_path)
                try:
                    for batch in parquet_file.iter_batches(batch_size=1024, use_threads=False):
                        for row in batch.to_pylist():
                            if row.get("license") != "apache-2.0":
                                continue
                            if not isinstance(row.get("path"), str):
                                raise ValueError("pinned CodeParrot Parquet row lacks a path")
                            if not row["path"].endswith(source["required_path_suffix"]):
                                continue
                            logical_row = dict(row)
                            logical_row["language"] = source["derived_language"]
                            yield logical_row
                finally:
                    parquet_file.close()
        return

    from datasets import load_dataset

    kwargs: dict[str, Any] = {
        "path": source["upstream_dataset"],
        "split": source["split"],
        "streaming": True,
        "revision": source["upstream_revision"],
    }
    if config is not None:
        kwargs["name"] = config
    yield from load_dataset(**kwargs)


def _build_tokenizer(manifest: dict[str, Any]):
    from megatron.bridge.training.tokenizers.config import TokenizerConfig
    from megatron.bridge.training.tokenizers.tokenizer import build_tokenizer

    model = manifest["model_and_tokenizer"]
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


def _materialize_source(
    *,
    source: dict[str, Any],
    tokenizer: Any,
    dtype: Any,
    workspace_root: Path,
    candidate_root: Path,
    final_root: Path,
) -> dict[str, Any]:
    from megatron.core.datasets.indexed_dataset import IndexedDatasetBuilder
    import torch

    prefix = workspace_root / source["indexed_prefix"]
    relative_prefix = prefix.relative_to(final_root)
    candidate_prefix = candidate_root / relative_prefix
    candidate_prefix.parent.mkdir(parents=True, exist_ok=True)
    candidate_bin = Path(f"{candidate_prefix}.bin")
    candidate_idx = Path(f"{candidate_prefix}.idx")
    candidate_ledger = candidate_root / "ledgers" / f"{source['name']}.jsonl"
    candidate_ledger.parent.mkdir(parents=True, exist_ok=True)
    if candidate_bin.exists() or candidate_idx.exists() or candidate_ledger.exists():
        raise FileExistsError(f"candidate already contains source output: {source['name']}")

    final_bin = Path(f"{prefix}.bin")
    final_idx = Path(f"{prefix}.idx")
    ledger = final_root / "ledgers" / f"{source['name']}.jsonl"
    builder = IndexedDatasetBuilder(str(candidate_bin), dtype=dtype)
    total_tokens = 0
    documents = 0
    per_config: dict[str, dict[str, int]] = {}
    records_seen = 0
    records_rejected_by_filter = 0
    records_skipped_empty = 0
    required_filters = _required_row_filters(source)
    shard_evidence: list[dict[str, Any]] = []
    # Candidate outputs deliberately remain beneath candidate_root until every
    # source succeeds. A failed source therefore never publishes a final prefix.
    with candidate_ledger.open("x", encoding="utf-8") as ledger_stream:
        for config, target_tokens in _source_configs(source):
            key = config if config is not None else "default"
            selected_tokens = 0
            selected_documents = 0
            config_records_seen = 0
            config_rejected = 0
            config_skipped_empty = 0
            row_stream = _stream_rows(source, config, shard_evidence)
            for row in row_stream:
                if not isinstance(row, dict):
                    raise ValueError(f"{source['name']} yielded a non-dictionary record")
                records_seen += 1
                config_records_seen += 1
                if not _row_matches_required_filters(row, source, required_filters):
                    records_rejected_by_filter += 1
                    config_rejected += 1
                    continue
                text = _required_string(row, source["text_field"], source["name"])
                # Installed MegatronTokenizerText exposes tokenize(), which
                # delegates to its HF text_to_ids implementation.
                token_ids = list(tokenizer.tokenize(text))
                if not token_ids:
                    records_skipped_empty += 1
                    config_skipped_empty += 1
                    continue
                token_ids.append(int(tokenizer.eod))
                builder.add_document(torch.tensor(token_ids, dtype=torch.int64), [len(token_ids)])
                identity = _identity(row, source["stable_document_identity"], text, source["name"])
                record = {
                    "source_name": source["name"],
                    "upstream_dataset": source["upstream_dataset"],
                    "upstream_revision": source["upstream_revision"],
                    "config": config,
                    "split": source["split"],
                    "stable_document_identity": identity,
                    "source_provenance": _provenance(
                        row, source.get("source_provenance_fields", []), source["name"]
                    ),
                    "text_sha256": sha256_bytes(text.encode("utf-8")),
                    "token_count_including_eod": len(token_ids),
                }
                ledger_stream.write(json.dumps(record, sort_keys=True) + "\n")
                total_tokens += len(token_ids)
                documents += 1
                selected_tokens += len(token_ids)
                selected_documents += 1
                if selected_tokens >= target_tokens:
                    break
            close = getattr(row_stream, "close", None)
            if callable(close):
                close()
            if selected_tokens < target_tokens:
                raise ValueError(
                    f"{source['name']}/{key} stream ended at {selected_tokens} tokens; "
                    f"planned target is {target_tokens}"
                )
            per_config[key] = {
                "target_tokens": target_tokens,
                "documents": selected_documents,
                "tokens_including_eod": selected_tokens,
                "records_seen": config_records_seen,
                "records_rejected_by_filter": config_rejected,
                "records_skipped_empty": config_skipped_empty,
            }
    builder.finalize(str(candidate_idx))
    result = {
        "name": source["name"],
        "indexed_prefix": str(prefix),
        "idx_path": str(final_idx),
        "bin_path": str(final_bin),
        "idx_bytes": candidate_idx.stat().st_size,
        "bin_bytes": candidate_bin.stat().st_size,
        "idx_sha256": sha256_file(candidate_idx),
        "bin_sha256": sha256_file(candidate_bin),
        "ledger_path": str(ledger),
        "ledger_sha256": sha256_file(candidate_ledger),
        "documents": documents,
        "tokens_including_eod": total_tokens,
        "per_config": per_config,
        "streaming_summary": {
            "records_seen": records_seen,
            "records_rejected_by_filter": records_rejected_by_filter,
            "records_skipped_empty": records_skipped_empty,
        },
    }
    if source.get("loader") == "pinned_parquet":
        result["source_parquet_shards"] = shard_evidence
    return result


def _lifecycle(status: str, **evidence: Any) -> dict[str, Any]:
    return {
        "status": status,
        "state_machine_version": LIFECYCLE_VERSION,
        "transition_sequence": list(TRANSITION_SEQUENCE),
        "canonical_training_allowed": False,
        **evidence,
    }


def _promote_candidate_root(candidate_root: Path, final_root: Path) -> None:
    """Atomically publish all indexed components only after every source succeeds."""
    if final_root.exists():
        raise FileExistsError(f"refusing to overwrite published indexed build: {final_root}")
    final_root.parent.mkdir(parents=True, exist_ok=True)
    os.replace(candidate_root, final_root)


def _write_json_staged(path: Path, payload: dict[str, Any], build_id: str) -> Path:
    staged = Path(f"{path}.partial-{build_id}")
    if staged.exists():
        raise FileExistsError(f"staged materialized manifest already exists: {staged}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with staged.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return staged


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_manifest.exists():
        raise FileExistsError(f"refusing to overwrite materialized manifest: {args.output_manifest}")
    plan = json.loads(args.plan_manifest.read_text(encoding="utf-8"))
    _validate_plan(plan)
    if (
        not isinstance(args.build_id, str)
        or not args.build_id
        or args.build_id in {".", ".."}
        or Path(args.build_id).name != args.build_id
    ):
        raise ValueError("build_id must be a non-path non-empty string")
    plan_sha256 = sha256_file(args.plan_manifest)
    model_vocab_size = _validate_model_manifest_binding(plan, args.workspace_root)
    tokenizer = _build_tokenizer(plan)
    eod = getattr(tokenizer, "eod", None)
    vocab_size = getattr(tokenizer, "vocab_size", None)
    if not isinstance(eod, int) or not isinstance(vocab_size, int):
        raise ValueError("pinned Bridge tokenizer does not expose integer eod and vocab_size")
    # Qwen's model embedding table includes reserved/padded rows.  The
    # tokenizer may have fewer IDs; every emitted ID must fit both bounds.
    if not 0 <= eod < vocab_size <= model_vocab_size:
        raise ValueError(
            f"pinned Bridge tokenizer EOD/vocabulary is outside model embedding capacity: "
            f"eod={eod}, tokenizer_vocab_size={vocab_size}, model_vocab_size={model_vocab_size}"
        )
    from megatron.core.datasets.indexed_dataset import DType

    dtype = DType.optimal_dtype(vocab_size)
    final_root = args.workspace_root / "dataset" / "indexed" / plan["manifest_id"]
    candidate_root = final_root.parent / ".candidates" / plan["manifest_id"] / args.build_id
    if final_root.exists():
        raise FileExistsError(f"refusing to overwrite published indexed build: {final_root}")
    if candidate_root.exists():
        raise FileExistsError(f"candidate build directory already exists: {candidate_root}")
    candidate_root.mkdir(parents=True, exist_ok=False)
    artifacts = [
        _materialize_source(
            source=source,
            tokenizer=tokenizer,
            dtype=dtype,
            workspace_root=args.workspace_root,
            candidate_root=candidate_root,
            final_root=final_root,
        )
        for source in plan["sources"]
    ]
    data_path_parts: list[str] = []
    for source, artifact in zip(plan["sources"], artifacts):
        data_path_parts.extend((str(source["weight"]), artifact["indexed_prefix"]))
    candidate = copy.deepcopy(plan)
    candidate["lifecycle"] = _lifecycle("materialized_pending_data_review", source_plan_sha256=plan_sha256)
    candidate["source_plan"] = {
        "path": str(args.plan_manifest.resolve()),
        "sha256": plan_sha256,
        "manifest_id": plan["manifest_id"],
        "schema_version": plan["schema_version"],
    }
    candidate["tokenizer_observed"] = {
        "eod": eod,
        "vocab_size": vocab_size,
        "indexed_dtype": dtype.__name__,
        "include_special_tokens": True,
        "model_manifest_sha256": plan["model_and_tokenizer"]["model_manifest_sha256"],
    }
    candidate["materialized_training_data_path"] = " ".join(data_path_parts)
    candidate["materialized_artifacts"] = artifacts
    staged_manifest = _write_json_staged(args.output_manifest, candidate, args.build_id)
    _promote_candidate_root(candidate_root, final_root)
    os.replace(staged_manifest, args.output_manifest)
    return candidate


def main() -> None:
    args = parse_args()
    result = materialize(args)
    print(
        json.dumps(
            {
                "manifest_id": result["manifest_id"],
                "status": result["lifecycle"]["status"],
                "total_tokens": sum(item["tokens_including_eod"] for item in result["materialized_artifacts"]),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
