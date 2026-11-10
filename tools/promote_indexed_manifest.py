#!/usr/bin/env python3
"""Promote independently reviewed indexed-data evidence to one final manifest.

This is intentionally separate from materialization and cache preparation.  It
never modifies an input candidate: the only state-changing action is an atomic,
write-once final manifest after it has revalidated the indexed data, cache, and
Bridge fast-cache-hit evidence.

The independent-review record is a small, explicit human-review binding::

    {
      "schema_version": 1,
      "kind": "ampa_dataset_independent_review",
      "verdict": "passed",
      "reviewer": "independent-reviewer-identity",
      "cache_hit_manifest_sha256": "...",
      "cache_validated_manifest_sha256": "..."
    }

Additional fields are allowed for review notes, but none can weaken these
required bindings.  The record is evidence of an independent review; this
program independently verifies the technical facts again rather than trusting
that assertion alone.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.megatron_cache_contract import (  # noqa: E402
    CacheContractError,
    cache_contract_sha256,
    canonical_json_sha256,
    expected_blended_train_size,
    sha256_file,
    validate_cache_inventory,
)
from tools.megatron_cache_stage import (  # noqa: E402
    CacheStageError,
    _cache_file_fingerprint,
    _contract_from_prepared_candidate,
    _validate_all_gpt_cache_key_fields,
    _validate_preparation_tokenizer,
    bridge_consumer_spec,
    resolve_pinned_hf_snapshot,
)
from tools.bridge_indexed_consumer import (  # noqa: E402
    IndexedConsumerError,
    expected_bridge_indexed_consumer,
)
from tools.c0_validation import (  # noqa: E402
    C0ValidationError,
    validate_c0_identity_manifest,
    validate_hf_snapshot_binding,
)
from tools.validate_indexed_mixture import (  # noqa: E402
    _load_bound_plan,
    _validate_artifact_schema,
    _validate_component,
    _validate_tokenizer_binding,
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
_CACHE_SCHEMA_VERSION = 1
_REVIEW_SCHEMA_VERSION = 1
_REVIEW_KIND = "ampa_dataset_independent_review"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_C0_MANIFEST = Path("checkpoint/megatron/qwen3-30b-a3b-base-c0/ampa_c0_manifest.json")


class PromotionError(ValueError):
    """Raised when a candidate cannot safely become a canonical manifest."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--cache-hit-manifest", type=Path, required=True)
    parser.add_argument("--cache-validated-manifest", type=Path, required=True)
    parser.add_argument("--independent-review", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--workspace-root", type=Path, required=True)
    return parser.parse_args(argv)


def _workspace_root(value: Path) -> Path:
    if value.is_symlink() or not value.is_dir():
        raise PromotionError(f"workspace root must be an existing non-symlink directory: {value}")
    return value.resolve()


def _require_under(path: Path, root: Path, label: str) -> None:
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise PromotionError(f"{label} escapes its required root: {path}") from error


def _regular_json(path: Path, label: str) -> tuple[Path, dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise PromotionError(f"{label} must be an existing non-symlink regular file: {path}")
    resolved = path.resolve()
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise PromotionError(f"{label} is not valid JSON: {error}") from error
    if not isinstance(value, dict):
        raise PromotionError(f"{label} must contain a JSON object")
    return resolved, value


def _sha256_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise PromotionError(f"{label} must be a lowercase SHA256")
    return value


def _require_lifecycle(value: Mapping[str, object], status: str, label: str) -> Mapping[str, object]:
    lifecycle = value.get("lifecycle")
    if not isinstance(lifecycle, Mapping) or lifecycle.get("status") != status:
        raise PromotionError(f"{label} is not a {status} candidate")
    if lifecycle.get("state_machine_version") != _LIFECYCLE_VERSION:
        raise PromotionError(f"{label} does not use lifecycle version {_LIFECYCLE_VERSION}")
    if lifecycle.get("transition_sequence") != list(_TRANSITION_SEQUENCE):
        raise PromotionError(f"{label} does not use the AMPA dataset lifecycle state machine")
    if lifecycle.get("canonical_training_allowed") is not False:
        raise PromotionError(f"{label} must still forbid canonical training")
    return lifecycle


def _require_same_mapping(observed: object, expected: object, label: str) -> None:
    if observed != expected:
        raise PromotionError(f"{label} differs from independently revalidated evidence")


def _require_probe_train_length(observed: object, contract: Mapping[str, object]) -> None:
    """Check the constructed MCore blend, not the requested sampler target."""
    if type(observed) is not int or observed != expected_blended_train_size(contract):
        raise PromotionError("Bridge cache-hit probe train dataset length differs from the native cache contract")


def _validate_review(
    review: Mapping[str, object], *, cache_hit_sha256: str, cache_validated_sha256: str
) -> None:
    if review.get("schema_version") != _REVIEW_SCHEMA_VERSION:
        raise PromotionError(f"independent review schema_version must be {_REVIEW_SCHEMA_VERSION}")
    if review.get("kind") != _REVIEW_KIND:
        raise PromotionError("independent review kind is invalid")
    if review.get("verdict") != "passed":
        raise PromotionError("independent review verdict is not passed")
    reviewer = review.get("reviewer")
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise PromotionError("independent review must identify a reviewer")
    if _sha256_text(review.get("cache_hit_manifest_sha256"), "review cache-hit manifest SHA256") != cache_hit_sha256:
        raise PromotionError("independent review is not bound to this cache-hit manifest")
    if (
        _sha256_text(review.get("cache_validated_manifest_sha256"), "review cache-validated manifest SHA256")
        != cache_validated_sha256
    ):
        raise PromotionError("independent review is not bound to this cache-validated manifest")


def _reverify_indexed_candidate(
    cache_validated: Mapping[str, object], *, workspace_root: Path
) -> tuple[int, Mapping[str, object]]:
    """Re-run the indexed-data checks retained by the immutable input chain."""
    preparation = cache_validated.get("cache_preparation")
    if not isinstance(preparation, Mapping):
        raise PromotionError("cache-validated manifest lacks cache preparation evidence")
    contract = preparation.get("cache_contract")
    if not isinstance(contract, Mapping):
        raise PromotionError("cache-validated manifest lacks its cache contract")
    indexed_binding = contract.get("indexed_manifest")
    if not isinstance(indexed_binding, Mapping):
        raise PromotionError("cache contract lacks indexed-validation binding")
    indexed_path_value = indexed_binding.get("path")
    if not isinstance(indexed_path_value, str):
        raise PromotionError("cache contract indexed-validation path is missing")
    indexed_path, indexed = _regular_json(Path(indexed_path_value), "indexed-validation manifest")
    _require_under(indexed_path, workspace_root / "dataset" / "manifests", "indexed-validation manifest")
    if sha256_file(indexed_path) != _sha256_text(indexed_binding.get("sha256"), "indexed-validation manifest SHA256"):
        raise PromotionError("indexed-validation manifest changed after cache preparation")
    _require_lifecycle(indexed, "indexed_validated_pending_cache", "indexed-validation manifest")
    # The final record is built from the cache-validated parent, so verify that
    # its data-bearing fields remain byte-for-byte equivalent to the indexed
    # validation candidate bound into the cache contract.
    for field in (
        "schema_version",
        "manifest_id",
        "model_and_tokenizer",
        "sampling",
        "sequence_construction",
        "initial_workload_plan",
        "sources",
        "planned_training_data_path",
        "materialization_requirements",
        "source_plan",
        "tokenizer_observed",
        "materialized_training_data_path",
        "materialized_artifacts",
        "indexed_validation",
    ):
        _require_same_mapping(
            cache_validated.get(field), indexed.get(field), f"cache-validated {field}"
        )

    plan_path, plan, plan_sha256 = _load_bound_plan(indexed)
    if plan_path.parents[2].resolve() != workspace_root:
        raise PromotionError("indexed-validation source plan belongs to another workspace")
    tokenizer = _validate_tokenizer_binding(plan_path, plan, indexed)
    artifacts = indexed.get("materialized_artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != len(plan["sources"]):
        raise PromotionError("indexed-validation artifact count differs from the immutable source plan")
    artifacts_by_name: dict[str, dict[str, Any]] = {}
    for artifact in artifacts:
        if not isinstance(artifact, dict) or not isinstance(artifact.get("name"), str) or artifact["name"] in artifacts_by_name:
            raise PromotionError("indexed-validation artifacts must have unique source names")
        artifacts_by_name[artifact["name"]] = artifact
    if set(artifacts_by_name) != {source["name"] for source in plan["sources"]}:
        raise PromotionError("indexed-validation artifacts do not match immutable source names")

    results: list[dict[str, Any]] = []
    data_path_parts: list[str] = []
    for source in plan["sources"]:
        artifact = artifacts_by_name[source["name"]]
        expected_prefix = workspace_root / source["indexed_prefix"]
        expected_ledger = workspace_root / "dataset" / "indexed" / plan["manifest_id"] / "ledgers" / f"{source['name']}.jsonl"
        _validate_artifact_schema(artifact, source, expected_prefix, expected_ledger)
        if artifact["tokens_including_eod"] < source["planned_tokens"]:
            raise PromotionError(f"{artifact['name']} is below its source-specific planned token target")
        results.append(_validate_component(artifact, source, tokenizer["eod"], tokenizer["indexed_dtype"]))
        data_path_parts.extend((str(source["weight"]), artifact["indexed_prefix"]))
    if indexed.get("materialized_training_data_path") != " ".join(data_path_parts):
        raise PromotionError("indexed-validation materialized data path differs from immutable source artifacts")
    indexed_validation = indexed.get("indexed_validation")
    if not isinstance(indexed_validation, Mapping):
        raise PromotionError("indexed-validation result is missing")
    _require_same_mapping(indexed_validation.get("components"), results, "indexed-validation components")
    total_tokens = sum(result["tokens_including_eod"] for result in results)
    if indexed_validation.get("total_tokens") != total_tokens:
        raise PromotionError("indexed-validation total token count differs from independently reread data")
    if indexed_validation.get("immutable_plan_sha256") != plan_sha256:
        raise PromotionError("indexed-validation immutable source-plan hash differs")
    _require_same_mapping(
        indexed_validation.get("tokenizer_independently_validated"), tokenizer, "indexed-validation tokenizer evidence"
    )
    return total_tokens, plan


def _reverify_cache_and_probe(
    cache_hit: Mapping[str, object], cache_validated: Mapping[str, object], *, workspace_root: Path, parent_sha256: str
) -> dict[str, object]:
    """Verify the fixed cache, local snapshot, and actual Bridge cache-hit proof."""
    parent_lifecycle = _require_lifecycle(
        cache_validated, "cache_validated_pending_independent_review", "cache-validated manifest"
    )
    contract, preparation = _contract_from_prepared_candidate(cache_validated, workspace_root=workspace_root)
    contract_sha256 = cache_contract_sha256(contract)
    if parent_lifecycle.get("cache_contract_sha256") != contract_sha256:
        raise PromotionError("cache-validated lifecycle is not bound to its cache contract")
    _sha256_text(parent_lifecycle.get("cache_prepared_manifest_sha256"), "cache-prepared manifest SHA256")
    model = cache_validated.get("model_and_tokenizer")
    if not isinstance(model, Mapping):
        raise PromotionError("cache-validated manifest lacks model/tokenizer identity")
    snapshot = resolve_pinned_hf_snapshot(workspace_root, model)
    c0_snapshot_binding = _validate_c0_snapshot_binding(workspace_root, model, snapshot)
    cache_dir = Path(str(preparation.get("cache_path", "")))
    _validate_all_gpt_cache_key_fields(contract, cache_dir)
    tokenizer_ids = _validate_preparation_tokenizer(
        preparation=preparation, cache_dir=cache_dir, snapshot=snapshot
    )
    inventory = validate_cache_inventory(contract, workspace_root=workspace_root, cache_dir=cache_dir)
    cache_validation = cache_validated.get("cache_validation")
    if not isinstance(cache_validation, Mapping) or cache_validation.get("schema_version") != _CACHE_SCHEMA_VERSION:
        raise PromotionError("cache-validated manifest has invalid cache-validation evidence")
    _require_same_mapping(cache_validation.get("inventory"), inventory, "cache-validation inventory")
    _require_same_mapping(
        cache_validation.get("mcore_tokenizer_unique_identifiers"), tokenizer_ids, "cache-validation tokenizer identity"
    )
    _require_same_mapping(cache_validation.get("pinned_hf_snapshot"), snapshot, "cache-validation pinned HF snapshot")

    expected_consumer = bridge_consumer_spec(contract, cache_dir=cache_dir)
    _require_same_mapping(preparation.get("bridge_consumer"), expected_consumer, "cache-preparation Bridge consumer")
    consumer_manifest = _consumer_manifest(
        cache_validated.get("model_and_tokenizer"),
        bridge_consumer=expected_consumer,
        inventory=inventory,
        snapshot=snapshot,
    )
    try:
        expected_consumer_view = expected_bridge_indexed_consumer(consumer_manifest)
    except IndexedConsumerError as error:
        raise PromotionError(f"final Bridge consumer contract is invalid: {error}") from error
    expected_pending_probe = preparation.get("bridge_cache_hit_probe")
    if not isinstance(expected_pending_probe, Mapping) or expected_pending_probe.get("status") != "required_not_run":
        raise PromotionError("cache preparation does not retain the required pre-probe state")

    hit_lifecycle = _require_lifecycle(
        cache_hit, "cache_validated_pending_independent_review", "cache-hit manifest"
    )
    if hit_lifecycle.get("cache_validated_manifest_sha256") != parent_sha256:
        raise PromotionError("cache-hit candidate is not bound to the supplied cache-validated parent")
    if hit_lifecycle.get("cache_contract_sha256") != contract_sha256:
        raise PromotionError("cache-hit candidate cache-contract binding differs")
    if hit_lifecycle.get("bridge_cache_hit_probe_passed") is not True:
        raise PromotionError("cache-hit candidate does not record a passed Bridge cache-hit probe")

    hit_validation = cache_hit.get("cache_validation")
    if not isinstance(hit_validation, Mapping):
        raise PromotionError("cache-hit candidate lacks cache-validation evidence")
    for field in ("schema_version", "inventory", "mcore_tokenizer_unique_identifiers", "pinned_hf_snapshot"):
        _require_same_mapping(hit_validation.get(field), cache_validation.get(field), f"cache-hit {field}")
    probe = hit_validation.get("bridge_cache_hit_probe")
    if not isinstance(probe, Mapping) or probe.get("required") is not True or probe.get("status") != "passed":
        raise PromotionError("cache-hit candidate lacks a passed required Bridge probe")
    current_fingerprint = canonical_json_sha256(_cache_file_fingerprint(cache_dir))
    if probe.get("cache_inventory_sha256_before") != current_fingerprint or probe.get(
        "cache_inventory_sha256_after"
    ) != current_fingerprint:
        raise PromotionError("Bridge cache-hit probe cache fingerprint no longer matches the current cache")
    if probe.get("validated_cache_inventory_sha256") != inventory["artifact_inventory_sha256"]:
        raise PromotionError("Bridge cache-hit probe inventory differs from the revalidated cache")
    _require_same_mapping(
        probe.get("mcore_tokenizer_unique_identifiers"), tokenizer_ids, "Bridge cache-hit probe tokenizer identity"
    )
    _require_full_probe_consumer(probe, expected_consumer_view=expected_consumer_view, snapshot=snapshot)
    _require_probe_train_length(probe.get("train_dataset_length"), contract)
    sample_keys = probe.get("sample_keys")
    if (
        not isinstance(sample_keys, list)
        or not sample_keys
        or any(not isinstance(key, str) or not key for key in sample_keys)
        or sample_keys != sorted(set(sample_keys))
    ):
        raise PromotionError("Bridge cache-hit probe did not record a non-empty canonical sample-key set")
    return {
        "contract": contract,
        "contract_sha256": contract_sha256,
        "inventory": inventory,
        "snapshot": snapshot,
        "c0_snapshot_binding": c0_snapshot_binding,
        "bridge_consumer": expected_consumer,
        "bridge_cache_hit_probe": dict(probe),
    }


def _consumer_manifest(
    model: object,
    *,
    bridge_consumer: Mapping[str, object],
    inventory: Mapping[str, object],
    snapshot: Mapping[str, object],
) -> dict[str, object]:
    """Make the finalized adapter input from independently observed values."""
    if not isinstance(model, Mapping):
        raise PromotionError("cache-validated manifest lacks model/tokenizer identity")
    return {
        "model_and_tokenizer": dict(model),
        "megatron_cache": {
            "status": "validated",
            "path": inventory["cache_path"],
            "per_dataset_sequences_path": inventory["per_dataset_sequences_path"],
            "per_dataset_sequences_sha256": inventory["per_dataset_sequences_sha256"],
            "bridge_consumer": dict(bridge_consumer),
            "pinned_hf_snapshot": dict(snapshot),
        },
    }


def _require_full_probe_consumer(
    probe: Mapping[str, object], *, expected_consumer_view: Mapping[str, object], snapshot: Mapping[str, object]
) -> None:
    """Reject a probe that reports a subset or altered finalized consumer field."""
    _require_same_mapping(
        probe.get("bridge_consumer"), expected_consumer_view, "Bridge full finalized consumer configuration"
    )
    _require_same_mapping(probe.get("pinned_hf_snapshot"), snapshot, "Bridge cache-hit pinned HF snapshot")


def _validate_c0_snapshot_binding(
    workspace_root: Path, model: Mapping[str, object], snapshot: Mapping[str, object]
) -> dict[str, object]:
    """Bind cache tokenization to C0 without rehashing C0 or HF payloads."""
    model_path_value = model.get("model_manifest")
    model_digest = _sha256_text(model.get("model_manifest_sha256"), "dataset model-manifest SHA256")
    if not isinstance(model_path_value, str) or not model_path_value:
        raise PromotionError("dataset model-manifest path is missing")
    model_path = Path(model_path_value)
    if model_path.is_absolute() or ".." in model_path.parts:
        raise PromotionError("dataset model-manifest path must be workspace-relative")
    model_path = workspace_root / model_path
    if model_path.is_symlink() or not model_path.is_file() or sha256_file(model_path) != model_digest:
        raise PromotionError("dataset model manifest is missing or differs from its pinned SHA256")
    model_id = model.get("model_id")
    revision = model.get("revision")
    if not isinstance(model_id, str) or not model_id or not isinstance(revision, str) or not revision:
        raise PromotionError("dataset model identity is incomplete")
    c0_path, c0_manifest = _regular_json(workspace_root / _C0_MANIFEST, "C0 identity manifest")
    _require_under(c0_path, workspace_root / "checkpoint" / "megatron", "C0 identity manifest")
    try:
        validate_c0_identity_manifest(
            c0_manifest,
            model_manifest_sha256=model_digest,
            model_id=model_id,
            revision=revision,
        )
    except C0ValidationError as error:
        raise PromotionError(f"C0 identity manifest is not a matching immutable import: {error}") from error
    snapshot_manifest_path = snapshot.get("manifest_path")
    snapshot_manifest_sha256 = _sha256_text(snapshot.get("manifest_sha256"), "local HF snapshot manifest SHA256")
    if not isinstance(snapshot_manifest_path, str):
        raise PromotionError("local HF snapshot manifest path is missing")
    local_snapshot_path, local_snapshot = _regular_json(Path(snapshot_manifest_path), "local HF snapshot manifest")
    _require_under(local_snapshot_path, workspace_root / "checkpoint" / "hf", "local HF snapshot manifest")
    if sha256_file(local_snapshot_path) != snapshot_manifest_sha256:
        raise PromotionError("local HF snapshot manifest changed after cache preparation")
    try:
        validate_hf_snapshot_binding(
            local_snapshot,
            snapshot_manifest_sha256=snapshot_manifest_sha256,
            expected_snapshot_manifest_sha256=_sha256_text(
                c0_manifest.get("hf_snapshot_manifest_sha256"), "C0 HF snapshot manifest SHA256"
            ),
            model_id=model_id,
            revision=revision,
        )
    except C0ValidationError as error:
        raise PromotionError(f"local HF snapshot is not the C0-bound materialized snapshot: {error}") from error
    return {
        "c0_identity_manifest_path": str(c0_path),
        "c0_identity_manifest_sha256": sha256_file(c0_path),
        "hf_snapshot_manifest_sha256": snapshot_manifest_sha256,
        "local_snapshot_lifecycle": dict(local_snapshot["lifecycle"]),
    }


def _validated_workload(cache_validated: Mapping[str, object], contract: Mapping[str, object], total_tokens: int) -> dict[str, object]:
    workload = contract.get("workload_identity")
    if not isinstance(workload, Mapping):
        raise PromotionError("cache contract lacks workload identity")
    plan = cache_validated.get("initial_workload_plan")
    if not isinstance(plan, Mapping):
        raise PromotionError("cache-validated manifest lacks its immutable workload plan")
    sampling = cache_validated.get("sampling")
    if not isinstance(sampling, Mapping):
        raise PromotionError("cache-validated manifest lacks its immutable sampling plan")
    for key, value in workload.items():
        observed = sampling.get("seed") if key == "sampler_seed" else plan.get(key)
        if observed != value:
            raise PromotionError(f"immutable workload plan differs from cache contract for {key}")
    safety_multiple = plan.get("safety_multiple")
    if (
        isinstance(safety_multiple, bool)
        or not isinstance(safety_multiple, (int, float))
        or safety_multiple < 1.0
    ):
        raise PromotionError("immutable workload safety_multiple must be a number at least one")
    required_tokens = (
        int(workload["warmup_steps"])
        + int(workload["measured_steps"])
    ) * int(workload["global_batch_size"]) * int(workload["sequence_length"])
    if total_tokens < required_tokens * safety_multiple:
        raise PromotionError("indexed token total is below the reviewed workload budget with its safety margin")
    return {**dict(workload), "safety_multiple": safety_multiple}


def _final_manifest(
    cache_validated: Mapping[str, object],
    *,
    cache_hit_path: Path,
    cache_hit_sha256: str,
    cache_validated_path: Path,
    cache_validated_sha256: str,
    review_path: Path,
    review_sha256: str,
    cache_evidence: Mapping[str, object],
    validated_workload: Mapping[str, object],
) -> dict[str, object]:
    """Build a final record only from verified parent and observed evidence."""
    result = copy.deepcopy(dict(cache_validated))
    contract = cache_evidence["contract"]
    inventory = cache_evidence["inventory"]
    if not isinstance(contract, Mapping) or not isinstance(inventory, Mapping):
        raise PromotionError("internal promotion evidence is malformed")
    data = contract.get("data")
    if not isinstance(data, Mapping) or not isinstance(data.get("flat_data_path"), str):
        raise PromotionError("cache contract lacks its derived flat data-path identity")
    result["lifecycle"] = {
        "status": "validated_for_canonical_run",
        "state_machine_version": _LIFECYCLE_VERSION,
        "transition_sequence": list(_TRANSITION_SEQUENCE),
        "canonical_training_allowed": True,
        "cache_hit_manifest_sha256": cache_hit_sha256,
        "cache_validated_manifest_sha256": cache_validated_sha256,
        "cache_contract_sha256": cache_evidence["contract_sha256"],
        "independent_review_sha256": review_sha256,
    }
    result["validated_training_data_path"] = data["flat_data_path"]
    result["validated_training_blend_per_split"] = cache_evidence["bridge_consumer"]["blend_per_split"]
    result["validated_workload"] = dict(validated_workload)
    result["megatron_cache"] = {
        "status": "validated",
        "path": inventory["cache_path"],
        "workload_identity": dict(contract["workload_identity"]),
        "per_dataset_sequences_path": inventory["per_dataset_sequences_path"],
        "per_dataset_sequences_sha256": inventory["per_dataset_sequences_sha256"],
        "artifacts": inventory["artifacts"],
        "artifact_inventory_sha256": inventory["artifact_inventory_sha256"],
        "cache_contract_sha256": cache_evidence["contract_sha256"],
        "bridge_consumer": cache_evidence["bridge_consumer"],
        "bridge_cache_hit_probe": cache_evidence["bridge_cache_hit_probe"],
        "pinned_hf_snapshot": cache_evidence["snapshot"],
        "c0_snapshot_binding": cache_evidence["c0_snapshot_binding"],
    }
    result["independent_review"] = {
        "path": str(review_path),
        "sha256": review_sha256,
        "cache_hit_manifest": {"path": str(cache_hit_path), "sha256": cache_hit_sha256},
        "cache_validated_manifest": {"path": str(cache_validated_path), "sha256": cache_validated_sha256},
    }
    return result


def _write_once_atomic(path: Path, payload: Mapping[str, object]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite final dataset manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(f"{path}.partial-{os.getpid()}")
    if staged.exists() or staged.is_symlink():
        raise FileExistsError(f"staged final dataset manifest already exists: {staged}")
    with staged.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(staged, path)


def promote(args: argparse.Namespace) -> dict[str, object]:
    workspace_root = _workspace_root(args.workspace_root)
    manifest_root = workspace_root / "dataset" / "manifests"
    cache_hit_path, cache_hit = _regular_json(args.cache_hit_manifest, "cache-hit manifest")
    parent_path, cache_validated = _regular_json(args.cache_validated_manifest, "cache-validated manifest")
    review_path, review = _regular_json(args.independent_review, "independent review")
    output_path = args.output_manifest
    if output_path.is_symlink():
        raise PromotionError(f"output manifest must not be a symlink: {output_path}")
    output_path = output_path.resolve()
    for path, label in ((cache_hit_path, "cache-hit manifest"), (parent_path, "cache-validated manifest"), (output_path, "output manifest")):
        _require_under(path, manifest_root, label)
    _require_under(review_path, workspace_root, "independent review")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite final dataset manifest: {output_path}")

    cache_hit_sha256 = sha256_file(cache_hit_path)
    parent_sha256 = sha256_file(parent_path)
    review_sha256 = sha256_file(review_path)
    _validate_review(review, cache_hit_sha256=cache_hit_sha256, cache_validated_sha256=parent_sha256)
    total_tokens, _plan = _reverify_indexed_candidate(cache_validated, workspace_root=workspace_root)
    cache_evidence = _reverify_cache_and_probe(
        cache_hit, cache_validated, workspace_root=workspace_root, parent_sha256=parent_sha256
    )
    workload = _validated_workload(cache_validated, cache_evidence["contract"], total_tokens)
    result = _final_manifest(
        cache_validated,
        cache_hit_path=cache_hit_path,
        cache_hit_sha256=cache_hit_sha256,
        cache_validated_path=parent_path,
        cache_validated_sha256=parent_sha256,
        review_path=review_path,
        review_sha256=review_sha256,
        cache_evidence=cache_evidence,
        validated_workload=workload,
    )
    _write_once_atomic(output_path, result)
    return result


def main() -> None:
    args = parse_args()
    try:
        result = promote(args)
    except (CacheContractError, CacheStageError, PromotionError, FileNotFoundError) as error:
        raise SystemExit(f"dataset promotion failed: {error}") from error
    print(
        json.dumps(
            {
                "manifest_id": result.get("manifest_id"),
                "status": result["lifecycle"]["status"],
                "canonical_training_allowed": result["lifecycle"]["canonical_training_allowed"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
