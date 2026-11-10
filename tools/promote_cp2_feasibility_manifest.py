#!/usr/bin/env python3
"""Promote a separately reviewed CP2/EP8 cache to a feasibility-only manifest.

The indexed tokens and immutable source plan remain the validated v2 CP1
artifacts.  Only the native GPT cache and runtime workload are CP2-specific.
Run this on an allocated node: promotion revalidates indexed data and cache
contents.  A successful manifest does not certify a completed training step.

The independent-review JSON must contain schema_version 1, kind
``ampa_cp2_feasibility_independent_review``, verdict ``passed``, a nonempty
reviewer, variant_id ``cp2_ep8``, and SHA256 bindings named
``parent_validated_manifest_sha256``, ``cache_hit_manifest_sha256``, and
``cache_validated_manifest_sha256``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Mapping

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import promote_indexed_manifest as common  # noqa: E402
from tools.megatron_cache_contract import (  # noqa: E402
    CacheContractError,
    sha256_file,
)
from tools.megatron_cache_stage import CacheStageError  # noqa: E402


_VARIANT_ID = "cp2_ep8"
_REVIEW_KIND = "ampa_cp2_feasibility_independent_review"
_EXPECTED_WORKLOAD = {
    "micro_batch_size": 1,
    "dense_data_parallel_size": 4,
    "gradient_accumulation": 16,
    "global_batch_size": 64,
    "sequence_length": 4096,
    "warmup_steps": 5,
    "measured_steps": 20,
    "sampler_seed": 1234,
}
_PARENT_WORKLOAD = {**_EXPECTED_WORKLOAD, "dense_data_parallel_size": 8, "gradient_accumulation": 8}
_DATA_FIELDS = (
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
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--parent-validated-manifest", type=Path, required=True)
    parser.add_argument("--cache-hit-manifest", type=Path, required=True)
    parser.add_argument("--cache-validated-manifest", type=Path, required=True)
    parser.add_argument("--independent-review", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--workspace-root", type=Path, required=True)
    return parser.parse_args(argv)


def _require_parent(
    parent: Mapping[str, object], candidate: Mapping[str, object], *, parent_sha256: str
) -> None:
    lifecycle = parent.get("lifecycle")
    if (
        not isinstance(lifecycle, Mapping)
        or lifecycle.get("status") != "validated_for_canonical_run"
        or lifecycle.get("canonical_training_allowed") is not True
    ):
        raise common.PromotionError("parent must be a validated canonical v2 data manifest")
    parent_workload = parent.get("validated_workload")
    if not isinstance(parent_workload, Mapping) or any(
        parent_workload.get(key) != value for key, value in _PARENT_WORKLOAD.items()
    ):
        raise common.PromotionError("parent is not the validated CP1/DP8/GA8 data workload")
    for field in _DATA_FIELDS:
        common._require_same_mapping(candidate.get(field), parent.get(field), f"CP2 versus parent {field}")
    parent_cache = parent.get("megatron_cache")
    if not isinstance(parent_cache, Mapping) or parent_cache.get("status") != "validated":
        raise common.PromotionError("parent native cache was not validated")
    if parent.get("validated_training_data_path") != candidate.get("materialized_training_data_path"):
        raise common.PromotionError("parent validated data path differs from CP2 indexed data")
    common._sha256_text(parent_sha256, "parent validated manifest SHA256")


def _validate_review(review: Mapping[str, object], *, parent_sha256: str, hit_sha256: str, validated_sha256: str) -> None:
    if review.get("schema_version") != 1 or review.get("kind") != _REVIEW_KIND:
        raise common.PromotionError("independent CP2 review schema or kind is invalid")
    if review.get("verdict") != "passed" or review.get("variant_id") != _VARIANT_ID:
        raise common.PromotionError("independent CP2 review did not pass for cp2_ep8")
    reviewer = review.get("reviewer")
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise common.PromotionError("independent CP2 review must identify its reviewer")
    for key, expected in (
        ("parent_validated_manifest_sha256", parent_sha256),
        ("cache_hit_manifest_sha256", hit_sha256),
        ("cache_validated_manifest_sha256", validated_sha256),
    ):
        if common._sha256_text(review.get(key), f"independent review {key}") != expected:
            raise common.PromotionError(f"independent CP2 review is not bound to this {key}")


def _validated_workload(
    candidate: Mapping[str, object], contract: Mapping[str, object], *, total_tokens: int
) -> dict[str, object]:
    if contract.get("variant_id") != _VARIANT_ID or contract.get("workload_identity") != _EXPECTED_WORKLOAD:
        raise common.PromotionError("CP2 cache contract has the wrong variant or workload identity")
    prepare = contract.get("prepare_cache")
    parallel = prepare.get("parallelism") if isinstance(prepare, Mapping) else None
    if parallel != {"tp": 1, "pp": 1, "cp": 2, "ep": 8, "etp": 1}:
        raise common.PromotionError("CP2 cache contract has the wrong parallelism")
    sampling = candidate.get("sampling")
    if not isinstance(sampling, Mapping) or sampling.get("seed") != _EXPECTED_WORKLOAD["sampler_seed"]:
        raise common.PromotionError("CP2 sampler seed differs from the immutable indexed data")
    initial = candidate.get("initial_workload_plan")
    if not isinstance(initial, Mapping):
        raise common.PromotionError("CP2 candidate lacks the immutable original workload plan")
    safety_multiple = initial.get("safety_multiple")
    if isinstance(safety_multiple, bool) or not isinstance(safety_multiple, (int, float)) or safety_multiple < 1.0:
        raise common.PromotionError("original workload safety_multiple must be at least one")
    required = (
        (_EXPECTED_WORKLOAD["warmup_steps"] + _EXPECTED_WORKLOAD["measured_steps"])
        * _EXPECTED_WORKLOAD["global_batch_size"]
        * _EXPECTED_WORKLOAD["sequence_length"]
    )
    if total_tokens < required * safety_multiple:
        raise common.PromotionError("validated indexed tokens do not cover the CP2 workload safety budget")
    return {**_EXPECTED_WORKLOAD, "safety_multiple": safety_multiple}


def promote(args: argparse.Namespace) -> dict[str, object]:
    workspace = common._workspace_root(args.workspace_root)
    manifest_root = workspace / "dataset" / "manifests"
    parent_path, parent = common._regular_json(args.parent_validated_manifest, "parent validated manifest")
    hit_path, hit = common._regular_json(args.cache_hit_manifest, "CP2 cache-hit manifest")
    validated_path, validated = common._regular_json(args.cache_validated_manifest, "CP2 cache-validated manifest")
    review_path, review = common._regular_json(args.independent_review, "independent CP2 review")
    output = args.output_manifest
    if output.is_symlink():
        raise common.PromotionError(f"output manifest must not be a symlink: {output}")
    output = output.resolve()
    for path, label in (
        (parent_path, "parent validated manifest"),
        (hit_path, "CP2 cache-hit manifest"),
        (validated_path, "CP2 cache-validated manifest"),
        (output, "CP2 output manifest"),
    ):
        common._require_under(path, manifest_root, label)
    common._require_under(review_path, workspace, "independent CP2 review")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite final CP2 feasibility manifest: {output}")

    parent_sha256 = sha256_file(parent_path)
    hit_sha256 = sha256_file(hit_path)
    validated_sha256 = sha256_file(validated_path)
    review_sha256 = sha256_file(review_path)
    _validate_review(
        review,
        parent_sha256=parent_sha256,
        hit_sha256=hit_sha256,
        validated_sha256=validated_sha256,
    )
    common._require_lifecycle(validated, "cache_validated_pending_independent_review", "CP2 cache-validated manifest")
    _require_parent(parent, validated, parent_sha256=parent_sha256)
    total_tokens, _ = common._reverify_indexed_candidate(validated, workspace_root=workspace)
    evidence = common._reverify_cache_and_probe(
        hit, validated, workspace_root=workspace, parent_sha256=validated_sha256
    )
    parent_cache = parent["megatron_cache"]
    if parent_cache.get("c0_snapshot_binding") != evidence.get("c0_snapshot_binding"):
        raise common.PromotionError("CP2 cache is not bound to the same C0 snapshot as its parent")
    contract = evidence["contract"]
    if not isinstance(contract, Mapping):
        raise common.PromotionError("CP2 cache contract evidence is malformed")
    workload = _validated_workload(validated, contract, total_tokens=total_tokens)
    result = common._final_manifest(
        validated,
        cache_hit_path=hit_path,
        cache_hit_sha256=hit_sha256,
        cache_validated_path=validated_path,
        cache_validated_sha256=validated_sha256,
        review_path=review_path,
        review_sha256=review_sha256,
        cache_evidence=evidence,
        validated_workload=workload,
    )
    result["variant_id"] = _VARIANT_ID
    result["parent_validated_manifest_sha256"] = parent_sha256
    result["parent_validated_manifest"] = {"path": str(parent_path), "sha256": parent_sha256}
    result["lifecycle"] = {
        "status": "validated_for_feasibility_run",
        "state_machine_version": 1,
        "transition_sequence": list(common._TRANSITION_SEQUENCE[:-1]) + ["validated_for_feasibility_run"],
        "canonical_training_allowed": False,
        "feasibility_training_allowed": True,
        "variant_id": _VARIANT_ID,
        "parent_validated_manifest_sha256": parent_sha256,
        "cache_hit_manifest_sha256": hit_sha256,
        "cache_validated_manifest_sha256": validated_sha256,
        "cache_contract_sha256": evidence["contract_sha256"],
        "independent_review_sha256": review_sha256,
    }
    common._write_once_atomic(output, result)
    return result


def main() -> None:
    try:
        result = promote(parse_args())
    except (CacheContractError, CacheStageError, common.PromotionError, FileNotFoundError) as error:
        raise SystemExit(f"CP2 feasibility promotion failed: {error}") from error
    print(
        f"{result['variant_id']}: {result['lifecycle']['status']} "
        f"(canonical_training_allowed=false)"
    )


if __name__ == "__main__":
    main()
