"""Reuse validated native cache arrays with a separately verified EP16 consumer.

Execute preparation only on allocated compute in the configured container.
The original cache contract and preparation provenance remain unchanged.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

from tools.bridge_indexed_consumer import expected_bridge_indexed_consumer
from tools.qwen_workload import workload

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "dataset/manifests/qwen3-30b-a3b-base-realistic-mix-v2-cp2-validated.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _derivative(parent: dict, variant: str) -> dict:
    spec = workload(variant)
    if spec["ep"] != 16:
        raise ValueError("runtime derivative is restricted to EP16")
    if (parent.get("variant_id") != "cp2_ep8"
            or parent.get("lifecycle", {}).get("feasibility_training_allowed") is not True
            or parent.get("megatron_cache", {}).get("status") != "validated"):
        raise ValueError("parent is not the validated CP2/EP8 indexed dataset")
    result = copy.deepcopy(parent)
    result["variant_id"] = variant
    result["validated_workload"]["dense_data_parallel_size"] = spec["dense_dp"]
    result["validated_workload"]["gradient_accumulation"] = spec["accumulation"]
    result["megatron_cache"]["workload_identity"]["dense_data_parallel_size"] = spec["dense_dp"]
    result["megatron_cache"]["workload_identity"]["gradient_accumulation"] = spec["accumulation"]
    result["megatron_cache"]["bridge_consumer"]["data_parallel_size"] = spec["dense_dp"]
    result["lifecycle"] = {
        "status": "validated_runtime_cache_reuse",
        "canonical_training_allowed": False, "feasibility_training_allowed": True,
        "variant_id": variant,
    }
    return result


def validate_runtime_derivative(manifest: dict, variant: str) -> None:
    """Fail closed on changes beyond runtime DP/GA and the verified probe."""
    if PARENT.is_symlink() or not PARENT.is_file():
        raise ValueError("validated cache parent is unavailable")
    parent = json.loads(PARENT.read_text())
    expected = _derivative(parent, variant)
    candidate = copy.deepcopy(manifest)
    reuse = candidate.pop("runtime_cache_reuse", {})
    if candidate != expected:
        raise ValueError("derivative changed immutable data, tokenizer, cache or preparation provenance")
    if reuse.get("parent_path") != str(PARENT) or reuse.get("parent_sha256") != sha256(PARENT):
        raise ValueError("runtime derivative is not bound to the pinned cache parent")
    probe = reuse.get("probe", {})
    parent_probe = parent["megatron_cache"]["bridge_cache_hit_probe"]
    parent_hash = parent_probe["cache_inventory_sha256_before"]
    if (probe.get("status") != "passed" or probe.get("train_samples") != 1600
            or probe.get("cache_sha256_before") != parent_hash
            or probe.get("cache_sha256_after") != parent_hash
            or parent_probe["cache_inventory_sha256_after"] != parent_hash
            or probe.get("dataset_length") != parent_probe["train_dataset_length"]
            or probe.get("sample_keys") != parent_probe["sample_keys"]
            or probe.get("actual_bridge_consumer") != expected_bridge_indexed_consumer(expected)):
        raise ValueError("missing matching cache-hit/consumer evidence for EP16")


def prepare(output: Path, variant: str) -> None:
    from tools.megatron_cache_stage import _bridge_fast_cache_dataset, _cache_file_fingerprint
    from tools.megatron_cache_contract import canonical_json_sha256
    parent = json.loads(PARENT.read_text())
    result = _derivative(parent, variant)
    cache = parent["megatron_cache"]
    cache_dir = Path(cache["path"])
    # Verify the parent artifacts before invoking the unchanged native fast path.
    for artifact in cache["artifacts"]:
        path = Path(artifact["path"])
        if path.is_symlink() or sha256(path) != artifact["sha256"]:
            raise ValueError(f"changed parent cache artifact: {path}")
    for artifact in parent["materialized_artifacts"]:
        for suffix in ("idx", "bin"):
            path = Path(f"{artifact['indexed_prefix']}.{suffix}")
            if path.is_symlink() or sha256(path) != artifact[f"{suffix}_sha256"]:
                raise ValueError(f"changed indexed artifact: {path}")
    before = _cache_file_fingerprint(cache_dir)
    # This copy configures a probe. It is not a new cache preparation contract.
    probe_contract = copy.deepcopy(parent["cache_preparation"]["cache_contract"])
    spec = workload(variant)
    probe_contract["workload_identity"]["dense_data_parallel_size"] = spec["dense_dp"]
    probe_contract["workload_identity"]["gradient_accumulation"] = spec["accumulation"]
    probe_contract["prepare_cache"]["parallelism"]["cp"] = spec["cp"]
    _, train, sample, actual = _bridge_fast_cache_dataset(
        probe_contract, cache_dir=cache_dir, snapshot=cache["pinned_hf_snapshot"],
        validated_tokenizer=parent["indexed_validation"]["tokenizer_independently_validated"],
    )
    after = _cache_file_fingerprint(cache_dir)
    if before != after or actual != expected_bridge_indexed_consumer(result):
        raise ValueError("native probe changed cache arrays or used the wrong runtime consumer")
    result["runtime_cache_reuse"] = {
        "parent_path": str(PARENT), "parent_sha256": sha256(PARENT),
        "probe": {"status": "passed", "train_samples": 1600,
                  "dataset_length": len(train), "sample_keys": sorted(sample),
                  "actual_bridge_consumer": actual,
                  "cache_sha256_before": canonical_json_sha256(before),
                  "cache_sha256_after": canonical_json_sha256(after)},
    }
    validate_runtime_derivative(result, variant)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"variant": variant, "output": str(output), "cache_reuse": "passed"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=("cp1_ep16", "cp2_ep16"), required=True)
    args = parser.parse_args()
    prepare(args.output, args.variant)
