#!/usr/bin/env python3
"""Compare matched profiling-off/on whole-step timing windows.

This is intentionally separate from training. It rejects comparisons whose
workload contracts, C0 manifest, or dataset manifest differ.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import uuid
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiling-off-run", type=Path, required=True)
    parser.add_argument("--profiling-on-run", type=Path, required=True)
    parser.add_argument("--max-overhead-percent", type=float, default=5.0)
    args = parser.parse_args()
    if args.max_overhead_percent < 0:
        parser.error("max overhead percent must be nonnegative")
    return args


def _sha256(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"required comparison input is missing: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _summary(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "analysis" / "optimizer_step_breakdown.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"timing summary is missing: {path}") from error
    if value.get("timing_unit") != "complete optimizer-step wall-time averaged over each logged window":
        raise ValueError(f"unsupported timing unit in {path}")
    if not isinstance(value.get("median_optimizer_step_ms"), (int, float)):
        raise ValueError(f"timing median is missing from {path}")
    return value


def _observation_mode(run_dir: Path) -> str:
    path = run_dir / "config" / "observation_mode.json"
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"observation mode record is missing: {path}") from error
    if record.get("schema_version") != 1 or record.get("data_mode") != "indexed":
        raise ValueError(f"invalid indexed-data observation mode record: {path}")
    mode = record.get("lightweight_profiling")
    if mode not in {"off", "on"}:
        raise ValueError(f"invalid lightweight profiling mode in {path}: {mode!r}")
    if record.get("nsys_trace", False) is not False:
        raise ValueError(f"Nsight diagnostic cannot be an overhead control: {path}")
    if record.get("torch_trace", False) is not False:
        raise ValueError(f"PyTorch profiler diagnostic cannot be an overhead control: {path}")
    expected_attached = mode == "on"
    for field in (
        "router_observer_attached",
        "route_wrapper_installed",
        "dispatcher_preprocess_wrapper_installed",
    ):
        if record.get(field) is not expected_attached:
            raise ValueError(f"observation mode {mode} has inconsistent {field}: {path}")
    return mode


def _environment(run_dir: Path, *, cp: int, world_size: int = 8, ep: int = 8) -> dict[str, Any]:
    if world_size not in (8, 16) or ep != world_size or cp not in (1, 2):
        raise ValueError("unsupported runtime provenance topology")
    path = run_dir / "environment.json"
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"runtime provenance is missing: {path}") from error
    placement = record.get("rank_placement")
    if not isinstance(placement, list) or len(placement) != world_size:
        raise ValueError(f"runtime provenance lacks {world_size} rank placements: {path}")
    if {entry.get("rank") for entry in placement} != set(range(world_size)):
        raise ValueError(f"runtime provenance has missing or duplicate ranks: {path}")
    host_counts = Counter(entry.get("hostname") for entry in placement)
    if (None in host_counts or len(host_counts) != world_size // 4
            or sorted(host_counts.values()) != [4] * (world_size // 4)):
        raise ValueError(f"runtime provenance is not a {world_size // 4}-node, four-rank placement: {path}")
    gpu_uuids = [entry.get("physical_gpu_uuid") for entry in placement]
    if len(set(gpu_uuids)) != world_size:
        raise ValueError(f"runtime provenance has duplicate physical GPUs: {path}")
    try:
        for gpu_uuid in gpu_uuids:
            if not isinstance(gpu_uuid, str) or not gpu_uuid.startswith("GPU-"):
                raise ValueError("missing GPU UUID")
            uuid.UUID(gpu_uuid.removeprefix("GPU-"))
    except ValueError as error:
        raise ValueError(f"runtime provenance has invalid physical GPU UUIDs: {path}") from error
    expected_topology = {
        "world_size": world_size, "tp": 1, "pp": 1, "cp": cp, "ep": ep, "etp": 1,
        "dense_dp": world_size // cp, "expert_dp": 1,
    }
    if cp == 2:
        expected_topology["dense_dp_cp"] = world_size
    if record.get("runtime_topology") != expected_topology:
        raise ValueError(f"runtime provenance has the wrong actual process-group topology: {path}")
    return record


def _rank_host_partition(environment: dict[str, Any]) -> tuple[tuple[int, ...], ...]:
    """Compare physical locality shape without requiring the same host names."""
    groups: dict[str, list[int]] = {}
    for entry in environment["rank_placement"]:
        groups.setdefault(entry["hostname"], []).append(int(entry["rank"]))
    return tuple(sorted(tuple(sorted(group)) for group in groups.values()))


def _physical_placement(environment: dict[str, Any]) -> tuple[tuple[int, str, str], ...]:
    return tuple(sorted(
        (entry["rank"], entry["hostname"], entry["physical_gpu_uuid"])
        for entry in environment["rank_placement"]
    ))


def _normalized_resolved_config(run_dir: Path) -> str:
    """Ignore only the three logger identities that necessarily name a run."""
    path = run_dir / "config" / "resolved.yaml"
    lines = path.read_text(encoding="utf-8").splitlines()
    expected = {
        "tensorboard_dir": str(run_dir.resolve() / "logs" / "tensorboard"),
        "wandb_exp_name": run_dir.name,
        "wandb_save_dir": str(run_dir.resolve() / "logs" / "wandb"),
    }
    seen: set[str] = set()
    normalized: list[str] = []
    for line in lines:
        key = next((key for key in expected if line.startswith(f"  {key}: ")), None)
        if key is None:
            normalized.append(line)
            continue
        if key in seen or line.partition(": ")[2] != expected[key]:
            raise ValueError(f"unexpected run-specific logger identity in {path}: {key}")
        seen.add(key)
        normalized.append(f"  {key}: <RUN_LOCAL>")
    if seen != set(expected):
        raise ValueError(f"resolved training config lacks run-specific logger identities: {path}")
    return "\n".join(normalized)


def compare(profiling_off_run: Path, profiling_on_run: Path, max_overhead_percent: float) -> dict[str, Any]:
    if _observation_mode(profiling_off_run) != "off" or _observation_mode(profiling_on_run) != "on":
        raise ValueError("overhead comparison requires an explicit profiling-off run and profiling-on run")
    required = (
        "comparison_contract.json",
        "c0_manifest.json",
        "dataset_manifest.json",
    )
    identities: dict[str, dict[str, str]] = {}
    for name in required:
        off_hash = _sha256(profiling_off_run / "config" / name)
        on_hash = _sha256(profiling_on_run / "config" / name)
        identities[name] = {"profiling_off_sha256": off_hash, "profiling_on_sha256": on_hash}
        if off_hash != on_hash:
            raise ValueError(f"profiling-off/on {name} differs; overhead comparison is invalid")

    contract = json.loads((profiling_off_run / "config" / "comparison_contract.json").read_text(encoding="utf-8"))
    scope = contract.get("claim_scope", {})
    logging = contract.get("logging", {})
    run_kind = contract.get("run_kind")
    canonical = run_kind == "canonical_indexed_profile"
    cp2_candidate = run_kind == "cp2_ep8_feasibility"
    if (
        contract.get("schema_version") != 2
        or not (canonical or cp2_candidate)
        or scope.get("kind") != "native_bridge_c0_measurement"
        or scope.get("hf_numerical_equivalence_claimed") is not False
        or scope.get("canonical_profile_data") is not canonical
        or contract.get("dataset", {}).get("canonical_profile_data") is not canonical
        or contract.get("parallelism", {}).get("cp") != (2 if cp2_candidate else 1)
        or contract.get("parallelism", {}).get("dense_dp") != (4 if cp2_candidate else 8)
        or logging.get("wandb_entity") != "anunay-yadav-epfl"
        or logging.get("wandb_project") != "moe-profile"
        or logging.get("wandb_mode") not in {"online", "offline", "disabled"}
        or logging.get("per_layer_moe_metrics") is not False
    ):
        raise ValueError("overhead comparison requires a native C0 canonical indexed-data or separate CP2 feasibility contract")

    cp = 2 if cp2_candidate else 1
    off_environment = _environment(profiling_off_run, cp=cp)
    on_environment = _environment(profiling_on_run, cp=cp)
    matching_runtime_fields = (
        "pytorch_version", "cuda_version", "cuda_allocator_conf", "nccl_version", "gpu_name",
        "container_sqsh_sha256", "megatron_lm", "megatron_bridge",
        "ampa_runtime_sources", "checkpoint", "dataset_blend_per_split", "parallelism",
        "activation_recomputation",
    )
    for field in matching_runtime_fields:
        if off_environment.get(field) is None or off_environment.get(field) != on_environment.get(field):
            raise ValueError(f"profiling-off/on runtime provenance differs or lacks {field}")
    if (not off_environment.get("slurm_job_id") or
            off_environment["slurm_job_id"] != on_environment.get("slurm_job_id")):
        raise ValueError("profiling-off/on runs must share the same Slurm allocation")
    if _rank_host_partition(off_environment) != _rank_host_partition(on_environment):
        raise ValueError("profiling-off/on logical rank-to-node partitions differ")
    if _physical_placement(off_environment) != _physical_placement(on_environment):
        raise ValueError("profiling-off/on physical rank-to-GPU placement differs")
    if _normalized_resolved_config(profiling_off_run) != _normalized_resolved_config(profiling_on_run):
        raise ValueError("profiling-off/on resolved training configs differ")

    off = _summary(profiling_off_run)
    on = _summary(profiling_on_run)
    training = contract.get("training", {})
    measured = training.get("measured_steps")
    interval = training.get("log_interval")
    if not isinstance(measured, int) or not isinstance(interval, int) or interval <= 0 or measured <= 0 or measured % interval:
        raise ValueError("overhead comparison lacks an exact measured timing-window contract")
    expected_windows = measured // interval
    if off.get("window_count") != expected_windows or on.get("window_count") != expected_windows:
        raise ValueError("profiling-off/on complete-step timing windows do not match the contract")
    validation_path = profiling_on_run / "validation.json"
    try:
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError("profiling-on accounting validation is missing") from error
    if validation.get("valid_profile") is not True or validation.get("expected_steps") != list(range(
        training.get("warmup_steps", -1), training.get("warmup_steps", -1) + measured
    )):
        raise ValueError("profiling-on accounting validation did not cover the complete measured window")
    off_median = float(off["median_optimizer_step_ms"])
    on_median = float(on["median_optimizer_step_ms"])
    if not all(math.isfinite(value) and value > 0 for value in (off_median, on_median)):
        raise ValueError("profiling-off/on median step times must be finite and positive")
    overhead_percent = 100.0 * (on_median - off_median) / off_median
    return {
        "valid_comparison": overhead_percent <= max_overhead_percent,
        "profiling_off_run": str(profiling_off_run),
        "profiling_on_run": str(profiling_on_run),
        "off_median_optimizer_step_ms": off_median,
        "on_median_optimizer_step_ms": on_median,
        "median_overhead_percent": overhead_percent,
        "max_overhead_percent": max_overhead_percent,
        "run_kind": run_kind,
        "claim_scope": "separate_cp2_ep8_with_cp_communication" if cp2_candidate else "cp1_ep8_canonical",
        "identity_hashes": identities,
        "note": "Only matched complete-step timing windows are compared; this does not measure component overhead.",
    }


def main() -> None:
    args = parse_args()
    result = compare(args.profiling_off_run, args.profiling_on_run, args.max_overhead_percent)
    destination = args.profiling_on_run / "analysis" / "profiler_overhead.json"
    destination.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    if not result["valid_comparison"]:
        raise SystemExit("lightweight profiler overhead exceeds the configured limit")


if __name__ == "__main__":
    main()
