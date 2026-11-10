#!/usr/bin/env python3
"""Compare a one-step all-rank deep trace with its same-allocation control."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from compare_profile_overhead import (
    _environment, _normalized_resolved_config, _physical_placement, _sha256,
)
from extract_step_timing import select_timer_log, timing_rows
from summarize_training_evidence import summarize
from tools.qwen_workload import workload


_FINGERPRINT_DTYPES = frozenset(f"torch.{name}" for name in (
    "bool", "uint8", "uint16", "uint32", "uint64", "int8", "int16", "int32", "int64",
    "float16", "bfloat16", "float32", "float64", "complex32", "complex64", "complex128",
))


def _mode(run: Path, expected_trace: bool) -> None:
    path = run / "config" / "observation_mode.json"
    mode = json.loads(path.read_text(encoding="utf-8"))
    if (mode.get("data_mode") != "indexed" or mode.get("lightweight_profiling") != "off"
            or mode.get("nsys_trace", False) is not False
            or mode.get("torch_trace", False) is not expected_trace):
        raise ValueError(f"wrong observation mode in {path}")
    if mode.get("sample_fingerprints") is not True:
        raise ValueError(f"same-token pair lacks actual consumed-batch fingerprints: {path}")
    if any(mode.get(key) is not False for key in (
        "router_observer_attached", "route_wrapper_installed", "dispatcher_preprocess_wrapper_installed"
    )):
        raise ValueError(f"lightweight observer unexpectedly attached in {path}")


def _training_resolved(run: Path) -> str:
    """Normalize exactly the four intentional profiler control differences."""
    lines = _normalized_resolved_config(run).splitlines()
    retained = []
    in_profiler = False
    allowed = {"use_pytorch_profiler", "profile_step_start", "profile_step_end", "profile_ranks"}
    seen = set()
    skip_rank_items = False
    for line in lines:
        if line == "profiling:":
            in_profiler = True
            retained.append(line)
            continue
        if in_profiler and line and not line.startswith(" "):
            in_profiler = False
        if in_profiler and skip_rank_items and line.startswith("  - "):
            continue
        skip_rank_items = False
        if in_profiler:
            key = line.strip().partition(":")[0]
            if key in allowed:
                if key in seen:
                    raise ValueError(f"duplicate profiler control {key}")
                seen.add(key)
                retained.append(f"  {key}: <CONTROL_OR_TRACE>")
                skip_rank_items = key == "profile_ranks"
                continue
        retained.append(line)
    if seen != allowed:
        raise ValueError(f"resolved profiler controls missing: {sorted(allowed - seen)}")
    return "\n".join(retained)


def _checked_fingerprint_batches(record, *, rank, accumulation, steps, token_shape, path):
    if (not isinstance(record, dict) or type(record.get("schema_version")) is not int
            or record["schema_version"] != 1
            or record.get("kind") != "consumed_cpu_token_batch_sha256"
            or type(record.get("rank")) is not int or record["rank"] != rank):
        raise ValueError(f"rank {rank}: invalid actual-batch fingerprint schema in {path}")
    records = record.get("steps")
    if not isinstance(records, list) or len(records) != steps:
        raise ValueError(f"rank {rank}: incomplete actual-batch fingerprint record in {path}")
    batches = []
    for step, row in enumerate(records):
        if (not isinstance(row, dict) or type(row.get("step")) is not int or row["step"] != step):
            raise ValueError(f"rank {rank}: incomplete actual-batch fingerprint record in {path}")
        microbatches = row.get("batches")
        if not isinstance(microbatches, list) or len(microbatches) != accumulation:
            raise ValueError(f"rank {rank}: expected {accumulation} microbatches per optimizer step in {path}")
        for batch in microbatches:
            if not isinstance(batch, dict) or not batch or "tokens" not in batch:
                raise ValueError(f"rank {rank}: actual-batch fingerprint lacks tokens in {path}")
            for name, tensor in batch.items():
                if (not isinstance(name, str) or not name or not isinstance(tensor, dict)
                        or set(tensor) != {"sha256", "shape", "dtype"}):
                    raise ValueError(f"rank {rank}: invalid tensor fingerprint metadata in {path}")
                digest, shape, dtype = tensor["sha256"], tensor["shape"], tensor["dtype"]
                if (not isinstance(digest, str) or len(digest) != 64
                        or any(character not in "0123456789abcdef" for character in digest)
                        or not isinstance(shape, list)
                        or any(type(dimension) is not int or dimension < 0 for dimension in shape)
                        or not isinstance(dtype, str) or dtype not in _FINGERPRINT_DTYPES):
                    raise ValueError(f"rank {rank}: invalid tensor fingerprint metadata in {path}")
            if batch["tokens"]["shape"] != token_shape or batch["tokens"]["dtype"] != "torch.int64":
                raise ValueError(f"rank {rank}: token fingerprint differs from the reviewed CPU workload in {path}")
        batches.append(microbatches)
    return batches


def _compare_consumed_batches(control: Path, trace: Path, *, world_size: int = 8,
                              accumulation: int = 16, steps: int = 25) -> dict:
    variants = [workload(variant) for variant in ("cp1_ep8", "cp2_ep8", "cp1_ep16", "cp2_ep16")]
    spec = next((item for item in variants
                 if (item["world_size"], item["accumulation"]) == (world_size, accumulation)), None)
    if (any(type(value) is not int for value in (world_size, accumulation, steps))
            or spec is None or steps not in (2, 25)):
        raise ValueError("unsupported consumed-batch comparison dimensions")
    # Fingerprints observe CPU inputs before context-parallel slicing.
    token_shape = [1, spec["token_positions"] * spec["cp"]]
    hashes = {}
    compared = 0
    for rank in range(world_size):
        rows = []
        for run in (control, trace):
            path = run / "raw" / f"sample_fingerprint_rank-{rank}.json"
            record = json.loads(path.read_text(encoding="utf-8"))
            batches = _checked_fingerprint_batches(
                record, rank=rank, accumulation=accumulation, steps=steps,
                token_shape=token_shape, path=path,
            )
            hashes[f"{run.name}:rank-{rank}"] = _sha256(path)
            rows.append(batches)
        if rows[0] != rows[1]:
            raise ValueError(f"rank {rank}: control and trace consumed different token batches or order")
        compared += steps * accumulation
    return {"all_ranks_exact_pre_cp_cpu_batches_equal": True, "compared_rank_microbatches": compared,
            "fingerprint_file_sha256": hashes}


def compare(control: Path, trace: Path) -> dict:
    _mode(control, False)
    _mode(trace, True)
    for name in ("comparison_contract.json", "c0_manifest.json", "dataset_manifest.json",
                 "model_manifest.json"):
        if _sha256(control / "config" / name) != _sha256(trace / "config" / name):
            raise ValueError(f"{name} differs between control and deep trace")
    contract = json.loads((control / "config" / "comparison_contract.json").read_text(encoding="utf-8"))
    if (contract.get("parallelism", {}).get("cp") != 2
            or contract.get("parallelism", {}).get("ep") != 8
            or contract.get("training", {}).get("log_interval") != 1
            or contract.get("training", {}).get("warmup_steps") != 5
            or contract.get("training", {}).get("measured_steps") != 20):
        raise ValueError("deep-trace pair requires the reviewed CP2, 5+20, one-step logging contract")
    if contract.get("sample_fingerprints") is not True:
        raise ValueError("deep-trace pair contract lacks actual-batch fingerprints")
    a, b = _environment(control, cp=2), _environment(trace, cp=2)
    if not a.get("slurm_job_id") or a["slurm_job_id"] != b.get("slurm_job_id"):
        raise ValueError("deep-trace control and capture did not share a Slurm allocation")
    if _physical_placement(a) != _physical_placement(b):
        raise ValueError("deep-trace control and capture used different physical GPUs")
    for field in (
        "pytorch_version", "cuda_version", "cuda_allocator_conf", "nccl_version", "gpu_name",
        "container_sqsh_sha256", "megatron_lm", "megatron_bridge", "ampa_runtime_sources",
        "checkpoint", "dataset_blend_per_split", "parallelism", "activation_recomputation",
    ):
        if a.get(field) is None or a[field] != b.get(field):
            raise ValueError(f"deep-trace pair has different runtime {field}")
    if _training_resolved(control) != _training_resolved(trace):
        raise ValueError("resolved training configuration differs outside profiler controls")
    validation = json.loads((trace / "analysis" / "torch_trace_validation.json").read_text(encoding="utf-8"))
    if (validation.get("valid_capture") is not True or validation.get("requested_ranks") != list(range(8))
            or validation.get("step") != 5 or validation.get("capture_steps") != 1):
        raise ValueError("deep trace has no valid all-rank capture")
    validation_records = validation.get("ranks", [])
    if len(validation_records) != 8 or {record.get("rank") for record in validation_records} != set(range(8)):
        raise ValueError("deep trace validation does not contain exactly eight distinct ranks")
    for record in validation_records:
        for path_key, digest_key in (("trace_path", "trace_sha256"), ("call_record_path", "call_record_sha256")):
            if _sha256(Path(record[path_key])) != record[digest_key]:
                raise ValueError(f"validated rank {record['rank']} {path_key} changed")
    index_path = trace / "analysis" / "moe_deep_timing_index.json"
    if not index_path.is_file():
        raise ValueError("all-rank trace aggregation did not finish")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    indexed = index.get("ranks", [])
    if (index.get("claim_scope") != "all_rank_diagnostic"
            or index.get("analysis_source_sha256") != validation.get("analysis_source_sha256")
            or len(indexed) != 8
            or {record.get("rank") for record in indexed} != set(range(8))):
        raise ValueError("timing index is not a validated, source-matched eight-rank diagnostic")
    for record in indexed:
        summary_path = Path(record["summary_path"])
        if not summary_path.is_file() or _sha256(summary_path) != record["summary_sha256"]:
            raise ValueError(f"rank {record['rank']} timing summary changed after aggregation")
    # A timed iteration is usable only when actual continued pretraining also
    # logged finite losses/gradients and no skipped or NaN updates in both legs.
    for run in (control, trace):
        summarize(run, steps=25, interval=1)
    consumed_batches = _compare_consumed_batches(control, trace)
    rows = {}
    for name, run in (("control", control), ("trace", trace)):
        records = timing_rows(select_timer_log(run / "logs"), warmup_steps=5,
                              measured_steps=20, log_interval=1, tokens_per_step=64 * 4096)
        rows[name] = {row["first_step"]: row["mean_optimizer_step_ms"] for row in records}
    control_selected, trace_selected = rows["control"][5], rows["trace"][5]
    other_steps = list(range(7, 25))
    other_control = statistics.median(rows["control"][step] for step in other_steps)
    other_trace = statistics.median(rows["trace"][step] for step in other_steps)
    return {
        "valid_comparison": True,
        "control_run": str(control), "trace_run": str(trace),
        "traced_step": 5,
        "control_step_5_ms": control_selected,
        "trace_step_5_ms": trace_selected,
        "selected_step_slowdown_percent": 100 * (trace_selected / control_selected - 1),
        "step_6_control_ms": rows["control"][6],
        "step_6_trace_ms": rows["trace"][6],
        "post_export_median_control_ms": other_control,
        "post_export_median_trace_ms": other_trace,
        "post_export_median_difference_percent": 100 * (other_trace / other_control - 1),
        "control_steps_ms": rows["control"], "trace_steps_ms": rows["trace"],
        "consumed_batches": consumed_batches,
        "claim_limit": "Both legs hash the same CPU batches before H2D transfer and CP slicing; that hash has matched CPU cost. Deep profiling perturbs the captured step. This ratio is not a correction factor for individual component timings.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-run", required=True, type=Path)
    parser.add_argument("--trace-run", required=True, type=Path)
    args = parser.parse_args()
    result = compare(args.control_run, args.trace_run)
    output = args.trace_run / "analysis" / "deep_trace_overhead.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"selected_step_slowdown_percent": result["selected_step_slowdown_percent"],
                      "post_export_median_difference_percent": result["post_export_median_difference_percent"]}))


if __name__ == "__main__":
    main()
