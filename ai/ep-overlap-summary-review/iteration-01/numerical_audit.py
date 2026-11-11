#!/usr/bin/env python3
"""Independent arithmetic and coverage audit. Execute on allocated compute."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[3]
LABELS = {f"{backend}-{routing}-{mode}" for backend in ("native", "uccl")
          for routing in ("ordinary", "grouped") for mode in ("off", "torch")}
LABELS |= {f"uccl-ordinary-nsys{budget}" for budget in (12, 20, 32, 40)}
METRICS = ("SMs Active [Throughput %]", "SM Issue [Throughput %]", "Compute Warps in Flight [Throughput %]")
TOKENS = 262144
HASHES = {}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, message):
    need(math.isfinite(float(actual)) and math.isclose(float(actual), expected, rel_tol=1e-12, abs_tol=1e-9), message)


def contents(path):
    path = Path(path)
    need(path.is_file() and not path.is_symlink(), f"regular input required: {path}")
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    need(str(path) not in HASHES or HASHES[str(path)] == digest, f"input changed: {path}")
    HASHES[str(path)] = digest
    return payload


def record(path):
    return json.loads(contents(path))


def table(path):
    return list(csv.DictReader(contents(path).decode().splitlines()))


def stats(values):
    need(len(values) >= 2 and all(math.isfinite(value) and value > 0 for value in values), "positive finite timers")
    mean = sum(values) / len(values)
    ordered = sorted(values)
    middle = len(values) // 2
    median = ordered[middle] if len(values) % 2 else sum(ordered[middle - 1:middle + 1]) / 2
    return {"count": len(values), "mean_ms": mean, "median_ms": median,
            "sample_sd_ms": math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1)),
            "min_ms": min(values), "max_ms": max(values), "aggregate_tokens_per_second": len(values) * TOKENS * 1000 / sum(values)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    need(os.environ.get("SLURM_JOB_PARTITION") == "debug" and os.environ.get("SLURM_JOB_ID", "").isdigit()
         and Path(sys.prefix) == Path("/opt/venvs/megatron-bridge"), "allocated debug configured Python required")
    need(args.summary_dir.is_absolute() and args.output.parent == args.summary_dir
         and not args.output.exists() and not args.output.is_symlink(), "new output inside summary directory required")
    contents(Path(__file__).resolve())
    summary = record(args.summary_dir / "campaign_summary.json")
    need(summary["valid_twelve_case_summary"] is True and len(summary["cases"]) == 12, "accepted twelve-case summary")
    cases = {row["label"]: row for row in summary["cases"]}
    need(set(cases) == LABELS, "exact twelve labels")
    rows = table(args.summary_dir / "optimizer_steps.csv")
    flat_rows = table(args.summary_dir / "eight_cases.csv") + table(args.summary_dir / "sm_budgets.csv")
    need(len(rows) == 120 and len(flat_rows) == 12, "120 timer rows and twelve table rows")
    flat = {row["label"]: row for row in flat_rows}
    need(set(flat) == LABELS and {row["label"] for row in rows} == LABELS, "exact CSV label sets")
    accounting = {}
    for line in contents(args.summary_dir / "evidence/accounting_utc.psv").decode().splitlines():
        fields = line.rstrip("|").split("|")
        need(len(fields) == 9 and fields[1:3] == ["COMPLETED", "0:0"] and fields[7] == "debug", "accepted accounting schema")
        need(fields[0] not in accounting, "unique accounting jobs")
        accounting[fields[0]] = int(fields[5])
    need(len(accounting) == 24, "24 distinct accepted allocations")
    results, hardware, captures, windows, prefixes = {}, {}, 0, 0, set()
    for label, case in cases.items():
        raw = record(Path(case["proof"]).parent / "timing.json")["all_updates"]
        need([row["first_step"] for row in raw] == list(range(10)), "ten ordered raw update timers")
        selected = sorted((row for row in rows if row["label"] == label), key=lambda row: int(row["human_update"]))
        need(len(selected) == 10 and [int(row["human_update"]) for row in selected] == list(range(1, 11)), "ten unique CSV updates")
        need([row["measured"] for row in selected] == ["False"] * 5 + ["True"] * 5, "five warmup and five measured updates")
        need(all(row["training_job"] == case["training_job"] and row["posthoc_job"] == case["posthoc_job"]
                 and row["run_id"] == case["id"] and row["capture"] == case["capture"] for row in selected), "CSV case identities")
        values = [float(row["mean_optimizer_step_ms"]) for row in raw]
        for index, row in enumerate(selected):
            close(row["optimizer_step_ms"], values[index], "raw optimizer timer copy")
            close(row["tokens_per_second"], TOKENS * 1000 / values[index], "per-update throughput")
            need(int(row["tokens_per_step"]) == TOKENS, "tokens per complete update")
        measured = stats(values[5:])
        for name, value in measured.items():
            close(case["timing"]["measured_five_statistics"][name], value, f"measured statistic {name}")
        for name, value in stats(values).items():
            close(case["timing"]["all_ten_statistics"][name], value, f"all-update statistic {name}")
        for column, key in (("mean_step_ms", "mean_ms"), ("median_step_ms", "median_ms"), ("sample_sd_ms", "sample_sd_ms"),
                            ("aggregate_tokens_per_second", "aggregate_tokens_per_second")):
            close(flat[label][column], measured[key], f"CSV statistic {column}")
        close(flat[label]["all_ten_timer_seconds"], sum(values) / 1000, "ten timer sum")
        close(case["timing"]["optimizer_update_sum_seconds"], sum(values) / 1000, "JSON ten timer sum")
        for role in ("training", "posthoc"):
            elapsed = accounting[case[role + "_job"]]
            close(flat[label][role + "_allocation_seconds"], elapsed, "CSV allocation elapsed")
            close(case["timing"][role + "_allocation_elapsed_seconds"], elapsed, "JSON allocation elapsed")
        prefixes.add(case["input_prefix_sha256"])
        if case["capture"] == "torch":
            gpu = record(Path(case["proof"]).parent / "gpu_activity/summary.json")
            need(len(gpu["captures"]) == gpu["capture_count"] == 160, "160 captures per Torch case")
            need({(row["rank"], row["step"]) for row in gpu["captures"]} == {(rank, step) for rank in range(16) for step in range(10)}, "full Torch grid")
            captures += len(gpu["captures"])
        if case["capture"] == "nsys":
            metrics = record(ROOT / "runs" / (case["id"] + "-nsys-posthoc") / "metrics.json")
            need(len(metrics["rank_update_windows"]) == 80 and metrics["valid_bound_metric_analysis"] is True, "80 accepted hardware windows")
            need({(row["rank"], row["step_index"]) for row in metrics["rank_update_windows"]} == {(rank, step) for rank in range(16) for step in range(5, 10)}, "full hardware grid")
            windows += 80
            for view, pooled in metrics["pooled_sample_summaries"].items():
                need(case["hardware"]["views"][view] == pooled, "hardware metric copies and availability")
                if pooled["all_80_windows_available"]:
                    for name in METRICS:
                        pieces = [row["views"][view]["metrics"][name] for row in metrics["rank_update_windows"]]
                        count = sum(item["count"] for item in pieces)
                        need(count == pooled["metrics"][name]["count"] > 0, "pooled sample count")
                        close(pooled["metrics"][name]["arithmetic_sample_mean_percent"], sum(item["count"] * item["arithmetic_sample_mean_percent"] for item in pieces) / count, "sample-count-weighted mean")
            hardware[label] = metrics["pooled_sample_summaries"]["during_any_payload"]
            for name in METRICS:
                close(flat[label][name + " during payload mean"], hardware[label]["metrics"][name]["arithmetic_sample_mean_percent"], "SM CSV mean copy")
        results[label] = {"ten_timer_seconds": sum(values) / 1000, "measured_five": measured}
    old = record(ROOT / "runs/qwen-debug-results-20261105T0925Z-cpu-validation/campaign_acceptance.json")
    need(captures == 640 and windows == 320 and old["valid_debug_campaign"] is True, "capture/window totals and old acceptance")
    need(prefixes == {old["sample_prefix_sha256"]}, "all full prefixes match original campaign")
    for path in list(HASHES):
        contents(path)
    output = {"valid_independent_numeric_audit": True, "cases": 12, "optimizer_updates": 120, "measured_updates": 60,
              "torch_captures": captures, "hardware_windows": windows, "original_input_prefix_sha256": next(iter(prefixes)),
              "timing_results": results, "hardware_during_payload": hardware, "input_hashes": HASHES,
              "execution": {"job": os.environ["SLURM_JOB_ID"], "partition": "debug", "hostname": socket.gethostname()},
              "claim_limit": "Arithmetic and copied evidence only. Existing producer validations establish raw event attribution. No causal speedup or free-SM claim."}
    with args.output.open("x") as stream:
        json.dump(output, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"valid_independent_numeric_audit": True, "output": str(args.output)}))


if __name__ == "__main__":
    main()
