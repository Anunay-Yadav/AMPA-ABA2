#!/usr/bin/env python3
"""Summarize ordinary Bridge real-training logs without changing training."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import statistics

try:
    from analysis.extract_step_timing import _ITERATION_TIME, select_timer_log
except ModuleNotFoundError:
    from extract_step_timing import _ITERATION_TIME, select_timer_log


FIELDS = {
    "consumed_samples": r"consumed samples:\s*(\d+)",
    "learning_rate": r"learning rate:\s*([0-9.Ee+-]+)",
    "global_batch_size": r"global batch size:\s*(\d+)",
    "lm_loss": r"lm loss:\s*([0-9.Ee+-]+)",
    "load_balancing_loss": r"load_balancing_loss:\s*([0-9.Ee+-]+)",
    "grad_norm": r"grad norm:\s*([0-9.Ee+-]+)",
    "skipped_iterations": r"number of skipped iterations:\s*(\d+)",
    "nan_iterations": r"number of nan iterations:\s*(\d+)",
}
INTEGERS = {"consumed_samples", "global_batch_size", "skipped_iterations", "nan_iterations"}


def summarize(run_dir: Path, *, steps: int = 25, interval: int = 5, sequence_length: int = 4096) -> dict:
    log_path = select_timer_log(run_dir / "logs")
    rows = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = _ITERATION_TIME.search(line)
        if match is None:
            continue
        row = {"iteration": int(match.group("iteration"))}
        for name, pattern in FIELDS.items():
            value = re.search(pattern, line)
            if value is None:
                raise ValueError(f"iteration {row['iteration']}: missing {name}")
            row[name] = int(value.group(1)) if name in INTEGERS else float(value.group(1))
        rows.append(row)
    expected = list(range(interval, steps + 1, interval))
    if [row["iteration"] for row in rows] != expected:
        raise ValueError(f"logged iterations {[r['iteration'] for r in rows]}, expected {expected}")
    for row in rows:
        if row["consumed_samples"] != row["iteration"] * row["global_batch_size"]:
            raise ValueError(f"iteration {row['iteration']}: consumed-sample accounting failed")
        if row["skipped_iterations"] or row["nan_iterations"]:
            raise ValueError(f"iteration {row['iteration']}: skipped or NaN iteration recorded")
        if not all(math.isfinite(row[name]) for name in ("learning_rate", "lm_loss", "load_balancing_loss", "grad_norm")):
            raise ValueError(f"iteration {row['iteration']}: nonfinite optimization metric")
        if row["learning_rate"] <= 0 or row["grad_norm"] <= 0:
            raise ValueError(f"iteration {row['iteration']}: nonpositive learning rate or gradient norm")
    if len({row["global_batch_size"] for row in rows}) != 1:
        raise ValueError("global batch size changed during the run")
    values = {name: [row[name] for row in rows] for name in ("lm_loss", "load_balancing_loss", "grad_norm")}
    result = {
        "schema_version": 1,
        "evidence_kind": "ordinary_Bridge_real_training_interval_logs",
        "run_id": run_dir.name,
        "timer_log": str(log_path),
        "completed_logged_optimizer_steps": steps,
        "global_batch_size": rows[0]["global_batch_size"],
        "final_consumed_samples": rows[-1]["consumed_samples"],
        "final_consumed_global_tokens": rows[-1]["consumed_samples"] * sequence_length,
        "logged_boundaries": rows,
        "ranges": {name: {"min": min(data), "max": max(data), "mean_of_logged_values": statistics.fmean(data)} for name, data in values.items()},
        "claim_limit": "Logged loss is a Bridge interval statistic; finite gradient norms and zero skipped iterations support real optimizer-step execution, not direct proof that every parameter changed.",
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--log-interval", type=int, default=5)
    args = parser.parse_args()
    result = summarize(args.run_dir, interval=args.log_interval)
    output = args.run_dir / "analysis" / "real_training_evidence.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"run_id": result["run_id"], "loss_range": result["ranges"]["lm_loss"], "grad_norm_range": result["ranges"]["grad_norm"]}))


if __name__ == "__main__":
    main()
