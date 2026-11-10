#!/usr/bin/env python3
"""Extract complete optimizer-step timing windows from Bridge's normal timer-bearing log.

Megatron's ``interval-time`` timer includes a synchronization at the log
boundary and therefore measures complete optimizer-step wall time.  This tool
does not add timing hooks to training; it only selects log intervals fully
contained in the configured measurement window.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Any


_ITERATION_TIME = re.compile(
    r"\biteration\s+(?P<iteration>\d+)\s*/\s*\d+\s+\|"
    r".*?\belapsed time per iteration \(ms\):\s*(?P<milliseconds>[0-9.]+)\s*\|"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--warmup-steps", type=int, required=True)
    parser.add_argument("--measured-steps", type=int, required=True)
    parser.add_argument("--log-interval", type=int, required=True)
    parser.add_argument("--global-batch-size", type=int, required=True)
    parser.add_argument("--sequence-length", type=int, required=True)
    args = parser.parse_args()
    if args.warmup_steps < 0 or min(
        args.measured_steps, args.log_interval, args.global_batch_size, args.sequence_length
    ) <= 0:
        parser.error("warmup must be nonnegative and all other timing controls must be positive")
    if args.warmup_steps % args.log_interval or args.measured_steps % args.log_interval:
        parser.error("warmup and measured steps must be exact multiples of log interval")
    return args


def timing_rows(log_path: Path, *, warmup_steps: int, measured_steps: int, log_interval: int, tokens_per_step: int) -> list[dict[str, Any]]:
    """Return only whole logged intervals contained within the selected window."""
    if not log_path.is_file():
        raise FileNotFoundError(f"timer-bearing training log is missing: {log_path}")
    measured_end = warmup_steps + measured_steps
    rows: list[dict[str, Any]] = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = _ITERATION_TIME.search(line)
        if match is None:
            continue
        end_step = int(match.group("iteration"))
        start_step = end_step - log_interval
        if start_step < warmup_steps or end_step > measured_end:
            continue
        milliseconds = float(match.group("milliseconds"))
        if milliseconds <= 0:
            raise ValueError(f"non-positive elapsed time in {log_path}: {line}")
        seconds_per_step = milliseconds / 1000.0
        rows.append(
            {
                "timer_log": str(log_path),
                "first_step": start_step,
                "last_step_exclusive": end_step,
                "optimizer_steps": log_interval,
                "mean_optimizer_step_ms": milliseconds,
                "tokens_per_optimizer_step": tokens_per_step,
                "tokens_per_second": tokens_per_step / seconds_per_step,
            }
        )
    expected_windows = measured_steps // log_interval
    if len(rows) != expected_windows:
        raise ValueError(
            f"found {len(rows)} whole timing windows in {log_path}; expected {expected_windows}. "
            "Do not use this run for timing conclusions."
        )
    expected_ranges = [(step, step + log_interval) for step in range(warmup_steps, measured_end, log_interval)]
    actual_ranges = [(row["first_step"], row["last_step_exclusive"]) for row in rows]
    if actual_ranges != expected_ranges:
        raise ValueError(f"timing windows are {actual_ranges}, expected {expected_ranges}")
    return rows


def select_timer_log(logs_dir: Path) -> Path:
    """Return the one Bridge rank log that emitted interval timers.

    Bridge uses ``print_rank_last`` for its progress/timer lines.  The last
    global rank is a runtime property, so selecting ``rank-0.out`` would
    silently reject valid jobs or, worse, parse a non-authoritative log.  A
    normal run has exactly one timer-bearing rank log; ambiguity invalidates
    timing evidence instead of guessing a rank.
    """
    candidates = [
        path
        for path in sorted(logs_dir.glob("rank-*.out"))
        if _ITERATION_TIME.search(path.read_text(encoding="utf-8", errors="replace")) is not None
    ]
    if len(candidates) != 1:
        names = [str(path) for path in candidates]
        raise ValueError(
            f"found {len(candidates)} timer-bearing rank logs in {logs_dir}: {names}; "
            "expected exactly one print_rank_last log"
        )
    return candidates[0]


def write_outputs(run_dir: Path, rows: list[dict[str, Any]]) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ModuleNotFoundError as error:
        raise RuntimeError("pyarrow is required to write timing_windows.parquet") from error
    metrics_dir = run_dir / "metrics"
    analysis_dir = run_dir / "analysis"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), metrics_dir / "timing_windows.parquet")
    values = [float(row["mean_optimizer_step_ms"]) for row in rows]
    token_rates = [float(row["tokens_per_second"]) for row in rows]
    summary = {
        "timing_unit": "complete optimizer-step wall-time averaged over each logged window",
        "window_count": len(rows),
        "mean_optimizer_step_ms": statistics.fmean(values),
        "median_optimizer_step_ms": statistics.median(values),
        "min_optimizer_step_ms": min(values),
        "max_optimizer_step_ms": max(values),
        "mean_tokens_per_second": statistics.fmean(token_rates),
        "note": "These are existing Megatron interval-timer windows; they are not isolated communication timings.",
    }
    (analysis_dir / "optimizer_step_breakdown.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    rows = timing_rows(
        select_timer_log(args.run_dir / "logs"),
        warmup_steps=args.warmup_steps,
        measured_steps=args.measured_steps,
        log_interval=args.log_interval,
        tokens_per_step=args.global_batch_size * args.sequence_length,
    )
    write_outputs(args.run_dir, rows)


if __name__ == "__main__":
    main()
