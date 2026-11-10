"""Run existing strict validators after the four-case UCCL acquisition.

This coordinator does not calculate performance metrics or weaken backend
checks. Child analyzers run only in allocated compute and preserve their logs.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orchestration-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8, choices=range(1, 17))
    args = parser.parse_args()
    if os.environ.get("SLURM_JOB_PARTITION") != "preemptable" or not os.environ.get("SLURM_JOB_ID"):
        parser.error("fresh preemptable compute required")
    orchestration, output = args.orchestration_dir.resolve(), args.output_dir.resolve()
    if output.exists() or output.is_symlink() or output.parent != ROOT / "runs":
        parser.error("output must be a new directory directly under runs")
    acquisition = json.loads((orchestration / "acquisition.json").read_text())
    mode = acquisition.get("mode")
    if mode not in ("pilot", "full"):
        parser.error("unknown acquisition mode")
    steps = 2 if mode == "pilot" else 25
    cases = [line.split("\t") for line in (orchestration / "cases.tsv").read_text().splitlines()]
    expected = [(case, leg) for leg in (("profile",) if mode == "pilot" else ("control", "profile"))
                for case in ("ordinary", "grouped", "uccl_ordinary", "uccl")]
    if len(cases) != len(expected) or any(row[:2] != list(pair) or len(row) != 3 for row, pair in zip(cases, expected)):
        parser.error("case order differs from the reviewed acquisition")
    for _, _, run in cases:
        if Path(run).resolve().parent != ROOT / "runs" or not Path(run).is_dir():
            parser.error("case run lies outside project runs")
    output.mkdir()
    (output / "logs").mkdir()
    commands = []
    for case, leg, raw_run in cases:
        run = Path(raw_run)
        label = case if mode == "pilot" else f"{case}-{leg}"
        destination = output / label
        destination.mkdir()
        commands.append((f"{label}-received", [sys.executable, str(ROOT / "analysis/validate_ep16_received_routes.py"),
            "--run-dir", str(run), "--steps", str(steps), "--output", str(destination / "received_routes.json")]))
        if leg != "profile":
            continue
        for rank in range(16):
            if case not in ("uccl_ordinary", "uccl"):
                commands.append((f"{label}-rank-{rank}", [sys.executable, str(ROOT / "analysis/analyze_all_step_trace.py"),
                    "--run-dir", str(run), "--rank", str(rank), "--steps", str(steps)]))
            else:
                for step in range(steps):
                    commands.append((f"{label}-rank-{rank}-step-{step:05d}",
                        [sys.executable, str(ROOT / "analysis/inspect_uccl_trace.py"),
                         "--trace", str(run / f"traces/torch_profile/step-{step:05d}/rank-{rank}.json.gz"),
                         "--record", str(run / f"raw/deep_trace/step-{step:05d}/rank-{rank}.json"),
                         "--output", str(destination / f"rank-{rank}-step-{step:05d}.json")]))
    acceptance = (("pilot-acceptance", [sys.executable, str(ROOT / "analysis/accept_uccl_pilot.py"),
        "--orchestration-dir", str(orchestration), "--analysis-root", str(output),
        "--output", str(output / "pilot_acceptance.json")]) if mode == "pilot" else None)
    recorded_commands = commands + ([acceptance] if acceptance else [])
    (output / "commands.json").write_text(json.dumps(recorded_commands, indent=2))

    def execute(task):
        name, command = task
        with (output / "logs" / f"{name}.log").open("x") as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        return name

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(execute, task) for task in commands]
        try:
            for future in as_completed(futures):
                print(json.dumps({"validator": future.result(), "completed": True}), flush=True)
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    if acceptance:
        execute(acceptance)
    (output / "COMPLETE.json").write_text(json.dumps({"all_invoked_validators_passed": True,
        "mode": mode, "steps": steps, "case_count": len(cases), "commands": len(recorded_commands),
        "job": os.environ["SLURM_JOB_ID"], "orchestration_dir": str(orchestration),
        "coordinator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "claim_limit": "Full acquisition still requires matched comparison/report validation; this marker records validator completion."},
        indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
