"""Summarize inclusive activity from accepted profile records, without a time partition."""
from __future__ import annotations

from pathlib import Path
import statistics

from analysis.compare_uccl_profiles import phase_rows

PHASES = ("router_total", "dispatch_total", "expert_total", "combine_total")


def collect(cases, inputs):
    result = {}
    inputs.bind(Path(__file__).resolve())
    for case, record in cases.items():
        directory = Path(record["legs"]["profile"]["validation_dir"])
        acceptance = inputs.read(directory / "acceptance.json")
        if acceptance.get("valid_capture") is not True:
            raise ValueError("phase summaries require accepted captures")
        actual = Path(acceptance.get("profile_output_dir", directory))
        accepted_inputs = {row["path"]: row["sha256"] for row in acceptance["input_files"]}
        backend = "uccl" if case in ("uccl", "uccl_ordinary") else "native"
        values = {phase: {"gpu": [], "cpu": []} for phase in PHASES}
        records = []
        for rank in range(16):
            for step in range(5, 10):
                relative = (f"profiles/rank-{rank}-step-{step:05d}.json" if backend == "uccl"
                    else f"profiles/rank-{rank}/step-{step:05d}.json")
                path = actual / relative
                normalized = phase_rows(inputs.read(path), backend)
                actual_sha = inputs.bind(path)
                if accepted_inputs.get(str(path)) != actual_sha:
                    raise ValueError(f"phase record differs from accepted capture: {path}")
                records.append({"rank": rank, "step": step, "path": str(path), "sha256": actual_sha})
                for phase in PHASES:
                    values[phase]["gpu"].append(normalized[phase]["gpu_inclusive_activity_ms"] / 1000)
                    values[phase]["cpu"].append(normalized[phase]["cpu_scope_ms"] / 1000)
        result[case] = {"population": 80, "window": "measured5", "human_updates": [6, 7, 8, 9, 10], "ranks": list(range(16)),
            "unit": "inclusive activity seconds, averaged per rank and optimizer update",
            "phases": {phase: {"gpu_inclusive_s_mean_per_rank_update": statistics.fmean(values[phase]["gpu"]),
                "cpu_scope_s_mean_per_rank_update": statistics.fmean(values[phase]["cpu"])} for phase in PHASES},
            "records": records,
            "claim_limit": "Nested scopes and concurrent activity are not additive step-time partitions, bandwidth, exposed latency, or unprofiled phase costs."}
    return result
