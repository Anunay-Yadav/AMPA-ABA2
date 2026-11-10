"""Gate full acquisition on two healthy real updates in each routing mode."""
import argparse
import json
from pathlib import Path
from summarize_training_evidence import summarize


def check(run: Path) -> dict:
    health = summarize(run, steps=2, interval=1)
    contract = json.loads((run / "config/comparison_contract.json").read_text())
    cp = contract["parallelism"]["cp"]
    ga = 4 if cp == 1 else 8
    expected = {"tp": 1, "pp": 1, "cp": cp, "ep": 16, "etp": 1, "dense_dp": 16 // cp}
    if cp not in (1, 2) or contract["parallelism"] != expected or contract["training"]["completed_steps"] != 2:
        raise ValueError("not a reviewed EP16 two-update smoke")
    capture = json.loads((run / "config/all_step_capture.json").read_text())
    if capture["enabled"] is not True or capture["steps"] != [0, 1] or capture["ranks"] != list(range(16)):
        raise ValueError("smoke capture coverage mismatch")
    for rank in range(16):
        for name in ("routing_audit", "sample_fingerprint"):
            raw = json.loads((run / "raw" / f"{name}_rank-{rank}.json").read_text())
            if ((name == "routing_audit" and raw.get("valid") is not True)
                    or raw.get("rank") != rank or [row["step"] for row in raw["steps"]] != [0, 1]):
                raise ValueError(f"rank {rank}: incomplete {name}")
            if name == "sample_fingerprint" and any(len(row["batches"]) != ga for row in raw["steps"]):
                raise ValueError("incorrect smoke gradient accumulation")
        for step in (0, 1):
            for relative in (f"traces/torch_profile/step-{step:05d}/rank-{rank}.json.gz",
                             f"raw/deep_trace/step-{step:05d}/rank-{rank}.json"):
                if not (run / relative).is_file():
                    raise ValueError(f"missing smoke artifact: {relative}")
    return {"valid": True, "cp": cp, "health": health, "ranks": 16, "steps": 2}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    result = check(args.run_dir)
    (args.run_dir / "analysis/smoke_health.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps({"valid": True, "cp": result["cp"], "ranks": 16, "steps": 2}))
