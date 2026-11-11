#!/usr/bin/env python3
"""Compare unprofiled same-allocation pairs and plan a bounded adaptive sweep.

All runtime entrypoints require allocated compute. A pair is one allocation
replicate. Five measured updates are subsamples, not five independent runs.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import statistics
import sys

import yaml

from analysis import accept_uccl_pilot as evidence
from profiling import run_pretrain_ep_overlap_budget as adapter
from tools.ep_overlap_campaign_summary import read_accounting

ROOT = Path(__file__).resolve().parents[1]
require, equal = evidence.require, evidence.equal
TOLERANCE = 0.02
INITIAL_PLAN = ((12, "reference-first"), (32, "candidate-first"), (40, "reference-first"),
                (12, "candidate-first"), (32, "reference-first"), (40, "candidate-first"))
ORDERS = ("reference-first", "candidate-first")


def write_new(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def finite_positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def bound_inputs(record, inputs):
    rows = record.get("input_files")
    require(type(rows) is list and rows, "proof lacks raw input bindings")
    seen = set()
    for row in rows:
        require(type(row) is dict and set(row) == {"path", "sha256"}, "invalid input binding")
        path = Path(row["path"])
        require(path.is_absolute() and str(path) not in seen, "duplicate or relative proof input")
        seen.add(str(path))
        inputs.bind(path, row["sha256"])


def placement_identity(rows):
    fields = ("rank", "hostname", "slurm_node_id", "local_rank", "cuda_current_device",
              "physical_gpu_uuid", "process_groups", "local_expert_indices")
    require(type(rows) is list and len(rows) == 16, "sixteen runtime placements required")
    require(sorted(row["rank"] for row in rows) == list(range(16)), "runtime rank grid differs")
    return [{field: row[field] for field in fields} for row in sorted(rows, key=lambda row: row["rank"])]


def normalized_config(configuration, run, budget):
    result = copy.deepcopy(configuration)
    equal(result["model"].pop("moe_flex_dispatcher_num_sms"), budget, "actual resolved budget")
    for name, value in {"tensorboard_dir": str(run / "logs/tensorboard"),
                        "wandb_exp_name": run.name, "wandb_save_dir": str(run / "logs/wandb")}.items():
        equal(result["logger"].pop(name), value, "actual per-case output path")
    return result


def checked_host_wall(record, start, leg, pair):
    for name, value in {"pair_id": pair["pair_id"], "experiment_id": leg["experiment_id"],
        "pair_leg": leg["role"], "leg_index": leg["index"], "communication_sms": leg["budget"],
        "training_slurm_job_id": pair["training_slurm_job_id"],
        "source_manifest_sha256": pair["source_manifest_sha256"], "allocation_scope": "shared_pair",
        "per_case_slurm_elapsed_available": False}.items():
        equal(record.get(name), value, f"per-case host wall {name}")
        equal(start.get(name), value, f"per-case wall start {name}")
    for name in start:
        if name != "kind":
            equal(record.get(name), start[name], "wall completion retains original start")
    equal(record.get("workers_srun_returned_zero"), True, "completed case workers")
    equal(record.get("health_timing_srun_returned_zero"), True, "completed case health/timing")
    begin, end = int(record["start_epoch_ns"]), int(record["end_epoch_ns"])
    require(end > begin and int(record["elapsed_nanoseconds"]) == end - begin, "wall interval is invalid")
    for key, nanoseconds in (("start_utc", begin), ("end_utc", end)):
        stamp = datetime.fromisoformat(record[key].replace("Z", "+00:00"))
        require(stamp.utcoffset() is not None and stamp.utcoffset().total_seconds() == 0
                and abs(stamp.timestamp() - nanoseconds / 1e9) <= 1.0, "UTC wall stamp differs from recorded epoch")
    elapsed = (end - begin) / 1e9
    require(finite_positive(record.get("host_elapsed_seconds"))
            and math.isclose(record["host_elapsed_seconds"], elapsed, rel_tol=1e-12), "wall elapsed differs")
    monotonic = float(record["end_host_uptime_seconds"]) - float(record["start_host_uptime_seconds"])
    require(monotonic > 0 and math.isclose(record["monotonic_host_elapsed_seconds"], monotonic, abs_tol=1e-8)
            and abs(monotonic - elapsed) <= 1.0, "host clock interval differs from monotonic interval")
    return elapsed


def checked_completed_case_ids(observed, expected):
    require(type(observed) is list and len(observed) == 2
            and all(type(value) is str for value in observed) and len(set(observed)) == 2,
            "completion requires two distinct actual case IDs")
    equal(sorted(observed), sorted(expected), "both actually executed independent case IDs")


def compare_pair(pair_dir, output_dir):
    pair_dir, output_dir = Path(pair_dir), Path(output_dir)
    require(pair_dir.is_absolute() and pair_dir.parent == ROOT / "runs" and pair_dir.is_dir()
            and not pair_dir.is_symlink(), "invalid pair directory")
    require(output_dir.is_absolute() and output_dir.parent == ROOT / "runs" and not output_dir.exists()
            and not output_dir.is_symlink(), "new pair output directory required")
    inputs = evidence.Inputs()
    inputs.bind(Path(__file__))
    pair = inputs.read(pair_dir / "pair.json")
    for name, value in {"schema_version": 1, "kind": "ampa_ep_overlap_budget_pair",
        "experiment_protocol": adapter.PROTOCOL, "partition": "debug", "requested_nodes": 4,
        "requested_time_limit": "00:21:00", "reference_budget": 20,
        "capture": "off", "case": "uccl_ordinary", "per_case_slurm_elapsed_available": False}.items():
        equal(pair.get(name), value, f"pair metadata {name}")
    equal(pair.get("pair_id"), pair_dir.name, "pair directory identity")
    require(adapter.allowed_budget(pair.get("candidate_budget")) and pair["candidate_budget"] != 20,
            "candidate requires a supported nonreference budget")
    require(pair.get("order") in ORDERS and pair.get("execution_steps") in (2, 10), "invalid pair order or length")
    equal(pair.get("qualification_only"), pair["execution_steps"] == 2, "pair qualification role")
    completed = inputs.read(pair_dir / "pair_completed.json")
    for name in pair:
        equal(completed.get(name), pair[name], "pair completion retains acquisition metadata")
    for name, value in {"success": True, "completed_legs": 2, "both_workers_and_health_srun_returned_zero": True}.items():
        equal(completed.get(name), value, "complete independent pair")
    source = adapter.verify_source_manifest(pair_dir / "campaign-source.sha256", pair["source_manifest_sha256"])
    inputs.bind(pair_dir / "campaign-source.sha256", source["sha256"])
    for name, digest in source["files"].items():
        inputs.bind(ROOT / name, digest)
    legs = pair.get("legs")
    require(type(legs) is list and len(legs) == 2, "two independent legs required")
    checked_completed_case_ids(completed.get("completed_case_ids"), [leg["experiment_id"] for leg in legs])
    roles = ["candidate", "reference"] if pair["order"] == "candidate-first" else ["reference", "candidate"]
    equal([leg.get("role") for leg in legs], roles, "recorded case order")
    proofs, timing, walls, identities, output_legs = {}, {}, {}, {}, []
    for index, leg in enumerate(legs):
        role = leg["role"]
        budget = pair["candidate_budget"] if role == "candidate" else 20
        equal((leg.get("index"), leg.get("budget")), (index, budget), "pair leg index and budget")
        run_id = f"{pair_dir.name}-{role}-b{budget}"
        root, run = ROOT / "runs" / run_id, ROOT / "runs" / f"{run_id}-training"
        equal((leg.get("experiment_id"), leg.get("orchestration_dir"), leg.get("run_dir")),
              (run_id, str(root), str(run)), "independent leg paths")
        proof_path = Path(str(root) + "-posthoc") / "execution_accounting.json"
        proof = inputs.read(proof_path)
        for name in ("valid_execution_and_accounting", "valid_training_health", "valid_send_receive_audit",
                     "valid_all_rank_startup_and_completion", "valid_actual_budget_runtime_guards"):
            equal(proof.get(name), True, f"strict leg acceptance {name}")
        for name, value in {"communication_sms": budget, "capture": "off", "case": "uccl_ordinary",
            "source_manifest_sha256": source["sha256"], "training_slurm_job_id": pair["training_slurm_job_id"],
            "completed_updates": pair["execution_steps"], "qualification_only": pair["qualification_only"],
            "orchestration_dir": str(root), "run_dir": str(run)}.items():
            equal(proof.get(name), value, f"actual leg identity {name}")
        equal(proof.get("execution", {}).get("slurm_job_id"), os.environ["SLURM_JOB_ID"], "same posthoc allocation")
        bound_inputs(proof, inputs)
        proofs[role] = proof
        placement = placement_identity(proof["rank_placement"])
        cfg_path = run / "config/resolved.yaml"
        inputs.bind(cfg_path)
        cfg = normalized_config(yaml.safe_load(cfg_path.read_text()), run, budget)
        identities[role] = {"placement": placement, "inputs": proof["immutable_inputs"],
            "prefix": proof["consumed_input_fingerprint_sha256"], "resolved": cfg}
        timing[role] = inputs.read(proof_path.parent / "timing.json")
        equal(timing[role].get("count"), 1 if pair["qualification_only"] else 5, "actual measured timer count")
        require(finite_positive(timing[role].get("mean")), "invalid complete-step mean")
        wall = inputs.read(root / "case_host_wall.json")
        wall_start = inputs.read(root / "case_host_wall_start.json")
        walls[role] = wall
        seconds = checked_host_wall(wall, wall_start, leg, pair)
        require(seconds + 1 >= timing[role]["sum_all_optimizer_update_seconds"], "update sum exceeds acquisition interval")
        output_legs.append({"role": role, "index": index, "orchestration_dir": str(root),
            "run_dir": str(run), "communication_sms": budget, "proof_path": str(proof_path),
            "proof_sha256": inputs.hashes[str(proof_path)], "mean_step_ms": timing[role]["mean"],
            "sample_sd_ms": timing[role]["sample_sd"], "host_wall_seconds": seconds,
            "all_optimizer_update_seconds": timing[role]["sum_all_optimizer_update_seconds"]})
    for name in ("placement", "inputs", "prefix"):
        equal(identities["candidate"][name], identities["reference"][name], f"actual paired equality {name}")
    evidence.equal_resolved_configuration(identities["candidate"]["resolved"], identities["reference"]["resolved"],
                                          "only budget and explicit output destinations differ")
    equal(walls[roles[0]]["hostname"], walls[roles[1]]["hostname"], "one allocated batch-host wall clock")
    require(int(walls[roles[0]]["end_epoch_ns"]) <= int(walls[roles[1]]["start_epoch_ns"]),
            "independent case acquisitions overlap or order differs")
    job_root = ROOT / "runs" / f"{pair_dir.name}-posthoc-job-{os.environ['SLURM_JOB_ID']}"
    account_path = job_root / "training_accounting_utc.psv"
    inputs.bind(account_path)
    accounting = read_accounting(account_path, {pair["training_slurm_job_id"]})[pair["training_slurm_job_id"]]
    equal(accounting["allocated_nodes"], 4, "one shared four-node training allocation")
    equal(accounting["nodes"], sorted({row["hostname"] for row in identities["candidate"]["placement"]}),
          "Slurm node list agrees with actual paired placement")
    allocated_start = datetime.fromisoformat(accounting["start_utc"]).timestamp()
    allocated_end = datetime.fromisoformat(accounting["end_utc"]).timestamp()
    require(allocated_start <= int(walls[roles[0]]["start_epoch_ns"]) / 1e9
            and int(walls[roles[1]]["end_epoch_ns"]) / 1e9 <= allocated_end + 1,
            "case wall intervals fall outside their shared allocation")
    allocation_seconds = accounting["elapsed_seconds"]
    require(allocation_seconds > 0 and sum(leg["host_wall_seconds"] for leg in output_legs) <= allocation_seconds + 2,
            "case intervals exceed the shared allocation")
    ratio = timing["candidate"]["mean"] / timing["reference"]["mean"]
    inputs.recheck()
    result = {"schema_version": 1, "kind": "ampa_ep_overlap_budget_pair_comparison",
        "valid_paired_budget_comparison": True,
        "pair_id": pair_dir.name, "experiment_protocol": adapter.PROTOCOL,
        "candidate_budget": pair["candidate_budget"], "reference_budget": 20, "order": pair["order"],
        "execution_steps": pair["execution_steps"], "completed_updates_per_case": pair["execution_steps"],
        "qualification_only": pair["qualification_only"], "source_manifest_sha256": source["sha256"],
        "training_slurm_job_id": pair["training_slurm_job_id"], "matched_actual_rank_placement": True,
        "rank_placement_identity": identities["candidate"]["placement"],
        "consumed_input_fingerprint_sha256": identities["candidate"]["prefix"],
        "resolved_configuration_sha256": hashlib.sha256(json.dumps(identities["candidate"]["resolved"],
            sort_keys=True, allow_nan=True).encode()).hexdigest(), "legs": output_legs,
        "candidate_over_reference_mean_ratio": ratio,
        "candidate_minus_reference_mean_ms": timing["candidate"]["mean"] - timing["reference"]["mean"],
        "timing": timing, "training_allocation_elapsed_seconds": allocation_seconds,
        "training_accounting": accounting,
        "per_case_slurm_elapsed_available": False, "whole_posthoc_elapsed_seconds": None,
        "analyzer_sha256": inputs.hashes[str(Path(__file__))],
        "execution": {"slurm_job_id": os.environ["SLURM_JOB_ID"], "partition": "debug", "hostname": socket.gethostname()},
        "input_files": [{"path": path, "sha256": digest} for path, digest in sorted(inputs.hashes.items())],
        "claim_limit": "One paired allocation replicate with five sequential measured subsamples per full case. Same GPU placement does not remove cache/order/time effects. No free-SM or global-optimum claim."}
    output_dir.mkdir()
    write_new(output_dir / "pair_comparison.json", result)
    return result


def budget_statistics(records):
    grouped, pair_ids, allocations = {}, set(), set()
    for row in records:
        equal((row.get("execution_steps"), row.get("qualification_only")), (10, False),
              "only full ten-update pairs are allocation replicates")
        require(row["pair_id"] not in pair_ids and row["training_slurm_job_id"] not in allocations,
                "a pair or allocation cannot supply duplicate replicates")
        pair_ids.add(row["pair_id"])
        allocations.add(row["training_slurm_job_id"])
        budget, ratio, order = row["candidate_budget"], row["candidate_over_reference_mean_ratio"], row["order"]
        require(adapter.allowed_budget(budget) and budget != 20 and finite_positive(ratio) and order in ORDERS,
                "invalid full paired observation")
        grouped.setdefault(budget, []).append(row)
    result = {20: {"ratios": [1.0], "orders": [], "pair_count": 0, "median_ratio": 1.0,
                   "minimum_ratio": 1.0, "maximum_ratio": 1.0, "maximum_step_cv": 0.0}}
    for budget, rows in grouped.items():
        ratios = [row["candidate_over_reference_mean_ratio"] for row in rows]
        cvs = [leg["sample_sd_ms"] / leg["mean_step_ms"] for row in rows for leg in row.get("legs", [])
               if leg.get("sample_sd_ms") is not None]
        result[budget] = {"ratios": ratios, "orders": [row["order"] for row in rows], "pair_count": len(rows),
            "median_ratio": statistics.median(ratios), "minimum_ratio": min(ratios), "maximum_ratio": max(ratios),
            "maximum_step_cv": max(cvs, default=0.0)}
    return result


def plan_next_pair(records, supported_budgets):
    require(set((12, 20, 32, 40)) <= set(supported_budgets)
            and all(adapter.allowed_budget(value) for value in supported_budgets), "qualified initial budgets unavailable")
    stats = budget_statistics(records)
    def next_pair(budget, reason):
        count = stats.get(budget, {}).get("pair_count", 0)
        seen = stats.get(budget, {}).get("orders", [])
        order = ORDERS[0] if seen.count(ORDERS[0]) <= seen.count(ORDERS[1]) else ORDERS[1]
        return {"action": "run_pair", "next_pair": {"candidate_budget": budget, "order": order,
                "replicate": count + 1}, "reason": reason, "budget_statistics": stats}
    for budget, order in INITIAL_PLAN:
        if not any(row["candidate_budget"] == budget and row["order"] == order for row in records):
            request = next_pair(budget, "complete the six predeclared balanced initial pairs")
            request["next_pair"]["order"] = order
            return request
    for budget in sorted(stats):
        if budget == 20:
            continue
        row = stats[budget]
        if row["pair_count"] < 2 or not set(ORDERS) <= set(row["orders"]):
            return next_pair(budget, "complete two reversed-order allocation replicates")
        uncertain = (row["maximum_ratio"] - row["minimum_ratio"] > TOLERANCE * row["median_ratio"]
            or row["maximum_step_cv"] > TOLERANCE or row["minimum_ratio"] < 1 < row["maximum_ratio"]
            or abs(row["median_ratio"] - 1) <= TOLERANCE)
        if uncertain and row["pair_count"] < 3:
            return next_pair(budget, "add the predeclared third pair for noise/order/near-threshold ambiguity")
    best = min(stats, key=lambda budget: (stats[budget]["median_ratio"], abs(budget - 20)))
    boundaries = {}
    for side, outer in (("lower", min(stats)), ("upper", max(stats))):
        current, incumbent = stats[outer], stats[best]
        worse = current["minimum_ratio"] > (1 + TOLERANCE) * incumbent["maximum_ratio"]
        if worse:
            boundaries[side] = {"status": "repeatably_worse", "outer_budget": outer, "incumbent_budget": best}
            continue
        target = max([value for value in (4, 8) if value < outer and value in supported_budgets], default=None) if side == "lower" else \
                 min([value for value in (48, max(supported_budgets)) if value > outer and value in supported_budgets], default=None)
        if target is not None:
            return next_pair(target, f"extend the competitive {side} boundary beyond {outer}")
        boundaries[side] = {"status": "supported_limit_without_clear_degradation", "outer_budget": outer,
                            "incumbent_budget": best}
    return {"action": "complete", "next_pair": None, "reason": "both directions reach repeatable loss or a qualified limit",
            "budget_statistics": stats, "best_tested_budget": best, "boundary_status": boundaries,
            "global_optimum_claimed": False}


def campaign_decision(campaign_dir, include_pair, preflight_proof, decision_output):
    campaign_dir, include_pair = Path(campaign_dir), Path(include_pair)
    require(campaign_dir.is_absolute() and campaign_dir.parent == ROOT / "runs" and campaign_dir.is_dir()
            and not campaign_dir.is_symlink(), "invalid campaign directory")
    inputs = evidence.Inputs()
    analyzer = inputs.bind(Path(__file__))
    preflight = inputs.read(preflight_proof)
    require(preflight.get("valid_preflight") is True and preflight.get("training_executed") is False,
            "source-qualified actual-device preflight required")
    equal(preflight.get("experiment_protocol"), adapter.PROTOCOL, "preflight protocol")
    ledger = campaign_dir / "accepted.tsv"
    ledger_bytes = ledger.read_bytes()
    ledger_sha = hashlib.sha256(ledger_bytes).hexdigest()
    snapshot = Path(decision_output).parent / "campaign_accepted_snapshot.tsv"
    with snapshot.open("xb") as stream:
        stream.write(ledger_bytes)
    inputs.bind(snapshot, ledger_sha)
    paths, declared = [], {}
    for line in ledger_bytes.decode().splitlines():
        fields = line.split("\t")
        require(len(fields) == 7, "accepted paired ledger requires seven fields")
        budget, order, replicate, pair_id, training, posthoc, raw_path = fields
        require(budget.isdigit() and replicate.isdigit() and training.isdigit() and posthoc.isdigit(), "invalid ledger identity")
        require(pair_id not in declared, "duplicate accepted paired identity")
        declared[pair_id] = (int(budget), order, training, posthoc)
        paths.append(Path(raw_path))
    require(include_pair not in paths, "prospective pair is already in the accepted ledger")
    paths.append(include_pair)
    records, seen, training_jobs = [], set(), set()
    for path in paths:
        row = inputs.read(path)
        for name in ("valid_paired_budget_comparison", "matched_actual_rank_placement"):
            equal(row.get(name), True, "accepted paired raw evidence")
        equal(row.get("experiment_protocol"), adapter.PROTOCOL, "paired decision protocol")
        equal(row.get("source_manifest_sha256"), preflight["source_manifest_sha256"], "common qualified source")
        equal((row.get("execution_steps"), row.get("qualification_only")), (10, False), "qualifications cannot become performance replicates")
        require(row["pair_id"] not in seen and row["training_slurm_job_id"] not in training_jobs,
                "one independent training allocation per pair replicate")
        seen.add(row["pair_id"])
        training_jobs.add(row["training_slurm_job_id"])
        if row["pair_id"] in declared:
            equal((row["candidate_budget"], row["order"], row["training_slurm_job_id"], row["execution"]["slurm_job_id"]),
                  declared[row["pair_id"]], "accepted paired ledger binding")
        bound_inputs(row, inputs)
        records.append(row)
    result = plan_next_pair(records, preflight["allowed_budgets"])
    result.update(schema_version=1, valid_campaign_decision=True, experiment_protocol=adapter.PROTOCOL,
        source_manifest_sha256=preflight["source_manifest_sha256"], analyzer_sha256=analyzer,
        accepted_pair_ids=sorted(seen), accepted_pair_count=len(seen), tolerance_fraction=TOLERANCE,
        accepted_ledger_snapshot={"path": str(snapshot), "sha256": ledger_sha},
        mutable_ledger_source_path=str(ledger),
        rule="2% is an operational tolerance, not a significance test. Repeatable loss requires min(new ratios)>1.02*max(incumbent ratios).",
        supported_budgets=preflight["allowed_budgets"], independent_replicate_unit="one paired training allocation",
        prospective_pair_path=str(include_pair),
        claim_limit="The included current pair becomes accepted only after successful terminal posthoc accounting. A local tested bracket is not a global optimum.")
    require(hashlib.sha256(ledger.read_bytes()).hexdigest() == ledger_sha,
            "accepted ledger changed during the prospective decision")
    inputs.recheck()
    result["input_files"] = [{"path": path, "sha256": digest} for path, digest in sorted(inputs.hashes.items())]
    write_new(decision_output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--include-pair", type=Path, required=True)
    parser.add_argument("--preflight-proof", type=Path, required=True)
    parser.add_argument("--decision-output", type=Path, required=True)
    args = parser.parse_args()
    require(os.environ.get("SLURM_JOB_PARTITION") == "debug" and os.environ.get("SLURM_JOB_ID", "").isdigit()
            and Path(sys.prefix) == Path("/opt/venvs/megatron-bridge"), "allocated debug configured Python required")
    result = campaign_decision(args.campaign_dir, args.include_pair, args.preflight_proof, args.decision_output)
    print(json.dumps({"valid_campaign_decision": True, "action": result["action"], "next_pair": result["next_pair"]}))


if __name__ == "__main__":
    main()
