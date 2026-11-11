#!/usr/bin/env python3
"""Reconcile a completed unprofiled adaptive sweep on allocated compute."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import statistics
import sys

import yaml

from analysis import accept_uccl_pilot as evidence
from analysis import ep_overlap_budget_compare as compare
from analysis.extract_step_timing import select_timer_log, timing_rows
from analysis.summarize_training_evidence import summarize
from profiling import run_pretrain_ep_overlap_budget as adapter
from tools import ep_overlap_budget_validate as validator
from tools.ep16_posthoc_proof import import_closure
from tools.ep_overlap_campaign_summary import Inputs, descriptive, read_accounting, write_csv, write_new


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
TOKENS_PER_STEP = 262144
NEW_SOURCES = ("tools/ep_overlap_budget_summary.py", "tests/test_ep_overlap_budget_summary.py",
               "jobs/analysis/ep_overlap_budget_summary.sbatch")
REPORT_NAME = "QWEN_UCCL_UNPROFILED_BUDGET_AUDIT.md"
REVIEW_TARGET = "docs/" + REPORT_NAME
PROOF_KIND = "ampa_ep_overlap_unprofiled_budget_execution_accounting_validation"
PAIR_KIND = "ampa_ep_overlap_budget_pair_comparison"
ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
SHA = re.compile(r"[0-9a-f]{64}")
require, equal = evidence.require, evidence.equal


def record_matches(record, fields, label):
    for name, value in fields.items():
        equal(record.get(name), value, f"{label}: {name}")


def read_ledger(payload, *, runs=RUNS):
    require(payload.endswith(b"\n"), "accepted ledger must contain complete lines")
    result, pairs, jobs, slots = [], set(), set(), set()
    for line in payload.decode().splitlines():
        fields = line.split("\t")
        require(len(fields) == 7, "accepted ledger must have seven fields per row")
        candidate, order, replicate, pair, train, post, proof = fields
        require(all(value.isascii() and value.isdecimal() for value in (candidate, replicate, train, post)),
                "invalid accepted ledger numeric identity")
        candidate, replicate = int(candidate), int(replicate)
        require(adapter.allowed_budget(candidate) and candidate != 20 and order in compare.ORDERS
                and 1 <= replicate <= 3 and ID.fullmatch(pair), "invalid full paired case")
        require(pair not in pairs and train != post and train not in jobs and post not in jobs,
                "duplicate paired case or allocation")
        require((candidate, replicate) not in slots, "duplicate candidate replicate")
        pairs.add(pair)
        jobs.update((train, post))
        slots.add((candidate, replicate))
        equal(proof, str(runs / f"{pair}-posthoc/pair_comparison.json"), "canonical accepted pair proof")
        result.append({"candidate_budget": candidate, "order": order, "replicate": replicate, "pair_id": pair,
                       "training_job": train, "posthoc_job": post, "proof_path": proof, "ledger_line": line})
    require(len(result) >= len(compare.INITIAL_PLAN), "six initial full allocation replicates required")
    return result


def read_keys(path):
    rows = [line.split("=", 1) for line in path.read_text().splitlines()]
    require(all(len(row) == 2 and row[0] for row in rows), "invalid completion marker")
    fields = dict(rows)
    require(len(fields) == len(rows), "duplicate completion marker key")
    return fields


def check_submission_ledgers(campaign, rows):
    acquired, posthoc = {}, {}
    for line in (campaign / "acquisitions.tsv").read_text().splitlines():
        budget, order, replicate, attempt, pair, train = line.split("\t")
        require(pair not in acquired and attempt in ("1", "2", "3"), "invalid acquisition attempt identity")
        acquired[pair] = (int(budget), order, int(replicate), train)
    for line in (campaign / "posthoc-jobs.tsv").read_text().splitlines():
        pair, train, post = line.split("\t")
        require(pair not in posthoc, "duplicate posthoc submission")
        posthoc[pair] = (train, post)
    for row in rows:
        pair = row["pair_id"]
        equal(acquired.get(pair), tuple(row[name] for name in ("candidate_budget", "order", "replicate", "training_job")),
              "accepted row names its submitted acquisition")
        equal(posthoc.get(pair), (row["training_job"], row["posthoc_job"]), "accepted row names its separate submitted analysis")


def bind_manifest(path, inputs):
    inputs.bind(path)
    result = {}
    for line in path.read_text().splitlines():
        match = re.fullmatch(r"([0-9a-f]{64}) [ *](.+)", line)
        require(match is not None, "invalid accepted evidence hash manifest")
        digest, name = match.groups()
        source = Path(name)
        require(source.is_absolute() and source.is_relative_to(ROOT) and ".." not in source.parts,
                "hash manifest path must be inside project")
        if name in result:
            equal(result[name], digest, "repeated hash manifest entry")
        inputs.bind(source, digest)
        result[name] = digest
    require(result, "empty evidence hash manifest")
    return result


def check_timer_windows(rows):
    equal([(row["first_step"], row["last_step_exclusive"], row["optimizer_steps"]) for row in rows],
          [(step, step + 1, 1) for step in range(10)], "ten complete ordered optimizer updates")
    values = [row["mean_optimizer_step_ms"] for row in rows]
    all_steps, measured = descriptive(values), descriptive(values[5:])
    return {"all_ten": all_steps, "measured_five": measured, "sum_optimizer_seconds": sum(values) / 1000}


def check_decision(record, planned, current, previous, source_sha, analyzer_sha, supported, inputs):
    # JSON normalizes integer budget keys to strings in the producer artifact.
    record_matches(record, json.loads(json.dumps(planned)), "replayed planner output")
    record_matches(record, {"valid_campaign_decision": True, "experiment_protocol": adapter.PROTOCOL,
        "source_manifest_sha256": source_sha, "analyzer_sha256": analyzer_sha, "supported_budgets": supported,
        "accepted_pair_ids": sorted([row["pair_id"] for row in previous] + [current["pair_id"]]),
        "accepted_pair_count": len(previous) + 1, "tolerance_fraction": compare.TOLERANCE,
        "prospective_pair_path": current["proof_path"]}, "source-bound decision identity")
    snapshot = Path(current["proof_path"]).parent / "campaign_accepted_snapshot.tsv"
    prefix = "".join(row["ledger_line"] + "\n" for row in previous).encode()
    equal(record.get("accepted_ledger_snapshot"), {"path": str(snapshot), "sha256": hashlib.sha256(prefix).hexdigest()},
          "immutable exact pre-append accepted ledger")
    compare.bound_inputs(record, inputs)
    require(snapshot.read_bytes() == prefix, "decision snapshot bytes differ from actual earlier accepted ledger")


def reconcile_received_routes(run, stored_path, contract, environment, inputs):
    ownership = inputs.read(run / "config/routing_ownership.json")
    audits = [inputs.read(run / "raw" / f"routing_audit_rank-{rank}.json") for rank in range(16)]
    received = validator.checked_receives(contract, ownership, audits, environment, 10)
    equal(inputs.read(stored_path), received, "fresh integer send/receive reconciliation")
    return received


def check_leg(row, pair, leg, result_leg, inputs, source_sha, source_files, reference_prefix):
    role, index = leg["role"], leg["index"]
    budget = row["candidate_budget"] if role == "candidate" else 20
    run_id = f"{row['pair_id']}-{role}-b{budget}"
    root, run = RUNS / run_id, RUNS / f"{run_id}-training"
    proof_path = RUNS / f"{run_id}-posthoc/execution_accounting.json"
    record_matches(leg, {"budget": budget, "experiment_id": run_id, "orchestration_dir": str(root), "run_dir": str(run)}, "leg identity")
    record_matches(result_leg, {"role": role, "index": index, "orchestration_dir": str(root), "run_dir": str(run),
        "communication_sms": budget, "proof_path": str(proof_path)}, "accepted pair leg")
    inputs.bind(proof_path, result_leg["proof_sha256"])
    proof = inputs.read(proof_path)
    record_matches(proof, {"kind": PROOF_KIND, "valid_execution_and_accounting": True,
        "valid_training_health": True, "valid_send_receive_audit": True, "valid_all_rank_startup_and_completion": True,
        "valid_actual_budget_runtime_guards": True, "qualification_only": False, "capture": "off",
        "case": "uccl_ordinary", "completed_updates": 10, "communication_backend": "uccl_ep", "communication_sms": budget,
        "training_slurm_job_id": row["training_job"], "source_manifest_sha256": source_sha,
        "orchestration_dir": str(root), "run_dir": str(run), "per_case_slurm_elapsed_available": False,
        "validator_sha256": source_files["tools/ep_overlap_budget_validate.py"],
        "gpu_event_correlation_acceptance": "not_requested_profiler_off",
        "hardware_metrics_acceptance": "not_requested_profiler_off"}, "strict accepted leg proof")
    record_matches(proof["execution"], {"slurm_job_id": row["posthoc_job"], "partition": "debug"}, "posthoc identity")
    require(proof["backend"].get("valid_backend_identity_and_cleanup") is True, "native backend/cleanup proof required")
    compare.bound_inputs(proof, inputs)
    execution = validator.checked_execution_contract(inputs.read(run / "config/ep_overlap_contract.json"),
        case="uccl_ordinary", capture="off", steps=10, source_sha256=source_sha)
    equal(execution, proof["execution_contract"], "actual runtime budget contract")
    contract = validator.checked_training_contract(inputs.read(run / "config/comparison_contract.json"), execution, "uccl_ordinary")
    resolved = run / "config/resolved.yaml"
    inputs.bind(resolved)
    cfg = validator.checked_resolved(yaml.safe_load(resolved.read_text()), execution, "uccl_ordinary", run)
    normalized = compare.normalized_config(cfg, run, budget)
    env = inputs.read(run / "environment.json")
    equal(env["slurm_job_id"], row["training_job"], "actual training job")
    placement = validator.validate_rank_placement(env["rank_placement"], expected_world_size=16, launch_policy="torchrun_node4")
    validator.validate_runtime_groups(placement, expected_world_size=16, cp=1)
    equal(placement, proof["rank_placement"], "actual validated rank placement")
    actual_guards, fingerprints = [], {}
    for slot in placement:
        rank = slot["rank"]
        guard = inputs.read(run / "raw" / f"ep_overlap_budget_rank-{rank}_runtime.json")
        actual_guards.append(validator.checked_budget_runtime_record(guard, execution, slot, row["training_job"]))
        path = run / "raw" / f"sample_fingerprint_rank-{rank}.json"
        fingerprints[str(rank)] = validator._checked_fingerprint_batches(inputs.read(path), rank=rank,
            accumulation=4, steps=10, token_shape=[1, 4096], path=path)
    equal(actual_guards, proof["budget_runtime_guards"], "actual requested Buffer budget on every rank")
    prefix = hashlib.sha256(json.dumps(fingerprints, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    equal(prefix, proof["consumed_input_fingerprint_sha256"], "all 640 consumed CPU batch fingerprints")
    equal(prefix, reference_prefix, "same original 640-sample prefix")
    health = summarize(run, steps=10, interval=1)
    equal(inputs.read(proof_path.parent / "training_health.json"), health, "fresh raw health summary")
    equal(health["final_consumed_samples"], 640, "actual full-run samples")
    equal(health["final_consumed_global_tokens"], 10 * TOKENS_PER_STEP, "actual full-run tokens")
    timer = select_timer_log(run / "logs")
    inputs.bind(timer)
    timers = timing_rows(timer, warmup_steps=0, measured_steps=10, log_interval=1, tokens_per_step=TOKENS_PER_STEP)
    stats = check_timer_windows(timers)
    stored = inputs.read(proof_path.parent / "timing.json")
    equal(stored["all_updates"], timers, "raw ten timers")
    equal(stored["by_update"], timers[5:], "actual five measured timers")
    record_matches(stored, {"mean": stats["measured_five"]["mean_ms"], "median": stats["measured_five"]["median_ms"],
        "sample_sd": stats["measured_five"]["sample_sd_ms"], "sum_all_optimizer_update_seconds": stats["sum_optimizer_seconds"],
        "count": 5, "qualification_only": False, "capture": "off"}, "recomputed full and measured timing")
    record_matches(result_leg, {"mean_step_ms": stored["mean"], "sample_sd_ms": stored["sample_sd"],
        "all_optimizer_update_seconds": stats["sum_optimizer_seconds"]}, "recomputed pair-leg timing")
    wall = inputs.read(root / "case_host_wall.json")
    seconds = compare.checked_host_wall(wall, inputs.read(root / "case_host_wall_start.json"), leg, pair)
    equal(seconds, result_leg["host_wall_seconds"], "actual per-case wall time")
    require(seconds + 1 >= stats["sum_optimizer_seconds"], "optimizer timers exceed case wall interval")
    received = reconcile_received_routes(run, proof_path.parent / "received_routes.json", contract, env, inputs)
    archive = Path("evidence/cases") / run_id
    for path in (proof_path, proof_path.parent / "timing.json", proof_path.parent / "training_health.json",
                 proof_path.parent / "received_routes.json", root / "case_host_wall.json", resolved, timer):
        inputs.archive(path, archive / path.name)
    common = {"immutable_inputs": proof["immutable_inputs"], "consumed_input_prefix_sha256": prefix,
        "model": contract["model"], "parallelism": contract["parallelism"], "dataset": contract["dataset"],
        "training": contract["training"], "routing": contract["routing"], "overlap_flags": execution["overlap_flags"],
        "actual_ddp_delay_wgrad_compute": execution["ddp_delay_wgrad_compute"], "cuda_connections": execution["CUDA_DEVICE_MAX_CONNECTIONS"],
        "native_backend": proof["backend"], "runtime_topology": env["runtime_topology"],
        "frameworks": {name: env[name] for name in ("megatron_lm", "megatron_bridge", "pytorch_version", "cuda_version", "nccl_version", "container_sqsh_sha256")}}
    return {"case_id": run_id, "role": role, "index": index, "budget": budget, "proof_path": str(proof_path),
        "raw_timer_log": str(timer), "optimizer_updates": timers, "statistics": stats, "health": health,
        "host_wall": wall, "host_wall_seconds": seconds, "actual_rank_placement": compare.placement_identity(placement),
        "actual_budget_guards": actual_guards, "actual_cpu_batches": 640,
        "logical_expert_assignments": received["total_source_expert_assignments"],
        "per_case_slurm_elapsed_available": False}, common, normalized, stored


def check_pair(row, account, inputs, source, reference_prefix, accepted_hashes):
    pair_id, source_sha = row["pair_id"], source["sha256"]
    pair_dir = RUNS / pair_id
    proof_path = Path(row["proof_path"])
    require(str(proof_path) in accepted_hashes, "accepted ledger lacks pinned pair proof")
    inputs.bind(proof_path, accepted_hashes[str(proof_path)])
    proof = inputs.read(proof_path)
    record_matches(proof, {"kind": PAIR_KIND, "valid_paired_budget_comparison": True,
        "matched_actual_rank_placement": True, "pair_id": pair_id, "experiment_protocol": adapter.PROTOCOL,
        "candidate_budget": row["candidate_budget"], "reference_budget": 20, "order": row["order"],
        "execution_steps": 10, "completed_updates_per_case": 10, "qualification_only": False,
        "training_slurm_job_id": row["training_job"], "source_manifest_sha256": source_sha,
        "consumed_input_fingerprint_sha256": reference_prefix, "per_case_slurm_elapsed_available": False,
        "analyzer_sha256": source["files"]["analysis/ep_overlap_budget_compare.py"]}, "accepted pair proof")
    record_matches(proof["execution"], {"slurm_job_id": row["posthoc_job"], "partition": "debug"}, "separate pair posthoc identity")
    compare.bound_inputs(proof, inputs)
    pair = inputs.read(pair_dir / "pair.json")
    record_matches(pair, {"pair_id": pair_id, "candidate_budget": row["candidate_budget"], "reference_budget": 20,
        "execution_steps": 10, "qualification_only": False, "capture": "off", "case": "uccl_ordinary", "order": row["order"],
        "training_slurm_job_id": row["training_job"], "source_manifest_sha256": source_sha,
        "requested_nodes": 4, "requested_time_limit": "00:21:00", "partition": "debug"}, "actual shared pair")
    completion = inputs.read(pair_dir / "pair_completed.json")
    record_matches(completion, pair, "pair completion metadata")
    record_matches(completion, {"success": True, "completed_legs": 2, "both_workers_and_health_srun_returned_zero": True}, "both case receipts")
    compare.checked_completed_case_ids(completion["completed_case_ids"], [leg["experiment_id"] for leg in pair["legs"]])
    receipt_path = RUNS / f"{pair_id}-posthoc-job-{row['posthoc_job']}/completed.json"
    require(str(receipt_path) in accepted_hashes, "accepted pair lacks pinned separate receipt")
    receipt = inputs.read(receipt_path)
    record_matches(receipt, {"success": True, "pair_id": pair_id, "experiment_protocol": adapter.PROTOCOL,
        "training_slurm_job_id": row["training_job"], "posthoc_slurm_job_id": row["posthoc_job"],
        "partition": "debug", "requested_nodes": 1, "source_manifest_sha256": source_sha}, "actual posthoc receipt")
    train, post = account[row["training_job"]], account[row["posthoc_job"]]
    equal(train, proof["training_accounting"], "fresh versus accepted shared allocation accounting")
    equal(train["elapsed_seconds"], proof["training_allocation_elapsed_seconds"], "shared training elapsed")
    equal((train["allocated_nodes"], post["allocated_nodes"]), (4, 1), "separate training and analysis node counts")
    require(train["end_utc"] <= post["start_utc"] and train["elapsed_seconds"] <= 1260
            and post["elapsed_seconds"] <= 5400, "separate allocations or wall limit differs")
    equal(post["nodes"], [proof["execution"]["hostname"]], "actual separate posthoc node")
    roles = ("candidate", "reference") if row["order"] == "candidate-first" else ("reference", "candidate")
    equal([leg["role"] for leg in pair["legs"]], list(roles), "actual declared execution order")
    equal(len(proof["legs"]), 2, "both accepted leg proofs")
    results, common, normalized, times = [], [], [], {}
    for index, leg in enumerate(pair["legs"]):
        equal(leg["index"], index, "actual leg execution index")
        result, identity, config, stored = check_leg(row, pair, leg, proof["legs"][index], inputs,
            source_sha, source["files"], reference_prefix)
        require(result["proof_path"] in accepted_hashes, "accepted pair lacks pinned leg proof")
        equal(inputs.hashes[result["proof_path"]], accepted_hashes[result["proof_path"]], "accepted leg proof hash")
        results.append(result)
        common.append(identity)
        normalized.append(config)
        times[result["role"]] = stored
    equal(results[0]["actual_rank_placement"], results[1]["actual_rank_placement"], "same actual rank/GPU placement within pair")
    equal(results[0]["actual_rank_placement"], proof["rank_placement_identity"], "paired actual GPU identities")
    equal(train["nodes"], sorted({slot["hostname"] for slot in results[0]["actual_rank_placement"]}), "allocation matches actual nodes")
    equal(results[0]["host_wall"]["hostname"], results[1]["host_wall"]["hostname"], "one shared allocation wall clock")
    require(int(results[0]["host_wall"]["end_epoch_ns"]) <= int(results[1]["host_wall"]["start_epoch_ns"]), "actual legs overlap or change order")
    require(datetime.fromisoformat(train["start_utc"]).timestamp() <= int(results[0]["host_wall"]["start_epoch_ns"]) / 1e9
            and int(results[1]["host_wall"]["end_epoch_ns"]) / 1e9 <= datetime.fromisoformat(train["end_utc"]).timestamp() + 1,
            "actual case wall intervals fall outside their shared allocation")
    require(sum(leg["host_wall_seconds"] for leg in results) <= train["elapsed_seconds"] + 2,
            "actual cases exceed shared allocation")
    equal(common[0], common[1], "same source, workload and input contract within pair")
    evidence.equal_resolved_configuration(normalized[0], normalized[1], "only budget and destinations differ within pair")
    equal(hashlib.sha256(json.dumps(normalized[0], sort_keys=True, allow_nan=True).encode()).hexdigest(),
          proof["resolved_configuration_sha256"], "paired normalized configuration digest")
    equal(times, proof["timing"], "pair proof preserves both leg timer records")
    ratio = times["candidate"]["mean"] / times["reference"]["mean"]
    equal(ratio, proof["candidate_over_reference_mean_ratio"], "actual measured optimizer-step mean ratio")
    for path in (proof_path, receipt_path, pair_dir / "pair.json", pair_dir / "pair_completed.json"):
        inputs.archive(path, Path("evidence/pairs") / pair_id / path.name)
    return {**{key: value for key, value in row.items() if key != "ledger_line"}, "legs": results,
        "candidate_over_reference_mean_ratio": ratio, "matched_actual_rank_placement": True,
        "training_accounting": train, "posthoc_accounting": post, "per_case_slurm_elapsed_available": False}, proof, common[0], normalized[0]


def export_tables(out, pairs, budget_statistics):
    updates, legs, pair_rows = [], [], []
    for pair in pairs:
        train, post = pair["training_accounting"], pair["posthoc_accounting"]
        base = {name: pair[name] for name in ("pair_id", "candidate_budget", "order", "replicate", "training_job", "posthoc_job")}
        means = {leg["role"]: leg["statistics"]["measured_five"]["mean_ms"] for leg in pair["legs"]}
        pair_rows.append({**base, "candidate_mean_step_ms": means["candidate"], "reference_mean_step_ms": means["reference"],
            "candidate_over_reference_mean_ratio": pair["candidate_over_reference_mean_ratio"],
            "shared_training_elapsed_seconds": train["elapsed_seconds"], "separate_posthoc_elapsed_seconds": post["elapsed_seconds"],
            "training_start_utc": train["start_utc"], "training_end_utc": train["end_utc"], "training_nodes": train["node_list"],
            "posthoc_start_utc": post["start_utc"], "posthoc_end_utc": post["end_utc"], "posthoc_node": post["node_list"]})
        for leg in pair["legs"]:
            identity = {**base, **{name: leg[name] for name in ("case_id", "role", "index", "budget")}}
            stats = leg["statistics"]["measured_five"]
            legs.append({**identity, "case_host_wall_seconds": leg["host_wall_seconds"],
                "case_wall_start_utc": leg["host_wall"]["start_utc"], "case_wall_end_utc": leg["host_wall"]["end_utc"],
                "all_ten_optimizer_timer_seconds": leg["statistics"]["sum_optimizer_seconds"],
                "measured_mean_ms": stats["mean_ms"], "measured_median_ms": stats["median_ms"],
                "measured_sample_sd_ms": stats["sample_sd_ms"], "measured_min_ms": stats["min_ms"],
                "measured_max_ms": stats["max_ms"], "measured_tokens_per_second": stats["aggregate_tokens_per_second"],
                "measured_step_cv": stats["sample_sd_ms"] / stats["mean_ms"], "actual_cpu_batches": leg["actual_cpu_batches"],
                "logical_expert_assignments": leg["logical_expert_assignments"], "per_case_slurm_elapsed_available": False})
            for timer, health in zip(leg["optimizer_updates"], leg["health"]["logged_boundaries"], strict=True):
                update = timer["first_step"] + 1
                equal(health["iteration"], update, "actual timer/health update identity")
                updates.append({**identity, "human_update": update, "measured": update >= 6,
                    "optimizer_step_ms": timer["mean_optimizer_step_ms"], "tokens_per_second": timer["tokens_per_second"],
                    "raw_timer_log": leg["raw_timer_log"], **health})
    budgets = [{"budget": int(budget), "candidate_allocation_replicates": stats["pair_count"],
                "reference_legs_at_budget20": len(pairs) if int(budget) == 20 else 0,
                "median_ratio": stats["median_ratio"], "minimum_ratio": stats["minimum_ratio"],
                "maximum_ratio": stats["maximum_ratio"], "maximum_measured_step_cv": None if int(budget) == 20 else stats["maximum_step_cv"],
                "ratio_one_is_reference_definition": int(budget) == 20}
               for budget, stats in sorted(budget_statistics.items(), key=lambda item: int(item[0]))]
    equal(len(updates), 20 * len(pairs), "all original optimizer timers retained")
    equal(len(legs), 2 * len(pairs), "both legs per accepted allocation")
    for name, rows in (("optimizer_steps.csv", updates), ("legs.csv", legs), ("pairs.csv", pair_rows), ("budgets.csv", budgets)):
        write_csv(out / name, rows)
    return pair_rows, legs, budgets


def plot_ratios(out, pairs, stats):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(9.4, 5.6))
    tolerance = compare.TOLERANCE
    axis.axhspan(1 - tolerance, 1 + tolerance, color="#f2f2f2", label="±2% operational band")
    axis.axhline(1, color="#525252", linestyle="--", linewidth=1, label="Paired 20-block reference")
    for order, marker, color, shift in (("candidate-first", "^", "#2365a2", -.22),
                                        ("reference-first", "o", "#ac4d22", .22)):
        selected = [row for row in pairs if row["order"] == order]
        axis.scatter([row["candidate_budget"] + shift for row in selected],
                     [row["candidate_over_reference_mean_ratio"] for row in selected],
                     marker=marker, color=color, label=order.replace("-", " "), alpha=.85, zorder=3)
    chosen = [(int(budget), row) for budget, row in stats.items() if int(budget) != 20]
    axis.errorbar([budget for budget, _ in chosen], [row["median_ratio"] for _, row in chosen],
        yerr=[[row["median_ratio"] - row["minimum_ratio"] for _, row in chosen],
              [row["maximum_ratio"] - row["median_ratio"] for _, row in chosen]],
        fmt="D", markersize=4, capsize=3, color="#252525", linestyle="none", label="Median and observed range", zorder=4)
    axis.set(xticks=sorted({20, *(budget for budget, _ in chosen)}), xlabel="Configured UCCL payload blocks",
             ylabel="Candidate / paired reference mean optimizer-step time",
             title="Unprofiled optimizer-step timing by payload budget")
    axis.grid(axis="y", alpha=.2)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(loc="best", frameon=False, fontsize=8)
    fig.text(.5, .025, "Each point is one same-placement allocation pair; updates 6–10 per leg. Lower is faster.\n"
             "Whiskers show observed ranges, not confidence intervals. Common audit observers remain enabled.",
             ha="center", va="bottom", fontsize=8)
    fig.tight_layout(rect=(0, .105, 1, 1))
    for suffix in ("png", "svg"):
        target = out / f"paired_budget_comparison.{suffix}"
        require(not target.exists() and not target.is_symlink(), "plot destination already exists")
        fig.savefig(target, dpi=180)
    plt.close(fig)


def render_report(result, out):
    decision = result["final_decision"]
    lines = ["# Qwen UCCL unprofiled payload-budget audit", "",
        f"The sweep reconciles {result['accepted_pair_count']} full paired allocations and {result['full_case_count']} independent C0 starts. "
        f"The planner selected budget {decision['best_tested_budget']} by the lowest observed median ratio. "
        "The recorded ranges and stopping rules define the supported conclusion. They do not establish a global optimum.", "",
        "Every case completed five warmup and five measured optimizer updates. Each candidate shares the same 16 physical GPUs "
        "with its own 20-block reference. Opposite execution orders and repeated allocations reduce placement and order effects. "
        "Each allocation is one replicate. The five measured updates within a leg are subsamples.", "",
        "The workload uses Qwen3-30B-A3B-Base, ordinary top-eight routing, BF16 and EP16/CP1/TP1/PP1/ETP1. "
        "Each update consumes 64 samples of length 4096. Each run starts from the same C0 and consumes the same 640-sample prefix. "
        "The main expert-parallel overlap, model and DDP delayed weight gradients, and early attention memory release stay enabled. "
        "Immutable payload verification is reused. The actual consumed prefix is rechecked.", "",
        "PyTorch and Nsight profilers are off. Common routing, receive, fingerprint and health observers remain enabled. "
        "Their costs are in the recorded timings. No SM activity, occupancy or free-SM measurement is available from this sweep.", "",
        f"The full-precision outputs live in `{out.relative_to(ROOT)}`. `optimizer_steps.csv` preserves every step and its health record. "
        "`legs.csv` contains per-case timing and wall intervals. `pairs.csv` contains the allocation times and paired ratios. "
        "`metrics.json` binds actual placement, source and input identities, accepted evidence and the planner decisions.", "",
        "## Budget comparisons", "",
        "A ratio below one means the candidate took less time than its same-allocation reference. "
        "The reference row defines ratio one; its individual 20-block timings remain in the leg and pair tables.", "",
        "| Blocks | Candidate allocation replicates | Reference legs | Median ratio | Observed ratio range | Maximum within-leg step CV |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in result["budget_table"]:
        cv = "n/a" if row["budget"] == 20 else f"{100 * row['maximum_measured_step_cv']:.2f}%"
        lines.append(f"| {row['budget']} | {row['candidate_allocation_replicates']} | {row['reference_legs_at_budget20']} | "
                     f"{row['median_ratio']:.5f} | {row['minimum_ratio']:.5f}–{row['maximum_ratio']:.5f} | {cv} |")
    lines += ["", "## Allocation and case timings", "",
        "The training elapsed time covers the single allocation shared by both cases. The case wall interval starts before its training "
        "launch and ends after its health/timing summary. It excludes shared setup and source archival. The optimizer timers cover the "
        "complete updates. The posthoc time is a separate allocation. There is no per-case Slurm elapsed time.", "",
        "| Pair | Budget | Order | Train job | Posthoc job | Candidate mean (s) | Reference mean (s) | Ratio | Shared allocation (s) | Posthoc (s) |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in result["pair_table"]:
        lines.append(f"| `{row['pair_id']}` | {row['candidate_budget']} | {row['order']} | {row['training_job']} | {row['posthoc_job']} | "
                     f"{row['candidate_mean_step_ms']/1000:.4f} | {row['reference_mean_step_ms']/1000:.4f} | "
                     f"{row['candidate_over_reference_mean_ratio']:.5f} | {row['shared_training_elapsed_seconds']} | {row['separate_posthoc_elapsed_seconds']} |")
    lines += ["", "| Case | Blocks | Case wall (s) | Ten timers sum (s) | Measured mean (s) | Measured median (s) | Measured SD (s) |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in result["leg_table"]:
        lines.append(f"| `{row['case_id']}` | {row['budget']} | {row['case_host_wall_seconds']:.3f} | {row['all_ten_optimizer_timer_seconds']:.4f} | "
                     f"{row['measured_mean_ms']/1000:.4f} | {row['measured_median_ms']/1000:.4f} | {row['measured_sample_sd_ms']/1000:.4f} |")
    lines += ["", "## Boundary decisions and limits", "",
        "The predeclared 2% threshold is an operational rule. It does not give a significance test or a confidence interval. "
        "The planner uses the observed replicate ranges, order coverage and within-case variation. It adds a third pair when the rule requires it.", ""]
    for side, boundary in decision["boundary_status"].items():
        outer, incumbent = boundary["outer_budget"], boundary["incumbent_budget"]
        stats = decision["budget_statistics"]
        minimum, threshold = stats[str(outer)]["minimum_ratio"], (1 + compare.TOLERANCE) * stats[str(incumbent)]["maximum_ratio"]
        if boundary["status"] == "repeatably_worse":
            reason = f"The minimum ratio {minimum:.5f} exceeds the incumbent's upper ratio times 1.02 ({threshold:.5f})."
        else:
            reason = f"The tested boundary reaches the qualified limit. The minimum ratio {minimum:.5f} does not exceed {threshold:.5f}."
        lines.append(f"- {side.capitalize()} boundary: budget {outer}, compared with {incumbent}. {reason}")
    limits = result["preflight_device_limits"]
    excluded = result["next_unsupported_upper_budget"]
    lines += ["", f"The qualified range is {min(result['qualified_budgets'])}–{max(result['qualified_budgets'])} even blocks. "
        f"The allocated device reports {limits['shared_memory_per_block_optin']:,} bytes of opt-in shared memory per block. "
        "These resource checks do not measure the number of free SMs or prove successful execution at every untested point.", ""]
    if decision["boundary_status"]["lower"]["outer_budget"] == 4:
        lines += ["The native cached notify path requires more than three blocks. The budget must be even, so four is the lower limit. "
                  "The bound follows from `num_channels * 2 > 3` in `uccl/ep/src/internode.cu` and `num_channels = budget / 2`.", ""]
    if not excluded["fits_actual_optin_shared_limit"]:
        lines += [f"Budget {excluded['budget']} would request {excluded['notify_dynamic_shared_bytes']:,} dynamic shared bytes per notify block. "
                  f"This exceeds the measured {limits['shared_memory_per_block_optin']:,}-byte device limit.", ""]
    lines += [
        f"The sweep excludes its two-update qualification and {result['failed_attempt_count']} failed full-pair attempts. "
        "Failures and incomplete acquisitions do not count as measured degradation. The input, health, source, routing and "
        "native budget checks support the stated short-run timings. They do not prove long-run quality or remove network, cache and clock variation.", "",
        f"Training source SHA-256: `{result['training_source_manifest']['sha256']}`. "
        f"Consumed input prefix SHA-256: `{result['common_workload']['consumed_input_prefix_sha256']}`.", "",
        "The generated report requires independent interpretation review before promotion to the project documentation.", ""]
    return "\n".join(lines)


def summarize_campaign(args):
    out, campaign = args.output_dir, args.campaign_dir
    require(out.is_absolute() and out.parent == RUNS and not out.exists() and not out.is_symlink(), "new summary run directory required")
    require(campaign.is_absolute() and campaign.parent == RUNS and campaign.is_dir() and not campaign.is_symlink(),
            "completed direct campaign run directory required")
    out.mkdir()
    inputs = Inputs(out)
    summary_source = adapter.verify_source_manifest(args.summary_source_manifest, args.summary_source_sha256)
    closure = set(import_closure(ROOT, NEW_SOURCES[:2])) | set(NEW_SOURCES)
    require(closure <= set(summary_source["files"]), "summary source inventory omits dependencies: " + ", ".join(sorted(closure - set(summary_source["files"]))))
    inputs.archive(args.summary_source_manifest, "source/summary-source.sha256")
    for name in sorted(closure):
        inputs.bind(ROOT / name, summary_source["files"][name])
        inputs.archive(ROOT / name, Path("source") / name)
    for name in ("accepted.tsv", "exit.txt", "ACQUISITIONS_COMPLETE.txt", "preflight.json", "campaign-source.sha256",
                 "controller.sh", "controller.sha256", "qualification-inputs.sha256", "qualification_pair_comparison.json",
                 "accepted-proofs.sha256", "acquisitions.tsv", "posthoc-jobs.tsv"):
        inputs.archive(campaign / name, Path("evidence/coordinator") / name)
    failed = campaign / "failed-attempts.tsv"
    failures = []
    if failed.exists():
        inputs.archive(failed, "evidence/coordinator/failed-attempts.tsv")
        failures = failed.read_text().splitlines()
    require(not (campaign / "RETRIES_EXHAUSTED.txt").exists(), "exhausted campaign cannot provide complete results")
    rows = read_ledger((campaign / "accepted.tsv").read_bytes())
    check_submission_ledgers(campaign, rows)
    record_matches(read_keys(campaign / "exit.txt"), {"exit_code": "0", "complete": "1", "phase": "bounded_acquisitions_complete",
        "active_job": ""}, "terminal completed coordinator")
    record_matches(read_keys(campaign / "ACQUISITIONS_COMPLETE.txt"), {"bounded_paired_sweep_complete": "true",
        "accepted_pairs": str(len(rows)), "full_cases": str(2 * len(rows)), "optimizer_updates": str(20 * len(rows)),
        "final_report_accepted": "false"}, "completed measurement counts")
    bind_manifest(campaign / "controller.sha256", inputs)
    accepted_hashes = bind_manifest(campaign / "accepted-proofs.sha256", inputs)
    qualification_hashes = bind_manifest(campaign / "qualification-inputs.sha256", inputs)
    preflight = inputs.read(campaign / "preflight.json")
    record_matches(preflight, {"valid_preflight": True, "training_executed": False, "native_buffer_allocated": False,
        "partition": "debug", "experiment_protocol": adapter.PROTOCOL}, "actual budget preflight")
    supported = preflight["allowed_budgets"]
    require(supported == sorted(set(supported)) and all(adapter.allowed_budget(value) for value in supported), "actual qualified budget list")
    equal(supported, preflight["device_launch_limits"]["supported_budgets"], "actual preflight device limits")
    equal(sorted(map(int, preflight["native_configs"])), supported, "native Config/hints for all qualified budgets")
    record_matches(preflight["paired_launcher_stdin_fixture"], {"valid_actual_pair_launcher_stdin_fixture": True,
        "orders_checked": ["candidate-first", "reference-first"], "pairs_checked": 2,
        "completed_fixture_cases": 4, "stub_srun_call_count": 8,
        "native_training_executed": False, "real_scheduler_commands_executed": False}, "both actual launcher orders passed QA")
    source = adapter.verify_source_manifest(campaign / "campaign-source.sha256", preflight["source_manifest_sha256"])
    require(all(summary_source["files"].get(name) == digest for name, digest in source["files"].items()), "summary changes qualified training source")
    qa_sources = [path for path in qualification_hashes if Path(path).name == "preflight.json"]
    require(len(qa_sources) == 1, "qualification must pin its actual preflight")
    equal(inputs.hashes[str(campaign / "preflight.json")], qualification_hashes[qa_sources[0]], "copied versus original qualified preflight")
    qualification = inputs.read(campaign / "qualification_pair_comparison.json")
    record_matches(qualification, {"valid_paired_budget_comparison": True, "qualification_only": True, "execution_steps": 2,
        "source_manifest_sha256": source["sha256"], "experiment_protocol": adapter.PROTOCOL}, "excluded real paired qualification")
    require(qualification["pair_id"] not in {row["pair_id"] for row in rows}, "qualification cannot become a full measurement")
    original_qualification = str(RUNS / f"{qualification['pair_id']}-posthoc/pair_comparison.json")
    equal(inputs.hashes[str(campaign / "qualification_pair_comparison.json")], qualification_hashes[original_qualification],
          "copied versus original excluded qualification")
    inputs.archive(args.accounting_file, "evidence/accounting_utc.psv")
    account = read_accounting(args.accounting_file, {row[name] for row in rows for name in ("training_job", "posthoc_job")})
    original_path = RUNS / "qwen-debug-results-20261105T0925Z-cpu-validation/campaign_acceptance.json"
    original = inputs.read(original_path)
    require(original.get("valid_debug_campaign") is True and SHA.fullmatch(original["sample_prefix_sha256"]), "original accepted full input prefix required")
    prefix = original["sample_prefix_sha256"]
    pairs, raw_pairs, decisions, workload, normalized = [], [], [], None, None
    for index, row in enumerate(rows):
        print(json.dumps({"stage": "reconcile_accepted_pair", "pair": row["pair_id"], "index": index + 1, "total": len(rows)}), flush=True)
        next_pair = compare.plan_next_pair(raw_pairs, supported)
        record_matches(next_pair, {"action": "run_pair", "next_pair": {key: row[key] for key in ("candidate_budget", "order", "replicate")}},
                       "next accepted pair matches predeclared planner")
        paired, raw, identity, cfg = check_pair(row, account, inputs, source, prefix, accepted_hashes)
        if workload is None:
            workload, normalized = identity, cfg
        equal(identity, workload, "common source, workload and consumed input across the entire sweep")
        evidence.equal_resolved_configuration(cfg, normalized, "only budget and output locations differ across full cases")
        raw_pairs.append(raw)
        pairs.append(paired)
        path = Path(row["proof_path"]).parent / "next_decision.json"
        require(str(path) in accepted_hashes, "accepted pair lacks pinned planner decision")
        decision = inputs.read(path)
        check_decision(decision, compare.plan_next_pair(raw_pairs, supported), row, rows[:index], source["sha256"],
                       source["files"]["analysis/ep_overlap_budget_compare.py"], supported, inputs)
        copy = campaign / "decisions" / f"{row['pair_id']}.json"
        equal(inputs.bind(copy), inputs.hashes[str(path)], "controller's retained planner decision")
        inputs.archive(path, Path("evidence/decisions") / f"{row['pair_id']}.json")
        decisions.append({"pair_id": row["pair_id"], "action": decision["action"], "next_pair": decision["next_pair"],
                          "reason": decision["reason"], "path": str(path), "sha256": inputs.hashes[str(path)]})
    record_matches(decision, {"action": "complete", "next_pair": None, "global_optimum_claimed": False}, "final adaptive stop")
    equal(sorted(decision["boundary_status"]), ["lower", "upper"], "both boundaries have evidence")
    inputs.recheck()
    pair_table, leg_table, budget_table = export_tables(out, pairs, decision["budget_statistics"])
    plot_ratios(out, pairs, decision["budget_statistics"])
    result = {"schema_version": 1, "valid_complete_adaptive_measurement_summary": True,
        "research_conclusions_independently_accepted": False, "experiment_protocol": adapter.PROTOCOL,
        "campaign_dir": str(campaign), "accepted_pair_count": len(pairs), "full_case_count": len(leg_table),
        "optimizer_updates": 20 * len(pairs), "measured_optimizer_updates": 10 * len(pairs),
        "excluded_qualification_pair_id": qualification["pair_id"], "failed_attempt_count": len(failures),
        "training_source_manifest": source, "summary_source_manifest": summary_source,
        "common_workload": workload, "unchanged_resolved_configuration_verified": True,
        "independent_replicate_unit": "one same-placement shared allocation pair",
        "per_case_slurm_elapsed_available": False, "qualified_budgets": supported,
        "preflight_device_limits": preflight["device_launch_limits"], "next_unsupported_upper_budget": preflight["next_unsupported_upper_budget"],
        "hardware_metrics": {"available": False, "reason": "Profiler OFF; no sampled device activity or achieved occupancy.",
                             "configured_budget_unit": "payload blocks", "free_sms": None},
        "decision_history": decisions, "final_decision": {key: value for key, value in decision.items() if key != "input_files"},
        "pairs": pairs, "pair_table": pair_table, "leg_table": leg_table, "budget_table": budget_table,
        "archive_manifest": inputs.archives, "report_promotion_target": REVIEW_TARGET,
        "claim_limits": ["Five measured updates per leg are sequential subsamples. Separate pairs are allocation replicates.",
            "Same GPUs within a pair and reversed order reduce confounds, but do not remove cache, clock or network drift.",
            "Common routing, receive, input and health observers remain enabled; their cost is in the timings.",
            "The 2% threshold and observed range are operational evidence, not significance or confidence intervals.",
            "No free-SM, achieved occupancy, physical traffic, global optimum or long-run training-quality claim."],
        "execution": {"slurm_job_id": os.environ["SLURM_JOB_ID"], "partition": "debug", "hostname": socket.gethostname(),
                      "python": sys.executable, "finished_utc": datetime.now(timezone.utc).isoformat(), "training_executed": False}}
    with (out / REPORT_NAME).open("x", encoding="utf-8") as stream:
        stream.write(render_report(result, out))
    references = [{"path": path, "sha256": digest} for path, digest in sorted(inputs.hashes.items())]
    write_new(out / "input_references.json", {"input_files": references, "bulk_artifacts_copied": False,
                                             "small_evidence_and_timer_logs_archived": True})
    result["artifacts"] = {name: {"path": str(out / name), "sha256": adapter.sha(out / name)} for name in (
        "optimizer_steps.csv", "legs.csv", "pairs.csv", "budgets.csv", "paired_budget_comparison.png", "paired_budget_comparison.svg",
        REPORT_NAME, "input_references.json")}
    write_new(out / "metrics.json", result)
    with (out / "VALIDATED.txt").open("x", encoding="utf-8") as stream:
        stream.write(f"valid_complete_adaptive_measurement_summary=true\nmetrics_sha256={adapter.sha(out / 'metrics.json')}\n"
                     "research_conclusions_independently_accepted=false\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--accounting-file", type=Path, required=True)
    parser.add_argument("--summary-source-manifest", type=Path, required=True)
    parser.add_argument("--summary-source-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    require(os.environ.get("SLURM_JOB_PARTITION") == "debug" and os.environ.get("SLURM_JOB_NUM_NODES") == "1"
            and os.environ.get("SLURM_JOB_ID", "").isdigit() and Path(sys.prefix) == Path("/opt/venvs/megatron-bridge"),
            "allocated one-node debug configured-container execution required")
    result = summarize_campaign(args)
    print(json.dumps({"valid_complete_adaptive_measurement_summary": True, "accepted_pairs": result["accepted_pair_count"],
                      "output_dir": str(args.output_dir)}))


if __name__ == "__main__":
    main()
