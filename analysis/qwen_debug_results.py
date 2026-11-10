#!/usr/bin/env python3
"""Describe eight separately allocated 5+5 experiments from bound raw records.

This report does not change acquisition data or treat separate allocations as
matched causal trials. A diagnostic export can precede full capture acceptance;
the final deck must use a report with all case and capture checks complete.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import socket

from analysis.compare_uccl_profiles import CASE_CONFIG, routing_summary
from analysis.extract_step_timing import select_timer_log, timing_rows
from analysis.summarize_training_evidence import summarize
from analysis.compare_deep_trace_overhead import _checked_fingerprint_batches

ROOT = Path(__file__).resolve().parents[1]
CASES = ("ordinary", "grouped", "uccl_ordinary", "uccl")
LEGS = ("control", "profile")
TOKENS = 64 * 4096
LIMITS = [
    "Five successive measured updates are descriptive samples from one allocation, not five independent experiments.",
    "Each case uses a separate allocation; placement, network contention, clocks and time may differ.",
    "Headline ratios use profiler-OFF complete optimizer-step timers, with routing, receive and fingerprint audits still enabled.",
    "UCCL membership/layout audit work differs from native audit work and remains in this measured setup.",
    "The native transport provider was not recorded; this is the native baseline as launched, not a demonstrated tuned transport.",
    "Observed ON/OFF changes do not isolate profiler overhead; routes and physical placement can differ.",
    "Routing counts and unique token destinations are forward logical quantities, not physical wire bytes or bandwidth.",
    "Inclusive CPU scopes and concurrent GPU activity do not partition step elapsed time; exposed latency and CPU proxy time are unavailable.",
    "Finite losses and gradients establish logged training health, not model-quality equivalence or proof that every parameter changed.",
    "Later acquisitions reuse the first successful immutable-input verification; they do not establish fresh payload verification.",
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


class Inputs:
    def __init__(self):
        self.hashes = {}
        self.events = []

    def bind(self, path):
        path = Path(path)
        require(path.is_absolute() and path.is_file() and not path.is_symlink(), f"regular absolute input required: {path}")
        digest = sha256(path)
        old = self.hashes.setdefault(str(path), digest)
        require(old == digest, f"input changed: {path}")
        self.events.append(str(path))
        return digest

    def read(self, path):
        expected = self.bind(path)
        payload = Path(path).read_bytes()
        require(hashlib.sha256(payload).hexdigest() == expected, f"JSON input changed while reading: {path}")
        return json.loads(payload)

    def recheck(self):
        for path, digest in self.hashes.items():
            require(sha256(path) == digest, f"input changed during reporting: {path}")


def describe(values):
    require(len(values) == 5 and all(math.isfinite(x) and x > 0 for x in values), "five finite positive step timers required")
    mean = statistics.fmean(values)
    sd = statistics.stdev(values)
    return {"values_s": values, "mean_s": mean, "median_s": statistics.median(values),
        "sd_s": sd, "cv_percent": 100 * sd / mean, "min_s": min(values), "max_s": max(values),
        "tokens_per_s": len(values) * TOKENS / sum(values), "sample_count": 5}


def load_case_list(path):
    rows = []
    for row in Path(path).read_text().splitlines():
        fields = row.split("\t")
        require(len(fields) == 4, "case list requires case, leg, job and acquisition root")
        case, leg, job, raw = fields
        require(case in CASES and leg in LEGS and job.isdigit(), "unexpected campaign case")
        orchestration = Path(raw)
        require(orchestration.is_absolute() and orchestration.parent == ROOT / "runs", "campaign root outside runs")
        rows.append((case, leg, job, orchestration))
    require(len(rows) == 8 and {(case, leg) for case, leg, _, _ in rows} == {(c, l) for c in CASES for l in LEGS}, "complete unique four-case/two-leg grid required")
    require(len({str(row[3]) for row in rows}) == 8, "acquisitions must be distinct")
    return rows


def build(case_list, output, validation_root=None, diagnostic=False, allow_login_cpu=False):
    allocated = bool(os.environ.get("SLURM_JOB_ID") and os.environ.get("SLURM_JOB_PARTITION") in ("debug", "preemptable"))
    require(allocated or allow_login_cpu, "allocated compute or explicit user-authorized lightweight CPU execution required")
    require(output.is_absolute() and output.parent == ROOT / "runs" and not output.exists(), "new output directly under runs required")
    require(diagnostic or validation_root is not None, "final reporting requires a validation root")
    inputs = Inputs()
    inputs.bind(case_list)
    inputs.bind(Path(__file__).resolve())
    for relative in ("analysis/compare_uccl_profiles.py", "analysis/extract_step_timing.py", "analysis/summarize_training_evidence.py", "analysis/compare_deep_trace_overhead.py"):
        inputs.bind(ROOT / relative)
    campaign = None
    if validation_root and not diagnostic:
        campaign = inputs.read(validation_root / "campaign_acceptance.json")
        require(campaign.get("valid_debug_campaign") is True, "campaign acceptance did not pass")
        require((campaign.get("warmup_steps"), campaign.get("measured_steps"), campaign.get("completed_updates_per_case"), campaign.get("rank_update_captures")) == (5, 5, 10, 640), "campaign acceptance window/capture coverage differs")
        expected_matrix = {(case, leg, job, str(directory)) for case, leg, job, directory in load_case_list(case_list)}
        actual_matrix = {(row["case"], row["leg"], row["training_slurm_job_id"], row["orchestration_dir"]) for row in campaign["cases"]}
        require(len(campaign["cases"]) == 8 and actual_matrix == expected_matrix, "campaign accepted acquisition grid differs")
    cases = {case: {"label": CASE_CONFIG[case]["label"], "backend": CASE_CONFIG[case]["backend"],
        "num_groups": CASE_CONFIG[case]["num_groups"], "group_topk": CASE_CONFIG[case]["group_topk"],
        "routing": "grouped" if case in ("grouped", "uccl") else "ordinary", "legs": {}, "routing_by_leg": {}} for case in CASES}
    common_identities, prefixes, capture_checks = [], {}, []
    route_hashes = {}
    source_windows = []
    for case, leg, job, orchestration in load_case_list(case_list):
        case_event_start = len(inputs.events)
        acquisition = inputs.read(orchestration / "acquisition.json")
        completed = inputs.read(orchestration / "acquisition_completed.json")
        require(completed.get("success") is True and completed.get("workers_srun_returned_zero") is True, "acquisition did not complete")
        require((completed["warmup_steps"], completed["measured_steps"], completed["completed_updates"]) == (5, 5, 10), "acquisition window differs")
        require(completed["job"] == acquisition["job"] == job, "job identity differs")
        rows = (orchestration / "cases.tsv").read_text().splitlines()
        inputs.bind(orchestration / "cases.tsv")
        require(len(rows) == 1 and rows[0].split("\t")[:2] == [case, leg], "acquisition case differs")
        run = Path(rows[0].split("\t")[2])
        require(run.is_absolute() and run.parent == ROOT / "runs", "run outside project")
        contract = inputs.read(run / "config/comparison_contract.json")
        environment = inputs.read(run / "environment.json")
        ownership = inputs.read(run / "config/routing_ownership.json")
        inputs.read(run / "config/debug_execution_contract.json")
        proof = inputs.read(orchestration / "reviewed_inputs.json")
        health = summarize(run, steps=10, interval=1)
        log = select_timer_log(run / "logs")
        inputs.bind(log)
        all_rows = timing_rows(log, warmup_steps=0, measured_steps=10, log_interval=1, tokens_per_step=TOKENS)
        all_values = [row["mean_optimizer_step_ms"] / 1000 for row in all_rows]
        timing = describe(all_values[5:])
        timing["all_values_s"] = all_values
        timing["timer_log"] = str(log)
        fingerprint_rows = {}
        for rank in range(16):
            path = run / "raw" / f"sample_fingerprint_rank-{rank}.json"
            fingerprint_rows[str(rank)] = _checked_fingerprint_batches(inputs.read(path), rank=rank, accumulation=4, steps=10, token_shape=[1, 4096], path=path)
        prefix = hashlib.sha256(json.dumps(fingerprint_rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        prefixes[f"{case}-{leg}"] = prefix
        common_identities.append((contract["c0_identity_manifest_sha256"], contract["dataset"]["manifest_sha256"], environment["ampa_runtime_sources"]["sha256"]))
        audits = [inputs.read(run / "raw" / f"routing_audit_rank-{rank}.json") for rank in range(16)]
        routing = routing_summary(audits, ownership, range(5, 10))
        totals, locality, dedup = routing["totals"], routing["locality_percent"], routing["dedup"]
        cases[case]["routing_by_leg"][leg] = {"total_assignments": totals["assignments"],
            "local_assignments": totals["local"], "intra_node_assignments": totals["intra_node"], "inter_node_assignments": totals["inter_node"],
            "local_percent": locality["local"], "intra_node_percent": locality["intra_node"], "inter_node_percent": locality["inter_node"],
            "token_rank_deliveries": dedup.get("unique_rank_deliveries"), "token_node_deliveries": dedup.get("unique_node_deliveries"),
            "routed_tokens": totals["assignments"] // 8, "dedup": dedup, "details": routing}
        route_hashes[f"{case}-{leg}"] = hashlib.sha256(json.dumps([
            [[{key: row[key] for key in ("expert_counts", "rank_traffic")} for row in step["layers"]] for step in audit["steps"]]
            for audit in audits], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        accepted, validation_dir = False, None
        if validation_root:
            validation_dir = validation_root / "cases" / f"{case}-{leg}"
            acceptance_path = validation_dir / "acceptance.json"
            if acceptance_path.is_file():
                acceptance = inputs.read(acceptance_path)
                require(acceptance.get("case") == case and acceptance.get("leg") == leg and acceptance.get("training_slurm_job_id") == job, "validation case/job differs")
                require(acceptance.get("sample_prefix_sha256") == prefix, "validation consumed-prefix mismatch")
                accepted = acceptance.get("valid_debug_case") is True
                require(accepted, "case acceptance failed")
                require(acceptance.get("valid_training_health") is True and acceptance.get("valid_send_receive_audit") is True, "case health/receive acceptance missing")
                accepted_inputs = {row["path"]: row["sha256"] for row in acceptance["input_files"]}
                for consumed in set(inputs.events[case_event_start:]):
                    if consumed != str(acceptance_path):
                        require(accepted_inputs.get(consumed) == inputs.hashes[consumed], f"report input differs from case acceptance: {consumed}")
                if campaign:
                    binding = campaign["case_acceptance"][case][leg]
                    require(binding["path"] == str(acceptance_path) and binding["sha256"] == inputs.bind(acceptance_path), "case proof differs from accepted campaign")
                    require(campaign["case_runs"][case][leg] == str(run), "case run differs from accepted campaign")
                if leg == "profile":
                    require(acceptance.get("valid_capture") is True, "profile not accepted")
                    capture_checks.append(True)
            else:
                require(diagnostic, f"missing case acceptance: {acceptance_path}")
        item = {"run_dir": str(run), "orchestration_dir": str(orchestration), "job": job,
            "timing": timing, "health": health, "rank_placement": environment["rank_placement"],
            "validation_dir": str(validation_dir) if validation_dir else None,
            "validation_status": "passed" if accepted else "pending", "sample_prefix_sha256": prefix,
            "input_verification_mode": proof.get("verification_mode", "verified_in_first_acquisition"),
            "first_update_s": all_values[0], "source_runtime_sha256": environment["ampa_runtime_sources"]["sha256"]}
        cases[case]["legs"][leg] = item
        source_windows.append({"case": case, "leg": leg, "log": str(log), "measured_human_updates": [6,7,8,9,10]})
        print(json.dumps({"reported": f"{case}-{leg}", "mean_step_s": timing["mean_s"], "validated": accepted}), flush=True)
    require(len(set(common_identities)) == 1, "C0, dataset or runtime source differs across runs")
    require(len(set(prefixes.values())) == 1, "consumed input fingerprints differ across runs")
    # Keep a distinct metadata field for routing mode; expose measured routing
    # populations under the schema shared with the deck and independent audit.
    for case in CASES:
        cases[case]["routing_mode"] = cases[case].pop("routing")
        cases[case]["routing"] = cases[case].pop("routing_by_leg")
        cases[case]["on_off_route_histograms_identical"] = route_hashes[f"{case}-control"] == route_hashes[f"{case}-profile"]
    contrasts = []
    for base, target, label in (("ordinary", "uccl_ordinary", "Ordinary: UCCL versus native"),
            ("grouped", "uccl", "Grouped: UCCL versus native"), ("ordinary", "grouped", "Native: grouped versus ordinary"),
            ("uccl_ordinary", "uccl", "UCCL: grouped versus ordinary")):
        first, second = (cases[c]["legs"]["control"]["timing"]["mean_s"] for c in (base, target))
        contrasts.append({"from": base, "to": target, "label": label, "speedup": first / second,
            "duration_change_percent": 100 * (second / first - 1), "leg": "control", "causal_effect_claimed": False})
    all_cases_valid = all(cases[c]["legs"][l]["validation_status"] == "passed" for c in CASES for l in LEGS)
    captures_valid = len(capture_checks) == 4 and all(capture_checks)
    if not diagnostic:
        require(all_cases_valid and captures_valid, "final report requires all cases and captures accepted")
        require(campaign["sample_prefix_sha256"] == next(iter(prefixes.values())), "campaign consumed-prefix identity differs")
    first = next(iter(cases.values()))["legs"]["control"]
    first_contract = inputs.read(Path(first["run_dir"]) / "config/comparison_contract.json")
    report = {"schema_version": 1, "kind": "qwen_debug_5plus5_report", "diagnostic": diagnostic,
        "model": first_contract["model"], "workload": {"sequence_length":4096,"global_batch_size":64,"micro_batch_size":1,"gradient_accumulation":4,
            "nodes":4,"gpus":16,"expert_parallel":16,"context_parallel":1,"precision":"BF16","top_k":8},
        "protocol": {"warmup_steps":5,"measured_steps":5,"completed_steps":10,"tokens_per_step":TOKENS},
        "routing_window": {"step_indices":[5,6,7,8,9],"human_updates":[6,7,8,9,10],"label":"5 measured updates"},
        "cases":cases,"contrasts":contrasts,
        "acceptance": {"valid_descriptive_comparison":all_cases_valid and captures_valid,
            "all_case_metadata_valid":all_cases_valid,"all_capture_valid":captures_valid,
            "rank_update_captures":640 if captures_valid else None,"identical_consumed_prefix":True,
            "same_checkpoint_dataset_runtime":True,"pure_profiler_overhead_available":False,
            "independent_numerical_audit":"pending","same_physical_placement_claimed":False},
        "sample_prefix_sha256":next(iter(prefixes.values())), "runtime_source_sha256": common_identities[0][2],
        "c0_identity_manifest_sha256":common_identities[0][0],"dataset_manifest_sha256":common_identities[0][1],
        "source_windows":source_windows,"claim_limits":LIMITS,
        "analysis_slurm_job_id":os.environ.get("SLURM_JOB_ID"),
        "analysis_execution": {"hostname":socket.gethostname(), "allocated_compute":allocated,
            "user_authorized_lightweight_cpu":bool(allow_login_cpu), "gpu_workload":False}}
    if not diagnostic:
        from analysis.qwen_debug_profile_phases import collect
        report["phase_summaries"] = collect(cases, inputs)
    inputs.recheck()
    report["input_files"] = [{"path":path,"sha256":digest} for path,digest in sorted(inputs.hashes.items())]
    output.mkdir()
    with (output / "report.json").open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    with (output / "timing.csv").open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("case","leg","job","mean_s","sample_sd_s","cv_percent","tokens_per_s","update6_s","update7_s","update8_s","update9_s","update10_s"))
        for case in CASES:
            for leg in LEGS:
                row = cases[case]["legs"][leg]
                timing = row["timing"]
                writer.writerow((case,leg,row["job"],timing["mean_s"],timing["sd_s"],timing["cv_percent"],timing["tokens_per_s"],*timing["values_s"]))
    print(json.dumps({"report":str(output/"report.json"),"final_acceptance":all_cases_valid and captures_valid}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-list", type=Path, required=True)
    parser.add_argument("--validation-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument("--allow-login-cpu", action="store_true", help="Use only for explicitly user-authorized lightweight CPU posthoc work.")
    args = parser.parse_args()
    build(args.case_list.resolve(), args.output_dir, args.validation_root, args.diagnostic, args.allow_login_cpu)


if __name__ == "__main__":
    main()
