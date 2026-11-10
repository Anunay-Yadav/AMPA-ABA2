#!/usr/bin/env python3
"""Independently reconstruct the short Qwen campaign from recorded evidence.

No report-builder, trace-validator, timing-helper, or routing-helper code is
imported. Use allocated compute, or the explicitly authorized isolated CPU venv.
Raw traces and immutable checkpoint/data payloads are not parsed or rehashed.
Their recorded validation and historical verification remain explicit.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import socket
import sys


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ACQUISITIONS = ROOT / "runs/uccl-debug5x5-reordered-20261104T2221Z-reordered-coordinator/acquisitions.tsv"
CASES = ("ordinary", "grouped", "uccl_ordinary", "uccl")
LEGS = ("control", "profile")
CONTRASTS = (("ordinary", "uccl_ordinary"), ("grouped", "uccl"),
             ("ordinary", "grouped"), ("uccl_ordinary", "uccl"))
TOKENS = 262144
WINDOW = list(range(5, 10))
TIMER = re.compile(r"\biteration\s+(\d+)\s*/\s*(\d+)\s*\|.*?elapsed time per iteration \(ms\):\s*([0-9.]+)\s*\|")
HEX = re.compile(r"[0-9a-f]{64}\Z")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, label):
    require(type(actual) in (int, float) and math.isfinite(actual)
            and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-9),
            f"{label}: report {actual!r}, independently reconstructed {expected!r}")


def exact(actual, expected, label):
    if actual != expected:
        raise ValueError(f"{label}: {repr(actual)[:400]} != {repr(expected)[:400]}")


def serialized(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def execution_context(allow_login_cpu):
    if allow_login_cpu:
        expected = ROOT / ".runtime/venvs/qwen-posthoc-cpu"
        exact(Path(sys.prefix).resolve(), expected.resolve(), "authorized isolated CPU venv")
        mode, job, partition = "user_authorized_login_cpu", None, None
    else:
        job = os.environ.get("SLURM_JOB_ID", "")
        partition = os.environ.get("SLURM_JOB_PARTITION")
        require(job.isdigit() and partition in ("debug", "preemptable"),
                "configured Slurm compute or explicit --allow-login-cpu authorization required")
        mode = "allocated_compute"
    packages = {}
    for distribution in metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            packages[name] = distribution.version
    return {"mode": mode, "hostname": socket.gethostname(), "slurm_job_id": job,
            "slurm_partition": partition, "python_executable": sys.executable,
            "python_version": platform.python_version(), "python_build": sys.version,
            "environment_prefix": sys.prefix, "installed_packages": dict(sorted(packages.items())),
            "external_packages_used_by_audit": [], "gpu_workload": False,
            "raw_trace_parsing": False,
            "authorization": "D-028: explicit lightweight CPU-only login analysis" if allow_login_cpu else "configured compute allocation"}


class Evidence:
    def __init__(self):
        self.hashes = {}

    def read(self, path):
        path = Path(path)
        require(path.is_absolute() and path.is_file() and not path.is_symlink(), f"regular absolute evidence required: {path}")
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        old = self.hashes.setdefault(str(path), digest)
        exact(old, digest, f"evidence changed while auditing {path}")
        return payload

    def json(self, path):
        return json.loads(self.read(path))

    def digest(self, path):
        self.read(path)
        return self.hashes[str(Path(path))]


def case_map(evidence, path):
    rows = []
    for line in evidence.read(path).decode().splitlines():
        fields = line.split("\t")
        require(len(fields) == 4, "acquisitions.tsv requires four columns")
        case, leg, job, directory = fields
        require(case in CASES and leg in LEGS and job.isdigit(), "unexpected case, leg, or job")
        orchestration = Path(directory)
        require(orchestration.is_absolute() and orchestration.parent == ROOT / "runs", "orchestration outside project runs")
        small_rows = evidence.read(orchestration / "cases.tsv").decode().splitlines()
        require(len(small_rows) == 1 and len(small_rows[0].split("\t")) == 3, "one case per allocation required")
        named_case, named_leg, directory = small_rows[0].split("\t")
        exact((named_case, named_leg), (case, leg), "acquisition case map")
        run = Path(directory)
        require(run.is_absolute() and run.parent == ROOT / "runs" and run.is_dir() and not run.is_symlink(), "case run outside project runs")
        rows.append((case, leg, job, orchestration, run))
    exact(len(rows), 8, "acquisition count")
    exact({(r[0], r[1]) for r in rows}, {(c, l) for c in CASES for l in LEGS}, "unique eight-case grid")
    exact(len({r[2] for r in rows}), 8, "independent training jobs")
    exact(len({r[4] for r in rows}), 8, "independent case runs")
    return rows


def timers(evidence, run):
    found = []
    for path in sorted((run / "logs").glob("rank-*.out")):
        lines = evidence.read(path).decode(errors="replace").splitlines()
        matched = [(line, TIMER.search(line)) for line in lines]
        matched = [(line, match) for line, match in matched if match]
        if matched:
            found.append((path, matched))
    exact(len(found), 1, "one timer-bearing rank log")
    path, rows = found[0]
    exact([int(m.group(1)) for _, m in rows], list(range(1, 11)), "ten consecutive completed updates")
    exact({int(m.group(2)) for _, m in rows}, {25}, "unchanged train_iters cache budget")
    values = []
    metrics = {"learning rate": [], "lm loss": [], "load_balancing_loss": [], "grad norm": []}
    for step, (line, match) in enumerate(rows, 1):
        value = float(match.group(3)) / 1000
        require(math.isfinite(value) and value > 0, "nonpositive or nonfinite step time")
        values.append(value)
        for name, expected in (("consumed samples", step * 64), ("global batch size", 64),
                               ("number of skipped iterations", 0), ("number of nan iterations", 0)):
            item = re.search(rf"{re.escape(name)}:\s*(\d+)\s*\|", line)
            require(item is not None and int(item.group(1)) == expected, f"step {step}: invalid {name}")
        for name in metrics:
            item = re.search(rf"(?:^|\|)\s*{re.escape(name)}:\s*([^| ]+)", line)
            require(item is not None, f"step {step}: missing {name}")
            number = float(item.group(1))
            require(math.isfinite(number), f"step {step}: nonfinite {name}")
            metrics[name].append(number)
        require(metrics["grad norm"][-1] > 0, "nonpositive gradient norm")
        close(metrics["learning rate"][-1], 1e-6, "constant learning rate")
    measured = values[5:]
    mean = sum(measured) / 5
    sd = math.sqrt(sum((x - mean) ** 2 for x in measured) / 4)
    return {"values_s": measured, "all_values_s": values, "mean_s": mean,
            "median_s": sorted(measured)[2], "sd_s": sd, "cv_percent": 100 * sd / mean,
            "min_s": min(measured), "max_s": max(measured), "tokens_per_s": 5 * TOKENS / sum(measured),
            "sample_count": 5, "timer_log": str(path), "health_ranges": {
                name: {"min": min(a), "max": max(a), "mean": sum(a) / len(a)} for name, a in metrics.items()}}


def fingerprints(evidence, run):
    result = {}
    for rank in range(16):
        record = evidence.json(run / "raw" / f"sample_fingerprint_rank-{rank}.json")
        exact((record.get("schema_version"), record.get("kind"), record.get("rank")),
              (1, "consumed_cpu_token_batch_sha256", rank), "fingerprint identity")
        rows = record["steps"]
        exact([r["step"] for r in rows], list(range(10)), "fingerprint step order")
        batches = []
        for row in rows:
            exact(len(row["batches"]), 4, "four real microbatches per update")
            for batch in row["batches"]:
                require("tokens" in batch and "labels" in batch and "loss_mask" in batch, "missing training tensor fingerprint")
                for tensor in batch.values():
                    exact(set(tensor), {"sha256", "shape", "dtype"}, "tensor fingerprint fields")
                    require(isinstance(tensor["sha256"], str) and HEX.fullmatch(tensor["sha256"]), "invalid tensor digest")
                    require(isinstance(tensor["shape"], list) and all(type(v) is int and v >= 0 for v in tensor["shape"]), "invalid tensor shape")
                    require(isinstance(tensor["dtype"], str) and tensor["dtype"].startswith("torch."), "invalid tensor dtype")
                exact(batch["tokens"]["shape"], [1, 4096], "actual token batch shape")
                exact(batch["tokens"]["dtype"], "torch.int64", "actual token dtype")
            batches.append(row["batches"])
        result[str(rank)] = batches
    return hashlib.sha256(serialized(result)).hexdigest()


def routing(evidence, run, backend, grouped, placement):
    ownership = evidence.json(run / "config/routing_ownership.json")
    hosts = {int(rank): host for rank, host in ownership["rank_hosts"].items()}
    exact(set(hosts), set(range(16)), "runtime rank host coverage")
    exact(len(set(hosts.values())), 4, "four actual nodes")
    exact(hosts, {row["rank"]: row["hostname"] for row in placement}, "ownership/placement host agreement")
    owner_maps = ownership["expert_owners"]
    exact(set(owner_maps), {str(n) for n in range(48)}, "48 runtime ownership maps")
    for owners in owner_maps.values():
        exact(len(owners), 128, "128 expert owners")
        exact(Counter(owners), Counter({rank: 8 for rank in range(16)}), "eight experts on every owner")
    outgoing = [[[0] * 128 for _ in range(48)] for _ in range(10)]
    incoming = [[[0] * 128 for _ in range(48)] for _ in range(10)]
    counts = {"total_assignments": 0, "local_assignments": 0, "intra_node_assignments": 0,
              "inter_node_assignments": 0, "token_rank_deliveries": 0 if backend == "uccl_ep" else None,
              "token_node_deliveries": 0 if backend == "uccl_ep" else None}
    remote_nodes = 0
    route_hash = hashlib.sha256()
    for rank in range(16):
        record = evidence.json(run / "raw" / f"routing_audit_rank-{rank}.json")
        exact((record["rank"], record["valid"], record["communication_backend"]), (rank, True, backend), "routing rank/backend identity")
        exact((record["num_groups"], record["group_topk"]), (4, 2) if grouped else (None, None), "routing mode")
        exact([r["step"] for r in record["steps"]], list(range(10)), "routing ten-step coverage")
        for step in record["steps"]:
            s = step["step"]
            exact([r["layer"] for r in step["layers"]], list(range(48)), "48 ordered routing layers")
            for row in step["layers"]:
                layer = row["layer"]
                experts, traffic = row["expert_counts"], row["rank_traffic"]
                exact((len(experts), len(traffic)), (128, 16), "routing vector sizes")
                require(all(type(v) is int and v >= 0 for v in experts + traffic), "invalid routing counts")
                exact((row["routed_tokens"], row["assignments"]), (16384, 131072), "routed positions times top8")
                exact((sum(experts), sum(traffic)), (131072, 131072), "top8 conservation")
                rebuilt = [0] * 16
                for expert, value in enumerate(experts):
                    rebuilt[owner_maps[str(layer)][expert]] += value
                    outgoing[s][layer][expert] += value
                exact(rebuilt, traffic, "traffic reconstructed from actual expert ownership")
                local = traffic[rank]
                intra = sum(v for destination, v in enumerate(traffic) if destination != rank and hosts[destination] == hosts[rank])
                inter = sum(v for destination, v in enumerate(traffic) if hosts[destination] != hosts[rank])
                exact((row["local"], row["intra_node"], row["inter_node"]), (local, intra, inter), "runtime locality reconstruction")
                exact(local + intra + inter, 131072, "locality conservation")
                owned = [e for e, owner in enumerate(owner_maps[str(layer)]) if owner == rank]
                exact(row["received_expert_ids"], owned, "actual receive expert ownership")
                received = row["received_expert_counts"]
                require(len(received) == len(owned) and all(type(v) is int and v >= 0 for v in received), "invalid receive counts")
                exact((sum(received), row["receive_calls"]), (row["received_assignments"], 4), "receive accounting")
                for expert, value in zip(owned, received):
                    incoming[s][layer][expert] += value
                route_hash.update(serialized([rank, s, layer, experts, traffic]))
                if backend == "uccl_ep":
                    ranks, nodes, names = row["unique_token_rank_counts"], row["unique_token_node_counts"], row["unique_token_node_hosts"]
                    require(len(ranks) == 16 and len(nodes) == len(names) == 4 and len(set(names)) == 4
                            and set(names) == set(hosts.values()), "unique-delivery destination coverage")
                    require(all(type(v) is int and 0 <= v <= 16384 for v in ranks + nodes), "invalid unique delivery count")
                    require(16384 <= sum(nodes) <= sum(ranks) <= 131072, "unique delivery conservation/bounds")
                    require(sum(nodes) <= 16384 * (2 if grouped else 4), "ordinary/grouped destination-node bound")
                    exact((row["layout_calls"], row["metadata_calls"]), (4, 4), "UCCL layout/metadata coverage")
                if s in WINDOW:
                    counts["total_assignments"] += 131072
                    counts["local_assignments"] += local
                    counts["intra_node_assignments"] += intra
                    counts["inter_node_assignments"] += inter
                    if backend == "uccl_ep":
                        counts["token_rank_deliveries"] += sum(ranks)
                        counts["token_node_deliveries"] += sum(nodes)
                        remote_nodes += sum(v for host, v in zip(names, nodes) if host != hosts[rank])
    exact(outgoing, incoming, "all-source routed assignments versus actual received expert counts")
    exact(counts["total_assignments"], 503316480, "five-update assignment population")
    for name in ("local", "intra_node", "inter_node"):
        counts[f"{name}_percent"] = 100 * counts[f"{name}_assignments"] / counts["total_assignments"]
    counts["routed_tokens"] = counts["total_assignments"] // 8
    counts["remote_token_node_deliveries"] = remote_nodes if backend == "uccl_ep" else None
    counts["route_fingerprint_sha256"] = route_hash.hexdigest()
    counts["all_ten_update_rows_checked"] = 16 * 10 * 48
    return counts


def identities(evidence, case, leg, job, orchestration, run):
    acquired = evidence.json(orchestration / "acquisition.json")
    completed = evidence.json(orchestration / "acquisition_completed.json")
    proof = evidence.json(orchestration / "reviewed_inputs.json")
    contract = evidence.json(run / "config/comparison_contract.json")
    execution = evidence.json(run / "config/debug_execution_contract.json")
    environment = evidence.json(run / "environment.json")
    for record in (acquired, completed):
        exact(record["job"], job, "current acquisition job")
        exact((record["warmup_steps"], record["measured_steps"], record["completed_updates"]), (5, 5, 10), "actual acquisition window")
    exact((acquired["case"], acquired["leg"]), (case, leg), "acquisition matrix identity")
    exact((completed["success"], completed["workers_srun_returned_zero"]), (True, True), "completed worker execution")
    exact((execution["train_iters"], execution["exit_interval"], execution["cache_budget_steps"], execution["cache_budget_samples"]),
          (25, 10, 25, 1600), "cache budget separate from completed updates")
    exact((execution["warmup_steps"], execution["measured_steps"], execution["completed_steps"]), (5, 5, 10), "execution contract window")
    exact(execution["measured_step_indices"], WINDOW, "execution measured indices")
    exact(execution["capture_step_indices"], list(range(10)), "requested capture window")
    training = contract["training"]
    for name, value in {"warmup_steps": 5, "measured_steps": 5, "completed_steps": 10,
                        "global_batch_size": 64, "micro_batch_size": 1, "gradient_accumulation": 4,
                        "sequence_length": 4096, "log_interval": 1, "learning_rate": 1e-6, "recompute_mode": "none"}.items():
        exact(training[name], value, f"training identity {name}")
    exact(contract["parallelism"], {"tp": 1, "pp": 1, "cp": 1, "ep": 16, "etp": 1, "dense_dp": 16}, "runtime topology contract")
    backend = "uccl_ep" if case in ("uccl", "uccl_ordinary") else "alltoall"
    grouped = case in ("grouped", "uccl")
    exact(contract["communication_backend"], backend, "requested backend")
    exact(contract["routing"], {"topk": 8, "num_groups": 4 if grouped else None, "group_topk": 2 if grouped else None}, "top8 routing mode")
    for name in ("routing_audit", "sample_fingerprints", "audit_received_experts"):
        exact(contract[name], True, f"required recorder {name}")
    exact((proof["valid_reviewed_inputs"], proof["slurm_job_id"], environment["slurm_job_id"]), (True, job, job), "proof/environment current job binding")
    source = environment["ampa_runtime_sources"]
    exact(proof["ampa_runtime_sources"], source, "recorded proof/runtime source identity")
    exact((acquired["runtime_source_sha256"], completed["runtime_source_sha256"]), (source["sha256"], source["sha256"]), "recorded acquisition runtime source")
    exact(proof["checkpoint_payload"]["verified"], True, "historical checkpoint verification")
    verification = proof.get("verification_mode", "verified_in_first_acquisition")
    if verification == "reused_first_case":
        reused = proof["verification_reuse"]
        exact(reused["payload_verified_in_current_allocation"], False, "no false fresh payload claim")
        exact(reused["indexed_payload_verified_in_current_allocation"], False, "no false fresh indexed-data claim")
        exact(reused["native_build_reverified_in_current_allocation"], False, "no false fresh native-build claim")
        reference = reused["first_verified_proof"]
        exact(reference["slurm_job_id"], "3851782", "historical verification job")
        exact(evidence.digest(Path(reference["path"])), reference["sha256"], "historical proof metadata identity")
        exact(proof["checkpoint_payload"]["verified_in_current_allocation"], False, "checkpoint historical verification label")
    else:
        exact(job, "3851782", "only first acquisition performs original verification")
    c0_path, data_path = run / "config/c0_manifest.json", run / "config/dataset_manifest.json"
    c0, data = evidence.json(c0_path), evidence.json(data_path)
    exact(evidence.hashes[str(c0_path)], contract["c0_identity_manifest_sha256"], "copied C0 manifest metadata")
    exact(evidence.hashes[str(data_path)], contract["dataset"]["manifest_sha256"], "copied indexed manifest metadata")
    exact(proof["c0_identity_manifest_sha256"], contract["c0_identity_manifest_sha256"], "proof/contract C0 identity")
    exact(proof["dataset_manifest_sha256"], contract["dataset"]["manifest_sha256"], "proof/contract dataset identity")
    exact((data["validated_workload"]["warmup_steps"], data["validated_workload"]["measured_steps"]), (5, 20), "original dataset cache geometry")
    exact((c0["hf_model"], c0["hf_revision"]), (contract["model"]["hf_model_id"], contract["model"]["hf_revision"]), "checkpoint model revision")
    placement = environment["rank_placement"]
    exact(sorted(row["rank"] for row in placement), list(range(16)), "sixteen actual ranks")
    exact(len({row["physical_gpu_uuid"] for row in placement}), 16, "sixteen unique physical GPUs")
    require(len(set(row["hostname"] for row in placement)) == 4
            and all(v == 4 for v in Counter(row["hostname"] for row in placement).values()), "four GPUs on each of four nodes")
    observation = evidence.json(run / "config/observation_mode.json")
    exact(observation["torch_trace"], leg == "profile", "actual capture recorder setting")
    if backend == "uccl_ep":
        for rank in range(16):
            startup = evidence.json(run / "raw" / f"uccl_backend_rank-{rank}_startup.json")
            cleanup = evidence.json(run / "raw" / f"uccl_backend_rank-{rank}_cleanup.json")
            exact((startup["rank"], startup["job"], startup["valid_identity"]), (rank, job, True), "native identity job/rank")
            exact(startup["binary"]["sha256"], acquired["binary_sha256"], "loaded native binary identity")
            exact(startup["requested_transport_environment"]["FI_MR_CACHE_MONITOR"], "disabled", "same UCCL workaround setting")
            exact((cleanup["rank"], cleanup["valid_cleanup"], cleanup["official_destroy_uccl_returned"]), (rank, True, True), "native cleanup completion")
    common = {"c0": contract["c0_identity_manifest_sha256"], "dataset": contract["dataset"]["manifest_sha256"],
              "runtime": source["sha256"], "model": contract["model"], "data_order": contract["dataset"],
              "checkpoint": contract["initialization"], "native_lm": environment["megatron_lm"],
              "native_bridge": environment["megatron_bridge"]}
    return {"backend": backend, "grouped": grouped, "placement": placement, "common": common,
            "verification_mode": verification, "contract": contract}


def phase_means(evidence, validation_directory, backend):
    """Re-sum the four shared forward ancestors in validated rank-update files."""
    phases = ("router_total", "dispatch_total", "expert_total", "combine_total")
    totals = {phase: {"gpu_ms": 0.0, "cpu_ms": 0.0, "calls": 0} for phase in phases}
    acceptance = evidence.json(Path(validation_directory) / "acceptance.json")
    exact(acceptance["valid_capture"], True, "accepted capture for phase audit")
    accepted_hashes = {row["path"]: row["sha256"] for row in acceptance["input_files"]}
    directory = Path(acceptance.get("profile_output_dir", validation_directory)) / "profiles"
    records = []
    for rank in range(16):
        for step in WINDOW:
            path = (directory / f"rank-{rank}-step-{step:05d}.json" if backend == "uccl_ep"
                    else directory / f"rank-{rank}" / f"step-{step:05d}.json")
            record = evidence.json(path)
            exact(accepted_hashes.get(str(path)), evidence.hashes[str(path)], "phase file bound by accepted capture")
            records.append({"rank": rank, "step": step, "path": str(path), "sha256": evidence.hashes[str(path)]})
            if backend == "uccl_ep":
                exact(record["valid_uccl_trace_capture"], True, "accepted UCCL phase source")
                rows = record["phases"]
            else:
                exact(record["rank"], rank, "native phase-source rank")
                exact(set(record["step_scopes_ms"]), {f"AMPA/STEP/{step}"}, "native phase-source update")
                rows = record["phase_rows"]
            seen = Counter()
            for row in rows:
                name = row["phase"]
                if backend == "uccl_ep":
                    match = re.fullmatch(r"AMPA/FWD/layer_\d{2}/(.+)", name)
                    if match is None:
                        continue
                    name = match.group(1)
                if name not in phases:
                    continue
                gpu = row["gpu_inclusive_activity_sum_ms"] if backend == "uccl_ep" else row["gpu_launched_activity_sum_ms"]
                cpu = row["cpu_scope_sum_ms"]
                require(all(type(v) in (float, int) and math.isfinite(v) and v >= 0 for v in (gpu, cpu)), "invalid inclusive phase time")
                totals[name]["gpu_ms"] += gpu
                totals[name]["cpu_ms"] += cpu
                totals[name]["calls"] += row["calls"]
                seen[name] += row["calls"]
            exact(dict(seen), {phase: 48 * 4 for phase in phases}, "shared forward phase call coverage per rank/update")
    return {"records": records, "phases": {phase: {"gpu_inclusive_s_mean_per_rank_update": values["gpu_ms"] / (80 * 1000),
                    "cpu_scope_s_mean_per_rank_update": values["cpu_ms"] / (80 * 1000),
                    "calls": values["calls"]} for phase, values in totals.items()}}


def compare_report(evidence, report_path, acquisitions_path):
    report = evidence.json(report_path)
    exact((report["schema_version"], report["kind"]), (1, "qwen_debug_5plus5_report"), "report schema")
    exact(report["protocol"], {"warmup_steps": 5, "measured_steps": 5, "completed_steps": 10, "tokens_per_step": TOKENS}, "report protocol")
    exact(report["workload"], {"sequence_length": 4096, "global_batch_size": 64,
          "micro_batch_size": 1, "gradient_accumulation": 4, "nodes": 4, "gpus": 16,
          "expert_parallel": 16, "context_parallel": 1, "precision": "BF16", "top_k": 8}, "report workload")
    exact(report["routing_window"]["step_indices"], WINDOW, "report routing window")
    exact(set(report["cases"]), set(CASES), "report case coverage")
    independently = {}
    common, prefix, placement_sets = [], [], []
    metadata_acceptances, capture_acceptances = [], []
    for case, leg, job, orchestration, run in case_map(evidence, acquisitions_path):
        label = f"{case}-{leg}"
        identity = identities(evidence, case, leg, job, orchestration, run)
        exact(report["cases"][case]["backend"], identity["backend"], f"{label} report backend")
        exact(report["cases"][case]["routing_mode"], "grouped" if identity["grouped"] else "ordinary", f"{label} report routing mode")
        exact((report["cases"][case]["num_groups"], report["cases"][case]["group_topk"]),
              (4, 2) if identity["grouped"] else (None, None), f"{label} actual conditional group configuration")
        timing = timers(evidence, run)
        routing_counts = routing(evidence, run, identity["backend"], identity["grouped"], identity["placement"])
        fingerprint = fingerprints(evidence, run)
        item = report["cases"][case]["legs"][leg]
        exact((item["run_dir"], item["orchestration_dir"], item["job"]), (str(run), str(orchestration), job), f"{label} report case/job")
        for key in ("values_s", "all_values_s"):
            exact(len(item["timing"][key]), len(timing[key]), f"{label} {key} length")
            for actual, expected in zip(item["timing"][key], timing[key]):
                close(actual, expected, f"{label} raw {key}")
        for key in ("mean_s", "median_s", "sd_s", "cv_percent", "min_s", "max_s", "tokens_per_s"):
            close(item["timing"][key], timing[key], f"{label} {key}")
        exact(item["sample_prefix_sha256"], fingerprint, f"{label} fingerprint prefix")
        exact(item["rank_placement"], identity["placement"], f"{label} reported actual placement")
        for key, value in {"completed_logged_optimizer_steps": 10, "global_batch_size": 64,
                           "final_consumed_samples": 640, "final_consumed_global_tokens": 2621440}.items():
            exact(item["health"][key], value, f"{label} training health {key}")
        for raw_name, reported_name in (("lm loss", "lm_loss"), ("grad norm", "grad_norm"), ("load_balancing_loss", "load_balancing_loss")):
            for statistic, reported_statistic in (("min", "min"), ("max", "max"), ("mean", "mean_of_logged_values")):
                close(item["health"]["ranges"][reported_name][reported_statistic], timing["health_ranges"][raw_name][statistic], f"{label} {reported_name} {statistic}")
        accepted, captured = False, False
        if item.get("validation_dir"):
            acceptance_path = Path(item["validation_dir"]) / "acceptance.json"
            if acceptance_path.is_file():
                proof = evidence.json(acceptance_path)
                exact((proof["case"], proof["leg"], proof["training_slurm_job_id"]), (case, leg, job), "posthoc acceptance identity")
                exact(proof["sample_prefix_sha256"], fingerprint, "posthoc accepted input prefix")
                accepted = proof.get("valid_debug_case") is True
                captured = proof.get("valid_capture") is True
        exact(item["validation_status"], "passed" if accepted else "pending", "reported case acceptance state")
        metadata_acceptances.append(accepted)
        if leg == "profile":
            capture_acceptances.append(captured)
        reported_routing = report["cases"][case]["routing"][leg]
        for key in ("total_assignments", "local_assignments", "intra_node_assignments", "inter_node_assignments", "token_rank_deliveries", "token_node_deliveries"):
            exact(reported_routing[key], routing_counts[key], f"{label} {key}")
        for key in ("local_percent", "intra_node_percent", "inter_node_percent"):
            close(reported_routing[key], routing_counts[key], f"{label} {key}")
        if identity["backend"] == "uccl_ep":
            dedup = reported_routing.get("dedup", {})
            for key, value in {"avg_unique_ranks_per_token": routing_counts["token_rank_deliveries"] / routing_counts["routed_tokens"],
                               "avg_unique_nodes_per_token": routing_counts["token_node_deliveries"] / routing_counts["routed_tokens"],
                               "avg_remote_nodes_per_token": routing_counts["remote_token_node_deliveries"] / routing_counts["routed_tokens"]}.items():
                if key in dedup:
                    close(dedup[key], value, f"{label} {key}")
        common.append(identity["common"])
        prefix.append(fingerprint)
        placement_sets.append(tuple((row["rank"], row["physical_gpu_uuid"]) for row in identity["placement"]))
        independently[label] = {"job": job, "run_dir": str(run), "timing": timing, "routing": routing_counts,
                                "sample_prefix_sha256": fingerprint, "input_verification_mode": identity["verification_mode"]}
        print(json.dumps({"independently_audited": label, "mean_s": timing["mean_s"]}), flush=True)
    require(all(item == common[0] for item in common), "checkpoint/data/order/native sources differ across acquisitions")
    exact(len(set(prefix)), 1, "same actual consumed batches across all eight acquisitions")
    exact(report["sample_prefix_sha256"], prefix[0], "global consumed-prefix identity")
    exact(report["model"], common[0]["model"], "report model identity")
    for key, expected in (("runtime_source_sha256", common[0]["runtime"]), ("c0_identity_manifest_sha256", common[0]["c0"]), ("dataset_manifest_sha256", common[0]["dataset"])):
        exact(report[key], expected, key)
    contrasts = report["contrasts"]
    exact([(c["from"], c["to"]) for c in contrasts], list(CONTRASTS), "four OFF-only contrast orientations")
    for contrast in contrasts:
        a = independently[contrast["from"] + "-control"]["timing"]["mean_s"]
        b = independently[contrast["to"] + "-control"]["timing"]["mean_s"]
        exact(contrast["leg"], "control", "headline contrast uses profiler OFF")
        close(contrast["speedup"], a / b, "OFF timing ratio")
        close(contrast["duration_change_percent"], 100 * (b / a - 1), "OFF duration change sign/denominator")
        exact(contrast["causal_effect_claimed"], False, "no causal cross-allocation claim")
    for case in CASES:
        identical = independently[case + "-control"]["routing"]["route_fingerprint_sha256"] == independently[case + "-profile"]["routing"]["route_fingerprint_sha256"]
        exact(report["cases"][case]["on_off_route_histograms_identical"], identical, "ON/OFF route trajectory claim")
    acceptance = report["acceptance"]
    exact(acceptance["all_case_metadata_valid"], all(metadata_acceptances), "all eight metadata acceptances")
    exact(acceptance["all_capture_valid"], all(capture_acceptances), "all four capture acceptances")
    exact(acceptance["rank_update_captures"], 640 if all(capture_acceptances) else None, "accepted rank-update capture count")
    if "valid_descriptive_comparison" in acceptance:
        exact(acceptance["valid_descriptive_comparison"], all(metadata_acceptances) and all(capture_acceptances), "final descriptive acceptance conjunction")
    if not report.get("diagnostic", False):
        require(all(metadata_acceptances) and all(capture_acceptances), "final report lacks full metadata/capture acceptance")
    exact(acceptance["same_physical_placement_claimed"], False, "no matched-placement assumption")
    exact(acceptance["pure_profiler_overhead_available"], False, "no isolated-overhead claim")
    exact((acceptance["identical_consumed_prefix"], acceptance["same_checkpoint_dataset_runtime"]), (True, True), "report common identity claims")
    require(isinstance(report["claim_limits"], list) and len(report["claim_limits"]) >= 8, "missing report qualification record")
    reconstructed_phases = {}
    if "phase_summaries" in report:
        exact(set(report["phase_summaries"]), set(CASES), "phase-summary case coverage")
        for case in CASES:
            case_report = report["cases"][case]
            phases = phase_means(evidence, case_report["legs"]["profile"]["validation_dir"], case_report["backend"])
            stated = report["phase_summaries"][case]
            exact(stated["population"], 80, "phase denominator in rank-updates")
            exact(stated["window"], "measured5", "phase measured window label")
            exact(stated["human_updates"], [6, 7, 8, 9, 10], "phase measured updates")
            exact(stated["ranks"], list(range(16)), "phase rank coverage")
            exact(stated["records"], phases["records"], "phase-source file/rank/step/hash binding")
            exact(set(stated["phases"]), set(phases["phases"]), "four common inclusive phases")
            for name, values in phases["phases"].items():
                for field in ("gpu_inclusive_s_mean_per_rank_update", "cpu_scope_s_mean_per_rank_update"):
                    close(stated["phases"][name][field], values[field], f"{case} {name} {field}")
            reconstructed_phases[case] = phases
    for item in report["input_files"]:
        path = Path(item["path"])
        require(not (ROOT / "checkpoint") in path.parents and not (ROOT / "dataset") in path.parents,
                "independent audit does not repeat upstream payload verification")
        actual = evidence.hashes.get(str(path)) or evidence.digest(path)
        exact(item["sha256"], actual, f"report evidence identity {path}")
    return {"valid_independent_audit": True, "report_sha256": evidence.hashes[str(report_path)],
            "report_path": str(report_path), "audit_source_sha256": evidence.digest(Path(__file__).resolve()),
            "acquisitions_path": str(acquisitions_path), "audit_slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "scope": "Independent raw timer, health, ownership, conservation, receive, fingerprint and recorded-identity reconstruction. Capture completeness remains a separate acceptance gate.",
            "cases": independently, "contrasts": contrasts, "all_eight_consumed_prefixes_equal": True,
            "distinct_observed_placements": len(set(placement_sets)), "same_physical_placement_assumed": False,
            "upstream_payload_reverified": False, "report_diagnostic": report.get("diagnostic"),
            "report_capture_acceptance": acceptance.get("all_capture_valid"),
            "phase_summaries": reconstructed_phases,
            "input_files": [{"path": path, "sha256": digest} for path, digest in sorted(evidence.hashes.items())]}


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--acquisitions", type=Path, default=DEFAULT_ACQUISITIONS)
    parser.add_argument("--allow-login-cpu", action="store_true",
                        help="Explicit D-028 authorization for lightweight CPU analysis in .runtime/venvs/qwen-posthoc-cpu.")
    args = parser.parse_args()
    execution = execution_context(args.allow_login_cpu)
    require(args.output.is_absolute() and not args.output.exists(), "new absolute audit output required")
    evidence = Evidence()
    try:
        result = compare_report(evidence, args.report.resolve(), args.acquisitions.resolve())
        result["execution"] = execution
        result["audit_slurm_job_id"] = execution["slurm_job_id"]
    except Exception as error:
        result = {"valid_independent_audit": False, "error": f"{type(error).__name__}: {error}",
                  "audit_slurm_job_id": execution["slurm_job_id"], "execution": execution,
                  "input_files": [{"path": path, "sha256": digest} for path, digest in sorted(evidence.hashes.items())]}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        raise
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"valid_independent_audit": True, "report_sha256": result["report_sha256"], "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
