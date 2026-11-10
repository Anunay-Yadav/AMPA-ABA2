#!/usr/bin/env python3
"""Accept the four-case pilot only from complete, unchanged validation evidence.

Consumes reviewed validator outputs; compressed traces are hashed, never parsed.
Small training logs are checked with the existing health helper because its
current smoke record does not contain log hashes.
"""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path

import yaml

from analysis.compare_deep_trace_overhead import _checked_fingerprint_batches
from analysis.summarize_training_evidence import summarize
from analysis.validate_ep16_received_routes import validate_run as validate_received_run
from profiling.ampa_profile.provenance import (
    _EP16_RUNTIME_TOPOLOGIES, _ampa_runtime_sources, validate_rank_placement, validate_runtime_groups,
)
from tools.uccl_ep_log_proof import inspect_node
from tools.uccl_ep_source_identity import verify_uccl_source


CASES = ("ordinary", "grouped", "uccl_ordinary", "uccl")
UCCL_CASES = ("uccl_ordinary", "uccl")
GROUPED_CASES = ("grouped", "uccl")
TRANSPORT_ENVIRONMENT_KEYS = (
    "FI_MR_CACHE_MONITOR", "FI_MR_CUDA_CACHE_MONITOR_ENABLED",
    "FI_MR_CACHE_MAX_COUNT", "FI_MR_CACHE_MAX_SIZE", "FI_LOG_LEVEL", "FI_LOG_SUBSYS",
)
UCCL_TRANSPORT_POLICY = {
    "FI_MR_CACHE_MONITOR": "disabled",
    "scope": "control_and_profile_workers_before_native_initialization",
    "claim_limit": "Disables the system monitor; registration overhead can change.",
}
ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_SOURCES = (
    "analysis/accept_uccl_pilot.py", "analysis/inspect_uccl_trace.py", "analysis/validate_ep16_received_routes.py",
    "analysis/analyze_all_step_trace.py", "analysis/validate_torch_trace.py", "analysis/aggregate_deep_trace.py",
    "analysis/check_ep16_smoke.py", "analysis/summarize_training_evidence.py", "analysis/extract_step_timing.py",
    "analysis/compare_deep_trace_overhead.py", "analysis/compare_profile_overhead.py", "tools/uccl_ep_log_proof.py",
    "tools/uccl_ep_source_identity.py",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def equal(actual, expected, label):
    # Canonical JSON keeps bool/int and int/float differences visible at every
    # nesting level; Python container equality otherwise aliases those values.
    require(json.dumps(actual, sort_keys=True, allow_nan=False) == json.dumps(expected, sort_keys=True, allow_nan=False),
            f"{label} differs")


def equal_resolved_configuration(actual, expected, label="cross-case resolved training"):
    """Allow only the documented positive-infinity optimizer skip default."""
    require(type(actual) is dict and type(expected) is dict, f"{label}: expected configuration mappings")

    def visit(left, right, path):
        location = ".".join(str(part) for part in path) or "<root>"
        require(type(left) is type(right), f"{label}: type differs at {location}")
        if type(left) is dict:
            require(all(type(key) is str for key in left) and all(type(key) is str for key in right),
                    f"{label}: non-string mapping key at {location}")
            require(sorted(left) == sorted(right), f"{label}: keys differ at {location}")
            for key in sorted(left):
                visit(left[key], right[key], (*path, key))
        elif type(left) is list:
            require(len(left) == len(right), f"{label}: list length differs at {location}")
            for index, (value, reference) in enumerate(zip(left, right)):
                visit(value, reference, (*path, index))
        elif type(left) is float:
            if not math.isfinite(left) or not math.isfinite(right):
                require(path == ("optimizer", "grad_norm_skip_threshold") and left == math.inf and right == math.inf,
                        f"{label}: nonfinite or unequal value at {location}")
            else:
                require(left.hex() == right.hex(), f"{label}: float differs at {location}")
        else:
            require(type(left) in (type(None), bool, int, str), f"{label}: unsupported value at {location}")
            require(left == right, f"{label}: value differs at {location}")

    visit(actual, expected, ())


class Inputs:
    """Bind files once and recheck every consumed byte before writing proof."""
    def __init__(self):
        self.hashes = {}

    def bind(self, path, expected=None):
        path = Path(path)
        require(path.is_file() and not path.is_symlink(), f"expected regular nonsymlink input: {path}")
        key = str(path.resolve())
        if key not in self.hashes:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            self.hashes[key] = digest.hexdigest()
        if expected is not None:
            equal(self.hashes[key], expected, f"input hash {path}")
        return self.hashes[key]

    def read(self, path):
        self.bind(path)
        payload = Path(path).read_bytes()
        equal(hashlib.sha256(payload).hexdigest(), self.hashes[str(Path(path).resolve())], "JSON read binding")
        return json.loads(payload)

    def recheck(self):
        fresh = Inputs()
        for path, digest in self.hashes.items():
            fresh.bind(path, digest)


def _path(value, expected, label):
    require(isinstance(value, str) and Path(value).is_absolute(), f"{label} must be an absolute path")
    equal(str(Path(value).resolve()), str(Path(expected).resolve()), label)


def _contract(contract, case):
    require(case in CASES, f"unknown pilot case {case}")
    result = copy.deepcopy(contract)
    equal(result["parallelism"], {"tp": 1, "pp": 1, "cp": 1, "ep": 16, "etp": 1, "dense_dp": 16}, "pilot topology")
    equal(result["model"]["precision"], "bf16", "pilot precision")
    for name, value in {"sequence_length": 4096, "micro_batch_size": 1, "global_batch_size": 64,
        "gradient_accumulation": 4, "warmup_steps": 5, "measured_steps": 20, "completed_steps": 2,
        "log_interval": 1, "recompute_mode": "none"}.items():
        equal(result["training"][name], value, f"pilot training {name}")
    for name in ("sample_fingerprints", "routing_audit", "audit_received_experts"):
        equal(result[name], True, f"pilot {name}")
    equal(result.pop("communication_backend"), "uccl_ep" if case in UCCL_CASES else "alltoall", "case backend")
    equal(result["routing"].pop("num_groups"), 4 if case in GROUPED_CASES else None, "case groups")
    equal(result["routing"].pop("group_topk"), 2 if case in GROUPED_CASES else None, "case selected groups")
    equal(result["baseline"].pop("dispatcher"), "flex" if case in UCCL_CASES else "alltoall", "case dispatcher")
    if case in UCCL_CASES:
        result.pop("uccl_build_dir")
    else:
        require("uccl_build_dir" not in result, "native case contains UCCL build selection")
    return result


def _resolved(configuration, case, run):
    require(case in CASES, f"unknown pilot case {case}")
    result = copy.deepcopy(configuration)
    model = result["model"]
    expected = {"moe_router_num_groups": 4 if case in GROUPED_CASES else None,
                "moe_router_group_topk": 2 if case in GROUPED_CASES else None,
                "moe_token_dispatcher_type": "flex" if case in UCCL_CASES else "alltoall",
                "moe_flex_dispatcher_backend": "deepep" if case in UCCL_CASES else None}
    for name, value in expected.items():
        equal(model.pop(name), value, f"resolved {name}")
    for name, value in {"tensor_model_parallel_size": 1, "pipeline_model_parallel_size": 1,
                        "context_parallel_size": 1, "expert_model_parallel_size": 16,
                        "expert_tensor_parallel_size": 1, "seq_length": 4096, "bf16": True,
                        "fp16": False, "moe_router_dtype": None, "moe_enable_deepep": False}.items():
        equal(model[name], value, f"resolved {name}")
    for name, value in {"train_iters": 25, "exit_interval": 2, "global_batch_size": 64, "micro_batch_size": 1}.items():
        equal(result["train"][name], value, f"resolved train.{name}")
    for name in ("use_pytorch_profiler", "use_nsys_profiler"):
        equal(result["profiling"][name], False, f"resolved Bridge profiler {name}")
    for name, value in {"tensorboard_dir": str(run / "logs/tensorboard"), "wandb_exp_name": run.name,
                        "wandb_save_dir": str(run / "logs/wandb")}.items():
        equal(result["logger"].pop(name), value, f"resolved run logger {name}")
    return result


def _bound_map(evidence, expected_paths, inputs, label):
    require(isinstance(evidence, dict) and set(evidence) == {str(path.resolve()) for path in expected_paths},
            f"{label} input paths differ")
    for path in expected_paths:
        inputs.bind(path, evidence[str(path.resolve())])


def _trace_paths(run, rank, step):
    return (run / "traces/torch_profile" / f"step-{step:05d}" / f"rank-{rank}.json.gz",
            run / "raw/deep_trace" / f"step-{step:05d}" / f"rank-{rank}.json")


def _trace_evidence(case, run, analysis, root, inputs):
    native_sources = {name: inputs.bind(root / "analysis" / name) for name in (
        "analyze_all_step_trace.py", "validate_torch_trace.py", "aggregate_deep_trace.py")}
    for rank in range(16):
        summary = None
        if case not in UCCL_CASES:
            summary = inputs.read(run / "analysis/all_steps" / f"rank-{rank}" / "summary.json")
            for key, value in {"valid_capture": True, "rank": rank, "steps": [0, 1], "world_size": 16,
                               "cp": 1, "gradient_accumulation": 4}.items():
                equal(summary.get(key), value, f"native trace {key}")
            equal(summary["analysis_source_sha256"], native_sources, "native trace analysis sources")
            audit = run / "raw" / f"routing_audit_rank-{rank}.json"
            _path(summary["routing_audit_path"], audit, "native trace audit")
            inputs.bind(audit, summary["routing_audit_sha256"])
            require(len(summary["validation"]) == len(summary["step_summaries"]) == 2, "native trace step coverage differs")
        for step in (0, 1):
            trace, call = _trace_paths(run, rank, step)
            if case in UCCL_CASES:
                record = inputs.read(analysis / case / f"rank-{rank}-step-{step:05d}.json")
                for key, value in {"valid_uccl_trace_capture": True, "synthetic": False,
                                   "pair_count": 384, "paired_direction_count": 768}.items():
                    equal(record.get(key), value, f"UCCL trace {key}")
                equal(record["analysis_source_sha256"], inputs.bind(root / "analysis/inspect_uccl_trace.py"), "UCCL trace analysis source")
                _bound_map(record["inputs"], (trace, call), inputs, "UCCL trace")
            else:
                record = summary["validation"][step]
                equal(record["rank"], rank, "native validation rank")
                equal(record["steps"], [step], "native validation step")
                for path, prefix in ((trace, "trace"), (call, "call_record")):
                    _path(record[f"{prefix}_path"], path, f"native {prefix}")
                    inputs.bind(path, record[f"{prefix}_sha256"])
                aggregate = summary["step_summaries"][step]
                equal(aggregate["step"], step, "native aggregation step")
                aggregate_path = run / "analysis/all_steps" / f"rank-{rank}" / f"step-{step:05d}.json"
                _path(aggregate["path"], aggregate_path, "native aggregation")
                inputs.bind(aggregate_path, aggregate["sha256"])
            call_record = inputs.read(call)
            equal(call_record["rank"], rank, "actual trace call rank")
            equal(call_record["schema_version"], 2 if case in UCCL_CASES else 1, "actual trace call schema")
            require(len(call_record["steps"]) == 1, "actual trace call step coverage differs")
            equal(call_record["steps"][0]["step"], step, "actual trace call step")
            equal(call_record["steps"][0]["skipped_iter"], False, "actual trace skipped update")


def _manifest(path, inputs):
    inputs.bind(path)
    lines = Path(path).read_text().splitlines()
    require(len(lines) == 1, "expected one library in build manifest")
    digest, name = lines[0].split(maxsplit=1)
    library = Path(name.lstrip(" *"))
    require(library.is_absolute(), "build library path must be absolute")
    inputs.bind(library.resolve(), digest)
    return {"path": str(library.resolve()), "sha256": digest}


def _uccl_identity(run, build, source_identity, placement, job, root, inputs):
    binary = _manifest(build / "binary.sha256", inputs)
    fabric = _manifest(build / "libfabric_runtime.sha256", inputs)
    require(Path(binary["path"]).parent == build / "site/uccl", "UCCL binary is outside isolated build")
    reference_transport = None
    for rank, slot in enumerate(placement):
        start = inputs.read(run / "raw" / f"uccl_backend_rank-{rank}_startup.json")
        for key, value in {"valid_identity": True, "rank": rank, "hostname": slot["hostname"], "job": job,
            "backend": "uccl_ep", "source_revision": source_identity["base_revision"], "source_identity": source_identity,
            "binary": binary, "libfabric": fabric, "mode": "normal_high_throughput", "requested_transport": "cxi",
            "low_latency_dedup_api_exercised": False, "build_dir": str(build)}.items():
            equal(start.get(key), value, f"UCCL startup {key}")
        transport = start.get("requested_transport_environment")
        require(type(transport) is dict and set(transport) == set(TRANSPORT_ENVIRONMENT_KEYS)
                and all(value is None or type(value) is str for value in transport.values()),
                "UCCL requested transport environment is missing or mistyped")
        equal(transport["FI_MR_CACHE_MONITOR"], "disabled", "UCCL system monitor workaround")
        if reference_transport is None:
            reference_transport = transport
        equal(transport, reference_transport, "UCCL cross-rank transport environment")
        equal(start["gpu_ordinals"], {"local_rank": slot["local_rank"], "cuda_current_device": slot["cuda_current_device"],
            "uccl_physical_gpu_rank": slot["local_rank"]}, "UCCL physical ordinal")
        adapter = root / "Megatron-LM/megatron/core/transformer/moe/fused_a2a.py"
        _path(start["native_adapter"]["path"], adapter, "UCCL native adapter")
        inputs.bind(adapter, start["native_adapter"]["sha256"])
        require(set(start["wrappers"]) == {"deep_ep", "deep_ep.buffer", "deep_ep.utils"}, "official wrapper coverage differs")
        for name, record in start["wrappers"].items():
            leaf = "__init__.py" if name == "deep_ep" else name.rsplit(".", 1)[1] + ".py"
            expected = (build / "uccl/ep/deep_ep_wrapper/deep_ep" / leaf).resolve()
            require(build / "uccl" in expected.parents, "official wrapper target escapes verified source")
            _path(record["path"], expected, "imported official wrapper target")
            equal(inputs.bind(record["path"], record["sha256"]), inputs.bind(expected), "official wrapper bytes")
        runtime = inputs.read(run / "raw" / f"uccl_backend_rank-{rank}_runtime.json")
        equal(runtime.get("requested_transport_environment"), transport, "UCCL startup/runtime transport environment")
        for key, value in {"valid_identity": True, "rank": rank, "first_completed_step": 0, "low_latency_mode": False,
                           "scratch_is_cuda": True, "scratch_device": f"cuda:{slot['cuda_current_device']}",
                           "buffer_type": "deep_ep.buffer.Buffer"}.items():
            equal(runtime.get(key), value, f"UCCL runtime {key}")
        for key in ("num_nvl_bytes", "num_rdma_bytes", "num_sms"):
            require(type(runtime.get(key)) is int and runtime[key] > 0, f"invalid UCCL runtime {key}")
        cleanup = inputs.read(run / "raw" / f"uccl_backend_rank-{rank}_cleanup.json")
        for key in ("valid_cleanup", "official_destroy_uccl_returned", "native_runtime_released_before_scratch",
                    "peer_closure_barrier_completed", "process_groups_still_alive",
                    "allocator_drained_before_native_stream_release"):
            equal(cleanup.get(key), True, f"UCCL cleanup {key}")
        equal(cleanup.get("transient_metadata_layers"), 48, "UCCL cleanup layer coverage")
        equal(cleanup.get("transient_attributes_cleared"), 384, "UCCL cleanup attribute coverage")
        equal(cleanup.get("known_transient_fields"), [
            "token_indices", "token_probs", "dispatched_indices", "dispatched_probs",
            "dispatched_routing_map", "reversed_mapping_for_combine", "pad_offsets", "tokens_per_expert",
        ], "UCCL cleanup transient fields")
        equal(cleanup["rank"], rank, "UCCL cleanup rank")
    seen = set()
    for node in range(4):
        proof = inputs.read(run / "raw" / f"native-cxi-node-{node}.json")
        equal(proof, inspect_node(run / "logs" / f"torchrun-node-{node}", node), "recomputed native CXI proof")
        equal(proof.get("valid_native_cxi_log_binding"), True, "native CXI log proof")
        require(len(proof["workers"]) == 4, "native CXI worker coverage differs")
        for local, worker in enumerate(proof["workers"]):
            rank = worker["rank"]
            require(type(rank) is int and rank in range(16) and rank not in seen, "duplicate or wrong native CXI rank")
            seen.add(rank)
            slot = placement[rank]
            equal(worker["node_rank"], node, "native CXI node")
            equal(slot["slurm_node_id"], str(node), "native CXI actual node placement")
            equal(worker["local_rank"], local, "native CXI worker slot")
            equal(slot["local_rank"], local, "native CXI actual worker slot")
            equal(worker["native_cxi_domain"], f"cxi{local}", "native CXI domain")
            require(all(type(worker.get(key)) is int and worker[key] >= 1 for key in ("domain_messages", "endpoint_messages")),
                    "native CXI initialization evidence is incomplete")
            parts = []
            require(set(worker["logs"]) == {"stdout", "stderr"}, "native CXI log stream coverage differs")
            for stream in ("stdout", "stderr"):
                record = worker["logs"][stream]
                path = Path(record["path"])
                expected_root = run / "logs" / f"torchrun-node-{node}"
                require(path.is_absolute() and expected_root in path.parents
                        and path.parts[-3:] == ("attempt_0", str(local), f"{stream}.log"), "native CXI log path differs")
                inputs.bind(path, record["sha256"])
                parts.append(path.read_bytes())
            equal(hashlib.sha256(b"".join(parts)).hexdigest(), inputs.bind(run / "logs" / f"rank-{rank}.out"), "flattened worker log")
    return binary["sha256"]


def accept_pilot(orchestration_dir, analysis_root, *, root=ROOT):
    root, orchestration, analysis = Path(root).resolve(), Path(orchestration_dir).resolve(), Path(analysis_root).resolve()
    inputs = Inputs()
    inputs.bind(orchestration / "cases.tsv")
    rows = [line.split("\t") for line in (orchestration / "cases.tsv").read_text().splitlines()]
    require(len(rows) == len(CASES) and all(len(row) == 3 for row in rows), "pilot must contain exactly four case rows")
    runs = {}
    for case, row in zip(CASES, rows):
        equal(row[:2], [case, "profile"], "pilot case/profile order")
        run = Path(row[2])
        require(run.is_absolute() and run.is_dir() and not run.is_symlink() and run.parent == root / "runs", "pilot run root differs")
        runs[case] = run.resolve()
    require(len(set(runs.values())) == len(CASES), "pilot case roots must be distinct")
    sources = _ampa_runtime_sources(root)
    for relative in ANALYSIS_SOURCES + ("jobs/training/uccl_ep_comparison.sbatch", "jobs/training/qwen3_uccl_ep_worker.sh"):
        require(relative in sources["files"], f"material source missing from runtime inventory: {relative}")
    for relative, digest in sources["files"].items():
        inputs.bind(root / relative, digest)
    build = Path(inputs.read(runs["uccl"] / "config/comparison_contract.json")["uccl_build_dir"])
    require(build.is_absolute() and build.is_dir() and not build.is_symlink(), "UCCL build root differs")
    for case in UCCL_CASES:
        equal(inputs.read(runs[case] / "config/comparison_contract.json")["uccl_build_dir"], str(build),
              "cross-case UCCL build")
    source_identity = verify_uccl_source(build)
    equal(inputs.read(build / "source_identity.json"), source_identity, "approved source manifest")
    inputs.bind(build / "source/uccl-borrowed-scratch.patch", source_identity["patch_sha256"])
    for relative, digest in source_identity["modified_files"].items():
        inputs.bind(build / "uccl" / relative, digest)
    acquisition = inputs.read(orchestration / "acquisition.json")
    for key, value in {"mode": "pilot", "topology": "cp1_ep16", "launch_policy": "torchrun_node4",
                       "runtime_source_sha256": sources["sha256"], "accepted_pilot_proof": None}.items():
        equal(acquisition.get(key), value, f"pilot acquisition {key}")
    equal(acquisition.get("uccl_source_identity"), source_identity, "acquisition approved source identity")
    equal(acquisition.get("uccl_transport_policy"), UCCL_TRANSPORT_POLICY, "acquisition UCCL transport policy")
    job = acquisition.get("job")
    require(type(job) is str and job, "pilot allocation identity is missing")
    completed = inputs.read(orchestration / "acquisition_completed.json")
    for key, value in {"success": True, "workers_srun_returned_zero": True, "mode": "pilot", "job": job,
        "runtime_source_sha256": sources["sha256"], "binary_sha256": acquisition["binary_sha256"],
        "cases_tsv_sha256": inputs.bind(orchestration / "cases.tsv")}.items():
        equal(completed.get(key), value, f"acquisition completion {key}")
    reference_contract = reference_resolved = reference_environment = reference_placement = None
    shared_manifests, fingerprints, initial_routes = {}, {}, {}
    placement_by_case = {}
    for case, run in runs.items():
        contract = inputs.read(run / "config/comparison_contract.json")
        equal(contract["training"]["completed_steps"], 2, "pilot completed updates")
        equal(contract["launch_policy"], "torchrun_node4", "pilot launch policy")
        equal(contract["audit_received_experts"], True, "pilot receive auditing")
        normalized_contract = _contract(contract, case)
        resolved_path = run / "config/resolved.yaml"
        inputs.bind(resolved_path)
        normalized_resolved = _resolved(yaml.safe_load(resolved_path.read_text()), case, run)
        environment = inputs.read(run / "environment.json")
        equal(environment["slurm_job_id"], job, "pilot allocation")
        equal(environment["launch_policy"], "torchrun_node4", "pilot environment launch policy")
        equal(environment["ampa_runtime_sources"], sources, "current pilot runtime sources")
        equal(environment["runtime_topology"], _EP16_RUNTIME_TOPOLOGIES["cp1_ep16"], "actual pilot runtime topology")
        placement = validate_rank_placement(environment["rank_placement"], expected_world_size=16, launch_policy="torchrun_node4")
        validate_runtime_groups(placement, expected_world_size=16, cp=1)
        placement_by_case[case] = placement
        common_environment = {name: environment[name] for name in ("pytorch_version", "cuda_version", "nccl_version",
            "cuda_allocator_conf", "container_sqsh_sha256", "megatron_lm", "megatron_bridge", "checkpoint",
            "dataset_blend_per_split", "parallelism", "runtime_topology", "activation_recomputation",
            "hf_model_id", "hf_model_revision", "dataset_consumer", "reviewed_inputs_proof")}
        if reference_contract is None:
            reference_contract, reference_resolved, reference_environment, reference_placement = (
                normalized_contract, normalized_resolved, common_environment, placement)
        equal(normalized_contract, reference_contract, "cross-case training contract")
        equal_resolved_configuration(normalized_resolved, reference_resolved, "cross-case resolved training")
        equal(common_environment, reference_environment, "cross-case native/runtime/input environment")
        equal(placement, reference_placement, "cross-case physical worker placement")
        for name in ("c0_manifest.json", "dataset_manifest.json", "model_manifest.json"):
            digest = inputs.bind(run / "config" / name)
            shared_manifests.setdefault(name, digest)
            equal(digest, shared_manifests[name], f"cross-case {name}")
        equal(inputs.bind(run / "config/c0_manifest.json"), contract["c0_identity_manifest_sha256"], "actual C0 contract binding")
        equal(inputs.bind(run / "config/dataset_manifest.json"), contract["dataset"]["manifest_sha256"], "actual dataset contract binding")
        mode = inputs.read(run / "config/observation_mode.json")
        for key, value in {"data_mode": "indexed", "lightweight_profiling": "off", "nsys_trace": False,
                           "torch_trace": True, "sample_fingerprints": True, "router_observer_attached": False,
                           "route_wrapper_installed": False, "dispatcher_preprocess_wrapper_installed": False}.items():
            equal(mode.get(key), value, f"pilot observation {key}")
        capture = inputs.read(run / "config/all_step_capture.json")
        for key, value in {"enabled": True, "steps": [0, 1], "ranks": list(range(16))}.items():
            equal(capture.get(key), value, f"pilot capture {key}")
        for rank in range(16):
            inputs.bind(run / "logs" / f"rank-{rank}.out")
            path = run / "raw" / f"sample_fingerprint_rank-{rank}.json"
            batches = _checked_fingerprint_batches(inputs.read(path), rank=rank, accumulation=4, steps=2,
                                                  token_shape=[1, 4096], path=path)
            fingerprints.setdefault(rank, batches)
            equal(batches, fingerprints[rank], "actual consumed CPU batch identity")
        health = inputs.read(run / "analysis/smoke_health.json")
        for key, value in {"valid": True, "cp": 1, "ranks": 16, "steps": 2}.items():
            equal(health.get(key), value, f"pilot health {key}")
        equal(health["health"], summarize(run, steps=2, interval=1), "actual authoritative health log")
        received = inputs.read(analysis / case / "received_routes.json")
        # Reuse the cheap integer/JSON validator. Matching hashes and a valid
        # flag alone cannot prove that the actual assignment records reconcile.
        equal(received, validate_received_run(run, 2), "recomputed receive proof")
        expected_paths = [run / "config/comparison_contract.json", run / "config/routing_ownership.json", run / "environment.json"] + [
            run / "raw" / f"routing_audit_rank-{rank}.json" for rank in range(16)]
        require(len(received["input_files"]) == 19, "receive proof input coverage differs")
        _bound_map({row["path"]: row["sha256"] for row in received["input_files"]}, expected_paths, inputs, "receive proof")
        _trace_evidence(case, run, analysis, root, inputs)
        initial_routes[case] = [inputs.read(run / "raw" / f"routing_audit_rank-{rank}.json")["steps"][0]["layers"][0]["expert_counts"]
                                for rank in range(16)]
    equal(initial_routes["grouped"], initial_routes["uccl"], "initial grouped/UCCL layer0 route histograms")
    equal(initial_routes["ordinary"], initial_routes["uccl_ordinary"], "initial ordinary/UCCL layer0 route histograms")
    proof_path = orchestration / "reviewed_inputs.json"
    reviewed = inputs.read(proof_path)
    equal(reviewed["valid_reviewed_inputs"], True, "immutable input proof")
    equal(reviewed["slurm_job_id"], job, "immutable input allocation")
    equal(reviewed["ampa_runtime_sources"], sources, "immutable input source proof")
    equal(reviewed["checkpoint_payload"]["verified"], True, "immutable checkpoint payload proof")
    equal(reviewed["native_checkouts"], {"megatron_lm": reference_environment["megatron_lm"],
        "megatron_bridge": reference_environment["megatron_bridge"]}, "immutable native checkout proof")
    canonical_c0 = root / "checkpoint/megatron/qwen3-30b-a3b-base-c0"
    equal(reference_contract["initialization"], {"mode": "pretrained", "checkpoint": str(canonical_c0)}, "canonical pilot C0")
    equal(reference_environment["checkpoint"], str(canonical_c0), "actual pilot C0")
    _path(reviewed["checkpoint_payload"]["path"], canonical_c0, "immutable checkpoint payload path")
    equal(inputs.bind(canonical_c0 / "ampa_c0_manifest.json"), shared_manifests["c0_manifest.json"], "canonical C0 manifest")
    equal(reviewed["c0_identity_manifest_sha256"], shared_manifests["c0_manifest.json"], "immutable C0 manifest binding")
    equal(reviewed["dataset_manifest_sha256"], shared_manifests["dataset_manifest.json"], "immutable dataset binding")
    equal(reference_environment["reviewed_inputs_proof"], {"path": str(proof_path), "sha256": inputs.bind(proof_path)}, "runtime immutable input binding")
    binary, reference_transport = None, None
    for case in UCCL_CASES:
        binary = _uccl_identity(runs[case], build, source_identity, placement_by_case[case], job, root, inputs)
        equal(binary, acquisition["binary_sha256"], "acquisition actual UCCL binary")
        transport = inputs.read(runs[case] / "raw/uccl_backend_rank-0_startup.json")["requested_transport_environment"]
        if reference_transport is None:
            reference_transport = transport
        equal(transport, reference_transport, "cross-case UCCL transport environment")
    inputs.recheck()
    return {"valid_pilot": True, "cases": list(CASES), "completed_updates": 2, "topology": "cp1_ep16",
        "valid_training_health": True, "valid_send_receive_audit": True, "valid_uccl_trace_capture": True,
        "binary_sha256": binary, "runtime_source_sha256": sources["sha256"], "training_slurm_job_id": job,
        "uccl_source_identity": source_identity,
        "case_runs": {case: str(run) for case, run in runs.items()}, "matched_cpu_batches_per_case": 128,
        "initial_grouped_uccl_layer0_histograms_equal": True,
        "initial_ordinary_uccl_layer0_histograms_equal": True,
        "uccl_transport_policy": UCCL_TRANSPORT_POLICY,
        "all_analysis_source_sha256": {name: inputs.hashes[str((root / name).resolve())] for name in ANALYSIS_SOURCES},
        "input_files": [{"path": path, "sha256": digest} for path, digest in sorted(inputs.hashes.items())],
        "claim_limit": "Two healthy updates establish only this tested native workload. Later routes/losses and quality equivalence are not required or claimed. Logical deliveries are not wire bytes. Trace acceptance covers recorded correlated activity, not unrecorded proxy/persistent activity, exposed latency or causal speedup."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orchestration-dir", type=Path, required=True)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(os.environ.get("SLURM_JOB_ID") and os.environ.get("SLURM_JOB_PARTITION") == "preemptable", "allocated preemptable compute is required")
    require(not args.output.exists() and not args.output.is_symlink(), "output must be new and unique")
    result = accept_pilot(args.orchestration_dir, args.analysis_root)
    output = args.output.resolve()
    require(all(output != Path(run) and Path(run) not in output.parents for run in result["case_runs"].values()),
            "proof output must be outside every case run")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"valid_pilot": True, "cases": result["cases"], "binary_sha256": result["binary_sha256"]}), flush=True)


if __name__ == "__main__":
    main()
