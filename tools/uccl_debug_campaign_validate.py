#!/usr/bin/env python3
"""Validate the completed eight-case5+5 debug campaign on allocated compute.

Historical input verification remains explicitly historical for reused cases.
All actual training, receive-accounting, identity, and trace checks are reused
from the strict individual-case validator. Source runs are never rewritten.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

import yaml

from tools import uccl_ep_debug_validate as single
from tools.uccl_ep_log_proof import inspect_node


ROOT = single.ROOT
FIRST = ROOT / "runs/uccl-debug5x5-20261104T1415Z-grouped-profile"
require, equal = single.require, single.equal
PAIR_SET = {(case, leg) for case in single.evidence.CASES for leg in ("control", "profile")}
CPU_VENV = ROOT / ".runtime/venvs/qwen-posthoc-cpu"
FABRIC_SHA256 = "998956619429c918c9dba07f8a68faf5d9e8c630b799496e261e54622797db91"
FABRIC_ALIAS = "/opt/amazon/efa/lib/libfabric.so.1"
FABRIC_CANONICAL = "/opt/amazon/efa/lib/libfabric.so.1.25.0"


def digest(path):
    return single.evidence.Inputs().bind(path)


def execution_record(allow_login_cpu=False):
    return {"hostname": socket.gethostname(), "python_prefix": str(Path(sys.prefix).resolve()),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
        "allocated_compute": bool(os.environ.get("SLURM_JOB_ID") and os.environ.get("SLURM_JOB_PARTITION") in ("debug", "preemptable")),
        "user_authorized_cpu_only": bool(allow_login_cpu), "gpu_workload": False}


def require_execution(allow_login_cpu=False):
    if allow_login_cpu:
        require(Path(sys.prefix).resolve() == CPU_VENV, "CPU-only execution requires the isolated qwen-posthoc-cpu virtual environment")
    else:
        require(os.environ.get("SLURM_JOB_ID") and os.environ.get("SLURM_JOB_PARTITION") in ("debug", "preemptable"),
                "configured Slurm compute or explicit authorized CPU-only mode required")


def container_library_evidence(path, inputs, build):
    """Bind an exact runtime library copy without changing its recorded path."""
    path, build = Path(path), Path(build)
    require(path.is_absolute() and path.is_file() and not path.is_symlink(), "regular absolute container library map required")
    record = inputs.read(path)
    expected_keys = {"manifest_path", "original_alias", "original_canonical_path", "local_copy_path", "sha256", "extraction_evidence"}
    require(type(record) is dict and set(record) == expected_keys, "container library map schema differs")
    equal(record["manifest_path"], str(build / "libfabric_runtime.sha256"), "mapped library manifest scope")
    equal(record["original_alias"], FABRIC_ALIAS, "mapped runtime alias")
    equal(record["original_canonical_path"], FABRIC_CANONICAL, "mapped runtime canonical path")
    equal(record["sha256"], FABRIC_SHA256, "mapped runtime library SHA")
    manifest_path = Path(record["manifest_path"])
    inputs.bind(manifest_path)
    lines = manifest_path.read_text().splitlines()
    require(len(lines) == 1 and len(lines[0].split(maxsplit=1)) == 2, "one original runtime library binding required")
    expected_sha, alias = lines[0].split(maxsplit=1)
    equal(expected_sha, FABRIC_SHA256, "recorded original library SHA")
    equal(alias.lstrip(" *"), FABRIC_ALIAS, "recorded original library alias")
    local = Path(record["local_copy_path"])
    require(local.is_absolute() and local.is_file() and not local.is_symlink(), "regular absolute extracted library copy required")
    inputs.bind(local, FABRIC_SHA256)
    evidence = record["extraction_evidence"]
    require(type(evidence) is list and bool(evidence) and len(set(evidence)) == len(evidence), "extraction evidence paths required")
    for value in evidence:
        require(type(value) is str and Path(value).is_absolute(), "extraction evidence must use absolute file paths")
        inputs.bind(value)
    return {**record, "map_path": str(path), "map_sha256": inputs.bind(path),
            "verification": "Exact local bytes match the runtime SHA; the recorded container canonical path is preserved."}


@contextmanager
def mapped_libfabric_manifest(mapping):
    if mapping is None:
        yield
        return
    previous = single.evidence._manifest

    def mapped(path, inputs):
        if Path(path) != Path(mapping["manifest_path"]):
            return previous(path, inputs)
        inputs.bind(path)
        inputs.bind(mapping["map_path"], mapping["map_sha256"])
        inputs.bind(mapping["local_copy_path"], mapping["sha256"])
        for item in mapping["extraction_evidence"]:
            inputs.bind(item)
        return {"path": mapping["original_canonical_path"], "sha256": mapping["sha256"]}

    single.evidence._manifest = mapped
    try:
        yield
    finally:
        single.evidence._manifest = previous


def stage(directory, name, **details):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    record = {"time": datetime.now(timezone.utc).isoformat(), "stage": name, **details}
    temporary = directory / "CURRENT.json.next"
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, directory / "CURRENT.json")
    with (directory / "timeline.jsonl").open("a") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")


def save_proof(path, record):
    single.write_new(path, record)
    single.write_new(Path(path).with_suffix(".sha256.json"), {"path": str(path), "sha256": digest(path)})


def cached_proof(path, expected=None):
    path = Path(path)
    inputs = single.evidence.Inputs()
    marker = inputs.read(path.with_suffix(".sha256.json"))
    equal(marker.get("path"), str(path), "cached proof path")
    inputs.bind(path, marker.get("sha256"))
    result = inputs.read(path)
    equal(result.get("posthoc_validator_sha256"), digest(Path(__file__)), "cached posthoc validator identity")
    if expected:
        for key, value in expected.items():
            equal(result.get(key), value, f"cached {key}")
    require(type(result.get("input_files")) is list and bool(result["input_files"]), "cached proof lacks input bindings")
    for item in result["input_files"]:
        require(set(item) == {"path", "sha256"}, "cached input binding schema differs")
        inputs.bind(item["path"], item["sha256"])
    inputs.recheck()
    return result


def read_case_list(path, *, root=ROOT):
    path, root = Path(path), Path(root)
    require(path.is_file() and not path.is_symlink(), "case list must be a regular file")
    rows = []
    for line in path.read_text().splitlines():
        fields = line.split("\t")
        require(len(fields) == 4, "case list requires case,leg,job,orchestration")
        case, leg, job, raw = fields
        directory = Path(raw)
        require((case, leg) in PAIR_SET and job.isdigit(), "invalid campaign case,leg,job")
        require(directory.is_absolute() and directory.parent == root / "runs" and directory.is_dir()
                and not directory.is_symlink(), "invalid campaign orchestration root")
        rows.append({"case": case, "leg": leg, "training_slurm_job_id": job, "orchestration_dir": str(directory)})
    equal(sorted((row["case"], row["leg"]) for row in rows), sorted(PAIR_SET), "campaign case matrix")
    require(len(rows) == 8 and len({row["orchestration_dir"] for row in rows}) == 8
            and len({row["training_slurm_job_id"] for row in rows}) == 8, "campaign must contain eight distinct acquisitions")
    return rows


def recorded_manifest(path, inputs, orchestration, root):
    """Check historical source bytes, using archived copies when necessary."""
    inputs.bind(path)
    records = {}
    for line in Path(path).read_text().splitlines():
        fields = line.split(maxsplit=1)
        require(len(fields) == 2, "malformed recorded source manifest")
        expected, raw = fields
        source = Path(raw.lstrip(" *"))
        require(".." not in source.parts, "recorded source path traverses a parent directory")
        source = source if source.is_absolute() else root / source
        require(len(expected) == 64 and all(c in "0123456789abcdef" for c in expected)
                and str(source) not in records, "invalid recorded source binding")
        require(source == root or root in source.parents, "recorded project source escapes root")
        archived = orchestration / "source" / source.relative_to(root)
        archive_runtime = orchestration / "source/runtime" / source.relative_to(root)
        candidates = [candidate for candidate in (archived, archive_runtime, source)
                      if candidate.is_file() and not candidate.is_symlink()]
        matching = [candidate for candidate in candidates if digest(candidate) == expected]
        require(bool(matching), f"recorded source bytes unavailable or changed: {source}")
        inputs.bind(matching[0], expected)
        records[str(source)] = expected
    require(bool(records), "empty recorded source manifest")
    return records


def execution_sources(launch, campaign, reused, root):
    required = (("jobs/training/uccl_ep_debug_reuse.sbatch", "jobs/training/qwen3_uccl_debug_reuse_worker.sh",
        "profiling/run_pretrain_debug_reuse.py", "tools/uccl_ep_verified_inputs_reuse.py", "profiling/run_pretrain_debug.py")
        if reused else ("jobs/training/uccl_ep_debug_case.sbatch", "jobs/training/qwen3_uccl_debug_worker.sh", "profiling/run_pretrain_debug.py"))
    # Some source-bound campaign manifests are QA inventories, not complete
    # execution inventories. The canonical debug adapter is bound by the
    # actual launch inventory even when that earlier QA subset omits it.
    qa_required = required[:4] if reused else required
    for relative in required:
        require(str(Path(root) / relative) in launch, f"recorded actual execution source missing: {relative}")
    for relative in qa_required:
        require(str(Path(root) / relative) in campaign, f"recorded campaign source missing: {relative}")


def verification_evidence(orchestration, acquisition, reviewed, inputs, *, first_root=FIRST):
    """Bind the exact first proof without relabelling it as fresh verification."""
    mode = acquisition.get("verification_mode")
    if mode is None:
        require(Path(orchestration) == Path(first_root), "only the original first case can declare fresh verification")
        require("verification_mode" not in reviewed, "fresh/reused verification declarations disagree")
        equal(reviewed["checkpoint_payload"].get("verified"), True, "first payload verification")
        return {"verification_mode": "fresh_first_case", "payload_verified_in_training_allocation": True,
                "first_verified_proof": {"path": str(Path(orchestration) / "reviewed_inputs.json"),
                    "sha256": inputs.bind(Path(orchestration) / "reviewed_inputs.json"),
                    "slurm_job_id": acquisition["job"]}}
    equal(mode, "reused_first_case", "reused acquisition verification mode")
    equal(reviewed.get("verification_mode"), mode, "reused reviewed-input verification mode")
    reuse = acquisition.get("verification_reuse")
    equal(reviewed.get("verification_reuse"), reuse, "reused verification declarations")
    for key in ("payload_verified_in_current_allocation", "indexed_payload_verified_in_current_allocation",
                "native_build_reverified_in_current_allocation"):
        equal(reuse.get(key), False, f"explicit historical verification {key}")
    reference = reuse.get("first_verified_proof")
    require(type(reference) is dict and set(reference) == {"path", "sha256", "slurm_job_id"}, "invalid first proof reference")
    first_path = Path(first_root) / "reviewed_inputs.json"
    equal(reference["path"], str(first_path), "first proof path")
    inputs.bind(first_path, reference["sha256"])
    first = inputs.read(first_path)
    equal(first.get("slurm_job_id"), reference["slurm_job_id"], "first proof allocation")
    equal(first.get("valid_reviewed_inputs"), True, "first proof validity")
    equal(first.get("checkpoint_payload", {}).get("verified"), True, "historical payload verified")
    first_completed = inputs.read(Path(first_root) / "acquisition_completed.json")
    for key, value in {"success": True, "workers_srun_returned_zero": True,
                       "completed_updates": 10, "job": reference["slurm_job_id"]}.items():
        equal(first_completed.get(key), value, f"historical completed acquisition {key}")
    payload = reviewed["checkpoint_payload"]
    for key, value in {"verified": True, "verified_in_current_allocation": False,
                       "verification_mode": mode, "verification_slurm_job_id": reference["slurm_job_id"],
                       "first_verified_proof": reference}.items():
        equal(payload.get(key), value, f"historical checkpoint proof {key}")
    for key in ("path", "file_count", "total_bytes", "inventory_sha256"):
        equal(payload.get(key), first["checkpoint_payload"].get(key), f"historical payload identity {key}")
    for key in ("c0_identity_manifest_sha256", "dataset_manifest_sha256", "native_checkouts", "runtime_versions"):
        equal(reviewed.get(key), first.get(key), f"reused immutable proof {key}")
    return {"verification_mode": mode, "payload_verified_in_training_allocation": False,
            "first_verified_proof": reference, "claim_limit": "Historical immutable payload verification was reused; no fresh payload verification is claimed."}


class IdentityInputs:
    """Redirect only posthoc CXI proof records; keep all original log bindings."""
    def __init__(self, inputs, run, output):
        self.inputs, self.run, self.output = inputs, run, output

    def bind(self, *args, **kwargs):
        return self.inputs.bind(*args, **kwargs)

    def read(self, path):
        path = Path(path)
        for node in range(4):
            if path == self.run / "raw" / f"native-cxi-node-{node}.json":
                proof = inspect_node(self.run / "logs" / f"torchrun-node-{node}", node)
                if path.exists():
                    equal(self.inputs.read(path), proof, "recorded CXI proof agrees with original logs")
                destination = self.output / "native-cxi" / f"node-{node}.json"
                if destination.exists():
                    equal(self.inputs.read(destination), proof, "existing posthoc CXI proof")
                else:
                    single.write_new(destination, proof)
                return self.inputs.read(destination)
        return self.inputs.read(path)


def bind_list(inputs):
    return [{"path": path, "sha256": value} for path, value in sorted(inputs.hashes.items())]


def validate_metadata(row, directory, *, root=ROOT, first_root=FIRST, allow_login_cpu=False, container_library_map=None):
    directory = Path(directory)
    expected = {key: row[key] for key in ("case", "leg", "training_slurm_job_id", "orchestration_dir")}
    metadata_path = directory / "metadata_acceptance.json"
    if metadata_path.exists():
        cached = cached_proof(metadata_path, {**expected, "valid_debug_metadata": True})
        if container_library_map is not None:
            mapping = cached.get("container_library_mapping")
            require(type(mapping) is dict, "cached metadata lacks the requested container library mapping")
            equal(mapping.get("map_path"), str(container_library_map), "cached container library map path")
            equal(mapping.get("map_sha256"), digest(container_library_map), "cached container library map identity")
        return cached
    stage(directory, "metadata_started", **expected)
    inputs = single.evidence.Inputs()
    inputs.bind(Path(__file__))
    inputs.bind(Path(single.__file__))
    orchestration = Path(row["orchestration_dir"])
    case, leg, run = single.one_case(orchestration, Path(root), inputs)
    equal((case, leg), (row["case"], row["leg"]), "case-list acquisition identity")
    acquisition = inputs.read(orchestration / "acquisition.json")
    for key, value in {"mode": "full", "experiment_protocol": single.PROTOCOL, "partition": "debug", "case": case,
        "leg": leg, "warmup_steps": 5, "measured_steps": 5, "completed_updates": 10,
        "cache_budget_updates": 25, "cache_budget_samples": 1600, "job": row["training_slurm_job_id"],
        "topology": "cp1_ep16", "launch_policy": "torchrun_node4", "accepted_pilot_proof": None,
        "uccl_transport_policy": single.DEBUG_TRANSPORT_POLICY}.items():
        equal(acquisition.get(key), value, f"posthoc acquisition {key}")
    job = acquisition["job"]
    environment = inputs.read(run / "environment.json")
    sources = environment.get("ampa_runtime_sources")
    equal(sources, single._ampa_runtime_sources(Path(root)), "current compatible canonical runtime sources")
    equal(acquisition["runtime_source_sha256"], sources["sha256"], "recorded runtime source identity")
    runtime_records = recorded_manifest(orchestration / "runtime-source.sha256", inputs, orchestration, Path(root))
    equal(runtime_records, {str(Path(root) / name): value for name, value in sources["files"].items()}, "recorded runtime coverage")
    campaign = recorded_manifest(orchestration / "campaign-source.sha256", inputs, orchestration, Path(root))
    equal(acquisition["campaign_source_sha256"], inputs.bind(orchestration / "campaign-source.sha256"), "recorded campaign identity")
    launch = recorded_manifest(orchestration / "source.sha256", inputs, orchestration, Path(root))
    reused = acquisition.get("verification_mode") == "reused_first_case"
    execution_sources(launch, campaign, reused, root)
    completed = inputs.read(orchestration / "acquisition_completed.json")
    for key, value in {"success": True, "workers_srun_returned_zero": True, "mode": "full", "partition": "debug", "job": job,
        "runtime_source_sha256": sources["sha256"], "binary_sha256": acquisition["binary_sha256"],
        "cases_tsv_sha256": inputs.bind(orchestration / "cases.tsv"), "warmup_steps": 5,
        "measured_steps": 5, "completed_updates": 10}.items():
        equal(completed.get(key), value, f"posthoc acquisition completion {key}")
    contract = single.checked_contract(inputs.read(run / "config/comparison_contract.json"), case)
    require(container_library_map is None or allow_login_cpu, "container library map is limited to explicitly authorized CPU-only execution")
    library_mapping = None
    if container_library_map is not None:
        build = Path(contract.get("uccl_build_dir") or Path(root) / "runs/uccl-ep-owned-build-20261101T1713Z")
        library_mapping = container_library_evidence(container_library_map, inputs, build)
    equal(inputs.read(run / "config/debug_execution_contract.json"), single.execution_contract(), "posthoc execution/cache contract")
    resolved_path = run / "config/resolved.yaml"
    inputs.bind(resolved_path)
    resolved = yaml.safe_load(resolved_path.read_text())
    single.checked_resolved(resolved, case, run)
    for key, value in {"slurm_job_id": job, "launch_policy": "torchrun_node4",
                       "runtime_topology": single._EP16_RUNTIME_TOPOLOGIES["cp1_ep16"]}.items():
        equal(environment.get(key), value, f"posthoc environment {key}")
    placement = single.validate_rank_placement(environment["rank_placement"], expected_world_size=16, launch_policy="torchrun_node4")
    single.validate_runtime_groups(placement, expected_world_size=16, cp=1)
    manifests = {name: inputs.bind(run / "config" / name) for name in ("c0_manifest.json", "dataset_manifest.json", "model_manifest.json")}
    equal(manifests["c0_manifest.json"], contract["c0_identity_manifest_sha256"], "posthoc C0 manifest")
    equal(manifests["dataset_manifest.json"], contract["dataset"]["manifest_sha256"], "posthoc dataset manifest")
    reviewed_path = orchestration / "reviewed_inputs.json"
    reviewed = inputs.read(reviewed_path)
    verification = verification_evidence(orchestration, acquisition, reviewed, inputs, first_root=first_root)
    for key, value in {"valid_reviewed_inputs": True, "slurm_job_id": job, "ampa_runtime_sources": sources,
        "c0_identity_manifest_sha256": manifests["c0_manifest.json"], "dataset_manifest_sha256": manifests["dataset_manifest.json"]}.items():
        equal(reviewed.get(key), value, f"posthoc reviewed input {key}")
    equal(reviewed["checkpoint_payload"]["verified"], True, "historical payload validity")
    c0 = Path(root) / "checkpoint/megatron/qwen3-30b-a3b-base-c0"
    single.evidence._path(reviewed["checkpoint_payload"]["path"], c0, "posthoc payload path")
    equal(contract["initialization"], {"mode": "pretrained", "checkpoint": str(c0)}, "posthoc initialization")
    equal(environment["checkpoint"], str(c0), "posthoc actual C0")
    equal(inputs.bind(c0 / "ampa_c0_manifest.json"), manifests["c0_manifest.json"], "canonical C0 manifest")
    equal(reviewed["native_checkouts"], {name: environment[name] for name in ("megatron_lm", "megatron_bridge")}, "posthoc native checkout identity")
    equal(environment["reviewed_inputs_proof"], {"path": str(reviewed_path), "sha256": inputs.bind(reviewed_path)}, "actual reviewed-input proof")
    for key, value in reviewed.get("runtime_versions", {}).items():
        equal(environment.get(key), value, f"reviewed runtime {key}")
    if reused:
        context = inputs.read(orchestration / "input_reuse_context.json")
        for key, value in {"verification_mode": "reused_first_case", "slurm_job_id": job, "run_dir": str(run),
            "pretrained_checkpoint": str(c0), "dataset_manifest_path": str(run / "config/dataset_manifest.json"),
            "c0_identity_manifest_sha256": manifests["c0_manifest.json"], "dataset_manifest_sha256": manifests["dataset_manifest.json"]}.items():
            equal(context.get(key), value, f"reuse context {key}")
        equal(context["indexed_manifest"], inputs.read(run / "config/dataset_manifest.json"), "actual copied indexed manifest")
        equal(context["verification_reuse"], reviewed["verification_reuse"], "reuse context proof")
    mode = inputs.read(run / "config/observation_mode.json")
    for key, value in {"data_mode": "indexed", "lightweight_profiling": "off", "nsys_trace": False, "torch_trace": leg == "profile",
        "sample_fingerprints": True, "router_observer_attached": False, "route_wrapper_installed": False,
        "dispatcher_preprocess_wrapper_installed": False}.items():
        equal(mode.get(key), value, f"posthoc observation {key}")
    capture = inputs.read(run / "config/all_step_capture.json")
    for key, value in {"enabled": leg == "profile", "steps": list(range(10)), "ranks": list(range(16)) if leg == "profile" else [0]}.items():
        equal(capture.get(key), value, f"posthoc capture declaration {key}")
    if leg == "control":
        require(not any((run / "traces/torch_profile").rglob("*.json*")) and not any((run / "raw/deep_trace").rglob("*.json")), "control contains capture artifacts")
    fingerprints = {}
    for rank in range(16):
        inputs.bind(run / "logs" / f"rank-{rank}.out")
        path = run / "raw" / f"sample_fingerprint_rank-{rank}.json"
        fingerprints[str(rank)] = single._checked_fingerprint_batches(inputs.read(path), rank=rank, accumulation=4,
            steps=10, token_shape=[1, 4096], path=path)
    prefix = hashlib.sha256(json.dumps(fingerprints, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    health, timing = single.checked_timing(run)
    ownership = inputs.read(run / "config/routing_ownership.json")
    audits = [inputs.read(run / "raw" / f"routing_audit_rank-{rank}.json") for rank in range(16)]
    received = single.checked_receives(contract, ownership, audits, environment)
    if case in single.evidence.UCCL_CASES:
        build = Path(contract["uccl_build_dir"])
        require(build.is_absolute() and build.is_dir() and not build.is_symlink(), "isolated UCCL build required")
        identity = single.verify_uccl_source(build)
        equal(inputs.read(build / "source_identity.json"), identity, "posthoc approved UCCL identity")
        equal(acquisition["uccl_source_identity"], identity, "recorded UCCL identity")
        inputs.bind(build / "source/uccl-borrowed-scratch.patch", identity["patch_sha256"])
        for relative, value in identity["modified_files"].items():
            inputs.bind(build / "uccl" / relative, value)
        with mapped_libfabric_manifest(library_mapping):
            binary = single.evidence._uccl_identity(run, build, identity, placement, job, Path(root), IdentityInputs(inputs, run, directory))
        equal(binary, acquisition["binary_sha256"], "posthoc actual UCCL binary")
    for filename, record in (("training_health.json", health), ("timing.json", timing), ("received_routes.json", received)):
        single.write_new(directory / filename, record)
        inputs.bind(directory / filename)
    normalized = copy.deepcopy(contract)
    normalized["training"].update(completed_steps=2, measured_steps=20)
    normalized = single.evidence._contract(normalized, case)
    normalized["training"].update(completed_steps=10, measured_steps=5)
    normalized_resolved = copy.deepcopy(resolved)
    normalized_resolved["train"]["exit_interval"] = 2
    normalized_resolved = single.evidence._resolved(normalized_resolved, case, run)
    normalized_resolved["train"]["exit_interval"] = 10
    inputs.recheck()
    result = {"valid_debug_metadata": True, "valid_debug_case": leg == "control", **expected,
        "experiment_protocol": single.PROTOCOL, "run_dir": str(run), "completed_updates": 10,
        "warmup_steps": 5, "measured_steps": 5, "valid_training_health": True, "valid_send_receive_audit": True,
        "valid_capture": None, "profile": None, "timing": timing, "rank_placement": placement,
        "sample_prefix_sha256": prefix, "consumed_cpu_batches": 640, "verification": verification,
        "c0_identity_manifest_sha256": manifests["c0_manifest.json"], "dataset_manifest_sha256": manifests["dataset_manifest.json"],
        "model_manifest_sha256": manifests["model_manifest.json"], "runtime_source_sha256": sources["sha256"],
        "binary_sha256": acquisition["binary_sha256"], "normalized_contract": normalized,
        "normalized_resolved_yaml": yaml.safe_dump(normalized_resolved),
        "same_actual_rank_placement_across_allocations_claimed": False, "pure_profiler_overhead_available": False,
        "validation_execution": execution_record(allow_login_cpu), "container_library_mapping": library_mapping,
        "posthoc_validator_sha256": inputs.bind(Path(__file__)), "input_files": bind_list(inputs)}
    save_proof(metadata_path, result)
    if leg == "control":
        save_proof(directory / "acceptance.json", result)
    stage(directory, "metadata_complete", case=case, leg=leg, valid_debug_case=leg == "control")
    return result


def profile_commands(case, run, output, *, root=ROOT, allow_login_cpu=False):
    if not allow_login_cpu or case in single.evidence.UCCL_CASES:
        return single.profile_commands(case, run, output, Path(root))
    return [(f"rank-{rank}", [sys.executable, str(Path(root) / "tools/uccl_debug_campaign_validate.py"),
        "--allow-login-cpu", "--native-run", str(run), "--native-rank", str(rank),
        "--native-output", str(Path(output) / "profiles" / f"rank-{rank}")]) for rank in range(16)]


def run_profile_validators(case, run, output, workers, *, root=ROOT, allow_login_cpu=False):
    (output / "profiles").mkdir()
    (output / "logs").mkdir()
    commands = profile_commands(case, run, output, root=root, allow_login_cpu=allow_login_cpu)
    single.write_new(output / "commands.json", commands)

    def execute(task):
        name, command = task
        with (output / "logs" / f"{name}.log").open("x") as stream:
            subprocess.run(command, cwd=root, stdout=stream, stderr=subprocess.STDOUT, check=True)
        return name

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(execute, task) for task in commands]
        for future in as_completed(futures):
            print(json.dumps({"validator": future.result(), "completed": True}), flush=True)


def validate_profile(row, directory, *, workers=4, root=ROOT, first_root=FIRST, allow_login_cpu=False, container_library_map=None):
    directory = Path(directory)
    proof = directory / "acceptance.json"
    expected = {key: row[key] for key in ("case", "leg", "training_slurm_job_id", "orchestration_dir")}
    if proof.exists():
        return cached_proof(proof, {**expected, "valid_debug_case": True})
    metadata = validate_metadata(row, directory, root=root, first_root=first_root,
                                 allow_login_cpu=allow_login_cpu, container_library_map=container_library_map)
    require(row["leg"] == "profile", "profile validation requires profile leg")
    inputs = single.evidence.Inputs()
    inputs.bind(directory / "metadata_acceptance.json")
    for item in metadata["input_files"]:
        inputs.bind(item["path"], item["sha256"])
    attempt = directory
    index = 1
    while (attempt / "profiles").exists():
        attempt = directory / f"profile-attempt-{index:03d}"
        index += 1
    attempt.mkdir(parents=True, exist_ok=True)
    stage(directory, "profile_started", case=row["case"], leg=row["leg"], profile_output_dir=str(attempt))
    run_profile_validators(row["case"], Path(metadata["run_dir"]), attempt, workers,
                           root=Path(root), allow_login_cpu=allow_login_cpu)
    profile = single.checked_profile(row["case"], Path(metadata["run_dir"]), attempt, Path(root), inputs)
    inputs.recheck()
    result = {**metadata, "valid_debug_case": True, "valid_capture": True, "profile": profile,
        "profile_output_dir": str(attempt), "validation_slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "validation_execution": execution_record(allow_login_cpu),
        "posthoc_validator_sha256": inputs.bind(Path(__file__)), "input_files": bind_list(inputs)}
    save_proof(proof, result)
    stage(directory, "case_accepted", case=row["case"], leg=row["leg"], rank_update_captures=160)
    return result


def finalize_campaign(rows, output, case_list, *, allow_login_cpu=False):
    output = Path(output)
    records, case_runs, acceptances = [], {}, {}
    inputs = single.evidence.Inputs()
    inputs.bind(case_list)
    reference = None
    for row in rows:
        path = output / "cases" / f"{row['case']}-{row['leg']}" / "acceptance.json"
        result = cached_proof(path, {"valid_debug_case": True, **row})
        inputs.bind(path)
        common = {name: result[name] for name in ("sample_prefix_sha256", "c0_identity_manifest_sha256", "dataset_manifest_sha256",
            "model_manifest_sha256", "runtime_source_sha256", "binary_sha256", "normalized_contract")}
        if reference is None:
            reference = result
        equal(common, {name: reference[name] for name in common}, "cross-case workload and actual input-prefix identity")
        single.evidence.equal_resolved_configuration(yaml.safe_load(result["normalized_resolved_yaml"]),
            yaml.safe_load(reference["normalized_resolved_yaml"]), "cross-case resolved workload")
        if row["leg"] == "profile":
            equal(result["valid_capture"], True, "complete profile capture")
            equal(result["profile"]["rank_update_captures"], 160, "profile capture count")
        case_runs.setdefault(row["case"], {})[row["leg"]] = result["run_dir"]
        acceptances.setdefault(row["case"], {})[row["leg"]] = {"path": str(path), "sha256": inputs.bind(path)}
        records.append({**row, "run_dir": result["run_dir"], "acceptance": acceptances[row["case"]][row["leg"]],
                        "verification": result["verification"]})
    result = {"valid_debug_campaign": True, "experiment_protocol": single.PROTOCOL, "cases": records,
        "case_runs": case_runs, "case_acceptance": acceptances, "completed_updates_per_case": 10,
        "warmup_steps": 5, "measured_steps": 5, "sample_prefix_sha256": reference["sample_prefix_sha256"],
        "rank_update_captures": 640, "same_actual_rank_placement_across_allocations_claimed": False,
        "pure_profiler_overhead_available": False, "validation_slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "validation_execution": execution_record(allow_login_cpu),
        "posthoc_validator_sha256": inputs.bind(Path(__file__)), "input_files": bind_list(inputs),
        "claim_limits": ["Different debug allocations can use different nodes and network conditions.",
            "Seven cases explicitly reuse first-case immutable verification; no fresh training-time payload check is claimed.",
            "Five measured updates describe a short dependent timing window, not five independent cluster repetitions.",
            "Observed profiler ON/OFF changes do not isolate profiler overhead.",
            "Logical assignments and correlated activity are not physical wire bytes or exposed communication latency."]}
    inputs.recheck()
    save_proof(output / "campaign_acceptance.json", result)
    stage(output, "campaign_accepted", case_count=8, rank_update_captures=640)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-list", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--workers", type=int, default=4, choices=range(1, 9))
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--case", choices=single.evidence.CASES)
    parser.add_argument("--leg", choices=("control", "profile"))
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--allow-login-cpu", action="store_true", help="Explicit user-authorized CPU-only execution in the isolated posthoc venv.")
    parser.add_argument("--container-library-map", type=Path)
    parser.add_argument("--native-run", type=Path)
    parser.add_argument("--native-rank", type=int, choices=range(16))
    parser.add_argument("--native-output", type=Path)
    args = parser.parse_args()
    require_execution(args.allow_login_cpu)
    require(args.container_library_map is None or args.allow_login_cpu, "container library map requires authorized CPU-only mode")
    native = (args.native_run, args.native_rank, args.native_output)
    if any(value is not None for value in native):
        require(all(value is not None for value in native) and args.case_list is None and args.output_dir is None
                and args.container_library_map is None and not args.metadata_only and not args.finalize
                and args.case is None and args.leg is None, "native worker arguments must be complete and exclusive")
        single.native_rank(args.native_run.resolve(), args.native_output.resolve(), args.native_rank)
        return
    require(args.case_list is not None and args.output_dir is not None, "case-list and output-dir required outside native worker mode")
    require(not args.output_dir.is_symlink(), "symlink output forbidden")
    output = args.output_dir.resolve()
    require(output.parent == ROOT / "runs", "campaign output must be directly under project runs")
    require((args.case is None) == (args.leg is None), "selected case and leg must be supplied together")
    rows = read_case_list(args.case_list)
    output.mkdir(parents=True, exist_ok=True)
    try:
        if args.finalize:
            finalize_campaign(rows, output, args.case_list, allow_login_cpu=args.allow_login_cpu)
            return
        selected = [row for row in rows if args.case is None or (row["case"], row["leg"]) == (args.case, args.leg)]
        for row in selected:
            directory = output / "cases" / f"{row['case']}-{row['leg']}"
            try:
                stage(output, "case_started", **row)
                validate_metadata(row, directory, allow_login_cpu=args.allow_login_cpu,
                                  container_library_map=args.container_library_map)
                if row["leg"] == "profile" and not args.metadata_only:
                    validate_profile(row, directory, workers=args.workers, allow_login_cpu=args.allow_login_cpu,
                                     container_library_map=args.container_library_map)
            except BaseException as error:
                stage(directory, "failed", error_type=type(error).__name__, error=str(error), **row)
                raise
        if not args.metadata_only and args.case is None:
            finalize_campaign(rows, output, args.case_list, allow_login_cpu=args.allow_login_cpu)
        else:
            stage(output, "selected_stage_complete", metadata_only=args.metadata_only, selected_cases=len(selected))
    except BaseException as error:
        stage(output, "failed", error_type=type(error).__name__, error=str(error))
        raise


if __name__ == "__main__":
    main()
