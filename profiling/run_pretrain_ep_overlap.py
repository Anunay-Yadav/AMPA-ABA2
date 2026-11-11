#!/usr/bin/env python3
"""Run the selected EP overlap bundle with explicit historical input reuse.

The original parser validates the 25-update indexed cache. The adapter exits
after two qualification updates or ten measured-workload updates. All training
and audit callbacks remain canonical, except for the separate capture owner.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import shutil
import sys

from profiling import run_pretrain_debug_reuse as reuse


canonical = reuse.debug.canonical
BASE_PARSE_ARGS = reuse.debug.BASE_PARSE_ARGS
BASE_BUILD_CONFIG = reuse.debug.BASE_BUILD_CONFIG
BASE_COMPARISON_CONTRACT = canonical.comparison_contract
ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "ep_overlap_main_delay_early_5_plus_5_v1"
CASES = ("ordinary", "grouped", "uccl_ordinary", "uccl")
CAPTURES = ("off", "torch", "nsys")
BUDGETS = (12, 20, 32, 40)
OVERLAP_FLAGS = {
    "overlap_moe_expert_parallel_comm": True,
    "delay_wgrad_compute": True,
    "ep_overlap_early_attn_memory_release": True,
    "overlap_dispatch_backward_with_experts_wgrad": False,
    "moe_shared_expert_overlap": False,
    "moe_shortcut_parallel": False,
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f"expected regular metadata: {path}")
    return json.loads(path.read_text())


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def require_allocation(nodes=4):
    require(os.environ.get("SLURM_JOB_PARTITION") == "debug"
            and os.environ.get("SLURM_JOB_NUM_NODES") == str(nodes)
            and os.environ.get("SLURM_JOB_ID", "").isdigit(),
            f"a fresh {nodes}-node debug allocation is required")
    require(os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS") == "32",
            "set CUDA_DEVICE_MAX_CONNECTIONS=32 inside the container before Torch import")


def selected_parser():
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--prepare-inputs", action="store_true")
    parser.add_argument("--orchestration-dir", type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--case", required=True, choices=CASES)
    parser.add_argument("--capture", required=True, choices=CAPTURES)
    parser.add_argument("--communication-sms", required=True, type=int, choices=BUDGETS)
    parser.add_argument("--execution-steps", required=True, type=int, choices=(2, 10))
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measured-steps", type=int, default=5)
    return parser


def validate_selection(selected, parser):
    if (selected.warmup_steps, selected.measured_steps) != (5, 5):
        parser.error("the requested workload must retain five warmup and five measured updates")
    if selected.capture == "nsys" and selected.case != "uccl_ordinary":
        parser.error("Nsight is restricted to ordinary UCCL")
    if selected.capture != "nsys" and selected.communication_sms != 20:
        parser.error("only the separate Nsight budget sweep changes the 20-block reference")


def execution_contract(args):
    steps = args.execution_steps
    warmup, measured = (5, 5) if steps == 10 else (1, 1)
    capture = args.overlap_capture
    captured = (list(range(steps)) if capture == "torch" or (capture == "nsys" and steps == 2)
                else list(range(5, 10)) if capture == "nsys" else [])
    return {
        "schema_version": 1, "experiment_protocol": PROTOCOL,
        "case": args.case, "capture": capture,
        "communication_backend": args.communication_backend,
        "communication_sms": args.communication_sms if args.communication_backend == "uccl_ep" else None,
        "budget_meaning": "UCCL payload block budget; not measured active SMs.",
        "execution_steps": steps, "completed_steps": steps,
        "warmup_steps": warmup, "measured_steps": measured,
        "qualification_only": steps == 2,
        "cache_budget_steps": 25, "cache_budget_samples": 1600,
        "train_iters": 25, "exit_interval": steps,
        "capture_step_indices": captured,
        "capture_ranks": list(range(16)) if capture != "off" else [],
        "overlap_flags": dict(OVERLAP_FLAGS), "bridge_comm_overlap_configured": False,
        "ddp_delay_wgrad_compute": True,
        "CUDA_DEVICE_MAX_CONNECTIONS": "32",
        "capture_owner": "profiling.ampa_profile.overlap_capture",
        "kineto_enabled": capture == "torch",
        "source_manifest_sha256": args.overlap_source_sha256,
        "input_verification": "historical first successful fixed input proof; no fresh payload verification",
        "claim_limit": "Qualification pilots establish execution only; instrumented runs are diagnostic.",
    }


def verify_source_manifest(path, expected, *, root=ROOT):
    """Bind the small frozen source inventory without reading training payloads."""
    path, root = Path(path), Path(root)
    require(path.is_absolute() and path.is_file() and not path.is_symlink(), "source manifest must be absolute and regular")
    require(re.fullmatch(r"[0-9a-f]{64}", expected or "") and sha(path) == expected,
            "source manifest digest differs")
    files = {}
    for line in path.read_text().splitlines():
        match = re.fullmatch(r"([0-9a-f]{64}) [ *](.+)", line)
        require(match is not None, "unsupported source manifest line")
        name = Path(match[2])
        target = name if name.is_absolute() else root / name
        require(target.is_relative_to(root) and ".." not in target.parts and target != path,
                "source manifest path is outside the project or self-referential")
        relative = str(target.relative_to(root))
        require(relative not in files and target.is_file() and not target.is_symlink(),
                "source inventory contains duplicate, missing, or symlink files")
        require(sha(target) == match[1], f"source bytes differ: {relative}")
        files[relative] = match[1]
    require(files, "source manifest is empty")
    return {"path": str(path), "sha256": expected, "files": files}


def prepare_inputs(selected, *, root=ROOT):
    """Copy only manifests and bind this allocation to the historical proof."""
    from tools.uccl_ep_verified_inputs_reuse import FIRST_ROOT, load_first_case
    from profiling.ampa_profile.provenance import _ampa_runtime_sources

    require_allocation()
    root, run, orchestration = Path(root), selected.run_dir, selected.orchestration_dir
    require(orchestration is not None and run.is_absolute() and orchestration.is_absolute()
            and run.parent == orchestration.parent == root / "runs" and run != orchestration
            and not run.is_symlink() and not orchestration.is_symlink(),
            "preparation requires separate direct project run directories")
    first_root = Path(os.environ.get("AMPA_UCCL_VERIFIED_FIRST_ROOT", str(FIRST_ROOT)))
    require(first_root == FIRST_ROOT, "the first successful fixed input proof must remain pinned")
    first = load_first_case(first_root, root=root)
    require(orchestration != first["root"] and run != first["run"], "preserve first-case evidence")
    source_path = orchestration / "campaign-source.sha256"
    source = verify_source_manifest(source_path, os.environ.get("AMPA_OVERLAP_SOURCE_SHA256"), root=root)
    sources = _ampa_runtime_sources(root)
    run.joinpath("config").mkdir(parents=True, exist_ok=True)
    for name in ("c0_manifest.json", "dataset_manifest.json", "model_manifest.json"):
        origin, destination = first["run"] / "config" / name, run / "config" / name
        require(origin.is_file() and not origin.is_symlink() and not destination.exists() and not destination.is_symlink(),
                "input manifest source is unavailable or destination already exists")
        with destination.open("xb") as stream:
            with origin.open("rb") as input_stream:
                shutil.copyfileobj(input_stream, stream)
    c0, dataset = read_json(run / "config/c0_manifest.json"), read_json(run / "config/dataset_manifest.json")
    proof = copy.deepcopy(first["proof"])
    require(sha(run / "config/dataset_manifest.json") == proof["dataset_manifest_sha256"],
            "copied dataset manifest differs from the first proof")
    verification = {
        "verification_mode": "reused_first_case", "first_verified_proof": first["proof_reference"],
        "payload_verified_in_current_allocation": False,
        "indexed_payload_verified_in_current_allocation": False,
        "native_build_reverified_in_current_allocation": False,
        "recording_policy": "Historical payload proof reused. Current source and actual execution are recorded separately.",
    }
    common = {"experiment_protocol": PROTOCOL, "slurm_job_id": os.environ["SLURM_JOB_ID"],
              "partition": "debug", "case": selected.case, "capture": selected.capture,
              "communication_sms": selected.communication_sms if selected.case.startswith("uccl") else None,
              "execution_steps": selected.execution_steps,
              "qualification_only": selected.execution_steps == 2, "source_manifest": source}
    proof.update(common, ampa_runtime_sources=sources, verification_mode="reused_first_case", verification_reuse=verification)
    proof["checkpoint_payload"].update(verification_mode="reused_first_case",
        verification_slurm_job_id=first["proof_reference"]["slurm_job_id"],
        verified_in_current_allocation=False, first_verified_proof=first["proof_reference"])
    context = {**common, "schema_version": 1, "verification_mode": "reused_first_case",
        "verification_reuse": verification, "run_dir": str(run),
        "pretrained_checkpoint": proof["checkpoint_payload"]["path"],
        "c0_identity_manifest_sha256": proof["c0_identity_manifest_sha256"],
        "c0_hf_snapshot_manifest_sha256": c0.get("hf_snapshot_manifest_sha256"),
        "hf_model_id": c0.get("hf_model"), "hf_revision": c0.get("hf_revision"),
        "dataset_manifest_path": str(run / "config/dataset_manifest.json"),
        "dataset_manifest_sha256": proof["dataset_manifest_sha256"], "indexed_manifest": dataset}
    acquired = copy.deepcopy(first["acquisition"])
    acquired.update(common, job=os.environ["SLURM_JOB_ID"], leg=selected.capture,
        run_dir=str(run), communication_backend="uccl_ep" if selected.case.startswith("uccl") else "alltoall",
        overlap_flags=dict(OVERLAP_FLAGS), cuda_device_max_connections="32",
        completed_updates=selected.execution_steps, warmup_steps=5 if selected.execution_steps == 10 else 1,
        measured_steps=5 if selected.execution_steps == 10 else 1,
        runtime_source_sha256=sources["sha256"], campaign_source_sha256=source["sha256"],
        accepted_pilot_proof=None, historical_pilot_proof=None,
        verification_mode="reused_first_case", verification_reuse=verification,
        acquisition_analysis_deferred=True,
        first_verified_case={"root": str(first["root"]), "run_dir": str(first["run"]), **first["proof_reference"]})
    for name, value in (("reviewed_inputs.json", proof), ("input_reuse_context.json", context), ("acquisition.json", acquired)):
        write_new(orchestration / name, value)
    with (orchestration / "runtime-source.sha256").open("x") as stream:
        for name, digest in sorted(sources["files"].items()):
            stream.write(f"{digest}  {root / name}\n")
    return acquired


def load_context(selected):
    path = Path(os.environ.get("AMPA_UCCL_REUSE_CONTEXT", ""))
    require(path.is_absolute(), "AMPA_UCCL_REUSE_CONTEXT must be absolute")
    context = read_json(path)
    expected = {"experiment_protocol": PROTOCOL, "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "partition": "debug", "case": selected.case, "capture": selected.capture,
        "run_dir": str(selected.run_dir), "execution_steps": selected.execution_steps,
        "communication_sms": selected.communication_sms if selected.case.startswith("uccl") else None,
        "verification_mode": "reused_first_case"}
    require(all(context.get(key) == value for key, value in expected.items()), "reuse context differs from this allocation or selected case")
    verification = context.get("verification_reuse", {})
    require(all(verification.get(key) is False for key in ("payload_verified_in_current_allocation",
            "indexed_payload_verified_in_current_allocation", "native_build_reverified_in_current_allocation")),
            "reuse context must state that payload verification is historical")
    manifest = selected.run_dir / "config/dataset_manifest.json"
    require(context.get("dataset_manifest_path") == str(manifest) and sha(manifest) == context.get("dataset_manifest_sha256")
            and read_json(manifest) == context.get("indexed_manifest"), "prepared dataset metadata differs")
    source = context.get("source_manifest", {})
    require(source.get("sha256") == os.environ.get("AMPA_OVERLAP_SOURCE_SHA256"), "current source identity differs from prepared context")
    require(source.get("path") == str(path.parent / "campaign-source.sha256")
            and sha(source["path"]) == source.get("sha256"), "prepared source inventory differs")
    return context


def parse_overlap_args(argv=None):
    parser = selected_parser()
    selected, remaining = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    validate_selection(selected, parser)
    require_allocation()
    require(not selected.prepare_inputs and selected.orchestration_dir is None, "training must omit preparation options")
    require("--smoke-steps" not in remaining, "use --execution-steps for qualification pilots")
    context = load_context(selected)
    old_validators = canonical._validate_c0_manifest, canonical._validate_dataset_manifest
    old_argv = sys.argv
    canonical._validate_c0_manifest = lambda args, p: reuse.cached_c0(args, p, context)
    canonical._validate_dataset_manifest = lambda args, p: reuse.cached_dataset(args, p, context)
    try:
        # Validate immutable input/cache metadata at its original 25-update budget.
        # Do not pass canonical --smoke-steps: it incorrectly requires Kineto for OFF/Nsight.
        sys.argv = [old_argv[0], *remaining, "--run-dir", str(selected.run_dir),
                    "--warmup-steps", "5", "--measured-steps", "20"]
        args = BASE_PARSE_ARGS()
    finally:
        canonical._validate_c0_manifest, canonical._validate_dataset_manifest = old_validators
        sys.argv = old_argv
    expected_backend = "uccl_ep" if selected.case.startswith("uccl") else "alltoall"
    expected_groups = (4, 2) if selected.case in ("grouped", "uccl") else (None, None)
    if (args.communication_backend != expected_backend or (args.router_num_groups, args.router_group_topk) != expected_groups
            or args.topology_variant != "cp1_ep16" or args.launch_policy != "torchrun_node4"
            or not args.audit_received_experts or args.audit_num_groups != 4
            or args.learning_rate != 1e-6 or args.dataset_sampler_seed != 1234):
        parser.error("case/backend/router and fixed EP16 workload flags must agree")
    if (args.torch_profile_steps or args.nsys_capture_steps
            or args.torch_profile_all_steps != (selected.capture == "torch")):
        parser.error("--capture must agree with the sole explicit all-rank PyTorch profiler flag")
    from profiling.run_pretrain_uccl_budget import reject_overrides
    reject_overrides()
    args.case, args.overlap_capture = selected.case, selected.capture
    args.communication_sms, args.execution_steps = selected.communication_sms, selected.execution_steps
    args.completed_steps = args.smoke_steps = selected.execution_steps
    args.warmup_steps, args.measured_steps = (5, 5) if selected.execution_steps == 10 else (1, 1)
    args.overlap_source_sha256 = context["source_manifest"]["sha256"]
    args.uccl_budget_setup = None
    return args


def build_overlap_config(args):
    require_allocation()
    if args.communication_backend == "uccl_ep":
        from profiling.run_pretrain_uccl_budget import configure_before_model, runtime_adapter
        require(os.environ.get("FI_MR_CACHE_MONITOR") == "disabled", "UCCL requires FI_MR_CACHE_MONITOR=disabled")
        args.uccl_budget_setup = configure_before_model(runtime_adapter(), args.communication_sms)
    cache_args = copy.copy(args)
    cache_args.warmup_steps, cache_args.measured_steps = 5, 20
    cache_args.smoke_steps = args.execution_steps
    config = BASE_BUILD_CONFIG(cache_args)
    require(config.model.hidden_size == 2048 and config.model.moe_router_topk == 8,
            "the overlap experiment requires the pinned Qwen model")
    require(config.train.train_iters == 25 and config.train.exit_interval == args.execution_steps,
            "preserve the 25-update cache budget and explicit real-update exit interval")
    require(hasattr(config, "ddp") and hasattr(config.ddp, "delay_wgrad_compute"),
            "pinned DDP config lacks required delay_wgrad_compute field")
    # CommOverlapConfig normally applies this override to both model and DDP.
    # With comm_overlap=None, set the companion explicitly so DDP attaches its
    # accumulation/reduction hooks to TE's delayed backward_dw() execution.
    config.ddp.delay_wgrad_compute = True
    require(config.ddp.delay_wgrad_compute is True, "DDP delay_wgrad_compute must be enabled")
    for key, value in OVERLAP_FLAGS.items():
        require(hasattr(config.model, key), f"pinned model config lacks selected field: {key}")
        setattr(config.model, key, value)
    config.comm_overlap = None
    if args.communication_backend == "uccl_ep":
        config.model.moe_flex_dispatcher_num_sms = args.communication_sms
    config.profiling.use_nsys_profiler = False
    config.profiling.use_pytorch_profiler = False
    config.profiling.nvtx_ranges = args.overlap_capture == "nsys"
    if int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0"))) == 0:
        write_new(args.run_dir / "config/ep_overlap_contract.json", execution_contract(args))
    return config


def comparison_contract(args):
    contract = BASE_COMPARISON_CONTRACT(args)
    contract["run_kind"] = "ep16_overlap_experiment"
    contract["measurement_role"] = "overlap_qualification" if args.execution_steps == 2 else "overlap_routing_comparison"
    contract["baseline"].update(ep_a2a_overlap=True, delay_expert_wgrad=True,
        dispatch_backward_expert_wgrad_overlap=False, ep_overlap_early_attn_memory_release=True,
        bridge_comm_overlap_configured=False, cuda_device_max_connections="32")
    contract["ep_overlap"] = execution_contract(args)
    return contract


def main(argv=None):
    selected, remaining = selected_parser().parse_known_args(sys.argv[1:] if argv is None else argv)
    validate_selection(selected, selected_parser())
    require_allocation()  # Must precede the first import of Torch or Bridge runtime.
    if selected.prepare_inputs:
        require(not remaining, "input preparation accepts only the overlap selection options")
        result = prepare_inputs(selected)
        print(json.dumps({"prepared": True, "job": result["job"], "experiment_protocol": PROTOCOL}), flush=True)
        return
    pretrain_module = importlib.import_module("megatron.bridge.training.pretrain")
    provenance = importlib.import_module("ampa_profile.provenance")
    deep_trace = importlib.import_module("ampa_profile.deep_trace")
    capture_module = importlib.import_module("profiling.ampa_profile.overlap_capture")
    previous = (canonical.parse_args, canonical.build_config, canonical.comparison_contract,
        pretrain_module.pretrain, provenance.make_indexed_run_provenance_callback,
        deep_trace.make_deep_trace_callback)
    state, suppressed_callback = {}, object()

    def parse():
        state["args"] = parse_overlap_args(argv)
        return state["args"]

    def provenance_callback(**kwargs):
        capture = state["args"].overlap_capture
        return previous[4](**{**kwargs, "nsys_trace": capture == "nsys", "torch_trace": capture == "torch"})

    def pretrain(*, config, forward_step_func, callbacks=None):
        callback = capture_module.make_overlap_callback(state["args"])
        preserved = [item for item in (callbacks or []) if item is not suppressed_callback]
        try:
            return previous[3](config=config, forward_step_func=forward_step_func, callbacks=[callback, *preserved])
        finally:
            callback.abort()

    canonical.parse_args, canonical.build_config, canonical.comparison_contract = parse, build_overlap_config, comparison_contract
    pretrain_module.pretrain = pretrain
    provenance.make_indexed_run_provenance_callback = provenance_callback
    deep_trace.make_deep_trace_callback = lambda **_kwargs: suppressed_callback
    try:
        canonical.main()
    finally:
        (canonical.parse_args, canonical.build_config, canonical.comparison_contract,
         pretrain_module.pretrain, provenance.make_indexed_run_provenance_callback,
         deep_trace.make_deep_trace_callback) = previous


if __name__ == "__main__":
    main()
