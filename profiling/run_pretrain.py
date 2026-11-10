#!/usr/bin/env python3
"""Run AMPA's fixed-topology continued-pretraining measurement baseline.

This deliberately uses Bridge's public ``pretrain`` entry point and only adds
an optional lifecycle callback. It is not a replacement trainer or dispatcher.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

from tools.c0_validation import C0ValidationError, validate_c0_identity_manifest
from tools.bridge_indexed_consumer import apply_bridge_indexed_consumer, expected_bridge_indexed_consumer
from tools.qwen_workload import workload


_CANONICAL_C0_NAME = "qwen3-30b-a3b-base-c0"
_BASELINE_PARALLELISM = {"tp": 1, "pp": 1, "cp": 1, "ep": 8, "etp": 1}
_MODEL_MANIFEST_PATH = Path(__file__).resolve().parents[1] / "configs" / "model" / "qwen3_30b_a3b_base.json"
_CP2_PARENT_MANIFEST_PATH = Path(__file__).resolve().parents[1] / "dataset" / "manifests" / "qwen3-30b-a3b-base-realistic-mix-v2-validated.json"


def _runtime_workload(topology_variant: str) -> tuple[int, int, int]:
    """Return CP size, distinct-data DP size, and GA for an explicit variant."""
    spec = workload(topology_variant)
    return spec["cp"], spec["dense_dp"], spec["accumulation"]


def _configure_nsys_capture(profiling: object, warmup_steps: int, capture_steps: int) -> None:
    """Use Bridge's existing all-rank CUDA-profiler-API step boundaries."""
    if not capture_steps:
        return
    if profiling is None:
        raise ValueError("Bridge profiling configuration is unavailable")
    profiling.use_nsys_profiler = True
    profiling.profile_step_start = warmup_steps
    profiling.profile_step_end = warmup_steps + capture_steps
    profiling.profile_ranks = list(range(8))
    profiling.nvtx_ranges = True


def _configure_torch_capture(
    profiling: object, warmup_steps: int, capture_steps: int, ranks: tuple[int, ...]
) -> None:
    """Account for Bridge advancing its profiler schedule at loop entry."""
    if not capture_steps:
        return
    if profiling is None:
        raise ValueError("Bridge profiling configuration is unavailable")
    profiling.use_pytorch_profiler = True
    profiling.profile_step_start = warmup_steps + 1
    profiling.profile_step_end = warmup_steps + 1 + capture_steps
    profiling.profile_ranks = list(ranks)
    profiling.pytorch_profiler_collect_shapes = False
    profiling.pytorch_profiler_collect_callstack = False
    profiling.pytorch_profiler_collect_chakra = False


def parse_args() -> argparse.Namespace:
    """Parse only the baseline controls that must remain identical across runs."""
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--pretrained-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--dataset-data-path",
        help="Megatron --data-path style value; required only with --data-mode indexed.",
    )
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        help="Validated JSON manifest for exactly the indexed data-path; required only with --data-mode indexed.",
    )
    parser.add_argument(
        "--data-mode",
        choices=("indexed", "mock"),
        default="indexed",
        help="Use canonical indexed data (default) or the explicitly noncanonical one-step mock bring-up path.",
    )
    parser.add_argument("--hf-model-id", default="Qwen/Qwen3-30B-A3B-Base")
    parser.add_argument("--hf-revision", required=True)
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--global-batch-size", type=int, default=64)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measured-steps", type=int, default=20)
    parser.add_argument("--log-interval", type=int)
    parser.add_argument("--lightweight-profiling", choices=("on", "off"), default="on")
    parser.add_argument(
        "--nsys-capture-steps", type=int, default=0,
        help="Trace-only CP2 diagnostic: capture 1–3 complete postwarmup optimizer steps on all ranks.",
    )
    parser.add_argument(
        "--torch-profile-steps", type=int, default=0,
        help="Trace-only CP2 diagnostic: capture 1–3 complete postwarmup optimizer steps.",
    )
    parser.add_argument(
        "--torch-profile-ranks", default="0",
        help="Comma-separated global ranks for the PyTorch profiler; use 0,1,2,3,4,5,6,7 for the final trace.",
    )
    parser.add_argument("--sample-fingerprints", action="store_true",
                        help="Hash actual CPU token batches for matched diagnostic run identity.")
    parser.add_argument("--torch-profile-all-steps", action="store_true",
                        help="Capture every optimizer step with separate AMPA-owned per-step exports.")
    parser.add_argument("--router-num-groups", type=int, default=None)
    parser.add_argument("--router-group-topk", type=int, default=None)
    parser.add_argument("--routing-audit", action="store_true",
                        help="Validate actual routing maps and dispatcher splits on every step.")
    parser.add_argument("--communication-backend", choices=("alltoall", "uccl_ep"), default="alltoall")
    parser.add_argument("--launch-policy", choices=("singleton", "torchrun_node4"), default="singleton")
    parser.add_argument("--audit-received-experts", action="store_true",
                        help="Record the native ready CPU expert-receive counts for cross-rank reconciliation.")
    parser.add_argument("--uccl-build-dir", type=Path,
                        help="Reviewed UCCL build evidence directory; required only for uccl_ep.")
    parser.add_argument("--topology-variant", choices=("cp1_ep8", "cp2_ep8", "cp1_ep16", "cp2_ep16"), default="cp1_ep8")
    parser.add_argument("--smoke-steps", type=int, choices=(2,),
                        help="Exit after two real updates; retain the full 1600-sample cache budget.")
    parser.add_argument("--audit-num-groups", type=int, default=None,
                        help="Diagnostic partition shared by ordinary and grouped routing.")
    parser.add_argument(
        "--recompute-mode",
        choices=("none", "moe_act"),
        default="none",
        help="Explicit training-workload variant; moe_act checkpoints only grouped-expert activation.",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--dataset-sampler-seed", type=int, default=1234)
    args = parser.parse_args()
    spec = workload(args.topology_variant)
    world = spec["world_size"]
    diagnostic_variant = args.topology_variant in ("cp2_ep8", "cp1_ep16", "cp2_ep16")
    args.completed_steps = args.smoke_steps or args.warmup_steps + args.measured_steps
    if args.audit_num_groups is None:
        args.audit_num_groups = 4 if world == 16 else 8
    if args.audit_num_groups <= 0 or 128 % args.audit_num_groups:
        parser.error("audit groups must divide 128 experts")
    if args.smoke_steps and (world != 16 or not args.routing_audit or not args.sample_fingerprints
                            or not args.torch_profile_all_steps or args.log_interval != 1):
        parser.error("two-update smoke requires the EP16 all-rank profiled/audited workload and per-step logs")
    if args.log_interval is None:
        args.log_interval = 1 if args.data_mode == "mock" else 5
    if args.sequence_length <= 0 or args.micro_batch_size <= 0 or args.global_batch_size <= 0:
        parser.error("batch sizes and sequence length must be positive")
    if args.warmup_steps < 0 or args.measured_steps <= 0:
        parser.error("warmup must be nonnegative and measured steps must be positive")
    if args.log_interval <= 0:
        parser.error("log interval must be positive")
    if args.nsys_capture_steps < 0 or args.nsys_capture_steps > 3:
        parser.error("--nsys-capture-steps must be between 0 and 3")
    if args.torch_profile_steps < 0 or args.torch_profile_steps > 3:
        parser.error("--torch-profile-steps must be between 0 and 3")
    if args.torch_profile_all_steps and (args.torch_profile_steps or args.nsys_capture_steps):
        parser.error("all-step profiling is separate from bounded Bridge capture")
    if (args.router_num_groups is None) != (args.router_group_topk is None):
        parser.error("set both --router-num-groups and --router-group-topk")
    if args.router_num_groups is not None:
        groups, selected = args.router_num_groups, args.router_group_topk
        if (groups <= 0 or selected <= 0 or selected > groups or 128 % groups
                or 8 % selected or 128 // groups < 8 // selected):
            parser.error("group routing requires equal expert groups and enough experts for top-k=8")
        if not diagnostic_variant:
            parser.error("group routing requires a reviewed diagnostic topology")
        if args.routing_audit and groups != args.audit_num_groups:
            parser.error("group routing and diagnostic audit must use the same partition")
    try:
        args.torch_profile_ranks = tuple(int(value) for value in args.torch_profile_ranks.split(","))
    except ValueError:
        parser.error("--torch-profile-ranks must be comma-separated integer ranks")
    if not args.torch_profile_ranks or len(set(args.torch_profile_ranks)) != len(args.torch_profile_ranks):
        parser.error("--torch-profile-ranks must list distinct ranks")
    if any(rank not in range(world) for rank in args.torch_profile_ranks):
        parser.error(f"--torch-profile-ranks must be within 0–{world - 1}")
    if args.torch_profile_all_steps and args.torch_profile_ranks != tuple(range(world)):
        parser.error(f"all-step profiling requires all {world} ranks in global-rank order")
    if args.nsys_capture_steps and args.torch_profile_steps:
        parser.error("Nsight and PyTorch profiler diagnostics are mutually exclusive")
    if args.nsys_capture_steps and (
        args.topology_variant != "cp2_ep8" or args.data_mode != "indexed"
        or args.lightweight_profiling != "off" or args.recompute_mode != "none"
        or args.warmup_steps != 5 or args.measured_steps != 20
    ):
        parser.error("Nsight is a separate profile-OFF CP2 indexed-data diagnostic with 5+20 steps")
    if (args.torch_profile_steps or args.torch_profile_all_steps) and (
        not diagnostic_variant or args.data_mode != "indexed"
        or args.lightweight_profiling != "off" or args.recompute_mode != "none"
        or args.warmup_steps != 5 or args.measured_steps != 20
    ):
        parser.error("PyTorch profiler requires a separate profile-OFF CP2 indexed-data run with 5+20 steps")
    if args.sample_fingerprints and (
        not diagnostic_variant or args.data_mode != "indexed"
        or args.lightweight_profiling != "off" or args.recompute_mode != "none"
        or args.warmup_steps != 5 or args.measured_steps != 20
    ):
        parser.error("sample fingerprints require the reviewed profile-OFF CP2 indexed workload")
    if args.warmup_steps % args.log_interval or args.measured_steps % args.log_interval:
        parser.error("warmup and measured steps must be exact multiples of log interval")
    cp, dense_dp, ga = _runtime_workload(args.topology_variant)
    if args.global_batch_size != args.micro_batch_size * dense_dp * ga:
        parser.error(f"{args.topology_variant} requires GBS = MBS × dense_DP({dense_dp}) × GA({ga})")
    if diagnostic_variant and (
        args.data_mode != "indexed" or args.recompute_mode != "none"
        or args.sequence_length != 4096 or args.micro_batch_size != 1 or args.global_batch_size != 64
    ):
        parser.error("CP2/EP8 requires indexed L4096 MBS1 GBS64 and no recomputation")
    if world == 16 and (args.lightweight_profiling != "off" or not args.routing_audit
                       or not args.sample_fingerprints or args.log_interval != 1):
        parser.error("EP16 requires matched audit/fingerprints and per-step logs, with legacy observer off")
    if args.audit_received_experts and not args.routing_audit:
        parser.error("receive recording requires --routing-audit")
    if args.communication_backend == "uccl_ep":
        if (args.topology_variant != "cp1_ep16" or args.launch_policy != "torchrun_node4"
                or (args.router_num_groups, args.router_group_topk) not in ((None, None), (4, 2))
                or args.audit_num_groups != 4 or not args.audit_received_experts):
            parser.error("uccl_ep requires CP1/EP16, torchrun_node4, ordinary or four-group/select2 routing, audit4 and receive recording")
        if args.uccl_build_dir is None or not args.uccl_build_dir.is_dir():
            parser.error("uccl_ep requires an existing --uccl-build-dir")
    elif args.uccl_build_dir is not None:
        parser.error("--uccl-build-dir is only valid for --communication-backend uccl_ep")
    if args.recompute_mode == "moe_act" and args.lightweight_profiling != "off":
        parser.error("selective moe_act is feasibility-only until the routing observer supports recomputation")
    if not args.pretrained_checkpoint.is_dir():
        parser.error(f"pretrained checkpoint directory does not exist: {args.pretrained_checkpoint}")
    _validate_c0_manifest(args, parser)
    if args.data_mode == "indexed":
        if args.dataset_data_path is None or args.dataset_manifest is None:
            parser.error("--dataset-data-path and --dataset-manifest are required with --data-mode indexed")
        _validate_dataset_manifest(args, parser)
        for prefix in _dataset_prefixes(args.dataset_data_path):
            if not Path(f"{prefix}.idx").is_file():
                parser.error(f"Megatron indexed dataset index is missing: {prefix}.idx")
            if not Path(f"{prefix}.bin").is_file():
                parser.error(f"Megatron indexed dataset data is missing: {prefix}.bin")
    else:
        _validate_mock_bringup(args, parser)
    return args


def _validate_c0_manifest(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Bind every run to the pinned immutable C0 before Bridge loads it."""
    manifest_path = args.pretrained_checkpoint / "ampa_c0_manifest.json"
    if (
        args.pretrained_checkpoint.name != _CANONICAL_C0_NAME
        or args.pretrained_checkpoint.parent.name != "megatron"
        or args.pretrained_checkpoint.parent.parent.name != "checkpoint"
    ):
        parser.error("C0 path must use the canonical checkpoint/megatron/qwen3-30b-a3b-base-c0 location")
    required_paths = (
        args.pretrained_checkpoint.parent.parent,
        args.pretrained_checkpoint.parent,
        args.pretrained_checkpoint,
        manifest_path,
    )
    if any(path.is_symlink() for path in required_paths):
        parser.error("C0 checkpoint and identity manifest paths must not use symlinks")
    if not manifest_path.is_file():
        parser.error(f"C0 manifest is missing: {manifest_path}")
    if not _MODEL_MANIFEST_PATH.is_file() or _MODEL_MANIFEST_PATH.is_symlink():
        parser.error(f"pinned model manifest is unavailable: {_MODEL_MANIFEST_PATH}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        parser.error(f"C0 manifest is not valid JSON: {error}")
    try:
        identity = validate_c0_identity_manifest(
            manifest,
            model_manifest_sha256=_sha256(_MODEL_MANIFEST_PATH),
            model_id=args.hf_model_id,
            revision=args.hf_revision,
        )
    except C0ValidationError as error:
        parser.error(f"C0 identity manifest is invalid: {error}")
    if identity["parallelism"] != _BASELINE_PARALLELISM:
        parser.error("C0 identity manifest parallelism is not the fixed TP1/PP1/CP1/EP8/ETP1 baseline")
    args.c0_identity_manifest_sha256 = _sha256(manifest_path)
    args.c0_hf_snapshot_manifest_sha256 = manifest.get("hf_snapshot_manifest_sha256")


def _validate_mock_bringup(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Keep the data-free C0 smoke distinct from an indexed profiling run."""
    if args.lightweight_profiling != "off":
        parser.error("--data-mode mock is noncanonical and requires --lightweight-profiling off")
    if args.warmup_steps != 0 or args.measured_steps != 1:
        parser.error("--data-mode mock is limited to one optimizer-step bring-up: warmup=0, measured-steps=1")
    if args.dataset_data_path is not None or args.dataset_manifest is not None:
        parser.error("--data-mode mock must not receive indexed dataset inputs")
    if args.recompute_mode != "none":
        parser.error("--data-mode mock must not change the established no-recompute bring-up workload")


def _validate_dataset_manifest(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Require an indexed-data contract rather than accepting an arbitrary existing prefix."""
    try:
        manifest = json.loads(args.dataset_manifest.read_text(encoding="utf-8"))
    except FileNotFoundError:
        parser.error(f"dataset manifest is missing: {args.dataset_manifest}")
    except json.JSONDecodeError as error:
        parser.error(f"dataset manifest is not valid JSON: {error}")
    variant = getattr(args, "topology_variant", "cp1_ep8")
    lifecycle = manifest.get("lifecycle", {})
    if variant in ("cp1_ep16", "cp2_ep16"):
        from tools.ep16_indexed_manifest import validate_runtime_derivative
        try:
            validate_runtime_derivative(manifest, variant)
        except (ValueError, OSError) as error:
            parser.error(f"invalid EP16 runtime dataset derivative: {error}")
    elif variant == "cp2_ep8":
        if (
            manifest.get("variant_id") != "cp2_ep8"
            or lifecycle.get("status") != "validated_for_feasibility_run"
            or lifecycle.get("feasibility_training_allowed") is not True
            or lifecycle.get("canonical_training_allowed") is not False
        ):
            parser.error("CP2/EP8 requires an explicitly validated feasibility-only manifest")
        if not _CP2_PARENT_MANIFEST_PATH.is_file() or _CP2_PARENT_MANIFEST_PATH.is_symlink():
            parser.error("pinned CP1 parent dataset manifest is unavailable")
        parent_bytes = _CP2_PARENT_MANIFEST_PATH.read_bytes()
        if manifest.get("parent_validated_manifest_sha256") != hashlib.sha256(parent_bytes).hexdigest():
            parser.error("CP2 variant is not bound to the pinned validated v2 parent manifest")
        parent = json.loads(parent_bytes)
        if (
            parent.get("lifecycle", {}).get("status") != "validated_for_canonical_run"
            or parent.get("materialized_artifacts") != manifest.get("materialized_artifacts")
            or parent.get("validated_training_data_path") != manifest.get("validated_training_data_path")
            or parent.get("model_and_tokenizer") != manifest.get("model_and_tokenizer")
        ):
            parser.error("CP2 variant changed the independently validated v2 indexed data identity")
    else:
        if lifecycle.get("status") != "validated_for_canonical_run":
            parser.error("dataset manifest is not validated_for_canonical_run; canonical profiling is blocked")
        if lifecycle.get("canonical_training_allowed") is not True:
            parser.error("dataset manifest has not explicitly allowed canonical training")
    model = manifest.get("model_and_tokenizer", {})
    if model.get("model_id") != args.hf_model_id or model.get("revision") != args.hf_revision:
        parser.error("dataset manifest model identity does not match --hf-model-id/--hf-revision")
    if model.get("tokenizer_id") != args.hf_model_id or model.get("tokenizer_revision") != args.hf_revision:
        parser.error("dataset manifest tokenizer identity does not match the pinned model revision")
    if manifest.get("validated_training_data_path") != args.dataset_data_path:
        parser.error("dataset manifest data path does not exactly match --dataset-data-path")
    workload = manifest.get("validated_workload", {})
    _, dense_dp, ga = _runtime_workload(variant)
    expected_workload = {
        "micro_batch_size": args.micro_batch_size,
        "dense_data_parallel_size": dense_dp,
        "gradient_accumulation": ga,
        "global_batch_size": args.global_batch_size,
        "sequence_length": args.sequence_length,
        "warmup_steps": args.warmup_steps,
        "measured_steps": args.measured_steps,
        "sampler_seed": args.dataset_sampler_seed,
    }
    if any(workload.get(key) != value for key, value in expected_workload.items()):
        parser.error("dataset manifest workload identity does not match the requested canonical run")
    artifacts = manifest.get("materialized_artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != 4:
        parser.error("dataset manifest must record four materialized source artifacts")
    required_tokens = (args.warmup_steps + args.measured_steps) * args.global_batch_size * args.sequence_length
    safety_multiple = workload.get("safety_multiple")
    available_tokens = sum(item.get("tokens_including_eod", 0) for item in artifacts if isinstance(item, dict))
    if not isinstance(safety_multiple, (int, float)) or safety_multiple < 1.0:
        parser.error("dataset manifest safety_multiple must be a number at least one")
    if available_tokens < required_tokens * safety_multiple:
        parser.error("indexed dataset token count is below the validated run budget with safety margin")
    artifact_prefixes = {item.get("indexed_prefix") for item in artifacts if isinstance(item, dict)}
    requested_prefixes = {str(prefix) for prefix in _dataset_prefixes(args.dataset_data_path)}
    if artifact_prefixes != requested_prefixes:
        parser.error("dataset manifest indexed artifact prefixes do not match --dataset-data-path")
    for item in artifacts:
        if not isinstance(item, dict) or not isinstance(item.get("idx_sha256"), str) or not isinstance(
            item.get("bin_sha256"), str
        ):
            parser.error("dataset manifest records an artifact without .idx/.bin SHA256 identities")
        prefix = Path(item["indexed_prefix"])
        for suffix, expected_hash in ((".idx", item["idx_sha256"]), (".bin", item["bin_sha256"])):
            artifact_path = Path(f"{prefix}{suffix}")
            if not artifact_path.is_file():
                parser.error(f"dataset manifest artifact is missing: {artifact_path}")
            if _sha256(artifact_path) != expected_hash:
                parser.error(f"dataset manifest SHA256 does not match artifact: {artifact_path}")
    _validate_dataset_cache(manifest, expected_workload, parser)
    cache = manifest["megatron_cache"]
    binding = cache.get("c0_snapshot_binding")
    if not isinstance(binding, dict):
        parser.error("indexed tokenizer/cache lacks a C0 snapshot binding")
    if binding.get("c0_identity_manifest_sha256") != args.c0_identity_manifest_sha256:
        parser.error("indexed tokenizer/cache is not bound to the selected C0 manifest")
    snapshot = cache.get("pinned_hf_snapshot")
    if not isinstance(snapshot, dict):
        parser.error("indexed tokenizer/cache lacks a pinned local HF snapshot")
    snapshot_hash = snapshot.get("manifest_sha256")
    if (
        not isinstance(args.c0_hf_snapshot_manifest_sha256, str)
        or snapshot_hash != args.c0_hf_snapshot_manifest_sha256
        or binding.get("hf_snapshot_manifest_sha256") != snapshot_hash
    ):
        parser.error("indexed tokenizer snapshot identity differs from selected C0")
    try:
        expected_bridge_indexed_consumer(manifest)
    except ValueError as error:
        parser.error(f"indexed Bridge consumer contract is invalid: {error}")
    args.indexed_manifest = manifest


def _validate_dataset_cache(
    manifest: dict[str, object], expected_workload: dict[str, int], parser: argparse.ArgumentParser
) -> None:
    """Ensure normal Megatron index caches were built off the measured run's path.

    Building these caches in the trainer would add rank-dependent startup work
    and can create files adjacent to immutable indexed components.  The data
    preparation stage owns that work; training merely validates and consumes it.
    """
    cache = manifest.get("megatron_cache")
    if not isinstance(cache, dict) or cache.get("status") != "validated":
        parser.error("dataset manifest has no validated offline Megatron cache")
    if cache.get("workload_identity") != expected_workload:
        parser.error("dataset cache workload identity does not match the requested canonical run")
    cache_dir = cache.get("path")
    sequences_path = cache.get("per_dataset_sequences_path")
    sequences_hash = cache.get("per_dataset_sequences_sha256")
    if not isinstance(cache_dir, str) or not Path(cache_dir).is_dir():
        parser.error("dataset cache directory is missing")
    if not isinstance(sequences_path, str) or not isinstance(sequences_hash, str):
        parser.error("dataset cache does not identify per-dataset sequence metadata")
    if not Path(sequences_path).is_file() or _sha256(Path(sequences_path)) != sequences_hash:
        parser.error("dataset cache per-dataset sequence metadata is missing or has a SHA256 mismatch")
    cache_artifacts = cache.get("artifacts")
    if not isinstance(cache_artifacts, list) or not cache_artifacts:
        parser.error("dataset cache does not list hashed cache artifacts")
    for artifact in cache_artifacts:
        if not isinstance(artifact, dict):
            parser.error("dataset cache artifact entry is invalid")
        path = artifact.get("path")
        digest = artifact.get("sha256")
        if not isinstance(path, str) or not isinstance(digest, str) or not Path(path).is_file():
            parser.error("dataset cache artifact is missing")
        if _sha256(Path(path)) != digest:
            parser.error(f"dataset cache artifact SHA256 does not match: {path}")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dataset_prefixes(data_path: str) -> list[Path]:
    """Extract paths from Megatron's accepted flat single/multi-blend notation."""
    tokens = data_path.split()
    if not tokens:
        raise ValueError("dataset data path is empty")
    try:
        weighted = len(tokens) % 2 == 0 and all(float(tokens[index]) > 0 for index in range(0, len(tokens), 2))
    except ValueError:
        weighted = False
    paths = tokens[1::2] if weighted else tokens
    return [Path(path) for path in paths]


def build_config(args: argparse.Namespace):
    """Create a clean Bridge configuration for the intentionally boring baseline."""
    from transformers import AutoConfig

    from megatron.bridge import AutoBridge
    from megatron.bridge.recipes.common import _pretrain_common
    from megatron.bridge.training.mixed_precision import bf16_mixed

    config = _pretrain_common()
    # C0, not Hugging Face, supplies the training weights.  ``from_hf_pretrained``
    # creates a lazy HF model wrapper whose eventual model access would load the
    # entire 30B checkpoint on every rank.  Configuration-only construction
    # preserves the architecture mapping without introducing a second weight
    # source in the canonical training path.
    hf_config = AutoConfig.from_pretrained(args.hf_model_id, revision=args.hf_revision)
    hf_config.name_or_path = args.hf_model_id
    bridge = AutoBridge.from_hf_config(hf_config)
    bridge.hf_model_id = args.hf_model_id
    bridge.hf_model_revision = args.hf_revision
    config.model = bridge.to_megatron_provider(load_weights=False)
    config.tokenizer.tokenizer_model = args.hf_model_id
    config.tokenizer.hf_tokenizer_kwargs = {"revision": args.hf_revision}

    config.model.tensor_model_parallel_size = 1
    config.model.pipeline_model_parallel_size = 1
    cp, _, _ = _runtime_workload(getattr(args, "topology_variant", "cp1_ep8"))
    config.model.context_parallel_size = cp
    config.model.virtual_pipeline_model_parallel_size = None
    config.model.expert_model_parallel_size = workload(args.topology_variant)["ep"]
    config.model.expert_tensor_parallel_size = 1
    config.model.sequence_parallel = False
    config.model.seq_length = args.sequence_length
    uccl_ep = getattr(args, "communication_backend", "alltoall") == "uccl_ep"
    config.model.moe_token_dispatcher_type = "flex" if uccl_ep else "alltoall"
    config.model.moe_flex_dispatcher_backend = "deepep" if uccl_ep else None
    config.model.moe_enable_deepep = False
    config.model.overlap_moe_expert_parallel_comm = False
    config.model.delay_wgrad_compute = False
    config.model.overlap_dispatch_backward_with_experts_wgrad = False
    config.model.ep_overlap_early_attn_memory_release = False
    config.model.moe_shared_expert_overlap = False
    config.model.moe_shortcut_parallel = False
    config.model.moe_router_force_load_balancing = False
    config.model.moe_router_num_groups = getattr(args, "router_num_groups", None)
    config.model.moe_router_group_topk = getattr(args, "router_group_topk", None)
    config.model.moe_expert_capacity_factor = None
    config.model.moe_router_padding_for_quantization = False
    config.model.moe_per_layer_logging = False
    config.model.cuda_graph_impl = "none"
    config.model.recompute_granularity = "selective" if args.recompute_mode == "moe_act" else None
    config.model.recompute_method = None
    config.model.recompute_num_layers = None
    config.model.recompute_modules = ["moe_act"] if args.recompute_mode == "moe_act" else None
    config.model.moe_layer_recompute = False

    config.mixed_precision = bf16_mixed()
    config.train.train_iters = args.warmup_steps + args.measured_steps
    if getattr(args, "smoke_steps", None):
        config.train.exit_interval = args.smoke_steps
    config.train.global_batch_size = args.global_batch_size
    config.train.micro_batch_size = args.micro_batch_size
    config.validation.eval_interval = 0
    config.validation.eval_iters = 0
    config.dataset.random_seed = args.dataset_sampler_seed
    config.dataset.seq_length = args.sequence_length
    if args.data_mode == "indexed":
        # The CLI flat path is identity-only.  Native fast-cache consumption
        # requires the reviewed train-only per-split blend and local tokenizer.
        apply_bridge_indexed_consumer(config, args.indexed_manifest)
    else:
        # `_pretrain_common` deliberately uses blend=None as Bridge's normal
        # MockGPTDataset path.  Do not attach a data path or cache: this is a
        # one-step C0 bring-up, never a canonical profiling workload.
        config.dataset.data_path = None
        config.dataset.blend = None
        config.dataset.blend_per_split = None
        config.dataset.path_to_cache = None
        config.dataset.fast_cache_load = False
        config.dataset.defer_npy_index_mmap = False
        config.dataset.per_dataset_sequences_path = None
    config.dataset.reset_attention_mask = False
    config.dataset.reset_position_ids = False
    config.optimizer.lr = args.learning_rate
    config.optimizer.min_lr = args.learning_rate
    config.optimizer.adam_beta1 = 0.9
    config.optimizer.adam_beta2 = 0.95
    config.optimizer.adam_eps = 1e-8
    config.optimizer.weight_decay = 0.1
    config.optimizer.clip_grad = 1.0
    config.scheduler.lr_warmup_iters = 0
    config.scheduler.lr_decay_iters = config.train.train_iters
    config.scheduler.lr_decay_style = "constant"
    config.checkpoint.load = None
    config.checkpoint.pretrained_checkpoint = str(args.pretrained_checkpoint)
    config.checkpoint.save = None
    config.logger.log_interval = args.log_interval
    # Bridge starts CUDA capture before step 5 and stops after the last
    # requested complete optimizer step; no AMPA training-loop rewrite.
    _configure_nsys_capture(config.profiling, args.warmup_steps, args.nsys_capture_steps)
    _configure_torch_capture(
        config.profiling, args.warmup_steps, args.torch_profile_steps, args.torch_profile_ranks
    )
    if getattr(args, "torch_profile_all_steps", False):
        config.profiling.use_pytorch_profiler = False
        config.profiling.use_nsys_profiler = False
    config.logger.tensorboard_dir = str(args.run_dir / "logs" / "tensorboard")
    if args.data_mode == "indexed":
        # Bridge already logs the ordinary loss/LR/grad-norm/step scalars on
        # one rank. Keep detailed AMPA routing/traffic artifacts local.
        config.logger.log_timers_to_tensorboard = False
        config.logger.timing_log_level = 0
        config.logger.wandb_entity = "anunay-yadav-epfl"
        config.logger.wandb_project = "moe-profile"
        config.logger.wandb_exp_name = args.run_dir.name
        config.logger.wandb_save_dir = str(args.run_dir / "logs" / "wandb")
        config.logger.tensorboard_log_interval = args.log_interval
        config.logger.log_throughput_to_tensorboard = True
        config.logger.throughput_window_size = 5
    return config


def comparison_contract(args: argparse.Namespace) -> dict[str, object]:
    """Return the run identity fields that must match for profiling-overhead comparisons.

    The profiling switch is intentionally omitted: the contract represents the
    training workload, not the observer being compared.
    """
    cp, dense_dp, ga = _runtime_workload(getattr(args, "topology_variant", "cp1_ep8"))
    ep = workload(getattr(args, "topology_variant", "cp1_ep8"))["ep"]
    backend = getattr(args, "communication_backend", "alltoall")
    launch_policy = getattr(args, "launch_policy", "singleton")
    record_receives = getattr(args, "audit_received_experts", False)
    is_cp2 = cp == 2
    dataset: dict[str, object]
    if args.data_mode == "indexed":
        dataset = {
            "mode": "indexed",
            "canonical_profile_data": not is_cp2,
            "data_path": args.dataset_data_path,
            "manifest_sha256": _sha256(args.dataset_manifest),
            "sampler_seed": args.dataset_sampler_seed,
        }
    else:
        dataset = {
            "mode": "mock",
            "canonical_profile_data": False,
            "reason": "one-step C0 optimizer bring-up before indexed dataset materialization",
            "sampler_seed": args.dataset_sampler_seed,
        }

    return {
        "schema_version": 2,
        "run_kind": "ep16_all_step_experiment" if ep == 16 else "cp2_ep8_feasibility" if is_cp2 else (
            "canonical_indexed_profile" if args.data_mode == "indexed" else "noncanonical_mock_c0_bringup"
        ),
        "measurement_role": "ep16_memory_smoke" if getattr(args, "smoke_steps", None) else (
            "ep16_routing_comparison" if ep == 16 else "selected_cp2_ep8_baseline" if is_cp2 else "legacy_cp1_workload"),
        "claim_scope": {
            "kind": "native_bridge_c0_measurement",
            "hf_numerical_equivalence_claimed": False,
            "canonical_profile_data": args.data_mode == "indexed" and not is_cp2,
        },
        "initialization": {"mode": "pretrained", "checkpoint": str(args.pretrained_checkpoint)},
        "c0_identity_manifest_sha256": args.c0_identity_manifest_sha256,
        "model": {"hf_model_id": args.hf_model_id, "hf_revision": args.hf_revision, "precision": "bf16"},
        "routing": {
            "topk": 8,
            "num_groups": getattr(args, "router_num_groups", None),
            "group_topk": getattr(args, "router_group_topk", None),
        },
        "parallelism": {"tp": 1, "pp": 1, "cp": cp, "ep": ep, "etp": 1, "dense_dp": dense_dp},
        "training": {
            "sequence_length": args.sequence_length,
            "micro_batch_size": args.micro_batch_size,
            "global_batch_size": args.global_batch_size,
            "gradient_accumulation": ga,
            "warmup_steps": args.warmup_steps,
            "measured_steps": args.measured_steps,
            "log_interval": args.log_interval,
            "learning_rate": args.learning_rate,
            "recompute_mode": args.recompute_mode,
            "completed_steps": getattr(args, "completed_steps", args.warmup_steps + args.measured_steps),
        },
        "logging": {
            "wandb_entity": "anunay-yadav-epfl" if args.data_mode == "indexed" else None,
            "wandb_project": "moe-profile" if args.data_mode == "indexed" else None,
            "wandb_mode": os.environ.get("WANDB_MODE", "online") if args.data_mode == "indexed" else None,
            "scalar_interval_steps": args.log_interval,
            "per_layer_moe_metrics": False,
        },
        "dataset": dataset,
        "baseline": {
            "cuda_allocator_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "default"),
            "dispatcher": "flex" if backend == "uccl_ep" else "alltoall",
            "capacity_drop": False,
            "router_padding": False,
            "activation_recomputation": args.recompute_mode == "moe_act",
            "recompute_granularity": "selective" if args.recompute_mode == "moe_act" else None,
            "recompute_modules": ["moe_act"] if args.recompute_mode == "moe_act" else [],
            "cuda_graphs": False,
            "ep_a2a_overlap": False,
            "delay_expert_wgrad": False,
            "dispatch_backward_expert_wgrad_overlap": False,
            "ep_overlap_early_attn_memory_release": False,
            "bridge_comm_overlap_configured": False,
            "dense_dp_overlap_grad_reduce": True,
            "dense_dp_overlap_param_gather": True,
        },
        "sample_fingerprints": bool(getattr(args, "sample_fingerprints", False)),
        "routing_audit": bool(getattr(args, "routing_audit", False)),
        **({"communication_backend": backend, "launch_policy": launch_policy,
            "audit_received_experts": record_receives,
            **({"uccl_build_dir": str(args.uccl_build_dir.resolve())} if backend == "uccl_ep" else {})}
           if backend != "alltoall" or launch_policy != "singleton" or record_receives else {}),
    }


def _completed_uccl_zero_exit(error: BaseException, identity_callback: object) -> bool:
    """Recognize Bridge's explicit zero exit only after UCCL cleanup finished."""
    return (isinstance(error, SystemExit) and type(error.code) is int and error.code == 0
            and getattr(identity_callback, "cleanup_completed", False) is True)


def _abort_uccl_pretrain(args: argparse.Namespace, rank: int, error: BaseException) -> None:
    """Persist host-only failure evidence, then exit before native destructors."""
    try:
        record = {"kind": "ampa_uccl_pretrain_aborted", "valid": False, "rank": rank,
            "pid": os.getpid(), "communication_backend": "uccl_ep", "launch_policy": args.launch_policy,
            "error": f"{type(error).__name__}: {error}",
            "traceback": "".join(traceback.format_exception(type(error), error, error.__traceback__)),
            "entrypoint_cuda_cleanup_attempted": False, "immediate_process_exit_requested": True}
        with (args.run_dir / "raw" / f"uccl_pretrain_aborted_rank-{rank}.json").open("x", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        traceback.print_exception(type(error), error, error.__traceback__, file=sys.stderr)
    except BaseException:
        traceback.print_exc(file=sys.stderr)
    finally:
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.flush()
            except BaseException:
                pass
        os._exit(1)


def main() -> None:
    """Run the normal Bridge trainer with optional AMPA observation."""
    args = parse_args()
    if not args.run_dir.is_dir():
        raise FileNotFoundError(
            f"run directory must be created once by the Slurm wrapper before ranks start: {args.run_dir}"
        )

    config = build_config(args)
    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))
    if rank == 0:
        config.to_yaml(str(args.run_dir / "config" / "requested.yaml"))
        (args.run_dir / "config" / "comparison_contract.json").write_text(
            json.dumps(comparison_contract(args), indent=2, sort_keys=True), encoding="utf-8"
        )
        (args.run_dir / "config" / "all_step_capture.json").write_text(json.dumps({
            "enabled": args.torch_profile_all_steps,
            "steps": list(range(args.completed_steps)),
            "ranks": list(args.torch_profile_ranks),
            "owner": "AMPA per-step callback",
            "record_shapes": False, "with_stack": False,
        }, indent=2, sort_keys=True), encoding="utf-8")

    from megatron.bridge.training.gpt_step import forward_step
    from megatron.bridge.training.pretrain import pretrain
    from megatron.bridge.training.config import apply_environment_variables

    apply_environment_variables(config)
    callbacks = []
    uccl_identity_callback = None
    if args.data_mode == "mock":
        from ampa_profile.c0_mock_sanity import make_native_c0_mock_sanity_callback

        callbacks.append(
            make_native_c0_mock_sanity_callback(
                run_dir=args.run_dir,
                c0_identity_manifest_sha256=args.c0_identity_manifest_sha256,
                hf_model_id=args.hf_model_id,
                hf_revision=args.hf_revision,
            )
        )
    else:
        from ampa_profile.provenance import make_indexed_run_provenance_callback

        profiling_callback = None
        if args.lightweight_profiling == "on":
            from ampa_profile.callback import AMPAProfilingCallback

            profiling_callback = AMPAProfilingCallback(
                run_dir=args.run_dir,
                profile_start_step=args.warmup_steps,
                profile_stop_step=args.warmup_steps + args.measured_steps,
                topology_variant=args.topology_variant,
                expected_calls_per_layer=_runtime_workload(args.topology_variant)[2],
                expected_tokens_per_call=(
                    args.micro_batch_size * args.sequence_length // _runtime_workload(args.topology_variant)[0]
                ),
            )
            callbacks.append(profiling_callback)
        if args.torch_profile_steps or args.torch_profile_all_steps:
            from ampa_profile.deep_trace import make_deep_trace_callback

            callbacks.append(make_deep_trace_callback(
                run_dir=args.run_dir,
                start_step=0 if args.torch_profile_all_steps else args.warmup_steps,
                capture_steps=(args.completed_steps
                               if args.torch_profile_all_steps else args.torch_profile_steps),
                selected_ranks=args.torch_profile_ranks,
                accumulation=_runtime_workload(args.topology_variant)[2],
                per_step_export=args.torch_profile_all_steps,
                expected_world_size=workload(args.topology_variant)["world_size"],
                communication_backend=args.communication_backend,
            ))
        if args.routing_audit:
            from ampa_profile.routing_audit import make_routing_audit_callback

            callbacks.append(make_routing_audit_callback(
                run_dir=args.run_dir,
                expected_steps=args.completed_steps,
                num_groups=args.router_num_groups, group_topk=args.router_group_topk,
                expected_world_size=workload(args.topology_variant)["world_size"],
                accumulation=workload(args.topology_variant)["accumulation"],
                token_positions=workload(args.topology_variant)["token_positions"],
                diagnostic_num_groups=args.audit_num_groups,
                communication_backend=args.communication_backend,
                record_receives=args.audit_received_experts,
            ))
        if args.sample_fingerprints:
            from ampa_profile.sample_fingerprint import make_sample_fingerprint_callback

            callbacks.append(make_sample_fingerprint_callback(
                run_dir=args.run_dir,
                expected_steps=args.completed_steps,
                accumulation=_runtime_workload(args.topology_variant)[2],
                expected_world_size=workload(args.topology_variant)["world_size"],
            ))
        if args.communication_backend == "uccl_ep":
            from ampa_profile.uccl_identity import make_uccl_identity_callback

            uccl_identity_callback = make_uccl_identity_callback(run_dir=args.run_dir, build_dir=args.uccl_build_dir)
            callbacks.append(uccl_identity_callback)
        # Bridge fires callbacks in registration order.  The observer (when
        # enabled) must attach before common provenance records its state.
        callbacks.append(
            make_indexed_run_provenance_callback(
                run_dir=args.run_dir,
                lightweight_profiling=args.lightweight_profiling,
                expected_recompute_mode=args.recompute_mode,
                topology_variant=args.topology_variant,
                nsys_trace=bool(args.nsys_capture_steps),
                torch_trace=bool(args.torch_profile_steps or args.torch_profile_all_steps),
                sample_fingerprints=args.sample_fingerprints,
                dataset_manifest=args.indexed_manifest,
                launch_policy=args.launch_policy,
                router_observer_attached=(
                    (lambda _context: profiling_callback.observer_attached)
                    if profiling_callback is not None
                    else None
                ),
            )
        )
    try:
        pretrain(config=config, forward_step_func=forward_step, callbacks=callbacks or None)
    except BaseException as error:
        if args.communication_backend == "uccl_ep":
            if _completed_uccl_zero_exit(error, uccl_identity_callback):
                raise
            # Keep config/callback references alive through immediate exit. A
            # transport failure may leave native kernels waiting for peers.
            _abort_uccl_pretrain(args, rank, error)
        raise


if __name__ == "__main__":
    main()
