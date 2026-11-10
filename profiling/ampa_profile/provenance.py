"""Common, non-step provenance for matched indexed profiling runs.

This module deliberately has no dependency on the routing observer.  The
callback it creates performs one all-rank placement/topology validation before
Bridge enters its training loop, then rank zero writes provenance artifacts.
It never registers a per-step hook or synchronizes CUDA.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
import uuid
from collections import Counter
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from tools.qwen_workload import workload


_GPU_UUID_SOURCE = "cuda_driver.cuDeviceGetUuid(cuda_runtime.cudaGetDevice())"
_AMPA_RUNTIME_SOURCE_PATHS = (
    "profiling/run_pretrain.py",
    "profiling/ampa_profile/__init__.py",
    "profiling/ampa_profile/callback.py",
    "profiling/ampa_profile/deep_trace.py",
    "profiling/ampa_profile/uccl_trace.py",
    "profiling/ampa_profile/uccl_identity.py",
    "profiling/ampa_profile/routing_audit.py",
    "profiling/ampa_profile/sample_fingerprint.py",
    "profiling/ampa_profile/observer.py",
    "profiling/ampa_profile/provenance.py",
    "jobs/training/qwen3_30b_a3b_base_ep8_profile.sbatch",
    "jobs/training/cp2_torchprof_pair.sbatch",
    "jobs/training/cp2_all_steps_group_routing.sbatch",
    "jobs/training/qwen3_all_steps_profile.sbatch",
    "jobs/training/qwen3_ep16_all_steps.sbatch",
    "jobs/training/ep16_all_steps_group_routing.sbatch",
    "jobs/training/uccl_ep_comparison.sbatch",
    "jobs/training/qwen3_uccl_ep_worker.sh",
    "jobs/training/qwen3_30b_a3b_base_cp2_paired_overhead.sbatch",
    "analysis/extract_step_timing.py",
    "analysis/aggregate_run.py",
    "analysis/analyze_all_step_trace.py",
    "analysis/inspect_uccl_trace.py",
    "analysis/validate_ep16_received_routes.py",
    "analysis/accept_uccl_pilot.py",
    "analysis/check_ep16_smoke.py",
    "analysis/summarize_training_evidence.py",
    "analysis/compare_all_step_profiles.py",
    "analysis/compare_deep_trace_overhead.py",
    "analysis/compare_profile_overhead.py",
    "analysis/validate_torch_trace.py",
    "analysis/aggregate_deep_trace.py",
    "tools/c0_validation.py",
    "tools/bridge_indexed_consumer.py",
    "tools/inspect_gpu_identity.py",
    "tools/uccl_ep_log_proof.py",
    "tools/uccl_ep_source_identity.py",
    "tools/uccl_ep_validate_profiles.py",
    "jobs/analysis/uccl_ep_validate_profiles.sbatch",
    "tools/megatron_cache_contract.py",
    "tools/megatron_cache_stage.py",
    "tools/qwen_workload.py",
    "tools/ep16_indexed_manifest.py",
    "tools/ep16_reviewed_inputs.py",
    "tools/ep16_recovery_placement.py",
    "analysis/compare_ep16_recovery.py",
    "jobs/training/ep16_all_steps_reviewed.sbatch",
)
_EXPECTED_RUNTIME_TOPOLOGY = {
    "world_size": 8,
    "tp": 1,
    "pp": 1,
    "cp": 1,
    "ep": 8,
    "etp": 1,
    "dense_dp": 8,
    "expert_dp": 1,
}
_CP2_RUNTIME_TOPOLOGY = {
    **_EXPECTED_RUNTIME_TOPOLOGY,
    "cp": 2,
    "dense_dp": 4,
    "dense_dp_cp": 8,
}
_EP16_RUNTIME_TOPOLOGIES = {
    f"cp{cp}_ep16": {
        **{key: value for key, value in workload(f"cp{cp}_ep16").items()
           if key in ("world_size", "cp", "ep", "dense_dp")},
        "tp": 1, "pp": 1, "etp": 1, "expert_dp": 1,
        **({"dense_dp_cp": 16} if cp == 2 else {}),
    }
    for cp in (1, 2)
}


class ProvenanceInvariantError(RuntimeError):
    """Raised rather than writing provenance for the wrong runtime."""


def _validate_launch_policy(launch_policy: str) -> None:
    if launch_policy not in ("singleton", "torchrun_node4"):
        raise ProvenanceInvariantError(f"unsupported GPU launch policy: {launch_policy!r}")


def build_observation_mode(
    *, lightweight_profiling: str, router_observer_attached: bool,
    nsys_trace: bool = False, torch_trace: bool = False,
    sample_fingerprints: bool = False,
) -> dict[str, object]:
    """Return an explicit observer-state record for an indexed run.

    ``router_observer_attached`` must come from the caller that owns observer
    installation.  This prevents a profile-on artifact from asserting that a
    wrapper exists merely because that mode was requested.
    """

    if lightweight_profiling not in {"off", "on"}:
        raise ProvenanceInvariantError(
            f"lightweight profiling mode must be 'off' or 'on', got {lightweight_profiling!r}"
        )
    if lightweight_profiling == "on" and not router_observer_attached:
        raise ProvenanceInvariantError("profile-on provenance requires an attached routing observer")
    if lightweight_profiling == "off" and router_observer_attached:
        raise ProvenanceInvariantError("profile-off provenance forbids an attached routing observer")
    if nsys_trace and lightweight_profiling != "off":
        raise ProvenanceInvariantError("Nsight diagnostic requires lightweight profiling off")
    if torch_trace and lightweight_profiling != "off":
        raise ProvenanceInvariantError("PyTorch diagnostic requires lightweight profiling off")
    if nsys_trace and torch_trace:
        raise ProvenanceInvariantError("only one deep tracing mode may be enabled")
    return {
        "schema_version": 1,
        "data_mode": "indexed",
        "lightweight_profiling": lightweight_profiling,
        "nsys_trace": nsys_trace,
        "torch_trace": torch_trace,
        "sample_fingerprints": sample_fingerprints,
        "router_observer_attached": router_observer_attached,
        "route_wrapper_installed": router_observer_attached,
        "dispatcher_preprocess_wrapper_installed": router_observer_attached,
        "emission": "after_runtime_topology_and_placement_validation",
    }


def validate_runtime_topology(
    runtime_topology: Mapping[str, object],
    *,
    expected: Mapping[str, int] = _EXPECTED_RUNTIME_TOPOLOGY,
) -> dict[str, int]:
    """Fail closed unless actual Core process groups match the reviewed workload."""

    normalized: dict[str, int] = {}
    for key, expected_value in expected.items():
        value = runtime_topology.get(key)
        if not isinstance(value, int):
            raise ProvenanceInvariantError(f"runtime topology {key!r} is not an integer: {value!r}")
        normalized[key] = value
        if value != expected_value:
            raise ProvenanceInvariantError(
                f"runtime topology {key!r} is {value}, expected {expected_value}"
            )
    return normalized


def validate_recomputation_runtime(
    model_config: Any, model_chunks: list[Any], *, expected_mode: str,
    strict_no_recompute: bool = False,
) -> dict[str, object]:
    """Prove the selected memory variant from constructed modules, not CLI intent.

    This is a one-time CPU metadata inspection. It never calls a module or
    changes the forward/backward path.
    """

    if expected_mode == "none":
        if getattr(model_config, "recompute_granularity", None) is not None:
            raise ProvenanceInvariantError("no-recompute workload has activation recomputation enabled")
        if strict_no_recompute:
            if (
                getattr(model_config, "recompute_modules", None) not in (None, ["core_attn"])
                or getattr(model_config, "recompute_method", None) is not None
                or getattr(model_config, "recompute_num_layers", None) is not None
                or getattr(model_config, "moe_layer_recompute", None) is not False
            ):
                raise ProvenanceInvariantError("CP2 no-recompute configuration has checkpoint settings enabled")
            return {
                "mode": "none",
                "verified_moe_layers": 48,
                "verified_attention_layers": 48,
                "attention_checkpoint_layers": 0,
                "moe_checkpoint_layers": 0,
                "grouped_expert_activation_checkpoint_layers": 0,
                "local_expert_indices": _local_expert_indices(model_chunks),
            }
        return {"mode": "none"}
    if expected_mode != "moe_act":
        raise ProvenanceInvariantError(f"unsupported activation recomputation mode: {expected_mode!r}")
    if (
        getattr(model_config, "recompute_granularity", None) != "selective"
        or getattr(model_config, "recompute_modules", None) != ["moe_act"]
        or getattr(model_config, "moe_grouped_gemm", None) is not True
        or getattr(model_config, "moe_layer_recompute", None) is not False
        or getattr(model_config, "bf16", None) is not True
        or bool(getattr(model_config, "fp8", None))
        or bool(getattr(model_config, "fp4", None))
        or getattr(model_config, "use_transformer_engine_op_fuser", None) is not False
        or getattr(model_config, "moe_token_dispatcher_type", None) != "alltoall"
        or getattr(model_config, "num_moe_experts", None) != 128
        or getattr(model_config, "recompute_method", None) is not None
        or getattr(model_config, "recompute_num_layers", None) is not None
        or getattr(model_config, "cuda_graph_impl", None) != "none"
    ):
        raise ProvenanceInvariantError("selective moe_act configuration is not the approved exact variant")
    layer_count = getattr(model_config, "num_layers", None)
    if layer_count != 48 or getattr(model_config, "moe_layer_freq", None) != 1:
        raise ProvenanceInvariantError("selective moe_act requires all 48 pinned Qwen layers to be MoE")

    observed_layers: set[int] = set()
    attention_layers: set[int] = set()
    attention_flagged_modules = 0
    for chunk in model_chunks:
        for _, module in chunk.named_modules():
            if hasattr(module, "checkpoint_core_attention"):
                attention_flagged_modules += 1
                if module.checkpoint_core_attention is not False:
                    raise ProvenanceInvariantError("attention recomputation was enabled in moe_act workload")
                if type(module).__name__ == "SelfAttention":
                    layer_number = getattr(module, "layer_number", None)
                    if not isinstance(layer_number, int) or layer_number < 1 or layer_number - 1 in attention_layers:
                        raise ProvenanceInvariantError("selective moe_act has invalid or duplicate attention layer identity")
                    attention_layers.add(layer_number - 1)
            if not (hasattr(module, "local_expert_indices") and hasattr(module, "token_dispatcher")):
                continue
            layer_number = getattr(module, "layer_number", None)
            if not isinstance(layer_number, int) or layer_number < 1 or layer_number - 1 in observed_layers:
                raise ProvenanceInvariantError("selective moe_act has an invalid or duplicate MoE layer identity")
            observed_layers.add(layer_number - 1)
            experts = getattr(module, "experts", None)
            if (
                type(experts).__name__ != "TEGroupedMLP"
                or getattr(experts, "activation_recompute", None) is not True
                or getattr(experts, "_with_fused_impl", None) is not False
                or getattr(module, "moe_layer_recompute", None) is not False
                or getattr(module, "shared_experts_recompute", None) is not False
                or len(module.local_expert_indices) != 16
                or type(module.token_dispatcher).__name__ != "MoEAlltoAllTokenDispatcher"
            ):
                raise ProvenanceInvariantError(
                    f"layer {layer_number - 1} does not use isolated TEGroupedMLP moe_act recomputation"
                )
    # MCore sets the flag on TransformerBlock too. Check every flag, but count
    # only the pinned Qwen SelfAttention modules by distinct layer identity.
    if observed_layers != set(range(layer_count)) or attention_layers != set(range(layer_count)):
        raise ProvenanceInvariantError(
            f"selective moe_act found {len(observed_layers)} MoE and {len(attention_layers)} attention layers; "
            f"expected {layer_count} of each"
        )
    return {
        "mode": "moe_act",
        "granularity": "selective",
        "modules": ["moe_act"],
        "moe_layers": layer_count,
        "grouped_expert_activation_checkpoint_layers": layer_count,
        "full_moe_checkpoint_layers": 0,
        "attention_checkpoint_layers": 0,
        "attention_flagged_modules": attention_flagged_modules,
    }


def validate_rank_placement(
    rank_placement: list[Mapping[str, object]], *, expected_world_size: int = 8,
    launch_policy: str = "singleton",
) -> list[dict[str, object]]:
    """Validate actual placement on the requested four-GPU nodes."""

    _validate_launch_policy(launch_policy)
    if type(expected_world_size) is not int or expected_world_size not in (8, 16):
        raise ProvenanceInvariantError("only reviewed eight- or sixteen-rank placement is supported")

    if len(rank_placement) != expected_world_size:
        raise ProvenanceInvariantError(
            f"expected {expected_world_size} placement records, got {len(rank_placement)}"
        )
    normalized = [dict(record) for record in rank_placement]
    ranks = [record.get("rank") for record in normalized]
    if any(type(rank) is not int for rank in ranks) or sorted(ranks) != list(range(expected_world_size)):
        raise ProvenanceInvariantError(
            f"runtime ranks are {sorted(ranks, key=str)}, expected {list(range(expected_world_size))}"
        )
    for record in normalized:
        if record.get("launch_policy", "singleton") != launch_policy:
            raise ProvenanceInvariantError(f"rank {record['rank']} has a different GPU launch policy")
        expected_count = 4 if launch_policy == "torchrun_node4" else 1
        count = record.get("cuda_visible_device_count")
        if type(count) is not int or count != expected_count:
            raise ProvenanceInvariantError(
                f"rank {record['rank']} has cuda_visible_device_count="
                f"{count!r}, expected {expected_count}"
            )
        if launch_policy == "torchrun_node4":
            local_rank, local_world = record.get("local_rank"), record.get("local_world_size")
            current = record.get("cuda_current_device")
            affinity = record.get("cpu_affinity")
            if (type(local_rank) is not int or local_rank not in range(4)
                    or type(local_world) is not int or local_world != 4
                    or type(current) is not int or current != local_rank):
                raise ProvenanceInvariantError(f"rank {record['rank']} has invalid torchrun CUDA/local rank metadata")
            if (not isinstance(affinity, list) or not affinity
                    or any(type(cpu) is not int or cpu < 0 for cpu in affinity)
                    or len(set(affinity)) != len(affinity)):
                raise ProvenanceInvariantError(f"rank {record['rank']} lacks valid worker CPU affinity")
            visibility = record.get("cuda_visible_devices")
            if visibility is not None and not isinstance(visibility, str):
                raise ProvenanceInvariantError(f"rank {record['rank']} has malformed CUDA visibility metadata")
        elif "cuda_current_device" in record:
            current = record["cuda_current_device"]
            if type(current) is not int or current != 0:
                raise ProvenanceInvariantError(f"rank {record['rank']} singleton CUDA ordinal is not zero")
        if record.get("physical_gpu_identity_source") != _GPU_UUID_SOURCE:
            raise ProvenanceInvariantError(
                f"rank {record['rank']} did not report the task-selected CUDA-driver GPU identity"
            )
        gpu_uuid = record.get("physical_gpu_uuid")
        if not isinstance(gpu_uuid, str) or not gpu_uuid.startswith("GPU-"):
            raise ProvenanceInvariantError(f"rank {record['rank']} has no valid physical GPU UUID")
        try:
            uuid.UUID(gpu_uuid.removeprefix("GPU-"))
        except ValueError as error:
            raise ProvenanceInvariantError(
                f"rank {record['rank']} has malformed physical GPU UUID {gpu_uuid!r}"
            ) from error
        for key in ("hostname", "slurm_node_id", "slurm_local_id"):
            if not isinstance(record.get(key), str) or not record[key]:
                raise ProvenanceInvariantError(f"rank {record['rank']} lacks non-empty {key}")

    gpu_uuids = [str(record["physical_gpu_uuid"]) for record in normalized]
    if len(set(gpu_uuids)) != expected_world_size:
        raise ProvenanceInvariantError("physical GPU UUIDs are not unique across the ranks")
    host_counts = Counter(str(record["hostname"]) for record in normalized)
    expected_nodes = expected_world_size // 4
    node_label = "two" if expected_nodes == 2 else "four"
    if len(host_counts) != expected_nodes or sorted(host_counts.values()) != [4] * expected_nodes:
        raise ProvenanceInvariantError(
            f"expected {node_label} hosts with four ranks each, observed {dict(host_counts)}"
        )
    node_ids = {str(record["slurm_node_id"]) for record in normalized}
    if len(node_ids) != expected_nodes:
        raise ProvenanceInvariantError(f"expected {node_label} Slurm node IDs, observed {sorted(node_ids)}")
    for hostname in host_counts:
        host_records = [record for record in normalized if record["hostname"] == hostname]
        host_node_ids = {str(record["slurm_node_id"]) for record in host_records}
        if len(host_node_ids) != 1:
            raise ProvenanceInvariantError(
                f"hostname {hostname!r} maps to multiple Slurm node IDs: {sorted(host_node_ids)}"
            )
        if launch_policy == "torchrun_node4":
            local_ids = sorted(record["local_rank"] for record in host_records)
            if local_ids != list(range(4)):
                raise ProvenanceInvariantError(f"hostname {hostname!r} has duplicate or missing torchrun local ranks")
            if any(record["slurm_local_id"] != "0" for record in host_records):
                raise ProvenanceInvariantError(f"hostname {hostname!r} requires one Slurm parent task with local ID zero")
            if len({record.get("cuda_visible_devices") for record in host_records}) != 1:
                raise ProvenanceInvariantError(f"hostname {hostname!r} has inconsistent worker CUDA visibility")
        else:
            local_ids = sorted(str(record["slurm_local_id"]) for record in host_records)
        if launch_policy == "singleton" and local_ids != ["0", "1", "2", "3"]:
            raise ProvenanceInvariantError(
                f"hostname {hostname!r} has local ranks {local_ids}, expected ['0', '1', '2', '3']"
            )
    return sorted(normalized, key=lambda record: int(record["rank"]))


def validate_cp2_groups(placement: list[Mapping[str, object]]) -> None:
    """Prove CP peers are co-located and EP still spans both physical nodes."""
    validate_runtime_groups(placement, expected_world_size=8, cp=2)


def validate_runtime_groups(
    placement: list[Mapping[str, object]], *, expected_world_size: int, cp: int,
) -> None:
    """Prove actual CP/DP partitions and balanced ownership in global EP."""
    if expected_world_size not in (8, 16) or cp not in (1, 2):
        raise ProvenanceInvariantError("unsupported runtime process-group workload")
    by_rank = {int(record["rank"]): record for record in placement}
    expected_all = list(range(expected_world_size))
    if len(placement) != expected_world_size or sorted(by_rank) != expected_all:
        raise ProvenanceInvariantError("process-group provenance lacks distinct runtime ranks")
    cp_groups: set[tuple[int, ...]] = set()
    dp_groups: set[tuple[int, ...]] = set()
    experts_per_rank = 128 // expected_world_size
    for rank, record in by_rank.items():
        groups = record.get("process_groups")
        if not isinstance(groups, Mapping):
            raise ProvenanceInvariantError(f"rank {rank} lacks CP{cp} process-group provenance")
        for name, expected_size in (("cp", cp), ("dense_dp", expected_world_size // cp),
                                    ("dense_dp_cp", expected_world_size), ("ep", expected_world_size)):
            ranks = groups.get(name)
            if not isinstance(ranks, list) or len(ranks) != expected_size or len(set(ranks)) != expected_size:
                raise ProvenanceInvariantError(f"rank {rank} has invalid {name} group ranks: {ranks!r}")
            if rank not in ranks or any(peer not in by_rank for peer in ranks):
                raise ProvenanceInvariantError(f"rank {rank} is not in a valid {name} process group")
        if sorted(groups["ep"]) != expected_all or sorted(groups["dense_dp_cp"]) != expected_all:
            raise ProvenanceInvariantError(f"CP{cp} EP and dense DP×CP must each span all ranks")
        cp_peer_hosts = {by_rank[peer]["hostname"] for peer in groups["cp"]}
        if len(cp_peer_hosts) != 1:
            raise ProvenanceInvariantError(f"CP{cp} peers cross physical nodes unexpectedly")
        for name in ("cp", "dense_dp", "dense_dp_cp", "ep"):
            for peer in groups[name]:
                peer_groups = by_rank[peer].get("process_groups")
                if not isinstance(peer_groups, Mapping) or peer_groups.get(name) != groups[name]:
                    raise ProvenanceInvariantError(f"rank {rank} has inconsistent {name} group membership")
        cp_groups.add(tuple(sorted(groups["cp"])))
        dp_groups.add(tuple(sorted(groups["dense_dp"])))
        local_experts = record.get("local_expert_indices")
        if (not isinstance(local_experts, list) or len(local_experts) != experts_per_rank
                or len(set(local_experts)) != experts_per_rank):
            raise ProvenanceInvariantError(f"rank {rank} lacks {experts_per_rank} distinct locally owned expert IDs")
    if (len(cp_groups) != expected_world_size // cp
            or {rank for group in cp_groups for rank in group} != set(expected_all)
            or sum(map(len, cp_groups)) != expected_world_size):
        raise ProvenanceInvariantError(f"CP{cp} process groups do not partition the ranks")
    if (len(dp_groups) != cp or sum(map(len, dp_groups)) != expected_world_size
            or {rank for group in dp_groups for rank in group} != set(expected_all)
            or any(len(set(cp_group) & set(dp_group)) != 1
                   for cp_group in cp_groups for dp_group in dp_groups)):
        raise ProvenanceInvariantError("dense DP groups do not form the expected CP/DP grid")
    expert_sets = [set(by_rank[rank]["local_expert_indices"]) for rank in expected_all]
    if set().union(*expert_sets) != set(range(128)) or sum(map(len, expert_sets)) != 128:
        raise ProvenanceInvariantError(f"CP{cp} expert ownership is not a disjoint 128-expert partition")


def build_environment_record(
    *,
    slurm_job_id: str | None,
    pytorch_version: str,
    cuda_version: str | None,
    nccl_version: list[int] | None,
    gpu_name: str,
    container_sqsh_sha256: str | None,
    megatron_lm: Mapping[str, object],
    megatron_bridge: Mapping[str, object],
    ampa_runtime_sources: Mapping[str, object],
    checkpoint: str | None,
    dataset_data_path: object,
    dataset_blend_per_split: object,
    parallelism: Mapping[str, object],
    runtime_topology: Mapping[str, object],
    rank_placement: list[Mapping[str, object]],
    hf_model_id: str | None,
    hf_model_revision: str | None,
    dataset_consumer: Mapping[str, object] | None = None,
    activation_recomputation: Mapping[str, object] | None = None,
    launch_policy: str = "singleton",
) -> dict[str, object]:
    """Build JSON-safe common provenance after all runtime checks have passed."""

    if dataset_blend_per_split is None:
        raise ProvenanceInvariantError("finalized indexed configuration lacks dataset.blend_per_split")
    _validate_launch_policy(launch_policy)
    if any(record.get("launch_policy", "singleton") != launch_policy for record in rank_placement):
        raise ProvenanceInvariantError("environment launch policy differs from runtime placement")
    return {
        "schema_version": 1,
        "slurm_job_id": slurm_job_id,
        "pytorch_version": pytorch_version,
        "cuda_version": cuda_version,
        "cuda_allocator_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "default"),
        "nccl_version": nccl_version,
        "gpu_name": gpu_name,
        "container_sqsh_sha256": container_sqsh_sha256,
        "megatron_lm": dict(megatron_lm),
        "megatron_bridge": dict(megatron_bridge),
        "ampa_runtime_sources": dict(ampa_runtime_sources),
        "reviewed_inputs_proof": _reviewed_inputs_proof(slurm_job_id, ampa_runtime_sources),
        "checkpoint": checkpoint,
        "dataset_data_path": _json_value(dataset_data_path),
        "dataset_blend_per_split": _json_value(dataset_blend_per_split),
        "parallelism": _json_value(parallelism),
        "runtime_topology": _json_value(runtime_topology),
        "rank_placement": [dict(record) for record in rank_placement],
        "hf_model_id": hf_model_id,
        "hf_model_revision": hf_model_revision,
        "dataset_consumer": _json_value(dataset_consumer),
        "activation_recomputation": _json_value(activation_recomputation),
        **({"launch_policy": launch_policy} if launch_policy != "singleton" else {}),
    }


def make_indexed_run_provenance_callback(
    *,
    run_dir: Path,
    lightweight_profiling: str,
    expected_recompute_mode: str = "none",
    topology_variant: str = "cp1_ep8",
    nsys_trace: bool = False,
    torch_trace: bool = False,
    sample_fingerprints: bool = False,
    router_observer_attached: Callable[[Any], bool] | None = None,
    dataset_manifest: Mapping[str, object] | None = None,
    launch_policy: str = "singleton",
) -> Any:
    """Create the later-driver-integrated, startup-only provenance callback.

    A profile-on caller must provide ``router_observer_attached`` and arrange for
    it to return true only after its observer has installed its wrappers.  The
    profile-off default is false and touches no routing object.
    """

    _validate_launch_policy(launch_policy)
    from megatron.bridge.training.callbacks import Callback

    class IndexedRunProvenanceCallback(Callback):
        def on_train_start(self, context: Any) -> None:
            import torch
            import torch.distributed as dist

            if not dist.is_initialized():
                raise ProvenanceInvariantError("indexed-run provenance requires an initialized process group")
            if topology_variant in _EP16_RUNTIME_TOPOLOGIES:
                expected_topology = _EP16_RUNTIME_TOPOLOGIES[topology_variant]
            elif topology_variant == "cp2_ep8":
                expected_topology = _CP2_RUNTIME_TOPOLOGY
            elif topology_variant == "cp1_ep8":
                expected_topology = _EXPECTED_RUNTIME_TOPOLOGY
            else:
                raise ProvenanceInvariantError(f"unsupported topology variant: {topology_variant!r}")
            strict_topology = topology_variant == "cp2_ep8" or topology_variant in _EP16_RUNTIME_TOPOLOGIES
            runtime_topology = validate_runtime_topology(_runtime_topology(dist), expected=expected_topology)
            local_placement = _local_placement(torch, rank=dist.get_rank(), launch_policy=launch_policy)
            local_placement["activation_recomputation"] = validate_recomputation_runtime(
                context.state.cfg.model, context.model, expected_mode=expected_recompute_mode,
                strict_no_recompute=strict_topology,
            )
            if strict_topology:
                local_placement["process_groups"] = _cp2_process_groups(dist)
                local_placement["local_expert_indices"] = local_placement["activation_recomputation"]["local_expert_indices"]
            gathered: list[dict[str, object] | None] = [None] * dist.get_world_size()
            dist.all_gather_object(gathered, local_placement)
            if any(record is None for record in gathered):
                raise ProvenanceInvariantError("rank placement gather returned an empty record")
            placement = validate_rank_placement(
                [record for record in gathered if record is not None],
                expected_world_size=runtime_topology["world_size"],
                launch_policy=launch_policy,
            )
            if strict_topology:
                validate_runtime_groups(placement, expected_world_size=runtime_topology["world_size"],
                                        cp=runtime_topology["cp"])
            if dataset_manifest is None:
                raise ProvenanceInvariantError("indexed-run provenance requires the reviewed dataset manifest")
            from tools.bridge_indexed_consumer import assert_bridge_indexed_consumer

            consumer = assert_bridge_indexed_consumer(context.state.cfg, dataset_manifest)
            attached = bool(router_observer_attached(context)) if router_observer_attached is not None else False
            observation_mode = build_observation_mode(
                lightweight_profiling=lightweight_profiling,
                router_observer_attached=attached,
                nsys_trace=nsys_trace,
                torch_trace=torch_trace,
                sample_fingerprints=sample_fingerprints,
            )
            if dist.get_rank() != 0:
                return
            config = context.state.cfg
            model = config.model
            root = run_dir.parents[1]
            environment = build_environment_record(
                slurm_job_id=os.environ.get("SLURM_JOB_ID"),
                pytorch_version=torch.__version__,
                cuda_version=torch.version.cuda,
                nccl_version=_nccl_version(torch),
                gpu_name=str(torch.cuda.get_device_name()),
                container_sqsh_sha256=_read_optional(root / "setup" / "gh200-cuda130-megatron.sqsh.sha256"),
                megatron_lm=_git_identity(root / "Megatron-LM"),
                megatron_bridge=_git_identity(root / "Megatron-Bridge"),
                ampa_runtime_sources=_ampa_runtime_sources(root),
                checkpoint=_string_or_none(getattr(config.checkpoint, "pretrained_checkpoint", None)),
                dataset_data_path=getattr(config.dataset, "data_path", None),
                dataset_blend_per_split=getattr(config.dataset, "blend_per_split", None),
                parallelism={
                    "tp": getattr(model, "tensor_model_parallel_size", None),
                    "pp": getattr(model, "pipeline_model_parallel_size", None),
                    "cp": getattr(model, "context_parallel_size", None),
                    "ep": getattr(model, "expert_model_parallel_size", None),
                    "etp": getattr(model, "expert_tensor_parallel_size", None),
                },
                runtime_topology=runtime_topology,
                rank_placement=placement,
                hf_model_id=_string_or_none(getattr(model, "hf_model_id", None)),
                hf_model_revision=_string_or_none(getattr(model, "hf_model_revision", None)),
                dataset_consumer=consumer,
                activation_recomputation={
                    "mode": expected_recompute_mode,
                    "ranks": [
                        {"rank": record["rank"], **record["activation_recomputation"]}
                        for record in placement
                    ],
                },
                launch_policy=launch_policy,
            )
            config_dir = run_dir / "config"
            config_dir.mkdir(parents=True, exist_ok=True)
            _atomic_write_yaml(config, config_dir / "resolved.yaml")
            _atomic_write_json(config_dir / "observation_mode.json", observation_mode)
            _atomic_write_json(run_dir / "environment.json", environment)

    return IndexedRunProvenanceCallback()


def _runtime_topology(dist: Any) -> dict[str, int]:
    from megatron.core import parallel_state

    return {
        "world_size": int(dist.get_world_size()),
        "tp": int(parallel_state.get_tensor_model_parallel_world_size()),
        "pp": int(parallel_state.get_pipeline_model_parallel_world_size()),
        "cp": int(parallel_state.get_context_parallel_world_size()),
        "ep": int(parallel_state.get_expert_model_parallel_world_size()),
        "etp": int(parallel_state.get_expert_tensor_parallel_world_size()),
        "dense_dp": int(parallel_state.get_data_parallel_world_size()),
        "dense_dp_cp": int(parallel_state.get_data_parallel_world_size(with_context_parallel=True)),
        "expert_dp": int(parallel_state.get_expert_data_parallel_world_size()),
    }


def _cp2_process_groups(dist: Any) -> dict[str, list[int]]:
    from megatron.core import parallel_state

    return {
        "cp": list(dist.get_process_group_ranks(parallel_state.get_context_parallel_group())),
        "dense_dp": list(dist.get_process_group_ranks(parallel_state.get_data_parallel_group())),
        "dense_dp_cp": list(dist.get_process_group_ranks(parallel_state.get_data_parallel_group(with_context_parallel=True))),
        "ep": list(dist.get_process_group_ranks(parallel_state.get_expert_model_parallel_group())),
    }


def _local_expert_indices(model_chunks: list[Any]) -> list[int]:
    observed: list[set[int]] = []
    attention_layers = 0
    for chunk in model_chunks:
        for module in chunk.modules():
            if hasattr(module, "checkpoint_core_attention") and module.checkpoint_core_attention is not False:
                raise ProvenanceInvariantError("CP2 no-recompute variant enabled attention checkpointing")
            if type(module).__name__ == "SelfAttention":
                attention_layers += 1
            if not (hasattr(module, "local_expert_indices") and hasattr(module, "token_dispatcher")):
                continue
            if (
                getattr(module, "moe_layer_recompute", None) is not False
                or getattr(module, "shared_experts_recompute", None) is not False
                or getattr(getattr(module, "experts", None), "activation_recompute", None) is not False
            ):
                raise ProvenanceInvariantError("CP2 no-recompute variant enabled MoE checkpointing")
            observed.append(set(int(index) for index in module.local_expert_indices))
    if len(observed) != 48 or attention_layers != 48 or any(indices != observed[0] for indices in observed):
        raise ProvenanceInvariantError("CP2 did not construct 48 MoE layers with stable expert ownership")
    return sorted(observed[0])


def _local_placement(torch: Any, *, rank: int, launch_policy: str = "singleton") -> dict[str, object]:
    _validate_launch_policy(launch_policy)
    if type(rank) is not int or rank < 0:
        raise ProvenanceInvariantError("runtime rank must be a nonnegative integer")
    count = torch.cuda.device_count()
    expected_count = 4 if launch_policy == "torchrun_node4" else 1
    if type(count) is not int or count != expected_count:
        raise ProvenanceInvariantError(f"expected exactly {expected_count} CUDA-visible GPUs under {launch_policy}")
    current = torch.cuda.current_device()
    policy_fields = {}
    if launch_policy == "torchrun_node4":
        try:
            local_rank, local_world = int(os.environ["LOCAL_RANK"]), int(os.environ["LOCAL_WORLD_SIZE"])
        except (KeyError, ValueError) as error:
            raise ProvenanceInvariantError("torchrun_node4 requires integer LOCAL_RANK and LOCAL_WORLD_SIZE") from error
        if (local_rank not in range(4) or local_world != 4
                or type(current) is not int or current != local_rank):
            raise ProvenanceInvariantError("selected CUDA device differs from the torchrun local worker slot")
        policy_fields = {"launch_policy": launch_policy, "local_rank": local_rank,
                         "local_world_size": local_world, "cpu_affinity": sorted(os.sched_getaffinity(0))}
    elif type(current) is not int or current != 0:
        raise ProvenanceInvariantError("singleton launch requires selected CUDA ordinal zero")
    from tools.inspect_gpu_identity import _cuda_driver_identity

    identity = _cuda_driver_identity()
    if launch_policy == "torchrun_node4" and (type(identity.get("device_id")) is not int
                                               or identity["device_id"] != current):
        raise ProvenanceInvariantError("CUDA-driver UUID identity differs from the selected worker ordinal")
    gpu_uuid = identity.get("uuid")
    if not isinstance(gpu_uuid, str) or not gpu_uuid.startswith("GPU-"):
        raise ProvenanceInvariantError(
            f"cannot identify task-selected physical GPU: {identity.get('error')}"
        )
    return {
        "rank": int(rank),
        "hostname": socket.gethostname(),
        "slurm_node_id": os.environ.get("SLURM_NODEID"),
        "slurm_local_id": os.environ.get("SLURM_LOCALID"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_visible_device_count": count,
        "cuda_current_device": current,
        "gpu_name": str(torch.cuda.get_device_name()),
        "physical_gpu_uuid": gpu_uuid,
        "physical_gpu_identity_source": _GPU_UUID_SOURCE,
        **policy_fields,
    }


def _atomic_write_yaml(config: Any, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    config.to_yaml(str(temporary))
    temporary.replace(destination)


def _atomic_write_json(destination: Path, value: Mapping[str, object]) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(destination)


def _json_value(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    raise ProvenanceInvariantError(f"provenance value is not JSON-safe: {type(value).__name__}")


def _string_or_none(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _nccl_version(torch: Any) -> list[int] | None:
    try:
        return list(torch.cuda.nccl.version())
    except (AttributeError, RuntimeError):
        return None


def _read_optional(path: Path) -> str | None:
    try:
        value = path.read_text(encoding="utf-8").split()
    except OSError:
        return None
    digest = value[0] if value else None
    if digest is None or len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ProvenanceInvariantError(f"invalid container SHA256 record: {path}")
    return digest


def _git_identity(path: Path) -> dict[str, str | None]:
    def output(*args: str) -> str | None:
        try:
            return subprocess.check_output(args, cwd=path, text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = output("git", "status", "--porcelain=v1")
    diff = output("git", "diff", "--binary", "HEAD")
    return {
        "revision": output("git", "rev-parse", "HEAD"),
        "status": status,
        "diff_sha256": None if diff is None else hashlib.sha256(diff.encode()).hexdigest(),
    }


def _reviewed_inputs_proof(slurm_job_id, sources):
    value = os.environ.get("AMPA_REVIEWED_INPUTS_PROOF")
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ProvenanceInvariantError("reviewed input proof is missing or not a regular absolute path")
    payload = path.read_bytes()
    record = json.loads(payload)
    if (record.get("valid_reviewed_inputs") is not True
            or record.get("slurm_job_id") != slurm_job_id
            or record.get("ampa_runtime_sources") != sources
            or record.get("checkpoint_payload", {}).get("verified") is not True):
        raise ProvenanceInvariantError("reviewed input proof has wrong allocation, source or payload validation")
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(payload).hexdigest()}


def _ampa_runtime_sources(root: Path) -> dict[str, object]:
    """Bind untracked AMPA execution code that root Git cannot identify."""
    file_hashes: dict[str, str] = {}
    for relative in _AMPA_RUNTIME_SOURCE_PATHS:
        source = root / relative
        if source.is_symlink() or not source.is_file():
            raise ProvenanceInvariantError(f"AMPA runtime source is missing or symlinked: {source}")
        file_hashes[relative] = hashlib.sha256(source.read_bytes()).hexdigest()
    encoded = json.dumps(file_hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {"sha256": hashlib.sha256(encoded).hexdigest(), "files": file_hashes}
