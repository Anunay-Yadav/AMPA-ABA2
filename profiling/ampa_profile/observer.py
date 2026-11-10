"""GPU-side summaries around the existing standard AllToAll dispatcher.

The observer wraps an existing MoE layer's ``route`` method and its standard
AllToAll dispatcher's ``preprocess`` method after model construction. The
latter is the last point at which the dispatcher's actual ``input_splits`` and
receiver-side expert counts are still GPU tensors. No Megatron behavior is
modified.
"""

from __future__ import annotations

import json
import os
import socket
import types
import uuid
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class ProfileInvariantError(RuntimeError):
    """Raised instead of emitting plausible-looking but invalid measurements."""


@dataclass(frozen=True)
class MoESite:
    """Actual runtime identity and ownership for one MoE layer."""

    layer: int
    module_name: str
    experts: tuple[int, ...]
    ep_global_ranks: tuple[int, ...]
    dispatcher_type: str
    router_topk: int


@dataclass
class LayerAccumulator:
    """Small GPU-resident summaries for one source rank and layer."""

    receiver_expert_counts: Any
    destination_counts: Any
    assignment_count: Any
    probability_sum: Any
    probability_square_sum: Any
    local_expert_indices: Any
    route_calls: int = 0
    route_token_positions: int = 0
    routed_token_positions: int = 0
    preprocess_calls: int = 0


@dataclass
class StepAccumulator:
    """A whole optimizer-step record retained on GPU until the run is complete."""

    step: int
    layers: dict[int, LayerAccumulator] = field(default_factory=dict)
    skipped_iter: bool | None = None
    grad_norm: float | None = None
    loss_is_finite: Any | None = None


class RoutingObserver:
    """Aggregate existing dispatcher metadata without per-layer D2H transfers.

    This observer is intentionally limited to the normal, dropless AllToAll
    path. It rejects capacity drop/padding, CUDA graphs, and recomputation,
    because an external Python wrapper cannot demonstrate correct replay
    suppression for those modes.
    """

    def __init__(
        self,
        *,
        run_dir: Path,
        profile_start_step: int,
        profile_stop_step: int,
        expected_calls_per_layer: int,
        expected_tokens_per_call: int,
    ) -> None:
        if profile_start_step < 0 or profile_stop_step <= profile_start_step:
            raise ValueError("profile window must satisfy 0 <= start < stop")
        if expected_calls_per_layer <= 0 or expected_tokens_per_call <= 0:
            raise ValueError("expected route coverage must be positive")
        self.run_dir = run_dir
        self.profile_start_step = profile_start_step
        self.profile_stop_step = profile_stop_step
        self.expected_calls_per_layer = expected_calls_per_layer
        self.expected_tokens_per_call = expected_tokens_per_call
        self.rank: int | None = None
        self.world_size: int | None = None
        self.current: StepAccumulator | None = None
        self.records: list[StepAccumulator] = []
        self.sites: dict[int, MoESite] = {}
        self.owner_by_layer: dict[int, tuple[int, ...]] = {}
        self.inventory: dict[str, Any] = {}
        self.invalid_reasons: list[str] = []

    def attach(self, model_chunks: list[Any]) -> None:
        """Validate the baseline path and wrap existing route/dispatcher methods."""
        import torch
        import torch.distributed as dist

        if not dist.is_initialized():
            raise ProfileInvariantError("routing observer requires an initialized process group")
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        local_sites: list[dict[str, Any]] = []
        seen_layers: set[int] = set()
        for chunk_index, model_chunk in enumerate(model_chunks):
            for module_name, module in model_chunk.named_modules():
                local_experts = getattr(module, "local_expert_indices", None)
                dispatcher = getattr(module, "token_dispatcher", None)
                ep_group = getattr(module, "ep_group", None)
                original_route = getattr(module, "route", None)
                original_dispatcher_preprocess = getattr(dispatcher, "preprocess", None)
                if (
                    local_experts is None
                    or dispatcher is None
                    or ep_group is None
                    or original_route is None
                    or original_dispatcher_preprocess is None
                ):
                    continue
                layer_number = getattr(module, "layer_number", None)
                if not isinstance(layer_number, int) or layer_number < 1:
                    raise ProfileInvariantError(
                        f"cannot derive a stable layer number for MoE module {module_name!r}"
                    )
                layer = layer_number - 1
                if layer in seen_layers:
                    raise ProfileInvariantError(f"MoE layer {layer} was discovered more than once")
                seen_layers.add(layer)
                self._validate_module_baseline(module, dispatcher, layer)
                ep_global_ranks = tuple(int(value) for value in dist.get_process_group_ranks(ep_group))
                site = MoESite(
                    layer=layer,
                    module_name=f"chunk{chunk_index}.{module_name}",
                    experts=tuple(int(value) for value in local_experts),
                    ep_global_ranks=ep_global_ranks,
                    dispatcher_type=type(dispatcher).__name__,
                    router_topk=int(getattr(module.config, "moe_router_topk")),
                )
                self.sites[id(module)] = site
                self._wrap_route(module, original_route)
                self._wrap_dispatcher_preprocess(module, dispatcher, original_dispatcher_preprocess)
                local_sites.append(
                    {
                        "layer": site.layer,
                        "module_name": site.module_name,
                        "experts": list(site.experts),
                        "ep_global_ranks": list(site.ep_global_ranks),
                        "dispatcher_type": site.dispatcher_type,
                        "router_topk": site.router_topk,
                    }
                )

        if not self.sites:
            raise ProfileInvariantError("no MoE layers were discovered after model setup")
        if seen_layers != set(range(48)):
            raise ProfileInvariantError(f"expected 48 distinct Qwen3 MoE layers, observed {sorted(seen_layers)}")
        local_metadata = {
            "rank": self.rank,
            "hostname": socket.gethostname(),
            "slurm_nodeid": os.environ.get("SLURM_NODEID"),
            "local_rank": os.environ.get("SLURM_LOCALID"),
            "cuda_device": torch.cuda.current_device(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu_name": torch.cuda.get_device_name(),
            "physical_gpu_uuid": self._physical_gpu_uuid(torch),
            "physical_gpu_identity_source": "cuda_driver.cuDeviceGetUuid(cuda_runtime.cudaGetDevice())",
            "sites": local_sites,
        }
        all_metadata: list[dict[str, Any] | None] = [None] * self.world_size
        dist.all_gather_object(all_metadata, local_metadata)
        if any(value is None for value in all_metadata):
            raise ProfileInvariantError("rank metadata gather returned an empty record")
        rank_metadata = [value for value in all_metadata if value is not None]
        self._validate_rank_placement(rank_metadata, self.world_size)
        self.owner_by_layer = self._build_ownership(rank_metadata)
        self.inventory = {
            "rank": self.rank,
            "world_size": self.world_size,
            "rank_metadata": rank_metadata,
            "local_sites": local_sites,
            "owner_by_layer": {str(layer): list(owners) for layer, owners in self.owner_by_layer.items()},
        }

    @staticmethod
    def _physical_gpu_uuid(torch: Any) -> str:
        """Identify the task-selected physical GPU, not its remapped CUDA ordinal."""
        if torch.cuda.device_count() != 1:
            raise ProfileInvariantError("expected exactly one CUDA-visible GPU per EP8 Slurm task")
        from tools.inspect_gpu_identity import _cuda_driver_identity

        identity = _cuda_driver_identity()
        gpu_uuid = identity.get("uuid")
        if not isinstance(gpu_uuid, str) or not gpu_uuid.startswith("GPU-"):
            raise ProfileInvariantError(f"cannot identify task-selected physical GPU: {identity.get('error')}")
        try:
            uuid.UUID(gpu_uuid.removeprefix("GPU-"))
        except ValueError as error:
            raise ProfileInvariantError("task-selected physical GPU UUID is malformed") from error
        return gpu_uuid

    @staticmethod
    def _validate_rank_placement(rank_metadata: list[dict[str, Any]], world_size: int) -> None:
        """Require the intended two-node/four-distinct-GPU placement before counting locality."""
        if len(rank_metadata) != world_size or {entry.get("rank") for entry in rank_metadata} != set(range(world_size)):
            raise ProfileInvariantError("incomplete or duplicate EP8 rank placement")
        gpu_uuids = [entry.get("physical_gpu_uuid") for entry in rank_metadata]
        if any(not isinstance(value, str) for value in gpu_uuids) or len(set(gpu_uuids)) != world_size:
            raise ProfileInvariantError("EP8 ranks do not have eight distinct physical GPU UUIDs")
        host_counts = Counter(entry.get("hostname") for entry in rank_metadata)
        if len(host_counts) != 2 or sorted(host_counts.values()) != [4, 4]:
            raise ProfileInvariantError(f"EP8 rank placement is not four GPUs on each of two nodes: {host_counts}")

    @staticmethod
    def _validate_module_baseline(module: Any, dispatcher: Any, layer: int) -> None:
        config = getattr(module, "config", None)
        if type(dispatcher).__name__ != "MoEAlltoAllTokenDispatcher":
            raise ProfileInvariantError(
                f"layer {layer} uses {type(dispatcher).__name__}, not the required standard AllToAll dispatcher"
            )
        if getattr(config, "moe_expert_capacity_factor", None) is not None:
            raise ProfileInvariantError(f"layer {layer} enables capacity dropping/padding")
        if getattr(config, "moe_router_padding_for_quantization", False):
            raise ProfileInvariantError(f"layer {layer} enables router padding")
        if getattr(config, "recompute_granularity", None) is not None:
            raise ProfileInvariantError(f"layer {layer} enables activation recomputation")
        if getattr(config, "cuda_graph_impl", "none") not in {None, "none"}:
            raise ProfileInvariantError(f"layer {layer} enables CUDA graph replay")

    def _wrap_route(self, module: Any, original_route: Any) -> None:
        """Observe existing sparse router probabilities without copying a routing map."""

        def observed_route(wrapped_module: Any, hidden_states: Any, padding_mask: Any = None) -> Any:
            probabilities, routing_map = original_route(hidden_states, padding_mask)
            self.observe_route(wrapped_module, probabilities, routing_map)
            return probabilities, routing_map

        module.route = types.MethodType(observed_route, module)

    def _wrap_dispatcher_preprocess(
        self, module: Any, dispatcher: Any, original_preprocess: Any
    ) -> None:
        """Observe existing GPU dispatcher metadata before Megatron's own D2H boundary."""

        def observed_preprocess(wrapped_dispatcher: Any, routing_map: Any) -> Any:
            result = original_preprocess(routing_map)
            self.observe_dispatcher_preprocess(module, wrapped_dispatcher, routing_map, result)
            return result

        dispatcher.preprocess = types.MethodType(observed_preprocess, dispatcher)

    def _build_ownership(self, rank_metadata: list[dict[str, Any]]) -> dict[int, tuple[int, ...]]:
        owners_by_layer: dict[int, dict[int, int]] = {}
        ep_ranks_by_layer: dict[int, tuple[int, ...]] = {}
        for rank_record in rank_metadata:
            source_rank = int(rank_record["rank"])
            for site in rank_record["sites"]:
                layer = int(site["layer"])
                ep_global_ranks = tuple(int(value) for value in site["ep_global_ranks"])
                prior_group = ep_ranks_by_layer.setdefault(layer, ep_global_ranks)
                if prior_group != ep_global_ranks:
                    raise ProfileInvariantError(
                        f"layer {layer} reports inconsistent expert-parallel groups: "
                        f"{prior_group} vs {ep_global_ranks}"
                    )
                owners = owners_by_layer.setdefault(layer, {})
                for expert in site["experts"]:
                    expert = int(expert)
                    previous = owners.setdefault(expert, source_rank)
                    if previous != source_rank:
                        raise ProfileInvariantError(
                            f"layer {layer} expert {expert} is owned by both ranks {previous} and {source_rank}"
                        )
        finalized: dict[int, tuple[int, ...]] = {}
        for layer, experts in owners_by_layer.items():
            expert_count = max(experts) + 1
            if set(experts) != set(range(expert_count)):
                raise ProfileInvariantError(f"layer {layer} expert IDs are not contiguous from zero")
            ep_global_ranks = ep_ranks_by_layer[layer]
            expected_ep_rank_set = set(range(self.world_size or 0))
            if len(ep_global_ranks) != (self.world_size or 0) or set(ep_global_ranks) != expected_ep_rank_set:
                raise ProfileInvariantError(
                    f"layer {layer} has EP ranks {ep_global_ranks}, expected a duplicate-free full world "
                    f"{tuple(range(self.world_size or 0))}"
                )
            if expert_count % len(ep_global_ranks):
                raise ProfileInvariantError(
                    f"layer {layer} has {expert_count} experts not divisible by EP size {len(ep_global_ranks)}"
                )
            experts_per_rank = expert_count // len(ep_global_ranks)
            expected_owners = tuple(ep_global_ranks[expert // experts_per_rank] for expert in range(expert_count))
            actual_owners = tuple(experts[index] for index in range(expert_count))
            if actual_owners != expected_owners:
                raise ProfileInvariantError(
                    f"layer {layer} expert ownership does not match its runtime EP-rank ordering"
                )
            finalized[layer] = actual_owners
        return finalized

    def start_step(self, step: int) -> None:
        """Start a complete optimizer-step record without a CUDA synchronization."""
        if self.current is not None:
            raise ProfileInvariantError("received a train-step start before finalizing the prior step")
        if self.profile_start_step <= step < self.profile_stop_step:
            self.current = StepAccumulator(step=step)

    def observe_route(self, module: Any, probabilities: Any, routing_map: Any) -> None:
        """Accumulate selected router-probability summaries while GPU-resident."""
        if self.current is None:
            return
        import torch

        site = self.sites.get(id(module))
        if site is None:
            raise ProfileInvariantError("wrapped MoE layer was absent from startup inventory")
        if not isinstance(probabilities, torch.Tensor) or not probabilities.is_cuda:
            raise ProfileInvariantError(f"layer {site.layer} router probabilities were not GPU-resident")
        if routing_map.ndim != 2 or probabilities.shape != routing_map.shape:
            raise ProfileInvariantError(
                f"layer {site.layer} router output shapes differ: "
                f"probabilities={tuple(probabilities.shape)}, routing_map={tuple(routing_map.shape)}"
            )
        accumulator = self._layer_accumulator(site, probabilities.device)
        if routing_map.shape[0] != self.expected_tokens_per_call:
            raise ProfileInvariantError(
                f"layer {site.layer} route saw {routing_map.shape[0]} token positions, "
                f"expected {self.expected_tokens_per_call} per microbatch"
            )
        accumulator.route_calls += 1
        accumulator.route_token_positions += int(routing_map.shape[0])
        # Preserve the input shape: masked_select has a data-dependent output
        # size on CUDA. Unselected weights become zero before FP32 moments.
        selected_probabilities = torch.where(routing_map, probabilities.detach(), 0).to(torch.float32)
        accumulator.probability_sum.add_(selected_probabilities.sum())
        accumulator.probability_square_sum.add_(selected_probabilities.square().sum())

    def observe_dispatcher_preprocess(
        self, module: Any, dispatcher: Any, routing_map: Any, tokens_per_expert: Any
    ) -> None:
        """Accumulate actual source/receiver assignment counts before D2H conversion."""
        if self.current is None:
            return
        import torch

        site = self.sites.get(id(module))
        if site is None:
            raise ProfileInvariantError("wrapped MoE layer was absent from startup inventory")
        input_splits = dispatcher.input_splits
        if not isinstance(input_splits, torch.Tensor) or not input_splits.is_cuda:
            raise ProfileInvariantError(
                f"layer {site.layer} input_splits were not GPU-resident at dispatcher preprocess"
            )
        if not isinstance(tokens_per_expert, torch.Tensor) or not tokens_per_expert.is_cuda:
            raise ProfileInvariantError(
                f"layer {site.layer} receiver expert counts were not GPU-resident at dispatcher preprocess"
            )
        if input_splits.numel() != len(site.ep_global_ranks):
            raise ProfileInvariantError(
                f"layer {site.layer} has {input_splits.numel()} split entries for {len(site.ep_global_ranks)} EP ranks"
            )
        if tokens_per_expert.numel() != len(site.experts):
            raise ProfileInvariantError(
                f"layer {site.layer} has {tokens_per_expert.numel()} local expert counts for "
                f"{len(site.experts)} local experts"
            )
        if routing_map.ndim != 2:
            raise ProfileInvariantError(
                f"layer {site.layer} routing map must be rank two, got shape {tuple(routing_map.shape)}"
            )
        accumulator = self._layer_accumulator(site, input_splits.device)
        if routing_map.shape[0] != self.expected_tokens_per_call:
            raise ProfileInvariantError(
                f"layer {site.layer} dispatch saw {routing_map.shape[0]} token positions, "
                f"expected {self.expected_tokens_per_call} per microbatch"
            )
        accumulator.receiver_expert_counts.scatter_add_(
            0, accumulator.local_expert_indices, tokens_per_expert.to(torch.int64)
        )
        accumulator.destination_counts.add_(input_splits.to(torch.int64))
        accumulator.assignment_count.add_(input_splits.to(torch.int64).sum())
        accumulator.routed_token_positions += int(routing_map.shape[0])
        accumulator.preprocess_calls += 1

    def _layer_accumulator(self, site: MoESite, device: Any) -> LayerAccumulator:
        import torch

        assert self.current is not None
        current = self.current.layers.get(site.layer)
        if current is not None:
            return current
        expert_count = len(self.owner_by_layer[site.layer])
        current = LayerAccumulator(
            receiver_expert_counts=torch.zeros(expert_count, dtype=torch.int64, device=device),
            destination_counts=torch.zeros(len(site.ep_global_ranks), dtype=torch.int64, device=device),
            assignment_count=torch.zeros((), dtype=torch.int64, device=device),
            probability_sum=torch.zeros((), dtype=torch.float32, device=device),
            probability_square_sum=torch.zeros((), dtype=torch.float32, device=device),
            local_expert_indices=torch.tensor(site.experts, dtype=torch.int64, device=device),
        )
        self.current.layers[site.layer] = current
        return current

    def finish_step(
        self, *, skipped_iter: bool | None, grad_norm: float | None, loss_dict: dict[str, Any] | None
    ) -> None:
        """Seal the optimizer-step record; no GPU data moves at this boundary."""
        current = self.current
        self.current = None
        if current is None:
            return
        expected_layers = set(self.owner_by_layer)
        if set(current.layers) != expected_layers:
            raise ProfileInvariantError(
                f"step {current.step} observed layers {sorted(current.layers)}, expected {sorted(expected_layers)}"
            )
        for layer, accumulator in current.layers.items():
            expected_positions = self.expected_calls_per_layer * self.expected_tokens_per_call
            if (
                accumulator.route_calls != self.expected_calls_per_layer
                or accumulator.preprocess_calls != self.expected_calls_per_layer
                or accumulator.route_token_positions != expected_positions
                or accumulator.routed_token_positions != expected_positions
            ):
                raise ProfileInvariantError(
                    f"step {current.step}, layer {layer} route/dispatch coverage is "
                    f"{accumulator.route_calls}/{accumulator.preprocess_calls} calls and "
                    f"{accumulator.route_token_positions}/{accumulator.routed_token_positions} positions; "
                    f"expected {self.expected_calls_per_layer} calls and {expected_positions} positions each"
                )
        current.skipped_iter = skipped_iter
        current.grad_norm = None if grad_norm is None else float(grad_norm)
        current.loss_is_finite = self._loss_is_finite(loss_dict)
        self.records.append(current)

    @staticmethod
    def _loss_is_finite(loss_dict: dict[str, Any] | None) -> Any | None:
        """Build one deferred GPU boolean, avoiding a per-step scalar transfer."""
        if not loss_dict:
            return None
        import torch

        checks = [torch.isfinite(value.detach()).all() for value in loss_dict.values() if isinstance(value, torch.Tensor)]
        return torch.stack(checks).all() if checks else None

    def write_raw(self) -> Path:
        """Synchronize once after training and write one compact rank-local record."""
        if self.rank is None or self.world_size is None:
            raise ProfileInvariantError("routing observer was never attached")
        if self.current is not None:
            raise ProfileInvariantError("cannot write a profile while an optimizer step is active")
        import torch

        torch.cuda.synchronize()
        payload = {
            "schema_version": 2,
            "run_id": self.run_dir.name,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "rank": self.rank,
            "world_size": self.world_size,
            "profile_start_step": self.profile_start_step,
            "profile_stop_step": self.profile_stop_step,
            "expected_calls_per_layer": self.expected_calls_per_layer,
            "expected_tokens_per_call": self.expected_tokens_per_call,
            "inventory": self.inventory,
            "invalid_reasons": self.invalid_reasons,
            "steps": [self._serialize_step(record) for record in self.records],
        }
        raw_dir = self.run_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        destination = raw_dir / f"routing-rank-{self.rank:04d}.json"
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        temporary.replace(destination)
        return destination

    @staticmethod
    def _scalar(value: Any) -> int | float:
        return value.detach().cpu().item()

    def _serialize_step(self, record: StepAccumulator) -> dict[str, Any]:
        return {
            "step": record.step,
            "skipped_iter": record.skipped_iter,
            "grad_norm": record.grad_norm,
            "loss_is_finite": None if record.loss_is_finite is None else bool(self._scalar(record.loss_is_finite)),
            "layers": [
                {
                    "layer": layer,
                    "receiver_expert_counts": accumulator.receiver_expert_counts.detach().cpu().tolist(),
                    "destination_counts": accumulator.destination_counts.detach().cpu().tolist(),
                    "assignment_count": self._scalar(accumulator.assignment_count),
                    "probability_sum": self._scalar(accumulator.probability_sum),
                    "probability_square_sum": self._scalar(accumulator.probability_square_sum),
                    "routed_token_positions": accumulator.routed_token_positions,
                    "route_calls": accumulator.route_calls,
                    "route_token_positions": accumulator.route_token_positions,
                    "preprocess_calls": accumulator.preprocess_calls,
                }
                for layer, accumulator in sorted(record.layers.items())
            ],
        }
