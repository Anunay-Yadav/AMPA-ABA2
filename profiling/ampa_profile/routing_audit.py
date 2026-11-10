"""Validate every actual top-k map and dispatcher split in matched experiments.

Small summaries stay on GPU during each optimizer step. Audit kernels have
their own trace label. The observer never changes returned tensors.
"""

from collections import Counter
from contextvars import ContextVar
import json
from pathlib import Path
import socket
from types import MethodType


def _layout_reference(torch, routing_map, destinations, rank_nodes, num_nodes):
    """Keep assignments and deduplicated rank/node deliveries separate."""
    if (routing_map.dtype != torch.bool or routing_map.ndim != 2 or routing_map.shape[1] != 128
            or destinations.shape != (128,) or destinations.dtype != torch.int64
            or rank_nodes.ndim != 1 or rank_nodes.dtype != torch.int64
            or destinations.device != routing_map.device or rank_nodes.device != routing_map.device):
        raise RuntimeError("invalid routing reference tensor metadata")
    positions, ranks = routing_map.shape[0], rank_nodes.numel()
    assignments = torch.zeros((positions, ranks), device=routing_map.device, dtype=torch.int64)
    assignments.scatter_add_(1, destinations.expand(positions, -1), routing_map.to(torch.int64))
    membership = assignments != 0
    node_assignments = torch.zeros((positions, num_nodes), device=routing_map.device, dtype=torch.int64)
    node_assignments.scatter_add_(1, rank_nodes.expand(positions, -1), membership.to(torch.int64))
    return {"expert_counts": routing_map.sum(0, dtype=torch.int64),
            "assignment_rank_counts": assignments.sum(0), "rank_membership": membership,
            "unique_rank_counts": membership.sum(0, dtype=torch.int64),
            "unique_node_counts": (node_assignments != 0).sum(0, dtype=torch.int64)}


def _selected_metadata_errors(torch, routing_map, probabilities, indices, weights):
    if (indices.shape != (routing_map.shape[0], 8) or indices.dtype != torch.int64
            or indices.device != routing_map.device or weights.shape != indices.shape
            or weights.dtype != probabilities.dtype or weights.device != probabilities.device):
        raise RuntimeError("native Flex selected metadata has unexpected shape, dtype or device")
    safe = indices.clamp(0, 127)
    reconstructed = torch.zeros_like(routing_map).scatter_(1, safe, True)
    # Clamp only new audit indices; never alter the native tensors. Invalid
    # native IDs are counted without launching an out-of-bounds gather.
    return ((indices < 0).sum() + (indices >= 128).sum()
            + (reconstructed != routing_map).sum() + (reconstructed.sum(1) != 8).sum()
            + (weights != probabilities.gather(1, safe)).sum())


def _layout_errors(torch, layout, expected):
    if not isinstance(layout, tuple) or len(layout) != 5:
        raise RuntimeError("unexpected UCCL dispatch layout result")
    error = torch.zeros((), device=expected["expert_counts"].device, dtype=torch.int64)
    for actual, key in zip(layout[:4], ("unique_rank_counts", "unique_node_counts", "expert_counts", "rank_membership")):
        reference = expected[key]
        allowed = (torch.bool,) if key == "rank_membership" else (torch.int32, torch.int64)
        if (not isinstance(actual, torch.Tensor) or actual.shape != reference.shape
                or actual.device != reference.device or actual.dtype not in allowed):
            raise RuntimeError(f"UCCL layout {key} metadata differs from the routing reference")
        error.add_((actual != reference).sum())
    return error


def _ready_receive_counts(torch, result, local_experts):
    if (not isinstance(result, tuple) or len(result) != 3
            or not isinstance(result[0], torch.Tensor) or not isinstance(result[1], torch.Tensor)):
        raise RuntimeError("dispatcher postprocess did not return the native token/count/probability tuple")
    counts = result[1]
    if (counts.device.type != "cpu" or counts.dtype not in (torch.int32, torch.int64)
            or counts.shape != (local_experts,) or counts.requires_grad):
        raise RuntimeError("received expert counts must already be ready CPU integer data")
    values = counts.detach().tolist()
    if any(value < 0 for value in values) or sum(values) != result[0].shape[0]:
        raise RuntimeError("received expert counts do not match the returned expert rows")
    return values


def make_routing_audit_callback(*, run_dir: Path, expected_steps: int,
                                num_groups: int | None, group_topk: int | None,
                                expected_world_size: int = 8, accumulation: int = 16,
                                token_positions: int = 2048, diagnostic_num_groups: int = 8,
                                communication_backend: str = "alltoall", record_receives: bool = False):
    if communication_backend not in ("alltoall", "uccl_ep") or type(record_receives) is not bool:
        raise ValueError("unsupported routing-audit communication backend or receive-recording flag")
    if communication_backend == "uccl_ep" and not record_receives:
        raise ValueError("UCCL routing audits require receive recording")
    if (expected_world_size not in (8, 16) or accumulation < 1 or token_positions < 1
            or diagnostic_num_groups < 1 or 128 % diagnostic_num_groups
            or expected_world_size % diagnostic_num_groups):
        raise ValueError("routing audit requires a supported, balanced workload")
    if ((num_groups is None) != (group_topk is None)
            or (num_groups is not None and (num_groups != diagnostic_num_groups
                                           or not 1 <= group_topk <= num_groups))):
        raise ValueError("routing groups must match the matched diagnostic grouping")
    group_size = 128 // diagnostic_num_groups
    experts_per_owner = 128 // expected_world_size
    from megatron.bridge.training.callbacks import Callback

    class RoutingAudit(Callback):
        def __init__(self):
            self.current = None
            self.records = []
            self.restore = []
            self.uccl_context = ContextVar("ampa_uccl_audit_layer", default=None)
            self.node_hosts = {}
            self.rank_nodes = {}

        def _set(self, owner, name, value):
            self.restore.append((owner, name, name in vars(owner), vars(owner).get(name)))
            setattr(owner, name, value)

        def _detach(self):
            for owner, name, had_own, original in reversed(self.restore):
                if had_own:
                    setattr(owner, name, original)
                else:
                    delattr(owner, name)
            self.restore.clear()

        def _attach_uccl_layer(self, torch, dist, module, layer, peers):
            from megatron.core.transformer.moe.token_dispatcher import MoEFlexTokenDispatcher, _DeepepManager

            dispatcher = module.token_dispatcher
            manager = getattr(dispatcher, "_comm_manager", None)
            if (type(dispatcher) is not MoEFlexTokenDispatcher or type(manager) is not _DeepepManager
                    or dispatcher.tp_size != 1
                    or manager.num_experts != 128 or manager.num_local_experts != experts_per_owner
                    or manager.router_topk != 8
                    or list(dist.get_process_group_ranks(manager.group)) != peers):
                raise RuntimeError("UCCL audit requires the existing native Flex/DeepEP manager and matching TP1/EP group")
            blocks = [peers[first:first + 4] for first in range(0, expected_world_size, 4)]
            block_hosts = [{self.hosts[rank] for rank in block} for block in blocks]
            if (any(len(hosts) != 1 for hosts in block_hosts)
                    or len({next(iter(hosts)) for hosts in block_hosts}) != expected_world_size // 4):
                raise RuntimeError("UCCL four-rank communication blocks do not match distinct physical nodes")
            self.node_hosts[layer] = [next(iter(hosts)) for hosts in block_hosts]
            self.rank_nodes[layer] = torch.tensor(
                [self.node_hosts[layer].index(self.hosts[peer]) for peer in peers], device="cuda", dtype=torch.int64)
            original_setup, original_dispatch = manager.setup_metadata, manager.dispatch

            def setup(_manager, routing_map, probabilities, *args, **kwargs):
                result = original_setup(routing_map, probabilities, *args, **kwargs)
                with torch.profiler.record_function("AMPA/TRAIN/routing_audit"):
                    routing = routing_map.detach().reshape(token_positions, 128)
                    probs = probabilities.detach().reshape(token_positions, 128)
                    if not routing.is_cuda or not probs.is_cuda:
                        raise RuntimeError("UCCL metadata audit requires actual CUDA routing tensors")
                    record = self.current[layer]
                    record["bad_metadata"].add_(_selected_metadata_errors(
                        torch, routing, probs, _manager.token_indices, _manager.token_probs))
                    reference = _layout_reference(torch, routing, self.destination_indices[layer],
                                                  self.rank_nodes[layer], len(self.node_hosts[layer]))
                    record["expected_layout"] = reference
                    record["splits"].add_(reference["assignment_rank_counts"])
                    record["metadata_calls"] += 1
                return result

            def dispatch(_manager, *args, **kwargs):
                if self.uccl_context.get() is not None:
                    raise RuntimeError("nested UCCL audit dispatch context")
                self.current[layer]["dispatch_calls"] += 1
                token = self.uccl_context.set((layer, _manager))
                try:
                    return original_dispatch(*args, **kwargs)
                finally:
                    self.uccl_context.reset(token)

            self._set(manager, "setup_metadata", MethodType(setup, manager))
            self._set(manager, "dispatch", MethodType(dispatch, manager))

        def _attach_uccl_layout(self, torch):
            import uccl.ep as ep
            import deep_ep.buffer as wrapper
            from megatron.core.transformer.moe import fused_a2a

            if (ep.num_max_nvl_peers() != 4 or wrapper.ep is not ep
                    or fused_a2a.Buffer is not wrapper.Buffer or not fused_a2a.HAVE_DEEP_EP):
                raise RuntimeError("routing audit backend is not the official four-peer UCCL compatibility Buffer")
            original = wrapper.Buffer.get_dispatch_layout

            def layout(buffer, *args, **kwargs):
                context = self.uccl_context.get()
                if context is None or self.current is None:
                    raise RuntimeError("UCCL dispatch layout called outside a forward manager context; cached backward must reuse its handle")
                layer, manager = context
                if buffer.group is not manager.group or not isinstance(buffer.runtime, ep.Buffer):
                    raise RuntimeError("UCCL layout used a different runtime communication buffer")
                result = original(buffer, *args, **kwargs)
                # Layout outputs are asynchronous. Audit on their own producer
                # stream; native dispatch's existing completion event covers
                # this work before the step-end host readback. No new wait.
                with torch.profiler.record_function("AMPA/TRAIN/routing_audit"), torch.cuda.stream(buffer.get_comm_stream()):
                    record = self.current[layer]
                    if "expected_layout" not in record:
                        raise RuntimeError("UCCL layout lacks its native setup reference")
                    record["bad_layout"].add_(_layout_errors(torch, result, record["expected_layout"]))
                    record["unique_rank_counts"].add_(result[0].to(torch.int64))
                    record["unique_node_counts"].add_(result[1].to(torch.int64))
                    record["layout_calls"] += 1
                return result

            self._set(wrapper.Buffer, "get_dispatch_layout", layout)

        def _attach_receives(self, torch, dispatcher, layer):
            original = dispatcher.dispatch_postprocess

            def postprocess(_dispatcher, *args, **kwargs):
                result = original(*args, **kwargs)
                # Both native paths have already produced ready CPU counts at
                # this boundary. Read them without a CUDA copy or extra wait.
                with torch.profiler.record_function("AMPA/TRAIN/routing_audit"):
                    values = _ready_receive_counts(torch, result, experts_per_owner)
                    record = self.current[layer]
                    record["received_expert_counts"] = [a + b for a, b in zip(record["received_expert_counts"], values)]
                    record["receive_calls"] += 1
                return result

            self._set(dispatcher, "dispatch_postprocess", MethodType(postprocess, dispatcher))

        def on_train_start(self, context):
            import torch
            import torch.distributed as dist

            self.rank = dist.get_rank()
            self.sites = {}
            for chunk in context.model:
                for module in chunk.modules():
                    if hasattr(module, "local_expert_indices") and hasattr(module, "token_dispatcher"):
                        layer = int(module.layer_number) - 1
                        if layer in self.sites:
                            raise RuntimeError("duplicate audit layer")
                        cfg = module.config
                        if (cfg.num_moe_experts != 128 or cfg.moe_router_topk != 8
                                or cfg.moe_router_num_groups != num_groups
                                or cfg.moe_router_group_topk != group_topk):
                            raise RuntimeError("runtime routing differs from requested configuration")
                        self.sites[layer] = module
            if set(self.sites) != set(range(48)):
                raise RuntimeError("routing audit requires all 48 MoE layers")
            local = {"rank": self.rank, "hostname": socket.gethostname(), "experts": {
                str(layer): list(module.local_expert_indices) for layer, module in self.sites.items()}}
            metadata = [None] * dist.get_world_size()
            dist.all_gather_object(metadata, local)
            if (len(metadata) != expected_world_size
                    or sorted(item["rank"] for item in metadata) != list(range(expected_world_size))):
                raise RuntimeError(f"routing audit requires {expected_world_size} distinct ranks")
            self.hosts = {item["rank"]: item["hostname"] for item in metadata}
            self.owners = {}
            self.group_owners = {}
            self.group_hosts = {}
            self.destination_indices = {}
            for layer, module in self.sites.items():
                owners = {}
                for item in metadata:
                    local_experts = item["experts"][str(layer)]
                    if len(local_experts) != experts_per_owner or len(set(local_experts)) != experts_per_owner:
                        raise RuntimeError("runtime expert ownership is not balanced")
                    for expert in local_experts:
                        if expert in owners:
                            raise RuntimeError("expert has multiple owners")
                        owners[expert] = item["rank"]
                if set(owners) != set(range(128)):
                    raise RuntimeError("incomplete expert ownership")
                self.owners[layer] = [owners[e] for e in range(128)]
                dispatcher = module.token_dispatcher
                peers = list(dist.get_process_group_ranks(dispatcher.ep_group))
                if sorted(peers) != list(range(expected_world_size)):
                    raise RuntimeError("audit dispatcher has wrong expert group")
                self.destination_indices[layer] = torch.tensor(
                    [peers.index(owners[e]) for e in range(128)], device="cuda", dtype=torch.int64)
                groups = [Counter(self.owners[layer][start:start + group_size])
                          for start in range(0, 128, group_size)]
                if any(len(group) != expected_world_size // diagnostic_num_groups
                       or set(group.values()) != {experts_per_owner} for group in groups):
                    raise RuntimeError("expert groups do not align with balanced runtime expert owners")
                self.group_owners[layer] = [sorted(group) for group in groups]
                self.group_hosts[layer] = [sorted({self.hosts[owner] for owner in group})
                                           for group in self.group_owners[layer]]
                if expected_world_size == 16 and diagnostic_num_groups == 4:
                    if (any(len(hosts) != 1 for hosts in self.group_hosts[layer])
                            or len({hosts[0] for hosts in self.group_hosts[layer]}) != 4):
                        raise RuntimeError("each expert group must map to one physical node, with four distinct nodes")
                original_route = module.route
                original_preprocess = dispatcher.preprocess if communication_backend == "alltoall" else None

                def route(_module, *args, _original=original_route, _layer=layer, **kwargs):
                    result = _original(*args, **kwargs)
                    probabilities, routing_map = result
                    with torch.profiler.record_function("AMPA/TRAIN/routing_audit"):
                        if routing_map.shape != (token_positions, 128) or probabilities.shape != routing_map.shape:
                            raise RuntimeError("audit router shape differs from the requested workload")
                        record = self.current[_layer]
                        record["calls"] += 1
                        counts = routing_map.sum(dim=0, dtype=torch.int64)
                        record["experts"].add_(counts)
                        record["bad_topk"].add_((routing_map.sum(dim=1) != 8).sum())
                        record["nonfinite"].add_((~torch.isfinite(probabilities)).sum())
                        # Both variants run the same audit operations. Ordinary
                        # routing permits all diagnostic ownership groups.
                        used = routing_map.reshape(token_positions, diagnostic_num_groups, group_size).any(dim=2).sum(dim=1)
                        record["bad_groups"].add_((used > (group_topk if num_groups else diagnostic_num_groups)).sum())
                    return result

                def preprocess(_dispatcher, *args, _original=original_preprocess,
                               _layer=layer, _peers=peers, **kwargs):
                    result = _original(*args, **kwargs)
                    with torch.profiler.record_function("AMPA/TRAIN/routing_audit"):
                        splits = _dispatcher.input_splits
                        if not isinstance(splits, torch.Tensor) or not splits.is_cuda or splits.numel() != expected_world_size:
                            raise RuntimeError("dispatcher splits unavailable before host conversion")
                        per_expert = args[0].sum(dim=0, dtype=torch.int64)
                        expected_splits = torch.zeros(expected_world_size, device=splits.device, dtype=torch.int64)
                        expected_splits.scatter_add_(0, self.destination_indices[_layer], per_expert)
                        self.current[_layer]["bad_splits"].add_((splits != expected_splits).sum())
                        self.current[_layer]["splits"].add_(splits)
                        self.current[_layer]["dispatch_calls"] += 1
                    return result

                owners_to_restore = ((module, "route"), (dispatcher, "preprocess")) if communication_backend == "alltoall" else ((module, "route"),)
                self.restore.extend([(owner, name, name in vars(owner), vars(owner).get(name))
                                     for owner, name in owners_to_restore])
                module.route = MethodType(route, module)
                if communication_backend == "alltoall":
                    dispatcher.preprocess = MethodType(preprocess, dispatcher)
                else:
                    self._attach_uccl_layer(torch, dist, module, layer, peers)
                if record_receives:
                    self._attach_receives(torch, dispatcher, layer)
                self.sites[layer] = (module, peers)
            if communication_backend == "uccl_ep":
                self._attach_uccl_layout(torch)
            if self.rank == 0:
                (run_dir / "config" / "routing_ownership.json").write_text(json.dumps({
                    "rank_hosts": self.hosts, "expert_owners": self.owners,
                    "num_groups": num_groups, "group_topk": group_topk,
                    "diagnostic_num_groups": diagnostic_num_groups,
                    "group_owners": self.group_owners,
                    "group_hosts": self.group_hosts,
                    "audit_label": "AMPA/TRAIN/routing_audit",
                    **({"communication_backend": communication_backend, "record_receives": True}
                       if record_receives else {}),
                    **({"uccl_node_hosts_in_pg_order": self.node_hosts} if communication_backend == "uccl_ep" else {}),
                }, indent=2, sort_keys=True))

        def on_train_step_start(self, context):
            import torch

            self.current = {layer: {
                "experts": torch.zeros(128, dtype=torch.int64, device="cuda"),
                "splits": torch.zeros(expected_world_size, dtype=torch.int64, device="cuda"),
                "bad_topk": torch.zeros((), dtype=torch.int64, device="cuda"),
                "bad_groups": torch.zeros((), dtype=torch.int64, device="cuda"),
                "nonfinite": torch.zeros((), dtype=torch.int64, device="cuda"),
                "bad_splits": torch.zeros((), dtype=torch.int64, device="cuda"),
                "calls": 0, "dispatch_calls": 0,
            } for layer in self.sites}
            if record_receives:
                for record in self.current.values():
                    record.update(received_expert_counts=[0] * experts_per_owner, receive_calls=0)
                    if communication_backend == "uccl_ep":
                        record.update(bad_metadata=torch.zeros((), dtype=torch.int64, device="cuda"),
                            bad_layout=torch.zeros((), dtype=torch.int64, device="cuda"),
                            unique_rank_counts=torch.zeros(expected_world_size, dtype=torch.int64, device="cuda"),
                            unique_node_counts=torch.zeros(expected_world_size // 4, dtype=torch.int64, device="cuda"),
                            metadata_calls=0, layout_calls=0)

        def on_train_step_end(self, context):
            step = int(context.state.train_state.step)
            layers = []
            for layer, record in self.current.items():
                experts = record["experts"].cpu().tolist()
                splits = record["splits"].cpu().tolist()
                errors = [int(record[key].item()) for key in ("bad_topk", "bad_groups", "nonfinite", "bad_splits")]
                peers = self.sites[layer][1]
                traffic = Counter()
                for expert, count in enumerate(experts):
                    traffic[self.owners[layer][expert]] += count
                expected = accumulation * token_positions * 8
                if (record["calls"] != accumulation or record["dispatch_calls"] != accumulation or any(errors)
                        or sum(experts) != expected or sum(splits) != expected
                        or splits != [traffic[peer] for peer in peers] or context.skipped_iter):
                    raise RuntimeError(f"invalid routing accounting at rank {self.rank}, step {step}, layer {layer}")
                extra = {} if record_receives else None
                if record_receives:
                    if record["receive_calls"] != accumulation:
                        raise RuntimeError("received expert-count coverage differs from accumulation")
                    expert_ids = list(self.sites[layer][0].local_expert_indices)
                    if len(expert_ids) != experts_per_owner:
                        raise RuntimeError("received expert-count ID order differs from actual local ownership")
                    extra.update(received_expert_ids=expert_ids,
                        received_expert_counts=list(record["received_expert_counts"]),
                        received_assignments=sum(record["received_expert_counts"]),
                        receive_calls=record["receive_calls"])
                if communication_backend == "uccl_ep":
                    if (int(record["bad_metadata"].item()) or int(record["bad_layout"].item())
                            or record["metadata_calls"] != accumulation or record["layout_calls"] != accumulation):
                        raise RuntimeError("UCCL selected metadata or deduplicated layout accounting failed")
                    rank_counts = record["unique_rank_counts"].cpu().tolist()
                    node_counts = record["unique_node_counts"].cpu().tolist()
                    bound = accumulation * token_positions
                    if (any(not 0 <= count <= bound for count in rank_counts + node_counts)
                            or not bound <= sum(node_counts) <= sum(rank_counts) <= expected):
                        raise RuntimeError("UCCL unique delivery counts violate token/assignment bounds")
                    extra.update(unique_token_rank_counts=[rank_counts[peers.index(rank)] for rank in range(expected_world_size)],
                        unique_token_node_counts=node_counts, unique_token_node_hosts=list(self.node_hosts[layer]),
                        layout_calls=record["layout_calls"], metadata_calls=record["metadata_calls"])
                local = traffic[self.rank]
                intra = sum(count for rank, count in traffic.items()
                            if rank != self.rank and self.hosts[rank] == self.hosts[self.rank])
                inter = sum(count for rank, count in traffic.items()
                            if self.hosts[rank] != self.hosts[self.rank])
                if local + intra + inter != expected:
                    raise RuntimeError("routing locality does not reconcile")
                layers.append({"layer": layer, "expert_counts": experts,
                               "rank_traffic": [traffic[rank] for rank in range(expected_world_size)],
                               "routed_tokens": accumulation * token_positions, "assignments": expected,
                               "local": local, "intra_node": intra, "inter_node": inter})
                if record_receives:
                    layers[-1].update(extra)
            self.current = None
            self.records.append({"step": step, "layers": layers})

        def on_train_end(self, context):
            if [record["step"] for record in self.records] != list(range(expected_steps)):
                raise RuntimeError("incomplete routing audit step coverage")
            self._detach()
            (run_dir / "raw" / f"routing_audit_rank-{self.rank}.json").write_text(json.dumps({
                "rank": self.rank, "steps": self.records, "valid": True,
                "num_groups": num_groups, "group_topk": group_topk,
                **({"communication_backend": communication_backend, "record_receives": True}
                   if record_receives else {}),
            }, separators=(",", ":")))

    return RoutingAudit()
