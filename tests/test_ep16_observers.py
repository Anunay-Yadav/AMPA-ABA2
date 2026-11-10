"""EP16 metadata contracts and real CUDA routing-audit checks for preflight."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from profiling.ampa_profile.deep_trace import DeepTraceError, make_deep_trace_callback
from profiling.ampa_profile.provenance import (
    ProvenanceInvariantError, _EP16_RUNTIME_TOPOLOGIES,
    validate_rank_placement, validate_runtime_groups, validate_runtime_topology,
)
from profiling.ampa_profile.routing_audit import make_routing_audit_callback
from profiling.ampa_profile.sample_fingerprint import (
    SampleFingerprintError, make_sample_fingerprint_callback,
)


def _placement(cp):
    """Permute expert ownership independently of physical rank placement."""
    ownership = list(reversed(range(16)))
    return [{
        "rank": rank, "hostname": f"node-{rank // 4}",
        "slurm_node_id": str(rank // 4), "slurm_local_id": str(rank % 4),
        "cuda_visible_device_count": 1,
        "physical_gpu_uuid": f"GPU-00000000-0000-0000-0000-{rank:012d}",
        "physical_gpu_identity_source": "cuda_driver.cuDeviceGetUuid(cuda_runtime.cudaGetDevice())",
        "process_groups": {
            "cp": [rank] if cp == 1 else [rank // 2 * 2, rank // 2 * 2 + 1],
            "dense_dp": list(range(16)) if cp == 1 else list(range(rank % 2, 16, 2)),
            "dense_dp_cp": list(range(16)), "ep": list(range(16)),
        },
        "local_expert_indices": list(range(ownership[rank] * 8, (ownership[rank] + 1) * 8)),
    } for rank in range(16)]


class EP16ObserverContracts(unittest.TestCase):
    def test_accepts_cp1_and_cp2_actual_groups_and_permuted_ownership(self):
        for cp in (1, 2):
            placement = validate_rank_placement(_placement(cp), expected_world_size=16)
            validate_runtime_groups(placement, expected_world_size=16, cp=cp)
            topology = _EP16_RUNTIME_TOPOLOGIES[f"cp{cp}_ep16"]
            self.assertEqual(validate_runtime_topology(dict(topology), expected=topology), topology)

    def test_rejects_inconsistent_membership_and_duplicate_expert_owner(self):
        placement = _placement(2)
        placement[1]["process_groups"]["cp"] = [1, 0]
        with self.assertRaisesRegex(ProvenanceInvariantError, "inconsistent cp"):
            validate_runtime_groups(placement, expected_world_size=16, cp=2)
        placement = _placement(1)
        placement[1]["local_expert_indices"] = placement[0]["local_expert_indices"]
        with self.assertRaisesRegex(ProvenanceInvariantError, "disjoint"):
            validate_runtime_groups(placement, expected_world_size=16, cp=1)

    def test_rejects_cross_node_cp_pairs_and_wrong_host_count(self):
        placement = _placement(2)
        placement[0]["process_groups"]["cp"] = [0, 4]
        with self.assertRaisesRegex(ProvenanceInvariantError, "CP2 peers cross"):
            validate_runtime_groups(placement, expected_world_size=16, cp=2)
        placement = _placement(1)
        placement[12]["hostname"] = "node-0"
        with self.assertRaisesRegex(ProvenanceInvariantError, "four hosts with four ranks"):
            validate_rank_placement(placement, expected_world_size=16)

    def test_observers_require_requested_world_size(self):
        with TemporaryDirectory() as directory:
            callbacks = [
                (make_deep_trace_callback(run_dir=Path(directory), start_step=0, capture_steps=2,
                                         selected_ranks=tuple(range(16)), accumulation=4,
                                         expected_world_size=16), DeepTraceError),
                (make_sample_fingerprint_callback(run_dir=Path(directory), expected_steps=2,
                                                  accumulation=4, expected_world_size=16),
                 SampleFingerprintError),
            ]
            with patch("torch.distributed.get_rank", return_value=0):
                for callback, exception in callbacks:
                    with patch("torch.distributed.get_world_size", return_value=16):
                        callback.on_train_start(None)
                    with patch("torch.distributed.get_world_size", return_value=8), self.assertRaises(exception):
                        callback.on_train_start(None)

    def _exercise_audit(self, *, cp, grouped, corrupt=False,
                        diagnostic_num_groups=4, group_topk=2, cross_node_groups=False):
        import torch

        positions, accumulation = (4096, 4) if cp == 1 else (2048, 8)
        routing_map = torch.zeros(positions, 128, device="cuda", dtype=torch.bool)
        selected_experts = ([0, 8, 16, 24, 32, 40, 48, 56] if diagnostic_num_groups == 4
                            else [0, 1, 8, 9, 16, 17, 24, 25] if group_topk == 4
                            else [0, 1, 2, 3, 8, 9, 10, 11])
        routing_map[:, selected_experts] = True
        probabilities = routing_map.to(torch.float32) / 8
        placement = _placement(cp)
        if cross_node_groups:
            placement[0]["local_expert_indices"], placement[4]["local_expert_indices"] = (
                placement[4]["local_expert_indices"], placement[0]["local_expert_indices"])
        peer_order = list(reversed(range(16)))
        destination = {expert: record["rank"] for record in placement
                       for expert in record["local_expert_indices"]}

        class Dispatcher:
            ep_group = object()

            def __init__(self, layer):
                self.layer, self.calls = layer, 0

            def preprocess(self, tensor):
                self.calls += 1
                counts = tensor.sum(dim=0, dtype=torch.int64)
                self.input_splits = torch.stack([
                    counts[placement[peer]["local_expert_indices"]].sum() for peer in peer_order
                ])
                if corrupt and self.layer == 0 and self.calls in (1, 2):
                    change = 1 if self.calls == 1 else -1
                    self.input_splits[0] += change
                    self.input_splits[1] -= change
                return tensor

        class MoE:
            def __init__(self, layer):
                self.layer_number = layer + 1
                self.local_expert_indices = placement[0]["local_expert_indices"]
                self.config = SimpleNamespace(num_moe_experts=128, moe_router_topk=8,
                                              moe_router_num_groups=diagnostic_num_groups if grouped else None,
                                              moe_router_group_topk=group_topk if grouped else None)
                self.token_dispatcher = Dispatcher(layer)

            def route(self, hidden_states):
                return probabilities, routing_map

        sites = [MoE(layer) for layer in range(48)]
        model = SimpleNamespace(modules=lambda: iter(sites))

        def gather(output, local):
            for record in placement:
                output[record["rank"]] = {"rank": record["rank"], "hostname": record["hostname"],
                                           "experts": {str(layer): record["local_expert_indices"]
                                                       for layer in range(48)}}

        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "raw").mkdir()
            (root / "config").mkdir()
            audit = make_routing_audit_callback(
                run_dir=root, expected_steps=1, num_groups=diagnostic_num_groups if grouped else None,
                group_topk=group_topk if grouped else None, expected_world_size=16,
                accumulation=accumulation, token_positions=positions,
                diagnostic_num_groups=diagnostic_num_groups,
            )
            context = SimpleNamespace(model=[model], skipped_iter=False,
                                      state=SimpleNamespace(train_state=SimpleNamespace(step=0)))
            with patch("torch.distributed.get_rank", return_value=0), \
                    patch("torch.distributed.get_world_size", return_value=16), \
                    patch("torch.distributed.all_gather_object", side_effect=gather), \
                    patch("torch.distributed.get_process_group_ranks", return_value=peer_order):
                audit.on_train_start(context)
                audit.on_train_step_start(context)
                for _ in range(accumulation):
                    for site in sites:
                        result = site.route(None)
                        self.assertIs(result[0], probabilities)
                        self.assertIs(result[1], routing_map)
                        site.token_dispatcher.preprocess(result[1])
                if corrupt:
                    with self.assertRaisesRegex(RuntimeError, "invalid routing accounting"):
                        audit.on_train_step_end(context)
                    return
                audit.on_train_step_end(context)
                audit.on_train_end(context)
            owners = json.loads((root / "config" / "routing_ownership.json").read_text())
            group_size = 128 // diagnostic_num_groups
            self.assertEqual(owners["group_owners"]["0"],
                             [sorted({destination[expert] for expert in range(start, start + group_size)})
                              for start in range(0, 128, group_size)])
            if diagnostic_num_groups == 16:
                self.assertTrue(all(len(group) == 1 for group in owners["group_owners"]["0"]))
            if diagnostic_num_groups == 4:
                self.assertTrue(all(len(hosts) == 1 for hosts in owners["group_hosts"]["0"]))
                self.assertEqual(len({hosts[0] for hosts in owners["group_hosts"]["0"]}), 4)
            record = json.loads((root / "raw" / "routing_audit_rank-0.json").read_text())
            layer = record["steps"][0]["layers"][0]
            self.assertEqual(layer["routed_tokens"], accumulation * positions)
            self.assertEqual(layer["assignments"], accumulation * positions * 8)
            self.assertEqual(layer["rank_traffic"], [sum(accumulation * positions for expert in
                                                      selected_experts
                                                      if destination[expert] == rank)
                                                   for rank in range(16)])
            self.assertEqual(layer["local"] + layer["intra_node"] + layer["inter_node"], layer["assignments"])
            self.assertNotIn("route", vars(sites[0]))
            self.assertNotIn("preprocess", vars(sites[0].token_dispatcher))

    def test_cuda_cp1_grouped_and_ordinary_preserve_tensors_and_ownership(self):
        for grouped in (False, True):
            self._exercise_audit(cp=1, grouped=grouped)

    def test_cuda_cp2_grouped_and_ordinary_preserve_tensors_and_ownership(self):
        for grouped in (False, True):
            self._exercise_audit(cp=2, grouped=grouped)

    def test_cuda_ep16_rejects_canceling_split_errors(self):
        self._exercise_audit(cp=1, grouped=True, corrupt=True)

    def test_cuda_four_groups_require_four_distinct_physical_nodes(self):
        with self.assertRaisesRegex(RuntimeError, "one physical node"):
            self._exercise_audit(cp=1, grouped=True, cross_node_groups=True)

    def test_cuda_cp1_sixteen_groups_top2_and_top4(self):
        for group_topk in (2, 4):
            for grouped in (False, True):
                self._exercise_audit(cp=1, grouped=grouped,
                                     diagnostic_num_groups=16, group_topk=group_topk)

    def test_cuda_cp2_sixteen_groups_top2_and_top4(self):
        for group_topk in (2, 4):
            for grouped in (False, True):
                self._exercise_audit(cp=2, grouped=grouped,
                                     diagnostic_num_groups=16, group_topk=group_topk)

    def test_native_four_group_topk_limits_eight_destination_owners(self):
        import torch
        from megatron.core.transformer.moe.moe_utils import group_limited_topk

        torch.manual_seed(1234)
        for scores in (torch.rand(257, 128, device="cuda"), torch.ones(257, 128, device="cuda")):
            values, indices = group_limited_topk(scores, 8, 257, 128, 4, 2)
            self.assertEqual(tuple(indices.shape), (257, 8))
            self.assertTrue(torch.isfinite(values).all().item())
            counts = torch.zeros(257, 4, device="cuda", dtype=torch.int64)
            counts.scatter_add_(1, indices // 32, torch.ones_like(indices))
            self.assertTrue(((counts > 0).sum(dim=1) <= 2).all().item())
            self.assertTrue((indices.sort(dim=1).values.diff(dim=1) != 0).all().item())

    def test_native_sixteen_groups_select_top2_or_top4(self):
        import torch
        from megatron.core.transformer.moe.moe_utils import group_limited_topk

        torch.manual_seed(1234)
        placement = _placement(1)
        owner_by_expert = torch.tensor([next(record["rank"] for record in placement
                                             if expert in record["local_expert_indices"])
                                       for expert in range(128)], device="cuda")
        for group_topk in (2, 4):
            for scores in (torch.rand(257, 128, device="cuda"), torch.ones(257, 128, device="cuda")):
                values, indices = group_limited_topk(scores, 8, 257, 128, 16, group_topk)
                self.assertEqual(tuple(indices.shape), (257, 8))
                self.assertTrue(torch.isfinite(values).all().item())
                group_counts = torch.zeros(257, 16, device="cuda", dtype=torch.int64)
                group_counts.scatter_add_(1, indices // 8, torch.ones_like(indices))
                owner_counts = torch.zeros_like(group_counts)
                owner_counts.scatter_add_(1, owner_by_expert[indices], torch.ones_like(indices))
                self.assertTrue(((group_counts > 0).sum(dim=1) <= group_topk).all().item())
                self.assertTrue(torch.equal((group_counts > 0).sum(dim=1), (owner_counts > 0).sum(dim=1)))
                self.assertTrue((indices.sort(dim=1).values.diff(dim=1) != 0).all().item())


if __name__ == "__main__":
    unittest.main()
