"""GPU preflight for first/final per-step export and native grouped routing."""

import gzip
import json
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
import sys
from unittest.mock import patch

import torch

from profiling.ampa_profile.deep_trace import make_deep_trace_callback
from profiling.ampa_profile.routing_audit import make_routing_audit_callback


class _Labels:
    def __init__(self, *args):
        self.layers = list(range(48))
        self.counts = {}
        self.payload_bytes = {}
        self.payload_dtypes = {}
        self.attached = False

    def attach(self):
        self.attached = True

    def detach(self):
        self.attached = False

    def validate_counts(self, accumulation):
        assert accumulation == 16


class AllStepPreflight(unittest.TestCase):
    def _parse_all_step_cli(self, ranks):
        from profiling import run_pretrain

        with TemporaryDirectory() as directory:
            argv = ["run_pretrain.py", "--run-dir", directory, "--pretrained-checkpoint", directory,
                    "--hf-revision", "test-revision", "--dataset-data-path", "test-prefix",
                    "--dataset-manifest", "test-manifest", "--topology-variant", "cp2_ep8",
                    "--lightweight-profiling", "off", "--torch-profile-all-steps",
                    "--torch-profile-ranks", ranks]
            with patch.object(sys, "argv", argv), \
                    patch.object(run_pretrain, "_validate_c0_manifest"), \
                    patch.object(run_pretrain, "_validate_dataset_manifest"), \
                    patch.object(run_pretrain, "_dataset_prefixes", return_value=[]):
                return run_pretrain.parse_args()

    def test_all_step_cli_requires_all_eight_ranks(self):
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            self._parse_all_step_cli("0")

    def test_all_step_cli_accepts_all_eight_ranks(self):
        self.assertEqual(self._parse_all_step_cli("0,1,2,3,4,5,6,7").torch_profile_ranks, tuple(range(8)))

    def _exercise_audit(self, corrupt=False):
        routing_map = torch.zeros(2048, 128, device="cuda", dtype=torch.bool)
        routing_map[:, [0, 1, 2, 3, 16, 17, 18, 19]] = True
        probabilities = routing_map.to(torch.float32) / 8

        class Dispatcher:
            ep_group = object()

            def __init__(self, layer):
                self.calls = 0
                self.layer = layer

            def preprocess(self, tensor):
                self.calls += 1
                self.input_splits = tensor.sum(dim=0, dtype=torch.int64).view(8, 16).sum(dim=1)
                if corrupt and self.layer == 0 and self.calls in (1, 2):
                    change = 1 if self.calls == 1 else -1
                    self.input_splits[0] += change
                    self.input_splits[1] -= change
                return tensor.sum(dim=0)[:16]

        class MoE:
            def __init__(self, layer):
                self.layer_number = layer + 1
                self.local_expert_indices = list(range(16))
                self.config = SimpleNamespace(num_moe_experts=128, moe_router_topk=8,
                                              moe_router_num_groups=8, moe_router_group_topk=2)
                self.token_dispatcher = Dispatcher(layer)

            def route(self, hidden_states):
                return probabilities, routing_map

        sites = [MoE(layer) for layer in range(48)]
        model = SimpleNamespace(modules=lambda: iter(sites))

        def gather(output, local):
            for rank in range(8):
                output[rank] = {"rank": rank, "hostname": f"node-{rank // 4}",
                                "experts": {str(layer): list(range(rank * 16, (rank + 1) * 16))
                                            for layer in range(48)}}

        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "raw").mkdir()
            (root / "config").mkdir()
            audit = make_routing_audit_callback(run_dir=root, expected_steps=1, num_groups=8, group_topk=2)
            context = SimpleNamespace(model=[model], skipped_iter=False,
                                      state=SimpleNamespace(train_state=SimpleNamespace(step=0)))
            with patch("torch.distributed.get_rank", return_value=0), \
                    patch("torch.distributed.get_world_size", return_value=8), \
                    patch("torch.distributed.all_gather_object", side_effect=gather), \
                    patch("torch.distributed.get_process_group_ranks", return_value=list(range(8))):
                audit.on_train_start(context)
                audit.on_train_step_start(context)
                for _ in range(16):
                    for site in sites:
                        result = site.route(None)
                        self.assertIs(result[0], probabilities)
                        self.assertIs(result[1], routing_map)
                        site.token_dispatcher.preprocess(result[1])
                if corrupt:
                    with self.assertRaisesRegex(RuntimeError, "invalid routing accounting"):
                        audit.on_train_step_end(context)
                else:
                    audit.on_train_step_end(context)
                    audit.on_train_end(context)
                    record = json.loads((root / "raw" / "routing_audit_rank-0.json").read_text())
                    self.assertTrue(record["valid"])
                    self.assertEqual(record["steps"][0]["layers"][0]["assignments"], 16 * 2048 * 8)
                    self.assertNotIn("route", vars(sites[0]))
                    self.assertNotIn("preprocess", vars(sites[0].token_dispatcher))

    def test_audit_preserves_tensors_and_counts(self):
        self._exercise_audit()

    def test_audit_rejects_canceling_split_errors(self):
        self._exercise_audit(corrupt=True)

    def test_first_and_final_step_export(self):
        self.assertTrue(torch.cuda.is_available())
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "raw").mkdir()
            callback = make_deep_trace_callback(run_dir=root, start_step=0, capture_steps=2,
                                                selected_ranks=(0,), per_step_export=True)
            context = SimpleNamespace(state=SimpleNamespace(train_state=SimpleNamespace(step=0)),
                                      model=[], optimizer=None, skipped_iter=False)
            with patch("torch.distributed.get_rank", return_value=0), \
                    patch("torch.distributed.get_world_size", return_value=8), \
                    patch("profiling.ampa_profile.deep_trace._MethodLabels", _Labels):
                callback.on_train_start(context)
                for step in range(2):
                    context.state.train_state.step = step
                    callback.on_train_step_start(context)
                    tensor = torch.ones(4096, device="cuda")
                    for _ in range(3):
                        tensor = tensor * 1.01
                    callback.on_train_step_end(context)
                callback.on_train_end(context)
            for step in range(2):
                path = root / "traces" / "torch_profile" / f"step-{step:05d}" / "rank-0.json.gz"
                with gzip.open(path, "rt") as stream:
                    trace = json.load(stream)
                scopes = [event["name"] for event in trace["traceEvents"]
                          if event.get("cat") == "user_annotation" and event.get("name", "").startswith("AMPA/STEP/")]
                self.assertEqual(scopes, [f"AMPA/STEP/{step}"])
                self.assertTrue(any(event.get("cat") == "kernel" for event in trace["traceEvents"]))
            record = json.loads((root / "raw" / "deep_trace_rank-0.json").read_text())
            self.assertEqual([item["step"] for item in record["steps"]], [0, 1])

    def test_native_group_topk(self):
        from megatron.core.transformer.moe.moe_utils import group_limited_topk

        torch.manual_seed(1234)
        for scores in (torch.rand(257, 128, device="cuda"), torch.ones(257, 128, device="cuda")):
            values, indices = group_limited_topk(scores, 8, 257, 128, 8, 2)
            self.assertEqual(tuple(indices.shape), (257, 8))
            self.assertTrue(torch.isfinite(values).all().item())
            groups = indices // 16
            counts = torch.zeros(257, 8, device="cuda", dtype=torch.int64)
            counts.scatter_add_(1, groups, torch.ones_like(groups))
            self.assertTrue(((counts > 0).sum(dim=1) <= 2).all().item())
            self.assertTrue((indices.sort(dim=1).values.diff(dim=1) != 0).all().item())


if __name__ == "__main__":
    unittest.main()
