"""Entrypoint and immutable-cache evidence guards for the EP16 experiment."""
import copy
from contextlib import redirect_stderr
from io import StringIO
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from profiling import run_pretrain
from tools import ep16_indexed_manifest as cache
from tools.bridge_indexed_consumer import expected_bridge_indexed_consumer


class EP16Runtime(unittest.TestCase):
    def parse(self, cp=1, ranks=None, smoke=False):
        ranks = ranks if ranks is not None else ",".join(map(str, range(16)))
        with TemporaryDirectory() as directory:
            argv = ["run_pretrain.py", "--run-dir", directory, "--pretrained-checkpoint", directory,
                    "--hf-revision", "test", "--dataset-data-path", "test", "--dataset-manifest", "test",
                    "--topology-variant", f"cp{cp}_ep16", "--lightweight-profiling", "off",
                    "--torch-profile-all-steps", "--torch-profile-ranks", ranks,
                    "--sample-fingerprints", "--routing-audit", "--log-interval", "1",
                    "--router-num-groups", "4", "--router-group-topk", "2"]
            if smoke:
                argv += ["--smoke-steps", "2"]
            with patch.object(sys, "argv", argv), patch.object(run_pretrain, "_validate_c0_manifest"), \
                    patch.object(run_pretrain, "_validate_dataset_manifest"), \
                    patch.object(run_pretrain, "_dataset_prefixes", return_value=[]):
                return run_pretrain.parse_args()

    def test_cp1_and_cp2_require_all16_ranks(self):
        for cp in (1, 2):
            args = self.parse(cp)
            self.assertEqual(args.torch_profile_ranks, tuple(range(16)))
            self.assertEqual(args.completed_steps, 25)
            self.assertEqual(args.audit_num_groups, 4)
            with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                self.parse(cp, ranks="0,1,2,3,4,5,6,7")

    def test_smoke_retains_full_cache_budget(self):
        args = self.parse(smoke=True)
        self.assertEqual(args.completed_steps, 2)
        self.assertEqual(args.warmup_steps + args.measured_steps, 25)
        self.assertEqual(args.global_batch_size * (args.warmup_steps + args.measured_steps), 1600)

    def derivative(self, cp):
        parent = json.loads(cache.PARENT.read_text())
        result = cache._derivative(parent, f"cp{cp}_ep16")
        probe = parent["megatron_cache"]["bridge_cache_hit_probe"]
        result["runtime_cache_reuse"] = {
            "parent_path": str(cache.PARENT), "parent_sha256": cache.sha256(cache.PARENT),
            "probe": {"status": "passed", "train_samples": 1600,
                      "dataset_length": probe["train_dataset_length"], "sample_keys": probe["sample_keys"],
                      "cache_sha256_before": probe["cache_inventory_sha256_before"],
                      "cache_sha256_after": probe["cache_inventory_sha256_after"],
                      "actual_bridge_consumer": expected_bridge_indexed_consumer(result)},
        }
        return result

    def test_cp_derivatives_preserve_preparation_identity(self):
        parent = json.loads(cache.PARENT.read_text())
        for cp in (1, 2):
            result = self.derivative(cp)
            cache.validate_runtime_derivative(result, f"cp{cp}_ep16")
            self.assertEqual(result["cache_preparation"], parent["cache_preparation"])
            self.assertEqual(result["materialized_artifacts"], parent["materialized_artifacts"])

    def test_rejects_equal_but_forged_cache_hashes(self):
        result = self.derivative(1)
        result["runtime_cache_reuse"]["probe"].update(cache_sha256_before="0" * 64, cache_sha256_after="0" * 64)
        with self.assertRaisesRegex(ValueError, "cache-hit"):
            cache.validate_runtime_derivative(result, "cp1_ep16")

    def test_rejects_changed_cache_or_sample_budget(self):
        original = self.derivative(1)
        for key in ("dataset_length", "train_samples"):
            result = copy.deepcopy(original)
            result["runtime_cache_reuse"]["probe"][key] = 128
            with self.assertRaises(ValueError):
                cache.validate_runtime_derivative(result, "cp1_ep16")
        result = copy.deepcopy(original)
        result["cache_preparation"]["cache_contract"]["prepare_cache"]["parallelism"]["ep"] = 16
        with self.assertRaisesRegex(ValueError, "immutable"):
            cache.validate_runtime_derivative(result, "cp1_ep16")
