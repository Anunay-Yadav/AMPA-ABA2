"""Check the short run's real-update window and unchanged cache budget."""
import argparse
from contextlib import redirect_stderr
from io import StringIO
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from profiling import run_pretrain_debug as debug
import test_uccl_pretrain_config as native_fixtures


class DebugEntrypointTests(unittest.TestCase):
    def args(self, **changes):
        values = dict(topology_variant="cp1_ep16", launch_policy="torchrun_node4", smoke_steps=None,
            nsys_capture_steps=0, torch_profile_steps=0, audit_received_experts=True,
            routing_audit=True, sample_fingerprints=True, log_interval=1,
            warmup_steps=5, measured_steps=20, completed_steps=25)
        values.update(changes)
        return argparse.Namespace(**values)

    def environment(self):
        return patch.dict(os.environ, {"SLURM_JOB_PARTITION": "debug", "SLURM_JOB_NUM_NODES": "4", "SLURM_JOB_ID": "12345"})

    def test_original_parser_validates_the_unmodified_cache_budget(self):
        seen = []
        def parse():
            seen.extend(sys.argv)
            return self.args()
        original = sys.argv
        with self.environment(), patch.object(debug, "BASE_PARSE_ARGS", side_effect=parse):
            args = debug.parse_debug_args(["--run-dir", "/tmp/run", "--warmup-steps", "5", "--measured-steps=5"])
        self.assertIs(sys.argv, original)
        self.assertEqual(seen[-4:], ["--warmup-steps", "5", "--measured-steps", "20"])
        self.assertEqual((args.warmup_steps, args.measured_steps, args.completed_steps), (5, 5, 10))
        self.assertIsNone(args.smoke_steps)

    def test_invalid_window_and_allocation_are_rejected_before_canonical_parser(self):
        for extra in (["--measured-steps", "10"], ["--warmup-steps", "2"]):
            with self.environment(), patch.object(debug, "BASE_PARSE_ARGS") as parser, redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                debug.parse_debug_args(extra)
            parser.assert_not_called()
        for changes in ({"SLURM_JOB_PARTITION": "preemptable"}, {"SLURM_JOB_NUM_NODES": "1"}, {"SLURM_JOB_ID": ""}):
            with self.environment(), patch.dict(os.environ, changes), redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                debug.parse_debug_args([])

    def test_required_observers_and_normal_error_propagation_are_preserved(self):
        for changes in ({"sample_fingerprints": False}, {"audit_received_experts": False}, {"smoke_steps": 2}, {"topology_variant": "cp2_ep16"}):
            with self.environment(), patch.object(debug, "BASE_PARSE_ARGS", return_value=self.args(**changes)), redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                debug.parse_debug_args([])
        with self.environment(), patch.object(debug, "BASE_PARSE_ARGS", side_effect=ValueError("immutable input failed")), self.assertRaisesRegex(ValueError, "immutable input"):
            debug.parse_debug_args([])

    def test_config_keeps_sample_budget_but_exits_at_ten_and_records_actual_window(self):
        with TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "config").mkdir()
            args = self.args(warmup_steps=5, measured_steps=5, completed_steps=10, run_dir=run)
            def build(cached):
                self.assertEqual((cached.warmup_steps, cached.measured_steps, cached.smoke_steps), (5, 20, 10))
                return SimpleNamespace(train=SimpleNamespace(train_iters=25, exit_interval=10))
            with patch.object(debug, "BASE_BUILD_CONFIG", side_effect=build), patch.dict(os.environ, {"RANK": "0"}):
                result = debug.build_debug_config(args)
            self.assertEqual((result.train.train_iters, result.train.exit_interval), (25, 10))
            self.assertEqual((args.measured_steps, args.completed_steps, args.smoke_steps), (5, 10, None))
            record = json.loads((run / "config/debug_execution_contract.json").read_text())
            self.assertEqual((record["measured_steps"], record["completed_steps"], record["cache_budget_samples"]), (5, 10, 1600))

    def test_native_budget_mismatch_cannot_silently_run_twenty_five_updates(self):
        args = self.args(warmup_steps=5, measured_steps=5, completed_steps=10)
        wrong = SimpleNamespace(train=SimpleNamespace(train_iters=10, exit_interval=None))
        with patch.object(debug, "BASE_BUILD_CONFIG", return_value=wrong), self.assertRaisesRegex(ValueError, "sample budget"):
            debug.build_debug_config(args)

    def test_real_canonical_parser_and_config_builder_support_all_eight_short_cases(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config").mkdir()
            base = ["--run-dir", str(root), "--pretrained-checkpoint", str(root),
                "--hf-revision", "test", "--dataset-data-path", "test",
                "--dataset-manifest", str(root / "data.json"), "--topology-variant", "cp1_ep16",
                "--lightweight-profiling", "off", "--sample-fingerprints", "--routing-audit",
                "--audit-received-experts", "--launch-policy", "torchrun_node4", "--log-interval", "1",
                "--warmup-steps", "5", "--measured-steps", "5"]
            def c0(args, parser):
                args.c0_identity_manifest_sha256 = "test-c0"
            def dataset(args, parser):
                self.assertEqual((args.warmup_steps, args.measured_steps, args.completed_steps), (5, 20, 25))
                args.indexed_manifest = {}
            fixture = native_fixtures.UcclPretrainConfigTests()
            for backend in ("alltoall", "uccl_ep"):
                for grouped in (False, True):
                    for profiled in (False, True):
                        flags = [*base, "--communication-backend", backend]
                        if backend == "uccl_ep":
                            flags += ["--uccl-build-dir", str(root)]
                        if grouped:
                            flags += ["--router-num-groups", "4", "--router-group-topk", "2"]
                        if profiled:
                            flags += ["--torch-profile-all-steps", "--torch-profile-ranks", ",".join(map(str, range(16)))]
                        with self.subTest(backend=backend, grouped=grouped, profiled=profiled), \
                             self.environment(), patch.object(debug.canonical, "_validate_c0_manifest", side_effect=c0), \
                             patch.object(debug.canonical, "_validate_dataset_manifest", side_effect=dataset), \
                             patch.object(debug.canonical, "_dataset_prefixes", return_value=[]):
                            args = debug.parse_debug_args(flags)
                        with patch.object(debug, "BASE_BUILD_CONFIG", side_effect=fixture.build), patch.dict(os.environ, {"RANK": "1"}):
                            config = debug.build_debug_config(args)
                        self.assertEqual((args.completed_steps, args.warmup_steps, args.measured_steps), (10, 5, 5))
                        self.assertEqual((config.train.train_iters, config.train.exit_interval), (25, 10))
                        self.assertEqual(config.scheduler.lr_decay_iters, 25)
                        self.assertEqual(config.model.moe_token_dispatcher_type, "flex" if backend == "uccl_ep" else "alltoall")


if __name__ == "__main__":
    unittest.main()
