"""CPU-only checks of cache identity, selected flags, and callback ownership."""
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
from unittest.mock import Mock, patch

from profiling import run_pretrain_ep_overlap as adapter
from profiling import run_pretrain_uccl_budget as budget


ALLOCATION = {"SLURM_JOB_PARTITION": "debug", "SLURM_JOB_NUM_NODES": "4",
              "SLURM_JOB_ID": "123456", "CUDA_DEVICE_MAX_CONNECTIONS": "32",
              "FI_MR_CACHE_MONITOR": "disabled", "RANK": "0"}


def selected(root, **changes):
    values = dict(run_dir=root, case="ordinary", overlap_capture="off", communication_backend="alltoall",
        communication_sms=20, execution_steps=10, completed_steps=10, smoke_steps=10,
        warmup_steps=5, measured_steps=5, overlap_source_sha256="a" * 64,
        uccl_budget_setup=None, torch_profile_all_steps=False)
    return argparse.Namespace(**{**values, **changes})


def config_fixture(steps=10):
    return SimpleNamespace(model=SimpleNamespace(hidden_size=2048, moe_router_topk=8,
        **{key: False for key in adapter.OVERLAP_FLAGS}),
        ddp=SimpleNamespace(delay_wgrad_compute=False),
        train=SimpleNamespace(train_iters=25, exit_interval=steps),
        profiling=SimpleNamespace(), comm_overlap=object())


class OverlapEntrypointTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, ALLOCATION, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.run = self.root / "runs/current"
        (self.run / "config").mkdir(parents=True)
        self.c0 = self.root / "checkpoint"
        self.c0.mkdir()
        self.prefix = self.root / "data"
        Path(f"{self.prefix}.idx").touch()
        Path(f"{self.prefix}.bin").touch()
        self.build = self.root / "build"
        self.build.mkdir()
        self.dataset = {
            "validated_training_data_path": str(self.prefix),
            "validated_workload": {"micro_batch_size": 1, "dense_data_parallel_size": 16,
                "gradient_accumulation": 4, "global_batch_size": 64, "sequence_length": 4096,
                "warmup_steps": 5, "measured_steps": 20, "sampler_seed": 1234},
            "model_and_tokenizer": {"model_id": "Qwen/Qwen3-30B-A3B-Base", "revision": "fixed",
                "tokenizer_id": "Qwen/Qwen3-30B-A3B-Base", "tokenizer_revision": "fixed"},
        }
        self.manifest = self.run / "config/dataset_manifest.json"
        self.manifest.write_text(json.dumps(self.dataset))
        self.context = {"run_dir": str(self.run), "dataset_manifest_path": str(self.manifest),
            "pretrained_checkpoint": str(self.c0), "hf_model_id": "Qwen/Qwen3-30B-A3B-Base",
            "hf_revision": "fixed", "c0_identity_manifest_sha256": "b" * 64,
            "c0_hf_snapshot_manifest_sha256": "c" * 64, "indexed_manifest": self.dataset,
            "source_manifest": {"sha256": "a" * 64}}

    def argv(self, case="ordinary", capture="off", steps=10, sms=20):
        values = ["--run-dir", str(self.run), "--case", case, "--capture", capture,
            "--communication-sms", str(sms), "--execution-steps", str(steps),
            "--pretrained-checkpoint", str(self.c0), "--hf-revision", "fixed",
            "--dataset-data-path", str(self.prefix), "--dataset-manifest", str(self.manifest),
            "--topology-variant", "cp1_ep16", "--lightweight-profiling", "off",
            "--warmup-steps", "5", "--measured-steps", "5", "--log-interval", "1",
            "--sample-fingerprints", "--routing-audit", "--audit-num-groups", "4",
            "--launch-policy", "torchrun_node4", "--audit-received-experts"]
        if case in ("grouped", "uccl"):
            values += ["--router-num-groups", "4", "--router-group-topk", "2"]
        if case.startswith("uccl"):
            values += ["--communication-backend", "uccl_ep", "--uccl-build-dir", str(self.build)]
        if capture == "torch":
            values += ["--torch-profile-all-steps", "--torch-profile-ranks", ",".join(map(str, range(16)))]
        return values

    def parse(self, *args, **kwargs):
        with patch.object(adapter, "load_context", return_value=self.context):
            return adapter.parse_overlap_args(self.argv(*args, **kwargs))

    def test_real_parser_all_cases_and_capture_modes_keep_cache_and_capture_truth(self):
        for case in adapter.CASES:
            for capture in ("off", "torch"):
                for steps in (2, 10):
                    with self.subTest(case=case, capture=capture, steps=steps):
                        args = self.parse(case, capture, steps)
                        self.assertEqual(args.completed_steps, steps)
                        self.assertEqual(args.smoke_steps, steps)
                        self.assertEqual((args.warmup_steps, args.measured_steps), (1, 1) if steps == 2 else (5, 5))
                        self.assertEqual(args.indexed_manifest["validated_workload"]["measured_steps"], 20)
                        self.assertEqual(args.torch_profile_all_steps, capture == "torch")
                        self.assertEqual(args.nsys_capture_steps, 0)
        for sms in adapter.BUDGETS:
            for steps in (2, 10):
                args = self.parse("uccl_ordinary", "nsys", steps, sms)
                self.assertFalse(args.torch_profile_all_steps)
                self.assertEqual(args.communication_sms, sms)

    def test_wrong_router_backend_or_second_profiler_is_rejected(self):
        changes = [(["--communication-backend", "uccl_ep", "--uccl-build-dir", str(self.build)], "off"),
                   (["--router-num-groups", "4", "--router-group-topk", "2"], "off"),
                   (["--torch-profile-all-steps", "--torch-profile-ranks", ",".join(map(str, range(16)))], "off"),
                   (["--nsys-capture-steps", "1"], "off"), (["--smoke-steps", "2"], "off"),
                   (["--learning-rate", "0.01"], "off")]
        for extra, capture in changes:
            with self.subTest(extra=extra), patch.object(adapter, "load_context", return_value=self.context), \
                    redirect_stderr(StringIO()), self.assertRaises((ValueError, SystemExit)):
                adapter.parse_overlap_args([*self.argv(capture=capture), *extra])

    def test_invalid_sweep_and_window_selection_rejected(self):
        for case, capture, sms in [("ordinary", "nsys", 20), ("uccl", "nsys", 32),
                                    ("ordinary", "off", 12), ("uccl_ordinary", "torch", 40)]:
            with self.subTest(case=case, capture=capture, sms=sms), redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                self.parse(case, capture, sms=sms)
        with patch.object(adapter, "load_context", return_value=self.context), redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            adapter.parse_overlap_args([*self.argv(), "--measured-steps", "20"])

    def test_allocation_and_connections_checked_before_parser_or_runtime(self):
        for key, value in [("SLURM_JOB_PARTITION", "preemptable"), ("SLURM_JOB_NUM_NODES", "1"),
                           ("SLURM_JOB_ID", "bad"), ("CUDA_DEVICE_MAX_CONNECTIONS", "1")]:
            with self.subTest(key=key), patch.dict(os.environ, {key: value}), \
                    patch.object(adapter, "load_context") as loader, self.assertRaises(ValueError):
                adapter.parse_overlap_args(self.argv())
            loader.assert_not_called()

    def test_cached_validator_and_argv_restored_after_parse_failure(self):
        before = adapter.canonical._validate_c0_manifest, adapter.canonical._validate_dataset_manifest, sys.argv
        context = {**self.context, "hf_revision": "wrong"}
        with patch.object(adapter, "load_context", return_value=context), redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            adapter.parse_overlap_args(self.argv())
        self.assertEqual((adapter.canonical._validate_c0_manifest, adapter.canonical._validate_dataset_manifest), before[:2])
        self.assertIs(sys.argv, before[2])

    def test_historical_cache_identity_and_payload_existence_remain_required(self):
        changed = {**self.dataset, "validated_workload": {**self.dataset["validated_workload"], "measured_steps": 5}}
        with patch.object(adapter, "load_context", return_value={**self.context, "indexed_manifest": changed}), \
                redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            adapter.parse_overlap_args(self.argv())
        Path(f"{self.prefix}.bin").unlink()
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            self.parse()

    def test_builder_sets_selected_fields_before_model_and_never_touches_native_buffer(self):
        args = selected(self.run, execution_steps=2, completed_steps=2, smoke_steps=2, warmup_steps=1, measured_steps=1)
        cfg = config_fixture(2)
        with patch.object(adapter, "BASE_BUILD_CONFIG", return_value=cfg) as build, \
                patch.object(budget, "runtime_adapter") as runtime:
            result = adapter.build_overlap_config(args)
        runtime.assert_not_called()
        cached = build.call_args.args[0]
        self.assertIsNot(cached, args)
        self.assertEqual((cached.warmup_steps, cached.measured_steps, cached.smoke_steps), (5, 20, 2))
        self.assertEqual((args.warmup_steps, args.measured_steps), (1, 1))
        self.assertIsNone(result.comm_overlap)
        self.assertEqual({key: getattr(result.model, key) for key in adapter.OVERLAP_FLAGS}, adapter.OVERLAP_FLAGS)
        self.assertTrue(result.ddp.delay_wgrad_compute)
        self.assertFalse(result.profiling.use_nsys_profiler)
        self.assertFalse(result.profiling.use_pytorch_profiler)
        record = json.loads((self.run / "config/ep_overlap_contract.json").read_text())
        self.assertTrue(record["qualification_only"])
        self.assertIsNone(record["communication_sms"])
        self.assertEqual(record["cache_budget_steps"], 25)
        self.assertTrue(record["ddp_delay_wgrad_compute"])

    def test_uccl_official_budget_setter_precedes_builder_and_manager_keeps_budget(self):
        args = selected(self.run, case="uccl_ordinary", communication_backend="uccl_ep", communication_sms=32, overlap_capture="nsys")
        events, setup = [], {"dispatch": {"config": {"num_sms": 32}}}
        cfg, fused = config_fixture(), object()
        with patch.object(budget, "runtime_adapter", return_value=fused), \
                patch.object(budget, "configure_before_model", side_effect=lambda *_: events.append("setter") or setup), \
                patch.object(adapter, "BASE_BUILD_CONFIG", side_effect=lambda _: events.append("builder") or cfg):
            result = adapter.build_overlap_config(args)
        self.assertEqual(events, ["setter", "builder"])
        self.assertEqual(args.uccl_budget_setup, setup)
        self.assertEqual(result.model.moe_flex_dispatcher_num_sms, 32)
        self.assertTrue(result.profiling.nvtx_ranges)
        self.assertIs(result.model.delay_wgrad_compute, True)
        self.assertIs(result.ddp.delay_wgrad_compute, True)
        self.assertIsNone(result.comm_overlap)

    def test_ddp_companion_field_missing_or_remaining_false_fails_before_model_construction(self):
        class IgnoredDelaySetting:
            @property
            def delay_wgrad_compute(self):
                return False

            @delay_wgrad_compute.setter
            def delay_wgrad_compute(self, value):
                pass

        for kind in ("missing_ddp", "missing_field", "remaining_false"):
            cfg = config_fixture()
            if kind == "missing_ddp":
                delattr(cfg, "ddp")
            elif kind == "missing_field":
                delattr(cfg.ddp, "delay_wgrad_compute")
            else:
                cfg.ddp = IgnoredDelaySetting()
            with self.subTest(kind=kind), patch.object(adapter, "BASE_BUILD_CONFIG", return_value=cfg), \
                    self.assertRaisesRegex(ValueError, "DDP"):
                adapter.build_overlap_config(selected(self.run))
            self.assertFalse((self.run / "config/ep_overlap_contract.json").exists())

    def test_missing_native_field_and_changed_train_budget_fail_before_training(self):
        for missing, train_iters in [(True, 25), (False, 10)]:
            cfg = config_fixture()
            if missing:
                delattr(cfg.model, "ep_overlap_early_attn_memory_release")
            cfg.train.train_iters = train_iters
            with self.subTest(missing=missing), patch.object(adapter, "BASE_BUILD_CONFIG", return_value=cfg), self.assertRaises(ValueError):
                adapter.build_overlap_config(selected(self.run))

    def test_contract_records_actual_capture_windows_and_selected_baseline(self):
        for capture, steps, expected in [("off", 10, []), ("torch", 10, list(range(10))),
                                         ("nsys", 10, list(range(5, 10))), ("nsys", 2, [0, 1])]:
            args = selected(self.run, overlap_capture=capture, execution_steps=steps)
            record = adapter.execution_contract(args)
            self.assertEqual(record["capture_step_indices"], expected)
            self.assertEqual(record["kineto_enabled"], capture == "torch")
            self.assertEqual(record["exit_interval"], steps)
            self.assertTrue(record["ddp_delay_wgrad_compute"])
        with patch.object(adapter, "BASE_COMPARISON_CONTRACT", return_value={"baseline": {}}):
            record = adapter.comparison_contract(selected(self.run))
        self.assertTrue(record["baseline"]["ep_a2a_overlap"])
        self.assertTrue(record["baseline"]["delay_expert_wgrad"])
        self.assertTrue(record["baseline"]["ep_overlap_early_attn_memory_release"])
        self.assertFalse(record["baseline"]["dispatch_backward_expert_wgrad_overlap"])

    def test_frozen_source_manifest_rejects_tamper_duplicate_and_escape(self):
        source = self.root / "entry.py"
        source.write_text("pass\n")
        manifest = self.root / "source.sha256"
        row = f"{adapter.sha(source)}  {source}\n"
        manifest.write_text(row)
        result = adapter.verify_source_manifest(manifest, adapter.sha(manifest), root=self.root)
        self.assertEqual(result["files"], {"entry.py": adapter.sha(source)})
        for text in (row + row, f"{'a' * 64}  {self.root}/../escape.py\n"):
            manifest.write_text(text)
            with self.assertRaises(ValueError):
                adapter.verify_source_manifest(manifest, adapter.sha(manifest), root=self.root)
        manifest.write_text(row)
        source.write_text("changed\n")
        with self.assertRaisesRegex(ValueError, "source bytes differ"):
            adapter.verify_source_manifest(manifest, adapter.sha(manifest), root=self.root)

    def test_preparation_is_historical_and_binds_new_job_source_protocol(self):
        from tools import uccl_ep_verified_inputs_reuse as inputs
        from profiling.ampa_profile import provenance
        first_run, orchestration = self.root / "runs/first-training", self.root / "runs/current-owner"
        (first_run / "config").mkdir(parents=True)
        orchestration.mkdir()
        # Use a different new run because the normal fixture already has a manifest.
        run = self.root / "runs/new-training"
        c0 = {"hf_model": self.context["hf_model_id"], "hf_revision": "fixed", "hf_snapshot_manifest_sha256": "c" * 64}
        for name, record in [("dataset_manifest.json", self.dataset), ("c0_manifest.json", c0), ("model_manifest.json", {})]:
            (first_run / "config" / name).write_text(json.dumps(record))
        proof = {"checkpoint_payload": {"path": str(self.c0), "verified": True},
            "dataset_manifest_sha256": adapter.sha(first_run / "config/dataset_manifest.json"),
            "c0_identity_manifest_sha256": "b" * 64, "valid_reviewed_inputs": True}
        first = {"root": self.root / "runs/first-owner", "run": first_run,
                 "proof": proof, "acquisition": {"experiment_protocol": "old"},
                 "proof_reference": {"path": "/historical/proof", "sha256": "d" * 64, "slurm_job_id": "100"}}
        source = self.root / "new.py"
        source.write_text("pass\n")
        manifest = orchestration / "campaign-source.sha256"
        manifest.write_text(f"{adapter.sha(source)}  {source}\n")
        options = argparse.Namespace(run_dir=run, orchestration_dir=orchestration,
            case="uccl_ordinary", capture="nsys", communication_sms=12, execution_steps=2)
        with patch.object(inputs, "load_first_case", return_value=first), \
                patch.object(provenance, "_ampa_runtime_sources", return_value={"sha256": "f" * 64, "files": {}}), \
                patch.dict(os.environ, {"AMPA_OVERLAP_SOURCE_SHA256": adapter.sha(manifest)}):
            acquired = adapter.prepare_inputs(options, root=self.root)
        self.assertEqual(acquired["experiment_protocol"], adapter.PROTOCOL)
        self.assertEqual(acquired["job"], "123456")
        self.assertEqual(acquired["partition"], "debug")
        self.assertEqual(acquired["leg"], "nsys")
        self.assertTrue(acquired["qualification_only"])
        reused = json.loads((orchestration / "reviewed_inputs.json").read_text())
        self.assertTrue(reused["checkpoint_payload"]["verified"])
        self.assertFalse(reused["checkpoint_payload"]["verified_in_current_allocation"])
        self.assertEqual(reused["checkpoint_payload"]["verification_slurm_job_id"], "100")
        self.assertFalse(reused["verification_reuse"]["indexed_payload_verified_in_current_allocation"])
        context = json.loads((orchestration / "input_reuse_context.json").read_text())
        self.assertEqual(context["partition"], "debug")
        self.assertEqual(context["source_manifest"]["sha256"], adapter.sha(manifest))
        self.assertEqual(context["run_dir"], str(run))
        self.assertEqual(proof["checkpoint_payload"], {"path": str(self.c0), "verified": True})
        with patch.dict(os.environ, {"AMPA_UCCL_REUSE_CONTEXT": str(orchestration / "input_reuse_context.json"),
                                     "AMPA_OVERLAP_SOURCE_SHA256": adapter.sha(manifest)}):
            self.assertEqual(adapter.load_context(options)["partition"], "debug")
            context["partition"] = "preemptable"
            (orchestration / "input_reuse_context.json").write_text(json.dumps(context))
            with self.assertRaisesRegex(ValueError, "reuse context differs"):
                adapter.load_context(options)

    def test_main_owns_only_capture_callback_preserves_audits_and_restores_on_exception(self):
        for fail in (False, True):
            args = selected(self.run, overlap_capture="torch", torch_profile_all_steps=True)
            observer = SimpleNamespace(abort=Mock())
            audit, fingerprint, proof = object(), object(), object()
            old_pretrain = Mock(side_effect=RuntimeError("training failure") if fail else None)
            old_provenance = Mock(return_value=proof)
            old_deep = Mock()
            pretrain = SimpleNamespace(pretrain=old_pretrain)
            provenance = SimpleNamespace(make_indexed_run_provenance_callback=old_provenance)
            deep = SimpleNamespace(make_deep_trace_callback=old_deep)
            capture = SimpleNamespace(make_overlap_callback=Mock(return_value=observer))
            modules = {"megatron.bridge.training.pretrain": pretrain, "ampa_profile.provenance": provenance,
                       "ampa_profile.deep_trace": deep, "profiling.ampa_profile.overlap_capture": capture}
            originals = adapter.canonical.parse_args, adapter.canonical.build_config, adapter.canonical.comparison_contract

            def canonical_main():
                adapter.canonical.parse_args()
                item = deep.make_deep_trace_callback()
                provenance.make_indexed_run_provenance_callback(nsys_trace=True, torch_trace=False)
                pretrain.pretrain(config="config", forward_step_func="forward", callbacks=[audit, item, fingerprint, proof])

            with self.subTest(fail=fail), patch.object(adapter, "parse_overlap_args", return_value=args), \
                    patch.object(adapter.importlib, "import_module", side_effect=modules.__getitem__), \
                    patch.object(adapter.canonical, "main", side_effect=canonical_main):
                if fail:
                    with self.assertRaisesRegex(RuntimeError, "training failure"):
                        adapter.main(self.argv(capture="torch"))
                else:
                    adapter.main(self.argv(capture="torch"))
            old_deep.assert_not_called()
            observer.abort.assert_called_once_with()
            self.assertEqual(old_pretrain.call_args.kwargs["callbacks"], [observer, audit, fingerprint, proof])
            old_provenance.assert_called_once_with(nsys_trace=False, torch_trace=True)
            self.assertIs(pretrain.pretrain, old_pretrain)
            self.assertIs(provenance.make_indexed_run_provenance_callback, old_provenance)
            self.assertIs(deep.make_deep_trace_callback, old_deep)
            self.assertEqual((adapter.canonical.parse_args, adapter.canonical.build_config,
                              adapter.canonical.comparison_contract), originals)


if __name__ == "__main__":
    unittest.main()
