"""Exercise explicit verification reuse without changing measured training."""
import argparse
from contextlib import redirect_stderr
import copy
from io import StringIO
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from profiling import run_pretrain_debug_reuse as reuse
from tools import uccl_ep_verified_inputs_reuse as inputs
import test_uccl_pretrain_config as native_fixtures


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def metadata_fixture(directory):
    root = Path(directory)
    first, first_run = root / "runs/first", root / "runs/first-grouped-profile"
    first.mkdir(parents=True)
    c0 = root / "checkpoint/megatron/qwen3-30b-a3b-base-c0"
    c0.mkdir(parents=True)
    sources = {"files": {"profiling/run_pretrain.py": "a" * 64}, "sha256": "b" * 64}
    proof = {"valid_reviewed_inputs": True, "slurm_job_id": "100", "ampa_runtime_sources": sources,
        "checkpoint_payload": {"verified": True, "path": str(c0), "file_count": 17, "total_bytes": 61129914201},
        "c0_identity_manifest_sha256": "c" * 64, "dataset_manifest_sha256": "d" * 64,
        "native_checkouts": {"megatron_lm": {"revision": "unchanged"}, "megatron_bridge": {"revision": "unchanged"}}}
    acquired = {"job": "100", "experiment_protocol": "debug_one_case_5_plus_5_v1", "case": "grouped",
        "leg": "profile", "mode": "full", "partition": "debug", "warmup_steps": 5, "measured_steps": 5,
        "completed_updates": 10, "cache_budget_updates": 25, "cache_budget_samples": 1600,
        "topology": "cp1_ep16", "launch_policy": "torchrun_node4", "runtime_source_sha256": sources["sha256"],
        "binary_sha256": "e" * 64, "uccl_source_identity": {"base_revision": "firstverified"}}
    write(first / "reviewed_inputs.json", proof)
    write(first / "acquisition.json", acquired)
    write(first / "acquisition_completed.json", {"job": "100", "success": True,
        "workers_srun_returned_zero": True, "completed_updates": 10})
    (first / "cases.tsv").write_text(f"grouped\tprofile\t{first_run}\n")
    dataset = {"validated_training_data_path": "first immutable data", "model_and_tokenizer": {
        "model_id": "Qwen/Qwen3-30B-A3B-Base", "revision": "pinned", "tokenizer_id": "Qwen/Qwen3-30B-A3B-Base",
        "tokenizer_revision": "pinned"}, "validated_workload": {"micro_batch_size": 1,
        "dense_data_parallel_size": 16, "gradient_accumulation": 4, "global_batch_size": 64, "sequence_length": 4096,
        "warmup_steps": 5, "measured_steps": 20, "sampler_seed": 1234}, "megatron_cache": {"untouched": True}}
    write(first_run / "config/c0_manifest.json", {"hf_model": "Qwen/Qwen3-30B-A3B-Base",
        "hf_revision": "pinned", "hf_snapshot_manifest_sha256": "f" * 64})
    write(first_run / "config/dataset_manifest.json", dataset)
    write(first_run / "config/model_manifest.json", {"model": "unchanged"})
    current, run = root / "runs/next", root / "runs/next-ordinary-control"
    current.mkdir()
    (current / "campaign-source.sha256").write_text("recorded current campaign source identity\n")
    return root, first, current, run, sources


def debug_environment():
    return patch.dict(os.environ, {"SLURM_JOB_ID": "200", "SLURM_JOB_PARTITION": "debug",
                                  "SLURM_JOB_NUM_NODES": "4"})


class DebugReuseTests(unittest.TestCase):
    def prepare(self, directory):
        root, first, current, run, sources = metadata_fixture(directory)
        with debug_environment(), patch.object(inputs, "_ampa_runtime_sources", return_value=sources):
            result = inputs.prepare(current, run, "ordinary", "control", first_root=first, root=root)
        return root, first, current, run, result

    def test_prepare_records_historical_verification_and_current_job_without_payload_files(self):
        with TemporaryDirectory() as directory:
            root, first, current, run, result = self.prepare(directory)
            proof = result["reviewed_inputs"]
            self.assertTrue(proof["valid_reviewed_inputs"])
            self.assertTrue(proof["checkpoint_payload"]["verified"])
            self.assertFalse(proof["checkpoint_payload"]["verified_in_current_allocation"])
            self.assertEqual(proof["checkpoint_payload"]["verification_slurm_job_id"], "100")
            self.assertEqual(proof["slurm_job_id"], "200")
            self.assertEqual(proof["verification_mode"], "reused_first_case")
            self.assertEqual(proof["verification_reuse"]["first_verified_proof"]["path"], str(first / "reviewed_inputs.json"))
            self.assertEqual(inputs.read_json(run / "config/dataset_manifest.json"), result["context"]["indexed_manifest"])
            self.assertTrue(result["acquisition"]["acquisition_analysis_deferred"])
            self.assertEqual(result["acquisition"]["binary_sha256"], "e" * 64)
            self.assertFalse((root / "checkpoint/megatron/qwen3-30b-a3b-base-c0/iter_0000000").exists())
            self.assertTrue((current / "runtime-source.sha256").is_file())

    def test_incomplete_first_case_or_wrong_job_does_not_become_reusable(self):
        for field, value in (("success", False), ("workers_srun_returned_zero", False),
                             ("completed_updates", 9), ("job", "other")):
            with self.subTest(field=field), TemporaryDirectory() as directory:
                root, first, current, run, sources = metadata_fixture(directory)
                path = first / "acquisition_completed.json"
                record = inputs.read_json(path)
                record[field] = value
                write(path, record)
                with debug_environment(), patch.object(inputs, "_ampa_runtime_sources", return_value=sources), self.assertRaises(ValueError):
                    inputs.prepare(current, run, "ordinary", "control", first_root=first, root=root)
                self.assertFalse((current / "reviewed_inputs.json").exists())

    def test_canonical_parser_keeps_all_eight_measurement_modes_without_bulk_handlers(self):
        with TemporaryDirectory() as directory:
            root, first, current, run, prepared = self.prepare(directory)
            context = current / "input_reuse_context.json"
            base = ["--run-dir", str(run), "--pretrained-checkpoint", prepared["context"]["pretrained_checkpoint"],
                "--hf-revision", "pinned", "--dataset-data-path", "first immutable data", "--dataset-manifest",
                str(run / "config/dataset_manifest.json"), "--topology-variant", "cp1_ep16", "--lightweight-profiling", "off",
                "--sample-fingerprints", "--routing-audit", "--audit-received-experts", "--audit-num-groups", "4",
                "--launch-policy", "torchrun_node4", "--log-interval", "1", "--warmup-steps", "5", "--measured-steps", "5"]
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
                        with self.subTest(backend=backend, grouped=grouped, profiled=profiled), debug_environment(), \
                             patch.dict(os.environ, {"AMPA_UCCL_REUSE_CONTEXT": str(context)}), \
                             patch.object(reuse.debug.canonical, "_validate_c0_manifest", side_effect=AssertionError("bulk C0 verifier called")) as c0, \
                             patch.object(reuse.debug.canonical, "_validate_dataset_manifest", side_effect=AssertionError("bulk dataset verifier called")) as dataset, \
                             patch.object(reuse.debug.canonical, "_dataset_prefixes", return_value=[]):
                            args = reuse.parse_reuse_args(flags)
                            self.assertIs(reuse.debug.canonical._validate_c0_manifest, c0)
                            self.assertIs(reuse.debug.canonical._validate_dataset_manifest, dataset)
                        with patch.object(reuse.debug, "BASE_BUILD_CONFIG", side_effect=fixture.build), patch.dict(os.environ, {"RANK": "1"}):
                            config = reuse.debug.build_debug_config(args)
                        self.assertTrue(args.sample_fingerprints and args.routing_audit and args.audit_received_experts)
                        self.assertEqual(args.torch_profile_all_steps, profiled)
                        self.assertEqual((args.warmup_steps, args.measured_steps, args.completed_steps), (5, 5, 10))
                        self.assertEqual((config.train.train_iters, config.train.exit_interval), (25, 10))
                        self.assertEqual(args.indexed_manifest, prepared["context"]["indexed_manifest"])

    def test_reuse_context_requires_current_job_and_explicit_not_fresh_status(self):
        for changes in ({"slurm_job_id": "other"}, {"verification_mode": "fresh"},
                        {"verification_reuse": {"payload_verified_in_current_allocation": True}}):
            with self.subTest(changes=changes), TemporaryDirectory() as directory:
                _, _, current, _, prepared = self.prepare(directory)
                context = prepared["context"]
                context.update(changes)
                write(current / "input_reuse_context.json", context)
                with debug_environment(), patch.dict(os.environ, {"AMPA_UCCL_REUSE_CONTEXT": str(current / "input_reuse_context.json")}), self.assertRaises(ValueError):
                    reuse.load_context()

    def test_cheap_semantic_checks_still_reject_changed_paths_and_data_order(self):
        with TemporaryDirectory() as directory:
            _, _, _, run, prepared = self.prepare(directory)
            context = prepared["context"]
            parser = argparse.ArgumentParser()
            args = argparse.Namespace(run_dir=run, dataset_manifest=run / "config/dataset_manifest.json",
                pretrained_checkpoint=Path(context["pretrained_checkpoint"]), hf_model_id=context["hf_model_id"],
                hf_revision=context["hf_revision"], dataset_data_path="first immutable data", topology_variant="cp1_ep16",
                micro_batch_size=1, global_batch_size=64, sequence_length=4096, warmup_steps=5, measured_steps=20,
                dataset_sampler_seed=1234)
            reuse.cached_c0(args, parser, context)
            reuse.cached_dataset(args, parser, context)
            for field, value in (("dataset_sampler_seed", 99), ("measured_steps", 5),
                                 ("dataset_data_path", "different data"), ("run_dir", run.parent / "other")):
                changed = copy.copy(args)
                setattr(changed, field, value)
                with self.subTest(field=field), redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                    reuse.cached_dataset(changed, parser, context)

    def test_main_restores_only_parser_and_builder_after_training_exception(self):
        canonical = reuse.debug.canonical
        original = canonical.parse_args, canonical.build_config, canonical._validate_c0_manifest, canonical._validate_dataset_manifest
        with patch.object(canonical, "main", side_effect=RuntimeError("training failed")), self.assertRaisesRegex(RuntimeError, "training failed"):
            reuse.main()
        self.assertEqual((canonical.parse_args, canonical.build_config, canonical._validate_c0_manifest,
                          canonical._validate_dataset_manifest), original)


if __name__ == "__main__":
    unittest.main()
