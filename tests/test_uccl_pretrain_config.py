"""CPU checks for explicit UCCL selection and failure-safe entrypoint wiring."""
import copy
from contextlib import contextmanager, redirect_stderr
from io import StringIO
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from profiling import run_pretrain


class ProcessExit(BaseException):
    pass


def module(name, **attributes):
    result = ModuleType(name)
    vars(result).update(attributes)
    return result


class UcclPretrainConfigTests(unittest.TestCase):
    def parse(self, root, extra=(), grouped=True):
        argv = ["run_pretrain.py", "--run-dir", str(root), "--pretrained-checkpoint", str(root),
            "--hf-revision", "test", "--dataset-data-path", "test", "--dataset-manifest", str(root / "data.json"),
            "--topology-variant", "cp1_ep16", "--lightweight-profiling", "off", "--torch-profile-all-steps",
            "--torch-profile-ranks", ",".join(map(str, range(16))), "--sample-fingerprints",
            "--routing-audit", "--log-interval", "1"]
        if grouped:
            argv += ["--router-num-groups", "4", "--router-group-topk", "2"]
        argv += list(extra)
        def c0(args, parser):
            args.c0_identity_manifest_sha256 = "c0-test"
        def dataset(args, parser):
            args.indexed_manifest = {}
        with patch.object(sys, "argv", argv), patch.object(run_pretrain, "_validate_c0_manifest", side_effect=c0), \
             patch.object(run_pretrain, "_validate_dataset_manifest", side_effect=dataset), \
             patch.object(run_pretrain, "_dataset_prefixes", return_value=[]):
            return run_pretrain.parse_args()

    def uccl_flags(self, root):
        return ["--communication-backend", "uccl_ep", "--launch-policy", "torchrun_node4",
                "--audit-received-experts", "--uccl-build-dir", str(root)]

    def test_native_defaults_and_fresh_node_visible_controls_remain_explicit(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for grouped in (False, True):
                with self.subTest(grouped=grouped):
                    args = self.parse(root, grouped=grouped)
                    self.assertEqual((args.communication_backend, args.launch_policy), ("alltoall", "singleton"))
                    self.assertFalse(args.audit_received_experts)
                    self.assertIsNone(args.uccl_build_dir)
                    args = self.parse(root, ["--launch-policy", "torchrun_node4", "--audit-received-experts"], grouped)
                    self.assertEqual(args.launch_policy, "torchrun_node4")
                    self.assertTrue(args.audit_received_experts)

    def test_uccl_accepts_both_routes_with_cp1_ep16_audit4_and_build_evidence(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            valid = self.uccl_flags(root)
            for grouped in (False, True):
                with self.subTest(grouped=grouped):
                    args = self.parse(root, valid, grouped=grouped)
                    self.assertEqual(args.completed_steps, 25)
                    expected = (4, 2, 4) if grouped else (None, None, 4)
                    self.assertEqual((args.router_num_groups, args.router_group_topk, args.audit_num_groups), expected)
            cases = [valid + ["--topology-variant", "cp2_ep16"],
                valid + ["--launch-policy", "singleton"], valid + ["--router-group-topk", "1"],
                valid + ["--router-num-groups", "8", "--audit-num-groups", "8"],
                [value for value in valid if value != "--audit-received-experts"],
                valid[:-2], valid[:-1] + [str(root / "missing")],
                ["--uccl-build-dir", str(root)], ["--communication-backend", "unknown"]]
            for extra in cases:
                with self.subTest(extra=extra), redirect_stderr(StringIO()), self.assertRaises(SystemExit) as raised:
                    self.parse(root, extra)
                self.assertEqual(raised.exception.code, 2)
            ordinary_faults = [valid + ["--audit-num-groups", "8"],
                valid + ["--router-num-groups", "4"], valid + ["--router-group-topk", "2"]]
            for extra in ordinary_faults:
                with self.subTest(extra=extra), redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                    self.parse(root, extra, grouped=False)

    def build(self, args):
        sections = ("tokenizer", "train", "validation", "dataset", "optimizer", "scheduler",
                    "checkpoint", "logger", "profiling")
        config = SimpleNamespace(**{name: SimpleNamespace() for name in sections})
        model = SimpleNamespace(moe_router_dtype=None, moe_permute_fusion=True, moe_grouped_gemm=True,
                                moe_router_num_groups=8, moe_router_group_topk=1)
        bridge = SimpleNamespace(to_megatron_provider=Mock(return_value=model))
        auto_bridge = SimpleNamespace(from_hf_config=Mock(return_value=bridge))
        auto_config = SimpleNamespace(from_pretrained=Mock(return_value=SimpleNamespace()))
        precision = object()
        modules = {
            "transformers": module("transformers", AutoConfig=auto_config),
            "megatron.bridge": module("megatron.bridge", AutoBridge=auto_bridge),
            "megatron.bridge.recipes.common": module("megatron.bridge.recipes.common", _pretrain_common=lambda: config),
            "megatron.bridge.training.mixed_precision": module("megatron.bridge.training.mixed_precision", bf16_mixed=lambda: precision),
        }
        with patch.dict(sys.modules, modules), patch.object(run_pretrain, "apply_bridge_indexed_consumer"):
            result = run_pretrain.build_config(args)
        bridge.to_megatron_provider.assert_called_once_with(load_weights=False)
        self.assertIs(result.mixed_precision, precision)
        return result

    def test_backend_selection_changes_only_dispatcher_configuration_and_preserves_precision(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for grouped in (False, True):
                with self.subTest(grouped=grouped):
                    native = self.build(self.parse(root, grouped=grouped))
                    uccl = self.build(self.parse(root, self.uccl_flags(root), grouped=grouped))
                    self.assertEqual(native.model.moe_token_dispatcher_type, "alltoall")
                    self.assertIsNone(native.model.moe_flex_dispatcher_backend)
                    self.assertEqual(uccl.model.moe_token_dispatcher_type, "flex")
                    self.assertEqual(uccl.model.moe_flex_dispatcher_backend, "deepep")
                    native_model, uccl_model = dict(vars(native.model)), dict(vars(uccl.model))
                    for key in ("moe_token_dispatcher_type", "moe_flex_dispatcher_backend"):
                        native_model.pop(key)
                        uccl_model.pop(key)
                    self.assertEqual(native_model, uccl_model)
                    expected = (4, 2) if grouped else (None, None)
                    self.assertEqual((uccl.model.moe_router_num_groups, uccl.model.moe_router_group_topk), expected)
                    self.assertIsNone(uccl.model.moe_router_dtype)
                    self.assertFalse(uccl.model.moe_enable_deepep)
                    self.assertTrue(uccl.model.moe_permute_fusion)
                    self.assertTrue(uccl.model.moe_grouped_gemm)
                    self.assertEqual((uccl.model.context_parallel_size, uccl.model.expert_model_parallel_size), (1, 16))

    def test_default_contract_schema_is_preserved_and_opt_in_records_backend_and_launch(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data.json").write_text("{}")
            args = self.parse(root)
            old_args = copy.deepcopy(args)
            for key in ("communication_backend", "launch_policy", "audit_received_experts", "uccl_build_dir"):
                delattr(old_args, key)
            native = run_pretrain.comparison_contract(args)
            self.assertEqual(native, run_pretrain.comparison_contract(old_args))
            self.assertNotIn("communication_backend", native)
            uccl = run_pretrain.comparison_contract(self.parse(root, self.uccl_flags(root)))
            self.assertEqual(uccl["communication_backend"], "uccl_ep")
            self.assertEqual(uccl["launch_policy"], "torchrun_node4")
            self.assertTrue(uccl["audit_received_experts"])
            self.assertEqual(uccl["uccl_build_dir"], str(root.resolve()))
            self.assertEqual(uccl["baseline"]["dispatcher"], "flex")
            for key in ("model", "parallelism", "training", "dataset", "routing"):
                self.assertEqual(native[key], uccl[key])

    @contextmanager
    def trainer(self, root, backend, failure=None):
        flags = self.uccl_flags(root) if backend == "uccl_ep" else ["--launch-policy", "torchrun_node4", "--audit-received-experts"]
        args = self.parse(root, flags)
        factories, modules = {}, {}
        for name, function in (("deep_trace", "make_deep_trace_callback"), ("routing_audit", "make_routing_audit_callback"),
                               ("sample_fingerprint", "make_sample_fingerprint_callback"),
                               ("uccl_identity", "make_uccl_identity_callback"), ("provenance", "make_indexed_run_provenance_callback")):
            factories[name] = Mock(name=function, return_value=object())
            modules["ampa_profile." + name] = module("ampa_profile." + name, **{function: factories[name]})
        pretrain = Mock(side_effect=failure)
        modules.update({"megatron.bridge.training.gpt_step": module("megatron.bridge.training.gpt_step", forward_step=object()),
            "megatron.bridge.training.pretrain": module("megatron.bridge.training.pretrain", pretrain=pretrain),
            "megatron.bridge.training.config": module("megatron.bridge.training.config", apply_environment_variables=Mock())})
        with patch.dict(sys.modules, modules), patch.dict("os.environ", {"RANK": "1"}), \
             patch.object(run_pretrain, "parse_args", return_value=args), \
             patch.object(run_pretrain, "build_config", return_value=SimpleNamespace()):
            yield args, factories, pretrain

    def test_callbacks_receive_backend_launch_and_receive_policy_with_identity_before_provenance(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for backend in ("alltoall", "uccl_ep"):
                with self.subTest(backend=backend), self.trainer(root, backend) as (args, factories, pretrain):
                    run_pretrain.main()
                    self.assertEqual(factories["deep_trace"].call_args.kwargs["communication_backend"], backend)
                    self.assertEqual(factories["routing_audit"].call_args.kwargs["communication_backend"], backend)
                    self.assertIs(factories["routing_audit"].call_args.kwargs["record_receives"], True)
                    self.assertEqual(factories["provenance"].call_args.kwargs["launch_policy"], "torchrun_node4")
                    callbacks = pretrain.call_args.kwargs["callbacks"]
                    self.assertIs(callbacks[-1], factories["provenance"].return_value)
                    if backend == "uccl_ep":
                        factories["uccl_identity"].assert_called_once_with(run_dir=root, build_dir=args.uccl_build_dir)
                        self.assertIs(callbacks[-2], factories["uccl_identity"].return_value)
                    else:
                        factories["uccl_identity"].assert_not_called()

    def test_uccl_base_exception_records_host_failure_and_exits_while_native_propagates(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "raw").mkdir()
            failure = KeyboardInterrupt("native body failed")
            with self.trainer(root, "alltoall", failure), patch.object(run_pretrain, "_abort_uccl_pretrain") as abort:
                with self.assertRaises(KeyboardInterrupt) as raised:
                    run_pretrain.main()
                self.assertIs(raised.exception, failure)
                abort.assert_not_called()
            failure = KeyboardInterrupt("uccl body failed")
            with self.trainer(root, "uccl_ep", failure), redirect_stderr(StringIO()), \
                 patch.object(run_pretrain.os, "_exit", side_effect=ProcessExit) as exit_process:
                with self.assertRaises(ProcessExit):
                    run_pretrain.main()
                exit_process.assert_called_once_with(1)
            record = json.loads((root / "raw/uccl_pretrain_aborted_rank-1.json").read_text())
            self.assertFalse(record["valid"])
            self.assertEqual(record["error"], "KeyboardInterrupt: uccl body failed")
            self.assertIn("uccl body failed", record["traceback"])
            self.assertFalse(record["entrypoint_cuda_cleanup_attempted"])
            self.assertTrue(record["immediate_process_exit_requested"])

    def test_completed_zero_exit_classification_requires_exact_code_and_cleanup_flag(self):
        self.assertTrue(run_pretrain._completed_uccl_zero_exit(SystemExit(0), SimpleNamespace(cleanup_completed=True)))
        for error in (SystemExit(1), SystemExit(-1), SystemExit(None), SystemExit(False), SystemExit(True),
                      SystemExit("0"), RuntimeError("failure"), KeyboardInterrupt()):
            with self.subTest(error=repr(error)):
                self.assertFalse(run_pretrain._completed_uccl_zero_exit(error, SimpleNamespace(cleanup_completed=True)))
        for callback in (None, object(), SimpleNamespace(cleanup_completed=False), SimpleNamespace(cleanup_completed=0),
                         SimpleNamespace(cleanup_completed=1), SimpleNamespace(cleanup_completed=None)):
            with self.subTest(callback=repr(callback)):
                self.assertFalse(run_pretrain._completed_uccl_zero_exit(SystemExit(0), callback))

    def test_main_reraises_only_completed_uccl_integer_zero_and_keeps_other_failures_fast(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for backend in ("alltoall", "uccl_ep"):
                failure = SystemExit(0)
                with self.subTest(backend=backend), self.trainer(root, backend, failure) as (_, factories, _), \
                     patch.object(run_pretrain, "_abort_uccl_pretrain") as abort:
                    factories["uccl_identity"].return_value = SimpleNamespace(cleanup_completed=True)
                    with self.assertRaises(SystemExit) as raised:
                        run_pretrain.main()
                    self.assertIs(raised.exception, failure)
                    abort.assert_not_called()
            for failure, completed in ((SystemExit(0), False), (SystemExit(1), True), (SystemExit(None), True),
                                       (SystemExit(False), True), (SystemExit(True), True), (KeyboardInterrupt(), True),
                                       (RuntimeError("later failure"), True)):
                with self.subTest(error=repr(failure), completed=completed), self.trainer(root, "uccl_ep", failure) as (args, factories, _), \
                     patch.object(run_pretrain, "_abort_uccl_pretrain", side_effect=ProcessExit) as abort:
                    factories["uccl_identity"].return_value = SimpleNamespace(cleanup_completed=completed)
                    with self.assertRaises(ProcessExit):
                        run_pretrain.main()
                    abort.assert_called_once_with(args, 1, failure)

    def test_abort_evidence_write_failure_still_exits(self):
        with TemporaryDirectory() as directory:
            args = SimpleNamespace(run_dir=Path(directory), launch_policy="torchrun_node4")
            # Missing raw directory must not let stack unwinding reach native
            # transport destructors after an evidence-write failure.
            with redirect_stderr(StringIO()), patch.object(run_pretrain.os, "_exit", side_effect=ProcessExit) as exit_process:
                with self.assertRaises(ProcessExit):
                    run_pretrain._abort_uccl_pretrain(args, 1, RuntimeError("transport failure"))
                exit_process.assert_called_once_with(1)


if __name__ == "__main__":
    unittest.main()
