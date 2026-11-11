"""Unprofiled budget selection and state guards. Run on allocated compute."""
import argparse
from contextlib import redirect_stderr
from io import StringIO
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from profiling import run_pretrain_ep_overlap as original
from profiling import run_pretrain_ep_overlap_budget as adapter
from profiling import run_pretrain_uccl_budget as native_budget


ALLOCATION = {"SLURM_JOB_PARTITION": "debug", "SLURM_JOB_NUM_NODES": "4", "SLURM_JOB_ID": "123456",
              "CUDA_DEVICE_MAX_CONNECTIONS": "32", "FI_MR_CACHE_MONITOR": "disabled", "LOCAL_RANK": "0", "RANK": "0"}


def execution(directory, budget=12, steps=10):
    warmup, measured = (1, 1) if steps == 2 else (5, 5)
    return argparse.Namespace(case="uccl_ordinary", communication_backend="uccl_ep", overlap_capture="off",
        communication_sms=budget, execution_steps=steps, completed_steps=steps, smoke_steps=steps,
        warmup_steps=warmup, measured_steps=measured, overlap_source_sha256="a" * 64,
        run_dir=Path(directory), uccl_budget_setup=None)


def config(steps=10):
    return SimpleNamespace(model=SimpleNamespace(hidden_size=2048, moe_router_topk=8,
        moe_router_num_groups=None, moe_router_group_topk=None,
        **{key: False for key in adapter.OVERLAP_FLAGS}),
        ddp=SimpleNamespace(delay_wgrad_compute=False), train=SimpleNamespace(train_iters=25, exit_interval=steps),
        profiling=SimpleNamespace(), comm_overlap=object())


class BudgetEntrypointTests(unittest.TestCase):
    def test_budget_bounds_follow_native_launch_constraints(self):
        self.assertTrue(all(adapter.allowed_budget(value) for value in (4, 12, 20, 32, 40, 48, 54, 56, 64)))
        self.assertFalse(any(adapter.allowed_budget(value) for value in (0, 2, 3, 55, 65, 66, 132, True, 20.0)))
        for value in (0, 2, 3, True, 20.0):
            with self.assertRaises(ValueError):
                adapter.kernel_launch_requirements(value)
        self.assertEqual(adapter.kernel_launch_requirements(4)["notify_dynamic_shared_bytes"], 32768)
        self.assertEqual(adapter.kernel_launch_requirements(40)["notify_threads"], 640)
        self.assertEqual(adapter.kernel_launch_requirements(54)["notify_dynamic_shared_bytes"], 221184)
        self.assertEqual(adapter.kernel_launch_requirements(56)["notify_dynamic_shared_bytes"], 229376)
        self.assertEqual(adapter.kernel_launch_requirements(58)["notify_dynamic_shared_bytes"], 237568)
        self.assertEqual(adapter.kernel_launch_requirements(64)["notify_threads"], 1024)
        self.assertEqual(adapter.kernel_launch_requirements(66)["notify_threads"], 1056)

    def test_full_and_qualification_contracts_keep_actual_budget_and_off(self):
        for steps in (2, 10):
            for budget in (12, 20, 48, 56):
                row = adapter.execution_contract(execution("/isolated/run", budget, steps))
                self.assertEqual(row["experiment_protocol"], adapter.PROTOCOL)
                self.assertEqual(row["communication_sms"], budget)
                self.assertEqual(row["qualification_only"], steps == 2)
                self.assertEqual(row["capture_step_indices"], [])
                self.assertEqual(row["capture_ranks"], [])
                self.assertFalse(row["kineto_enabled"])
                self.assertEqual(row["cache_budget_steps"], 25)
                self.assertEqual((row["warmup_steps"], row["measured_steps"]), (1, 1) if steps == 2 else (5, 5))

    def test_nonordinary_or_profiled_execution_is_rejected(self):
        for field, value in (("case", "uccl"), ("communication_backend", "alltoall"), ("overlap_capture", "torch"),
                             ("overlap_capture", "nsys"), ("communication_sms", 66)):
            args = execution("/isolated/run")
            setattr(args, field, value)
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                adapter.execution_contract(args)

    def test_parser_rejects_incompatible_window_without_changing_cache(self):
        parser = adapter.selected_parser()
        common = ["--run-dir", "/isolated/run", "--case", "uccl_ordinary", "--capture", "off", "--communication-sms", "12"]
        for steps, warmup, measured in ((10, 1, 1), (2, 5, 5)):
            selected = parser.parse_args([*common, "--execution-steps", str(steps), "--warmup-steps", str(warmup), "--measured-steps", str(measured)])
            with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                adapter.validate_selection(selected, parser)
        selected = parser.parse_args([*common, "--execution-steps", "2", "--warmup-steps", "1", "--measured-steps", "1"])
        adapter.validate_selection(selected, parser)

    def test_scoped_protocol_restores_original_adapter_after_exception(self):
        from profiling.ampa_profile import overlap_capture
        names = ("PROTOCOL", "BUDGETS", "selected_parser", "validate_selection", "execution_contract", "build_overlap_config", "comparison_contract")
        before = {name: getattr(original, name) for name in names}
        before_capture = overlap_capture.make_overlap_callback
        with self.assertRaisesRegex(RuntimeError, "intentional"):
            with adapter.compat_context():
                self.assertEqual(original.PROTOCOL, adapter.PROTOCOL)
                self.assertIs(original.execution_contract, adapter.execution_contract)
                with adapter.compat_context():
                    self.assertEqual(original.BUDGETS, adapter.BUDGETS)
                self.assertIs(overlap_capture.make_overlap_callback, adapter.make_budget_callback)
                raise RuntimeError("intentional")
        self.assertEqual({name: getattr(original, name) for name in names}, before)
        self.assertIs(overlap_capture.make_overlap_callback, before_capture)

    def test_actual_device_check_precedes_native_setter_and_builder(self):
        with TemporaryDirectory() as directory, patch.dict(os.environ, ALLOCATION, clear=True):
            args = execution(directory, 48)
            events = []
            with adapter.compat_context(), \
                    patch.object(adapter, "device_budget_limits", side_effect=lambda **_: events.append("device") or {"supported_budgets": [20, 48]}), \
                    patch.object(native_budget, "runtime_adapter", return_value=object()), \
                    patch.object(native_budget, "configure_before_model", side_effect=lambda _, budget: events.append(("setter", budget)) or {"budget": budget}), \
                    patch.object(original, "BASE_BUILD_CONFIG", side_effect=lambda _: events.append("builder") or config()):
                result = adapter.build_budget_config(args)
            self.assertEqual(events, ["device", ("setter", 48), "builder"])
            self.assertEqual(result.model.moe_flex_dispatcher_num_sms, 48)
            self.assertTrue(result.model.overlap_moe_expert_parallel_comm)
            self.assertTrue(result.model.delay_wgrad_compute)
            self.assertTrue(result.model.ep_overlap_early_attn_memory_release)
            self.assertTrue(result.ddp.delay_wgrad_compute)
            self.assertFalse(result.profiling.use_pytorch_profiler)
            self.assertFalse(result.profiling.use_nsys_profiler)
            self.assertFalse(result.profiling.nvtx_ranges)

    def test_device_limit_rejects_before_native_configuration(self):
        with patch.dict(os.environ, ALLOCATION, clear=True), \
                patch.object(adapter, "device_budget_limits", return_value={"supported_budgets": [12, 20, 32, 40]}), \
                patch.object(adapter, "_BASE_BUILD") as builder:
            with self.assertRaises(ValueError):
                adapter.build_budget_config(execution("/isolated/run", 54))
            builder.assert_not_called()

    def test_runtime_guard_checks_exercised_buffer_budget(self):
        args = execution("/isolated/run", 48)
        buffer = SimpleNamespace(num_sms=48, low_latency_mode=False, group_size=16, num_nvl_bytes=100, num_rdma_bytes=200)
        fused = SimpleNamespace(_buffer=buffer, Buffer=SimpleNamespace(num_sms=48))
        device = {"supported_budgets": [20, 48]}
        record = adapter.runtime_budget_record(fused, args, device, rank=7, job="123456")
        self.assertEqual((record["requested_budget"], record["buffer_num_sms"], record["rank"]), (48, 48, 7))
        for value in (20, 54):
            buffer.num_sms = value
            with self.assertRaises(ValueError):
                adapter.runtime_budget_record(fused, args, device, rank=7, job="123456")

    def test_explicit_local_device_geometry_constrains_supported_range(self):
        enum_names = ("MAX_SHARED_MEMORY_PER_BLOCK_OPTIN", "MAX_THREADS_PER_BLOCK", "WARP_SIZE", "COMPUTE_CAPABILITY_MAJOR", "MULTIPROCESSOR_COUNT")
        enums = SimpleNamespace(**{"CU_DEVICE_ATTRIBUTE_" + name: name for name in enum_names})
        values = dict(zip(enum_names, (232448, 1024, 32, 9, 132)))
        selected = []
        driver = SimpleNamespace(CUresult=SimpleNamespace(CUDA_SUCCESS=0), CUdevice_attribute=enums,
            cuInit=lambda _: (0,), cuDeviceGet=lambda ordinal: selected.append(ordinal) or (0, ordinal),
            cuDeviceGetAttribute=lambda attr, device: (0, values[attr]))
        module = ModuleType("cuda.bindings")
        module.driver = driver
        with patch.dict(sys.modules, {"cuda.bindings": module}):
            caps = adapter.device_budget_limits(ordinal=3)
            self.assertEqual(selected, [3])
            self.assertEqual(caps["supported_budgets"][-1], 56)
            self.assertNotIn(58, caps["supported_budgets"])
            values["MAX_SHARED_MEMORY_PER_BLOCK_OPTIN"] = 227328
            self.assertEqual(adapter.device_budget_limits(ordinal=3)["supported_budgets"][-1], 54)
            values["MAX_SHARED_MEMORY_PER_BLOCK_OPTIN"] = 220000
            self.assertEqual(adapter.device_budget_limits(ordinal=3)["supported_budgets"][-1], 52)


if __name__ == "__main__":
    unittest.main()
