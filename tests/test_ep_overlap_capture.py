"""Check actual capture callbacks with mocked CUDA and distributed services.

Run on allocated compute. These checks never simulate GPU timing evidence.
"""

from collections import Counter
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from profiling.ampa_profile.deep_trace import DeepTraceError
from profiling.ampa_profile import overlap_capture as capture


class _Profiler:
    def __init__(self, events):
        self.events = events
        self.instances = []

    @contextmanager
    def record_function(self, label):
        self.events.append(("scope_enter", label))
        try:
            yield
        finally:
            self.events.append(("scope_exit", label))

    def profile(self, **kwargs):
        self.events.append(("profile", kwargs))
        owner = self

        class Profile:
            def start(self):
                owner.events.append(("profile_start",))

            def stop(self):
                owner.events.append(("profile_stop",))

            def export_chrome_trace(self, destination):
                owner.events.append(("profile_export", destination))
                Path(destination).write_bytes(b"mock trace export")

        instance = Profile()
        self.instances.append(instance)
        return instance


class _Cuda:
    def __init__(self, events):
        self.events = events
        self.nvtx = SimpleNamespace(range_push=lambda label: events.append(("nvtx_push", label)),
                                    range_pop=lambda: events.append(("nvtx_pop",)))

    def synchronize(self):
        self.events.append(("synchronize",))

    def cudart(self):
        return self

    def cudaProfilerStart(self):
        self.events.append(("cuda_profiler_start",))
        return 0

    def cudaProfilerStop(self):
        self.events.append(("cuda_profiler_stop",))
        return 0

    def check_error(self, status):
        if status != 0:
            raise RuntimeError("mock CUDA failure")


class _Dist:
    def __init__(self, events):
        self.events = events
        self.world_size = 16
        self.disagree = False

    def get_rank(self):
        return 0

    def get_world_size(self):
        return self.world_size

    def all_gather_object(self, peers, local):
        self.events.append(("all_gather_object",))
        peers[:] = [local] * 16
        if self.disagree:
            peers[-1] = dict(local, capture_mode="mismatched")

    def barrier(self):
        self.events.append(("barrier",))


class _Labels:
    def __init__(self, events, proxy, model, optimizer, communication_backend):
        self.events, self.proxy = events, proxy
        self.model, self.optimizer, self.backend = model, optimizer, communication_backend
        self.layers = list(range(48))
        self.counts = Counter({"AMPA/OVERLAP/TRAIN/combined_pass_00": 1})
        self.payload_bytes = Counter({"payload": 123})
        self.payload_dtypes = {"payload": "torch.bfloat16"}
        self.validation_error = self.attach_error = None

    def attach(self):
        self.events.append(("attach",))
        if self.attach_error:
            raise self.attach_error

    def detach(self):
        self.events.append(("detach",))

    def validate_counts(self, accumulation):
        self.events.append(("validate", accumulation))
        if self.validation_error:
            raise self.validation_error

    def pair_records(self):
        return [{"call_id": 7, "forward_returned": True, "backward_returned": True}]

    def contract_record(self, accumulation):
        return {"contract": capture.CONTRACT, "gradient_accumulation": accumulation,
                "expected_counts": dict(self.counts)}

    def combined_pass_records(self):
        return [{"pass": 0, "forward_microbatch": 0, "backward_microbatch": None, "returned": True}]


class EpOverlapCaptureTest(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.events, self.labels = [], []
        self.profiler = _Profiler(self.events)
        self.torch = SimpleNamespace(__version__="mock", profiler=self.profiler, cuda=_Cuda(self.events))
        self.dist = _Dist(self.events)
        self.context = SimpleNamespace(model=[SimpleNamespace(ddp_config=SimpleNamespace(delay_wgrad_compute=True))],
            optimizer=object(), skipped_iter=False,
            state=SimpleNamespace(train_state=SimpleNamespace(step=0)))
        self.config = SimpleNamespace(**capture.SELECTED_FLAGS,
            pipeline_model_parallel_size=1, tensor_model_parallel_size=1, context_parallel_size=1,
            expert_model_parallel_size=16, expert_tensor_parallel_size=1, num_layers=48,
            cuda_graph_impl="none", recompute_granularity=None, moe_router_topk=8,
            moe_flex_dispatcher_num_sms=20)
        names = ("megatron", "megatron.core", "megatron.core.utils")
        modules = {name: ModuleType(name) for name in names}
        for name, module in modules.items():
            parent, _, child = name.rpartition(".")
            if parent in modules:
                setattr(modules[parent], child, module)
        modules["megatron.core.utils"].get_model_config = lambda _: self.config
        for patcher in (patch.dict(sys.modules, modules),
                        patch.dict(os.environ, {"CUDA_DEVICE_MAX_CONNECTIONS": "32"}),
                        patch.object(capture.importlib.metadata, "version", return_value="mock")):
            patcher.start()
            self.addCleanup(patcher.stop)

    def factory(self, proxy, model, optimizer, communication_backend):
        labeler = _Labels(self.events, proxy, model, optimizer, communication_backend)
        self.labels.append(labeler)
        return labeler

    def test_startup_requires_actual_ddp_delay_before_any_update(self):
        for value in (False, None):
            with self.subTest(ddp_delay=value):
                self.context.model[0].ddp_config.delay_wgrad_compute = value
                callback = self.make("off")
                with self.assertRaises(DeepTraceError):
                    callback.on_train_start(self.context)
                self.assertNotIn(("all_gather_object",), self.events)

    def make(self, mode="torch", steps=2, backend="alltoall"):
        args = SimpleNamespace(run_dir=self.root / f"{mode}-{steps}", overlap_capture=mode,
                               completed_steps=steps, communication_backend=backend,
                               communication_sms=20 if backend == "uccl_ep" else None)
        callback = capture._OverlapCapture(args, self.torch, self.dist, self.factory)
        self.addCleanup(callback.abort)
        return callback

    def drive(self, callback):
        callback.on_train_start(self.context)
        for step in range(callback.steps):
            self.context.state.train_state.step = step
            callback.on_train_step_start(self.context)
            callback.on_train_step_end(self.context)
        callback.on_train_end(self.context)

    def names(self):
        return [row[0] for row in self.events]

    def test_mode_step_grids(self):
        for steps in (2, 10):
            for mode in ("off", "torch", "nsys"):
                callback = self.make(mode, steps)
                expected = ([] if mode == "off" else list(range(steps)) if mode == "torch"
                            else [0, 1] if steps == 2 else list(range(5, 10)))
                self.assertEqual(callback.capture_steps, expected)
                callback.finished = True
        for mode, steps in (("invalid", 2), ("torch", 3)):
            with self.assertRaises(DeepTraceError):
                self.make(mode, steps)

    def test_off_never_attaches_wrappers_or_calls_profilers(self):
        callback = self.make("off", 10)
        self.drive(callback)
        self.assertFalse(self.labels)
        self.assertEqual(self.names(), ["all_gather_object"])
        record = json.loads((callback.run / "raw/ep_overlap_rank-0_capture.json").read_text())
        self.assertEqual(record["capture_steps"], [])
        self.assertEqual(record["ended_steps"], list(range(10)))
        self.assertTrue(record["valid_step_execution"])
        startup = json.loads((callback.run / "raw/ep_overlap_rank-0_startup.json").read_text())
        self.assertIs(startup["ddp_delay_wgrad_compute"], True)

    def test_torch_lifecycle_preserves_returned_counts_and_contract(self):
        callback = self.make()
        self.drive(callback)
        self.assertEqual(self.names().count("profile_start"), 2)
        self.assertEqual(self.names().count("profile_stop"), 2)
        self.assertEqual(self.names().count("profile_export"), 2)
        self.assertEqual(self.names().count("attach"), 2)
        self.assertEqual(self.names().count("detach"), 2)
        self.assertNotIn("synchronize", self.names())
        self.assertNotIn("barrier", self.names())
        self.assertNotIn("cuda_profiler_start", self.names())
        for step, labeler in enumerate(self.labels):
            record = json.loads((callback.run / "raw/overlap_calls" / f"step-{step:05d}/rank-0.json").read_text())
            self.assertEqual(record["counts"], labeler.counts)
            self.assertEqual(record["payload_bytes"], labeler.payload_bytes)
            self.assertEqual(record["payload_dtypes"], labeler.payload_dtypes)
            self.assertEqual(record["uccl_autograd_pairs"], labeler.pair_records())
            self.assertEqual(record["combined_pass_records"], labeler.combined_pass_records())
            self.assertEqual(record["observation_contract"], labeler.contract_record(4))
            self.assertEqual(record["capture_mode"], "torch")
            self.assertFalse(record["skipped_iter"])
            self.assertTrue((callback.run / "traces/torch_profile" / f"step-{step:05d}/rank-0.json.gz").is_file())

    def test_nsight_warmup_and_capture_boundaries(self):
        callback = self.make("nsys", 10)
        callback.on_train_start(self.context)
        for step in range(10):
            self.context.state.train_state.step = step
            callback.on_train_step_start(self.context)
            if step < 5:
                self.assertFalse(self.labels)
                self.assertNotIn("synchronize", self.names())
            callback.on_train_step_end(self.context)
        callback.on_train_end(self.context)
        self.assertEqual(len(self.labels), 5)
        self.assertEqual(self.names().count("synchronize"), 2)
        self.assertEqual(self.names().count("barrier"), 3)
        self.assertEqual(self.names().count("cuda_profiler_start"), 1)
        self.assertEqual(self.names().count("cuda_profiler_stop"), 1)
        self.assertNotIn("profile_start", self.names())
        self.assertEqual(callback.captured, list(range(5, 10)))
        self.assertEqual([row["action"] for row in callback.boundaries], ["start", "stop"])

    def test_nsight_two_step_qualification(self):
        callback = self.make("nsys", 2)
        self.drive(callback)
        record = json.loads((callback.run / "raw/ep_overlap_rank-0_capture.json").read_text())
        self.assertTrue(record["qualification_only"])
        self.assertEqual(record["capture_steps"], [0, 1])

    def test_abort_torch_restores_host_hooks_without_cuda_wait_stop_or_export(self):
        callback = self.make()
        callback.on_train_start(self.context)
        callback.on_train_step_start(self.context)
        callback.abort()
        callback.abort()
        self.assertEqual(self.names().count("detach"), 1)
        self.assertEqual(self.names().count("scope_exit"), 1)
        for forbidden in ("synchronize", "barrier", "cuda_profiler_stop", "profile_stop", "profile_export"):
            self.assertNotIn(forbidden, self.names())
        self.assertIsNotNone(callback.profiler)
        self.assertFalse(callback.finished)
        record = json.loads((callback.run / "raw/ep_overlap_rank-0_aborted.json").read_text())
        self.assertFalse(record["cuda_cleanup_attempted"])
        self.assertFalse(record["trace_exported"])

    def test_abort_nsight_does_not_wait_or_stop_active_capture(self):
        callback = self.make("nsys", 2)
        callback.on_train_start(self.context)
        callback.on_train_step_start(self.context)
        previous = len(self.events)
        callback.abort()
        cleanup = [row[0] for row in self.events[previous:]]
        self.assertEqual(cleanup, ["nvtx_pop", "detach"])
        self.assertTrue(callback.started)
        self.assertFalse(callback.stopped)

    def test_failed_validation_does_not_stop_or_export_cuda_trace(self):
        callback = self.make()
        callback.on_train_start(self.context)
        callback.on_train_step_start(self.context)
        self.labels[-1].validation_error = DeepTraceError("incomplete node grid")
        with self.assertRaisesRegex(DeepTraceError, "incomplete node grid"):
            callback.on_train_step_end(self.context)
        callback.abort()
        self.assertEqual(self.names().count("detach"), 1)
        self.assertNotIn("profile_stop", self.names())
        self.assertNotIn("profile_export", self.names())
        self.assertFalse((callback.run / "raw/overlap_calls/step-00000/rank-0.json").exists())

    def test_failed_attach_and_skipped_update_allow_host_cleanup(self):
        callback = self.make()
        original = callback.label_type

        def failed_factory(*args, **kwargs):
            labeler = original(*args, **kwargs)
            labeler.attach_error = LookupError("attach failed")
            return labeler

        callback.label_type = failed_factory
        callback.on_train_start(self.context)
        with self.assertRaisesRegex(LookupError, "attach failed"):
            callback.on_train_step_start(self.context)
        callback.abort()
        self.assertEqual(self.names().count("detach"), 1)
        self.assertNotIn("profile_stop", self.names())
        self.assertNotIn("synchronize", self.names())

    def test_skipped_update_and_wrong_order_fail_closed(self):
        callback = self.make()
        callback.on_train_start(self.context)
        callback.on_train_step_start(self.context)
        self.context.skipped_iter = True
        with self.assertRaisesRegex(DeepTraceError, "skipped"):
            callback.on_train_step_end(self.context)
        callback.abort()
        self.assertEqual(self.names().count("detach"), 1)
        self.assertNotIn("profile_stop", self.names())
        with self.assertRaises(DeepTraceError):
            callback.on_train_end(self.context)

    def test_startup_rejects_configuration_and_rank_disagreement(self):
        callback = self.make("off")
        self.dist.disagree = True
        with self.assertRaisesRegex(DeepTraceError, "ranks disagree"):
            callback.on_train_start(self.context)
        self.dist.disagree = False
        self.config.delay_wgrad_compute = False
        with self.assertRaisesRegex(DeepTraceError, "flags"):
            callback.on_train_start(self.context)
        self.assertFalse(self.labels)
        self.assertNotIn("synchronize", self.names())

    def test_uccl_startup_records_official_effective_budget(self):
        callback = self.make("off", backend="uccl_ep")
        effective = {"dispatch": {"config": {"num_sms": 20}}, "combine": {"config": {"num_sms": 20}}}
        callback.args.uccl_budget_setup = effective
        budget = ModuleType("profiling.run_pretrain_uccl_budget")
        budget.runtime_adapter = lambda: SimpleNamespace(_buffer=None)
        budget.effective_configs = lambda fused, sms: effective
        with patch.dict(sys.modules, {budget.__name__: budget}):
            callback.on_train_start(self.context)
        record = json.loads((callback.run / "raw/ep_overlap_rank-0_startup.json").read_text())
        self.assertEqual(record["effective_configs"], effective)
        self.assertEqual(record["communication_sms"], 20)
        self.assertTrue(record["all_rank_configuration_agreement"])
        self.assertEqual(record["cuda_device_max_connections"], "32")
        self.assertIs(record["ddp_delay_wgrad_compute"], True)


if __name__ == "__main__":
    unittest.main()
