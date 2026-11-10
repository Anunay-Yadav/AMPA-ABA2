"""CPU-only checks for early budgets, unchanged call wiring, and NVTX boundaries."""
import argparse
from contextlib import redirect_stderr
from io import StringIO
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from profiling import run_pretrain_uccl_budget as budget
from profiling.ampa_profile.deep_trace import _MethodLabels


def args(root=None, **changes):
    values = dict(communication_sms=32, communication_backend="uccl_ep", router_num_groups=None,
        router_group_topk=None, torch_profile_all_steps=False, torch_profile_steps=0, nsys_capture_steps=0,
        warmup_steps=5, measured_steps=5, completed_steps=10, run_dir=root)
    return argparse.Namespace(**{**values, **changes})


def fused_fixture(events):
    def config(sms, send):
        result = SimpleNamespace(**dict(zip(budget.CONFIG_FIELDS, [sms, send, 288, 20 if send == 36 else 12, 128])))
        result.get_nvl_buffer_size_hint = lambda hidden, ranks: sms * hidden + ranks
        result.get_rdma_buffer_size_hint = lambda hidden, ranks: sms * hidden * 2 + ranks
        return result

    class Buffer:
        num_sms = 20

        @classmethod
        def set_num_sms(cls, value):
            events.append(("set_budget", value))
            cls.num_sms = value

        @classmethod
        def get_dispatch_config(cls, ranks):
            assert ranks == 16
            events.append(("dispatch_config", cls.num_sms))
            return config(cls.num_sms, 36)

        @classmethod
        def get_combine_config(cls, ranks):
            assert ranks == 16
            events.append(("combine_config", cls.num_sms))
            return config(cls.num_sms, 4)
    return SimpleNamespace(Buffer=Buffer, HAVE_DEEP_EP=True, _buffer=None)


def capture_fixture(root, label_type=None):
    events = []
    fused = fused_fixture(events)
    selected = args(root)
    selected.uccl_budget_setup = budget.configure_before_model(fused, selected.communication_sms)

    def start():
        events.append(("start",))
        return 0

    def stop():
        events.append(("stop",))
        return 0

    cuda = SimpleNamespace(synchronize=lambda: events.append(("synchronize",)),
        cudart=lambda: SimpleNamespace(cudaProfilerStart=start, cudaProfilerStop=stop),
        check_error=lambda result: None if result == 0 else (_ for _ in ()).throw(RuntimeError("CUDA failure")),
        nvtx=SimpleNamespace(range_push=lambda label: events.append(("push", label)),
                             range_pop=lambda: events.append(("pop",))))
    torch = SimpleNamespace(cuda=cuda)  # No global torch.profiler interface.
    dist = SimpleNamespace(get_rank=lambda: 0, get_world_size=lambda: 16,
                           barrier=lambda: events.append(("barrier",)),
                           all_gather_object=lambda output, value: output.__setitem__(slice(None), [value] * 16))

    class Labels:
        def __init__(self, proxy, model, optimizer):
            self.proxy = proxy
            self.layers = list(range(48))
            self.counts = {"AMPA/TRAIN/microbatch_forward": 4, "AMPA/TRAIN/microbatch_backward": 4,
                           "AMPA/TRAIN/gradient_finalize": 1, "AMPA/TRAIN/optimizer_step": 1}

        def attach(self):
            events.append(("attach",))
            with self.proxy.profiler.record_function("fake_phase"):
                pass

        def validate_counts(self, accumulation):
            assert accumulation == 4
            events.append(("validate",))

        def pair_records(self):
            return [{"call_id": n, "forward_returned": True, "backward_returned": True} for n in range(384)]

        def detach(self):
            events.append(("detach",))

    cb = budget._BudgetCapture(selected, torch, dist, fused, label_type or Labels)
    context = SimpleNamespace(state=SimpleNamespace(cfg=SimpleNamespace(model=SimpleNamespace(moe_flex_dispatcher_num_sms=32)),
                                                   train_state=SimpleNamespace(step=0)),
                              model=[object()], optimizer=object(), skipped_iter=False)
    return cb, context, events


class UcclBudgetEntrypointTests(unittest.TestCase):
    def test_parse_uses_existing_reuse_path_and_rejects_grouped_or_profiler_modes(self):
        for sms in budget.BUDGETS:
            with self.subTest(sms=sms), patch.object(budget.reuse, "parse_reuse_args", return_value=args()) as parse:
                parsed = budget.parse_budget_args(["--communication-sms", str(sms), "--run-dir", "/tmp/run"])
            self.assertEqual(parsed.communication_sms, sms)
            parse.assert_called_once_with(["--run-dir", "/tmp/run"])
        for change in [dict(communication_backend="alltoall"), dict(router_num_groups=4, router_group_topk=2),
                       dict(torch_profile_all_steps=True), dict(nsys_capture_steps=1), dict(completed_steps=8)]:
            with self.subTest(change=change), patch.object(budget.reuse, "parse_reuse_args", return_value=args(**change)), \
                    redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                budget.parse_budget_args(["--communication-sms", "32"])

    def test_only_requested_even_budgets_and_no_inherited_overrides(self):
        self.assertEqual(budget.BUDGETS, (20, 12, 32, 40))
        for value in ["0", "21", "24", "64"]:
            with self.subTest(value=value), redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                budget.parse_budget_args(["--communication-sms", value])
        for key in budget.OVERRIDES:
            for value in ["", "20,36,288,20,128"]:
                with self.subTest(key=key, value=value), patch.dict(os.environ, {key: value}), \
                        patch.object(budget.reuse, "parse_reuse_args") as parse, self.assertRaisesRegex(RuntimeError, "unset"):
                    budget.parse_budget_args(["--communication-sms", "32"])
                parse.assert_not_called()

    def test_official_setter_precedes_configs_and_records_budget_dependent_hints(self):
        for sms in budget.BUDGETS:
            events = []
            fused = fused_fixture(events)
            configs = budget.configure_before_model(fused, sms)
            self.assertEqual(events[0], ("set_budget", sms))
            for direction in ["dispatch", "combine"]:
                self.assertEqual(configs[direction]["config"]["num_sms"], sms)
                self.assertEqual(configs[direction]["nvl_buffer_size_hint"], sms * 4096 + 16)
                self.assertEqual(configs[direction]["rdma_buffer_size_hint"], sms * 8192 + 16)

    def test_existing_buffer_or_effective_config_override_is_rejected(self):
        fused = fused_fixture([])
        fused._buffer = object()
        with self.assertRaisesRegex(RuntimeError, "before any"):
            budget.configure_before_model(fused, 32)
        fused._buffer = None
        original = fused.Buffer.get_combine_config(16)
        fused.Buffer.get_combine_config = lambda ranks: original
        with self.assertRaisesRegex(RuntimeError, "effective combine"):
            budget.configure_before_model(fused, 32)

    def test_builder_keeps_debug_cache_and_sets_native_manager_option(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            selected = args(root)
            config = SimpleNamespace(model=SimpleNamespace(hidden_size=2048, moe_router_topk=8,
                moe_router_num_groups=None, moe_router_group_topk=None), profiling=SimpleNamespace(),
                train=SimpleNamespace(train_iters=25, exit_interval=10))
            events = []
            with patch.object(budget, "runtime_adapter", return_value=fused_fixture(events)), \
                    patch.object(budget.reuse.debug, "build_debug_config", return_value=config) as build, \
                    patch.dict(os.environ, {"RANK": "0"}):
                result = budget.build_budget_config(selected)
            build.assert_called_once_with(selected)
            self.assertEqual((result.train.train_iters, result.train.exit_interval), (25, 10))
            self.assertEqual(result.model.moe_flex_dispatcher_num_sms, 32)
            self.assertFalse(result.profiling.use_nsys_profiler)
            self.assertFalse(result.profiling.use_pytorch_profiler)
            self.assertTrue(result.profiling.nvtx_ranges)
            record = json.loads((root / "config/uccl_budget_contract.json").read_text())
            self.assertEqual(record["capture_step_indices"], list(range(5, 10)))
            self.assertEqual(record["capture_ranks"], list(range(16)))

    def test_nvtx_facade_preserves_original_arguments_results_and_restores_methods(self):
        events = []
        torch = SimpleNamespace(cuda=SimpleNamespace(nvtx=SimpleNamespace(
            range_push=lambda name: events.append(name), range_pop=lambda: events.append("pop"))))
        result, argument = object(), object()

        class Owner:
            def work(self, value, *, named):
                self.calls.append((value, named))
                return result

        owner = Owner()
        owner.calls = []
        labels = _MethodLabels(budget.NvtxTorchProxy(torch), [])
        labels._label_method(owner, "work", "test_scope")
        self.assertIs(owner.work(argument, named=argument), result)
        self.assertEqual(owner.calls, [(argument, argument)])
        self.assertEqual(events, ["test_scope", "pop"])
        labels.detach()
        self.assertNotIn("work", vars(owner))

    def test_all_ten_updates_preserve_warmup_and_capture_exactly_five_measured_steps(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            cb, context, events = capture_fixture(root)
            cb.on_train_start(context)
            for step in range(10):
                context.state.train_state.step = step
                cb.on_train_step_start(context)
                if step < 5:
                    self.assertFalse(cb.started)
                cb.on_train_step_end(context)
            cb.on_train_end(context)
            boundary = [e[0] for e in events if e[0] in ("synchronize", "barrier", "start", "stop")]
            self.assertEqual(boundary, ["synchronize", "barrier", "start", "barrier", "synchronize", "barrier", "stop"])
            self.assertEqual(sum(e[0] == "attach" for e in events), 5)
            self.assertEqual(sum(e[0] == "detach" for e in events), 5)
            before = list(events)
            cb.abort()
            self.assertEqual(events, before)
            record = json.loads((root / "raw/uccl_budget_rank-0_capture.json").read_text())
            self.assertTrue(record["valid_nvtx_capture"])
            self.assertEqual(record["capture_steps"], list(range(5, 10)))
            self.assertEqual(len(list((root / "raw/nsys_calls").glob("step-*/rank-0.json"))), 5)
            self.assertTrue((root / "uccl_budget_environment.json").is_file())
            # Final step range closes and labels restore before the last synchronization.
            last_sync = max(i for i, e in enumerate(events) if e[0] == "synchronize")
            self.assertEqual(events[last_sync - 1][0], "detach")

    def test_changed_budget_skipped_updates_and_incomplete_capture_fail(self):
        with TemporaryDirectory() as directory:
            cb, context, _ = capture_fixture(Path(directory))
            cb.fused.Buffer.num_sms = 20
            with self.assertRaisesRegex(RuntimeError, "budget changed"):
                cb.on_train_step_start(context)
            cb.fused.Buffer.num_sms = 32
            cb.on_train_step_start(context)
            context.skipped_iter = True
            with self.assertRaisesRegex(RuntimeError, "skipped"):
                cb.on_train_step_end(context)
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                cb.on_train_end(context)

    def test_cross_rank_budget_or_hint_disagreement_fails_before_training(self):
        with TemporaryDirectory() as directory:
            cb, context, _ = capture_fixture(Path(directory))
            def gather(output, value):
                output[:] = [value] * 15 + [{"communication_sms": 20, "effective_configs": value["effective_configs"]}]
            cb.dist.all_gather_object = gather
            with self.assertRaisesRegex(RuntimeError, "ranks disagree"):
                cb.on_train_start(context)
            self.assertFalse((Path(directory) / "raw/uccl_budget_rank-0_startup.json").exists())

    def test_abort_restores_active_labels_and_stops_without_cross_rank_wait(self):
        with TemporaryDirectory() as directory:
            cb, context, events = capture_fixture(Path(directory))
            cb.on_train_start(context)
            for step in range(5):
                context.state.train_state.step = step
                cb.on_train_step_start(context)
                cb.on_train_step_end(context)
            context.state.train_state.step = 5
            cb.on_train_step_start(context)
            before = len(events)
            cb.abort()
            self.assertEqual([e[0] for e in events[before:]], ["pop", "detach", "stop"])
            self.assertFalse((Path(directory) / "raw/uccl_budget_rank-0_capture.json").exists())

    def test_scoped_main_calls_original_trainer_once_and_restores_on_failure(self):
        original_train = Mock(side_effect=RuntimeError("trainer failure"))
        original_provenance = Mock(return_value=object())
        trainer = SimpleNamespace(pretrain=original_train)
        provenance = SimpleNamespace(make_indexed_run_provenance_callback=original_provenance)
        callback = SimpleNamespace(abort=Mock())
        selected = args()
        original = budget.canonical.parse_args, budget.canonical.build_config, budget.canonical.comparison_contract
        forward, retained_callback = object(), object()

        def original_main():
            budget.canonical.parse_args()
            provenance.make_indexed_run_provenance_callback(nsys_trace=False, run_dir="same")
            trainer.pretrain(config="same_config", forward_step_func=forward, callbacks=[retained_callback])

        def imported(name):
            return trainer if name == "megatron.bridge.training.pretrain" else provenance

        with patch.object(budget.importlib, "import_module", side_effect=imported), \
                patch.object(budget, "parse_budget_args", return_value=selected), \
                patch.object(budget, "make_budget_callback", return_value=callback), \
                patch.object(budget.canonical, "main", side_effect=original_main), \
                self.assertRaisesRegex(RuntimeError, "trainer failure"):
            budget.main()
        original_train.assert_called_once_with(config="same_config", forward_step_func=forward,
                                                callbacks=[callback, retained_callback])
        original_provenance.assert_called_once_with(nsys_trace=True, run_dir="same")
        callback.abort.assert_called_once()
        self.assertEqual((budget.canonical.parse_args, budget.canonical.build_config, budget.canonical.comparison_contract), original)
        self.assertIs(trainer.pretrain, original_train)
        self.assertIs(provenance.make_indexed_run_provenance_callback, original_provenance)


if __name__ == "__main__":
    unittest.main()
