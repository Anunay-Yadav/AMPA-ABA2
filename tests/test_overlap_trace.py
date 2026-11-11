"""Overlap observer lifecycle checks; execute on allocated compute only.

The fake nodes check observer semantics, not Megatron math or GPU overlap.
Existing native/UCCL label tests cover the reused communication wrappers.
"""

from collections import Counter
from contextlib import contextmanager
import gc
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

from profiling.ampa_profile import overlap_trace as observer
from profiling.ampa_profile.deep_trace import DeepTraceError


class _Profiler:
    def __init__(self):
        self.entered = []
        self.open = []

    @contextmanager
    def record_function(self, label):
        self.entered.append(label)
        self.open.append(label)
        try:
            yield
        finally:
            if self.open.pop() != label:
                raise AssertionError("scope nesting changed")


class _Node:
    def __init__(self, name="node"):
        self.name = name
        self.calls = []
        self.fail = False

    def forward(self, *args, **kwargs):
        self.calls.append(("FWD", args, kwargs))
        if self.fail:
            raise ValueError("original node failed")
        return args[0] if args else self

    def backward(self, *args, **kwargs):
        self.calls.append(("BWD", args, kwargs))
        return args[0] if args else self


class _LayerNode(_Node):
    delay_wgrad_compute = True

    def backward_dw(self, *args, **kwargs):
        self.calls.append(("WGRAD", args, kwargs))
        return args[0] if args else self


class _Plan:
    def __init__(self):
        self.pre_process = _Node("preprocess")
        self.post_process = _Node("postprocess")
        self.layers = []
        for index in range(48):
            layer = SimpleNamespace(layer=SimpleNamespace(layer_number=index + 1))
            for slot in observer.NODE_SLOTS:
                setattr(layer, slot, _LayerNode(slot))
            self.layers.append(layer)

    def num_layers(self):
        return len(self.layers)

    def get_layer(self, index):
        return self.layers[index]


class _Model:
    def __init__(self):
        self.build_calls = []

    def build_schedule_plan(self, *args, **kwargs):
        self.build_calls.append((args, kwargs))
        return _Plan()


class _Inner:
    """Isolate the new observer from separately tested legacy phase hooks."""
    def __init__(self, torch, model, optimizer=None):
        self.counts = Counter()
        self.payload_bytes = Counter()
        self.payload_dtypes = {}
        self.layers = list(range(48))
        self.attached = False
        self.pairs = []

    def attach(self):
        self.attached = True

    def detach(self):
        self.attached = False

    def _required_phases(self):
        return {"FWD": ("pre_mlp_norm", "moe_total", "residual_bda", "router_total")}

    def pair_records(self):
        return self.pairs


class OverlapTraceTest(unittest.TestCase):
    def setUp(self):
        self.profiler = _Profiler()
        self.torch = SimpleNamespace(profiler=self.profiler)
        self.config = SimpleNamespace(**observer.SELECTED_FLAGS,
                                      pipeline_model_parallel_size=1,
                                      cuda_graph_impl="none")
        names = ("megatron", "megatron.core", "megatron.core.models",
                 "megatron.core.models.common", "megatron.core.models.common.utils",
                 "megatron.core.pipeline_parallel", "megatron.core.pipeline_parallel.utils",
                 "megatron.core.pipeline_parallel.combined_1f1b",
                 "megatron.core.utils")
        modules = {name: ModuleType(name) for name in names}
        for name, module in modules.items():
            parent, _, child = name.rpartition(".")
            if parent in modules:
                setattr(modules[parent], child, module)
        modules["megatron.core.models.common.utils"].TransformerLayerNode = _LayerNode
        modules["megatron.core.pipeline_parallel.utils"].ScheduleNode = _Node
        modules["megatron.core.utils"].get_model_config = lambda _: self.config
        modules["megatron.core.utils"].get_attr_wrapped_model = lambda model, *a, **kw: model
        self.combined_module = modules["megatron.core.pipeline_parallel.combined_1f1b"]

        def combined(forward_step_func, data_iterator, f_model, num_microbatches,
                     input_tensor, forward_data_store, b_model, b_input_tensor,
                     b_output_tensor, b_output_tensor_grad, config, current_microbatch=None):
            plan = f_model.build_schedule_plan() if f_model is not None else None
            if plan is not None:
                self.execute_forward(plan)
            if b_model is not None:
                self.execute_backward(b_output_tensor)
            return plan

        self.combined_module.combined_forward_backward_step = combined
        for patcher in (patch.dict(sys.modules, modules),
                        patch.object(observer, "_MethodLabels", _Inner),
                        patch.object(observer, "UcclMethodLabels", _Inner)):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.model = _Model()
        self.labels = observer.OverlapMethodLabels(self.torch, [self.model])
        self.addCleanup(self.labels.detach)

    def attach(self):
        self.labels.attach()

    def execute_forward(self, plan):
        value = object()
        plan.pre_process.forward(value)
        for layer in plan.layers:
            for slot in observer.NODE_SLOTS:
                getattr(layer, slot).forward(value)
        plan.post_process.forward(value)

    def execute_backward(self, plan):
        value = object()
        plan.post_process.backward(value)
        for layer in reversed(plan.layers):
            for slot in reversed(observer.NODE_SLOTS):
                getattr(layer, slot).backward(value)
            for slot in observer.WGRAD_SLOTS:
                getattr(layer, slot).backward_dw()
        plan.pre_process.backward(value)

    def complete(self):
        plan = None
        for index in range(5):
            plan = self.combined_module.combined_forward_backward_step(
                None, None, self.model if index < 4 else None, 4, None, [],
                self.model if index else None, None, plan, None, self.config,
                current_microbatch=index if index < 4 else None)
        for label, count in self.labels.expected_counts(4).items():
            if not label.startswith("AMPA/OVERLAP/"):
                self.labels.counts[label] = count

    def test_four_complete_microbatches_and_explicit_contract(self):
        self.attach()
        self.complete()
        self.labels.validate_counts(4)
        expected = self.labels.expected_counts(4)
        self.assertEqual(self.labels.counts, expected)
        self.assertEqual(len(self.labels._plans), 4)
        self.assertEqual(self.labels.counts[observer.PLAN_BUILD], 4)
        self.assertEqual(self.labels.combined_pass_records(),
                         [dict(row, returned=True) for row in observer.combined_pass_layout()])
        self.assertFalse(self.profiler.open)
        self.assertEqual(self.labels.pair_records(), [])
        self.assertFalse(any("microbatch_forward" in label or "microbatch_backward" in label
                             for label in expected))
        for phase in observer.DORMANT_FORWARD_PHASES:
            self.assertFalse(any(label.endswith("/" + phase) for label in expected))
        self.assertEqual(self.labels.contract_record()["contract"], observer.CONTRACT)

    def test_passthrough_and_exact_restoration_of_inherited_methods(self):
        node_forward, node_backward, wgrad = _Node.forward, _Node.backward, _LayerNode.backward_dw
        combined = self.combined_module.combined_forward_backward_step
        self.assertNotIn("build_schedule_plan", vars(self.model))
        self.attach()
        token, keyword = object(), object()
        plan = self.model.build_schedule_plan(token, named=keyword)
        self.assertEqual(self.model.build_calls, [((token,), {"named": keyword})])
        self.assertIs(plan.pre_process.forward(token, named=keyword), token)
        self.assertEqual(plan.pre_process.calls, [("FWD", (token,), {"named": keyword})])
        untouched = _Node()
        self.assertIs(untouched.forward(token, named=keyword), token)
        self.assertEqual(self.labels.counts[observer.node_label(0, "FWD", "pre_process")], 1)
        self.labels.detach()
        self.labels.detach()
        self.assertIs(_Node.forward, node_forward)
        self.assertIs(_Node.backward, node_backward)
        self.assertIs(_LayerNode.backward_dw, wgrad)
        self.assertIs(self.combined_module.combined_forward_backward_step, combined)
        self.assertNotIn("build_schedule_plan", vars(self.model))

    def test_node_exception_closes_scope_and_fails_validation(self):
        self.attach()
        plan = self.model.build_schedule_plan()
        plan.pre_process.fail = True
        with self.assertRaisesRegex(ValueError, "original node failed"):
            plan.pre_process.forward(object())
        self.assertFalse(self.profiler.open)
        with self.assertRaises(DeepTraceError):
            self.labels.validate_counts(4)

    def test_backward_and_weight_gradient_require_matching_forward(self):
        self.attach()
        plan = self.model.build_schedule_plan()
        node = plan.layers[0].mlp
        with self.assertRaisesRegex(DeepTraceError, "completed FWD"):
            node.backward(object())
        node.forward(object())
        with self.assertRaisesRegex(DeepTraceError, "completed BWD"):
            node.backward_dw()
        node.backward(object())
        node.backward_dw()
        with self.assertRaisesRegex(DeepTraceError, "repeated overlap node"):
            node.backward_dw()

    def test_missing_delayed_weight_gradient_and_extra_legacy_scope_rejected(self):
        self.attach()
        self.complete()
        missing = observer.node_label(3, "WGRAD", "pre_dispatch_computation", 0)
        del self.labels.counts[missing]
        with self.assertRaises(DeepTraceError):
            self.labels.validate_counts(4)
        self.labels.counts[missing] = 1
        self.labels.counts["AMPA/TRAIN/microbatch_backward"] = 4
        with self.assertRaises(DeepTraceError):
            self.labels.validate_counts(4)

    def test_observer_does_not_retain_plan_or_nodes(self):
        self.attach()
        plan = self.model.build_schedule_plan()
        weak_plan, weak_node = weakref.ref(plan), weakref.ref(plan.layers[0].mlp)
        del plan
        gc.collect()
        self.assertIsNone(weak_plan())
        self.assertIsNone(weak_node())
        self.assertEqual(len(self.labels._nodes), 0)

    def test_configuration_or_geometry_mismatch_fails_closed(self):
        self.config.delay_wgrad_compute = False
        with self.assertRaises(DeepTraceError):
            self.attach()
        self.config.delay_wgrad_compute = True
        self.attach()
        plan = _Plan()
        plan.layers[0].layer.layer_number = 2
        with self.assertRaisesRegex(DeepTraceError, "layer identity"):
            self.labels._register_plan(plan, _Node, _LayerNode)
        self.assertEqual(len(self.labels._nodes), 0)
        with self.assertRaisesRegex(DeepTraceError, "accumulation four"):
            self.labels.expected_counts(3)

    def test_partial_attach_restores_class_and_model_methods(self):
        forward, backward, wgrad = _Node.forward, _Node.backward, _LayerNode.backward_dw
        combined = self.combined_module.combined_forward_backward_step
        original_set = self.labels._set

        def fail_model_patch(owner, name, value):
            if owner is self.model:
                raise ValueError("failed setup")
            return original_set(owner, name, value)

        with patch.object(self.labels, "_set", side_effect=fail_model_patch):
            with self.assertRaisesRegex(ValueError, "failed setup"):
                self.attach()
        self.assertIs(_Node.forward, forward)
        self.assertIs(_Node.backward, backward)
        self.assertIs(_LayerNode.backward_dw, wgrad)
        self.assertIs(self.combined_module.combined_forward_backward_step, combined)
        self.assertFalse(self.labels._inner.attached)
        self.assertFalse(self.labels._restore)
        self.assertNotIn("build_schedule_plan", vars(self.model))

    def test_builder_exception_is_propagated_and_reversible(self):
        def failed(*args, **kwargs):
            raise LookupError("native plan builder failed")

        self.model.build_schedule_plan = failed
        self.attach()
        with self.assertRaisesRegex(LookupError, "native plan builder failed"):
            self.model.build_schedule_plan(object(), named=object())
        self.assertFalse(self.profiler.open)
        self.assertEqual(self.labels.counts[observer.PLAN_BUILD], 1)
        self.assertFalse(self.labels._completed)
        self.labels.detach()
        self.assertIs(self.model.build_schedule_plan, failed)

    def test_native_and_uccl_require_complete_pair_grid(self):
        self.labels = observer.OverlapMethodLabels(self.torch, [self.model], communication_backend="uccl_ep")
        self.addCleanup(self.labels.detach)
        self.attach()
        self.complete()
        with self.assertRaisesRegex(DeepTraceError, "journal"):
            self.labels.validate_counts(4)

        pairs = []
        for layer in range(48):
            for family in ("dispatch", "combine"):
                for _ in range(4):
                    pairs.append({"call_id": len(pairs), "layer": layer, "family": family,
                                  "forward_returned": True, "backward_returned": True})
        self.labels._inner.pairs = pairs
        self.labels.validate_counts(4)
        pairs[-1]["backward_returned"] = False
        with self.assertRaisesRegex(DeepTraceError, "journal"):
            self.labels.validate_counts(4)

    def test_combined_pass_rejects_wrong_forward_backward_schedule(self):
        self.attach()
        with self.assertRaisesRegex(DeepTraceError, "native GA4 schedule"):
            self.combined_module.combined_forward_backward_step(
                None, None, self.model, 4, None, [], self.model, None, None, None,
                self.config, current_microbatch=0)
        self.assertFalse(self.labels.combined_pass_records())
        self.assertFalse(self.profiler.open)


if __name__ == "__main__":
    unittest.main()
