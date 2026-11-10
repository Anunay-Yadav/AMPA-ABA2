"""The diagnostic wrappers preserve calls and restore the original methods."""

from __future__ import annotations

from contextlib import nullcontext
import sys
import types
import unittest
from unittest.mock import patch

from profiling.ampa_profile.deep_trace import _MethodLabels


class _Tensor:
    dtype = "bf16"

    def __init__(self, value):
        self.value = value

    def numel(self):
        return 1

    def element_size(self):
        return 2

    def __add__(self, other):
        return _Tensor(self.value + (other.value if isinstance(other, _Tensor) else other))


class _Router:
    module = None

    def gating(self, value):
        return value

    def routing(self, value):
        selected = self.module.topk_routing_with_score_function(value)
        scores = self.module.compute_routing_scores_for_aux_loss(selected)
        return self._apply_aux_loss(scores)

    def _apply_aux_loss(self, value):
        return value


class _Linear:
    def forward(self, value):
        return value


class _Experts:
    def __init__(self):
        self.linear_fc1 = _Linear()
        self.linear_fc2 = _Linear()

    def forward(self, value):
        return self.linear_fc2.forward(self.linear_fc1.forward(value))


class MoEAlltoAllTokenDispatcher:
    def __init__(self, group, module):
        self.ep_group = group
        self.module = module

    def preprocess(self, value):
        return value

    def _maybe_dtoh_and_synchronize(self, point, value):
        return value

    def dispatch_preprocess(self, value):
        return self.module.permute(self.preprocess(self._maybe_dtoh_and_synchronize("before_permutation_1", value)))

    def token_dispatch(self, value):
        return self.module.all_to_all(self.ep_group, value) + self.module.all_to_all(self.ep_group, value)

    def dispatch_postprocess(self, value):
        return self.module.sort_chunks_by_idxs(value)

    def combine_preprocess(self, value):
        return self.module.sort_chunks_by_idxs(value)

    def token_combine(self, value):
        return self.module.all_to_all(self.ep_group, value)

    def combine_postprocess(self, value):
        return self.module.unpermute(value)


class _MoELayer:
    use_shared_expert = False
    moe_layer_recompute = False

    def __init__(self, group, module):
        self.router = _Router()
        self.token_dispatcher = MoEAlltoAllTokenDispatcher(group, module)
        self.experts = _Experts()

    def route(self, value):
        return self.router.routing(self.router.gating(value))

    def forward(self, value):
        value = self.route(value)
        value = self.token_dispatcher.dispatch_preprocess(value)
        value = self.token_dispatcher.token_dispatch(value)
        value = self.token_dispatcher.dispatch_postprocess(value)
        value = self.experts.forward(value)
        value = self.token_dispatcher.combine_preprocess(value)
        value = self.token_dispatcher.token_combine(value)
        return self.token_dispatcher.combine_postprocess(value)


class _Transformer:
    def __init__(self, number, group, module):
        self.layer_number = number
        self.mlp = _MoELayer(group, module)

    def _pre_mlp_layernorm_and_residual(self, value):
        return value

    def _apply_mlp_bda_step(self, value):
        return value


class _Model:
    def __init__(self, group, module):
        self.layers = [_Transformer(number, group, module) for number in range(1, 49)]

    def named_modules(self):
        return ((f"layer.{number}", layer) for number, layer in enumerate(self.layers))


class _Optimizer:
    def step(self):
        return "updated"


class DeepTraceTest(unittest.TestCase):
    def test_forward_a2a_labels_do_not_change_result_and_are_restored(self):
        modules = {name: types.ModuleType(name) for name in (
            "megatron", "megatron.core", "megatron.core.transformer",
            "megatron.core.transformer.moe", "megatron.core.transformer.moe.moe_layer",
            "megatron.core.transformer.moe.token_dispatcher", "megatron.core.transformer.moe.router",
            "megatron.core.pipeline_parallel", "megatron.core.pipeline_parallel.schedules",
            "megatron.core.utils",
        )}
        modules["megatron.core.transformer.moe.moe_layer"].MoELayer = _MoELayer
        dispatcher_module = modules["megatron.core.transformer.moe.token_dispatcher"]
        dispatcher_module.all_to_all = lambda group, value: value + 1
        dispatcher_module.permute = lambda value: value
        dispatcher_module.sort_chunks_by_idxs = lambda value: value
        dispatcher_module.unpermute = lambda value: value
        modules["megatron.core.transformer.moe"].token_dispatcher = dispatcher_module
        router_module = modules["megatron.core.transformer.moe.router"]
        router_module.topk_routing_with_score_function = lambda value: value
        router_module.compute_routing_scores_for_aux_loss = lambda value: value
        modules["megatron.core.transformer.moe"].router = router_module
        _Router.module = router_module
        original_a2a = dispatcher_module.all_to_all
        schedules = modules["megatron.core.pipeline_parallel.schedules"]
        schedules.forward_step = lambda value: value
        schedules.backward_step = lambda value: value
        modules["megatron.core.pipeline_parallel"].schedules = schedules
        config = types.SimpleNamespace(finalize_model_grads_func=lambda value: value)
        modules["megatron.core.utils"].get_model_config = lambda model: config
        group = object()
        model = _Model(group, dispatcher_module)
        optimizer = _Optimizer()
        torch = types.SimpleNamespace(profiler=types.SimpleNamespace(record_function=lambda _: nullcontext()))
        with patch.dict(sys.modules, modules):
            labeler = _MethodLabels(torch, [model], optimizer)
            labeler.attach()
            try:
                self.assertEqual(schedules.forward_step("forward"), "forward")
                self.assertEqual(schedules.backward_step("backward"), "backward")
                self.assertEqual(config.finalize_model_grads_func("gradients"), "gradients")
                self.assertEqual(optimizer.step(), "updated")
                self.assertEqual(model.layers[0].mlp.forward(_Tensor(2)).value, 7)
                self.assertEqual(labeler.counts["AMPA/FWD/layer_00/dispatch_activations_a2a"], 1)
                self.assertEqual(labeler.counts["AMPA/FWD/layer_00/dispatch_probabilities_a2a"], 1)
                self.assertEqual(labeler.counts["AMPA/FWD/layer_00/combine_expert_output_a2a"], 1)
                self.assertEqual(labeler.counts["AMPA/FWD/layer_00/first_permute"], 1)
                self.assertEqual(labeler.counts["AMPA/FWD/layer_00/router_aux_scores"], 1)
                self.assertEqual(labeler.payload_bytes["AMPA/FWD/layer_00/combine_expert_output_a2a"], 2)
                self.assertEqual(labeler.counts["AMPA/TRAIN/gradient_finalize"], 1)
                self.assertEqual(labeler.counts["AMPA/TRAIN/optimizer_step"], 1)
            finally:
                labeler.detach()
        self.assertIs(dispatcher_module.all_to_all, original_a2a)
        self.assertNotIn("forward", vars(model.layers[0].mlp))


if __name__ == "__main__":
    unittest.main()
