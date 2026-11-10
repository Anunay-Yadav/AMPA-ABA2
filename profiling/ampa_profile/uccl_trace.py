"""Labels for unchanged native Flex/UCCL calls, including their adjoints.

These scopes identify host launches. They do not measure wire traffic, CPU
proxy work or exposed communication latency. Fine backward children require
complete semantic rollups when compared with a broad native backward scope.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from itertools import count

from .deep_trace import DeepTraceError, _DISPATCH_STAGE, _MethodLabels


_EPOCHS = count(1)
_CALL = ContextVar("ampa_uccl_trace_call", default=None)
_PAIR_ATTRIBUTE = "_ampa_uccl_trace_pair"


class UcclMethodLabels(_MethodLabels):
    dispatcher_type = "MoEFlexTokenDispatcher"

    def __init__(self, torch, model, optimizer=None):
        super().__init__(torch, model, optimizer)
        self.epoch = next(_EPOCHS)
        self.pairs = {}

    @contextmanager
    def _call_scope(self, direction, layer, phase, call_id):
        label = f"AMPA/{direction}/layer_{layer:02d}/{phase}"
        token = _CALL.set((direction, layer, call_id))
        self.counts[label] += 1
        try:
            with self.torch.profiler.record_function(label):
                with self.torch.profiler.record_function(f"AMPA/UCCL_CALL/{call_id}/{direction}"):
                    yield
        finally:
            _CALL.reset(token)

    def _label_autograd(self, owner, family, forward_phase, backward_phase):
        original_forward, original_backward = owner.forward, owner.backward
        expected_stage = "dispatch_total" if family == "dispatch" else "combine_total"

        def forward(ctx, *args, **kwargs):
            context = _DISPATCH_STAGE.get()
            if context is None:
                return original_forward(ctx, *args, **kwargs)
            layer, stage = context
            if type(layer) is not int or layer not in range(48) or stage != expected_stage:
                raise DeepTraceError(f"unexpected {family} forward context: {context}")
            if hasattr(ctx, _PAIR_ATTRIBUTE):
                raise DeepTraceError("autograd context already has an AMPA call identity")
            call_id = len(self.pairs)
            setattr(ctx, _PAIR_ATTRIBUTE, (self.epoch, call_id, layer, family))
            self.pairs[call_id] = {"call_id": call_id, "layer": layer, "family": family,
                                   "context_id": id(ctx),
                                   "forward_returned": False, "backward_returned": False}
            with self._call_scope("FWD", layer, forward_phase, call_id):
                result = original_forward(ctx, *args, **kwargs)
            self.pairs[call_id]["forward_returned"] = True
            return result

        def backward(ctx, *args, **kwargs):
            identity = getattr(ctx, _PAIR_ATTRIBUTE, None)
            if identity is None:
                # Only calls whose forward ran in an observed MoE dispatcher
                # context belong to this journal. Full-model count gates still
                # require every expected observed forward and backward.
                return original_backward(ctx, *args, **kwargs)
            if (type(identity) is not tuple or len(identity) != 4
                    or any(type(value) is not int for value in identity[:3])):
                raise DeepTraceError("malformed UCCL autograd trace identity")
            epoch, call_id, layer, saved_family = identity
            if epoch != self.epoch or saved_family != family or call_id not in self.pairs:
                raise DeepTraceError("backward context belongs to another trace or operation")
            pair = self.pairs[call_id]
            if (pair["layer"] != layer or pair["family"] != family
                    or pair["context_id"] != id(ctx)
                    or not pair["forward_returned"] or pair["backward_returned"]):
                raise DeepTraceError("missing, repeated or mismatched UCCL backward call")
            with self._call_scope("BWD", layer, backward_phase, call_id):
                result = original_backward(ctx, *args, **kwargs)
            pair["backward_returned"] = True
            return result

        self._set(owner, "forward", staticmethod(forward))
        self._set(owner, "backward", staticmethod(backward))

    def _attach_communication(self, dispatcher_module, observed_a2a):
        del observed_a2a
        import uccl.ep as ep
        import deep_ep.buffer as wrapper
        from megatron.core.transformer.moe import fused_a2a

        if (ep.num_max_nvl_peers() != 4 or wrapper.ep is not ep
                or fused_a2a.Buffer is not wrapper.Buffer or not fused_a2a.HAVE_DEEP_EP):
            raise DeepTraceError("trace backend is not the four-peer UCCL compatibility Buffer")
        self._label_autograd(fused_a2a.FusedDispatch, "dispatch", "uccl_dispatch", "uccl_combine_dinput_dprob")
        self._label_autograd(fused_a2a.FusedCombine, "combine", "uccl_combine", "uccl_dispatch_doutput")
        original_layout = wrapper.Buffer.get_dispatch_layout

        def layout(buffer, *args, **kwargs):
            context = _CALL.get()
            if context is None:
                return original_layout(buffer, *args, **kwargs)
            direction, layer, call_id = context
            if direction != "FWD" or self.pairs[call_id]["family"] != "dispatch":
                raise DeepTraceError("cached backward or combine unexpectedly recomputed dispatch layout")
            label = f"AMPA/FWD/layer_{layer:02d}/uccl_layout"
            self.counts[label] += 1
            with self.torch.profiler.record_function(label):
                return original_layout(buffer, *args, **kwargs)

        self._set(wrapper.Buffer, "get_dispatch_layout", layout)
        self._label_alias(dispatcher_module, "permute", _DISPATCH_STAGE,
                          {(layer, "dispatch_postprocess"): "local_expert_sort_core" for layer in range(48)})
        self._label_alias(dispatcher_module, "unpermute", _DISPATCH_STAGE,
                          {(layer, "combine_preprocess"): "inverse_local_sort_core" for layer in range(48)})

    def _attach_dispatcher(self, dispatcher, layer):
        if type(dispatcher._comm_manager).__name__ != "_DeepepManager":
            raise DeepTraceError("UCCL trace requires the existing native DeepEP manager")
        for method, phase, context in (
            ("dispatch_preprocess", "dispatch_preprocess", "dispatch_preprocess"),
            ("token_dispatch", "dispatch_total", "dispatch_total"),
            ("dispatch_postprocess", "local_expert_sort", "dispatch_postprocess"),
            ("combine_preprocess", "inverse_local_sort", "combine_preprocess"),
            ("token_combine", "combine_total", "combine_total"),
            ("combine_postprocess", "final_unpermute", "combine_postprocess"),
        ):
            self._label_context_method(dispatcher, method, f"AMPA/FWD/layer_{layer:02d}/{phase}",
                                       _DISPATCH_STAGE, (layer, context))

    def _required_phases(self):
        return {"FWD": (
            "pre_mlp_norm", "moe_total", "router_total", "router_gate", "router_routing",
            "router_main_topk", "router_aux_scores", "router_aux_loss", "dispatch_preprocess",
            "dispatch_total", "uccl_layout", "uccl_dispatch", "local_expert_sort", "local_expert_sort_core",
            "expert_total", "expert_fc1", "expert_fc2", "inverse_local_sort", "inverse_local_sort_core",
            "combine_total", "uccl_combine", "final_unpermute", "residual_bda",
        ), "BWD": ("uccl_combine_dinput_dprob", "uccl_dispatch_doutput")}

    def validate_counts(self, accumulation):
        super().validate_counts(accumulation)
        if len(self.pairs) != 2 * len(self.layers) * accumulation:
            raise DeepTraceError("UCCL forward/backward journal has an incomplete call grid")
        for pair in self.pairs.values():
            if not pair["forward_returned"] or not pair["backward_returned"]:
                raise DeepTraceError("UCCL autograd call lacks a completed forward or backward")

    def pair_records(self):
        return [dict(self.pairs[key], epoch=self.epoch) for key in sorted(self.pairs)]
