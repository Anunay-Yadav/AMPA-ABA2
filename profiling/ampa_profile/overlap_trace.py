"""Observe the native EP-overlap schedule without changing its execution.

The combined-1F1B path has no enclosing ordinary microbatch forward/backward
calls. Its forward, backward and delayed weight-gradient nodes are labelled
separately. These host scopes identify launches, not additive GPU step time.
Existing router/dispatcher/expert labels and the UCCL autograd journal remain
in use. No tensor, stream, event, synchronization or schedule is changed.
"""

from __future__ import annotations

from collections import Counter
from inspect import signature
from types import MethodType
from weakref import WeakKeyDictionary

from .deep_trace import DeepTraceError, _MethodLabels
from .uccl_trace import UcclMethodLabels


CONTRACT = "ampa_ep_overlap_nodes_v1"
NODE_SLOTS = ("pre_dispatch_computation", "moe_dispatch", "mlp", "moe_combine")
WGRAD_SLOTS = ("pre_dispatch_computation", "mlp")
DORMANT_FORWARD_PHASES = frozenset(("pre_mlp_norm", "moe_total", "residual_bda"))
PLAN_BUILD = "AMPA/OVERLAP/TRAIN/plan_build"
SELECTED_FLAGS = {
    "overlap_moe_expert_parallel_comm": True,
    "delay_wgrad_compute": True,
    "ep_overlap_early_attn_memory_release": True,
    "overlap_dispatch_backward_with_experts_wgrad": False,
    "moe_shared_expert_overlap": False,
}


def node_label(microbatch, direction, slot, layer=None):
    """Name one real schedule-node invocation, including its microbatch."""
    root = f"AMPA/OVERLAP/{direction}/microbatch_{microbatch:02d}"
    return f"{root}/{slot}" if layer is None else f"{root}/layer_{layer:02d}/{slot}"


def combined_pass_label(index):
    """Name one real combined forward/backward scheduler call."""
    return f"AMPA/OVERLAP/TRAIN/combined_pass_{index:02d}"


def combined_pass_layout():
    return [{"pass": index, "forward_microbatch": index if index < 4 else None,
             "backward_microbatch": index - 1 if index else None}
            for index in range(5)]


class OverlapMethodLabels:
    """Selected PP1, GA4, 48-layer Qwen overlap observation contract.

    ``counts`` and ``pair_records`` retain the legacy labeler interfaces.
    Consumers must select this contract and use ``expected_counts`` instead
    of the ordinary schedule's validation rules.
    """

    def __init__(self, torch, model, optimizer=None, communication_backend="alltoall"):
        if communication_backend not in ("alltoall", "uccl_ep"):
            raise DeepTraceError("unsupported overlap communication backend")
        owner = UcclMethodLabels if communication_backend == "uccl_ep" else _MethodLabels
        self._inner = owner(torch, model, optimizer)
        self.torch, self.model = torch, model
        self.communication_backend = communication_backend
        self.counts = self._inner.counts
        self.payload_bytes = self._inner.payload_bytes
        self.payload_dtypes = self._inner.payload_dtypes
        self._restore = []
        # Weak keys are essential: retaining a node until step end would keep
        # its detached activation references alive after the native release.
        self._nodes = WeakKeyDictionary()
        self._plans = []
        self._passes = []
        self._active_pass = None
        self._completed = Counter()
        self._failed = []
        self._attached = False
        self._used = False

    @property
    def layers(self):
        return self._inner.layers

    def _set(self, owner, name, value):
        own = vars(owner)
        self._restore.append((owner, name, name in own, own.get(name)))
        setattr(owner, name, value)

    def _observe_node(self, node, direction, original, args, kwargs):
        tag = self._nodes.get(node)
        if tag is None:
            return original(node, *args, **kwargs)
        microbatch, layer, slot = tag
        label = node_label(microbatch, direction, slot, layer)
        if direction == "WGRAD" and slot not in WGRAD_SLOTS:
            self._failed.append(label)
            raise DeepTraceError("unexpected delayed weight-gradient node")
        if self.counts.get(label, 0):
            self._failed.append(label)
            raise DeepTraceError(f"repeated overlap node: {label}")
        predecessor = {"BWD": "FWD", "WGRAD": "BWD"}.get(direction)
        if predecessor and self._completed.get(node_label(microbatch, predecessor, slot, layer)) != 1:
            self._failed.append(label)
            raise DeepTraceError(f"overlap node lacks its completed {predecessor}: {label}")
        self.counts[label] += 1
        try:
            with self.torch.profiler.record_function(label):
                result = original(node, *args, **kwargs)
        except BaseException:
            self._failed.append(label)
            raise
        self._completed[label] += 1
        return result

    def _wrap_node_method(self, owner, name, direction):
        original = getattr(owner, name)

        def observed(node, *args, **kwargs):
            return self._observe_node(node, direction, original, args, kwargs)

        self._set(owner, name, observed)

    def _register_plan(self, plan, node_type, layer_node_type):
        microbatch = len(self._plans)
        if microbatch >= 4 or plan.num_layers() != 48:
            raise DeepTraceError("overlap plan differs from four-microbatch/48-layer workload")
        pending = []
        for slot in ("pre_process", "post_process"):
            node = getattr(plan, slot, None)
            if not isinstance(node, node_type):
                raise DeepTraceError(f"overlap plan lacks its {slot} schedule node")
            pending.append((node, (microbatch, None, slot)))
        for index in range(48):
            layer_plan = plan.get_layer(index)
            if getattr(layer_plan.layer, "layer_number", None) != index + 1:
                raise DeepTraceError("overlap plan layer identity differs from runtime model")
            for slot in NODE_SLOTS:
                node = getattr(layer_plan, slot, None)
                if type(node) is not layer_node_type or node.name != slot or not node.delay_wgrad_compute:
                    raise DeepTraceError(f"unexpected overlap schedule node at layer {index}: {slot}")
                pending.append((node, (microbatch, index, slot)))
        if len({id(node) for node, _ in pending}) != len(pending) or any(node in self._nodes for node, _ in pending):
            raise DeepTraceError("overlap plans reuse an observed node")
        for node, tag in pending:
            self._nodes[node] = tag
        # Store scalar identity only; never retain a plan or its nodes.
        self._plans.append({"microbatch": microbatch, "layers": 48})

    def _wrap_combined_passes(self, module):
        original = module.combined_forward_backward_step
        call_signature = signature(original)
        layout = combined_pass_layout()

        def combined(*args, **kwargs):
            index = len(self._passes)
            label = combined_pass_label(index)
            if index >= len(layout) or self._active_pass is not None:
                self._failed.append(label)
                raise DeepTraceError("extra or nested combined overlap pass")
            values = call_signature.bind(*args, **kwargs).arguments
            expected = layout[index]
            has_forward = values.get("f_model") is not None
            has_backward = values.get("b_model") is not None
            if (has_forward != (expected["forward_microbatch"] is not None)
                    or has_backward != (expected["backward_microbatch"] is not None)
                    or values.get("num_microbatches") != 4
                    or values.get("current_microbatch") != expected["forward_microbatch"]
                    or len(self._plans) != min(index, 4)):
                self._failed.append(label)
                raise DeepTraceError("combined overlap pass differs from native GA4 schedule")
            # Only scalars survive the call. Do not retain output tensors,
            # backward inputs or a schedule plan in the journal.
            del values
            record = dict(expected, returned=False)
            self._passes.append(record)
            self._active_pass = index
            self.counts[label] += 1
            try:
                with self.torch.profiler.record_function(label):
                    result = original(*args, **kwargs)
                    if len(self._plans) != min(index + 1, 4):
                        raise DeepTraceError("combined overlap pass did not build exactly its forward plan")
            except BaseException:
                self._failed.append(label)
                raise
            else:
                record["returned"] = True
                self._completed[label] += 1
                return result
            finally:
                self._active_pass = None

        self._set(module, "combined_forward_backward_step", combined)

    def attach(self):
        if self._attached or self._used:
            raise DeepTraceError("overlap labelers are single-step and cannot be attached twice")
        from megatron.core.models.common.utils import TransformerLayerNode
        from megatron.core.pipeline_parallel import combined_1f1b
        from megatron.core.pipeline_parallel.utils import ScheduleNode
        from megatron.core.utils import get_attr_wrapped_model, get_model_config

        if len(self.model) != 1:
            raise DeepTraceError("overlap observation requires one PP1 model chunk")
        config = get_model_config(self.model[0])
        if (any(getattr(config, key, None) is not value for key, value in SELECTED_FLAGS.items())
                or getattr(config, "pipeline_model_parallel_size", None) != 1
                or getattr(config, "cuda_graph_impl", None) != "none"):
            raise DeepTraceError("resolved model differs from selected native overlap configuration")
        model = get_attr_wrapped_model(self.model[0], "build_schedule_plan", return_model_obj=True)
        original = model.build_schedule_plan
        if not callable(original):
            raise DeepTraceError("model has no native schedule-plan builder")
        self._used = True
        try:
            self._inner.attach()
            self._wrap_node_method(ScheduleNode, "forward", "FWD")
            self._wrap_node_method(ScheduleNode, "backward", "BWD")
            self._wrap_node_method(TransformerLayerNode, "backward_dw", "WGRAD")
            self._wrap_combined_passes(combined_1f1b)

            def build(_model, *args, **kwargs):
                self.counts[PLAN_BUILD] += 1
                try:
                    with self.torch.profiler.record_function(PLAN_BUILD):
                        plan = original(*args, **kwargs)
                        self._register_plan(plan, ScheduleNode, TransformerLayerNode)
                except BaseException:
                    self._failed.append(PLAN_BUILD)
                    raise
                self._completed[PLAN_BUILD] += 1
                return plan

            self._set(model, "build_schedule_plan", MethodType(build, model))
            self._attached = True
        except BaseException:
            self.detach()
            raise

    def detach(self):
        try:
            while self._restore:
                owner, name, had_value, old = self._restore.pop()
                if had_value:
                    setattr(owner, name, old)
                else:
                    delattr(owner, name)
        finally:
            self._inner.detach()
            self._nodes.clear()
            self._attached = False

    def required_phases(self):
        return {direction: tuple(phase for phase in phases
                                 if direction != "FWD" or phase not in DORMANT_FORWARD_PHASES)
                for direction, phases in self._inner._required_phases().items()}

    def expected_counts(self, accumulation):
        if type(accumulation) is not int or accumulation != 4:
            raise DeepTraceError("overlap observation requires gradient accumulation four")
        expected = Counter({PLAN_BUILD: accumulation,
                            "AMPA/TRAIN/gradient_finalize": 1,
                            "AMPA/TRAIN/optimizer_step": 1})
        for index in range(accumulation + 1):
            expected[combined_pass_label(index)] = 1
        for direction, phases in self.required_phases().items():
            for layer in range(48):
                for phase in phases:
                    expected[f"AMPA/{direction}/layer_{layer:02d}/{phase}"] = accumulation
        for microbatch in range(accumulation):
            for slot in ("pre_process", "post_process"):
                for direction in ("FWD", "BWD"):
                    expected[node_label(microbatch, direction, slot)] = 1
            for layer in range(48):
                for slot in NODE_SLOTS:
                    for direction in ("FWD", "BWD"):
                        expected[node_label(microbatch, direction, slot, layer)] = 1
                for slot in WGRAD_SLOTS:
                    expected[node_label(microbatch, "WGRAD", slot, layer)] = 1
        return expected

    def validate_counts(self, accumulation):
        expected = self.expected_counts(accumulation)
        if (not self._used or self.layers != list(range(48)) or len(self._plans) != accumulation
                or self._failed or self._active_pass is not None
                or self._passes != [dict(row, returned=True) for row in combined_pass_layout()]):
            raise DeepTraceError("incomplete or failed overlap schedule lifecycle")
        if self.counts != expected:
            differences = {key: (expected.get(key, 0), self.counts.get(key, 0))
                           for key in expected.keys() | self.counts.keys()
                           if expected.get(key, 0) != self.counts.get(key, 0)}
            raise DeepTraceError(f"overlap phase counts differ: {str(differences)[:2000]}")
        completed = Counter({key: count for key, count in expected.items() if key.startswith("AMPA/OVERLAP/")})
        if self._completed != completed:
            raise DeepTraceError("overlap node invocation did not return successfully")
        pairs = self.pair_records()
        if self.communication_backend == "uccl_ep":
            grid = Counter((pair.get("layer"), pair.get("family")) for pair in pairs)
            required = Counter({(layer, family): accumulation for layer in range(48)
                                for family in ("dispatch", "combine")})
            if (grid != required or [pair.get("call_id") for pair in pairs] != list(range(len(pairs)))
                    or any(pair.get("forward_returned") is not True or pair.get("backward_returned") is not True
                           for pair in pairs)):
                raise DeepTraceError("incomplete UCCL overlap forward/backward journal")

    def pair_records(self):
        return self._inner.pair_records() if self.communication_backend == "uccl_ep" else []

    def combined_pass_records(self):
        return [dict(row) for row in self._passes]

    def contract_record(self, accumulation=4):
        return {"contract": CONTRACT, "communication_backend": self.communication_backend,
                "gradient_accumulation": accumulation, "layers": list(range(48)),
                "selected_flags": dict(SELECTED_FLAGS), "expected_counts": dict(self.expected_counts(accumulation)),
                "required_legacy_phases": self.required_phases(),
                "combined_pass_layout": combined_pass_layout(),
                "absent_ordinary_scopes": ["microbatch_forward", "microbatch_backward", *sorted(DORMANT_FORWARD_PHASES)],
                "claim_limit": "Host node scopes and overlapping device work are not additive optimizer-step time."}
