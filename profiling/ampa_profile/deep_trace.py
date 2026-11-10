"""Opt-in PyTorch profiler labels for the existing CP2 MoE training path.

Only a selected diagnostic step is wrapped. Each wrapper calls the original
method once and returns its result unchanged; no CUDA event or synchronization
is inserted. Record-function ranges identify CPU launches for offline CUDA
correlation and are not themselves GPU elapsed times.
"""

from __future__ import annotations

from collections import Counter
from contextvars import ContextVar
import json
from pathlib import Path
from types import MethodType
from typing import Any


_A2A_CONTEXT: ContextVar[dict[str, Any] | None] = ContextVar("ampa_a2a_context", default=None)
_ROUTER_LAYER: ContextVar[int | None] = ContextVar("ampa_router_layer", default=None)
_DISPATCH_STAGE: ContextVar[tuple[int, str] | None] = ContextVar("ampa_dispatch_stage", default=None)


class DeepTraceError(RuntimeError):
    """A diagnostic trace cannot be interpreted safely."""


class _MethodLabels:
    dispatcher_type = "MoEAlltoAllTokenDispatcher"

    def __init__(self, torch: Any, model: list[Any], optimizer: Any = None) -> None:
        self.torch = torch
        self.model = model
        self.optimizer = optimizer
        self.counts: Counter[str] = Counter()
        self.payload_bytes: Counter[str] = Counter()
        self.payload_dtypes: dict[str, str] = {}
        self._restore: list[tuple[Any, str, bool, Any]] = []
        self.layers: list[int] = []

    def _set(self, owner: Any, name: str, value: Any) -> None:
        own_attributes = vars(owner)
        self._restore.append((owner, name, name in own_attributes, own_attributes.get(name)))
        setattr(owner, name, value)

    def _label_method(self, owner: Any, name: str, label: str) -> None:
        original = getattr(owner, name)
        if not callable(original):
            raise DeepTraceError(f"{label}: {name} is not callable")

        def observed(_owner: Any, *args: Any, **kwargs: Any) -> Any:
            self.counts[label] += 1
            with self.torch.profiler.record_function(label):
                return original(*args, **kwargs)

        self._set(owner, name, MethodType(observed, owner))

    def _label_function(self, owner: Any, name: str, label: str) -> None:
        original = getattr(owner, name)
        if not callable(original):
            raise DeepTraceError(f"{label}: {name} is not callable")

        def observed(*args: Any, **kwargs: Any) -> Any:
            self.counts[label] += 1
            with self.torch.profiler.record_function(label):
                return original(*args, **kwargs)

        self._set(owner, name, observed)

    def _label_metadata_point(self, dispatcher: Any, layer: int) -> None:
        original = dispatcher._maybe_dtoh_and_synchronize
        points = {
            "before_permutation_1", "before_ep_alltoall",
            "before_permutation_2", "before_finish",
        }

        def observed(_owner: Any, point: str, *args: Any, **kwargs: Any) -> Any:
            if point not in points:
                raise DeepTraceError(f"unexpected dispatcher metadata point: {point}")
            label = f"AMPA/FWD/layer_{layer:02d}/metadata_{point}"
            self.counts[label] += 1
            with self.torch.profiler.record_function(label):
                return original(point, *args, **kwargs)

        self._set(dispatcher, "_maybe_dtoh_and_synchronize", MethodType(observed, dispatcher))

    def _label_context_method(
        self, owner: Any, name: str, label: str, variable: ContextVar[Any], context: Any
    ) -> None:
        original = getattr(owner, name)

        def observed(_owner: Any, *args: Any, **kwargs: Any) -> Any:
            token = variable.set(context)
            self.counts[label] += 1
            try:
                with self.torch.profiler.record_function(label):
                    return original(*args, **kwargs)
            finally:
                variable.reset(token)

        self._set(owner, name, MethodType(observed, owner))

    def _label_alias(
        self, module: Any, name: str, variable: ContextVar[Any], suffixes: dict[Any, str]
    ) -> None:
        original = getattr(module, name)

        def observed(*args: Any, **kwargs: Any) -> Any:
            context = variable.get()
            if context is None:
                return original(*args, **kwargs)
            if context not in suffixes:
                raise DeepTraceError(f"{name}: unexpected MoE diagnostic context {context}")
            layer = context if isinstance(context, int) else context[0]
            label = f"AMPA/FWD/layer_{layer:02d}/{suffixes[context]}"
            self.counts[label] += 1
            with self.torch.profiler.record_function(label):
                return original(*args, **kwargs)

        self._set(module, name, observed)

    def _label_a2a_method(self, dispatcher: Any, name: str, layer: int, expected: int) -> None:
        original = getattr(dispatcher, name)
        phase = "dispatch" if name == "token_dispatch" else "combine"
        label = f"AMPA/FWD/layer_{layer:02d}/{phase}_total"

        def observed(_owner: Any, *args: Any, **kwargs: Any) -> Any:
            context = {"layer": layer, "phase": phase, "calls": 0, "group": dispatcher.ep_group}
            token = _A2A_CONTEXT.set(context)
            self.counts[label] += 1
            try:
                with self.torch.profiler.record_function(label):
                    result = original(*args, **kwargs)
            finally:
                _A2A_CONTEXT.reset(token)
            if context["calls"] != expected:
                raise DeepTraceError(
                    f"layer {layer} {phase} made {context['calls']} EP AllToAll calls; expected {expected}"
                )
            return result

        self._set(dispatcher, name, MethodType(observed, dispatcher))

    def _attach_communication(self, dispatcher_module: Any, observed_a2a: Any) -> None:
        self._set(dispatcher_module, "all_to_all", observed_a2a)
        self._label_alias(dispatcher_module, "permute", _DISPATCH_STAGE,
                          {(layer, "dispatch_preprocess"): "first_permute" for layer in range(48)})
        self._label_alias(dispatcher_module, "sort_chunks_by_idxs", _DISPATCH_STAGE, {
            **{(layer, "dispatch_postprocess"): "local_expert_sort_core" for layer in range(48)},
            **{(layer, "combine_preprocess"): "inverse_local_sort_core" for layer in range(48)},
        })
        self._label_alias(dispatcher_module, "unpermute", _DISPATCH_STAGE,
                          {(layer, "combine_postprocess"): "final_unpermute_core" for layer in range(48)})

    def _attach_dispatcher(self, dispatcher: Any, layer: int) -> None:
        root = f"AMPA/FWD/layer_{layer:02d}"
        self._label_context_method(dispatcher, "dispatch_preprocess", f"{root}/dispatch_preprocess",
                                   _DISPATCH_STAGE, (layer, "dispatch_preprocess"))
        self._label_method(dispatcher, "preprocess", f"{root}/dispatch_metadata")
        self._label_metadata_point(dispatcher, layer)
        self._label_a2a_method(dispatcher, "token_dispatch", layer, expected=2)
        self._label_context_method(dispatcher, "dispatch_postprocess", f"{root}/local_expert_sort",
                                   _DISPATCH_STAGE, (layer, "dispatch_postprocess"))
        self._label_context_method(dispatcher, "combine_preprocess", f"{root}/inverse_local_sort",
                                   _DISPATCH_STAGE, (layer, "combine_preprocess"))
        self._label_a2a_method(dispatcher, "token_combine", layer, expected=1)
        self._label_context_method(dispatcher, "combine_postprocess", f"{root}/final_unpermute",
                                   _DISPATCH_STAGE, (layer, "combine_postprocess"))

    def attach(self) -> None:
        from megatron.core.transformer.moe.moe_layer import MoELayer
        from megatron.core.transformer.moe import token_dispatcher as dispatcher_module
        from megatron.core.transformer.moe import router as router_module
        from megatron.core.pipeline_parallel import schedules
        from megatron.core.utils import get_model_config

        original_a2a = dispatcher_module.all_to_all

        def observed_a2a(group: Any, *args: Any, **kwargs: Any) -> Any:
            context = _A2A_CONTEXT.get()
            if context is None:
                return original_a2a(group, *args, **kwargs)
            if group is not context["group"]:
                raise DeepTraceError("MoE payload AllToAll did not use its runtime EP group")
            ordinal = context["calls"]
            phase = context["phase"]
            if phase == "dispatch":
                if ordinal >= 2:
                    raise DeepTraceError("more than two dispatch payload AllToAll calls")
                payload = ("activations", "probabilities")[ordinal]
            else:
                if ordinal != 0:
                    raise DeepTraceError("more than one combine payload AllToAll call")
                payload = "expert_output"
            context["calls"] += 1
            label = f"AMPA/FWD/layer_{context['layer']:02d}/{phase}_{payload}_a2a"
            self.counts[label] += 1
            if not args or not hasattr(args[0], "numel"):
                raise DeepTraceError(f"{label}: missing input tensor for payload accounting")
            dtype = str(args[0].dtype)
            if label in self.payload_dtypes and self.payload_dtypes[label] != dtype:
                raise DeepTraceError(f"{label}: input dtype changed during the traced step")
            self.payload_dtypes[label] = dtype
            # Tensor metadata are host-resident; no CUDA transfer or synchronization.
            self.payload_bytes[label] += args[0].numel() * args[0].element_size()
            with self.torch.profiler.record_function(label):
                return original_a2a(group, *args, **kwargs)

        try:
            self._label_function(schedules, "forward_step", "AMPA/TRAIN/microbatch_forward")
            self._label_function(schedules, "backward_step", "AMPA/TRAIN/microbatch_backward")
            config = get_model_config(self.model[0])
            self._label_function(config, "finalize_model_grads_func", "AMPA/TRAIN/gradient_finalize")
            if self.optimizer is None:
                raise DeepTraceError("selected-step optimizer is unavailable")
            self._label_method(self.optimizer, "step", "AMPA/TRAIN/optimizer_step")
            self._attach_communication(dispatcher_module, observed_a2a)
            self._label_alias(router_module, "topk_routing_with_score_function", _ROUTER_LAYER,
                              {layer: "router_main_topk" for layer in range(48)})
            self._label_alias(router_module, "compute_routing_scores_for_aux_loss", _ROUTER_LAYER,
                              {layer: "router_aux_scores" for layer in range(48)})
            sites: dict[int, tuple[Any, Any]] = {}
            for chunk in self.model:
                for _, module in chunk.named_modules():
                    moe = getattr(module, "mlp", None)
                    if isinstance(moe, MoELayer):
                        layer_number = getattr(module, "layer_number", None)
                        if not isinstance(layer_number, int) or layer_number not in range(1, 49):
                            raise DeepTraceError(f"invalid MoE transformer layer number: {layer_number}")
                        layer = layer_number - 1
                        if layer in sites:
                            raise DeepTraceError(f"duplicate MoE transformer layer {layer}")
                        sites[layer] = (module, moe)
            if set(sites) != set(range(48)):
                raise DeepTraceError(f"expected 48 MoE layers, found {sorted(sites)}")

            for layer, (transformer, moe) in sorted(sites.items()):
                dispatcher = moe.token_dispatcher
                if type(dispatcher).__name__ != self.dispatcher_type:
                    raise DeepTraceError(f"layer {layer} requires {self.dispatcher_type}")
                if getattr(moe, "use_shared_expert", False) or getattr(moe, "moe_layer_recompute", False):
                    raise DeepTraceError(f"layer {layer} changes the reviewed MoE execution path")
                root = f"AMPA/FWD/layer_{layer:02d}"
                self._label_method(transformer, "_pre_mlp_layernorm_and_residual", f"{root}/pre_mlp_norm")
                self._label_method(moe, "forward", f"{root}/moe_total")
                self._label_method(moe, "route", f"{root}/router_total")
                self._label_method(moe.router, "gating", f"{root}/router_gate")
                self._label_context_method(moe.router, "routing", f"{root}/router_routing",
                                           _ROUTER_LAYER, layer)
                self._label_method(moe.router, "_apply_aux_loss", f"{root}/router_aux_loss")
                self._attach_dispatcher(dispatcher, layer)
                self._label_method(moe.experts, "forward", f"{root}/expert_total")
                self._label_method(moe.experts.linear_fc1, "forward", f"{root}/expert_fc1")
                self._label_method(moe.experts.linear_fc2, "forward", f"{root}/expert_fc2")
                self._label_method(transformer, "_apply_mlp_bda_step", f"{root}/residual_bda")
                self.layers.append(layer)
        except BaseException:
            self.detach()
            raise

    def detach(self) -> None:
        while self._restore:
            owner, name, had_own_value, old_value = self._restore.pop()
            if had_own_value:
                setattr(owner, name, old_value)
            else:
                delattr(owner, name)

    def validate_counts(self, accumulation: int) -> None:
        train_required = {
            "AMPA/TRAIN/microbatch_forward": accumulation,
            "AMPA/TRAIN/microbatch_backward": accumulation,
            "AMPA/TRAIN/gradient_finalize": 1,
            "AMPA/TRAIN/optimizer_step": 1,
        }
        train_mismatches = {
            label: self.counts.get(label, 0) for label, expected in train_required.items()
            if self.counts.get(label, 0) != expected
        }
        if train_mismatches:
            raise DeepTraceError(f"selected-step training phase call counts differ: {train_mismatches}")
        mismatches = {
            f"{direction}/layer_{layer:02d}/{phase}": self.counts.get(f"AMPA/{direction}/layer_{layer:02d}/{phase}", 0)
            for direction, phases in self._required_phases().items()
            for layer in self.layers for phase in phases
            if self.counts.get(f"AMPA/{direction}/layer_{layer:02d}/{phase}", 0) != accumulation
        }
        if mismatches:
            raise DeepTraceError(f"MoE diagnostic call counts differ from GA={accumulation}: {mismatches}")

    def _required_phases(self) -> dict[str, tuple[str, ...]]:
        return {"FWD": (
            "pre_mlp_norm", "moe_total", "router_total", "router_gate", "router_routing",
            "router_main_topk", "router_aux_scores", "router_aux_loss",
            "dispatch_preprocess", "dispatch_metadata", "first_permute",
            "metadata_before_permutation_1", "metadata_before_ep_alltoall",
            "metadata_before_permutation_2", "metadata_before_finish", "dispatch_total",
            "dispatch_activations_a2a", "dispatch_probabilities_a2a", "local_expert_sort",
            "local_expert_sort_core", "expert_total", "expert_fc1", "expert_fc2",
            "inverse_local_sort", "inverse_local_sort_core", "combine_total",
            "combine_expert_output_a2a", "final_unpermute", "final_unpermute_core", "residual_bda",
        )}


def make_deep_trace_callback(
    *, run_dir: Path, start_step: int, capture_steps: int, selected_ranks: tuple[int, ...],
    accumulation: int = 16,
    per_step_export: bool = False,
    expected_world_size: int = 8,
    communication_backend: str = "alltoall",
) -> Any:
    """Build Bridge callback without importing Bridge or CUDA on the login node."""
    if communication_backend not in ("alltoall", "uccl_ep"):
        raise DeepTraceError("unsupported trace communication backend")
    schema_version = 1 if communication_backend == "alltoall" else 2
    record_kind = ("ampa_pytorch_moe_deep_trace_calls" if communication_backend == "alltoall"
                   else "ampa_pytorch_uccl_deep_trace_calls")
    from megatron.bridge.training.callbacks import Callback

    class DeepTraceCallback(Callback):
        def __init__(self) -> None:
            self.rank: int | None = None
            self.labeler: _MethodLabels | None = None
            self.step_scope: Any = None
            self.records: list[dict[str, Any]] = []
            self.profiler: Any = None
            self.aborted = False

        def _record_uccl_abort(self, step: int, error: BaseException,
                               cleanup_error: BaseException | None = None) -> None:
            # Keep self.profiler alive until the UCCL entrypoint exits. Do not
            # wait for peer-dependent kernels, stop, or export after failure.
            self.aborted = True
            try:
                destination = run_dir / "raw" / f"uccl_trace_aborted_rank-{self.rank}_step-{step}.json"
                destination.parent.mkdir(parents=True, exist_ok=True)
                with destination.open("x", encoding="utf-8") as handle:
                    json.dump({"schema_version": 2, "kind": "ampa_uccl_trace_aborted", "valid": False,
                               "rank": self.rank, "step": step, "error": f"{type(error).__name__}: {error}",
                               "cleanup_error": (f"{type(cleanup_error).__name__}: {cleanup_error}"
                                                 if cleanup_error is not None else None),
                               "wrappers_restored": cleanup_error is None,
                               "trace_exported": False, "cuda_cleanup_attempted": False}, handle)
            except OSError:
                pass  # Preserve the original backend exception.

        def on_train_start(self, context: Any) -> None:
            import torch.distributed as dist

            self.rank = dist.get_rank()
            if dist.get_world_size() != expected_world_size:
                raise DeepTraceError(f"deep trace requires the reviewed {expected_world_size}-rank workload")

        def on_train_step_start(self, context: Any) -> None:
            if self.aborted:
                raise DeepTraceError("an aborted UCCL trace cannot resume in this process")
            step = int(context.state.train_state.step)
            if self.rank not in selected_ranks or not start_step <= step < start_step + capture_steps:
                return
            if self.labeler is not None:
                raise DeepTraceError("deep trace step scopes overlap")
            import torch

            profiler_started = False
            try:
                if per_step_export:
                    self.profiler = torch.profiler.profile(record_shapes=False, with_stack=False)
                    self.profiler.start()
                    profiler_started = True
                if communication_backend == "uccl_ep":
                    from .uccl_trace import UcclMethodLabels
                    self.labeler = UcclMethodLabels(torch, context.model, context.optimizer)
                else:
                    self.labeler = _MethodLabels(torch, context.model, context.optimizer)
                self.labeler.attach()
                self.step_scope = torch.profiler.record_function(f"AMPA/STEP/{step}")
                self.step_scope.__enter__()
            except BaseException as error:
                if communication_backend == "uccl_ep":
                    cleanup_error = None
                    try:
                        if self.labeler is not None:
                            self.labeler.detach()
                    except BaseException as failure:
                        cleanup_error = failure
                    self.labeler = None
                    self._record_uccl_abort(step, error, cleanup_error)
                else:
                    if self.labeler is not None:
                        self.labeler.detach()
                    self.labeler = None
                    if self.profiler is not None and profiler_started:
                        self.profiler.stop()
                        self.profiler = None
                raise

        def on_train_step_end(self, context: Any) -> None:
            if self.labeler is None:
                return
            step = int(context.state.train_state.step)
            active_error = None
            cleanup_error = None
            try:
                self.step_scope.__exit__(None, None, None)
                self.labeler.validate_counts(accumulation)
                if context.skipped_iter:
                    raise DeepTraceError(f"deep-traced optimizer step {step} was skipped")
                self.records.append({
                    "step": step, "rank": self.rank, "layers": self.labeler.layers,
                    "counts": dict(sorted(self.labeler.counts.items())),
                    "payload_bytes": dict(sorted(self.labeler.payload_bytes.items())),
                    "payload_dtypes": dict(sorted(self.labeler.payload_dtypes.items())),
                    "skipped_iter": False,
                })
                if communication_backend == "uccl_ep":
                    self.records[-1]["uccl_autograd_pairs"] = self.labeler.pair_records()
            except BaseException as error:
                active_error = error
                raise
            finally:
                try:
                    self.labeler.detach()
                except BaseException as error:
                    cleanup_error = error
                    if active_error is None:
                        active_error = error
                        raise
                    if communication_backend != "uccl_ep":
                        raise
                finally:
                    self.labeler = None
                    self.step_scope = None
                    if communication_backend == "uccl_ep" and active_error is not None:
                        self._record_uccl_abort(step, active_error, cleanup_error)
                    elif self.profiler is not None:
                        import time

                        stop_start = time.monotonic()
                        self.profiler.stop()
                        if self.records and self.records[-1]["step"] == step:
                            self.records[-1]["profiler_stop_seconds"] = time.monotonic() - stop_start
            if per_step_export:
                import time

                trace_dir = run_dir / "traces" / "torch_profile" / f"step-{step:05d}"
                raw_dir = run_dir / "raw" / "deep_trace" / f"step-{step:05d}"
                trace_dir.mkdir(parents=True, exist_ok=True)
                raw_dir.mkdir(parents=True, exist_ok=True)
                trace_path = trace_dir / f"rank-{self.rank}.json.gz"
                if trace_path.exists():
                    raise DeepTraceError(f"refusing to replace trace: {trace_path}")
                export_start = time.monotonic()
                self.profiler.export_chrome_trace(str(trace_path))
                self.records[-1]["export_seconds"] = time.monotonic() - export_start
                self.profiler = None
                (raw_dir / f"rank-{self.rank}.json").write_text(json.dumps({
                    "schema_version": schema_version, "kind": record_kind,
                    "rank": self.rank, "steps": [self.records[-1]],
                }, indent=2, sort_keys=True), encoding="utf-8")

        def on_train_end(self, context: Any) -> None:
            del context
            if self.aborted:
                raise DeepTraceError("an aborted UCCL trace cannot be accepted")
            if self.labeler is not None or self.profiler is not None:
                raise DeepTraceError("deep trace ended with attached method wrappers")
            if self.rank not in selected_ranks:
                return
            if len(self.records) != capture_steps:
                raise DeepTraceError(f"expected {capture_steps} traced steps, got {len(self.records)}")
            destination = run_dir / "raw" / f"deep_trace_rank-{self.rank}.json"
            destination.write_text(json.dumps({
                "schema_version": schema_version, "kind": record_kind,
                "rank": self.rank, "steps": self.records,
            }, indent=2, sort_keys=True), encoding="utf-8")

    return DeepTraceCallback()
