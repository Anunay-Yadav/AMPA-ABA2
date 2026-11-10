"""Bridge callback that attaches and flushes the AMPA routing observer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from megatron.bridge.training.callbacks import Callback, CallbackContext

from .observer import ProfileInvariantError, RoutingObserver


class AMPAProfilingCallback(Callback):
    """Lifecycle glue only; routing aggregation remains in the dispatcher wrapper."""

    def __init__(
        self, *, run_dir: Path, profile_start_step: int, profile_stop_step: int,
        topology_variant: str, expected_calls_per_layer: int, expected_tokens_per_call: int,
    ) -> None:
        self.run_dir = run_dir
        if topology_variant not in {"cp1_ep8", "cp2_ep8"}:
            raise ValueError(f"unsupported topology variant: {topology_variant}")
        self.topology_variant = topology_variant
        self.observer = RoutingObserver(
            run_dir=run_dir,
            profile_start_step=profile_start_step,
            profile_stop_step=profile_stop_step,
            expected_calls_per_layer=expected_calls_per_layer,
            expected_tokens_per_call=expected_tokens_per_call,
        )
        self.observer_attached = False

    def on_train_start(self, context: CallbackContext) -> None:
        """Validate target topology before installing transparent instance wrappers."""
        self._validate_target_topology(context)
        self.observer.attach(context.model)
        # The common indexed-run provenance callback runs next and records the
        # actual attachment state for both profile-on and profile-off runs.
        self.observer_attached = True

    def on_train_step_start(self, context: CallbackContext) -> None:
        self.observer.start_step(int(context.state.train_state.step))

    def on_train_step_end(self, context: CallbackContext) -> None:
        self.observer.finish_step(
            skipped_iter=context.skipped_iter,
            grad_norm=context.grad_norm,
            loss_dict=context.loss_dict,
        )

    def on_train_end(self, context: CallbackContext) -> None:
        del context
        self.observer.write_raw()

    def _validate_target_topology(self, context: CallbackContext) -> None:
        import torch.distributed as dist

        config = context.state.cfg
        model = config.model
        expected = {
            "tensor_model_parallel_size": 1,
            "pipeline_model_parallel_size": 1,
            "context_parallel_size": 2 if self.topology_variant == "cp2_ep8" else 1,
            "expert_model_parallel_size": 8,
        }
        mismatches = {
            field: (getattr(model, field, None), value)
            for field, value in expected.items()
            if getattr(model, field, None) != value
        }
        if mismatches:
            raise ProfileInvariantError(f"AMPA baseline topology mismatch: {mismatches}")
        if dist.get_world_size() != 8:
            raise ProfileInvariantError(
                f"AMPA baseline requires world size 8, got {dist.get_world_size()}"
            )
        if getattr(model, "moe_token_dispatcher_type", None) != "alltoall":
            raise ProfileInvariantError("AMPA baseline requires moe_token_dispatcher_type='alltoall'")
        if getattr(model, "moe_flex_dispatcher_backend", None) is not None:
            raise ProfileInvariantError("AMPA baseline forbids a Flex dispatcher backend")
        if getattr(model, "moe_router_force_load_balancing", False):
            raise ProfileInvariantError("AMPA baseline forbids forced router load balancing")
        disabled_controls = (
            "overlap_moe_expert_parallel_comm",
            "delay_wgrad_compute",
            "overlap_dispatch_backward_with_experts_wgrad",
            "ep_overlap_early_attn_memory_release",
            "moe_shared_expert_overlap",
            "moe_shortcut_parallel",
        )
        enabled_controls = {
            field: getattr(model, field, None) for field in disabled_controls if getattr(model, field, None) is not False
        }
        if enabled_controls:
            raise ProfileInvariantError(
                f"AMPA baseline requires standard non-overlapped MoE controls: {enabled_controls}"
            )
        if config.comm_overlap is not None:
            raise ProfileInvariantError("AMPA baseline requires Bridge comm_overlap=None")
