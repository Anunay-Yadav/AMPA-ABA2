"""Noncanonical native-C0 mock-data optimizer sanity evidence.

This module deliberately observes Bridge's public callback lifecycle rather
than wrapping an optimizer or modifying Megatron.  It is only instantiated for
the one-step mock-data bring-up path; canonical indexed profiling never reads
or promotes this record.
"""

from __future__ import annotations

import hashlib
import json
import math
import socket
from pathlib import Path
from typing import Any, Mapping, Sequence


_EXPECTED_PARALLELISM = {"tp": 1, "pp": 1, "cp": 1, "ep": 8, "etp": 1, "dense_dp": 8}
_EXPECTED_RUNTIME_TOPOLOGY = {"world_size": 8, **_EXPECTED_PARALLELISM}


class NativeC0MockSanityError(RuntimeError):
    """Raised when the limited native C0 smoke contract is not satisfied."""


def build_native_c0_mock_sanity_record(
    *,
    c0_identity_manifest_sha256: str,
    hf_model_id: str,
    hf_revision: str,
    rank_records: Sequence[Mapping[str, Any]],
    expected_world_size: int = 8,
) -> dict[str, object]:
    """Build a fail-closed summary from post-optimizer callback observations.

    A successful record establishes one finite, non-skipped Bridge optimizer
    step with mock tokens.  A sampled parameter digest may show that one local
    model-parameter sample changed, but cannot establish full-model mutation
    or equivalence to Hugging Face.
    """
    ordered = sorted((dict(record) for record in rank_records), key=lambda record: int(record["rank"]))
    ranks = [int(record["rank"]) for record in ordered]
    expected_ranks = list(range(expected_world_size))
    complete = ranks == expected_ranks and all(len(record.get("steps", [])) == 1 for record in ordered)
    runtime_topology_passed = complete and all(
        record.get("runtime_topology") == _EXPECTED_RUNTIME_TOPOLOGY for record in ordered
    )
    steps = [record["steps"][0] for record in ordered if len(record.get("steps", [])) == 1]
    skipped_or_failed = sum(1 for step in steps if step.get("skipped_iter") is not False)
    finite_loss = complete and all(step.get("loss_is_finite") is True for step in steps)
    finite_grad_norm = complete and all(step.get("grad_norm_is_finite") is True for step in steps)
    post_optimizer_callback_observed = complete and all(
        step.get("callback_after_wrapped_train_step") is True for step in steps
    )
    optimizer_step_reported_success = post_optimizer_callback_observed and skipped_or_failed == 0
    samples = [step.get("parameter_sample") for step in steps]
    sample_captured_on_all_ranks = complete and all(
        isinstance(sample, Mapping)
        and isinstance(sample.get("pre"), Mapping)
        and isinstance(sample.get("post"), Mapping)
        and sample["pre"].get("finite") is True
        and sample["post"].get("finite") is True
        and sample["pre"].get("parameter") == sample["post"].get("parameter")
        for sample in samples
    )
    sampled_parameter_change_observed = bool(
        sample_captured_on_all_ranks and any(sample.get("changed") is True for sample in samples)
    )
    sampled_parameter_change_observed_on_all_ranks = bool(
        sample_captured_on_all_ranks and all(sample.get("changed") is True for sample in samples)
    )
    status = "native_mock_optimizer_sanity_passed" if (
        runtime_topology_passed and optimizer_step_reported_success and finite_loss
        and finite_grad_norm and sample_captured_on_all_ranks
    ) else "native_mock_optimizer_sanity_failed"

    return {
        "schema_version": 1,
        "kind": "ampa_native_c0_mock_optimizer_sanity",
        "c0_identity_manifest_sha256": c0_identity_manifest_sha256,
        "hf_model": hf_model_id,
        "hf_revision": hf_revision,
        "claim_scope": {
            "kind": "native_bridge_c0_measurement",
            "data_mode": "mock",
            "hf_numerical_equivalence_claimed": False,
            "canonical_profile_data": False,
            "eligible_for_profile": False,
        },
        "trainer": {
            "kind": "bridge_pretrain",
            "callback_event": "on_train_step_end",
            "callback_event_contract": "Bridge fires this event after wrapped_train_step returns.",
        },
        "topology": _EXPECTED_RUNTIME_TOPOLOGY,
        "checks": {
            "complete_optimizer_steps": 1 if complete else 0,
            "skipped_or_failed_optimizer_steps": skipped_or_failed + (0 if complete else 1),
            "runtime_topology_passed": runtime_topology_passed,
            "finite_loss": finite_loss,
            "finite_grad_norm": finite_grad_norm,
            "post_optimizer_callback_observed": post_optimizer_callback_observed,
            "optimizer_step_reported_success": optimizer_step_reported_success,
            "parameter_sample_captured_on_all_ranks": sample_captured_on_all_ranks,
            "sampled_parameter_change_observed": sampled_parameter_change_observed,
            "sampled_parameter_change_observed_on_all_ranks": sampled_parameter_change_observed_on_all_ranks,
            "parameter_update_observed": None,
        },
        "per_rank": ordered,
        "claim_limits": [
            "The callback observes Bridge after wrapped_train_step; it does not wrap or alter optimizer.step().",
            "A non-skipped step reports Bridge optimizer success, but this record does not claim a whole-model parameter update.",
            "A changed sampled local parameter digest is supporting evidence only; an unchanged sample does not prove no parameter changed.",
            "Mock generated tokens are not realistic-mixture data and this record cannot make a profiling run eligible.",
            "No Hugging Face numerical, router, logits, or loss equivalence is claimed.",
        ],
        "lifecycle": {
            "status": status,
            "eligible_for_profile": False,
        },
    }


def make_native_c0_mock_sanity_callback(
    *, run_dir: Path, c0_identity_manifest_sha256: str, hf_model_id: str, hf_revision: str
) -> Any:
    """Create a Bridge callback without importing Bridge during CPU-only record tests."""
    from megatron.bridge.training.callbacks import Callback

    class NativeC0MockSanityCallback(Callback):
        """Observe one standard Bridge step after the optimizer has returned."""

        def __init__(self) -> None:
            self._rank: int | None = None
            self._world_size: int | None = None
            self._runtime_topology: dict[str, int] | None = None
            self._pre_step_parameter_sample: dict[str, object] | None = None
            self._steps: list[dict[str, object]] = []

        def on_train_start(self, context: Any) -> None:
            import torch.distributed as dist

            if not dist.is_initialized():
                raise NativeC0MockSanityError("native C0 mock sanity requires an initialized process group")
            self._rank = int(dist.get_rank())
            self._world_size = int(dist.get_world_size())
            if self._world_size != 8:
                raise NativeC0MockSanityError(f"native C0 mock sanity requires world size 8, got {self._world_size}")
            self._runtime_topology = _validate_runtime_baseline(context)

        def on_train_step_start(self, context: Any) -> None:
            if self._pre_step_parameter_sample is not None:
                raise NativeC0MockSanityError("received a new train step before closing the previous sample")
            self._pre_step_parameter_sample = _sample_trainable_parameter(context.model)

        def on_train_step_end(self, context: Any) -> None:
            if self._pre_step_parameter_sample is None:
                raise NativeC0MockSanityError("train-step end arrived without a pre-step parameter sample")
            post_sample = _sample_trainable_parameter(context.model)
            pre_sample = self._pre_step_parameter_sample
            self._pre_step_parameter_sample = None
            changed = (
                post_sample is not None
                and pre_sample["parameter"] == post_sample["parameter"]
                and pre_sample["digest_sha256"] != post_sample["digest_sha256"]
            )
            self._steps.append(
                {
                    "step_index_before_increment": int(context.state.train_state.step),
                    "callback_after_wrapped_train_step": True,
                    "skipped_iter": context.skipped_iter,
                    "loss_is_finite": _loss_is_finite(context.loss_dict),
                    "grad_norm": _finite_float(context.grad_norm),
                    "grad_norm_is_finite": _is_finite_float(context.grad_norm),
                    "parameter_sample": {
                        "pre": pre_sample,
                        "post": post_sample,
                        "changed": changed,
                    },
                }
            )

        def on_train_end(self, context: Any) -> None:
            import torch.distributed as dist

            if self._rank is None or self._world_size is None:
                raise NativeC0MockSanityError("train ended before native C0 mock sanity initialized")
            local_record = {
                "rank": self._rank,
                "hostname": socket.gethostname(),
                "slurm_node_id": _environment("SLURM_NODEID"),
                "slurm_local_id": _environment("SLURM_LOCALID"),
                "runtime_topology": self._runtime_topology,
                "steps": self._steps,
            }
            gathered: list[dict[str, object] | None] = [None] * self._world_size
            dist.all_gather_object(gathered, local_record)
            if any(record is None for record in gathered):
                raise NativeC0MockSanityError("native C0 mock sanity rank gather returned an empty record")
            if self._rank != 0:
                return
            record = build_native_c0_mock_sanity_record(
                c0_identity_manifest_sha256=c0_identity_manifest_sha256,
                hf_model_id=hf_model_id,
                hf_revision=hf_revision,
                rank_records=[record for record in gathered if record is not None],
                expected_world_size=self._world_size,
            )
            destination = run_dir / "validation" / "native_c0_mock_optimizer_sanity.json"
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
            temporary.replace(destination)
            if record["lifecycle"]["status"] != "native_mock_optimizer_sanity_passed":
                raise NativeC0MockSanityError("native C0 mock optimizer sanity checks failed")

    return NativeC0MockSanityCallback()


def _validate_runtime_baseline(context: Any) -> dict[str, int]:
    from megatron.core import parallel_state

    model = context.state.cfg.model
    configured = {
        "tp": getattr(model, "tensor_model_parallel_size", None),
        "pp": getattr(model, "pipeline_model_parallel_size", None),
        "cp": getattr(model, "context_parallel_size", None),
        "ep": getattr(model, "expert_model_parallel_size", None),
        "etp": getattr(model, "expert_tensor_parallel_size", None),
        "dense_dp": 8,
    }
    if configured != _EXPECTED_PARALLELISM:
        raise NativeC0MockSanityError(f"native C0 mock sanity configured topology mismatch: {configured}")
    actual = {
        "tp": parallel_state.get_tensor_model_parallel_world_size(),
        "pp": parallel_state.get_pipeline_model_parallel_world_size(),
        "cp": parallel_state.get_context_parallel_world_size(),
        "ep": parallel_state.get_expert_model_parallel_world_size(),
        "etp": parallel_state.get_expert_tensor_parallel_world_size(),
        "dense_dp": parallel_state.get_data_parallel_world_size(),
    }
    if actual != _EXPECTED_PARALLELISM:
        raise NativeC0MockSanityError(f"native C0 mock sanity runtime topology mismatch: {actual}")
    if getattr(model, "moe_token_dispatcher_type", None) != "alltoall":
        raise NativeC0MockSanityError("native C0 mock sanity requires the standard AllToAll dispatcher")
    if getattr(model, "moe_enable_deepep", False) or getattr(model, "moe_flex_dispatcher_backend", None) is not None:
        raise NativeC0MockSanityError("native C0 mock sanity forbids DeepEP and Flex dispatch")
    return {"world_size": 8, **actual}


def _environment(name: str) -> str | None:
    import os

    return os.environ.get(name)


def _sample_trainable_parameter(model_chunks: Sequence[Any]) -> dict[str, object] | None:
    """Hash a tiny local parameter sample; this is evidence, not a full-state hash."""
    import torch

    for chunk_index, chunk in enumerate(model_chunks):
        for name, parameter in chunk.named_parameters():
            if not getattr(parameter, "requires_grad", False) or parameter.numel() == 0:
                continue
            sample = parameter.detach().reshape(-1)[:16].to(dtype=torch.float32).cpu().contiguous()
            return {
                "parameter": f"chunk{chunk_index}.{name}",
                "elements": int(sample.numel()),
                "finite": bool(torch.isfinite(sample).all().item()),
                "digest_sha256": hashlib.sha256(sample.numpy().tobytes()).hexdigest(),
            }
    return None


def _loss_is_finite(loss_dict: Mapping[str, Any] | None) -> bool | None:
    if not loss_dict:
        return None
    import torch

    checks = []
    for value in loss_dict.values():
        if isinstance(value, torch.Tensor):
            checks.append(bool(torch.isfinite(value.detach()).all().item()))
        elif isinstance(value, (int, float)):
            checks.append(math.isfinite(float(value)))
    return all(checks) if checks else None


def _is_finite_float(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _finite_float(value: Any) -> float | None:
    return float(value) if _is_finite_float(value) else None
