"""Opt-in CPU batch fingerprints for exact same-token run comparisons.

The existing Bridge data iterator produces CPU token tensors. Hashing that
tensor before Bridge transfers it to the GPU observes the actual consumed
batch without changing the iterator, token IDs, or model input. Enable this
only in matched diagnostic legs; its CPU cost is included in both legs.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class SampleFingerprintError(RuntimeError):
    """The requested batch identity could not be observed without perturbation."""


def make_sample_fingerprint_callback(
    *, run_dir: Path, expected_steps: int = 25, accumulation: int = 16,
    expected_world_size: int = 8,
) -> Any:
    from megatron.bridge.training.callbacks import Callback

    class FingerprintCallback(Callback):
        def __init__(self) -> None:
            self.rank: int | None = None
            self._module: Any = None
            self._original: Any = None
            self._step: int | None = None
            self.records: list[dict] = []

        def on_train_start(self, context: Any) -> None:
            del context
            import torch.distributed as dist

            self.rank = dist.get_rank()
            if dist.get_world_size() != expected_world_size:
                raise SampleFingerprintError(
                    f"batch fingerprint requires the reviewed {expected_world_size}-rank workload"
                )

        def on_train_step_start(self, context: Any) -> None:
            step = int(context.state.train_state.step)
            if step != len(self.records) or self._original is not None:
                raise SampleFingerprintError("unexpected or overlapping batch-fingerprint step")
            from megatron.bridge.training import gpt_step

            original = gpt_step.get_batch_from_iterator
            record: dict[str, Any] = {"step": step, "batches": []}

            class ObservedIterator:
                def __init__(self, source: Any) -> None:
                    self.source = source

                def __iter__(self) -> Any:
                    return self

                def __next__(self) -> Any:
                    batch = next(self.source)
                    tokens = batch.get("tokens") if isinstance(batch, dict) else None
                    if tokens is None or tokens.device.type != "cpu":
                        raise SampleFingerprintError("Bridge yielded no CPU token tensor")
                    # A CPU view is sufficient; no CUDA transfer, CUDA event, or
                    # synchronization is introduced by this observation.
                    fields = {}
                    for name, value in sorted(batch.items()):
                        if value is None or not hasattr(value, "device"):
                            continue
                        if value.device.type != "cpu":
                            raise SampleFingerprintError(f"Bridge yielded a non-CPU {name} before transfer")
                        host = value.detach().contiguous().numpy()
                        fields[name] = {
                            "sha256": hashlib.sha256(memoryview(host).cast("B")).hexdigest(),
                            "shape": list(value.shape), "dtype": str(value.dtype),
                        }
                    record["batches"].append(fields)
                    return batch

            def observed(data_iterator: Any, *args: Any, **kwargs: Any) -> Any:
                return original(ObservedIterator(data_iterator), *args, **kwargs)

            self._module = gpt_step
            self._original = original
            self._step = step
            self.records.append(record)
            gpt_step.get_batch_from_iterator = observed

        def on_train_step_end(self, context: Any) -> None:
            if self._original is None:
                raise SampleFingerprintError("batch fingerprint step ended without an attached observer")
            self._module.get_batch_from_iterator = self._original
            self._module = None
            self._original = None
            step = int(context.state.train_state.step)
            if step != self._step or len(self.records[-1]["batches"]) != accumulation or context.skipped_iter:
                raise SampleFingerprintError(
                    f"step {step}: expected {accumulation} real token batches and no skipped update"
                )
            self._step = None

        def on_train_end(self, context: Any) -> None:
            del context
            if self._original is not None or len(self.records) != expected_steps:
                raise SampleFingerprintError("incomplete batch-fingerprint run")
            path = run_dir / "raw" / f"sample_fingerprint_rank-{self.rank}.json"
            path.write_text(json.dumps({
                "schema_version": 1, "kind": "consumed_cpu_token_batch_sha256",
                "rank": self.rank, "steps": self.records,
            }, indent=2, sort_keys=True), encoding="utf-8")

    return FingerprintCallback()
