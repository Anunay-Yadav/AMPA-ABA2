"""Step capture for the native PP1 expert-parallel overlap schedule.

OFF controls retain runtime identity checks without attaching trace wrappers.
PyTorch and Nsight use the same node observation contract. Boundaries and
exports are recorded separately from optimizer-step device activity.
"""

from __future__ import annotations

from contextlib import contextmanager
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import time
from types import SimpleNamespace

from .deep_trace import DeepTraceError
from .overlap_trace import CONTRACT, SELECTED_FLAGS, OverlapMethodLabels


def require(value, message):
    if not value:
        raise DeepTraceError(message)


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


class NvtxProxy:
    """Replace only the observer's record_function with NVTX ranges."""

    def __init__(self, torch):
        @contextmanager
        def scope(label):
            torch.cuda.nvtx.range_push(label)
            try:
                yield
            finally:
                torch.cuda.nvtx.range_pop()
        self.profiler = SimpleNamespace(record_function=scope)


class _OverlapCapture:
    def __init__(self, args, torch, dist, label_type=OverlapMethodLabels):
        self.args, self.torch, self.dist, self.label_type = args, torch, dist, label_type
        self.run = Path(args.run_dir)
        self.mode, self.steps = args.overlap_capture, args.completed_steps
        require(self.mode in ("off", "torch", "nsys") and self.steps in (2, 10),
                "unsupported overlap capture protocol")
        self.capture_steps = (list(range(self.steps)) if self.mode == "torch" else
                              (list(range(5, 10)) if self.steps == 10 else [0, 1])
                              if self.mode == "nsys" else [])
        self.rank = None
        self.labeler = self.scope = self.profiler = None
        self.ended, self.captured, self.boundaries = [], [], []
        self.active_step = None
        self.started = self.stopped = self.finished = self.aborted = False

    def on_train_start(self, context):
        from megatron.core.utils import get_model_config

        self.rank = self.dist.get_rank()
        require(self.dist.get_world_size() == 16, "overlap capture requires sixteen ranks")
        require(os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS") == "32",
                "set CUDA_DEVICE_MAX_CONNECTIONS=32 before CUDA initialization")
        require(len(context.model) == 1, "overlap capture requires one PP1 model chunk")
        ddp_config = getattr(context.model[0], "ddp_config", None)
        ddp_delay = getattr(ddp_config, "delay_wgrad_compute", None)
        require(ddp_delay is True,
                "actual DDP must defer gradient accumulation/reduction hooks until delayed weight gradients")
        config = get_model_config(context.model[0])
        flags = {name: getattr(config, name, None) for name in SELECTED_FLAGS}
        require(flags == SELECTED_FLAGS, "finalized model differs from selected overlap flags")
        geometry = {name: getattr(config, name, None) for name in (
            "pipeline_model_parallel_size", "tensor_model_parallel_size", "context_parallel_size",
            "expert_model_parallel_size", "expert_tensor_parallel_size", "num_layers",
            "cuda_graph_impl", "recompute_granularity", "moe_router_topk")}
        expected = dict(pipeline_model_parallel_size=1, tensor_model_parallel_size=1,
                        context_parallel_size=1, expert_model_parallel_size=16,
                        expert_tensor_parallel_size=1, num_layers=48, cuda_graph_impl="none",
                        recompute_granularity=None, moe_router_topk=8)
        require(geometry == expected, "finalized overlap topology or workload differs")
        local = {"selected_flags": flags, "geometry": geometry, "capture_mode": self.mode,
                 "completed_steps": self.steps, "communication_backend": self.args.communication_backend,
                 "communication_sms": self.args.communication_sms, "cuda_device_max_connections": "32",
                 "ddp_delay_wgrad_compute": ddp_delay}
        if self.args.communication_backend == "uccl_ep":
            from profiling.run_pretrain_uccl_budget import effective_configs, runtime_adapter
            fused = runtime_adapter()
            require(fused._buffer is None, "UCCL buffer allocated before startup checks")
            require(config.moe_flex_dispatcher_num_sms == self.args.communication_sms,
                    "resolved UCCL payload budget differs")
            local["effective_configs"] = effective_configs(fused, self.args.communication_sms)
            require(local["effective_configs"] == self.args.uccl_budget_setup,
                    "UCCL configuration changed after its official setter")
        peers = [None] * 16
        self.dist.all_gather_object(peers, local)
        require(all(peer == local for peer in peers), "ranks disagree on overlap execution")
        versions = {"torch": self.torch.__version__}
        for package in ("transformer-engine", "megatron-core", "megatron-bridge"):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = None
        write_new(self.run / "raw" / f"ep_overlap_rank-{self.rank}_startup.json", {
            "schema_version": 1, "contract": CONTRACT, "valid_overlap_startup": True,
            "rank": self.rank, "hostname": socket.gethostname(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "local_rank": os.environ.get("LOCAL_RANK"), "world_size": 16,
            "all_rank_configuration_agreement": True, "configuration_ranks_checked": 16,
            "versions": versions, **local})

    def _boundary(self, action, step):
        record = {"action": action, "step": step, "rank": self.rank}
        before = time.monotonic()
        self.torch.cuda.synchronize()
        record["cuda_synchronize_seconds"] = time.monotonic() - before
        before = time.monotonic()
        self.dist.barrier()
        record["pre_api_global_barrier_seconds"] = time.monotonic() - before
        before = time.monotonic()
        api = self.torch.cuda.cudart()
        self.torch.cuda.check_error(api.cudaProfilerStart() if action == "start" else api.cudaProfilerStop())
        record["profiler_api_seconds"] = time.monotonic() - before
        if action == "start":
            self.started = True
            before = time.monotonic()
            self.dist.barrier()
            record["post_start_global_barrier_seconds"] = time.monotonic() - before
        else:
            self.stopped = True
        self.boundaries.append(record)
        print(json.dumps({"ampa_overlap_nsight_boundary": record}), flush=True)

    def on_train_step_start(self, context):
        step = int(context.state.train_state.step)
        require(not self.aborted and self.active_step is None and step == len(self.ended)
                and step < self.steps, "unexpected overlap optimizer-step order")
        self.active_step = step
        if step not in self.capture_steps:
            return
        if self.mode == "nsys" and step == self.capture_steps[0]:
            self._boundary("start", step)
        proxy = self.torch
        if self.mode == "torch":
            self.profiler = self.torch.profiler.profile(record_shapes=False, with_stack=False)
            self.profiler.start()
        else:
            proxy = NvtxProxy(self.torch)
        self.labeler = self.label_type(proxy, context.model, context.optimizer,
                                      communication_backend=self.args.communication_backend)
        self.labeler.attach()
        self.scope = proxy.profiler.record_function(f"AMPA/STEP/{step}")
        self.scope.__enter__()

    def on_train_step_end(self, context):
        step = int(context.state.train_state.step)
        require(step == self.active_step == len(self.ended) and not context.skipped_iter,
                "incomplete or skipped overlap update")
        if step in self.capture_steps:
            require(self.labeler is not None and self.scope is not None, "missing overlap step observer")
            self.scope.__exit__(None, None, None)
            self.scope = None
            labeler = self.labeler
            try:
                labeler.validate_counts(4)
                record = {"schema_version": 1, "kind": "ampa_ep_overlap_calls", "rank": self.rank,
                    "step": step, "capture_mode": self.mode, "skipped_iter": False,
                    "layers": labeler.layers, "counts": dict(labeler.counts),
                    "payload_bytes": dict(labeler.payload_bytes), "payload_dtypes": dict(labeler.payload_dtypes),
                    "uccl_autograd_pairs": labeler.pair_records(),
                    "combined_pass_records": labeler.combined_pass_records(),
                    "observation_contract": labeler.contract_record(4)}
            finally:
                labeler.detach()
                self.labeler = None
            if self.mode == "torch":
                before = time.monotonic()
                self.profiler.stop()
                record["profiler_stop_seconds"] = time.monotonic() - before
                destination = self.run / "traces/torch_profile" / f"step-{step:05d}" / f"rank-{self.rank}.json.gz"
                destination.parent.mkdir(parents=True, exist_ok=True)
                require(not destination.exists(), "refusing to replace overlap trace")
                before = time.monotonic()
                self.profiler.export_chrome_trace(str(destination))
                record["export_seconds"] = time.monotonic() - before
                self.profiler = None
            write_new(self.run / "raw/overlap_calls" / f"step-{step:05d}" / f"rank-{self.rank}.json", record)
            self.captured.append(step)
            if self.mode == "nsys" and step == self.capture_steps[-1]:
                self._boundary("stop", step)
        self.ended.append(step)
        self.active_step = None

    def on_train_end(self, context):
        del context
        require(self.ended == list(range(self.steps)) and self.captured == self.capture_steps
                and self.active_step is None and self.scope is None and self.labeler is None
                and self.profiler is None and not self.aborted, "incomplete overlap acquisition")
        require(self.mode != "nsys" or (self.started and self.stopped), "incomplete Nsight capture boundaries")
        write_new(self.run / "raw" / f"ep_overlap_rank-{self.rank}_capture.json", {
            "schema_version": 1, "contract": CONTRACT, "valid_step_execution": True, "rank": self.rank,
            "capture_mode": self.mode, "completed_updates": self.steps, "ended_steps": self.ended,
            "capture_steps": self.captured, "boundaries": self.boundaries,
            "qualification_only": self.steps == 2,
            "claim_limit": "Executed host-call proof. GPU correlation and hardware metrics require separate validation."})
        self.finished = True

    def abort(self):
        if self.finished or self.aborted:
            return
        self.aborted = True
        # A failed distributed operation can leave peer-dependent kernels in
        # flight. Restore host wrappers without a CUDA wait/stop/export here.
        cleanup_error = None
        try:
            if self.scope is not None:
                self.scope.__exit__(None, None, None)
                self.scope = None
        except BaseException as error:
            cleanup_error = str(error)
        try:
            if self.labeler is not None:
                self.labeler.detach()
                self.labeler = None
        except BaseException as error:
            cleanup_error = str(error)
        try:
            write_new(self.run / "raw" / f"ep_overlap_rank-{self.rank}_aborted.json", {
                "valid": False, "rank": self.rank, "step": self.active_step,
                "cleanup_error": cleanup_error, "cuda_cleanup_attempted": False,
                "trace_exported": False, "claim_limit": "Original worker exception remains authoritative."})
        except OSError:
            pass


def make_overlap_callback(args):
    import torch
    import torch.distributed as dist
    from megatron.bridge.training.callbacks import Callback

    class OverlapCapture(_OverlapCapture, Callback):
        pass
    return OverlapCapture(args, torch, dist)
