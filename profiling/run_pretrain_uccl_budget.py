#!/usr/bin/env python3
"""Run the audited ordinary-UCCL workload with one explicit communication budget.

Nsight Systems owns CUDA tracing. Local NVTX ranges reuse the existing call
labels without creating a PyTorch/Kineto profiler or changing training math.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib
import json
import os
from pathlib import Path
import socket
import sys
import time
from types import SimpleNamespace

from profiling import run_pretrain_debug_reuse as reuse


canonical = reuse.debug.canonical
PROTOCOL = "uccl_ordinary_budget_nsys_5_plus_5_v1"
BUDGETS = (20, 12, 32, 40)
CONFIG_FIELDS = ("num_sms", "num_max_nvl_chunked_send_tokens", "num_max_nvl_chunked_recv_tokens",
                 "num_max_rdma_chunked_send_tokens", "num_max_rdma_chunked_recv_tokens")
OVERRIDES = ("UCCL_EP_DISPATCH_CONFIG", "UCCL_EP_COMBINE_CONFIG")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def write_new(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def reject_overrides():
    require(not any(name in os.environ for name in OVERRIDES),
            "unset UCCL_EP_DISPATCH_CONFIG and UCCL_EP_COMBINE_CONFIG for the budget experiment")


def execution_contract(budget):
    require(type(budget) is int and budget in BUDGETS, "communication budget must be 20, 12, 32, or 40")
    return {"schema_version": 1, "experiment_protocol": PROTOCOL, "communication_sms": budget,
        "budget_meaning": "Normal UCCL payload block budget; not measured physical active SMs.",
        "router_topk": 8, "router_num_groups": None, "router_group_topk": None,
        "warmup_steps": 5, "measured_steps": 5, "completed_steps": 10,
        "cache_budget_steps": 25, "cache_budget_samples": 1600, "train_iters": 25, "exit_interval": 10,
        "capture_step_indices": list(range(5, 10)), "capture_ranks": list(range(16)),
        "capture_owner": "External Nsight Systems, one collector per node around torchrun",
        "capture_api_owner": "AMPA budget callback", "kineto_enabled": False,
        "boundary_policy": "CUDA synchronize and global barrier before start/stop; barrier after every rank starts.",
        "claim_limit": "Nsight diagnostics, not profiler-free throughput or a claim of achieved occupancy."}


def parse_budget_args(argv=None):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--communication-sms", required=True, type=int, choices=BUDGETS)
    selected, remaining = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    reject_overrides()
    args = reuse.parse_reuse_args(remaining)
    if (args.communication_backend != "uccl_ep" or (args.router_num_groups, args.router_group_topk) != (None, None)
            or args.torch_profile_all_steps or args.torch_profile_steps or args.nsys_capture_steps
            or (args.warmup_steps, args.measured_steps, args.completed_steps) != (5, 5, 10)):
        parser.error("budget runs require ordinary UCCL, five warmup plus five measured updates, and no other profiler flags")
    args.communication_sms = selected.communication_sms
    return args


def runtime_adapter():
    return importlib.import_module("megatron.core.transformer.moe.fused_a2a")


def effective_configs(fused, budget):
    reject_overrides()
    require(fused.HAVE_DEEP_EP and fused.Buffer.num_sms == budget, "native Buffer budget differs")
    result = {}
    for name in ("dispatch", "combine"):
        config = getattr(fused.Buffer, f"get_{name}_config")(16)
        values = {field: int(getattr(config, field)) for field in CONFIG_FIELDS}
        require(values["num_sms"] == budget, f"effective {name} configuration overrides the requested budget")
        result[name] = {"config": values, "hidden_bytes": 4096, "ep_ranks": 16,
            "nvl_buffer_size_hint": int(config.get_nvl_buffer_size_hint(4096, 16)),
            "rdma_buffer_size_hint": int(config.get_rdma_buffer_size_hint(4096, 16))}
    return result


def configure_before_model(fused, budget):
    reject_overrides()
    require(fused.HAVE_DEEP_EP and fused._buffer is None, "set the UCCL budget before any native Buffer allocation")
    fused.Buffer.set_num_sms(budget)
    return effective_configs(fused, budget)


def build_budget_config(args):
    # The setter runs before the original configuration builder and before
    # native model construction. Its size hints therefore see the new budget.
    args.uccl_budget_setup = configure_before_model(runtime_adapter(), args.communication_sms)
    config = reuse.debug.build_debug_config(args)
    require(config.model.hidden_size == 2048, "budget hints require the pinned BF16 Qwen hidden size")
    require(config.model.moe_router_topk == 8 and config.model.moe_router_num_groups is None
            and config.model.moe_router_group_topk is None, "budget tuning must preserve ordinary top-eight routing")
    # Native manager construction calls the official setter again. Preserve
    # this value there instead of allowing its historical default of 20.
    config.model.moe_flex_dispatcher_num_sms = args.communication_sms
    config.profiling.use_nsys_profiler = False
    config.profiling.use_pytorch_profiler = False
    config.profiling.nvtx_ranges = True
    if int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0"))) == 0:
        write_new(args.run_dir / "config/uccl_budget_contract.json", execution_contract(args.communication_sms))
    return config


class NvtxTorchProxy:
    """Only the labeler's record_function interface is replaced, locally."""
    def __init__(self, torch):
        @contextmanager
        def nvtx_range(label):
            torch.cuda.nvtx.range_push(label)
            try:
                yield
            finally:
                torch.cuda.nvtx.range_pop()
        self.profiler = SimpleNamespace(record_function=nvtx_range)


class _BudgetCapture:
    def __init__(self, args, torch, dist, fused, label_type):
        self.args, self.torch, self.dist, self.fused = args, torch, dist, fused
        self.label_type, self.proxy = label_type, NvtxTorchProxy(torch)
        self.rank = None
        self.labeler = self.scope = None
        self.started = self.stopped = False
        self.next_step = 0
        self.completed, self.ended, self.boundaries = [], [], []

    def on_train_start(self, context):
        self.rank = self.dist.get_rank()
        require(self.dist.get_world_size() == 16, "budget capture requires sixteen ranks")
        require(self.fused._buffer is None, "native Buffer was allocated before budget startup checks")
        require(context.state.cfg.model.moe_flex_dispatcher_num_sms == self.args.communication_sms,
                "resolved model budget differs before the first update")
        configs = effective_configs(self.fused, self.args.communication_sms)
        require(configs == self.args.uccl_budget_setup, "effective configurations changed during model construction")
        local = {"communication_sms": self.args.communication_sms, "effective_configs": configs}
        peers = [None] * 16
        self.dist.all_gather_object(peers, local)
        require(all(item == local for item in peers), "ranks disagree on communication budgets or buffer hints")
        source = Path(__file__).resolve()
        record = {"valid_budget_startup": True, "rank": self.rank, "hostname": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "local_rank": os.environ.get("LOCAL_RANK"),
            "run_dir": str(self.args.run_dir), "communication_sms": self.args.communication_sms,
            "effective_configs": configs, "native_buffer_allocated": False,
            "all_rank_configuration_agreement": True, "configuration_ranks_checked": 16,
            "override_environment": {name: os.environ.get(name) for name in OVERRIDES},
            "adapter_source": {"path": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()},
            "contract": execution_contract(self.args.communication_sms)}
        write_new(self.args.run_dir / "raw" / f"uccl_budget_rank-{self.rank}_startup.json", record)
        if self.rank == 0:
            write_new(self.args.run_dir / "uccl_budget_environment.json", record)

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
        print(json.dumps({"ampa_nsight_boundary": record}), flush=True)

    def on_train_step_start(self, context):
        step = int(context.state.train_state.step)
        require(step == self.next_step == len(self.ended) and step in range(10) and self.labeler is None,
                "unexpected budget update order")
        require(self.fused.Buffer.num_sms == self.args.communication_sms, "native budget changed during training")
        reject_overrides()
        self.next_step += 1
        if step < 5:
            return
        if step == 5:
            self._boundary("start", step)
        self.labeler = self.label_type(self.proxy, context.model, context.optimizer)
        self.labeler.attach()
        self.scope = self.proxy.profiler.record_function(f"AMPA/STEP/{step}")
        self.scope.__enter__()

    def on_train_step_end(self, context):
        step = int(context.state.train_state.step)
        require(step == self.next_step - 1 == len(self.ended) and not context.skipped_iter,
                "incomplete or skipped budget update")
        if step < 5:
            self.ended.append(step)
            return
        require(self.labeler is not None and self.scope is not None, "missing measured NVTX update")
        self.scope.__exit__(None, None, None)
        self.scope = None
        labeler, self.labeler = self.labeler, None
        try:
            labeler.validate_counts(4)
            record = {"schema_version": 1, "kind": "ampa_nsight_uccl_calls", "rank": self.rank,
                "step": step, "skipped_iter": False, "layers": labeler.layers,
                "counts": dict(labeler.counts), "uccl_autograd_pairs": labeler.pair_records()}
        finally:
            labeler.detach()
        write_new(self.args.run_dir / "raw/nsys_calls" / f"step-{step:05d}" / f"rank-{self.rank}.json", record)
        self.completed.append(step)
        if step == 9:
            # Close step NVTX before the boundary wait. Every rank completes
            # its device work before the first node-local stop can truncate it.
            self._boundary("stop", step)
        self.ended.append(step)

    def on_train_end(self, context):
        del context
        require(self.next_step == 10 and self.ended == list(range(10))
                and self.completed == list(range(5, 10)) and self.started and self.stopped
                and self.scope is None and self.labeler is None, "incomplete Nsight budget capture")
        write_new(self.args.run_dir / "raw" / f"uccl_budget_rank-{self.rank}_capture.json", {
            "valid_nvtx_capture": True, "rank": self.rank, "communication_sms": self.args.communication_sms,
            "completed_updates": 10, "capture_steps": self.completed, "boundaries": self.boundaries,
            "claim_limit": "Call/range execution proof; external Nsight reports and hardware metrics need separate validation."})

    def abort(self):
        # Do not wait for other ranks on an exception path.
        try:
            if self.scope is not None:
                self.scope.__exit__(None, None, None)
                self.scope = None
        finally:
            try:
                if self.labeler is not None:
                    self.labeler.detach()
                    self.labeler = None
            finally:
                if self.started and not self.stopped:
                    self.torch.cuda.check_error(self.torch.cuda.cudart().cudaProfilerStop())
                    self.stopped = True


def make_budget_callback(args):
    import torch
    import torch.distributed as dist
    from megatron.bridge.training.callbacks import Callback
    from ampa_profile.uccl_trace import UcclMethodLabels

    class BudgetCapture(_BudgetCapture, Callback):
        pass
    return BudgetCapture(args, torch, dist, runtime_adapter(), UcclMethodLabels)


def main():
    pretrain_module = importlib.import_module("megatron.bridge.training.pretrain")
    provenance = importlib.import_module("ampa_profile.provenance")
    previous = (canonical.parse_args, canonical.build_config, canonical.comparison_contract,
                pretrain_module.pretrain, provenance.make_indexed_run_provenance_callback)
    state = {}

    def parse():
        state["args"] = parse_budget_args()
        return state["args"]

    def contract(args):
        return {**previous[2](args), "communication_budget": execution_contract(args.communication_sms)}

    def provenance_callback(**kwargs):
        # The original parser keeps nsys_capture_steps=0 to avoid its unrelated
        # CP2-only capture path. Runtime observation must still name Nsight.
        return previous[4](**{**kwargs, "nsys_trace": True})

    def pretrain(*, config, forward_step_func, callbacks=None):
        callback = make_budget_callback(state["args"])
        try:
            return previous[3](config=config, forward_step_func=forward_step_func,
                               callbacks=[callback, *(callbacks or [])])
        finally:
            callback.abort()

    canonical.parse_args, canonical.build_config, canonical.comparison_contract = parse, build_budget_config, contract
    pretrain_module.pretrain = pretrain
    provenance.make_indexed_run_provenance_callback = provenance_callback
    try:
        canonical.main()
    finally:
        (canonical.parse_args, canonical.build_config, canonical.comparison_contract,
         pretrain_module.pretrain, provenance.make_indexed_run_provenance_callback) = previous


if __name__ == "__main__":
    main()
