#!/usr/bin/env python3
"""Run ordinary UCCL without tracing at an explicit supported payload budget.

Reuse the original overlap preparation, cache parser, model builder and OFF
capture owner. Scoped module overrides change experiment selection and records,
then restore the imported AMPA modules. Native training code stays unchanged.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import importlib
import os
from pathlib import Path
import sys

from profiling import run_pretrain_ep_overlap as base


PROTOCOL = "ep_overlap_ordinary_off_budget_5_plus_5_v1"
BUDGETS = tuple(range(4, 65, 2))
OVERLAP_FLAGS = dict(base.OVERLAP_FLAGS)
ROOT, canonical = base.ROOT, base.canonical
write_new, verify_source_manifest, sha = base.write_new, base.verify_source_manifest, base.sha
_BASE_EXECUTION = base.execution_contract
_BASE_BUILD = base.build_overlap_config
_BASE_COMPARISON = base.comparison_contract


def allowed_budget(value):
    return type(value) is int and value in BUDGETS


def require(condition, message):
    if not condition:
        raise ValueError(message)


def kernel_launch_requirements(budget):
    # Describe the adjacent unsupported budget too. Admission remains bounded
    # by allowed_budget and then by each actual device's resource limits.
    require(type(budget) is int and budget >= 4 and budget % 2 == 0,
            "payload budget formula requires an even integer of at least 4")
    threads = max(128, 32 * (budget // 2))
    return {"payload_blocks": budget, "channels": budget // 2, "notify_threads": threads,
            "notify_dynamic_shared_bytes": 8192 * (threads // 32)}


def device_budget_limits(torch=None, *, ordinal=None):
    """Query an explicit CUDA device without creating tensors or a Buffer."""
    if ordinal is None:
        ordinal = torch.cuda.current_device() if torch is not None else int(os.environ["LOCAL_RANK"])
    require(type(ordinal) is int and ordinal in range(4), "invalid local device ordinal")
    try:
        from cuda.bindings import driver
    except ImportError:
        import cuda.cuda as driver
    require(driver.cuInit(0)[0] == driver.CUresult.CUDA_SUCCESS, "CUDA driver initialization failed")
    status, device = driver.cuDeviceGet(ordinal)
    require(status == driver.CUresult.CUDA_SUCCESS, "CUDA driver device selection failed")
    fields = (("shared_memory_per_block_optin", "CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN"),
              ("max_threads_per_block", "CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK"),
              ("warp_size", "CU_DEVICE_ATTRIBUTE_WARP_SIZE"),
              ("major", "CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR"),
              ("multi_processor_count", "CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT"))
    values = {}
    for name, enum_name in fields:
        status, value = driver.cuDeviceGetAttribute(getattr(driver.CUdevice_attribute, enum_name), device)
        require(status == driver.CUresult.CUDA_SUCCESS, f"CUDA device-limit query failed: {name}")
        values[name] = int(value)
    require(values["major"] == 9 and values["multi_processor_count"] == 132, "expected GH200 geometry differs")
    require(values["warp_size"] == 32, "native notify formula requires 32-thread CUDA warps")
    supported = [budget for budget in BUDGETS
                 if kernel_launch_requirements(budget)["notify_threads"] <= values["max_threads_per_block"]
                 and kernel_launch_requirements(budget)["notify_dynamic_shared_bytes"] <= values["shared_memory_per_block_optin"]]
    return {"ordinal": ordinal, **values, "supported_budgets": supported,
            "claim_limit": "Native launch resource bounds, not achieved active SMs, occupancy or performance."}


def selected_parser():
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--prepare-inputs", action="store_true")
    parser.add_argument("--orchestration-dir", type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--case", required=True, choices=("uccl_ordinary",))
    parser.add_argument("--capture", required=True, choices=("off",))
    parser.add_argument("--communication-sms", required=True, type=int, choices=BUDGETS)
    parser.add_argument("--execution-steps", required=True, type=int, choices=(2, 10))
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measured-steps", type=int, default=5)
    return parser


def validate_selection(selected, parser):
    expected = (1, 1) if selected.execution_steps == 2 else (5, 5)
    if (selected.case != "uccl_ordinary" or selected.capture != "off"
            or not allowed_budget(selected.communication_sms)
            or (selected.warmup_steps, selected.measured_steps) != expected):
        parser.error("budget comparisons require ordinary UCCL OFF and either 2 qualification updates (1+1) or 10 full updates (5+5)")


def checked_execution(args):
    require(args.case == "uccl_ordinary" and args.communication_backend == "uccl_ep"
            and args.overlap_capture == "off" and args.execution_steps in (2, 10)
            and allowed_budget(args.communication_sms), "invalid unprofiled budget execution")
    expected = (1, 1) if args.execution_steps == 2 else (5, 5)
    require((args.warmup_steps, args.measured_steps) == expected, "budget update window differs")


def execution_contract(args):
    checked_execution(args)
    record = _BASE_EXECUTION(args)
    record.update(experiment_protocol=PROTOCOL,
                  measurement_role="budget_qualification" if args.execution_steps == 2 else "unprofiled_budget_comparison",
                  static_payload_budget_candidates=list(BUDGETS),
                  budget_support_requirement="Actual per-device thread/shared-memory limits and the qualified preflight must admit the selected candidate.",
                  native_launch_requirements=kernel_launch_requirements(args.communication_sms),
                  claim_limit="Unprofiled full optimizer updates. Qualification updates are excluded from performance results. Block budgets are not reserved SM counts.")
    return record


def build_budget_config(args):
    checked_execution(args)
    base.require_allocation()
    args.budget_device_limits = device_budget_limits(ordinal=int(os.environ["LOCAL_RANK"]))
    require(args.communication_sms in args.budget_device_limits["supported_budgets"],
            "payload budget exceeds actual worker-device launch limits before native configuration")
    config = _BASE_BUILD(args)
    require(config.model.moe_router_num_groups is None and config.model.moe_router_group_topk is None,
            "budget comparison must preserve ordinary top-eight routing")
    require(config.model.moe_flex_dispatcher_num_sms == args.communication_sms,
            "model manager changed the actual requested payload budget")
    require(config.profiling.use_pytorch_profiler is False and config.profiling.use_nsys_profiler is False
            and config.profiling.nvtx_ranges is False, "unprofiled budget execution attached a profiler")
    return config


def comparison_contract(args):
    record = _BASE_COMPARISON(args)
    record["measurement_role"] = execution_contract(args)["measurement_role"]
    return record


def runtime_budget_record(fused, args, device, *, rank, job):
    require(allowed_budget(args.communication_sms)
            and args.communication_sms in device["supported_budgets"], "requested budget exceeds actual GPU launch limits")
    buffer = fused._buffer
    require(buffer is not None and buffer.num_sms == args.communication_sms
            and fused.Buffer.num_sms == args.communication_sms,
            "actual allocated UCCL Buffer budget differs from the requested budget")
    require(buffer.low_latency_mode is False and buffer.group_size == 16
            and buffer.num_nvl_bytes > 0 and buffer.num_rdma_bytes > 0,
            "budget guard requires normal internode EP16 Buffer execution")
    return {"schema_version": 1, "valid_actual_budget": True, "experiment_protocol": PROTOCOL,
            "rank": rank, "job": job, "first_completed_update": 1,
            "requested_budget": args.communication_sms, "buffer_num_sms": buffer.num_sms,
            "buffer_class_num_sms": fused.Buffer.num_sms, "low_latency_mode": False,
            "group_size": 16, "device_launch_limits": device,
            "claim_limit": "Configured and exercised payload block budget, not a measured active-SM count."}


def make_budget_callback(args):
    checked_execution(args)
    import torch
    import torch.distributed as dist
    from megatron.bridge.training.callbacks import Callback
    from profiling.ampa_profile.overlap_capture import _OverlapCapture
    from profiling.run_pretrain_uccl_budget import runtime_adapter

    class BudgetCapture(_OverlapCapture, Callback):
        def on_train_start(self, context):
            super().on_train_start(context)
            self.budget_device = device_budget_limits(self.torch)
            require(self.budget_device == args.budget_device_limits,
                    "actual selected worker device differs from the early native-launch guard")
            require(args.communication_sms in self.budget_device["supported_budgets"],
                    "requested payload budget exceeds actual selected-device launch limits")

        def on_train_step_end(self, context):
            super().on_train_step_end(context)
            if len(self.ended) == 1:
                record = runtime_budget_record(runtime_adapter(), args, self.budget_device,
                                               rank=self.rank, job=os.environ["SLURM_JOB_ID"])
                write_new(Path(args.run_dir) / "raw" / f"ep_overlap_budget_rank-{self.rank}_runtime.json", record)

    return BudgetCapture(args, torch, dist)


@contextmanager
def compat_context():
    """Apply the separate AMPA protocol without editing any imported source."""
    capture = importlib.import_module("profiling.ampa_profile.overlap_capture")
    changes = {"PROTOCOL": PROTOCOL, "BUDGETS": BUDGETS, "selected_parser": selected_parser,
               "validate_selection": validate_selection, "execution_contract": execution_contract,
               "build_overlap_config": build_budget_config, "comparison_contract": comparison_contract}
    previous = {name: getattr(base, name) for name in changes}
    previous_capture = capture.make_overlap_callback
    try:
        for name, value in changes.items():
            setattr(base, name, value)
        capture.make_overlap_callback = make_budget_callback
        yield base
    finally:
        for name, value in previous.items():
            setattr(base, name, value)
        capture.make_overlap_callback = previous_capture


def main(argv=None):
    values = sys.argv[1:] if argv is None else argv
    parser = selected_parser()
    selected, _ = parser.parse_known_args(values)
    validate_selection(selected, parser)
    base.require_allocation()
    with compat_context():
        return base.main(values)


if __name__ == "__main__":
    main()
