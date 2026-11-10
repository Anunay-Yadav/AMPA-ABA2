#!/usr/bin/env python3
"""Run ten real updates while retaining the reviewed 25-update data cache.

The normal entrypoint validates the original checkpoint/data/cache contract.
This adapter changes only the requested measurement window and native exit
interval. The normal trainer, model, optimizer, callbacks, and sample order
remain authoritative.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import sys

from profiling import run_pretrain as canonical


BASE_PARSE_ARGS = canonical.parse_args
BASE_BUILD_CONFIG = canonical.build_config


def execution_contract() -> dict:
    return {
        "schema_version": 1,
        "warmup_steps": 5,
        "measured_steps": 5,
        "completed_steps": 10,
        "cache_budget_steps": 25,
        "cache_budget_samples": 1600,
        "train_iters": 25,
        "exit_interval": 10,
        "measured_step_indices": list(range(5, 10)),
        "capture_step_indices": list(range(10)),
        "sample_order_policy": "Original 1600-sample indexed cache; consume its first 640 samples.",
        "placement_policy": "Record actual placement independently for each allocation.",
        "claim_limit": "Short descriptive timing; no assumed matched placement across allocations.",
    }


def parse_debug_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measured-steps", type=int, default=5)
    requested, remaining = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    if (requested.warmup_steps, requested.measured_steps) != (5, 5):
        parser.error("the debug entrypoint requires exactly five warmup and five measured updates")
    if (os.environ.get("SLURM_JOB_PARTITION") != "debug"
            or os.environ.get("SLURM_JOB_NUM_NODES") != "4"
            or not os.environ.get("SLURM_JOB_ID", "").isdigit()):
        parser.error("the debug entrypoint requires a fresh four-node debug allocation")
    saved_argv = sys.argv
    try:
        sys.argv = [str(Path(__file__)), *remaining, "--warmup-steps", "5", "--measured-steps", "20"]
        args = BASE_PARSE_ARGS()
    finally:
        sys.argv = saved_argv
    if (args.topology_variant != "cp1_ep16" or args.launch_policy != "torchrun_node4"
            or args.smoke_steps is not None or args.nsys_capture_steps or args.torch_profile_steps
            or not args.audit_received_experts or not args.routing_audit
            or not args.sample_fingerprints or args.log_interval != 1):
        parser.error("debug requires the audited CP1/EP16 node4 workload without another smoke/capture window")
    args.warmup_steps = 5
    args.measured_steps = 5
    args.completed_steps = 10
    return args


def build_debug_config(args: argparse.Namespace):
    if (args.warmup_steps, args.measured_steps, args.completed_steps) != (5, 5, 10):
        raise ValueError("debug execution must contain exactly five warmup and five measured updates")
    cache_args = copy.copy(args)
    cache_args.measured_steps = 20
    cache_args.smoke_steps = 10
    config = BASE_BUILD_CONFIG(cache_args)
    if config.train.train_iters != 25 or config.train.exit_interval != 10:
        raise ValueError("debug must preserve the 25-update sample budget and exit after ten updates")
    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))
    if rank == 0:
        output = args.run_dir / "config" / "debug_execution_contract.json"
        with output.open("x") as stream:
            json.dump(execution_contract(), stream, indent=2, sort_keys=True)
            stream.write("\n")
    return config


def main() -> None:
    original_parse, original_build = canonical.parse_args, canonical.build_config
    canonical.parse_args, canonical.build_config = parse_debug_args, build_debug_config
    try:
        canonical.main()
    finally:
        canonical.parse_args, canonical.build_config = original_parse, original_build


if __name__ == "__main__":
    main()
