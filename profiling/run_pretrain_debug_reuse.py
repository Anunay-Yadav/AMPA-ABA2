#!/usr/bin/env python3
"""Run the unchanged short debug workload using first-case input verification.

Only the two immutable-input validation handlers are replaced during argument
parsing. Every training, routing, receive, fingerprint, profiler, provenance,
native identity and cleanup callback remains the canonical implementation.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

from profiling import run_pretrain_debug as debug


def load_context():
    path = Path(os.environ.get("AMPA_UCCL_REUSE_CONTEXT", ""))
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise ValueError("an absolute regular AMPA_UCCL_REUSE_CONTEXT is required")
    context = json.loads(path.read_text())
    if (context.get("verification_mode") != "reused_first_case"
            or context.get("slurm_job_id") != os.environ.get("SLURM_JOB_ID")
            or context.get("verification_reuse", {}).get("payload_verified_in_current_allocation") is not False):
        raise ValueError("reuse context must name the current allocation and explicit historical verification")
    return context


def cached_c0(args, parser, context):
    if str(args.pretrained_checkpoint) != context["pretrained_checkpoint"]:
        parser.error("checkpoint path differs from the first verified immutable C0")
    if args.pretrained_checkpoint.is_symlink() or not args.pretrained_checkpoint.is_dir():
        parser.error("the reused C0 must remain an existing nonsymlink checkpoint directory")
    if (args.hf_model_id, args.hf_revision) != (context["hf_model_id"], context["hf_revision"]):
        parser.error("model identity differs from the first verified immutable C0")
    args.c0_identity_manifest_sha256 = context["c0_identity_manifest_sha256"]
    args.c0_hf_snapshot_manifest_sha256 = context["c0_hf_snapshot_manifest_sha256"]


def cached_dataset(args, parser, context):
    if str(args.run_dir) != context["run_dir"] or str(args.dataset_manifest) != context["dataset_manifest_path"]:
        parser.error("dataset manifest/run path differs from the current prepared reuse context")
    if not args.dataset_manifest.is_file() or args.dataset_manifest.is_symlink():
        parser.error("the reused dataset manifest must remain a regular file")
    manifest = context["indexed_manifest"]
    model = manifest.get("model_and_tokenizer", {})
    if any(model.get(key) != expected for key, expected in {
        "model_id": args.hf_model_id, "revision": args.hf_revision,
        "tokenizer_id": args.hf_model_id, "tokenizer_revision": args.hf_revision}.items()):
        parser.error("reused model/tokenizer metadata differs from CLI identity")
    if args.dataset_data_path != manifest.get("validated_training_data_path"):
        parser.error("dataset data path differs from the first verified indexed data")
    expected = {"micro_batch_size": 1, "dense_data_parallel_size": 16, "gradient_accumulation": 4,
        "global_batch_size": 64, "sequence_length": 4096, "warmup_steps": 5, "measured_steps": 20, "sampler_seed": 1234}
    workload = manifest.get("validated_workload", {})
    if (any(workload.get(key) != value for key, value in expected.items())
            or (args.topology_variant, args.micro_batch_size, args.global_batch_size, args.sequence_length,
                args.warmup_steps, args.measured_steps, args.dataset_sampler_seed) != ("cp1_ep16", 1, 64, 4096, 5, 20, 1234)):
        parser.error("reused dataset requires the original 25-update CP1/EP16 sample budget")
    args.indexed_manifest = copy.deepcopy(manifest)


def parse_reuse_args(argv=None):
    context = load_context()
    canonical = debug.canonical
    previous = canonical._validate_c0_manifest, canonical._validate_dataset_manifest
    canonical._validate_c0_manifest = lambda args, parser: cached_c0(args, parser, context)
    canonical._validate_dataset_manifest = lambda args, parser: cached_dataset(args, parser, context)
    try:
        return debug.parse_debug_args(argv)
    finally:
        canonical._validate_c0_manifest, canonical._validate_dataset_manifest = previous


def main():
    canonical = debug.canonical
    previous = canonical.parse_args, canonical.build_config
    canonical.parse_args, canonical.build_config = parse_reuse_args, debug.build_debug_config
    try:
        canonical.main()
    finally:
        canonical.parse_args, canonical.build_config = previous


if __name__ == "__main__":
    main()
