#!/usr/bin/env python3
"""Allocated checks of overlap source, focused tests, versions and native Config.

This command does not construct a model, allocate a UCCL Buffer, or train.
The following four-node qualification remains a separate required gate.
"""
from __future__ import annotations

import argparse
import ast
import datetime
import importlib.metadata
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "runs/uccl-ep-owned-build-20261101T1713Z"
NEW_PYTHON = (
    "profiling/run_pretrain_ep_overlap.py", "profiling/ampa_profile/overlap_trace.py",
    "profiling/ampa_profile/overlap_capture.py", "tools/ep_overlap_preflight.py",
    "tools/ep_overlap_validate.py", "tools/ep_overlap_nsys_posthoc.py", "analysis/ep_overlap_gpu_activity.py",
    "tests/test_ep_overlap_entrypoint.py",
    "tests/test_overlap_trace.py", "tests/test_ep_overlap_capture.py",
    "tests/test_ep_overlap_validate.py", "tests/test_ep_overlap_gpu_activity.py", "tests/test_ep_overlap_nsys_posthoc.py",
)
LAUNCHERS = ("jobs/training/ep_overlap_case.sbatch", "jobs/training/ep_overlap_worker.sh",
             "jobs/smoke/ep_overlap_preflight.sbatch", "jobs/training/ep_overlap_checked.sbatch",
             "jobs/training/ep_overlap_posthoc.sh", "jobs/analysis/ep_overlap_posthoc.sbatch",
             "jobs/coordination/ep_overlap_campaign.sh")
FOCUSED_TESTS = ("test_ep_overlap_entrypoint.py", "test_overlap_trace.py", "test_ep_overlap_capture.py",
                 "test_ep_overlap_validate.py", "test_ep_overlap_gpu_activity.py", "test_ep_overlap_nsys_posthoc.py")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def inspect_capability(root, sha):
    """Record existing capability evidence without claiming a fresh capture."""
    root = Path(root)
    require(root.is_absolute() and root.is_dir() and not root.is_symlink(), "capability directory is unavailable")
    paths = [root / name for name in ("capabilities.json", "commands.json")]
    require(all(path.is_file() and not path.is_symlink() for path in paths), "capability metadata is unavailable")
    proof, commands = [json.loads(path.read_text()) for path in paths]
    require(proof.get("native_report_generated") is True and proof.get("cuda_profiler_api_capture_requested") is True
            and proof.get("native_capture_exit_code") == 0, "CUDAProfilerAPI capability proof did not pass")
    capture = [row for row in commands if row.get("name") == "capture_native"]
    require(len(capture) == 1 and capture[0].get("exit_code") == 0, "capability capture command is missing or failed")
    argv = capture[0].get("argv", [])
    require("--capture-range=cudaProfilerApi" in argv and "--capture-range-end=stop" in argv,
            "capability command does not prove the selected API range")
    binary = Path(proof.get("nsys_path", ""))
    require(binary.is_absolute() and binary.is_file() and not binary.is_symlink() and argv[0] == str(binary),
            "capability executable differs or is unavailable")
    return {"root": str(root), "metadata_sha256": {path.name: sha(path) for path in paths},
            "nsys_path": str(binary), "nsys_sha256": sha(binary),
            "historical_capability_job": proof.get("job_id"), "fresh_capture_executed": False,
            "claim_limit": "Historical tool capability only. New training reports require separate validation."}


def checks(args):
    # Check this before any package import that can initialize Torch/CUDA.
    require(os.environ.get("SLURM_JOB_PARTITION") == "debug"
            and os.environ.get("SLURM_JOB_NUM_NODES") == "1"
            and os.environ.get("SLURM_JOB_ID", "").isdigit(), "one-node debug allocation required")
    require(os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS") == "32", "set CUDA_DEVICE_MAX_CONNECTIONS=32 before Torch import")
    require("torch" not in sys.modules, "preflight must start before Torch import")
    from profiling.run_pretrain_ep_overlap import OVERLAP_FLAGS, sha, verify_source_manifest

    out = args.output_dir
    require(out.is_absolute() and out.parent == ROOT / "runs" and out.is_dir() and not out.is_symlink()
            and (out / "logs").is_dir(), "output must be a prepared direct project run directory with logs")
    require(not (out / "preflight.json").exists() and not (out / "PASS.txt").exists(), "preflight result already exists")
    result = {"schema_version": 1, "valid_preflight": False, "job": os.environ["SLURM_JOB_ID"],
        "hostname": socket.gethostname(), "partition": "debug", "python": sys.executable,
        "execution_location": "allocated_configured_container", "training_executed": False,
        "native_buffer_allocated": False, "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "cuda_device_max_connections": "32", "selected_flags": dict(OVERLAP_FLAGS),
        "source_manifest_sha256": args.source_sha256, "four_node_pilot_required": True}
    commands, deadline = [], time.monotonic() + 480

    def run(name, argv, timeout, env=None):
        available = min(timeout, max(0, deadline - time.monotonic() - 5))
        require(available > 0, "preflight deadline exhausted")
        row = {"name": name, "argv": [str(value) for value in argv], "timeout_seconds": available}
        commands.append(row)
        try:
            with (out / "logs" / f"{name}.out").open("x") as stdout, (out / "logs" / f"{name}.err").open("x") as stderr:
                process = subprocess.Popen(row["argv"], cwd=ROOT, env=env, stdout=stdout, stderr=stderr, start_new_session=True)
                try:
                    row["exit_code"] = process.wait(timeout=available)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    row["exit_code"] = 124
            require(row["exit_code"] == 0, f"{name} failed; inspect saved stdout/stderr")
        finally:
            write(out / "commands.json", commands)

    try:
        source = verify_source_manifest(args.source_manifest, args.source_sha256)
        require(set(NEW_PYTHON + LAUNCHERS) <= set(source["files"]), "frozen source omits overlap implementation or focused tests")
        result["source_manifest"] = source
        for relative in NEW_PYTHON:
            ast.parse((ROOT / relative).read_text(), filename=relative)
        for index, relative in enumerate(LAUNCHERS):
            run(f"launcher_syntax_{index}", ["bash", "-n", ROOT / relative], 5)
        masked = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"}
        test_records = []
        for name in FOCUSED_TESTS:
            label = name.removesuffix(".py")
            run(label, [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", name, "-v"], 90, masked)
            logs = (out / "logs" / f"{label}.out").read_text() + (out / "logs" / f"{label}.err").read_text()
            count = re.search(r"Ran ([1-9][0-9]*) tests?\b", logs)
            require(count is not None and not re.search(r"\bskipped=|\bexpected failures=", logs), "focused tests absent or skipped")
            test_records.append({"file": name, "tests": int(count[1])})
        result["focused_tests"] = test_records

        import torch
        import transformer_engine.pytorch as te
        import uccl.ep as native
        from packaging.version import Version
        from megatron.core.transformer.transformer_config import TransformerConfig
        from megatron.core.distributed.distributed_data_parallel_config import DistributedDataParallelConfig
        from profiling.run_pretrain_uccl_budget import configure_before_model, runtime_adapter
        from tools.uccl_ep_source_identity import verify_uccl_source

        require(Version(torch.__version__) >= Version("2.6.0"), "EP overlap requires PyTorch >=2.6")
        te_version = importlib.metadata.version("transformer-engine")
        require(Version(te_version) >= Version("2.7.0"), "selected delayed-gradient integration requires TE >=2.7")
        require(torch.cuda.is_available(), "allocated GPU is unavailable")
        torch.cuda.set_device(0)
        device = torch.cuda.get_device_properties(0)
        require(device.major == 9 and device.multi_processor_count == 132, "expected GH200 device geometry differs")
        require(os.environ.get("FI_MR_CACHE_MONITOR") == "disabled", "UCCL requires FI_MR_CACHE_MONITOR=disabled")
        result["versions"] = {"torch": torch.__version__, "transformer_engine": te_version,
            "cuda_runtime": torch.version.cuda, "transformer_engine_module": te.__file__}
        result["device"] = {"name": device.name, "major": device.major, "minor": device.minor,
                            "multi_processor_count": device.multi_processor_count, "total_memory": device.total_memory}
        # Construct configuration only. This exercises upstream compatibility
        # assertions without constructing weights or a distributed process group.
        config_checks = {}
        for dispatcher in ("alltoall", "flex"):
            cfg = TransformerConfig(num_layers=48, hidden_size=2048, num_attention_heads=32,
                num_moe_experts=128, moe_ffn_hidden_size=768, moe_router_topk=8,
                expert_model_parallel_size=16, expert_tensor_parallel_size=1,
                pipeline_model_parallel_size=1, tensor_model_parallel_size=1, context_parallel_size=1,
                bf16=True, params_dtype=torch.bfloat16, use_cpu_initialization=True,
                moe_token_dispatcher_type=dispatcher, moe_flex_dispatcher_backend="deepep" if dispatcher == "flex" else None,
                moe_grouped_gemm=True, cuda_graph_impl="none", **OVERLAP_FLAGS)
            require(all(getattr(cfg, key) is value for key, value in OVERLAP_FLAGS.items()), "native model configuration changed overlap fields")
            config_checks[dispatcher] = {"selected_flags": {key: getattr(cfg, key) for key in OVERLAP_FLAGS}, "weights_constructed": False}
        result["model_config_checks"] = config_checks
        ddp = DistributedDataParallelConfig(delay_wgrad_compute=True, overlap_grad_reduce=True,
            overlap_param_gather=True, use_distributed_optimizer=True)
        require(ddp.delay_wgrad_compute is True, "native DDP must delay its gradient hooks with the model")
        result["distributed_gradient_config"] = {"delay_wgrad_compute": ddp.delay_wgrad_compute,
            "overlap_grad_reduce": ddp.overlap_grad_reduce, "overlap_param_gather": ddp.overlap_param_gather,
            "use_distributed_optimizer": ddp.use_distributed_optimizer, "weights_constructed": False}
        require(Path(native.__file__).resolve() == BUILD / "site/uccl/ep.abi3.so", "native extension is not the pinned build")
        source_identity = verify_uccl_source(BUILD)
        fused = runtime_adapter()
        require(fused.HAVE_DEEP_EP and fused._buffer is None and native.num_max_nvl_peers() == 4,
                "native topology differs or a Buffer already exists")
        configs = {}
        try:
            for budget in (20, 12, 32, 40):
                configs[str(budget)] = configure_before_model(fused, budget)
                require(fused._buffer is None, "configuration check allocated a native Buffer")
                for direction in ("dispatch", "combine"):
                    cfg = getattr(fused.Buffer, f"get_{direction}_config")(16)
                    row, baseline = configs[str(budget)][direction], configs["20"][direction]["config"]
                    require(isinstance(cfg, native.Config) and row["config"]["num_sms"] == budget
                            and row["nvl_buffer_size_hint"] > 0 and row["rdma_buffer_size_hint"] > 0,
                            "native Config or buffer hints differ")
                    require(all(value == baseline[key] for key, value in row["config"].items() if key != "num_sms"),
                            "a nonbudget configuration field changed")
        finally:
            fused.Buffer.set_num_sms(20)
        require(fused.Buffer.num_sms == 20 and fused._buffer is None, "native configuration state was not restored")
        result["native_configs"] = configs
        result["native_identity"] = {"extension": native.__file__, "sha256": sha(native.__file__),
            "fused_module": fused.__file__, "buffer_module": sys.modules[fused.Buffer.__module__].__file__,
            "num_max_nvl_peers": 4, "native_buffer_allocated": False, "restored_budget": 20,
            "source_identity": source_identity}
        if args.nsys_capability_root is not None:
            result["nsys_capability"] = inspect_capability(args.nsys_capability_root, sha)
        require(verify_source_manifest(args.source_manifest, args.source_sha256) == source, "source identity changed during preflight")
        result["valid_preflight"] = True
        result["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        write(out / "preflight.json", result)
        (out / "PASS.txt").write_text(f"preflight_passed=true\njob={result['job']}\nsource_sha256={args.source_sha256}\n"
                                     f"preflight_sha256={sha(out / 'preflight.json')}\n")
    except BaseException as error:
        result["error"] = f"{type(error).__name__}: {error}"
        write(out / "preflight.json", result)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--nsys-capability-root", type=Path)
    checks(parser.parse_args())


if __name__ == "__main__":
    main()
