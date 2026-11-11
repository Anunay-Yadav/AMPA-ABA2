#!/usr/bin/env python3
"""Qualify separate unprofiled budget sources and native Config on compute.

No model, process group or UCCL Buffer is constructed. Driver queries read the
allocated device limits. Real paired training remains a separate qualification.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "runs/uccl-ep-owned-build-20261101T1713Z"
FROZEN = ROOT / "runs/ep-overlap-debug-ddp-source-20261110T111031Z/source.sha256"
FROZEN_SHA = "ed74f4058652bb6f96dfa0496f70f7a888a150a2b2ee639df2d7edd499c8620a"
REFERENCE_QA = ROOT / "runs/ep-overlap-debug-ddp-qa-20261110T111031Z"
NEW_PYTHON = ("profiling/run_pretrain_ep_overlap_budget.py", "tools/ep_overlap_budget_preflight.py",
              "tools/ep_overlap_budget_validate.py", "analysis/ep_overlap_budget_compare.py",
              "tests/test_ep_overlap_budget_entrypoint.py", "tests/test_ep_overlap_budget_validate.py")
LAUNCHERS = ("jobs/training/ep_overlap_budget_pair.sbatch", "jobs/training/ep_overlap_budget_worker.sh",
             "jobs/analysis/ep_overlap_budget_posthoc.sbatch", "jobs/smoke/ep_overlap_budget_preflight.sbatch")
TESTS = ("test_ep_overlap_entrypoint.py", "test_ep_overlap_budget_entrypoint.py", "test_ep_overlap_budget_validate.py")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _paired_launcher_stdin_order_fixture(out, adapter, order):
    """Exercise one real launcher order in an independent fixture project."""
    require(order in ("candidate-first", "reference-first"), "invalid fixture order")
    fake = out / f"paired_launcher_stdin_fixture_{order}"
    fake.mkdir()
    pair_id = f"stdin-fixture-{order}"
    production = ROOT / "jobs/training/ep_overlap_budget_pair.sbatch"
    launcher = fake / "jobs/training/ep_overlap_budget_pair.sbatch"
    launcher.parent.mkdir(parents=True)
    shutil.copyfile(production, launcher)
    worker = launcher.with_name("ep_overlap_budget_worker.sh")
    worker.write_text("# Fixture-only worker path. The Slurm double never executes training.\n")
    binary = fake / "bin"
    binary.mkdir()

    def command(name, body):
        path = binary / name
        path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + body)
        path.chmod(0o755)

    command("squeue", "printf '700001 debug RUNNING 4 21:00 nid[000001-000004]\\n'\n")
    command("scontrol", "printf 'nid000001\\nnid000002\\nnid000003\\nnid000004\\n'\n")
    command("srun", r'''isolated=0
for value in "$@"; do
    if [[ $value == --input=/dev/null ]]; then isolated=1; fi
done
if (( isolated == 1 )); then exec < /dev/null; fi
cat > /dev/null
if [[ ${STDIN_FIXTURE_PROBE:-0} == 1 ]]; then
    printf 'probe\n' >> "$STDIN_FIXTURE_PROBE_LOG"
    exit 0
fi
phase=${!#}
printf '%s\t%s\n' "$AMPA_BUDGET_LEG" "$phase" >> "$STDIN_FIXTURE_LOG"
if [[ $phase == --validate ]]; then
    jq -n --arg job "$SLURM_JOB_ID" --arg run "$AMPA_OVERLAP_RUN" --arg source "$AMPA_OVERLAP_SOURCE_SHA256" \
        --arg pair "$AMPA_BUDGET_PAIR_ID" --arg role "$AMPA_BUDGET_LEG" \
        --argjson budget "$AMPA_OVERLAP_BUDGET" --argjson steps "$AMPA_OVERLAP_STEPS" \
        '{success:true,workers_srun_returned_zero:true,training_health_and_timer_summary_complete:true,
          job:$job,run_dir:$run,case:"uccl_ordinary",capture:"off",communication_sms:$budget,
          completed_updates:$steps,campaign_source_sha256:$source,qualification_only:($steps==2),
          pair_id:$pair,pair_leg:$role}' > "$AMPA_OVERLAP_ROOT/acquisition_completed.json"
fi
''')
    (fake / "runs").mkdir()
    source_manifest = fake / "runs/source.sha256"
    source_manifest.write_text("".join(f"{adapter.sha(path)}  {path}\n" for path in (launcher, worker)))
    source_sha = adapter.sha(source_manifest)
    qa = fake / "runs/fixture-qa"
    qa.mkdir()
    adapter.write_new(qa / "preflight.json", {"valid_preflight": True, "training_executed": False,
        "partition": "debug", "source_manifest_sha256": source_sha, "experiment_protocol": adapter.PROTOCOL,
        "job": "700000", "allowed_budgets": [12, 20], "focused_tests": [{"file": "fixture", "tests": 1}]})
    (qa / "PASS.txt").write_text(f"job=700000\nsource_sha256={source_sha}\npreflight_sha256={adapter.sha(qa / 'preflight.json')}\n")
    session = fake / "progress/sessions/fixture"
    session.mkdir(parents=True)
    for name in ("timeline.md", "session_state.md"):
        (session / name).write_text("Fixture only. No training or real scheduler call.\n")
    status = fake / "ai/fixture/iteration-01/STATUS.md"
    status.parent.mkdir(parents=True)
    status.write_text("Fixture only. No training or real scheduler call.\n")
    probe_log, call_log = fake / "probe_calls.txt", fake / "srun_calls.tsv"
    environment = {**os.environ, "PATH": str(binary) + os.pathsep + os.environ["PATH"],
        "SLURM_SUBMIT_DIR": str(fake), "SLURM_JOB_ID": "700001", "SLURM_JOB_PARTITION": "debug",
        "SLURM_JOB_NUM_NODES": "4", "SLURM_JOB_NODELIST": "nid[000001-000004]",
        "AMPA_BUDGET_PAIR_ID": pair_id, "AMPA_BUDGET_PAIR_CANDIDATE": "12",
        "AMPA_BUDGET_PAIR_ORDER": order, "AMPA_BUDGET_PAIR_STEPS": "2",
        "AMPA_BUDGET_CANDIDATE_RESERVE_SECONDS": "360", "AMPA_BUDGET_REFERENCE_RESERVE_SECONDS": "360",
        "AMPA_BUDGET_SOURCE_MANIFEST": str(source_manifest), "AMPA_OVERLAP_SOURCE_MANIFEST": str(source_manifest),
        "AMPA_BUDGET_SOURCE_SHA256": source_sha, "AMPA_OVERLAP_SOURCE_SHA256": source_sha,
        "AMPA_BUDGET_PREFLIGHT_ROOT": str(qa), "AMPA_BUDGET_SESSION_DIR": str(session),
        "AMPA_BUDGET_STATUS_FILE": str(status), "STDIN_FIXTURE_LOG": str(call_log),
        "STDIN_FIXTURE_PROBE_LOG": str(probe_log)}
    probe_input = fake / "probe_input.txt"
    probe_input.write_text("candidate\nreference\n")
    probe = subprocess.run(["bash", "-c", 'while IFS= read -r role; do srun "$role"; done < "$1"',
                            "stdin-probe", str(probe_input)],
        env={**environment, "STDIN_FIXTURE_PROBE": "1"}, stdin=subprocess.DEVNULL,
        capture_output=True, text=True, timeout=10)
    require(probe.returncode == 0 and probe_log.read_text().splitlines() == ["probe"],
            "stdin fixture did not reproduce inherited-stream consumption")
    with (out / "logs" / f"paired_launcher_stdin_{order}.out").open("x") as stdout, (out / "logs" / f"paired_launcher_stdin_{order}.err").open("x") as stderr:
        completed = subprocess.run(["bash", str(launcher)], cwd=fake, env=environment,
            stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, timeout=45)
    require(completed.returncode == 0, "actual paired launcher failed the inherited-stdin regression fixture")
    calls = [line.split("\t") for line in call_log.read_text().splitlines()]
    roles = ("candidate", "reference") if order == "candidate-first" else ("reference", "candidate")
    require(calls == [[role, phase] for role in roles for phase in ("--node", "--validate")],
            "actual paired loop skipped or duplicated a case/phase")
    completed_pair = json.loads((fake / "runs" / pair_id / "pair_completed.json").read_text())
    expected_ids = [f"{pair_id}-candidate-b12", f"{pair_id}-reference-b20"]
    require(completed_pair.get("success") is True and completed_pair.get("completed_legs") == 2
            and completed_pair.get("completed_case_ids") == expected_ids, "pair completion does not reflect both actual fixture cases")
    for run_id in expected_ids:
        require((fake / "runs" / run_id / "acquisition_completed.json").is_file(), "missing actual fixture case receipt")
    return {"valid_actual_pair_launcher_stdin_fixture": True, "order": order, "pair_id": pair_id,
            "unsafe_probe_calls": 1,
            "stub_srun_calls": calls, "completed_legs": 2, "completed_case_ids": expected_ids,
            "production_launcher_sha256": adapter.sha(production), "native_training_executed": False,
            "real_scheduler_commands_executed": False,
            "claim_limit": "Actual launcher control flow with explicit test doubles, not training or measurement evidence."}


def paired_launcher_stdin_fixture(out, adapter):
    """Require both actual launcher orders to survive stdin-draining doubles."""
    orders = ("candidate-first", "reference-first")
    fixtures = [_paired_launcher_stdin_order_fixture(out, adapter, order) for order in orders]
    case_ids = [run_id for fixture in fixtures for run_id in fixture["completed_case_ids"]]
    require(len(case_ids) == len(set(case_ids)) == 4, "both-order fixtures reused a case directory")
    require(sum(len(fixture["stub_srun_calls"]) for fixture in fixtures) == 8,
            "both-order fixtures did not execute all eight training/health phases")
    return {"valid_actual_pair_launcher_stdin_fixture": True, "orders_checked": list(orders),
            "pairs_checked": 2, "completed_fixture_cases": 4, "stub_srun_call_count": 8,
            "pairs": fixtures, "native_training_executed": False, "real_scheduler_commands_executed": False,
            "claim_limit": "Both actual launcher orders with test doubles only. These are not training or performance results."}


def checks(args):
    require(os.environ.get("SLURM_JOB_PARTITION") == "debug" and os.environ.get("SLURM_JOB_NUM_NODES") == "1"
            and os.environ.get("SLURM_JOB_ID", "").isdigit()
            and Path(sys.prefix) == Path("/opt/venvs/megatron-bridge"), "one allocated debug node and configured Python required")
    require(os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS") == "32" and "torch" not in sys.modules,
            "start with CUDA connections 32 before importing Torch")
    from profiling import run_pretrain_ep_overlap_budget as adapter
    from tools.ep16_posthoc_proof import import_closure

    out = args.output_dir
    require(out.is_absolute() and out.parent == ROOT / "runs" and out.is_dir() and not out.is_symlink()
            and (out / "logs").is_dir(), "prepare a unique project run directory and logs")
    require(not (out / "preflight.json").exists() and not (out / "PASS.txt").exists(), "preserve existing preflight attempts")
    result = {"schema_version": 1, "valid_preflight": False, "training_executed": False,
              "native_buffer_allocated": False, "partition": "debug", "job": os.environ["SLURM_JOB_ID"],
              "hostname": socket.gethostname(), "python": sys.executable,
              "experiment_protocol": adapter.PROTOCOL, "source_manifest_sha256": args.source_sha256,
              "started_utc": datetime.now(timezone.utc).isoformat(), "focused_tests": []}
    started = time.monotonic()
    try:
        source = adapter.verify_source_manifest(args.source_manifest, args.source_sha256)
        frozen = adapter.verify_source_manifest(FROZEN, FROZEN_SHA)
        require(all(source["files"].get(name) == digest for name, digest in frozen["files"].items()),
                "new budget source omitted or changed an original acquisition source")
        required = set(NEW_PYTHON + LAUNCHERS) | set(import_closure(ROOT, NEW_PYTHON))
        missing = sorted(required - set(source["files"]))
        adapter.write_new(out / "closure_paths.json", {"all_paths": sorted(required), "missing_paths": missing})
        require(not missing, "budget source inventory omits: " + ", ".join(missing))
        result["source_manifest"] = source
        proof = json.loads((REFERENCE_QA / "preflight.json").read_text())
        marker = dict(line.split("=", 1) for line in (REFERENCE_QA / "PASS.txt").read_text().splitlines())
        require(proof.get("valid_preflight") is True and proof.get("training_executed") is False
                and proof.get("source_manifest_sha256") == FROZEN_SHA
                and marker.get("preflight_sha256") == adapter.sha(REFERENCE_QA / "preflight.json")
                and sum(row["tests"] for row in proof["focused_tests"]) == 72,
                "the original source-bound 72-test qualification must remain valid")
        result["original_qa_reference"] = {"path": str(REFERENCE_QA / "preflight.json"),
                                           "sha256": adapter.sha(REFERENCE_QA / "preflight.json"),
                                           "tests": 72, "repeated": False}
        for name in NEW_PYTHON:
            ast.parse((ROOT / name).read_text(), filename=name)
        for index, name in enumerate(LAUNCHERS):
            completed = subprocess.run(["bash", "-n", str(ROOT / name)], capture_output=True, text=True, timeout=5)
            (out / "logs" / f"syntax-{index}.out").write_text(completed.stdout)
            (out / "logs" / f"syntax-{index}.err").write_text(completed.stderr)
            require(completed.returncode == 0, f"shell syntax failed: {name}")
        for name in TESTS:
            timeout = min(90, 420 - (time.monotonic() - started))
            require(timeout > 0, "budget preflight test deadline exhausted")
            label = name.removesuffix(".py")
            with (out / "logs" / f"{label}.out").open("x") as stdout, (out / "logs" / f"{label}.err").open("x") as stderr:
                completed = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", name, "-v"],
                    cwd=ROOT, env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"},
                    stdout=stdout, stderr=stderr, timeout=timeout)
            logs = (out / "logs" / f"{label}.out").read_text() + (out / "logs" / f"{label}.err").read_text()
            count = re.search(r"Ran ([1-9][0-9]*) tests?\b", logs)
            require(completed.returncode == 0 and count is not None
                    and not re.search(r"\bskipped=|\bexpected failures=", logs), f"focused tests failed or were skipped: {name}")
            result["focused_tests"].append({"file": name, "tests": int(count[1])})

        result["paired_launcher_stdin_fixture"] = paired_launcher_stdin_fixture(out, adapter)

        import torch
        import uccl.ep as native
        from megatron.core.transformer.transformer_config import TransformerConfig
        from megatron.core.distributed.distributed_data_parallel_config import DistributedDataParallelConfig
        from profiling.run_pretrain_uccl_budget import configure_before_model, runtime_adapter
        from tools.uccl_ep_source_identity import verify_uccl_source

        require(torch.cuda.is_available() and os.environ.get("FI_MR_CACHE_MONITOR") == "disabled",
                "allocated GPU and reviewed UCCL transport policy required")
        torch.cuda.set_device(0)
        limits = adapter.device_budget_limits(torch)
        supported = limits["supported_budgets"]
        require(set((12, 20, 32, 40)) <= set(supported), "initial budget matrix exceeds actual device launch limits")
        result.update(device_launch_limits=limits, allowed_budgets=supported, supported_budgets=supported,
                      native_launch_requirements={str(budget): adapter.kernel_launch_requirements(budget) for budget in supported})
        next_budget = max(supported) + 2
        next_requirements = adapter.kernel_launch_requirements(next_budget)
        next_threads_fit = next_requirements["notify_threads"] <= limits["max_threads_per_block"]
        next_shared_fit = next_requirements["notify_dynamic_shared_bytes"] <= limits["shared_memory_per_block_optin"]
        require(not (next_threads_fit and next_shared_fit), "adjacent upper budget is not excluded by actual launch limits")
        result["static_budget_thread_ceiling"] = 64
        result["next_unsupported_upper_budget"] = {"budget": next_budget, **next_requirements,
            "fits_actual_thread_limit": next_threads_fit,
            "fits_actual_optin_shared_limit": next_shared_fit,
            "claim_limit": "The supported range derives from actual selected-device resource limits. Config construction alone does not qualify kernel execution."}
        cfg = TransformerConfig(num_layers=48, hidden_size=2048, num_attention_heads=32,
            num_moe_experts=128, moe_ffn_hidden_size=768, moe_router_topk=8,
            expert_model_parallel_size=16, expert_tensor_parallel_size=1, pipeline_model_parallel_size=1,
            tensor_model_parallel_size=1, context_parallel_size=1, bf16=True, params_dtype=torch.bfloat16,
            use_cpu_initialization=True, moe_token_dispatcher_type="flex", moe_flex_dispatcher_backend="deepep",
            moe_grouped_gemm=True, cuda_graph_impl="none", **adapter.OVERLAP_FLAGS)
        require(all(getattr(cfg, key) is value for key, value in adapter.OVERLAP_FLAGS.items()), "native model overlap fields differ")
        ddp = DistributedDataParallelConfig(delay_wgrad_compute=True, overlap_grad_reduce=True,
                                          overlap_param_gather=True, use_distributed_optimizer=True)
        require(ddp.delay_wgrad_compute is True, "native DDP delay companion differs")
        result["configuration_only"] = {"model_overlap_flags": dict(adapter.OVERLAP_FLAGS),
            "ddp_delay_wgrad_compute": ddp.delay_wgrad_compute, "weights_constructed": False}
        require(Path(native.__file__).resolve() == BUILD / "site/uccl/ep.abi3.so", "native extension is outside the pinned build")
        source_identity = verify_uccl_source(BUILD)
        fused = runtime_adapter()
        require(fused.HAVE_DEEP_EP and fused._buffer is None and native.num_max_nvl_peers() == 4,
                "native topology differs or a Buffer was allocated before QA")
        configs = {}
        try:
            for budget in (20, *[value for value in supported if value != 20]):
                configs[str(budget)] = configure_before_model(fused, budget)
                require(fused._buffer is None, "native Config check allocated a Buffer")
                for family in ("dispatch", "combine"):
                    row, reference = configs[str(budget)][family], configs["20"][family]["config"]
                    require(isinstance(getattr(fused.Buffer, f"get_{family}_config")(16), native.Config)
                            and row["config"]["num_sms"] == budget
                            and row["nvl_buffer_size_hint"] > 0 and row["rdma_buffer_size_hint"] > 0,
                            "actual native Config or buffer hints differ")
                    require(all(value == reference[key] for key, value in row["config"].items() if key != "num_sms"),
                            "a nonbudget native Config field changed")
        finally:
            fused.Buffer.set_num_sms(20)
        require(fused._buffer is None and fused.Buffer.num_sms == 20, "native preflight did not restore its state")
        result["native_configs"] = configs
        result["native_identity"] = {"extension": native.__file__, "binary_sha256": adapter.sha(native.__file__),
            "source_identity": source_identity, "native_buffer_allocated": False, "restored_budget": 20}
        result["native_limit_sources"] = {str(BUILD / "uccl/ep" / name): adapter.sha(BUILD / "uccl/ep" / name)
            for name in ("src/internode.cu", "include/ep_launch.cuh", "include/ep_config.hpp",
                         "include/ep_utils.cuh", "include/uccl_ibgda.cuh", "deep_ep_wrapper/deep_ep/buffer.py")}
        result["versions"] = {"torch": torch.__version__, "cuda": torch.version.cuda,
                               "transformer_engine": importlib.metadata.version("transformer-engine")}
        require(adapter.verify_source_manifest(args.source_manifest, args.source_sha256) == source, "source changed during preflight")
        result.update(valid_preflight=True, finished_utc=datetime.now(timezone.utc).isoformat())
        adapter.write_new(out / "preflight.json", result)
        with (out / "PASS.txt").open("x") as stream:
            stream.write(f"preflight_passed=true\njob={result['job']}\nsource_sha256={args.source_sha256}\n"
                         f"preflight_sha256={adapter.sha(out / 'preflight.json')}\n")
    except BaseException as error:
        result["error"] = f"{type(error).__name__}: {error}"
        if not (out / "preflight.json").exists():
            adapter.write_new(out / "preflight.json", result)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    checks(parser.parse_args())


if __name__ == "__main__":
    main()
