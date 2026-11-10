#!/usr/bin/env bash
# One independent C0 start, five warmup plus five measured real updates.
#SBATCH --job-name=ampa-uccl-debug-case
#SBATCH --account=infra01
#SBATCH --partition=debug
#SBATCH --qos=normal
#SBATCH --constraint=gh
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --exclusive
#SBATCH --time=00:22:30
#SBATCH --no-requeue
#SBATCH --output=.runtime/slurms/slurm-%x-%j.out
#SBATCH --error=.runtime/slurms/slurm-%x-%j.err

set -euo pipefail
if [[ "${1:-}" == --selfcheck ]]; then
    [[ $# == 1 && -n "${SLURM_JOB_ID:-}" ]] || exit 2
    readonly CHECK_ROOT="${SLURM_SUBMIT_DIR:?}"
    for script in jobs/training/uccl_ep_debug_case.sbatch jobs/training/qwen3_uccl_debug_worker.sh; do
        bash -n "$CHECK_ROOT/$script"
    done
    /opt/venvs/megatron-bridge/bin/python - "$CHECK_ROOT" <<'PY'
import ast
from pathlib import Path
import re
import sys

root = Path(sys.argv[1])
for relative in ("profiling/run_pretrain_debug.py", "tools/uccl_ep_debug_validate.py"):
    ast.parse((root / relative).read_text(), filename=relative)
for relative in ("jobs/training/uccl_ep_debug_case.sbatch", "jobs/training/qwen3_uccl_debug_worker.sh"):
    content = (root / relative).read_text()
    for index, body in enumerate(re.findall(r"<<'PY'\n(.*?)\nPY", content, re.DOTALL)):
        ast.parse(body, filename=f"{relative}:heredoc-{index}")
print("debug_launcher_syntax_passed=true")
PY
    exit 0
fi
[[ "${SLURM_JOB_PARTITION:-}" == debug && "${SLURM_JOB_NUM_NODES:-}" == 4 ]] || exit 2
readonly ROOT_DIR="${SLURM_SUBMIT_DIR:?}"
[[ "${AMPA_UCCL_DEBUG_ID:?}" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
[[ "${AMPA_UCCL_DEBUG_CASE:?}" =~ ^(ordinary|grouped|uccl_ordinary|uccl)$ ]] || exit 2
[[ "${AMPA_UCCL_DEBUG_LEG:?}" =~ ^(control|profile)$ ]] || exit 2
export AMPA_UCCL_COMPARISON_MODE=full
export AMPA_UCCL_COMPARISON_ID="$AMPA_UCCL_DEBUG_ID"
export AMPA_UCCL_COMPARISON_ROOT="$ROOT_DIR/runs/$AMPA_UCCL_DEBUG_ID"
export AMPA_UCCL_DEBUG_SESSION_DIR="${AMPA_UCCL_DEBUG_SESSION_DIR:-$ROOT_DIR/progress/sessions/2026-11-01-uccl-ep-comparison}"
export AMPA_UCCL_DEBUG_STATUS_FILE="${AMPA_UCCL_DEBUG_STATUS_FILE:-$ROOT_DIR/ai/qwen-debug-case-launcher/iteration-01/STATUS.md}"
[[ "$AMPA_UCCL_DEBUG_SESSION_DIR" == "$ROOT_DIR"/progress/sessions/*
    && "$AMPA_UCCL_DEBUG_STATUS_FILE" == "$ROOT_DIR"/ai/*/iteration-*/STATUS.md ]] || exit 2
export AMPA_UCCL_BUILD_ROOT="$ROOT_DIR/runs/uccl-ep-owned-build-20261101T1713Z"
export AMPA_C0="$ROOT_DIR/checkpoint/megatron/qwen3-30b-a3b-base-c0"
export AMPA_HF_REVISION=1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9
export AMPA_DATA_MANIFEST="$ROOT_DIR/dataset/manifests/ep16-cache-20261011T1757Z-cp1-ep16.json"
export AMPA_DATA_PATH="0.55 $ROOT_DIR/dataset/indexed/qwen3-30b-a3b-base-realistic-mix-v2/web/qwen3_30b_a3b_base_realistic_mix_web 0.15 $ROOT_DIR/dataset/indexed/qwen3-30b-a3b-base-realistic-mix-v2/multilingual/qwen3_30b_a3b_base_realistic_mix_multilingual 0.15 $ROOT_DIR/dataset/indexed/qwen3-30b-a3b-base-realistic-mix-v2/code/qwen3_30b_a3b_base_realistic_mix_code 0.15 $ROOT_DIR/dataset/indexed/qwen3-30b-a3b-base-realistic-mix-v2/math/qwen3_30b_a3b_base_realistic_mix_math"
export AMPA_REVIEWED_INPUTS_PROOF="$AMPA_UCCL_COMPARISON_ROOT/reviewed_inputs.json"
export EFA_HOME="$AMPA_UCCL_BUILD_ROOT/not-efa"
case "${LD_LIBRARY_PATH:-}" in
    /opt/amazon/efa/lib|/opt/amazon/efa/lib:*) ;;
    *) export LD_LIBRARY_PATH="/opt/amazon/efa/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
esac
export UCCL_EP_TRANSPORT=cxi CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$AMPA_UCCL_BUILD_ROOT/site:$AMPA_UCCL_BUILD_ROOT/deps:$AMPA_UCCL_BUILD_ROOT/uccl/ep/deep_ep_wrapper:$ROOT_DIR:$ROOT_DIR/profiling:$ROOT_DIR/analysis${PYTHONPATH:+:$PYTHONPATH}"

if [[ "${1:-}" == --preflight ]]; then
    [[ $# == 1 ]] || exit 2
    cd "$ROOT_DIR"
    readonly PREFLIGHT_PY=/opt/venvs/megatron-bridge/bin/python
    sha256sum -c "$AMPA_UCCL_BUILD_ROOT/binary.sha256"
    sha256sum -c "$AMPA_UCCL_BUILD_ROOT/libfabric_runtime.sha256"
    sha256sum -c "$AMPA_UCCL_COMPARISON_ROOT/source.sha256"
    sha256sum -c "$AMPA_UCCL_COMPARISON_ROOT/campaign-source.sha256"
    "$PREFLIGHT_PY" - <<'PY'
import hashlib
import json
import os
from pathlib import Path
import shutil

from ampa_profile.provenance import _ampa_runtime_sources
from tools.ep16_reviewed_inputs import reviewed_inputs
from tools.ep16_recovery_placement import sha256, write_new
from tools.uccl_ep_source_identity import verify_uccl_source

root = Path(os.environ["SLURM_SUBMIT_DIR"])
build = Path(os.environ["AMPA_UCCL_BUILD_ROOT"])
output = Path(os.environ["AMPA_UCCL_COMPARISON_ROOT"])
sources = _ampa_runtime_sources(root)
uccl_source_identity = verify_uccl_source(build)
lines = (build / "binary.sha256").read_text().splitlines()
if len(lines) != 1:
    raise ValueError("one reviewed UCCL binary is required")
binary_hash, binary_name = lines[0].split(maxsplit=1)
binary = Path(binary_name.lstrip(" *"))
if not binary.is_absolute() or binary.parent != build / "site/uccl" or binary.is_symlink() or not binary.is_file():
    raise ValueError("build binary must use its isolated absolute path")
if sha256(binary) != binary_hash:
    raise ValueError("actual UCCL binary differs from build manifest")
with (output / "runtime-source.sha256").open("x") as handle:
    for relative, digest in sorted(sources["files"].items()):
        handle.write(f"{digest}  {root / relative}\n")
# Reuse the input-verification function directly. Its CLI's old partition guard
# is not applied or spoofed for this explicitly authorized debug experiment.
reviewed = reviewed_inputs(
    root / "runs/ep16-all-steps-group4-top2-fresh-20261011T1957Z-cp1-normal-torchprof-all",
    root, Path(os.environ["AMPA_DATA_MANIFEST"]))
if reviewed["ampa_runtime_sources"] != sources:
    raise ValueError("runtime source changed during immutable input verification")
snapshot = output / "source/runtime"
snapshot.mkdir(parents=True, exist_ok=False)
for relative, expected in sources["files"].items():
    destination = snapshot / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(root / relative, destination)
    if sha256(destination) != expected or sha256(root / relative) != expected:
        raise ValueError(f"runtime source changed during snapshot: {relative}")
write_new(output / "reviewed_inputs.json", reviewed)
historical = None
precursor_path = os.environ.get("AMPA_UCCL_DEBUG_HISTORICAL_PILOT_PROOF")
if precursor_path:
    precursor = Path(precursor_path)
    if not precursor.is_absolute() or precursor.is_symlink() or not precursor.is_file():
        raise ValueError("historical precursor must be an absolute regular file")
    proof = json.loads(precursor.read_text())
    if (proof.get("valid_pilot") is not True or proof.get("completed_updates") != 2
            or proof.get("binary_sha256") != binary_hash
            or proof.get("runtime_source_sha256") != sources["sha256"]):
        raise ValueError("historical precursor identity does not match current source/build")
    historical = {"path": str(precursor), "sha256": sha256(precursor),
                  "role": "historical precursor only; not acceptance of the shortened debug experiment"}
record = {
    "mode": "full", "experiment_protocol": "debug_one_case_5_plus_5_v1",
    "job": os.environ["SLURM_JOB_ID"], "partition": "debug",
    "case": os.environ["AMPA_UCCL_DEBUG_CASE"], "leg": os.environ["AMPA_UCCL_DEBUG_LEG"],
    "warmup_steps": 5, "measured_steps": 5, "completed_updates": 10,
    "cache_budget_updates": 25, "cache_budget_samples": 1600,
    "topology": "cp1_ep16", "launch_policy": "torchrun_node4",
    "binary_sha256": binary_hash, "runtime_source_sha256": sources["sha256"],
    "campaign_source_sha256": sha256(output / "campaign-source.sha256"),
    "accepted_pilot_proof": None, "historical_pilot_proof": historical,
    "uccl_source_identity": uccl_source_identity,
    "uccl_transport_policy": {
        "FI_MR_CACHE_MONITOR": "disabled",
        "scope": "both_uccl_routing_modes_and_profiler_settings_before_native_initialization",
        "claim_limit": "Disables the system monitor; registration overhead can change."},
    "comparison_scope": {
        "one_case_per_allocation": True, "same_cross_case_physical_placement": False,
        "claim_limit": "Different allocations can use different nodes and network conditions."},
    "claim_limit": "Acquisition metadata does not replace health, routing, receive, input-prefix, backend or trace acceptance."
}
write_new(output / "acquisition.json", record)
print(json.dumps({"valid_reviewed_inputs": True, "mode": "debug_one_case",
                  "case": record["case"], "leg": record["leg"], "completed_updates": 10,
                  "runtime_source_sha256": sources["sha256"]}), flush=True)
PY
    exit 0
fi
[[ -z "${1:-}" && ! -e "$EFA_HOME" ]] || exit 2
[[ ! -e "$AMPA_UCCL_COMPARISON_ROOT" && ! -L "$AMPA_UCCL_COMPARISON_ROOT" ]] || exit 2
[[ -f "$AMPA_UCCL_BUILD_ROOT/binary.sha256" && -f "$AMPA_DATA_MANIFEST" ]] || exit 2
[[ -f "$AMPA_UCCL_DEBUG_SESSION_DIR/timeline.md" && -f "$AMPA_UCCL_DEBUG_SESSION_DIR/session_state.md"
    && -f "$AMPA_UCCL_DEBUG_STATUS_FILE" ]] || exit 2
[[ "${AMPA_UCCL_DEBUG_EXPECTED_SOURCE_MANIFEST:?}" == "$ROOT_DIR"/runs/*
    && -f "$AMPA_UCCL_DEBUG_EXPECTED_SOURCE_MANIFEST" && ! -L "$AMPA_UCCL_DEBUG_EXPECTED_SOURCE_MANIFEST"
    && "${AMPA_UCCL_DEBUG_EXPECTED_SOURCE_SHA256:?}" =~ ^[a-f0-9]{64}$ ]] || exit 2
[[ $(sha256sum "$AMPA_UCCL_DEBUG_EXPECTED_SOURCE_MANIFEST" | cut -d' ' -f1) == "$AMPA_UCCL_DEBUG_EXPECTED_SOURCE_SHA256" ]] || exit 2
sha256sum -c "$AMPA_UCCL_DEBUG_EXPECTED_SOURCE_MANIFEST"
mkdir -p "$AMPA_UCCL_COMPARISON_ROOT"/{source,logs}
cp "$AMPA_UCCL_DEBUG_EXPECTED_SOURCE_MANIFEST" "$AMPA_UCCL_COMPARISON_ROOT/campaign-source.sha256"
sha256sum "$AMPA_UCCL_COMPARISON_ROOT/campaign-source.sha256" >> "$AMPA_UCCL_COMPARISON_ROOT/source.sha256"
for relative in jobs/training/uccl_ep_debug_case.sbatch jobs/training/qwen3_uccl_debug_worker.sh \
    profiling/run_pretrain_debug.py tools/uccl_ep_debug_validate.py tools/uccl_ep_log_proof.py \
    tools/ep16_reviewed_inputs.py analysis/summarize_training_evidence.py analysis/extract_step_timing.py; do
    mkdir -p "$AMPA_UCCL_COMPARISON_ROOT/source/$(dirname "$relative")"
    cp "$ROOT_DIR/$relative" "$AMPA_UCCL_COMPARISON_ROOT/source/$relative"
    sha256sum "$ROOT_DIR/$relative" >> "$AMPA_UCCL_COMPARISON_ROOT/source.sha256"
done
sha256sum "$AMPA_UCCL_BUILD_ROOT/binary.sha256" "$AMPA_UCCL_BUILD_ROOT/libfabric_runtime.sha256" \
    "$AMPA_UCCL_BUILD_ROOT/source_identity.json" "$AMPA_UCCL_BUILD_ROOT/source/uccl-borrowed-scratch.patch" \
    "$AMPA_UCCL_BUILD_ROOT/verification.json" >> "$AMPA_UCCL_COMPARISON_ROOT/source.sha256"
for relative in ep/include/uccl_proxy.hpp ep/src/uccl_proxy.cpp ep/src/uccl_ep.cc ep/bench/utils.py; do
    sha256sum "$AMPA_UCCL_BUILD_ROOT/uccl/$relative" >> "$AMPA_UCCL_COMPARISON_ROOT/source.sha256"
done
record() {
    local note="$(date -u '+%Y-%m-%d %H:%M:%S UTC') job $SLURM_JOB_ID: $*"
    printf '%s\n' "$note" >> "$AMPA_UCCL_COMPARISON_ROOT/timeline.md"
    printf '%s\n' "$note" > "$AMPA_UCCL_COMPARISON_ROOT/CURRENT.md"
    printf '\n- %s\n' "$note" >> "$AMPA_UCCL_DEBUG_SESSION_DIR/timeline.md"
    printf '\n- %s\n' "$note" >> "$AMPA_UCCL_DEBUG_SESSION_DIR/session_state.md"
    printf '\nCompute milestone: %s\n' "$note" >> "$AMPA_UCCL_DEBUG_STATUS_FILE"
}
complete=0
on_exit() {
    local status=$?
    trap - EXIT
    if (( status != 0 || complete != 1 )); then
        record "EXIT status=$status; debug acquisition incomplete; preserve all artifacts"
    fi
    exit "$status"
}
trap on_exit EXIT
record "START debug case $AMPA_UCCL_DEBUG_CASE/$AMPA_UCCL_DEBUG_LEG on $SLURM_JOB_NODELIST; 5 warmup + 5 measured; configured container setup/gh200-cuda130-runtime.toml"
run="$ROOT_DIR/runs/$AMPA_UCCL_DEBUG_ID-$AMPA_UCCL_DEBUG_CASE-$AMPA_UCCL_DEBUG_LEG"
[[ ! -e "$run" && ! -L "$run" ]] || exit 2
mkdir -p "$run"/{config,raw,metrics,analysis,plots,traces,logs/wandb}
cp "$AMPA_DATA_MANIFEST" "$run/config/dataset_manifest.json"
cp "$AMPA_C0/ampa_c0_manifest.json" "$run/config/c0_manifest.json"
cp "$ROOT_DIR/configs/model/qwen3_30b_a3b_base.json" "$run/config/model_manifest.json"
printf '%s\t%s\t%s\n' "$AMPA_UCCL_DEBUG_CASE" "$AMPA_UCCL_DEBUG_LEG" "$run" > "$AMPA_UCCL_COMPARISON_ROOT/cases.tsv"
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
export MASTER_PORT="$((21000 + SLURM_JOB_ID % 20000))"
record "BEFORE compute immutable input/source/native preflight"
squeue -u "$USER"
srun --mpi=pmix --cpu-bind=none --container-remap-root --environment="$ROOT_DIR/setup/gh200-cuda130-runtime.toml" \
    --nodes=1 --ntasks=1 --gpus-per-task=1 --output="$AMPA_UCCL_COMPARISON_ROOT/logs/preflight.out" \
    --error="$AMPA_UCCL_COMPARISON_ROOT/logs/preflight.err" \
    bash "$ROOT_DIR/jobs/training/uccl_ep_debug_case.sbatch" --preflight
record "AFTER compute preflight passed; BEFORE compute one real10-update worker case"
squeue -u "$USER"
srun --mpi=pmix --cpu-bind=none --container-remap-root --environment="$ROOT_DIR/setup/gh200-cuda130-runtime.toml" \
    --nodes=4 --ntasks=4 --ntasks-per-node=1 --gpus-per-task=4 --kill-on-bad-exit=1 \
    --output="$AMPA_UCCL_COMPARISON_ROOT/logs/node-%t.out" --error="$AMPA_UCCL_COMPARISON_ROOT/logs/node-%t.err" \
    bash "$ROOT_DIR/jobs/training/qwen3_uccl_debug_worker.sh" --node
export AMPA_UCCL_WORKERS_SUCCEEDED=1
record "AFTER compute all16 workers exited0; BEFORE compute10-update health and5-update timing checks"
squeue -u "$USER"
srun --mpi=pmix --cpu-bind=none --container-remap-root --environment="$ROOT_DIR/setup/gh200-cuda130-runtime.toml" \
    --nodes=1 --ntasks=1 --gpus-per-task=1 --output="$AMPA_UCCL_COMPARISON_ROOT/logs/health-timing.out" \
    --error="$AMPA_UCCL_COMPARISON_ROOT/logs/health-timing.err" \
    bash "$ROOT_DIR/jobs/training/qwen3_uccl_debug_worker.sh" --validate
sha256sum -c "$AMPA_UCCL_COMPARISON_ROOT/source.sha256"
sha256sum -c "$AMPA_UCCL_COMPARISON_ROOT/runtime-source.sha256"
sha256sum -c "$AMPA_UCCL_COMPARISON_ROOT/campaign-source.sha256"
complete=1
record "ACQUISITION COMPLETE; all10 logged updates healthy, five timing samples saved. Separate allocated receive/trace/backend acceptance remains required."

