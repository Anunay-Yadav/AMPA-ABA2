#!/usr/bin/env bash
# One C0 start with unchanged measurements and reused first-case input verification.
#SBATCH --job-name=ampa-uccl-debug-reuse
#SBATCH --account=infra01
#SBATCH --partition=debug
#SBATCH --qos=normal
#SBATCH --constraint=gh
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --exclusive
#SBATCH --time=00:22:00
#SBATCH --no-requeue
#SBATCH --output=.runtime/slurms/slurm-%x-%j.out
#SBATCH --error=.runtime/slurms/slurm-%x-%j.err

set -euo pipefail
if [[ "${1:-}" == --selfcheck ]]; then
    [[ $# == 1 && -n "${SLURM_JOB_ID:-}" ]] || exit 2
    readonly CHECK_ROOT="${SLURM_SUBMIT_DIR:?}"
    for script in jobs/training/uccl_ep_debug_reuse.sbatch jobs/training/qwen3_uccl_debug_reuse_worker.sh; do
        bash -n "$CHECK_ROOT/$script"
    done
    /opt/venvs/megatron-bridge/bin/python - "$CHECK_ROOT" <<'PY'
import ast
from pathlib import Path
import re
import sys

root = Path(sys.argv[1])
for relative in ("profiling/run_pretrain_debug_reuse.py", "tools/uccl_ep_verified_inputs_reuse.py"):
    ast.parse((root / relative).read_text(), filename=relative)
for relative in ("jobs/training/uccl_ep_debug_reuse.sbatch", "jobs/training/qwen3_uccl_debug_reuse_worker.sh"):
    for index, body in enumerate(re.findall(r"<<'PY'\n(.*?)\nPY", (root / relative).read_text(), re.DOTALL)):
        ast.parse(body, filename=f"{relative}:heredoc-{index}")
print("debug_reuse_launcher_syntax_passed=true")
PY
    exit 0
fi
[[ "${SLURM_JOB_PARTITION:-}" == debug && "${SLURM_JOB_NUM_NODES:-}" == 4 ]] || exit 2
readonly ROOT_DIR="${SLURM_SUBMIT_DIR:?}"
[[ "${AMPA_UCCL_DEBUG_ID:?}" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
[[ "${AMPA_UCCL_DEBUG_CASE:?}" =~ ^(ordinary|grouped|uccl_ordinary|uccl)$ ]] || exit 2
[[ "${AMPA_UCCL_DEBUG_LEG:?}" =~ ^(control|profile)$ ]] || exit 2
[[ "${AMPA_UCCL_VERIFIED_FIRST_ROOT:?}" == "$ROOT_DIR"/runs/* ]] || exit 2
export AMPA_UCCL_VERIFIED_FIRST_ROOT
export AMPA_UCCL_COMPARISON_MODE=full
export AMPA_UCCL_COMPARISON_ID="$AMPA_UCCL_DEBUG_ID"
export AMPA_UCCL_COMPARISON_ROOT="$ROOT_DIR/runs/$AMPA_UCCL_DEBUG_ID"
export AMPA_UCCL_REUSE_CONTEXT="$AMPA_UCCL_COMPARISON_ROOT/input_reuse_context.json"
export AMPA_UCCL_DEBUG_SESSION_DIR="${AMPA_UCCL_DEBUG_SESSION_DIR:-$ROOT_DIR/progress/sessions/2026-11-01-uccl-ep-comparison}"
export AMPA_UCCL_DEBUG_STATUS_FILE="${AMPA_UCCL_DEBUG_STATUS_FILE:-$ROOT_DIR/ai/qwen-debug-5plus5/iteration-01/STATUS.md}"
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
[[ -z "${1:-}" ]] || exit 2
[[ ! -e "$AMPA_UCCL_COMPARISON_ROOT" && ! -L "$AMPA_UCCL_COMPARISON_ROOT" ]] || exit 2
mkdir -p "$AMPA_UCCL_COMPARISON_ROOT"/{source,logs}
# These are records only. No manifest or source comparison gates run here.
if [[ -n "${AMPA_UCCL_DEBUG_EXPECTED_SOURCE_MANIFEST:-}" ]]; then
    cp "$AMPA_UCCL_DEBUG_EXPECTED_SOURCE_MANIFEST" "$AMPA_UCCL_COMPARISON_ROOT/campaign-source.sha256"
fi
for relative in jobs/training/uccl_ep_debug_reuse.sbatch jobs/training/qwen3_uccl_debug_reuse_worker.sh \
    profiling/run_pretrain_debug_reuse.py tools/uccl_ep_verified_inputs_reuse.py \
    profiling/run_pretrain_debug.py analysis/summarize_training_evidence.py analysis/extract_step_timing.py; do
    mkdir -p "$AMPA_UCCL_COMPARISON_ROOT/source/$(dirname "$relative")"
    cp "$ROOT_DIR/$relative" "$AMPA_UCCL_COMPARISON_ROOT/source/$relative"
    sha256sum "$ROOT_DIR/$relative" >> "$AMPA_UCCL_COMPARISON_ROOT/source.sha256"
done
if [[ ! -f "$AMPA_UCCL_COMPARISON_ROOT/campaign-source.sha256" ]]; then
    cp "$AMPA_UCCL_COMPARISON_ROOT/source.sha256" "$AMPA_UCCL_COMPARISON_ROOT/campaign-source.sha256"
fi
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
        record "EXIT status=$status; debug reused-input acquisition incomplete; preserve all artifacts"
    fi
    exit "$status"
}
trap on_exit EXIT
record "START debug reused-input case $AMPA_UCCL_DEBUG_CASE/$AMPA_UCCL_DEBUG_LEG on $SLURM_JOB_NODELIST; 5warmup+5measured; unchanged measurement callbacks; first verification $AMPA_UCCL_VERIFIED_FIRST_ROOT"
run="$ROOT_DIR/runs/$AMPA_UCCL_DEBUG_ID-$AMPA_UCCL_DEBUG_CASE-$AMPA_UCCL_DEBUG_LEG"
[[ ! -e "$run" && ! -L "$run" ]] || exit 2
mkdir -p "$run"/{config,raw,metrics,analysis,plots,traces,logs/wandb}
printf '%s\t%s\t%s\n' "$AMPA_UCCL_DEBUG_CASE" "$AMPA_UCCL_DEBUG_LEG" "$run" > "$AMPA_UCCL_COMPARISON_ROOT/cases.tsv"
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
export MASTER_PORT="$((21000 + SLURM_JOB_ID % 20000))"
record "BEFORE compute node0 prepares reused-input records; all4nodes then execute10realupdates; no repeated payload/source checks or separate preflight"
squeue -u "$USER"
srun --mpi=pmix --cpu-bind=none --container-remap-root --environment="$ROOT_DIR/setup/gh200-cuda130-runtime.toml" \
    --nodes=4 --ntasks=4 --ntasks-per-node=1 --gpus-per-task=4 --kill-on-bad-exit=1 \
    --output="$AMPA_UCCL_COMPARISON_ROOT/logs/node-%t.out" --error="$AMPA_UCCL_COMPARISON_ROOT/logs/node-%t.err" \
    bash "$ROOT_DIR/jobs/training/qwen3_uccl_debug_reuse_worker.sh" --node
export AMPA_UCCL_WORKERS_SUCCEEDED=1
record "AFTER compute all16 workers exited0; BEFORE compute small10-update health and5-update timing records"
squeue -u "$USER"
srun --mpi=pmix --cpu-bind=none --container-remap-root --environment="$ROOT_DIR/setup/gh200-cuda130-runtime.toml" \
    --nodes=1 --ntasks=1 --gpus-per-task=1 --output="$AMPA_UCCL_COMPARISON_ROOT/logs/health-timing.out" \
    --error="$AMPA_UCCL_COMPARISON_ROOT/logs/health-timing.err" \
    bash "$ROOT_DIR/jobs/training/qwen3_uccl_debug_reuse_worker.sh" --validate
complete=1
record "ACQUISITION COMPLETE;10healthyloggedupdates and5timings recorded. Input verification reused without fresh payload checks. Routing/trace/backend/comparison acceptance remains pending after all acquisitions."

