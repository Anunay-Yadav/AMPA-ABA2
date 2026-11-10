#!/usr/bin/env bash
# One independent UCCL ordinary-routing C0 start, with a node-wide Nsight owner.
#SBATCH --job-name=ampa-uccl-budget
#SBATCH --account=infra01
#SBATCH --partition=debug
#SBATCH --qos=normal
#SBATCH --constraint=gh
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --exclusive
#SBATCH --time=00:21:00
#SBATCH --no-requeue
#SBATCH --output=.runtime/slurms/slurm-%x-%j.out
#SBATCH --error=.runtime/slurms/slurm-%x-%j.err
set -euo pipefail
if [[ ${1:-} == --selfcheck ]]; then
    [[ $# == 1 && -n ${SLURM_JOB_ID:-} ]] || exit 2
    readonly CHECK_ROOT=${SLURM_SUBMIT_DIR:?}
    bash -n "$CHECK_ROOT/jobs/training/uccl_budget_sweep.sbatch"
    bash -n "$CHECK_ROOT/jobs/training/uccl_budget_sweep_worker.sh"
    /opt/venvs/megatron-bridge/bin/python - "$CHECK_ROOT" <<'PY'
import ast
from pathlib import Path
import re
import sys

root = Path(sys.argv[1])
ast.parse((root / 'profiling/run_pretrain_uccl_budget.py').read_text(), filename='run_pretrain_uccl_budget.py')
for relative in ('jobs/training/uccl_budget_sweep.sbatch', 'jobs/training/uccl_budget_sweep_worker.sh'):
    for index, body in enumerate(re.findall(r"<<'PY'\n(.*?)\nPY", (root / relative).read_text(), re.DOTALL)):
        ast.parse(body, filename=f'{relative}:heredoc-{index}')
print('uccl_budget_launcher_source_syntax_passed=true')
PY
    exit 0
fi
[[ ${SLURM_JOB_PARTITION:-} == debug && ${SLURM_JOB_NUM_NODES:-} == 4 && $# == 0 ]] || exit 2
readonly ROOT_DIR=${SLURM_SUBMIT_DIR:?}
[[ ${AMPA_UCCL_SWEEP_ID:?} =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*$ ]] || exit 2
[[ ${AMPA_UCCL_SWEEP_SMS:?} =~ ^(12|20|32|40)$ ]] || exit 2
[[ ${AMPA_UCCL_NSYS_CAP_ROOT:?} == "$ROOT_DIR"/runs/* && -f "$AMPA_UCCL_NSYS_CAP_ROOT/capabilities.json" ]] || exit 2
export AMPA_UCCL_SWEEP_ID AMPA_UCCL_SWEEP_SMS AMPA_UCCL_NSYS_CAP_ROOT
export AMPA_UCCL_VERIFIED_FIRST_ROOT=${AMPA_UCCL_VERIFIED_FIRST_ROOT:-$ROOT_DIR/runs/uccl-debug5x5-20261104T1415Z-grouped-profile}
[[ $AMPA_UCCL_VERIFIED_FIRST_ROOT == "$ROOT_DIR"/runs/* ]] || exit 2
# The historical helper retains its internal input-reuse names. The new
# acquisition/NSYS contract below identifies this separate diagnostic sweep.
export AMPA_UCCL_DEBUG_ID=$AMPA_UCCL_SWEEP_ID AMPA_UCCL_DEBUG_CASE=uccl_ordinary AMPA_UCCL_DEBUG_LEG=control
export AMPA_UCCL_COMPARISON_MODE=full AMPA_UCCL_COMPARISON_ID=$AMPA_UCCL_SWEEP_ID
export AMPA_UCCL_COMPARISON_ROOT="$ROOT_DIR/runs/$AMPA_UCCL_SWEEP_ID"
export AMPA_UCCL_SWEEP_RUN="$ROOT_DIR/runs/$AMPA_UCCL_SWEEP_ID-uccl_ordinary-control"
export AMPA_UCCL_REUSE_CONTEXT="$AMPA_UCCL_COMPARISON_ROOT/input_reuse_context.json"
export AMPA_REVIEWED_INPUTS_PROOF="$AMPA_UCCL_COMPARISON_ROOT/reviewed_inputs.json"
export AMPA_UCCL_SWEEP_SESSION_DIR=${AMPA_UCCL_SWEEP_SESSION_DIR:-$ROOT_DIR/progress/sessions/2026-11-01-uccl-ep-comparison}
export AMPA_UCCL_SWEEP_STATUS_FILE=${AMPA_UCCL_SWEEP_STATUS_FILE:-$ROOT_DIR/ai/qwen-uccl-block-sweep/iteration-01/STATUS.md}
[[ $AMPA_UCCL_SWEEP_SESSION_DIR == "$ROOT_DIR"/progress/sessions/* && $AMPA_UCCL_SWEEP_STATUS_FILE == "$ROOT_DIR"/ai/*/iteration-*/STATUS.md ]] || exit 2
export AMPA_UCCL_BUILD_ROOT="$ROOT_DIR/runs/uccl-ep-owned-build-20261101T1713Z"
export AMPA_C0="$ROOT_DIR/checkpoint/megatron/qwen3-30b-a3b-base-c0"
export AMPA_HF_REVISION=1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9
export AMPA_DATA_PATH="0.55 $ROOT_DIR/dataset/indexed/qwen3-30b-a3b-base-realistic-mix-v2/web/qwen3_30b_a3b_base_realistic_mix_web 0.15 $ROOT_DIR/dataset/indexed/qwen3-30b-a3b-base-realistic-mix-v2/multilingual/qwen3_30b_a3b_base_realistic_mix_multilingual 0.15 $ROOT_DIR/dataset/indexed/qwen3-30b-a3b-base-realistic-mix-v2/code/qwen3_30b_a3b_base_realistic_mix_code 0.15 $ROOT_DIR/dataset/indexed/qwen3-30b-a3b-base-realistic-mix-v2/math/qwen3_30b_a3b_base_realistic_mix_math"
export EFA_HOME="$AMPA_UCCL_BUILD_ROOT/not-efa"
case ${LD_LIBRARY_PATH:-} in
    /opt/amazon/efa/lib|/opt/amazon/efa/lib:*) ;;
    *) export LD_LIBRARY_PATH="/opt/amazon/efa/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
esac
export UCCL_EP_TRANSPORT=cxi FI_MR_CACHE_MONITOR=disabled CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$AMPA_UCCL_BUILD_ROOT/site:$AMPA_UCCL_BUILD_ROOT/deps:$AMPA_UCCL_BUILD_ROOT/uccl/ep/deep_ep_wrapper:$ROOT_DIR:$ROOT_DIR/profiling:$ROOT_DIR/analysis${PYTHONPATH:+:$PYTHONPATH}"
[[ ! -e $AMPA_UCCL_COMPARISON_ROOT && ! -L $AMPA_UCCL_COMPARISON_ROOT
    && ! -e $AMPA_UCCL_SWEEP_RUN && ! -L $AMPA_UCCL_SWEEP_RUN ]] || exit 2
mkdir "$AMPA_UCCL_COMPARISON_ROOT"
mkdir -p "$AMPA_UCCL_COMPARISON_ROOT"/{source,logs}
mkdir -p "$AMPA_UCCL_SWEEP_RUN"/{config,raw,metrics,analysis,plots,traces,nsight,logs/wandb}
printf '%s\n' "$SLURM_JOB_ID" > "$AMPA_UCCL_COMPARISON_ROOT/job-id"
printf 'uccl_ordinary\tcontrol\t%s\n' "$AMPA_UCCL_SWEEP_RUN" > "$AMPA_UCCL_COMPARISON_ROOT/cases.tsv"
# Snapshot current sources as records. Do not assert that drifted native
# checkouts match the historical accepted campaign, or rehash input payloads.
for relative in jobs/training/uccl_budget_sweep.sbatch jobs/training/uccl_budget_sweep_worker.sh \
    profiling/run_pretrain_uccl_budget.py profiling/run_pretrain_debug_reuse.py \
    profiling/run_pretrain_debug.py profiling/run_pretrain.py \
    tools/uccl_ep_verified_inputs_reuse.py analysis/summarize_training_evidence.py analysis/extract_step_timing.py; do
    mkdir -p "$AMPA_UCCL_COMPARISON_ROOT/source/$(dirname "$relative")"
    cp "$ROOT_DIR/$relative" "$AMPA_UCCL_COMPARISON_ROOT/source/$relative"
    sha256sum "$ROOT_DIR/$relative" >> "$AMPA_UCCL_COMPARISON_ROOT/source.sha256"
done
cp "$AMPA_UCCL_COMPARISON_ROOT/source.sha256" "$AMPA_UCCL_COMPARISON_ROOT/campaign-source.sha256"
record() {
    local note="$(date -u '+%Y-%m-%d %H:%M:%S UTC') job $SLURM_JOB_ID: $*"
    printf '%s\n' "$note" >> "$AMPA_UCCL_COMPARISON_ROOT/timeline.md"
    printf '%s\n' "$note" > "$AMPA_UCCL_COMPARISON_ROOT/CURRENT.md"
    printf '\n- %s\n' "$note" >> "$AMPA_UCCL_SWEEP_SESSION_DIR/timeline.md"
    printf '\n- %s\n' "$note" >> "$AMPA_UCCL_SWEEP_SESSION_DIR/session_state.md"
    printf '\nCompute milestone: %s\n' "$note" >> "$AMPA_UCCL_SWEEP_STATUS_FILE"
}
complete=0
on_exit() {
    local status=$?
    trap - EXIT
    if (( status != 0 || complete != 1 )); then
        record "EXIT status=$status; budget$AMPA_UCCL_SWEEP_SMS acquisition incomplete; preserve all outputs"
    fi
    exit "$status"
}
trap on_exit EXIT
record "START separate Nsight diagnostic budget$AMPA_UCCL_SWEEP_SMS on $SLURM_JOB_NODELIST; four exclusive debug nodes/21minutes,5+5; original input proof reused, actual source/placement recorded"
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
export MASTER_PORT="$((21000 + SLURM_JOB_ID % 20000))"
record "BEFORE compute one Nsight owner pernode captures all4GPUchildren, CUDAProfilerAPI measured5window, KinetoOFF"
squeue -u "$USER"
srun --mpi=pmix --cpu-bind=none --container-remap-root --environment="$ROOT_DIR/setup/gh200-cuda130-runtime.toml" \
    --nodes=4 --ntasks=4 --ntasks-per-node=1 --gpus-per-task=4 --kill-on-bad-exit=1 \
    --output="$AMPA_UCCL_COMPARISON_ROOT/logs/node-%t.out" --error="$AMPA_UCCL_COMPARISON_ROOT/logs/node-%t.err" \
    bash "$ROOT_DIR/jobs/training/uccl_budget_sweep_worker.sh" --node
export AMPA_UCCL_WORKERS_SUCCEEDED=1
record "AFTER all4Nsightowners/all16workers exited0; BEFORE small10update health and5update diagnostic timing summary"
squeue -u "$USER"
srun --mpi=pmix --cpu-bind=none --container-remap-root --environment="$ROOT_DIR/setup/gh200-cuda130-runtime.toml" \
    --nodes=1 --ntasks=1 --gpus-per-task=1 \
    --output="$AMPA_UCCL_COMPARISON_ROOT/logs/health-timing.out" --error="$AMPA_UCCL_COMPARISON_ROOT/logs/health-timing.err" \
    bash "$ROOT_DIR/jobs/training/uccl_budget_sweep_worker.sh" --validate
complete=1
record "ACQUISITION COMPLETE budget$AMPA_UCCL_SWEEP_SMS;10healthyupdates/5diagnostictimers/fourNsightreports. Full route/input/backend/metric acceptance deferred; no unprofiled throughput or hardwarecounter claim yet."
