#!/usr/bin/env bash
# Allocated source, environment, native configuration, and focused observer QA.
#SBATCH --job-name=ampa-budget-off-qa
#SBATCH --account=infra01
#SBATCH --partition=debug
#SBATCH --qos=normal
#SBATCH --constraint=gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --no-requeue
#SBATCH --output=.runtime/slurms/slurm-%x-%j.out
#SBATCH --error=.runtime/slurms/slurm-%x-%j.err
set -euo pipefail
[[ ${SLURM_JOB_PARTITION:-} == debug && ${SLURM_JOB_NUM_NODES:-} == 1
    && ${SLURM_JOB_ID:-} =~ ^[0-9]+$ ]] || exit 2
readonly ROOT_DIR=${SLURM_SUBMIT_DIR:?}
cd "$ROOT_DIR"
export AMPA_OVERLAP_ID=${AMPA_BUDGET_PREFLIGHT_ID:?}
export AMPA_OVERLAP_SOURCE_MANIFEST=${AMPA_BUDGET_SOURCE_MANIFEST:?}
export AMPA_OVERLAP_SOURCE_SHA256=${AMPA_BUDGET_SOURCE_SHA256:?}
export AMPA_UCCL_BUILD_ROOT="$ROOT_DIR/runs/uccl-ep-owned-build-20261101T1713Z"
export EFA_HOME="$AMPA_UCCL_BUILD_ROOT/not-efa"
export FI_MR_CACHE_MONITOR=disabled UCCL_EP_TRANSPORT=cxi CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_DEVICE_MAX_CONNECTIONS=32
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/profiling:$ROOT_DIR/analysis:$ROOT_DIR/Megatron-LM:$ROOT_DIR/Megatron-Bridge/src:$AMPA_UCCL_BUILD_ROOT/site:$AMPA_UCCL_BUILD_ROOT/deps:$AMPA_UCCL_BUILD_ROOT/uccl/ep/deep_ep_wrapper${PYTHONPATH:+:$PYTHONPATH}"
case ${LD_LIBRARY_PATH:-} in
    /opt/amazon/efa/lib|/opt/amazon/efa/lib:*) ;;
    *) export LD_LIBRARY_PATH="/opt/amazon/efa/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
esac
if [[ ${1:-} == --inside-container ]]; then
    [[ $# == 1 ]] || exit 2
    args=(--output-dir "$AMPA_OVERLAP_ROOT" --source-manifest "$AMPA_OVERLAP_ROOT/source.sha256"
        --source-sha256 "$AMPA_OVERLAP_SOURCE_SHA256")
    exec timeout --signal=TERM --kill-after=5s 540s /opt/venvs/megatron-bridge/bin/python \
        "$ROOT_DIR/tools/ep_overlap_budget_preflight.py" "${args[@]}"
fi
[[ $# == 0 && ${AMPA_OVERLAP_ID:?} =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*$ ]] || exit 2
: "${AMPA_OVERLAP_SOURCE_MANIFEST:?}" "${AMPA_OVERLAP_SOURCE_SHA256:?}"
[[ $AMPA_OVERLAP_SOURCE_MANIFEST == "$ROOT_DIR"/* && -f $AMPA_OVERLAP_SOURCE_MANIFEST
    && ! -L $AMPA_OVERLAP_SOURCE_MANIFEST && $AMPA_OVERLAP_SOURCE_SHA256 =~ ^[0-9a-f]{64}$ ]] || exit 2
export AMPA_OVERLAP_ROOT="$ROOT_DIR/runs/$AMPA_OVERLAP_ID"
export AMPA_OVERLAP_SESSION_DIR=${AMPA_OVERLAP_SESSION_DIR:-$ROOT_DIR/progress/sessions/2026-11-10-qwen-ep-overlap}
export AMPA_OVERLAP_STATUS_FILE=${AMPA_OVERLAP_STATUS_FILE:-$ROOT_DIR/ai/qwen-uccl-unprofiled-budget-sweep/iteration-01/STATUS.md}
[[ $AMPA_OVERLAP_SESSION_DIR == "$ROOT_DIR"/progress/sessions/* && -d $AMPA_OVERLAP_SESSION_DIR
    && -f $AMPA_OVERLAP_SESSION_DIR/session_state.md
    && $AMPA_OVERLAP_STATUS_FILE == "$ROOT_DIR"/ai/*/iteration-*/STATUS.md
    && -f $AMPA_OVERLAP_STATUS_FILE
    && ! -e $AMPA_OVERLAP_ROOT && ! -L $AMPA_OVERLAP_ROOT ]] || exit 2
mkdir "$AMPA_OVERLAP_ROOT"
mkdir "$AMPA_OVERLAP_ROOT/logs"
cp "$AMPA_OVERLAP_SOURCE_MANIFEST" "$AMPA_OVERLAP_ROOT/source.sha256"
printf '%s\n' "$SLURM_JOB_ID" > "$AMPA_OVERLAP_ROOT/job-id"
record() {
    local note="$(date -u '+%Y-%m-%d %H:%M:%S UTC') job $SLURM_JOB_ID: $*"
    printf '%s\n' "$note" >> "$AMPA_OVERLAP_ROOT/timeline.md"
    printf '%s\n' "$note" > "$AMPA_OVERLAP_ROOT/CURRENT.md"
    printf '\n- %s\n' "$note" >> "$AMPA_OVERLAP_SESSION_DIR/timeline.md"
    printf '\nCompute milestone: %s\n' "$note" >> "$AMPA_OVERLAP_SESSION_DIR/session_state.md"
    printf '\nCompute milestone: %s\n' "$note" >> "$AMPA_OVERLAP_STATUS_FILE"
}
on_exit() {
    local status=$?
    trap - EXIT
    record "AFTER unprofiled-budget preflight status=$status node=$SLURM_JOB_NODELIST container=setup/gh200-cuda130-runtime.toml; no training executed; next inspect source/config/native bounds before paired pilot"
    printf 'exit_code=%s\njob=%s\nfinished_utc=%s\n' "$status" "$SLURM_JOB_ID" "$(date -u +%FT%TZ)" > "$AMPA_OVERLAP_ROOT/exit.txt"
    exit "$status"
}
trap on_exit EXIT
record "BEFORE allocated unprofiled-budget preflight node=$SLURM_JOB_NODELIST container=setup/gh200-cuda130-runtime.toml source=$AMPA_OVERLAP_SOURCE_SHA256 CUDA_DEVICE_MAX_CONNECTIONS=32; native launch bounds and OFF-only source/configuration QA, no training"
queue=$(squeue --array -u "$USER" --noheader --format='%i %P %T %D %R')
printf '%s\n' "$queue"
requested=$(awk '{total += $4} END {print total+0}' <<< "$queue")
(( requested <= 4 )) || exit 2
srun --mpi=pmix --cpu-bind=none --container-remap-root --environment="$ROOT_DIR/setup/gh200-cuda130-runtime.toml" \
    --nodes=1 --ntasks=1 --gpus-per-task=1 \
    --output="$AMPA_OVERLAP_ROOT/logs/checks.out" --error="$AMPA_OVERLAP_ROOT/logs/checks.err" \
    bash "$ROOT_DIR/jobs/smoke/ep_overlap_budget_preflight.sbatch" --inside-container
