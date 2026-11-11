#!/usr/bin/env bash
# One independent overlap case. The root agent owns sequential submission.
#SBATCH --job-name=ampa-ep-overlap
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
[[ $# == 0 && ${SLURM_JOB_PARTITION:-} == debug
    && ${SLURM_JOB_NUM_NODES:-} == 4 && ${SLURM_JOB_ID:-} =~ ^[0-9]+$ ]] || exit 2
readonly ROOT_DIR=${SLURM_SUBMIT_DIR:?}
cd "$ROOT_DIR"
[[ ${AMPA_OVERLAP_ID:?} =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*$
    && ${AMPA_OVERLAP_CASE:?} =~ ^(ordinary|grouped|uccl_ordinary|uccl)$
    && ${AMPA_OVERLAP_CAPTURE:?} =~ ^(off|torch|nsys)$ ]] || exit 2
export AMPA_OVERLAP_BUDGET=${AMPA_OVERLAP_BUDGET:-20}
export AMPA_OVERLAP_STEPS=${AMPA_OVERLAP_STEPS:-10}
[[ $AMPA_OVERLAP_BUDGET =~ ^(12|20|32|40)$ && $AMPA_OVERLAP_STEPS =~ ^(2|10)$ ]] || exit 2
if [[ $AMPA_OVERLAP_CAPTURE == nsys ]]; then
    [[ $AMPA_OVERLAP_CASE == uccl_ordinary && ${AMPA_OVERLAP_NSYS_CAP_ROOT:?} == "$ROOT_DIR"/runs/*
        && -f $AMPA_OVERLAP_NSYS_CAP_ROOT/capabilities.json
        && -f $AMPA_OVERLAP_NSYS_CAP_ROOT/commands.json ]] || exit 2
    export AMPA_OVERLAP_NSYS_CAP_ROOT
else
    [[ $AMPA_OVERLAP_BUDGET == 20 ]] || exit 2
fi
: "${AMPA_OVERLAP_SOURCE_MANIFEST:?}" "${AMPA_OVERLAP_SOURCE_SHA256:?}"
[[ $AMPA_OVERLAP_SOURCE_MANIFEST == "$ROOT_DIR"/*
    && -f $AMPA_OVERLAP_SOURCE_MANIFEST && ! -L $AMPA_OVERLAP_SOURCE_MANIFEST
    && $AMPA_OVERLAP_SOURCE_SHA256 =~ ^[0-9a-f]{64}$ ]] || exit 2
read -r manifest_digest _ < <(sha256sum "$AMPA_OVERLAP_SOURCE_MANIFEST")
[[ $manifest_digest == "$AMPA_OVERLAP_SOURCE_SHA256" ]] || exit 2
# These checks bind current source, not the historical immutable payload.
sha256sum --check --strict "$AMPA_OVERLAP_SOURCE_MANIFEST"
export AMPA_OVERLAP_ROOT="$ROOT_DIR/runs/$AMPA_OVERLAP_ID"
export AMPA_OVERLAP_RUN="$ROOT_DIR/runs/$AMPA_OVERLAP_ID-training"
export AMPA_OVERLAP_SESSION_DIR=${AMPA_OVERLAP_SESSION_DIR:-$ROOT_DIR/progress/sessions/2026-11-10-qwen-ep-overlap}
export AMPA_OVERLAP_STATUS_FILE=${AMPA_OVERLAP_STATUS_FILE:-$ROOT_DIR/ai/qwen-ep-overlap-campaign/iteration-01/STATUS.md}
[[ $AMPA_OVERLAP_SESSION_DIR == "$ROOT_DIR"/progress/sessions/*
    && -d $AMPA_OVERLAP_SESSION_DIR && -f $AMPA_OVERLAP_SESSION_DIR/session_state.md
    && $AMPA_OVERLAP_STATUS_FILE == "$ROOT_DIR"/ai/*/iteration-*/STATUS.md
    && -f $AMPA_OVERLAP_STATUS_FILE ]] || exit 2
[[ ! -e $AMPA_OVERLAP_ROOT && ! -L $AMPA_OVERLAP_ROOT
    && ! -e $AMPA_OVERLAP_RUN && ! -L $AMPA_OVERLAP_RUN ]] || exit 2
mkdir "$AMPA_OVERLAP_ROOT" "$AMPA_OVERLAP_RUN"
mkdir -p "$AMPA_OVERLAP_ROOT"/{source,logs} "$AMPA_OVERLAP_RUN"/{config,raw,metrics,analysis,plots,traces,nsight,logs/wandb}
cp "$AMPA_OVERLAP_SOURCE_MANIFEST" "$AMPA_OVERLAP_ROOT/campaign-source.sha256"
while IFS= read -r line; do
    source_path=${line:66}
    [[ ${line:0:64} =~ ^[0-9a-f]{64}$ && ${line:64:2} == '  '
        && $source_path == "$ROOT_DIR"/* && -f $source_path ]] || exit 2
    relative=${source_path#"$ROOT_DIR/"}
    [[ $relative != ../* && $relative != */../* ]] || exit 2
    mkdir -p "$AMPA_OVERLAP_ROOT/source/$(dirname "$relative")"
    cp "$source_path" "$AMPA_OVERLAP_ROOT/source/$relative"
done < "$AMPA_OVERLAP_SOURCE_MANIFEST"
printf '%s\n' "$SLURM_JOB_ID" > "$AMPA_OVERLAP_ROOT/job-id"
printf '%s\t%s\t%s\n' "$AMPA_OVERLAP_CASE" "$AMPA_OVERLAP_CAPTURE" "$AMPA_OVERLAP_RUN" > "$AMPA_OVERLAP_ROOT/cases.tsv"
export AMPA_UCCL_BUILD_ROOT="$ROOT_DIR/runs/uccl-ep-owned-build-20261101T1713Z"
export AMPA_UCCL_VERIFIED_FIRST_ROOT="$ROOT_DIR/runs/uccl-debug5x5-20261104T1415Z-grouped-profile"
export AMPA_UCCL_REUSE_CONTEXT="$AMPA_OVERLAP_ROOT/input_reuse_context.json"
export AMPA_REVIEWED_INPUTS_PROOF="$AMPA_OVERLAP_ROOT/reviewed_inputs.json"
export AMPA_C0="$ROOT_DIR/checkpoint/megatron/qwen3-30b-a3b-base-c0"
export AMPA_HF_REVISION=1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9
export AMPA_DATA_MANIFEST="$ROOT_DIR/dataset/manifests/ep16-cache-20261011T1757Z-cp1-ep16.json"
export AMPA_DATA_PATH="0.55 $ROOT_DIR/dataset/indexed/qwen3-30b-a3b-base-realistic-mix-v2/web/qwen3_30b_a3b_base_realistic_mix_web 0.15 $ROOT_DIR/dataset/indexed/qwen3-30b-a3b-base-realistic-mix-v2/multilingual/qwen3_30b_a3b_base_realistic_mix_multilingual 0.15 $ROOT_DIR/dataset/indexed/qwen3-30b-a3b-base-realistic-mix-v2/code/qwen3_30b_a3b_base_realistic_mix_code 0.15 $ROOT_DIR/dataset/indexed/qwen3-30b-a3b-base-realistic-mix-v2/math/qwen3_30b_a3b_base_realistic_mix_math"
export EFA_HOME="$AMPA_UCCL_BUILD_ROOT/not-efa"
export UCCL_EP_TRANSPORT=cxi CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONDONTWRITEBYTECODE=1
export CUDA_DEVICE_MAX_CONNECTIONS=32
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$AMPA_UCCL_BUILD_ROOT/site:$AMPA_UCCL_BUILD_ROOT/deps:$AMPA_UCCL_BUILD_ROOT/uccl/ep/deep_ep_wrapper:$ROOT_DIR:$ROOT_DIR/profiling:$ROOT_DIR/analysis${PYTHONPATH:+:$PYTHONPATH}"
case ${LD_LIBRARY_PATH:-} in
    /opt/amazon/efa/lib|/opt/amazon/efa/lib:*) ;;
    *) export LD_LIBRARY_PATH="/opt/amazon/efa/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
esac
record() {
    local note="$(date -u '+%Y-%m-%d %H:%M:%S UTC') job $SLURM_JOB_ID: $*"
    printf '%s\n' "$note" >> "$AMPA_OVERLAP_ROOT/timeline.md"
    printf '%s\n' "$note" > "$AMPA_OVERLAP_ROOT/CURRENT.md"
    printf '\n- %s\n' "$note" >> "$AMPA_OVERLAP_SESSION_DIR/timeline.md"
    printf '\nCompute milestone: %s\n' "$note" >> "$AMPA_OVERLAP_SESSION_DIR/session_state.md"
    printf '\nCompute milestone: %s\n' "$note" >> "$AMPA_OVERLAP_STATUS_FILE"
}
check_queue() {
    local queue requested
    queue=$(squeue --array -u "$USER" --noheader --format='%i %P %T %D %R')
    printf '%s\n' "$queue"
    requested=$(awk '{total += $4} END {print total+0}' <<< "$queue")
    (( requested <= 4 )) || { record "STOP global node cap exceeded: requested/allocated=$requested"; return 2; }
}
complete=0
on_exit() {
    local status=$?
    trap - EXIT
    if (( status != 0 || complete != 1 )); then
        (( status != 0 )) || status=1
        record "EXIT status=$status; acquisition incomplete; preserve all outputs"
    fi
    exit "$status"
}
trap on_exit EXIT
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
export MASTER_PORT="$((21000 + SLURM_JOB_ID % 20000))"
record "START $AMPA_OVERLAP_CASE/$AMPA_OVERLAP_CAPTURE budget=$AMPA_OVERLAP_BUDGET steps=$AMPA_OVERLAP_STEPS nodes=$SLURM_JOB_NODELIST container=setup/gh200-cuda130-runtime.toml source=$AMPA_OVERLAP_SOURCE_SHA256 CUDA_DEVICE_MAX_CONNECTIONS=32"
record "BEFORE compute prepare explicit historical input-reuse records, then independent C0 training; no repeated bulk payload checks"
check_queue
srun --mpi=pmix --cpu-bind=none --container-remap-root --environment="$ROOT_DIR/setup/gh200-cuda130-runtime.toml" \
    --nodes=4 --ntasks=4 --ntasks-per-node=1 --gpus-per-task=4 --kill-on-bad-exit=1 \
    --output="$AMPA_OVERLAP_ROOT/logs/node-%t.out" --error="$AMPA_OVERLAP_ROOT/logs/node-%t.err" \
    bash "$ROOT_DIR/jobs/training/ep_overlap_worker.sh" --node
export AMPA_OVERLAP_WORKERS_SUCCEEDED=1
record "AFTER compute all16 workers exited0; BEFORE compute small health and per-update timing summaries"
check_queue
srun --mpi=pmix --cpu-bind=none --container-remap-root --environment="$ROOT_DIR/setup/gh200-cuda130-runtime.toml" \
    --nodes=1 --ntasks=1 --gpus-per-task=1 \
    --output="$AMPA_OVERLAP_ROOT/logs/health-timing.out" --error="$AMPA_OVERLAP_ROOT/logs/health-timing.err" \
    bash "$ROOT_DIR/jobs/training/ep_overlap_worker.sh" --validate
complete=1
record "ACQUISITION COMPLETE; health and timing recorded; historical payload verification reused. Routing/backend/trace/comparison acceptance remains pending separate validation."
