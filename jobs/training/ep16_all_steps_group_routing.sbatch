#!/usr/bin/env bash
# CP1 memory pilots gate four independent-C0 matched acquisition legs.
#SBATCH --job-name=ampa-ep16-group4-top2
#SBATCH --account=infra01
#SBATCH --partition=preemptable
#SBATCH --qos=normal
#SBATCH --constraint=gh
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --no-requeue
#SBATCH --output=.runtime/slurms/slurm-%x-%j.out
#SBATCH --error=.runtime/slurms/slurm-%x-%j.err

set -euo pipefail
[[ "${SLURM_JOB_PARTITION:-}" == preemptable && "${SLURM_JOB_NUM_NODES:-}" == 4 ]] || exit 2
readonly ROOT_DIR="${SLURM_SUBMIT_DIR}"
readonly PAIR_ID="${AMPA_PAIR_ID:?}"
readonly CONTAINER_ENV="${ROOT_DIR}/setup/gh200-cuda130-runtime.toml"
readonly SESSION_DIR="${AMPA_SESSION_DIR:-${ROOT_DIR}/progress/sessions/2026-09-24-cp2-deep-profile}"
readonly TASK_STATUS_FILE="${AMPA_TASK_STATUS_FILE:-${ROOT_DIR}/ai/qwen-ep16-all-steps/iteration-01/STATUS.md}"
readonly REQUIRED_CP="${AMPA_REQUIRED_CP:-}"
[[ -z "${REQUIRED_CP}" || "${REQUIRED_CP}" == 1 ]] || exit 2
: "${AMPA_MANIFEST_CP1:?}" "${AMPA_MANIFEST_CP2:?}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

record() {
    local note="$(date -u '+%Y-%m-%d %H:%M UTC') job ${SLURM_JOB_ID}: $*"
    printf '%s\n' "${note}" >> "${ROOT_DIR}/runs/${PAIR_ID}-suite/timeline.md"
    printf '\n- %s\n' "${note}" >> "${SESSION_DIR}/timeline.md"
    printf '%s\n' "${note}" > "${ROOT_DIR}/runs/${PAIR_ID}-suite/CURRENT.md"
    printf '\n- %s\n' "${note}" >> "${SESSION_DIR}/session_state.md"
    printf '\nCompute milestone: %s\n' "${note}" >> "${TASK_STATUS_FILE}"
}
mkdir -p "${ROOT_DIR}/runs/${PAIR_ID}-suite"
trap 'record "EXIT code=$?; inspect logs and CURRENT.md before resuming"' EXIT
record "START four nodes ${SLURM_JOB_NODELIST}; container ${CONTAINER_ENV}; CP1 pilots before full acquisition"

route_config() {
    unset AMPA_ROUTER_NUM_GROUPS AMPA_ROUTER_GROUP_TOPK
    [[ "$1" == normal ]] || export AMPA_ROUTER_NUM_GROUPS=4 AMPA_ROUTER_GROUP_TOPK=2
}
cp_config() {
    export AMPA_TOPOLOGY_VARIANT="cp${1}_ep16"
    if [[ "$1" == 1 ]]; then export AMPA_DATA_MANIFEST="${AMPA_MANIFEST_CP1}";
    else export AMPA_DATA_MANIFEST="${AMPA_MANIFEST_CP2}"; fi
}
pilots() {
    local cp="$1" routing run status
    cp_config "${cp}"
    export AMPA_SMOKE_STEPS=2 AMPA_TORCH_PROFILE_ALL_STEPS=1
    for routing in normal group4-top2; do
        route_config "${routing}"
        export AMPA_RUN_ID="${PAIR_ID}-cp${cp}-${routing}-smoke"
        run="${ROOT_DIR}/runs/${AMPA_RUN_ID}"
        record "BEFORE compute ${AMPA_RUN_ID}: two real updates, all16rank traces"
        if bash "${ROOT_DIR}/jobs/training/qwen3_ep16_all_steps.sbatch"; then status=0; else status=$?; fi
        record "AFTER compute ${AMPA_RUN_ID}: trainer status=${status}"
        if (( status != 0 )); then
            if rg -q 'CUDA out of memory|torch\.OutOfMemoryError' "${run}/logs"; then return 42; fi
            return "${status}"
        fi
        squeue -u "$USER"
        srun --mpi=pmix --container-remap-root --environment="${CONTAINER_ENV}" \
            --nodes=1 --ntasks=1 --gpus-per-task=1 \
            /opt/venvs/megatron-bridge/bin/python "${ROOT_DIR}/analysis/check_ep16_smoke.py" --run-dir "${run}" || return 1
        record "HEALTH ${AMPA_RUN_ID}: two healthy updates/all16rank files; memory feasible"
    done
}

selected_cp=1
if pilots 1; then
    record "CP1 selected: both routes passed two real updates without memory failure"
else
    pilot_status=$?
    [[ "${pilot_status}" == 42 ]] || exit "${pilot_status}"
    if [[ "${REQUIRED_CP}" == 1 ]]; then
        record "CP1 memory failure retained; requested CP1 acquisition stopped, CP2 fallback disabled; inspect pilot logs before another allocation"
        exit 2
    fi
    selected_cp=2
    record "CP1 memory failure retained; testing both modes with CP2"
    pilots 2
    record "CP2 selected after CP1 memory failure; both routes passed two real updates"
fi

for routing in normal group4-top2; do
    export AMPA_RUN_DIR="${ROOT_DIR}/runs/${PAIR_ID}-cp${selected_cp}-${routing}-smoke"
    record "BEFORE compute pilot event validation ${AMPA_RUN_DIR}"
    squeue -u "$USER"
    srun --mpi=pmix --container-remap-root --environment="${CONTAINER_ENV}" \
        --nodes=4 --ntasks=16 --ntasks-per-node=4 --gpus-per-task=1 --kill-on-bad-exit=1 \
        --output="${AMPA_RUN_DIR}/logs/all-step-analysis-rank-%t.out" \
        bash -c 'exec /opt/venvs/megatron-bridge/bin/python "$WORKSPACE/analysis/analyze_all_step_trace.py" --run-dir "$AMPA_RUN_DIR" --rank "$SLURM_PROCID" --steps 2'
    record "AFTER compute pilot event validation ${AMPA_RUN_DIR}: passed"
done

export AMPA_SMOKE_STEPS=0
cp_config "${selected_cp}"
# Both requested traces precede timing controls, all on the same physical GPUs.
for leg in torchprof-all control; do
    export AMPA_TORCH_PROFILE_ALL_STEPS=0
    [[ "${leg}" == control ]] || export AMPA_TORCH_PROFILE_ALL_STEPS=1
    for routing in normal group4-top2; do
        route_config "${routing}"
        export AMPA_RUN_ID="${PAIR_ID}-cp${selected_cp}-${routing}-${leg}"
        record "BEFORE compute ${AMPA_RUN_ID}: independent C0 start,25 real updates"
        bash "${ROOT_DIR}/jobs/training/qwen3_ep16_all_steps.sbatch"
        record "AFTER compute ${AMPA_RUN_ID}: trainer and timing extraction passed"
    done
done
for routing in normal group4-top2; do
    export AMPA_RUN_DIR="${ROOT_DIR}/runs/${PAIR_ID}-cp${selected_cp}-${routing}-torchprof-all"
    record "BEFORE compute all-step validation ${AMPA_RUN_DIR}"
    squeue -u "$USER"
    srun --mpi=pmix --container-remap-root --environment="${CONTAINER_ENV}" \
        --nodes=4 --ntasks=16 --ntasks-per-node=4 --gpus-per-task=1 --kill-on-bad-exit=1 \
        --output="${AMPA_RUN_DIR}/logs/all-step-analysis-rank-%t.out" \
        bash -c 'exec /opt/venvs/megatron-bridge/bin/python "$WORKSPACE/analysis/analyze_all_step_trace.py" --run-dir "$AMPA_RUN_DIR" --rank "$SLURM_PROCID"'
    record "AFTER compute all-step validation ${AMPA_RUN_DIR}: passed"
done
record "BEFORE compute matched comparison cp${selected_cp}/EP16 groups4/top2"
squeue -u "$USER"
srun --mpi=pmix --container-remap-root --environment="${CONTAINER_ENV}" \
    --nodes=1 --ntasks=1 --gpus-per-task=1 \
    /opt/venvs/megatron-bridge/bin/python "${ROOT_DIR}/analysis/compare_all_step_profiles.py" \
    --runs-root "${ROOT_DIR}/runs" --pair-id "${PAIR_ID}" --cp "${selected_cp}" \
    --router-num-groups 4 --router-group-topk 2
record "COMPLETE matched comparison ${ROOT_DIR}/runs/${PAIR_ID}-comparison"
