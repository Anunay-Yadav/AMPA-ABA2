#!/usr/bin/env bash
# Acquire separate CP1 profiles and controls after the EP16 review fixes.
#SBATCH --job-name=ampa-ep16-reviewed
#SBATCH --account=infra01
#SBATCH --partition=preemptable
#SBATCH --qos=normal
#SBATCH --constraint=gh
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --no-requeue
#SBATCH --output=.runtime/slurms/slurm-%x-%j.out
#SBATCH --error=.runtime/slurms/slurm-%x-%j.err

set -euo pipefail
[[ "${SLURM_JOB_PARTITION:-}" == preemptable && "${SLURM_JOB_NUM_NODES:-}" == 4 ]] || exit 2
readonly ROOT_DIR="${SLURM_SUBMIT_DIR}"
readonly PAIR_ID="${AMPA_PAIR_ID:?}"
[[ "${PAIR_ID}" == *ep16* && "${PAIR_ID}" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
readonly CONTAINER_ENV="${ROOT_DIR}/setup/gh200-cuda130-runtime.toml"
readonly ORCHESTRATION_DIR="${ROOT_DIR}/runs/${PAIR_ID}-reviewed-orchestration"
readonly COMPARISON_DIR="${ROOT_DIR}/runs/${PAIR_ID}-reviewed-comparison"
readonly REFERENCE_RUN="${ROOT_DIR}/runs/ep16-all-steps-group4-top2-fresh-20261011T1957Z-cp1-normal-torchprof-all"
export AMPA_SESSION_DIR="${ROOT_DIR}/progress/sessions/2026-10-12-ep16-reviewed-profile"
export AMPA_TASK_STATUS_FILE="${ROOT_DIR}/ai/qwen-ep16-all-steps/iteration-02/STATUS.md"
export AMPA_REQUIRED_CP=1
[[ -f "${AMPA_SESSION_DIR}/timeline.md" && -f "${AMPA_SESSION_DIR}/session_state.md" \
    && -f "${AMPA_TASK_STATUS_FILE}" ]] || exit 2
[[ ! -e "${ORCHESTRATION_DIR}" && ! -e "${COMPARISON_DIR}" \
    && ! -e "${ROOT_DIR}/runs/${PAIR_ID}-suite" \
    && ! -e "${ROOT_DIR}/runs/${PAIR_ID}-comparison" ]] || exit 2
if compgen -G "${ROOT_DIR}/runs/${PAIR_ID}-cp[12]-*" >/dev/null; then exit 2; fi
mkdir -p "${ORCHESTRATION_DIR}"/{source,logs}

record() {
    local note="$(date -u '+%Y-%m-%d %H:%M UTC') job ${SLURM_JOB_ID}: $*"
    printf '%s\n' "${note}" >> "${ORCHESTRATION_DIR}/timeline.md"
    printf '%s\n' "${note}" > "${ORCHESTRATION_DIR}/CURRENT.md"
    printf '\n- %s\n' "${note}" >> "${AMPA_SESSION_DIR}/timeline.md"
    printf '\n- %s\n' "${note}" >> "${AMPA_SESSION_DIR}/session_state.md"
    printf '\nCompute milestone: %s\n' "${note}" >> "${AMPA_TASK_STATUS_FILE}"
}
complete=0
on_exit() {
    local status=$?
    trap - EXIT
    if (( status != 0 )); then
        record "EXIT status=${status}; reviewed comparison incomplete; preserve artifacts and inspect logs"
    elif (( complete != 1 )); then
        record "EXIT status=0 before final comparison; no final result accepted"
    fi
    exit "${status}"
}
trap on_exit EXIT
record "START separate reviewed CP1 experiment on ${SLURM_JOB_NODELIST}; container ${CONTAINER_ENV}; all four full legs share this allocation; historical captures remain separate"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export AMPA_C0="${ROOT_DIR}/checkpoint/megatron/qwen3-30b-a3b-base-c0"
export AMPA_HF_REVISION=1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9
export AMPA_DATA_PATH="0.55 ${ROOT_DIR}/dataset/indexed/qwen3-30b-a3b-base-realistic-mix-v2/web/qwen3_30b_a3b_base_realistic_mix_web 0.15 ${ROOT_DIR}/dataset/indexed/qwen3-30b-a3b-base-realistic-mix-v2/multilingual/qwen3_30b_a3b_base_realistic_mix_multilingual 0.15 ${ROOT_DIR}/dataset/indexed/qwen3-30b-a3b-base-realistic-mix-v2/code/qwen3_30b_a3b_base_realistic_mix_code 0.15 ${ROOT_DIR}/dataset/indexed/qwen3-30b-a3b-base-realistic-mix-v2/math/qwen3_30b_a3b_base_realistic_mix_math"
export AMPA_MANIFEST_CP1="${ROOT_DIR}/dataset/manifests/ep16-cache-20261011T1757Z-cp1-ep16.json"
export AMPA_MANIFEST_CP2="${ROOT_DIR}/dataset/manifests/ep16-cache-20261011T1757Z-cp2-ep16.json"
export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/profiling:${ROOT_DIR}/analysis${PYTHONPATH:+:${PYTHONPATH}}"
cp -- "${ROOT_DIR}/jobs/training/ep16_all_steps_reviewed.sbatch" "${ORCHESTRATION_DIR}/source/"
cp -- "${ROOT_DIR}/analysis/compare_ep16_recovery.py" "${ORCHESTRATION_DIR}/source/"
cp -- "${ROOT_DIR}/tools/ep16_reviewed_inputs.py" "${ORCHESTRATION_DIR}/source/"
sha256sum "${ROOT_DIR}/jobs/training/ep16_all_steps_reviewed.sbatch" \
    "${ROOT_DIR}/analysis/compare_ep16_recovery.py" > "${ORCHESTRATION_DIR}/source_sha256.txt"
(
    cd "${ROOT_DIR}"
    sha256sum tools/ep16_reviewed_inputs.py jobs/training/ep16_all_steps_group_routing.sbatch \
        dataset/manifests/ep16-cache-20261011T1757Z-cp1-ep16.json \
        dataset/manifests/ep16-cache-20261011T1757Z-cp2-ep16.json
) > "${ORCHESTRATION_DIR}/frozen_input_sha256.txt"
record "BEFORE compute reviewed source/input gate; source archive ${ORCHESTRATION_DIR}/source_sha256.txt"
squeue -u "$USER"
srun --mpi=pmix --container-remap-root --environment="${CONTAINER_ENV}" \
    --nodes=1 --ntasks=1 --gpus-per-task=1 --output="${ORCHESTRATION_DIR}/logs/reviewed-inputs.out" \
    /opt/venvs/megatron-bridge/bin/python "${ROOT_DIR}/tools/ep16_reviewed_inputs.py" \
    --root "${ROOT_DIR}" --reference-run "${REFERENCE_RUN}" \
    --manifest "${AMPA_MANIFEST_CP1}" --output-dir "${ORCHESTRATION_DIR}"
record "AFTER compute reviewed source/input gate: native/input identity and current C0 payload passed; updated AMPA sources snapshotted; actual fresh placement follows"
export AMPA_REVIEWED_INPUTS_PROOF="${ORCHESTRATION_DIR}/reviewed_inputs.json"

# Preserve the suite's strict comparison result and require the requested CP1.
record "BEFORE compute CP1-only reviewed suite: both route pilots, independent ordinary/grouped 25-step captures and controls, then all-rank event analysis"
if bash "${ROOT_DIR}/jobs/training/ep16_all_steps_group_routing.sbatch"; then
    original_status=0
else
    original_status=$?
fi
record "AFTER compute reviewed suite: status=${original_status}; inspect its strict comparison result"

# Complete capture and analysis evidence is required before descriptive output.
ready=1
for routing in normal group4-top2; do
    for leg in control torchprof-all; do
        run="${ROOT_DIR}/runs/${PAIR_ID}-cp1-${routing}-${leg}"
        [[ -f "${run}/environment.json" && -f "${run}/config/comparison_contract.json" ]] || ready=0
        if [[ "${leg}" == torchprof-all ]]; then
            for rank in {0..15}; do
                [[ -f "${run}/analysis/all_steps/rank-${rank}/summary.json" ]] || ready=0
            done
        fi
    done
done
if compgen -G "${ROOT_DIR}/runs/${PAIR_ID}-cp2-*" >/dev/null; then
    record "FAIL unexpected CP2 artifacts; only CP1 acquisition was requested; preserve evidence and inspect the suite"
    exit 2
fi
if (( ready != 1 )); then
    record "INCOMPLETE reviewed CP1 acquisition or all-step summaries; no descriptive comparison attempted"
    (( original_status != 0 )) && exit "${original_status}"
    exit 1
fi
record "BEFORE compute reviewed whole-step comparison CP1/EP16 groups4/top2; strict status=${original_status} retained"
squeue -u "$USER"
if srun --mpi=pmix --container-remap-root --environment="${CONTAINER_ENV}" \
    --nodes=1 --ntasks=1 --gpus-per-task=1 --output="${ORCHESTRATION_DIR}/logs/reviewed-comparison.out" \
    /opt/venvs/megatron-bridge/bin/python "${ROOT_DIR}/analysis/compare_ep16_recovery.py" \
    --normal-control "${ROOT_DIR}/runs/${PAIR_ID}-cp1-normal-control" \
    --normal-trace "${ROOT_DIR}/runs/${PAIR_ID}-cp1-normal-torchprof-all" \
    --grouped-control "${ROOT_DIR}/runs/${PAIR_ID}-cp1-group4-top2-control" \
    --grouped-trace "${ROOT_DIR}/runs/${PAIR_ID}-cp1-group4-top2-torchprof-all" \
    --same-allocation --reviewed-acquisition \
    --orchestration-sources "${ORCHESTRATION_DIR}/source_sha256.txt" \
    --output "${COMPARISON_DIR}"; then
    complete=1
    record "COMPLETE reviewed whole-step comparison ${COMPARISON_DIR}; strict status=${original_status}; inspect actual parity/overhead limits before conclusions"
else
    comparison_status=$?
    record "AFTER compute reviewed whole-step comparison: failed status=${comparison_status}; strict status=${original_status}; no failure waived"
    (( original_status != 0 )) && exit "${original_status}"
    exit "${comparison_status}"
fi
