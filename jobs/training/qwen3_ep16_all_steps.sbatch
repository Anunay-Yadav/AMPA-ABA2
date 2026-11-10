#!/usr/bin/env bash
# Reviewed all-step launcher adapted to four nodes without training-source edits.
#SBATCH --job-name=ampa-qwen-ep16-all
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
readonly CONTAINER_ENV="${ROOT_DIR}/setup/gh200-cuda130-runtime.toml"
: "${AMPA_C0:?}" "${AMPA_HF_REVISION:?}" "${AMPA_DATA_PATH:?}" "${AMPA_DATA_MANIFEST:?}"
readonly RUN_ID="${AMPA_RUN_ID:?}"
readonly RUN_DIR="${ROOT_DIR}/runs/${RUN_ID}"
readonly TOPOLOGY="${AMPA_TOPOLOGY_VARIANT:?}"
readonly ALL_STEPS="${AMPA_TORCH_PROFILE_ALL_STEPS:?}"
readonly SMOKE_STEPS="${AMPA_SMOKE_STEPS:-0}"
readonly PROFILE_RANKS=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
[[ "${TOPOLOGY}" == cp1_ep16 || "${TOPOLOGY}" == cp2_ep16 ]] || exit 2
[[ "${RUN_ID}" == *ep16* && "${RUN_ID}" == *"${TOPOLOGY%%_*}"* ]] || exit 2
[[ "${ALL_STEPS}" == 0 || "${ALL_STEPS}" == 1 ]] || exit 2
[[ "${SMOKE_STEPS}" == 0 || ( "${SMOKE_STEPS}" == 2 && "${ALL_STEPS}" == 1 ) ]] || exit 2
[[ "${PYTORCH_CUDA_ALLOC_CONF:-}" == expandable_segments:True ]] || exit 2

container_train() {
    export RANK="${SLURM_PROCID}" WORLD_SIZE="${SLURM_NTASKS}" LOCAL_RANK=0
    [[ "${WORLD_SIZE}" == 16 ]] || exit 2
    local overlay="${WORKSPACE}/.runtime/ampa-overlay/${SLURM_JOB_ID}/${RUN_ID}"
    local ready="${overlay}/ready-${SLURM_NODEID}" failed="${overlay}/failed-${SLURM_NODEID}"
    mkdir -p "${overlay}"
    if [[ "${SLURM_LOCALID}" == 0 ]]; then
        if uv pip install --python /opt/venvs/megatron-bridge/bin/python --no-deps --no-build-isolation \
            --editable "${WORKSPACE}/Megatron-LM" --editable "${WORKSPACE}/Megatron-Bridge"; then
            touch "${ready}"
        else
            touch "${failed}"
            return 1
        fi
    else
        while [[ ! -e "${ready}" && ! -e "${failed}" ]]; do sleep 1; done
        [[ -e "${ready}" ]] || return 1
    fi
    local args=(--sample-fingerprints --routing-audit --audit-num-groups 4)
    if [[ "${ALL_STEPS}" == 1 ]]; then
        args+=(--torch-profile-all-steps --torch-profile-ranks "${PROFILE_RANKS}")
    fi
    if [[ "${SMOKE_STEPS}" == 2 ]]; then args+=(--smoke-steps 2); fi
    if [[ -n "${AMPA_ROUTER_NUM_GROUPS:-}" ]]; then
        [[ "${AMPA_ROUTER_NUM_GROUPS}" == 4 && "${AMPA_ROUTER_GROUP_TOPK:-}" == 2 ]] || exit 2
        args+=(--router-num-groups 4 --router-group-topk 2)
    fi
    cd "${WORKSPACE}"
    PYTHONPATH="${WORKSPACE}:${WORKSPACE}/profiling${PYTHONPATH:+:${PYTHONPATH}}" \
        /opt/venvs/megatron-bridge/bin/python "${WORKSPACE}/profiling/run_pretrain.py" \
        --run-dir "${RUN_DIR}" --pretrained-checkpoint "${AMPA_C0}" \
        --hf-revision "${AMPA_HF_REVISION}" --data-mode indexed \
        --dataset-data-path "${AMPA_DATA_PATH}" --dataset-manifest "${AMPA_DATA_MANIFEST}" \
        --topology-variant "${TOPOLOGY}" --lightweight-profiling off --recompute-mode none \
        --sequence-length 4096 --micro-batch-size 1 --global-batch-size 64 \
        --warmup-steps 5 --measured-steps 20 --log-interval 1 --dataset-sampler-seed 1234 \
        "${args[@]}"
}

if [[ "${1:-}" == --container-train ]]; then container_train; exit 0; fi
[[ -z "${1:-}" ]] || exit 2
[[ ! -e "${RUN_DIR}" && -f "${AMPA_C0}/ampa_c0_manifest.json" && -f "${AMPA_DATA_MANIFEST}" ]] || exit 2
mkdir -p "${RUN_DIR}"/{config,raw,metrics,analysis,plots,traces,logs/wandb}
cp -- "${AMPA_DATA_MANIFEST}" "${RUN_DIR}/config/dataset_manifest.json"
cp -- "${AMPA_C0}/ampa_c0_manifest.json" "${RUN_DIR}/config/c0_manifest.json"
cp -- "${ROOT_DIR}/configs/model/qwen3_30b_a3b_base.json" "${RUN_DIR}/config/model_manifest.json"
export AMPA_DATA_MANIFEST="${RUN_DIR}/config/dataset_manifest.json"
export MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)"
export MASTER_PORT="$((20000 + SLURM_JOB_ID % 20000))"
squeue -u "$USER"
srun --mpi=pmix --container-remap-root --environment="${CONTAINER_ENV}" \
    --nodes=4 --ntasks=16 --ntasks-per-node=4 --gpus-per-task=1 \
    --kill-on-bad-exit=1 --output="${RUN_DIR}/logs/rank-%t.out" \
    bash "${ROOT_DIR}/jobs/training/qwen3_ep16_all_steps.sbatch" --container-train
if [[ "${SMOKE_STEPS}" == 0 ]]; then
    squeue -u "$USER"
    srun --mpi=pmix --container-remap-root --environment="${CONTAINER_ENV}" \
        --nodes=1 --ntasks=1 --gpus-per-task=1 --output="${RUN_DIR}/logs/timing.out" \
        /opt/venvs/megatron-bridge/bin/python "${ROOT_DIR}/analysis/extract_step_timing.py" \
        --run-dir "${RUN_DIR}" --warmup-steps 5 --measured-steps 20 --log-interval 1 \
        --global-batch-size 64 --sequence-length 4096
fi
