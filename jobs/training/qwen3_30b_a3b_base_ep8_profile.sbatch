#!/usr/bin/env bash
# Standard AMPA continued-pretraining measurement job.  Submit only after
# inspecting `squeue -u "$USER"`; this allocation consumes the full two-node cap.

#SBATCH --job-name=ampa-qwen3-base-ep8
#SBATCH --account=infra01
#SBATCH --partition=preemptable
#SBATCH --qos=normal
#SBATCH --constraint=gh
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=.runtime/slurms/slurm-%x-%j.out
#SBATCH --error=.runtime/slurms/slurm-%x-%j.err

set -euo pipefail

[[ "${SLURM_JOB_PARTITION:-}" == preemptable ]] || {
    echo "AMPA jobs must run on the preemptable partition." >&2
    exit 2
}

readonly ROOT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)}"
readonly CONTAINER_ENV="${ROOT_DIR}/setup/gh200-cuda130-runtime.toml"
: "${AMPA_C0:?Set AMPA_C0 to the pinned native C0 directory.}"
: "${AMPA_HF_REVISION:?Set AMPA_HF_REVISION to the immutable Hugging Face revision used for C0.}"

readonly DATA_MODE="${AMPA_DATA_MODE:-indexed}"
readonly PROFILE_MODE="${AMPA_PROFILE_MODE:-on}"
readonly NSYS_CAPTURE_STEPS="${AMPA_NSYS_CAPTURE_STEPS:-0}"
readonly TORCH_PROFILE_STEPS="${AMPA_TORCH_PROFILE_STEPS:-0}"
readonly TORCH_PROFILE_RANKS="${AMPA_TORCH_PROFILE_RANKS:-0}"
readonly SAMPLE_FINGERPRINTS="${AMPA_SAMPLE_FINGERPRINTS:-0}"
readonly TOPOLOGY_VARIANT="${AMPA_TOPOLOGY_VARIANT:-cp1_ep8}"
readonly RECOMPUTE_MODE="${AMPA_RECOMPUTE_MODE:-none}"
readonly SEQUENCE_LENGTH="${AMPA_SEQUENCE_LENGTH:-4096}"
readonly MICRO_BATCH_SIZE="${AMPA_MICRO_BATCH_SIZE:-1}"
readonly GLOBAL_BATCH_SIZE="${AMPA_GLOBAL_BATCH_SIZE:-$((MICRO_BATCH_SIZE * 64))}"
readonly WARMUP_STEPS="${AMPA_WARMUP_STEPS:-5}"
readonly MEASURED_STEPS="${AMPA_MEASURED_STEPS:-20}"
readonly DATASET_SAMPLER_SEED="${AMPA_DATASET_SAMPLER_SEED:-1234}"
readonly RUN_ID="${AMPA_RUN_ID:-qwen3-30b-a3b-base-c0-ep8-${DATA_MODE}-${PROFILE_MODE}-${SLURM_JOB_ID}}"
readonly RUN_DIR="${ROOT_DIR}/runs/${RUN_ID}"
readonly OVERLAY_DIR="${ROOT_DIR}/.runtime/ampa-overlay/${SLURM_JOB_ID}/${RUN_ID}"

case "${DATA_MODE}" in
    indexed)
        : "${AMPA_DATA_PATH:?Set AMPA_DATA_PATH to a Megatron data-path value for the indexed realistic mixture.}"
        : "${AMPA_DATA_MANIFEST:?Set AMPA_DATA_MANIFEST to the validated immutable dataset JSON manifest.}"
        default_log_interval=5
        ;;
    mock)
        default_log_interval=1
        ;;
    *) echo "AMPA_DATA_MODE must be 'indexed' or 'mock', got ${DATA_MODE}" >&2; exit 2 ;;
esac
readonly LOG_INTERVAL="${AMPA_LOG_INTERVAL:-${default_log_interval}}"

case "${PROFILE_MODE}" in
    on|off) ;;
    *) echo "AMPA_PROFILE_MODE must be 'on' or 'off', got ${PROFILE_MODE}" >&2; exit 2 ;;
esac
[[ "${NSYS_CAPTURE_STEPS}" =~ ^[0-9]+$ ]] || {
    echo "AMPA_NSYS_CAPTURE_STEPS must be an integer from 0 to 3." >&2
    exit 2
}
if (( NSYS_CAPTURE_STEPS > 0 )); then
    [[ "${NSYS_CAPTURE_STEPS}" -le 3 && "${PROFILE_MODE}" == off \
        && "${TOPOLOGY_VARIANT}" == cp2_ep8 && "${DATA_MODE}" == indexed \
        && "${RECOMPUTE_MODE}" == none && "${WARMUP_STEPS}" == 5 \
        && "${MEASURED_STEPS}" == 20 && "${RUN_ID}" == *cp2*nsys* ]] || {
        echo "Nsight requires a distinct cp2-nsys run, profile OFF, indexed 5+20 steps and no recomputation." >&2
        exit 2
    }
fi
[[ "${TORCH_PROFILE_STEPS}" =~ ^[0-9]+$ ]] || {
    echo "AMPA_TORCH_PROFILE_STEPS must be an integer from 0 to 3." >&2
    exit 2
}
if (( TORCH_PROFILE_STEPS > 0 )); then
    [[ "${TORCH_PROFILE_STEPS}" -le 3 && "${NSYS_CAPTURE_STEPS}" == 0 \
        && "${PROFILE_MODE}" == off && "${TOPOLOGY_VARIANT}" == cp2_ep8 \
        && "${DATA_MODE}" == indexed && "${RECOMPUTE_MODE}" == none \
        && "${WARMUP_STEPS}" == 5 && "${MEASURED_STEPS}" == 20 \
        && "${RUN_ID}" == *cp2*torchprof* ]] || {
        echo "PyTorch deep trace requires a distinct cp2-torchprof run, profile OFF, indexed 5+20 steps and no recomputation." >&2
        exit 2
    }
fi
if [[ "${SAMPLE_FINGERPRINTS}" == 1 ]]; then
    [[ "${DATA_MODE}" == indexed && "${PROFILE_MODE}" == off \
        && "${TOPOLOGY_VARIANT}" == cp2_ep8 && "${RECOMPUTE_MODE}" == none \
        && "${NSYS_CAPTURE_STEPS}" == 0 && "${LOG_INTERVAL}" == 1 \
        && "${WARMUP_STEPS}" == 5 && "${MEASURED_STEPS}" == 20 ]] || {
        echo "Batch fingerprints require profile-OFF CP2 indexed 5+20 with one-step logging." >&2
        exit 2
    }
elif [[ "${SAMPLE_FINGERPRINTS}" != 0 ]]; then
    echo "AMPA_SAMPLE_FINGERPRINTS must be 0 or 1." >&2
    exit 2
fi
case "${RECOMPUTE_MODE}" in
    none|moe_act) ;;
    *) echo "AMPA_RECOMPUTE_MODE must be 'none' or 'moe_act', got ${RECOMPUTE_MODE}" >&2; exit 2 ;;
esac
if [[ "${RECOMPUTE_MODE}" == moe_act && "${PROFILE_MODE}" != off ]]; then
    echo "Selective moe_act is profile-OFF only until observer recomputation accounting is reviewed." >&2
    exit 2
fi
case "${TOPOLOGY_VARIANT}" in
    cp1_ep8) ;;
    cp2_ep8)
        [[ "${DATA_MODE}" == indexed && "${RECOMPUTE_MODE}" == none \
            && "${SEQUENCE_LENGTH}" == 4096 && "${MICRO_BATCH_SIZE}" == 1 && "${GLOBAL_BATCH_SIZE}" == 64 ]] || {
            echo "CP2/EP8 requires indexed L4096 MBS1 GBS64 and no recomputation." >&2
            exit 2
        }
        [[ -n "${AMPA_RUN_ID:-}" && "${RUN_ID}" == *cp2* ]] || {
            echo "CP2/EP8 requires an explicit run ID containing cp2 to prevent baseline mislabeling." >&2
            exit 2
        }
        ;;
    *) echo "Unsupported AMPA_TOPOLOGY_VARIANT=${TOPOLOGY_VARIANT}" >&2; exit 2 ;;
esac

if [[ "${DATA_MODE}" == mock ]]; then
    [[ "${PROFILE_MODE}" == off ]] || {
        echo "AMPA_DATA_MODE=mock is noncanonical and requires AMPA_PROFILE_MODE=off." >&2
        exit 2
    }
    [[ "${WARMUP_STEPS}" == 0 && "${MEASURED_STEPS}" == 1 && "${LOG_INTERVAL}" == 1 ]] || {
        echo "AMPA_DATA_MODE=mock is limited to one optimizer step: warmup=0, measured=1, log_interval=1." >&2
        exit 2
    }
fi

install_editable_overlays() {
    uv pip install --python /opt/venvs/megatron-bridge/bin/python --no-deps --no-build-isolation \
        --editable "${WORKSPACE}/Megatron-LM"
    uv pip install --python /opt/venvs/megatron-bridge/bin/python --no-deps --no-build-isolation \
        --editable "${WORKSPACE}/Megatron-Bridge"
}

container_train() {
    export RANK="${SLURM_PROCID}"
    export WORLD_SIZE="${SLURM_NTASKS}"
    # Slurm exposes one allocated GPU per task as CUDA device zero in this EDF.
    export LOCAL_RANK=0
    local ready_marker="${WORKSPACE}/.runtime/ampa-overlay/${SLURM_JOB_ID}/${RUN_ID}/ready-${SLURM_NODEID}"
    local failed_marker="${WORKSPACE}/.runtime/ampa-overlay/${SLURM_JOB_ID}/${RUN_ID}/failed-${SLURM_NODEID}"
    mkdir -p "${WORKSPACE}/.runtime/ampa-overlay/${SLURM_JOB_ID}/${RUN_ID}"
    if [[ "${SLURM_LOCALID}" == 0 ]]; then
        if install_editable_overlays; then
            touch "${ready_marker}"
        else
            touch "${failed_marker}"
            return 1
        fi
    else
        while [[ ! -e "${ready_marker}" && ! -e "${failed_marker}" ]]; do sleep 1; done
        [[ -e "${ready_marker}" ]] || return 1
    fi
    cd "${WORKSPACE}"
    local data_args=()
    if [[ "${DATA_MODE}" == indexed ]]; then
        data_args=(
            --data-mode indexed
            --dataset-data-path "${AMPA_DATA_PATH}"
            --dataset-manifest "${AMPA_DATA_MANIFEST}"
        )
    else
        data_args=(--data-mode mock)
    fi
    local trace_command=()
    local trace_args=()
    if (( NSYS_CAPTURE_STEPS > 0 )); then
        trace_command=(
            nsys profile --sample=none --trace=cuda,nvtx
            --capture-range=cudaProfilerApi --capture-range-end=stop
            --output="${RUN_DIR}/traces/rank-${SLURM_PROCID}"
        )
        trace_args=(--nsys-capture-steps "${NSYS_CAPTURE_STEPS}")
    fi
    if (( TORCH_PROFILE_STEPS > 0 )); then
        trace_args=(--torch-profile-steps "${TORCH_PROFILE_STEPS}" --torch-profile-ranks "${TORCH_PROFILE_RANKS}")
    fi
    if [[ "${SAMPLE_FINGERPRINTS}" == 1 ]]; then
        trace_args+=(--sample-fingerprints)
    fi
    PYTHONPATH="${WORKSPACE}:${WORKSPACE}/profiling${PYTHONPATH:+:${PYTHONPATH}}" \
        "${trace_command[@]}" /opt/venvs/megatron-bridge/bin/python "${WORKSPACE}/profiling/run_pretrain.py" \
        --run-dir "${RUN_DIR}" \
        --pretrained-checkpoint "${AMPA_C0}" \
        "${data_args[@]}" \
        --hf-revision "${AMPA_HF_REVISION}" \
        --sequence-length "${SEQUENCE_LENGTH}" \
        --micro-batch-size "${MICRO_BATCH_SIZE}" \
        --global-batch-size "${GLOBAL_BATCH_SIZE}" \
        --warmup-steps "${WARMUP_STEPS}" \
        --measured-steps "${MEASURED_STEPS}" \
        --log-interval "${LOG_INTERVAL}" \
        --dataset-sampler-seed "${DATASET_SAMPLER_SEED}" \
        --recompute-mode "${RECOMPUTE_MODE}" \
        --topology-variant "${TOPOLOGY_VARIANT}" \
        --lightweight-profiling "${PROFILE_MODE}" \
        "${trace_args[@]}"
}

case "${1:-}" in
    --container-train)
        container_train
        exit 0
        ;;
    "")
        ;;
    *)
        echo "Unknown internal mode: $1" >&2
        exit 2
        ;;
esac

if [[ -e "${RUN_DIR}" ]]; then
    echo "Refusing to reuse run directory: ${RUN_DIR}" >&2
    exit 2
fi
[[ -f "${AMPA_C0}/ampa_c0_manifest.json" ]] || { echo "C0 manifest not found: ${AMPA_C0}/ampa_c0_manifest.json" >&2; exit 2; }
if [[ "${DATA_MODE}" == indexed ]]; then
    [[ -f "${AMPA_DATA_MANIFEST}" ]] || { echo "Dataset manifest not found: ${AMPA_DATA_MANIFEST}" >&2; exit 2; }
fi
mkdir -p "${RUN_DIR}"/{config,raw,metrics,analysis,plots,traces,logs}
if (( TORCH_PROFILE_STEPS > 0 )); then
    # Bridge writes alongside its tensorboard directory. Keep the actual
    # Chrome traces in the run's traces/ tree without changing Bridge.
    mkdir -p "${RUN_DIR}/traces/torch_profile" "${RUN_DIR}/logs/tensorboard"
    ln -s ../traces/torch_profile "${RUN_DIR}/logs/torch_profile"
fi
if [[ "${DATA_MODE}" == indexed ]]; then
    mkdir -p "${RUN_DIR}/logs/wandb"
    cp -- "${AMPA_DATA_MANIFEST}" "${RUN_DIR}/config/dataset_manifest.json"
    # Every rank must consume the exact run-local bytes later used by the
    # overhead comparator, not a source manifest that may change meanwhile.
    AMPA_DATA_MANIFEST="${RUN_DIR}/config/dataset_manifest.json"
else
    printf '%s\n' \
        'This run uses generated mock tokens for a one-step C0 optimizer bring-up.' \
        'It is noncanonical, unprofiled, and must not be used as realistic-mixture profiling data.' \
        > "${RUN_DIR}/NONCANONICAL_MOCK_BRINGUP.txt"
fi
cp -- "${AMPA_C0}/ampa_c0_manifest.json" "${RUN_DIR}/config/c0_manifest.json"
cp -- "${ROOT_DIR}/configs/model/qwen3_30b_a3b_base.json" "${RUN_DIR}/config/model_manifest.json"

export AMPA_PROFILE_MODE="${PROFILE_MODE}"
export AMPA_NSYS_CAPTURE_STEPS="${NSYS_CAPTURE_STEPS}"
export AMPA_TORCH_PROFILE_STEPS="${TORCH_PROFILE_STEPS}"
export AMPA_TORCH_PROFILE_RANKS="${TORCH_PROFILE_RANKS}"
export AMPA_SAMPLE_FINGERPRINTS="${SAMPLE_FINGERPRINTS}"
export AMPA_TOPOLOGY_VARIANT="${TOPOLOGY_VARIANT}"
export AMPA_RECOMPUTE_MODE="${RECOMPUTE_MODE}"
export AMPA_SEQUENCE_LENGTH="${SEQUENCE_LENGTH}"
export AMPA_MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE}"
export AMPA_GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE}"
export AMPA_WARMUP_STEPS="${WARMUP_STEPS}"
export AMPA_MEASURED_STEPS="${MEASURED_STEPS}"
export AMPA_LOG_INTERVAL="${LOG_INTERVAL}"
export AMPA_DATASET_SAMPLER_SEED="${DATASET_SAMPLER_SEED}"
export AMPA_RUN_ID="${RUN_ID}"
export AMPA_DATA_MODE="${DATA_MODE}"
export RUN_DIR AMPA_C0 AMPA_HF_REVISION PROFILE_MODE AMPA_DATA_MODE
if [[ "${DATA_MODE}" == indexed ]]; then
    export AMPA_DATA_PATH AMPA_DATA_MANIFEST
else
    unset AMPA_DATA_PATH AMPA_DATA_MANIFEST
fi
mkdir -p "${OVERLAY_DIR}"
export MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)"
export MASTER_PORT="$((20000 + SLURM_JOB_ID % 20000))"
srun --mpi=pmix --container-remap-root --environment="${CONTAINER_ENV}" \
    --nodes=2 --ntasks=8 --ntasks-per-node=4 --gpus-per-task=1 \
    --output="${RUN_DIR}/logs/rank-%t.out" \
    bash "${ROOT_DIR}/jobs/training/qwen3_30b_a3b_base_ep8_profile.sbatch" --container-train

if [[ "${DATA_MODE}" == indexed ]]; then
    srun --mpi=pmix --container-remap-root --environment="${CONTAINER_ENV}" \
        --nodes=1 --ntasks=1 --gpus-per-task=1 \
        --output="${RUN_DIR}/logs/timing.out" \
        /opt/venvs/megatron-bridge/bin/python "${ROOT_DIR}/analysis/extract_step_timing.py" \
        --run-dir "${RUN_DIR}" \
        --warmup-steps "${WARMUP_STEPS}" \
        --measured-steps "${MEASURED_STEPS}" \
        --log-interval "${LOG_INTERVAL}" \
        --global-batch-size "${GLOBAL_BATCH_SIZE}" \
        --sequence-length "${SEQUENCE_LENGTH}"

    if [[ "${PROFILE_MODE}" == on ]]; then
        srun --mpi=pmix --container-remap-root --environment="${CONTAINER_ENV}" \
            --nodes=1 --ntasks=1 --gpus-per-task=1 \
            --output="${RUN_DIR}/logs/aggregate.out" \
            /opt/venvs/megatron-bridge/bin/python "${ROOT_DIR}/analysis/aggregate_run.py" --run-dir "${RUN_DIR}"
    fi
    if (( TORCH_PROFILE_STEPS > 0 )); then
        srun --mpi=pmix --container-remap-root --environment="${CONTAINER_ENV}" \
            --nodes=1 --ntasks=1 --gpus-per-task=1 \
            --output="${RUN_DIR}/logs/torch_trace_validation.out" \
            /opt/venvs/megatron-bridge/bin/python "${ROOT_DIR}/analysis/validate_torch_trace.py" \
            --run-dir "${RUN_DIR}" --ranks "${TORCH_PROFILE_RANKS}" \
            --step "${WARMUP_STEPS}" --capture-steps "${TORCH_PROFILE_STEPS}"
        srun --mpi=pmix --container-remap-root --environment="${CONTAINER_ENV}" \
            --nodes=1 --ntasks=1 --gpus-per-task=1 \
            --output="${RUN_DIR}/logs/torch_trace_aggregation.out" \
            /opt/venvs/megatron-bridge/bin/python "${ROOT_DIR}/analysis/aggregate_deep_trace.py" \
            --run-dir "${RUN_DIR}" --ranks "${TORCH_PROFILE_RANKS}"
    fi
fi
