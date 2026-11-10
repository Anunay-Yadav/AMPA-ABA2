#!/usr/bin/env bash
# One-time C0 import. Submit only after `squeue -u "$USER"`; this uses two nodes.

#SBATCH --job-name=ampa-qwen3-base-c0
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
#SBATCH --requeue
#SBATCH --output=.runtime/slurms/slurm-%x-%j.out
#SBATCH --error=.runtime/slurms/slurm-%x-%j.err

set -euo pipefail

[[ "${SLURM_JOB_PARTITION:-}" == preemptable ]] || {
    echo "AMPA jobs must run on the preemptable partition." >&2
    exit 2
}

readonly ROOT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)}"
readonly CONTAINER_ENV="${ROOT_DIR}/setup/gh200-cuda130-runtime.toml"
readonly C0_DIR="${ROOT_DIR}/checkpoint/megatron/qwen3-30b-a3b-base-c0"
readonly HF_MODEL="Qwen/Qwen3-30B-A3B-Base"
readonly HF_REVISION="1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9"
readonly DEFAULT_SNAPSHOT_DIR="${ROOT_DIR}/checkpoint/hf/qwen3-30b-a3b-base/${HF_REVISION}"
readonly HF_SNAPSHOT_DIR="${AMPA_HF_SNAPSHOT_DIR:-${DEFAULT_SNAPSHOT_DIR}}"
readonly HF_SNAPSHOT_MANIFEST="${HF_SNAPSHOT_DIR}/ampa_hf_snapshot_manifest.json"

install_editable_overlays() {
    uv pip install --python /opt/venvs/megatron-bridge/bin/python --no-deps --no-build-isolation \
        --editable "${WORKSPACE}/Megatron-LM"
    uv pip install --python /opt/venvs/megatron-bridge/bin/python --no-deps --no-build-isolation \
        --editable "${WORKSPACE}/Megatron-Bridge"
}

container_import() {
    export RANK="${SLURM_PROCID}"
    export WORLD_SIZE="${SLURM_NTASKS}"
    export LOCAL_RANK=0
    local marker_dir="${WORKSPACE}/.runtime/ampa-overlay/${SLURM_JOB_ID}"
    local ready_marker="${marker_dir}/ready-${SLURM_NODEID}"
    local failed_marker="${marker_dir}/failed-${SLURM_NODEID}"
    mkdir -p "${marker_dir}"
    if [[ "${SLURM_LOCALID}" == 0 ]]; then
        if install_editable_overlays; then touch "${ready_marker}"; else touch "${failed_marker}"; return 1; fi
    else
        while [[ ! -e "${ready_marker}" && ! -e "${failed_marker}" ]]; do sleep 1; done
        [[ -e "${ready_marker}" ]] || return 1
    fi
    cd "${WORKSPACE}/Megatron-Bridge/scripts/conversion"
    /opt/venvs/megatron-bridge/bin/python run_conversion.py import \
        --device gpu --hf-model "${HF_SNAPSHOT_DIR}" \
        --megatron-path "${C0_DIR}" --tp 1 --pp 1 --ep 8 --etp 1 \
        --torch-dtype bfloat16 --low-memory-save --distributed-timeout-minutes 60
}

case "${1:-}" in
    --container-import) container_import; exit 0 ;;
    "") ;;
    *) echo "Unknown internal mode: $1" >&2; exit 2 ;;
esac

if [[ -e "${C0_DIR}" ]]; then
    echo "Refusing to overwrite existing C0 target: ${C0_DIR}" >&2
    exit 2
fi
[[ -f "${HF_SNAPSHOT_MANIFEST}" ]] || { echo "HF snapshot manifest not found: ${HF_SNAPSHOT_MANIFEST}" >&2; exit 2; }
export C0_DIR HF_SNAPSHOT_DIR HF_SNAPSHOT_MANIFEST
export MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)"
export MASTER_PORT="$((20000 + SLURM_JOB_ID % 20000))"
srun --mpi=pmix --container-remap-root --environment="${CONTAINER_ENV}" \
    --nodes=1 --ntasks=1 --gpus-per-task=1 \
    /opt/venvs/megatron-bridge/bin/python "${ROOT_DIR}/tools/materialize_hf_snapshot.py" \
    --model-id "${HF_MODEL}" --revision "${HF_REVISION}" --output-dir "${HF_SNAPSHOT_DIR}" --verify-only
srun --mpi=pmix --container-remap-root --environment="${CONTAINER_ENV}" \
    --nodes=2 --ntasks=8 --ntasks-per-node=4 --gpus-per-task=1 \
    --output="${ROOT_DIR}/.runtime/ampa-c0-import-${SLURM_JOB_ID}-%t.out" \
    bash "${ROOT_DIR}/jobs/checkpoint/import_qwen3_30b_a3b_base_c0.sbatch" --container-import

srun --mpi=pmix --container-remap-root --environment="${CONTAINER_ENV}" \
    --nodes=1 --ntasks=1 --gpus-per-task=1 \
    /opt/venvs/megatron-bridge/bin/python "${ROOT_DIR}/tools/write_c0_manifest.py" \
    --checkpoint "${C0_DIR}" --hf-model "${HF_MODEL}" --hf-revision "${HF_REVISION}" \
    --model-manifest "${ROOT_DIR}/configs/model/qwen3_30b_a3b_base.json" \
    --hf-snapshot-manifest "${HF_SNAPSHOT_MANIFEST}" --dataset-independent

# The imported payload and identity are now immutable. Validation state is external.
chmod -R a-w "${C0_DIR}"
