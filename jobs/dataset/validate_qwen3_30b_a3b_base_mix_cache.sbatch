#!/usr/bin/env bash
# Independently validate native MCore cache artifacts. Submit only after `squeue -u "$USER"`.

#SBATCH --job-name=ampa-qwen3-cache-validate
#SBATCH --account=infra01
#SBATCH --partition=preemptable
#SBATCH --qos=normal
#SBATCH --constraint=gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
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
case "${AMPA_CACHE_VARIANT:-cp1_ep8}" in
    cp1_ep8) readonly INPUT_MANIFEST="${ROOT_DIR}/dataset/manifests/qwen3-30b-a3b-base-realistic-mix-v2-cache-prepared.json"
             readonly OUTPUT_MANIFEST="${ROOT_DIR}/dataset/manifests/qwen3-30b-a3b-base-realistic-mix-v2-cache-validated.json" ;;
    cp2_ep8) readonly INPUT_MANIFEST="${ROOT_DIR}/dataset/manifests/qwen3-30b-a3b-base-realistic-mix-v2-cp2-cache-prepared.json"
             readonly OUTPUT_MANIFEST="${ROOT_DIR}/dataset/manifests/qwen3-30b-a3b-base-realistic-mix-v2-cp2-cache-validated.json" ;;
    *) echo "Unsupported AMPA_CACHE_VARIANT" >&2; exit 2 ;;
esac

container_validate() {
    cd "${WORKSPACE}"
    uv pip install --python /opt/venvs/megatron-bridge/bin/python --no-deps --no-build-isolation \
        --editable "${WORKSPACE}/Megatron-LM"
    /opt/venvs/megatron-bridge/bin/python "${WORKSPACE}/tools/megatron_cache_stage.py" validate \
        --cache-prepared-manifest "${INPUT_MANIFEST}" --output-manifest "${OUTPUT_MANIFEST}" \
        --workspace-root "${WORKSPACE}"
}

case "${1:-}" in
    --container-validate) container_validate; exit 0 ;;
    "") ;;
    *) echo "Unknown internal mode: $1" >&2; exit 2 ;;
esac

[[ -f "${INPUT_MANIFEST}" ]] || { echo "Cache-prepared manifest not found: ${INPUT_MANIFEST}" >&2; exit 2; }
[[ ! -e "${OUTPUT_MANIFEST}" ]] || { echo "Refusing to overwrite ${OUTPUT_MANIFEST}" >&2; exit 2; }
srun --mpi=pmix --container-remap-root --environment="${CONTAINER_ENV}" \
    --nodes=1 --ntasks=1 --gpus-per-task=1 \
    --output="${ROOT_DIR}/.runtime/ampa-cache-validate-${SLURM_JOB_ID}-%t.out" \
    bash "$0" --container-validate
