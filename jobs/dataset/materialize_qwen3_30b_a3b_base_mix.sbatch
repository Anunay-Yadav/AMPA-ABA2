#!/usr/bin/env bash
# Stream and index the pinned AMPA mixture. Submit only after `squeue -u "$USER"`.

#SBATCH --job-name=ampa-qwen3-mix-index
#SBATCH --account=infra01
#SBATCH --partition=preemptable
#SBATCH --qos=normal
#SBATCH --constraint=gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
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
readonly PLAN_MANIFEST="${ROOT_DIR}/dataset/manifests/qwen3-30b-a3b-base-realistic-mix-v2.json"
readonly OUTPUT_MANIFEST="${ROOT_DIR}/dataset/manifests/qwen3-30b-a3b-base-realistic-mix-v2-materialized.json"

container_materialize() {
    cd "${WORKSPACE}"
    uv pip install --python /opt/venvs/megatron-bridge/bin/python --no-deps --no-build-isolation \
        --editable "${WORKSPACE}/Megatron-LM"
    uv pip install --python /opt/venvs/megatron-bridge/bin/python --no-deps --no-build-isolation \
        --editable "${WORKSPACE}/Megatron-Bridge"
    /opt/venvs/megatron-bridge/bin/python "${WORKSPACE}/tools/materialize_indexed_mixture.py" \
        --plan-manifest "${PLAN_MANIFEST}" --output-manifest "${OUTPUT_MANIFEST}" \
        --workspace-root "${WORKSPACE}" --build-id "${SLURM_JOB_ID}"
}

case "${1:-}" in
    --container-materialize) container_materialize; exit 0 ;;
    "") ;;
    *) echo "Unknown internal mode: $1" >&2; exit 2 ;;
esac

[[ ! -e "${OUTPUT_MANIFEST}" ]] || { echo "Refusing to overwrite ${OUTPUT_MANIFEST}" >&2; exit 2; }
srun --mpi=pmix --container-remap-root --environment="${CONTAINER_ENV}" \
    --nodes=1 --ntasks=1 --gpus-per-task=1 \
    --output="${ROOT_DIR}/.runtime/ampa-index-${SLURM_JOB_ID}-%t.out" \
    bash "$0" --container-materialize
