#!/usr/bin/env bash
# One-time pinned HF snapshot. Submit only after `squeue -u "$USER"`; this uses one node.

#SBATCH --job-name=ampa-qwen3-base-hf-snapshot
#SBATCH --account=infra01
#SBATCH --partition=preemptable
#SBATCH --qos=normal
#SBATCH --constraint=gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
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
readonly HF_MODEL="Qwen/Qwen3-30B-A3B-Base"
readonly HF_REVISION="1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9"
readonly SNAPSHOT_DIR="${ROOT_DIR}/checkpoint/hf/qwen3-30b-a3b-base/${HF_REVISION}"

container_materialize() {
    cd "${WORKSPACE}"
    /opt/venvs/megatron-bridge/bin/python "${WORKSPACE}/tools/materialize_hf_snapshot.py" \
        --model-id "${HF_MODEL}" --revision "${HF_REVISION}" --output-dir "${SNAPSHOT_DIR}"
}

case "${1:-}" in
    --container-materialize) container_materialize; exit 0 ;;
    "") ;;
    *) echo "Unknown internal mode: $1" >&2; exit 2 ;;
esac

[[ ! -e "${SNAPSHOT_DIR}" ]] || { echo "Refusing to overwrite snapshot: ${SNAPSHOT_DIR}" >&2; exit 2; }
mkdir -p "$(dirname "${SNAPSHOT_DIR}")"
srun --mpi=pmix --container-remap-root --environment="${CONTAINER_ENV}" \
    --nodes=1 --ntasks=1 --gpus-per-task=1 \
    --output="${ROOT_DIR}/.runtime/ampa-hf-snapshot-${SLURM_JOB_ID}-%t.out" \
    bash "$0" --container-materialize

# The snapshot is now a conversion input, not a mutable cache. Preserve read and execute bits
# while removing write permission after its manifest has been written and the job has succeeded.
chmod -R a-w "${SNAPSHOT_DIR}"
