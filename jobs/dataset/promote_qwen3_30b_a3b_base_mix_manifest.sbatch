#!/usr/bin/env bash
# Revalidate and publish the reviewed final indexed-data manifest on compute.
# Submit only after `squeue -u "$USER"`; this hashes/walks indexed artifacts.

#SBATCH --job-name=ampa-qwen3-mix-promote
#SBATCH --account=infra01
#SBATCH --partition=preemptable
#SBATCH --qos=normal
#SBATCH --constraint=gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
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
    cp1_ep8)
        readonly CACHE_HIT="${ROOT_DIR}/dataset/manifests/qwen3-30b-a3b-base-realistic-mix-v2-cache-bridge-hit.json"
        readonly CACHE_VALIDATED="${ROOT_DIR}/dataset/manifests/qwen3-30b-a3b-base-realistic-mix-v2-cache-validated.json"
        readonly REVIEW_RECORD="${ROOT_DIR}/progress/03_dataset/evidence/qwen3_mix_v2_independent_review.json"
        readonly FINAL_MANIFEST="${ROOT_DIR}/dataset/manifests/qwen3-30b-a3b-base-realistic-mix-v2-validated.json"
        readonly PROMOTER="${ROOT_DIR}/tools/promote_indexed_manifest.py"
        readonly PARENT_MANIFEST=""
        ;;
    cp2_ep8)
        readonly CACHE_HIT="${ROOT_DIR}/dataset/manifests/qwen3-30b-a3b-base-realistic-mix-v2-cp2-cache-bridge-hit.json"
        readonly CACHE_VALIDATED="${ROOT_DIR}/dataset/manifests/qwen3-30b-a3b-base-realistic-mix-v2-cp2-cache-validated.json"
        readonly REVIEW_RECORD="${ROOT_DIR}/progress/05_smoke/evidence/qwen3_mix_v2_cp2_independent_review.json"
        readonly FINAL_MANIFEST="${ROOT_DIR}/dataset/manifests/qwen3-30b-a3b-base-realistic-mix-v2-cp2-validated.json"
        readonly PROMOTER="${ROOT_DIR}/tools/promote_cp2_feasibility_manifest.py"
        readonly PARENT_MANIFEST="${ROOT_DIR}/dataset/manifests/qwen3-30b-a3b-base-realistic-mix-v2-validated.json"
        ;;
    *) echo "Unsupported AMPA_CACHE_VARIANT" >&2; exit 2 ;;
esac

container_promote() {
    cd "${WORKSPACE}"
    uv pip install --python /opt/venvs/megatron-bridge/bin/python --no-deps --no-build-isolation \
        --editable "${WORKSPACE}/Megatron-LM"
    uv pip install --python /opt/venvs/megatron-bridge/bin/python --no-deps --no-build-isolation \
        --editable "${WORKSPACE}/Megatron-Bridge"
    local parent_args=()
    if [[ -n "${PARENT_MANIFEST}" ]]; then parent_args=(--parent-validated-manifest "${PARENT_MANIFEST}"); fi
    /opt/venvs/megatron-bridge/bin/python "${PROMOTER}" \
        --cache-hit-manifest "${CACHE_HIT}" --cache-validated-manifest "${CACHE_VALIDATED}" \
        --independent-review "${REVIEW_RECORD}" --output-manifest "${FINAL_MANIFEST}" \
        --workspace-root "${WORKSPACE}" "${parent_args[@]}"
}

case "${1:-}" in
    --container-promote) container_promote; exit 0 ;;
    "") ;;
    *) echo "Unknown internal mode: $1" >&2; exit 2 ;;
esac

[[ -f "${CACHE_HIT}" ]] || { echo "Bridge cache-hit manifest not found: ${CACHE_HIT}" >&2; exit 2; }
[[ -f "${CACHE_VALIDATED}" ]] || { echo "Cache-validated manifest not found: ${CACHE_VALIDATED}" >&2; exit 2; }
[[ -f "${REVIEW_RECORD}" ]] || { echo "Independent review record not found: ${REVIEW_RECORD}" >&2; exit 2; }
[[ ! -e "${FINAL_MANIFEST}" ]] || { echo "Refusing to overwrite ${FINAL_MANIFEST}" >&2; exit 2; }
srun --mpi=pmix --container-remap-root --environment="${CONTAINER_ENV}" \
    --nodes=1 --ntasks=1 --gpus-per-task=1 \
    --output="${ROOT_DIR}/.runtime/ampa-mix-promote-${SLURM_JOB_ID}-%t.out" \
    bash "$0" --container-promote
