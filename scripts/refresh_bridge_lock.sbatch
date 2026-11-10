#!/usr/bin/env bash
# Refresh Megatron-Bridge's lockfile after deliberately advancing its
# Megatron-LM submodule. This runs only on a GH200 compute node.

#SBATCH --job-name=bridge-lock-gh200
#SBATCH --account=infra01
#SBATCH --partition=preemptable
#SBATCH --qos=normal
#SBATCH --constraint=gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=.runtime/slurms/slurm-%x-%j.out
#SBATCH --error=.runtime/slurms/slurm-%x-%j.err

set -euo pipefail

readonly ROOT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)}"
readonly BRIDGE_DIR="${ROOT_DIR}/Megatron-Bridge"
readonly CONTAINER_ENV="${ROOT_DIR}/setup/gh200-cuda130-runtime.toml"
readonly SUBMODULE_PATH="3rdparty/Megatron-LM"

[[ $(uname -m) == aarch64 ]] || { echo "Expected an aarch64 GH200 node." >&2; exit 2; }
[[ -d ${BRIDGE_DIR}/.git ]] || { echo "Missing Bridge checkout: ${BRIDGE_DIR}" >&2; exit 2; }

git -C "${BRIDGE_DIR}" submodule sync --recursive
git -C "${BRIDGE_DIR}" submodule update --init --recursive
readonly MCORE_REF="$(git -C "${BRIDGE_DIR}" ls-files --stage "${SUBMODULE_PATH}" | awk '$1 == "160000" { print $2; exit }')"
[[ -n ${MCORE_REF} ]] || { echo "Missing submodule gitlink: ${SUBMODULE_PATH}" >&2; exit 2; }
test "$(git -C "${BRIDGE_DIR}/${SUBMODULE_PATH}" rev-parse HEAD)" = "${MCORE_REF}"

srun --mpi=pmix --container-remap-root --environment="${CONTAINER_ENV}" \
    --nodes=1 --ntasks=1 \
    bash -lc '
        set -euo pipefail
        cd "${WORKSPACE}/Megatron-Bridge"
        uv lock \
            --upgrade-package transformer-engine \
            --upgrade-package nvidia-cudnn-frontend \
            --upgrade-package fast-hadamard-transform
        uv lock --check
    '

git -C "${BRIDGE_DIR}" diff --check
git -C "${BRIDGE_DIR}" diff -- uv.lock
