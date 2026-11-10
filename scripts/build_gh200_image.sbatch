#!/usr/bin/env bash
# Build the dependency image with Podman and convert it to an Enroot squashfs
# on a GH200 compute node. Submit from the workspace root:
#   sbatch scripts/build_gh200_image.sbatch

#SBATCH --job-name=gh200-cuda130-image
#SBATCH --account=infra01
#SBATCH --partition=preemptable
#SBATCH --qos=normal
#SBATCH --constraint=gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
# The node-local Enroot base image and compiler peak well below this; declaring
# it avoids the site's 450 GB implicit memory request for 32 CPUs.
#SBATCH --mem=160G
# The observed full base import plus dependency build is measured in minutes;
# two hours leaves diagnostic margin while allowing the job to backfill.
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --output=.runtime/slurms/slurm-%x-%j.out
#SBATCH --error=.runtime/slurms/slurm-%x-%j.err

set -euo pipefail

# Slurm stages batch scripts under /var/spool/slurmd; SLURM_SUBMIT_DIR retains
# the actual workspace from which the user submitted this job.
readonly ROOT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)}"
readonly SETUP_DIR="${ROOT_DIR}/setup"
readonly DOCKERFILE="${SETUP_DIR}/Dockerfile.gh200-cuda130"
readonly IMAGE_TAG="gh200-cuda130-megatron:cuda13.0-gh200"
readonly SQSH_PATH="${SETUP_DIR}/gh200-cuda130-megatron.sqsh"
readonly BASE_IMAGE="nvcr.io/nvidia/pytorch:25.08-py3"
readonly BRIDGE_REF="3a459fe4ab5e5e6b6c489434c687cbc4a01a55b3"
readonly SQSH_SHA256_PATH="${SQSH_PATH}.sha256"

[[ $(uname -m) == aarch64 ]] || { echo "This image must be built on aarch64 GH200 nodes." >&2; exit 2; }
if [[ -f ${SQSH_PATH} && ! -e ${SQSH_SHA256_PATH} ]]; then
    # A preemptable job can be interrupted after mksquashfs atomically finishes
    # but before this script writes its checksum.  Validate that exact artifact
    # on a new compute allocation and attest it; never overwrite it blindly.
    echo "Resuming completed SQSH without checksum: ${SQSH_PATH}"
    unsquashfs -s "${SQSH_PATH}" >/dev/null
    sha256sum "${SQSH_PATH}" | tee "${SQSH_SHA256_PATH}"
    ls -lh "${SQSH_PATH}" "${SQSH_SHA256_PATH}"
    exit 0
fi
[[ ! -e ${SQSH_PATH} ]] || {
    echo "Refusing to overwrite existing image: ${SQSH_PATH}" >&2
    echo "Remove that exact file deliberately, then resubmit." >&2
    exit 2
}

echo "Building ${IMAGE_TAG} from ${BASE_IMAGE} on $(hostname)"
echo "Persistent output: ${SQSH_PATH}"
podman build \
    --tag "${IMAGE_TAG}" \
    --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
    --build-arg "MEGATRON_BRIDGE_REF=${BRIDGE_REF}" \
    --file "${DOCKERFILE}" \
    "${SETUP_DIR}"

podman run --rm "${IMAGE_TAG}" \
    python - <<'PY'
import sys
import torch

assert sys.version_info[:2] == (3, 12), sys.version
assert torch.version.cuda and torch.version.cuda.startswith("13.0"), torch.version.cuda
print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}; CUDA: {torch.version.cuda}")
PY
enroot import --arch aarch64 --output "${SQSH_PATH}" "podman://${IMAGE_TAG}"
sha256sum "${SQSH_PATH}" | tee "${SQSH_SHA256_PATH}"
ls -lh "${SQSH_PATH}" "${SQSH_SHA256_PATH}"
