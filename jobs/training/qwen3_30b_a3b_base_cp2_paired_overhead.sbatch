#!/usr/bin/env bash
# Run two independent C0 starts sequentially on the same two nodes. Submit
# only after checking `squeue -u "$USER"`; this uses the full two-node cap.

#SBATCH --job-name=ampa-cp2-overhead
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
: "${AMPA_PAIR_PREFIX:?Set a unique run-ID prefix containing cp2.}"
: "${AMPA_PAIR_ORDER:?Set off-on or on-off.}"
[[ "${AMPA_PAIR_PREFIX}" == *cp2* ]] || { echo "Pair prefix must contain cp2." >&2; exit 2; }
[[ "${AMPA_PAIR_PREFIX}" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "Pair prefix must be a single safe run-ID component." >&2; exit 2; }
[[ "${AMPA_TOPOLOGY_VARIANT:-}" == cp2_ep8 ]] || { echo "Pair requires AMPA_TOPOLOGY_VARIANT=cp2_ep8." >&2; exit 2; }
[[ "${AMPA_RECOMPUTE_MODE:-none}" == none ]] || { echo "Pair requires no recomputation." >&2; exit 2; }

case "${AMPA_PAIR_ORDER}" in
    off-on) modes=(off on) ;;
    on-off) modes=(on off) ;;
    *) echo "AMPA_PAIR_ORDER must be off-on or on-off." >&2; exit 2 ;;
esac

for mode in "${modes[@]}"; do
    export AMPA_PROFILE_MODE="${mode}"
    export AMPA_RUN_ID="${AMPA_PAIR_PREFIX}-${mode}"
    # This invokes the existing, tested AMPA job body inside the current
    # allocation; it does not submit a second allocation or resume weights.
    bash "${ROOT_DIR}/jobs/training/qwen3_30b_a3b_base_ep8_profile.sbatch"
done
