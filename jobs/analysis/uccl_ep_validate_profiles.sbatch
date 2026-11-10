#!/usr/bin/env bash
#SBATCH --job-name=ampa-uccl-validate
#SBATCH --account=infra01
#SBATCH --partition=preemptable
#SBATCH --qos=normal
#SBATCH --constraint=gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=.runtime/slurms/slurm-%x-%j.out
#SBATCH --error=.runtime/slurms/slurm-%x-%j.err
set -euo pipefail
[[ "${SLURM_JOB_PARTITION:-}" == preemptable ]] || exit 2
readonly ROOT_DIR="$SLURM_SUBMIT_DIR"
[[ "${AMPA_UCCL_VALIDATION_ID:?}" =~ ^[a-zA-Z0-9_-]+$ && "${AMPA_UCCL_ORCHESTRATION_DIR:?}" == /* ]] || exit 2
export AMPA_UCCL_VALIDATION_ROOT="$ROOT_DIR/runs/$AMPA_UCCL_VALIDATION_ID"
[[ ! -e "$AMPA_UCCL_VALIDATION_ROOT" && ! -L "$AMPA_UCCL_VALIDATION_ROOT" ]] || exit 2
squeue -u "$USER"
srun --mpi=pmix --container-remap-root --environment="$ROOT_DIR/setup/gh200-cuda130-runtime.toml" \
    --nodes=1 --ntasks=1 --gpus-per-task=1 bash -c 'set -euo pipefail
    cd "$WORKSPACE"
    export PYTHONPATH="$WORKSPACE:$WORKSPACE/profiling:$WORKSPACE/analysis${PYTHONPATH:+:$PYTHONPATH}"
    exec /opt/venvs/megatron-bridge/bin/python tools/uccl_ep_validate_profiles.py \
        --orchestration-dir "$AMPA_UCCL_ORCHESTRATION_DIR" --output-dir "$AMPA_UCCL_VALIDATION_ROOT" --workers 8
    '
