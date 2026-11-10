#!/usr/bin/env bash
#SBATCH --job-name=ampa-uccl-debug-validate
#SBATCH --account=infra01
#SBATCH --partition=debug
#SBATCH --qos=normal
#SBATCH --constraint=gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --exclusive
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=.runtime/slurms/slurm-%x-%j.out
#SBATCH --error=.runtime/slurms/slurm-%x-%j.err
set -euo pipefail
[[ ${SLURM_JOB_PARTITION:-} == debug && ${SLURM_JOB_NUM_NODES:-} == 1 ]] || exit 2
readonly ROOT_DIR=${SLURM_SUBMIT_DIR:?}
[[ ${AMPA_UCCL_DEBUG_VALIDATION_ID:?} =~ ^[a-zA-Z0-9_-]+$ \
    && ${AMPA_UCCL_DEBUG_ORCHESTRATION_DIR:?} == "$ROOT_DIR"/runs/* ]] || exit 2
export AMPA_UCCL_DEBUG_VALIDATION_OUT="$ROOT_DIR/runs/$AMPA_UCCL_DEBUG_VALIDATION_ID"
[[ ! -e "$AMPA_UCCL_DEBUG_VALIDATION_OUT" && ! -L "$AMPA_UCCL_DEBUG_VALIDATION_OUT" ]] || exit 2
: "${AMPA_UCCL_DEBUG_EXPECTED_SOURCE_MANIFEST:?}" "${AMPA_UCCL_DEBUG_EXPECTED_SOURCE_SHA256:?}"
[[ "$AMPA_UCCL_DEBUG_EXPECTED_SOURCE_MANIFEST" == "$ROOT_DIR"/runs/* \
    && -f "$AMPA_UCCL_DEBUG_EXPECTED_SOURCE_MANIFEST" && ! -L "$AMPA_UCCL_DEBUG_EXPECTED_SOURCE_MANIFEST" \
    && "$AMPA_UCCL_DEBUG_EXPECTED_SOURCE_SHA256" =~ ^[0-9a-f]{64}$ ]] || exit 2
[[ $(sha256sum "$AMPA_UCCL_DEBUG_EXPECTED_SOURCE_MANIFEST" | cut -d ' ' -f 1) == "$AMPA_UCCL_DEBUG_EXPECTED_SOURCE_SHA256" ]] || exit 2
cd "$ROOT_DIR"
sha256sum --check "$AMPA_UCCL_DEBUG_EXPECTED_SOURCE_MANIFEST"
squeue -u "$USER"
srun --mpi=pmix --cpu-bind=none --container-remap-root --environment="$ROOT_DIR/setup/gh200-cuda130-runtime.toml" \
    --nodes=1 --ntasks=1 --gpus-per-task=1 bash -c 'set -euo pipefail
    cd "$WORKSPACE"
    export CUDA_VISIBLE_DEVICES=""
    export PYTHONPATH="$WORKSPACE:$WORKSPACE/profiling:$WORKSPACE/analysis${PYTHONPATH:+:$PYTHONPATH}"
    exec /opt/venvs/megatron-bridge/bin/python tools/uccl_ep_debug_validate.py \
        --orchestration-dir "$AMPA_UCCL_DEBUG_ORCHESTRATION_DIR" \
        --output-dir "$AMPA_UCCL_DEBUG_VALIDATION_OUT" --workers 4
    '
sha256sum --check "$AMPA_UCCL_DEBUG_EXPECTED_SOURCE_MANIFEST"
