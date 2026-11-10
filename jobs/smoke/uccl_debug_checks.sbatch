#!/usr/bin/env bash
#SBATCH --job-name=ampa-uccl-debug-checks
#SBATCH --account=infra01
#SBATCH --partition=debug
#SBATCH --qos=normal
#SBATCH --constraint=gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --exclusive
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --no-requeue
#SBATCH --output=.runtime/slurms/slurm-%x-%j.out
#SBATCH --error=.runtime/slurms/slurm-%x-%j.err
set -euo pipefail
[[ ${SLURM_JOB_PARTITION:-} == debug && ${SLURM_JOB_NUM_NODES:-} == 1 ]] || exit 2
readonly ROOT_DIR=${SLURM_SUBMIT_DIR:?}
[[ ${AMPA_UCCL_DEBUG_CHECK_ID:?} =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
export AMPA_UCCL_DEBUG_CHECK_OUT="$ROOT_DIR/runs/$AMPA_UCCL_DEBUG_CHECK_ID"
[[ ! -e "$AMPA_UCCL_DEBUG_CHECK_OUT" && ! -L "$AMPA_UCCL_DEBUG_CHECK_OUT" ]] || exit 2
: "${AMPA_UCCL_DEBUG_EXPECTED_SOURCE_MANIFEST:?}" "${AMPA_UCCL_DEBUG_EXPECTED_SOURCE_SHA256:?}"
[[ "$AMPA_UCCL_DEBUG_EXPECTED_SOURCE_MANIFEST" == "$ROOT_DIR"/runs/* \
    && -f "$AMPA_UCCL_DEBUG_EXPECTED_SOURCE_MANIFEST" && ! -L "$AMPA_UCCL_DEBUG_EXPECTED_SOURCE_MANIFEST" \
    && "$AMPA_UCCL_DEBUG_EXPECTED_SOURCE_SHA256" =~ ^[0-9a-f]{64}$ ]] || exit 2
[[ $(sha256sum "$AMPA_UCCL_DEBUG_EXPECTED_SOURCE_MANIFEST" | cut -d ' ' -f 1) == "$AMPA_UCCL_DEBUG_EXPECTED_SOURCE_SHA256" ]] || exit 2
cd "$ROOT_DIR"
sha256sum --check "$AMPA_UCCL_DEBUG_EXPECTED_SOURCE_MANIFEST"
mkdir -p "$AMPA_UCCL_DEBUG_CHECK_OUT/logs"
printf '%s\n' "$SLURM_JOB_ID" > "$AMPA_UCCL_DEBUG_CHECK_OUT/job-id"
cp "$AMPA_UCCL_DEBUG_EXPECTED_SOURCE_MANIFEST" "$AMPA_UCCL_DEBUG_CHECK_OUT/source.sha256"
squeue -u "$USER"
srun --mpi=pmix --cpu-bind=none --container-remap-root --environment="$ROOT_DIR/setup/gh200-cuda130-runtime.toml" \
    --nodes=1 --ntasks=1 --output="$AMPA_UCCL_DEBUG_CHECK_OUT/logs/checks.out" \
    --error="$AMPA_UCCL_DEBUG_CHECK_OUT/logs/checks.err" bash -c 'set -euo pipefail
    cd "$WORKSPACE"
    export CUDA_VISIBLE_DEVICES=""
    export PYTHONPATH="$WORKSPACE:$WORKSPACE/profiling:$WORKSPACE/analysis${PYTHONPATH:+:$PYTHONPATH}"
    readonly CHECK_PY=/opt/venvs/megatron-bridge/bin/python
    sha256sum --check "$AMPA_UCCL_DEBUG_CHECK_OUT/source.sha256"
    for script in jobs/smoke/uccl_debug_checks.sbatch jobs/training/uccl_ep_debug_case.sbatch \
        jobs/training/qwen3_uccl_debug_worker.sh jobs/training/uccl_debug_coordinator.sh \
        jobs/analysis/uccl_debug_validate.sbatch; do
        bash -n "$script"
    done
    bash "$WORKSPACE/jobs/training/uccl_ep_debug_case.sbatch" --selfcheck
    "$CHECK_PY" -m py_compile profiling/run_pretrain_debug.py tools/uccl_ep_debug_validate.py \
        tests/test_uccl_debug_entrypoint.py tests/test_uccl_ep_debug_validate.py
    for pattern in test_uccl_debug_entrypoint.py test_uccl_ep_debug_validate.py; do
        "$CHECK_PY" -m unittest discover -s tests -p "$pattern" -v \
            > "$AMPA_UCCL_DEBUG_CHECK_OUT/logs/$pattern.log" 2>&1
        tail -n 5 "$AMPA_UCCL_DEBUG_CHECK_OUT/logs/$pattern.log"
    done
    sha256sum --check "$AMPA_UCCL_DEBUG_CHECK_OUT/source.sha256"
    printf "debug_checks_passed=true\nGPU_initialization_requested=false\n" > "$AMPA_UCCL_DEBUG_CHECK_OUT/PASS.txt"
    '
sha256sum --check "$AMPA_UCCL_DEBUG_CHECK_OUT/source.sha256"
