#!/usr/bin/env bash
#SBATCH --job-name=ampa-uccl-reuse-checks
#SBATCH --account=infra01
#SBATCH --partition=debug
#SBATCH --qos=normal
#SBATCH --constraint=gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --exclusive
#SBATCH --mem=32G
#SBATCH --time=00:02:00
#SBATCH --no-requeue
#SBATCH --output=.runtime/slurms/slurm-%x-%j.out
#SBATCH --error=.runtime/slurms/slurm-%x-%j.err
set -euo pipefail
[[ ${SLURM_JOB_PARTITION:-} == debug && ${SLURM_JOB_NUM_NODES:-} == 1 ]] || exit 2
readonly ROOT_DIR=${SLURM_SUBMIT_DIR:?}
[[ ${AMPA_UCCL_REUSE_CHECK_ID:?} =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
export AMPA_UCCL_REUSE_CHECK_OUT="$ROOT_DIR/runs/$AMPA_UCCL_REUSE_CHECK_ID"
[[ ! -e "$AMPA_UCCL_REUSE_CHECK_OUT" ]] || exit 2
mkdir -p "$AMPA_UCCL_REUSE_CHECK_OUT/logs"
printf '%s\n' "$SLURM_JOB_ID" > "$AMPA_UCCL_REUSE_CHECK_OUT/job-id"
cd "$ROOT_DIR"
sha256sum profiling/run_pretrain_debug_reuse.py tools/uccl_ep_verified_inputs_reuse.py \
    jobs/training/uccl_ep_debug_reuse.sbatch jobs/training/qwen3_uccl_debug_reuse_worker.sh \
    jobs/training/uccl_debug_reuse_coordinator.sh jobs/smoke/uccl_debug_reuse_checks.sbatch \
    tests/test_uccl_debug_reuse.py > "$AMPA_UCCL_REUSE_CHECK_OUT/source.sha256"
squeue -u "$USER"
srun --mpi=pmix --cpu-bind=none --container-remap-root --environment="$ROOT_DIR/setup/gh200-cuda130-runtime.toml" \
    --nodes=1 --ntasks=1 --output="$AMPA_UCCL_REUSE_CHECK_OUT/logs/checks.out" \
    --error="$AMPA_UCCL_REUSE_CHECK_OUT/logs/checks.err" bash -c 'set -euo pipefail
    cd "$WORKSPACE"
    export CUDA_VISIBLE_DEVICES=""
    export PYTHONPATH="$WORKSPACE:$WORKSPACE/profiling:$WORKSPACE/analysis${PYTHONPATH:+:$PYTHONPATH}"
    readonly CHECK_PY=/opt/venvs/megatron-bridge/bin/python
    for script in jobs/training/uccl_ep_debug_reuse.sbatch jobs/training/qwen3_uccl_debug_reuse_worker.sh \
        jobs/training/uccl_debug_reuse_coordinator.sh jobs/smoke/uccl_debug_reuse_checks.sbatch; do
        bash -n "$script"
    done
    bash jobs/training/uccl_ep_debug_reuse.sbatch --selfcheck
    "$CHECK_PY" -m py_compile profiling/run_pretrain_debug_reuse.py tools/uccl_ep_verified_inputs_reuse.py
    "$CHECK_PY" -m unittest discover -s tests -p test_uccl_debug_reuse.py -v \
        > "$AMPA_UCCL_REUSE_CHECK_OUT/logs/test_uccl_debug_reuse.log" 2>&1
    tail -n 6 "$AMPA_UCCL_REUSE_CHECK_OUT/logs/test_uccl_debug_reuse.log"
    printf "reuse_launcher_checks_passed=true\n" > "$AMPA_UCCL_REUSE_CHECK_OUT/PASS.txt"
    '
