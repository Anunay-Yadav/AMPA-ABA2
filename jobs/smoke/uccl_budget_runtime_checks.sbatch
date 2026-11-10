#!/usr/bin/env bash
# Frozen-source QA, native Config checks, then one-owner/two-child Nsight smoke.
#SBATCH --job-name=ampa-uccl-budget-qa
#SBATCH --account=infra01
#SBATCH --partition=debug
#SBATCH --qos=normal
#SBATCH --constraint=gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --exclusive
#SBATCH --time=00:05:00
#SBATCH --no-requeue
#SBATCH --output=.runtime/slurms/slurm-%x-%j.out
#SBATCH --error=.runtime/slurms/slurm-%x-%j.err
set -euo pipefail
[[ ${SLURM_JOB_PARTITION:-} == debug && ${SLURM_JOB_NUM_NODES:-} == 1 && $# == 0 ]] || exit 2
readonly ROOT_DIR=${SLURM_SUBMIT_DIR:?}
[[ ${AMPA_UCCL_BUDGET_QA_ID:?} =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*$ ]] || exit 2
[[ ${AMPA_UCCL_NSYS_CAP_ROOT:?} == "$ROOT_DIR"/runs/* ]] || exit 2
: "${AMPA_UCCL_SWEEP_EXPECTED_SOURCE_MANIFEST:?}" "${AMPA_UCCL_SWEEP_EXPECTED_SOURCE_SHA256:?}"
[[ $AMPA_UCCL_SWEEP_EXPECTED_SOURCE_MANIFEST == "$ROOT_DIR"/runs/*
    && -f $AMPA_UCCL_SWEEP_EXPECTED_SOURCE_MANIFEST && ! -L $AMPA_UCCL_SWEEP_EXPECTED_SOURCE_MANIFEST
    && $AMPA_UCCL_SWEEP_EXPECTED_SOURCE_SHA256 =~ ^[0-9a-f]{64}$ ]] || exit 2
export AMPA_UCCL_BUDGET_QA_ROOT="$ROOT_DIR/runs/$AMPA_UCCL_BUDGET_QA_ID"
[[ ! -e $AMPA_UCCL_BUDGET_QA_ROOT && ! -L $AMPA_UCCL_BUDGET_QA_ROOT ]] || exit 2
mkdir "$AMPA_UCCL_BUDGET_QA_ROOT"
mkdir -p "$AMPA_UCCL_BUDGET_QA_ROOT/logs"
cp "$AMPA_UCCL_SWEEP_EXPECTED_SOURCE_MANIFEST" "$AMPA_UCCL_BUDGET_QA_ROOT/source.sha256"
printf '%s\n' "$SLURM_JOB_ID" > "$AMPA_UCCL_BUDGET_QA_ROOT/job-id"
export AMPA_UCCL_NSYS_CAP_ROOT AMPA_UCCL_SWEEP_EXPECTED_SOURCE_SHA256
export AMPA_UCCL_BUILD_ROOT="$ROOT_DIR/runs/uccl-ep-owned-build-20261101T1713Z"
export EFA_HOME="$AMPA_UCCL_BUILD_ROOT/not-efa" FI_MR_CACHE_MONITOR=disabled UCCL_EP_TRANSPORT=cxi
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/profiling:$ROOT_DIR/analysis:$AMPA_UCCL_BUILD_ROOT/site:$AMPA_UCCL_BUILD_ROOT/deps:$AMPA_UCCL_BUILD_ROOT/uccl/ep/deep_ep_wrapper:$ROOT_DIR/Megatron-LM:$ROOT_DIR/Megatron-Bridge/src${PYTHONPATH:+:$PYTHONPATH}"
case ${LD_LIBRARY_PATH:-} in
    /opt/amazon/efa/lib|/opt/amazon/efa/lib:*) ;;
    *) export LD_LIBRARY_PATH="/opt/amazon/efa/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
esac
squeue -u "$USER"
set +e
srun --mpi=pmix --cpu-bind=none --container-remap-root --environment="$ROOT_DIR/setup/gh200-cuda130-runtime.toml" \
    --nodes=1 --ntasks=1 --gpus-per-task=1 \
    --output="$AMPA_UCCL_BUDGET_QA_ROOT/logs/checks.out" --error="$AMPA_UCCL_BUDGET_QA_ROOT/logs/checks.err" \
    timeout --signal=TERM --kill-after=3s 245s /opt/venvs/megatron-bridge/bin/python "$ROOT_DIR/tools/check_uccl_budget_runtime.py" \
    --output-dir "$AMPA_UCCL_BUDGET_QA_ROOT" --capability-root "$AMPA_UCCL_NSYS_CAP_ROOT" \
    --source-manifest "$AMPA_UCCL_BUDGET_QA_ROOT/source.sha256" --source-sha256 "$AMPA_UCCL_SWEEP_EXPECTED_SOURCE_SHA256"
qa_status=$?
set -e
printf 'exit_code=%s\njob=%s\nfinished_utc=%s\n' "$qa_status" "$SLURM_JOB_ID" "$(date -u +%FT%TZ)" > "$AMPA_UCCL_BUDGET_QA_ROOT/exit.txt"
exit "$qa_status"
