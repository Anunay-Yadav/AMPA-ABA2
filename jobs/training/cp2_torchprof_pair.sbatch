#!/usr/bin/env bash
# Same-allocation CP2 control/trace pair. The existing AMPA training launcher
# remains authoritative; this file only gives its two runs identical hosts.

#SBATCH --job-name=ampa-cp2-torchprof-pair
#SBATCH --account=infra01
#SBATCH --partition=preemptable
#SBATCH --qos=normal
#SBATCH --constraint=gh
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:30:00
#SBATCH --no-requeue
#SBATCH --output=.runtime/slurms/slurm-%x-%j.out
#SBATCH --error=.runtime/slurms/slurm-%x-%j.err

set -euo pipefail
[[ "${SLURM_JOB_PARTITION:-}" == preemptable ]] || { echo "Require preemptable partition" >&2; exit 2; }
: "${AMPA_C0:?}"
: "${AMPA_HF_REVISION:?}"
: "${AMPA_DATA_MANIFEST:?}"
: "${AMPA_DATA_PATH:?}"
readonly ROOT_DIR="${SLURM_SUBMIT_DIR}"
readonly PAIR_ID="${AMPA_PAIR_ID:-cp2-torchprof-pair-${SLURM_JOB_ID}}"
readonly FIRST="${AMPA_PAIR_FIRST:-control}"
[[ "${FIRST}" == control || "${FIRST}" == trace ]] || { echo "AMPA_PAIR_FIRST must be control or trace" >&2; exit 2; }

run_leg() {
    local leg="$1"
    export AMPA_TOPOLOGY_VARIANT=cp2_ep8
    export AMPA_DATA_MODE=indexed
    export AMPA_RECOMPUTE_MODE=none
    export AMPA_PROFILE_MODE=off
    export AMPA_NSYS_CAPTURE_STEPS=0
    export AMPA_LOG_INTERVAL=1
    export AMPA_WARMUP_STEPS=5
    export AMPA_MEASURED_STEPS=20
    export AMPA_TORCH_PROFILE_RANKS=0,1,2,3,4,5,6,7
    export AMPA_SAMPLE_FINGERPRINTS=1
    export AMPA_TORCH_PROFILE_STEPS=0
    if [[ "${leg}" == trace ]]; then
        export AMPA_TORCH_PROFILE_STEPS=1
        export AMPA_RUN_ID="${PAIR_ID}-cp2-torchprof"
    else
        export AMPA_RUN_ID="${PAIR_ID}-cp2-control"
    fi
    bash "${ROOT_DIR}/jobs/training/qwen3_30b_a3b_base_ep8_profile.sbatch"
}

if [[ "${FIRST}" == control ]]; then
    run_leg control
    run_leg trace
else
    run_leg trace
    run_leg control
fi

srun --mpi=pmix --container-remap-root \
    --environment="${ROOT_DIR}/setup/gh200-cuda130-runtime.toml" \
    --nodes=1 --ntasks=1 --gpus-per-task=1 \
    /opt/venvs/megatron-bridge/bin/python "${ROOT_DIR}/analysis/compare_deep_trace_overhead.py" \
    --control-run "${ROOT_DIR}/runs/${PAIR_ID}-cp2-control" \
    --trace-run "${ROOT_DIR}/runs/${PAIR_ID}-cp2-torchprof"
printf '{"pair_order": "%s", "slurm_job_id": "%s"}\n' "${FIRST}" "${SLURM_JOB_ID}" \
    > "${ROOT_DIR}/runs/${PAIR_ID}-cp2-torchprof/analysis/pair_order.json"
