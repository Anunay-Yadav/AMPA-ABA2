#!/usr/bin/env bash
# Four matched legs share one physical allocation and independent C0 starts.
#SBATCH --job-name=ampa-qwen-all-steps-group2
#SBATCH --account=infra01
#SBATCH --partition=preemptable
#SBATCH --qos=normal
#SBATCH --constraint=gh
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --no-requeue
#SBATCH --output=.runtime/slurms/slurm-%x-%j.out
#SBATCH --error=.runtime/slurms/slurm-%x-%j.err

set -euo pipefail
[[ "${SLURM_JOB_PARTITION:-}" == preemptable ]] || exit 2
readonly ROOT_DIR="${SLURM_SUBMIT_DIR}"
readonly PAIR_ID="${AMPA_PAIR_ID:?}"
export AMPA_TOPOLOGY_VARIANT=cp2_ep8 AMPA_DATA_MODE=indexed AMPA_RECOMPUTE_MODE=none
export AMPA_PROFILE_MODE=off AMPA_NSYS_CAPTURE_STEPS=0 AMPA_TORCH_PROFILE_STEPS=0
export AMPA_LOG_INTERVAL=1 AMPA_WARMUP_STEPS=5 AMPA_MEASURED_STEPS=20
export AMPA_TORCH_PROFILE_RANKS=0,1,2,3,4,5,6,7 AMPA_SAMPLE_FINGERPRINTS=1
export AMPA_SEQUENCE_LENGTH=4096 AMPA_MICRO_BATCH_SIZE=1 AMPA_GLOBAL_BATCH_SIZE=64
# Accepted CP2 pair2 needed this allocator to avoid large loss-buffer fragmentation.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for routing in normal group8-top2; do
    unset AMPA_ROUTER_NUM_GROUPS AMPA_ROUTER_GROUP_TOPK
    if [[ "${routing}" == group8-top2 ]]; then
        export AMPA_ROUTER_NUM_GROUPS=8 AMPA_ROUTER_GROUP_TOPK=2
    fi
    for leg in torchprof-all control; do
        export AMPA_TORCH_PROFILE_ALL_STEPS=0
        [[ "${leg}" == control ]] || export AMPA_TORCH_PROFILE_ALL_STEPS=1
        export AMPA_RUN_ID="${PAIR_ID}-cp2-${routing}-${leg}"
        bash "${ROOT_DIR}/jobs/training/qwen3_all_steps_profile.sbatch"
    done
done

for routing in normal group8-top2; do
    export AMPA_RUN_DIR="${ROOT_DIR}/runs/${PAIR_ID}-cp2-${routing}-torchprof-all"
    squeue -u "$USER"
    srun --mpi=pmix --container-remap-root --environment="${ROOT_DIR}/setup/gh200-cuda130-runtime.toml" \
        --nodes=2 --ntasks=8 --ntasks-per-node=4 --gpus-per-task=1 \
        --output="${AMPA_RUN_DIR}/logs/all-step-analysis-rank-%t.out" \
        bash -c 'exec /opt/venvs/megatron-bridge/bin/python "$WORKSPACE/analysis/analyze_all_step_trace.py" --run-dir "$AMPA_RUN_DIR" --rank "$SLURM_PROCID"'
done
squeue -u "$USER"
srun --mpi=pmix --container-remap-root --environment="${ROOT_DIR}/setup/gh200-cuda130-runtime.toml" \
    --nodes=1 --ntasks=1 --gpus-per-task=1 \
    /opt/venvs/megatron-bridge/bin/python "${ROOT_DIR}/analysis/compare_all_step_profiles.py" \
    --runs-root "${ROOT_DIR}/runs" --pair-id "${PAIR_ID}"
