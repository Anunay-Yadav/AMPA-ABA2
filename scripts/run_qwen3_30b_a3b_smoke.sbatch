#!/usr/bin/env bash
# Overlay the mounted local modules as editable packages, then run five Qwen3
# 30B-A3B mock-data optimizer steps on 2x4 GH200 GPUs:
#   sbatch scripts/run_qwen3_30b_a3b_smoke.sbatch

#SBATCH --job-name=qwen3-30b-a3b-gh200-smoke
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
#SBATCH --requeue
#SBATCH --output=.runtime/slurms/slurm-%x-%j.out
#SBATCH --error=.runtime/slurms/slurm-%x-%j.err

set -euo pipefail

# Slurm stages batch scripts under /var/spool/slurmd; SLURM_SUBMIT_DIR names
# the checkout that supplied this job and its Enroot environment definition.
readonly ROOT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)}"
readonly CONTAINER_ENV="${ROOT_DIR}/setup/gh200-cuda130-runtime.toml"

install_editable_overlays() {
    # The image owns Bridge's locked dependencies.  These only link the
    # mounted source trees; they neither resolve nor install dependencies.
    uv pip install --python /opt/venvs/megatron-bridge/bin/python --no-deps --no-build-isolation \
        --editable "${WORKSPACE}/Megatron-LM"
    uv pip install --python /opt/venvs/megatron-bridge/bin/python --no-deps --no-build-isolation \
        --editable "${WORKSPACE}/Megatron-Bridge"
}

install_editable_overlays_once_per_node() {
    local marker_base="${WORKSPACE}/.runtime/qwen3-30b-a3b-smoke/${SLURM_JOB_ID}/editable-${SLURM_STEP_ID}-${SLURM_NODEID}"
    local ready_marker="${marker_base}.ready"
    local failed_marker="${marker_base}.failed"

    # Tasks on the same node share the writable container layer.  Installing
    # the two editable packages on every task races over site-packages, so the
    # node's local rank zero performs that mutation and the other ranks wait.
    if [[ ${SLURM_LOCALID} == 0 ]]; then
        rm -f "${ready_marker}" "${failed_marker}"
        if install_editable_overlays; then
            touch "${ready_marker}"
        else
            touch "${failed_marker}"
            return 1
        fi
        return 0
    fi

    while [[ ! -e ${ready_marker} && ! -e ${failed_marker} ]]; do
        sleep 1
    done
    [[ -e ${ready_marker} ]] || {
        echo "Editable overlay installation failed on local rank zero." >&2
        return 1
    }
}

container_smoke() {
    export RANK="${SLURM_PROCID}"
    export WORLD_SIZE="${SLURM_NTASKS}"
    # --gpus-per-task=1 makes each task's allocated GPU device 0 inside its
    # container, even though SLURM_LOCALID is 0--3 on a four-GPU node.
    export LOCAL_RANK=0

    install_editable_overlays_once_per_node

    cd "${WORKSPACE}/Megatron-Bridge"
    /opt/venvs/megatron-bridge/bin/python \
        scripts/training/run_recipe.py \
        --recipe qwen3_30b_a3b_pretrain_8gpu_h100_bf16_config \
        --mode pretrain \
        --dataset mock \
        --max_steps 5 \
        --global_batch_size 8 \
        --micro_batch_size 1 \
        --seq_length 512 \
        --warmup_iters 1 \
        scheduler.lr_decay_iters=5 \
        model.attention_backend=auto \
        model.moe_token_dispatcher_type=alltoall \
        model.moe_flex_dispatcher_backend=null \
        validation.eval_iters=0 \
        validation.eval_interval=0 \
        checkpoint.load=null \
        checkpoint.save=null \
        logger.log_interval=1 \
        logger.tensorboard_dir=null
}

case "${1:-}" in
    --container-smoke)
        container_smoke
        exit 0
        ;;
    "")
        ;;
    *)
        echo "Unknown internal mode: $1" >&2
        exit 2
        ;;
esac

readonly RESULTS_DIR="${ROOT_DIR}/.runtime/qwen3-30b-a3b-smoke/${SLURM_JOB_ID}"

mkdir -p "${RESULTS_DIR}"

export MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)"
export MASTER_PORT="$((20000 + SLURM_JOB_ID % 20000))"

# Bridge initializes distributed state from the Slurm rank variables, so launch
# one rank per GPU.  Do not wrap this in torchrun: that would double-spawn.
srun --mpi=pmix --container-remap-root --environment="${CONTAINER_ENV}" \
    --nodes=2 --ntasks=8 --ntasks-per-node=4 --gpus-per-task=1 \
    --output="${RESULTS_DIR}/rank-%t.out" \
    bash "${ROOT_DIR}/scripts/run_qwen3_30b_a3b_smoke.sbatch" --container-smoke

rg -q -i 'iteration[[:space:]]+5' "${RESULTS_DIR}"/rank-*.out || {
    echo "The distributed job exited successfully but no fifth iteration was found in rank logs." >&2
    exit 1
}
if rg -i 'lm loss[^[:alpha:]]*(nan|inf)' "${RESULTS_DIR}"/rank-*.out; then
    echo "Detected non-finite LM loss in the smoke-test logs." >&2
    exit 1
fi

printf 'Qwen3-30B-A3B GH200 five-step smoke PASSED (job %s).\n' "${SLURM_JOB_ID}" \
    | tee "${RESULTS_DIR}/PASS"
