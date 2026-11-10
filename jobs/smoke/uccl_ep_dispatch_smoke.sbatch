#!/usr/bin/env bash
#SBATCH --job-name=ampa-uccl-dispatch-smoke
#SBATCH --account=infra01
#SBATCH --partition=preemptable
#SBATCH --qos=normal
#SBATCH --constraint=gh
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=64
#SBATCH --time=00:15:00
#SBATCH --no-requeue
#SBATCH --output=.runtime/slurms/slurm-%x-%j.out
#SBATCH --error=.runtime/slurms/slurm-%x-%j.err
set -euo pipefail
[[ "${SLURM_JOB_PARTITION:-}" == preemptable ]] || exit 2
export AMPA_UCCL_SMOKE_MODE="${AMPA_UCCL_SMOKE_MODE:-distributed}"
case "$AMPA_UCCL_SMOKE_MODE" in
    distributed) smoke_nodes=4; smoke_gpus=4 ;;
    preflight) smoke_nodes=1; smoke_gpus=1 ;;
    *) exit 2 ;;
esac
[[ "${SLURM_JOB_NUM_NODES:-}" == "$smoke_nodes" ]] || exit 2
readonly ROOT_DIR="${SLURM_SUBMIT_DIR}"
[[ "${AMPA_UCCL_SMOKE_ID:?}" =~ ^[a-zA-Z0-9._-]+$ ]] || exit 2
export AMPA_UCCL_SMOKE_ROOT="${ROOT_DIR}/runs/${AMPA_UCCL_SMOKE_ID}"
export AMPA_UCCL_BUILD_ROOT="${ROOT_DIR}/runs/uccl-ep-owned-build-20261101T1713Z"
export EFA_HOME="$AMPA_UCCL_BUILD_ROOT/not-efa"
[[ ! -e "$EFA_HOME" ]] || exit 2
[[ ! -e "$AMPA_UCCL_SMOKE_ROOT" && ! -L "$AMPA_UCCL_SMOKE_ROOT" ]] || exit 2
[[ -f "$AMPA_UCCL_BUILD_ROOT/binary.sha256" ]] || exit 2
mkdir -p "$AMPA_UCCL_SMOKE_ROOT"/{source,plain,observed,logs}
files=(tools/uccl_ep_dispatch_smoke.py tools/uccl_ep_log_proof.py
    tools/uccl_ep_source_identity.py
    jobs/smoke/uccl_ep_dispatch_smoke.sbatch profiling/ampa_profile/deep_trace.py
    profiling/ampa_profile/uccl_trace.py profiling/ampa_profile/routing_audit.py
    profiling/ampa_profile/uccl_identity.py tests/test_uccl_identity.py
    profiling/run_pretrain.py tests/test_all_step_profiling.py
    Megatron-LM/megatron/core/transformer/moe/fused_a2a.py
    Megatron-LM/megatron/core/transformer/moe/token_dispatcher.py)
files+=(runs/uccl-ep-owned-build-20261101T1713Z/verification.json
    runs/uccl-ep-owned-build-20261101T1713Z/binary.sha256
    runs/uccl-ep-owned-build-20261101T1713Z/source_identity.json
    runs/uccl-ep-owned-build-20261101T1713Z/source/uccl-borrowed-scratch.patch)
for relative in ep/include/uccl_proxy.hpp ep/src/uccl_proxy.cpp ep/src/uccl_ep.cc ep/bench/utils.py; do
    files+=("runs/uccl-ep-owned-build-20261101T1713Z/uccl/$relative")
done
for relative in "${files[@]}"; do
    mkdir -p "$AMPA_UCCL_SMOKE_ROOT/source/$(dirname "$relative")"
    cp "$ROOT_DIR/$relative" "$AMPA_UCCL_SMOKE_ROOT/source/$relative"
done
sha256sum "${files[@]/#/$ROOT_DIR/}" > "$AMPA_UCCL_SMOKE_ROOT/source.sha256"
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
export MASTER_PORT="$((21000 + SLURM_JOB_ID % 20000))"
export UCCL_EP_TRANSPORT=cxi CUDA_DEVICE_ORDER=PCI_BUS_ID OMP_NUM_THREADS=2
squeue -u "$USER"
srun --mpi=pmix --container-remap-root --environment="$ROOT_DIR/setup/gh200-cuda130-runtime.toml" \
    --nodes="$smoke_nodes" --ntasks="$smoke_nodes" --ntasks-per-node=1 --gpus-per-task="$smoke_gpus" --kill-on-bad-exit=1 \
    --output="$AMPA_UCCL_SMOKE_ROOT/logs/node-%t.out" \
    --error="$AMPA_UCCL_SMOKE_ROOT/logs/node-%t.err" bash -c 'set -euo pipefail
    cd "$WORKSPACE"
    readonly SMOKE_PY=/opt/venvs/megatron-bridge/bin/python
    export LD_LIBRARY_PATH="/opt/amazon/efa/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    sha256sum -c "$AMPA_UCCL_BUILD_ROOT/binary.sha256"
    sha256sum -c "$AMPA_UCCL_SMOKE_ROOT/source.sha256"
    uv pip install --python "$SMOKE_PY" --no-deps --no-build-isolation \
        --editable "$WORKSPACE/Megatron-LM" --editable "$WORKSPACE/Megatron-Bridge"
    export PYTHONPATH="$AMPA_UCCL_BUILD_ROOT/site:$AMPA_UCCL_BUILD_ROOT/deps:$AMPA_UCCL_BUILD_ROOT/uccl/ep/deep_ep_wrapper:$WORKSPACE:$WORKSPACE/profiling${PYTHONPATH:+:$PYTHONPATH}"
    "$SMOKE_PY" -m py_compile tools/uccl_ep_dispatch_smoke.py
    printf "node=%s Slurm_node=%s Slurm_local=%s visible=%s\n" "$(hostname)" "$SLURM_NODEID" "$SLURM_LOCALID" "${CUDA_VISIBLE_DEVICES:-}"
    cat /proc/net/route > "$AMPA_UCCL_SMOKE_ROOT/logs/network-node-$SLURM_NODEID.txt"
    "$SMOKE_PY" -m unittest discover -s tests -p test_all_step_profiling.py -v \
        > "$AMPA_UCCL_SMOKE_ROOT/logs/gpu-preflight-node-$SLURM_NODEID.log" 2>&1
    "$SMOKE_PY" -m unittest discover -s tests -p test_uccl_identity.py -v \
        > "$AMPA_UCCL_SMOKE_ROOT/logs/identity-checks-node-$SLURM_NODEID.log" 2>&1
    if [[ "$AMPA_UCCL_SMOKE_MODE" == preflight ]]; then exit 0; fi
    read -r expected_binary_sha expected_binary_path < "$AMPA_UCCL_BUILD_ROOT/binary.sha256"
    for mode in plain observed; do
        extra=()
        if [[ "$mode" == observed ]]; then extra+=(--trace-observer); fi
        log_dir="$AMPA_UCCL_SMOKE_ROOT/logs/$mode-node-$SLURM_NODEID"
        "$SMOKE_PY" -m torch.distributed.run --nnodes=4 --nproc_per_node=4 \
            --node_rank="$SLURM_NODEID" --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
            --max_restarts=0 --log_dir="$log_dir" --redirects=3 \
            "$WORKSPACE/tools/uccl_ep_dispatch_smoke.py" --output-dir "$AMPA_UCCL_SMOKE_ROOT/$mode" \
            --tokens 64 --hidden 2048 --cycles 2 --uccl-source "$AMPA_UCCL_BUILD_ROOT/uccl" \
            --expected-extension-sha256 "$expected_binary_sha" "${extra[@]}"
        "$SMOKE_PY" tools/uccl_ep_log_proof.py --log-dir "$log_dir" --node-rank "$SLURM_NODEID" \
            --output "$AMPA_UCCL_SMOKE_ROOT/$mode/native-cxi-node-$SLURM_NODEID.json"
        export MASTER_PORT="$((MASTER_PORT + 1))"
    done
    '
sha256sum -c "$AMPA_UCCL_SMOKE_ROOT/source.sha256"
printf 'UCCL_SMOKE_FINISHED mode=%s\n' "$AMPA_UCCL_SMOKE_MODE"
