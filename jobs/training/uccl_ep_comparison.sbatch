#!/usr/bin/env bash
# Four backend/routing cases, each with a fresh C0 start per profiler setting.
#SBATCH --job-name=ampa-uccl-ep-comparison
#SBATCH --account=infra01
#SBATCH --partition=preemptable
#SBATCH --qos=normal
#SBATCH --constraint=gh
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=64
#SBATCH --time=05:00:00
#SBATCH --no-requeue
#SBATCH --output=.runtime/slurms/slurm-%x-%j.out
#SBATCH --error=.runtime/slurms/slurm-%x-%j.err

set -euo pipefail
[[ "${SLURM_JOB_PARTITION:-}" == preemptable && "${SLURM_JOB_NUM_NODES:-}" == 4 ]] || exit 2
readonly ROOT_DIR="${SLURM_SUBMIT_DIR}"
export AMPA_UCCL_COMPARISON_MODE="${AMPA_UCCL_COMPARISON_MODE:-pilot}"
[[ "$AMPA_UCCL_COMPARISON_MODE" == pilot || "$AMPA_UCCL_COMPARISON_MODE" == full ]] || exit 2
[[ "${AMPA_UCCL_COMPARISON_ID:?}" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
export AMPA_UCCL_COMPARISON_ROOT="$ROOT_DIR/runs/$AMPA_UCCL_COMPARISON_ID"
export AMPA_UCCL_BUILD_ROOT="$ROOT_DIR/runs/uccl-ep-owned-build-20261101T1713Z"
export AMPA_C0="$ROOT_DIR/checkpoint/megatron/qwen3-30b-a3b-base-c0"
export AMPA_HF_REVISION=1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9
export AMPA_DATA_MANIFEST="$ROOT_DIR/dataset/manifests/ep16-cache-20261011T1757Z-cp1-ep16.json"
export AMPA_DATA_PATH="0.55 $ROOT_DIR/dataset/indexed/qwen3-30b-a3b-base-realistic-mix-v2/web/qwen3_30b_a3b_base_realistic_mix_web 0.15 $ROOT_DIR/dataset/indexed/qwen3-30b-a3b-base-realistic-mix-v2/multilingual/qwen3_30b_a3b_base_realistic_mix_multilingual 0.15 $ROOT_DIR/dataset/indexed/qwen3-30b-a3b-base-realistic-mix-v2/code/qwen3_30b_a3b_base_realistic_mix_code 0.15 $ROOT_DIR/dataset/indexed/qwen3-30b-a3b-base-realistic-mix-v2/math/qwen3_30b_a3b_base_realistic_mix_math"
export AMPA_REVIEWED_INPUTS_PROOF="$AMPA_UCCL_COMPARISON_ROOT/reviewed_inputs.json"
export EFA_HOME="$AMPA_UCCL_BUILD_ROOT/not-efa"
case "${LD_LIBRARY_PATH:-}" in
    /opt/amazon/efa/lib|/opt/amazon/efa/lib:*) ;;
    *) export LD_LIBRARY_PATH="/opt/amazon/efa/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
esac
export UCCL_EP_TRANSPORT=cxi CUDA_DEVICE_ORDER=PCI_BUS_ID OMP_NUM_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$AMPA_UCCL_BUILD_ROOT/site:$AMPA_UCCL_BUILD_ROOT/deps:$AMPA_UCCL_BUILD_ROOT/uccl/ep/deep_ep_wrapper:$ROOT_DIR:$ROOT_DIR/profiling:$ROOT_DIR/analysis${PYTHONPATH:+:$PYTHONPATH}"

if [[ "${1:-}" == --preflight ]]; then
    [[ $# == 1 ]] || exit 2
    cd "$ROOT_DIR"
    readonly PREFLIGHT_PY=/opt/venvs/megatron-bridge/bin/python
    sha256sum -c "$AMPA_UCCL_BUILD_ROOT/binary.sha256"
    sha256sum -c "$AMPA_UCCL_BUILD_ROOT/libfabric_runtime.sha256"
    sha256sum -c "$AMPA_UCCL_COMPARISON_ROOT/source.sha256"
    "$PREFLIGHT_PY" - <<'PY'
import hashlib
import json
import os
from pathlib import Path
from ampa_profile.provenance import _ampa_runtime_sources
from tools.uccl_ep_source_identity import verify_uccl_source

root = Path(os.environ["SLURM_SUBMIT_DIR"])
build = Path(os.environ["AMPA_UCCL_BUILD_ROOT"])
output = Path(os.environ["AMPA_UCCL_COMPARISON_ROOT"])
sources = _ampa_runtime_sources(root)
uccl_source_identity = verify_uccl_source(build)
lines = (build / "binary.sha256").read_text().splitlines()
if len(lines) != 1:
    raise ValueError("one reviewed UCCL binary is required")
binary_hash, binary_name = lines[0].split(maxsplit=1)
binary = Path(binary_name.lstrip(" *"))
if not binary.is_absolute() or binary.parent != build / "site/uccl" or binary.is_symlink() or not binary.is_file():
    raise ValueError("build binary must use its isolated absolute path")
if hashlib.sha256(binary.read_bytes()).hexdigest() != binary_hash:
    raise ValueError("actual UCCL binary differs from build manifest")
with (output / "runtime-source.sha256").open("x") as handle:
    for relative, digest in sorted(sources["files"].items()):
        handle.write(f"{digest}  {root / relative}\n")
pilot_binding = None
if os.environ["AMPA_UCCL_COMPARISON_MODE"] == "full":
    proof_path = Path(os.environ.get("AMPA_UCCL_PILOT_PROOF", ""))
    if not proof_path.is_absolute() or proof_path.is_symlink() or not proof_path.is_file():
        raise ValueError("full mode requires an absolute regular AMPA_UCCL_PILOT_PROOF")
    payload = proof_path.read_bytes()
    proof = json.loads(payload)
    required = {"valid_pilot": True, "cases": ["ordinary", "grouped", "uccl_ordinary", "uccl"],
        "completed_updates": 2, "topology": "cp1_ep16", "valid_training_health": True,
        "valid_send_receive_audit": True, "valid_uccl_trace_capture": True,
        "binary_sha256": binary_hash, "runtime_source_sha256": sources["sha256"]}
    if any(type(proof.get(key)) is not type(value) or proof.get(key) != value for key, value in required.items()):
        raise ValueError("accepted pilot proof differs from required case/health/audit/trace/source/build evidence")
    pilot_binding = {"path": str(proof_path), "sha256": hashlib.sha256(payload).hexdigest()}
with (output / "acquisition.json").open("x") as handle:
    json.dump({"mode": os.environ["AMPA_UCCL_COMPARISON_MODE"], "job": os.environ["SLURM_JOB_ID"],
        "topology": "cp1_ep16", "launch_policy": "torchrun_node4", "binary_sha256": binary_hash,
        "runtime_source_sha256": sources["sha256"], "accepted_pilot_proof": pilot_binding,
        "uccl_source_identity": uccl_source_identity,
        "uccl_transport_policy": {"FI_MR_CACHE_MONITOR": "disabled",
            "scope": "control_and_profile_workers_before_native_initialization",
            "claim_limit": "Disables the system monitor; registration overhead can change."},
        "claim_limit": "Acquisition metadata does not replace runtime health, routing, transport or trace acceptance."},
        handle, indent=2, sort_keys=True)
PY
    "$PREFLIGHT_PY" "$ROOT_DIR/tools/ep16_reviewed_inputs.py" --root "$ROOT_DIR" \
        --reference-run "$ROOT_DIR/runs/ep16-all-steps-group4-top2-fresh-20261011T1957Z-cp1-normal-torchprof-all" \
        --manifest "$AMPA_DATA_MANIFEST" --output-dir "$AMPA_UCCL_COMPARISON_ROOT"
    exit 0
fi
[[ -z "${1:-}" && ! -e "$EFA_HOME" ]] || exit 2
[[ ! -e "$AMPA_UCCL_COMPARISON_ROOT" && ! -L "$AMPA_UCCL_COMPARISON_ROOT" ]] || exit 2
[[ -f "$AMPA_UCCL_BUILD_ROOT/binary.sha256" && -f "$AMPA_DATA_MANIFEST" ]] || exit 2
if [[ "$AMPA_UCCL_COMPARISON_MODE" == full ]]; then : "${AMPA_UCCL_PILOT_PROOF:?}"; fi
readonly SESSION_DIR="$ROOT_DIR/progress/sessions/2026-11-01-uccl-ep-comparison"
readonly TASK_STATUS="$ROOT_DIR/ai/qwen-uccl-ep-comparison/iteration-01/STATUS.md"
[[ -f "$SESSION_DIR/timeline.md" && -f "$SESSION_DIR/session_state.md" && -f "$TASK_STATUS" ]] || exit 2
mkdir -p "$AMPA_UCCL_COMPARISON_ROOT"/{source,logs}
for relative in jobs/training/uccl_ep_comparison.sbatch jobs/training/qwen3_uccl_ep_worker.sh tools/uccl_ep_log_proof.py; do
    mkdir -p "$AMPA_UCCL_COMPARISON_ROOT/source/$(dirname "$relative")"
    cp "$ROOT_DIR/$relative" "$AMPA_UCCL_COMPARISON_ROOT/source/$relative"
    sha256sum "$ROOT_DIR/$relative" >> "$AMPA_UCCL_COMPARISON_ROOT/source.sha256"
done
sha256sum "$AMPA_UCCL_BUILD_ROOT/binary.sha256" "$AMPA_UCCL_BUILD_ROOT/libfabric_runtime.sha256" \
    "$AMPA_UCCL_BUILD_ROOT/source_identity.json" "$AMPA_UCCL_BUILD_ROOT/source/uccl-borrowed-scratch.patch" \
    "$AMPA_UCCL_BUILD_ROOT/verification.json" \
    >> "$AMPA_UCCL_COMPARISON_ROOT/source.sha256"
for relative in ep/include/uccl_proxy.hpp ep/src/uccl_proxy.cpp ep/src/uccl_ep.cc ep/bench/utils.py; do
    sha256sum "$AMPA_UCCL_BUILD_ROOT/uccl/$relative" >> "$AMPA_UCCL_COMPARISON_ROOT/source.sha256"
done
record() {
    local note="$(date -u '+%Y-%m-%d %H:%M UTC') job $SLURM_JOB_ID: $*"
    printf '%s\n' "$note" >> "$AMPA_UCCL_COMPARISON_ROOT/timeline.md"
    printf '%s\n' "$note" > "$AMPA_UCCL_COMPARISON_ROOT/CURRENT.md"
    printf '\n- %s\n' "$note" >> "$SESSION_DIR/timeline.md"
    printf '\n- %s\n' "$note" >> "$SESSION_DIR/session_state.md"
    printf '\nCompute milestone: %s\n' "$note" >> "$TASK_STATUS"
}
complete=0
on_exit() {
    local status=$?
    trap - EXIT
    if (( status != 0 || complete != 1 )); then
        record "EXIT status=$status; acquisition incomplete; preserve all artifacts"
    fi
    exit "$status"
}
trap on_exit EXIT
record "START $AMPA_UCCL_COMPARISON_MODE on $SLURM_JOB_NODELIST; fresh C0 starts with common node-visible launch/environment"
legs=(profile)
[[ "$AMPA_UCCL_COMPARISON_MODE" == pilot ]] || legs=(control profile)
for leg in "${legs[@]}"; do
    for experiment in ordinary grouped uccl_ordinary uccl; do
        run="$ROOT_DIR/runs/$AMPA_UCCL_COMPARISON_ID-$experiment-$leg"
        [[ ! -e "$run" && ! -L "$run" ]] || exit 2
        mkdir -p "$run"/{config,raw,metrics,analysis,plots,traces,logs/wandb}
        cp "$AMPA_DATA_MANIFEST" "$run/config/dataset_manifest.json"
        cp "$AMPA_C0/ampa_c0_manifest.json" "$run/config/c0_manifest.json"
        cp "$ROOT_DIR/configs/model/qwen3_30b_a3b_base.json" "$run/config/model_manifest.json"
        printf '%s\t%s\t%s\n' "$experiment" "$leg" "$run" >> "$AMPA_UCCL_COMPARISON_ROOT/cases.tsv"
    done
done
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
export MASTER_PORT="$((21000 + SLURM_JOB_ID % 20000))"
record "BEFORE compute input/pilot gate; current source snapshot and immutable payload verification"
squeue -u "$USER"
srun --mpi=pmix --container-remap-root --environment="$ROOT_DIR/setup/gh200-cuda130-runtime.toml" \
    --nodes=1 --ntasks=1 --gpus-per-task=1 --output="$AMPA_UCCL_COMPARISON_ROOT/logs/preflight.out" \
    --error="$AMPA_UCCL_COMPARISON_ROOT/logs/preflight.err" bash "$ROOT_DIR/jobs/training/uccl_ep_comparison.sbatch" --preflight
record "AFTER compute input/pilot gate passed; BEFORE compute sequential fresh worker cases (see cases.tsv)"
squeue -u "$USER"
srun --mpi=pmix --container-remap-root --environment="$ROOT_DIR/setup/gh200-cuda130-runtime.toml" \
    --nodes=4 --ntasks=4 --ntasks-per-node=1 --gpus-per-task=4 --kill-on-bad-exit=1 \
    --output="$AMPA_UCCL_COMPARISON_ROOT/logs/node-%t.out" --error="$AMPA_UCCL_COMPARISON_ROOT/logs/node-%t.err" \
    bash "$ROOT_DIR/jobs/training/qwen3_uccl_ep_worker.sh" --node
export AMPA_UCCL_WORKERS_SUCCEEDED=1
record "AFTER compute all worker cases exited successfully; BEFORE compute existing health/timing checks"
squeue -u "$USER"
srun --mpi=pmix --container-remap-root --environment="$ROOT_DIR/setup/gh200-cuda130-runtime.toml" \
    --nodes=1 --ntasks=1 --gpus-per-task=1 --output="$AMPA_UCCL_COMPARISON_ROOT/logs/health-timing.out" \
    --error="$AMPA_UCCL_COMPARISON_ROOT/logs/health-timing.err" \
    bash "$ROOT_DIR/jobs/training/qwen3_uccl_ep_worker.sh" --validate
sha256sum -c "$AMPA_UCCL_COMPARISON_ROOT/source.sha256"
sha256sum -c "$AMPA_UCCL_COMPARISON_ROOT/runtime-source.sha256"
complete=1
record "ACQUISITION COMPLETE; existing health/timing passed. Backend-aware all-source receive, endpoint and correlated trace acceptance remain separate gates; no performance result accepted here"
