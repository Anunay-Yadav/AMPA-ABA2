#!/usr/bin/env bash
set -euo pipefail
[[ "${SLURM_JOB_PARTITION:-}" == preemptable && "${SLURM_JOB_NUM_NODES:-}" == 4 ]] || exit 2
readonly ROOT_DIR="${SLURM_SUBMIT_DIR}"
readonly TRAIN_PY=/opt/venvs/megatron-bridge/bin/python
cd "$ROOT_DIR"
# Container startup replaces the inherited search path. Bind the reviewed
# fabric before uv, torchrun or any worker imports a native library.
case "${LD_LIBRARY_PATH:-}" in
    /opt/amazon/efa/lib|/opt/amazon/efa/lib:*) ;;
    *) export LD_LIBRARY_PATH="/opt/amazon/efa/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
esac

if [[ "${1:-}" == --worker ]]; then
    [[ $# == 4 && "${WORLD_SIZE:-}" == 16 && "${LOCAL_WORLD_SIZE:-}" == 4 ]] || exit 2
    [[ "${LOCAL_RANK:-}" =~ ^[0-3]$ && "${RANK:-}" =~ ^([0-9]|1[0-5])$ && "${SLURM_LOCALID:-}" == 0 ]] || exit 2
    experiment="$2" leg="$3" run="$4"
    args=(--sample-fingerprints --routing-audit --audit-num-groups 4
        --launch-policy torchrun_node4 --audit-received-experts)
    if [[ "$leg" == profile ]]; then
        args+=(--torch-profile-all-steps --torch-profile-ranks 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15)
    else [[ "$leg" == control && "$AMPA_UCCL_COMPARISON_MODE" == full ]] || exit 2; fi
    if [[ "$AMPA_UCCL_COMPARISON_MODE" == pilot ]]; then args+=(--smoke-steps 2); fi
    case "$experiment" in
        ordinary) ;;
        grouped) args+=(--router-num-groups 4 --router-group-topk 2) ;;
        uccl|uccl_ordinary)
            # CUPTI buffer allocation can reenter libfabric's system memhooks
            # while its nonrecursive MR-cache mutex is held. Apply the same
            # supported workaround to both control and profile workers.
            export FI_MR_CACHE_MONITOR=disabled
            args+=(--communication-backend uccl_ep --uccl-build-dir "$AMPA_UCCL_BUILD_ROOT")
            if [[ "$experiment" == uccl ]]; then
                args+=(--router-num-groups 4 --router-group-topk 2)
            fi ;;
        *) exit 2 ;;
    esac
    printf 'AMPA_WORKER_ID rank=%s local_rank=%s node_rank=%s hostname=%s\n' "$RANK" "$LOCAL_RANK" "$SLURM_NODEID" "$(hostname)"
    exec "$TRAIN_PY" "$ROOT_DIR/profiling/run_pretrain.py" \
        --run-dir "$run" --pretrained-checkpoint "$AMPA_C0" --hf-revision "$AMPA_HF_REVISION" \
        --data-mode indexed --dataset-data-path "$AMPA_DATA_PATH" --dataset-manifest "$run/config/dataset_manifest.json" \
        --topology-variant cp1_ep16 --lightweight-profiling off --recompute-mode none \
        --sequence-length 4096 --micro-batch-size 1 --global-batch-size 64 \
        --warmup-steps 5 --measured-steps 20 --log-interval 1 --dataset-sampler-seed 1234 "${args[@]}"
elif [[ "${1:-}" == --node ]]; then
    [[ $# == 1 && "${SLURM_LOCALID:-}" == 0 && "${SLURM_NTASKS:-}" == 4 && "${SLURM_NODEID:-}" =~ ^[0-3]$ ]] || exit 2
    sha256sum -c "$AMPA_UCCL_COMPARISON_ROOT/source.sha256"
    sha256sum -c "$AMPA_UCCL_COMPARISON_ROOT/runtime-source.sha256"
    sha256sum -c "$AMPA_UCCL_BUILD_ROOT/binary.sha256"
    sha256sum -c "$AMPA_UCCL_BUILD_ROOT/libfabric_runtime.sha256"
    uv pip install --python "$TRAIN_PY" --no-deps --no-build-isolation \
        --editable "$ROOT_DIR/Megatron-LM" --editable "$ROOT_DIR/Megatron-Bridge"
    printf 'hostname=%s Slurm_node=%s Slurm_local=%s visible=%s\nLD_LIBRARY_PATH=%s\nEFA_HOME=%s\n' \
        "$(hostname)" "$SLURM_NODEID" "$SLURM_LOCALID" "${CUDA_VISIBLE_DEVICES:-}" "$LD_LIBRARY_PATH" "$EFA_HOME"
    cat /proc/net/route > "$AMPA_UCCL_COMPARISON_ROOT/logs/network-node-$SLURM_NODEID.txt"
    shopt -s nullglob globstar
    mark_case() {
        if [[ "$SLURM_NODEID" == 0 ]]; then
            local note="$(date -u '+%Y-%m-%d %H:%M UTC') job $SLURM_JOB_ID: $*"
            printf '%s\n' "$note" >> "$AMPA_UCCL_COMPARISON_ROOT/timeline.md"
            printf '%s\n' "$note" > "$AMPA_UCCL_COMPARISON_ROOT/CURRENT.md"
            printf '\n- %s\n' "$note" >> "$ROOT_DIR/progress/sessions/2026-11-01-uccl-ep-comparison/timeline.md"
            printf '\n- %s\n' "$note" >> "$ROOT_DIR/progress/sessions/2026-11-01-uccl-ep-comparison/session_state.md"
            printf '\nCompute milestone: %s\n' "$note" >> "$ROOT_DIR/ai/qwen-uccl-ep-comparison/iteration-01/STATUS.md"
        fi
    }
    while IFS=$'\t' read -r experiment leg run; do
        sha256sum -c "$AMPA_UCCL_COMPARISON_ROOT/source.sha256"
        sha256sum -c "$AMPA_UCCL_COMPARISON_ROOT/runtime-source.sha256"
        log_dir="$run/logs/torchrun-node-$SLURM_NODEID"
        mark_case "BEFORE compute $experiment/$leg; independent C0 start; run $run"
        "$TRAIN_PY" -m torch.distributed.run --nnodes=4 --nproc_per_node=4 \
            --node_rank="$SLURM_NODEID" --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
            --max_restarts=0 --log_dir="$log_dir" --redirects=3 --no-python \
            bash "$ROOT_DIR/jobs/training/qwen3_uccl_ep_worker.sh" --worker "$experiment" "$leg" "$run"
        if [[ "$experiment" == uccl || "$experiment" == uccl_ordinary ]]; then
            "$TRAIN_PY" "$ROOT_DIR/tools/uccl_ep_log_proof.py" --log-dir "$log_dir" --node-rank "$SLURM_NODEID" \
                --output "$run/raw/native-cxi-node-$SLURM_NODEID.json"
        fi
        # Retain original stream files for proof; copies let existing native
        # health/timing tools discover the one print_rank_last worker log.
        for local_rank in {0..3}; do
            stdout=("$log_dir"/**/attempt_0/"$local_rank"/stdout.log)
            stderr=("$log_dir"/**/attempt_0/"$local_rank"/stderr.log)
            [[ ${#stdout[@]} == 1 && ${#stderr[@]} == 1 ]] || exit 2
            global_rank=$((SLURM_NODEID * 4 + local_rank))
            [[ ! -e "$run/logs/rank-$global_rank.out" ]] || exit 2
            cat "${stdout[0]}" "${stderr[0]}" > "$run/logs/rank-$global_rank.out"
            sha256sum "${stdout[0]}" "${stderr[0]}" "$run/logs/rank-$global_rank.out" \
                >> "$run/logs/log-map-node-$SLURM_NODEID.sha256"
        done
        sha256sum -c "$AMPA_UCCL_COMPARISON_ROOT/source.sha256"
        sha256sum -c "$AMPA_UCCL_COMPARISON_ROOT/runtime-source.sha256"
        mark_case "AFTER compute $experiment/$leg; local torchrun workers exited0; log proof/copy complete; aggregate health/audit/trace gates remain pending"
        MASTER_PORT=$((MASTER_PORT + 1))
        export MASTER_PORT
    done < "$AMPA_UCCL_COMPARISON_ROOT/cases.tsv"
elif [[ "${1:-}" == --validate ]]; then
    [[ $# == 1 && "${AMPA_UCCL_WORKERS_SUCCEEDED:-}" == 1 ]] || exit 2
    while IFS=$'\t' read -r experiment leg run; do
        if [[ "$AMPA_UCCL_COMPARISON_MODE" == pilot ]]; then
            "$TRAIN_PY" "$ROOT_DIR/analysis/check_ep16_smoke.py" --run-dir "$run"
        else
            "$TRAIN_PY" "$ROOT_DIR/analysis/summarize_training_evidence.py" --run-dir "$run" --log-interval 1
            "$TRAIN_PY" "$ROOT_DIR/analysis/extract_step_timing.py" --run-dir "$run" \
                --warmup-steps 5 --measured-steps 20 --log-interval 1 --global-batch-size 64 --sequence-length 4096
        fi
    done < "$AMPA_UCCL_COMPARISON_ROOT/cases.tsv"
    sha256sum -c "$AMPA_UCCL_COMPARISON_ROOT/source.sha256"
    sha256sum -c "$AMPA_UCCL_COMPARISON_ROOT/runtime-source.sha256"
    sha256sum -c "$AMPA_UCCL_BUILD_ROOT/binary.sha256"
    "$TRAIN_PY" - <<'PY'
import hashlib
import json
import os
from pathlib import Path
from ampa_profile.provenance import _ampa_runtime_sources

root = Path(os.environ["SLURM_SUBMIT_DIR"])
output = Path(os.environ["AMPA_UCCL_COMPARISON_ROOT"])
acquisition = json.loads((output / "acquisition.json").read_text())
sources = _ampa_runtime_sources(root)
if sources["sha256"] != acquisition["runtime_source_sha256"]:
    raise ValueError("runtime source changed before acquisition completion")
lines = (Path(os.environ["AMPA_UCCL_BUILD_ROOT"]) / "binary.sha256").read_text().splitlines()
if len(lines) != 1:
    raise ValueError("one reviewed UCCL binary is required")
binary_hash, binary_name = lines[0].split(maxsplit=1)
binary = Path(binary_name.lstrip(" *"))
if not binary.is_absolute() or hashlib.sha256(binary.read_bytes()).hexdigest() != binary_hash or binary_hash != acquisition["binary_sha256"]:
    raise ValueError("UCCL binary changed before acquisition completion")
record = {"success": True, "workers_srun_returned_zero": True, "mode": os.environ["AMPA_UCCL_COMPARISON_MODE"],
    "job": os.environ["SLURM_JOB_ID"], "runtime_source_sha256": sources["sha256"], "binary_sha256": binary_hash,
    "cases_tsv_sha256": hashlib.sha256((output / "cases.tsv").read_bytes()).hexdigest()}
with (output / "acquisition_completed.json").open("x") as handle:
    json.dump(record, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY
else
    exit 2
fi
