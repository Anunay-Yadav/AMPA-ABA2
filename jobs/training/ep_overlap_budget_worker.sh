#!/usr/bin/env bash
# Each invocation owns one independently loaded, unprofiled UCCL case.
set -euo pipefail
[[ ${SLURM_JOB_PARTITION:-} == debug && ${SLURM_JOB_NUM_NODES:-} == 4 ]] || exit 2
readonly ROOT_DIR=${SLURM_SUBMIT_DIR:?}
readonly TRAIN_PY=/opt/venvs/megatron-bridge/bin/python
readonly PROTOCOL=ep_overlap_ordinary_off_budget_5_plus_5_v1
cd "$ROOT_DIR"
[[ ${AMPA_OVERLAP_ID:-} =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$
    && ${AMPA_OVERLAP_CASE:-} == uccl_ordinary && ${AMPA_OVERLAP_CAPTURE:-} == off
    && ${AMPA_OVERLAP_STEPS:-} =~ ^(2|10)$
    && ${AMPA_OVERLAP_BUDGET:-} =~ ^[1-9][0-9]*$
    && ${AMPA_OVERLAP_ROOT:-} == "$ROOT_DIR/runs/$AMPA_OVERLAP_ID"
    && ${AMPA_OVERLAP_RUN:-} == "$ROOT_DIR/runs/$AMPA_OVERLAP_ID-training"
    && ${AMPA_BUDGET_PAIR_ID:-} =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$
    && ${AMPA_BUDGET_PAIR_ROOT:-} == "$ROOT_DIR/runs/$AMPA_BUDGET_PAIR_ID"
    && ${AMPA_BUDGET_LEG:-} =~ ^(candidate|reference)$
    && ${AMPA_BUDGET_LEG_INDEX:-} =~ ^[01]$ ]] || exit 2
# Static ceiling only. The launcher consumes the qualified-device budget
# list, and every worker checks its actual device before native allocation.
(( AMPA_OVERLAP_BUDGET >= 4 && AMPA_OVERLAP_BUDGET <= 64 && AMPA_OVERLAP_BUDGET % 2 == 0 )) || exit 2
export AMPA_UCCL_REUSE_CONTEXT="$AMPA_OVERLAP_ROOT/input_reuse_context.json"
export CUDA_DEVICE_MAX_CONNECTIONS=32 PYTHONDONTWRITEBYTECODE=1 FI_MR_CACHE_MONITOR=disabled
case ${LD_LIBRARY_PATH:-} in
    /opt/amazon/efa/lib|/opt/amazon/efa/lib:*) ;;
    *) export LD_LIBRARY_PATH="/opt/amazon/efa/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
esac
warmup=5
measured=5
if [[ $AMPA_OVERLAP_STEPS == 2 ]]; then warmup=1; measured=1; fi

if [[ ${1:-} == --worker ]]; then
    [[ $# == 1 && ${WORLD_SIZE:-} == 16 && ${LOCAL_WORLD_SIZE:-} == 4
        && ${LOCAL_RANK:-} =~ ^[0-3]$ && ${RANK:-} =~ ^([0-9]|1[0-5])$ && ${SLURM_LOCALID:-} == 0 ]] || exit 2
    printf 'AMPA_WORKER_ID rank=%s local_rank=%s node_rank=%s hostname=%s case=uccl_ordinary capture=off budget=%s steps=%s CUDA_DEVICE_MAX_CONNECTIONS=%s\n' \
        "$RANK" "$LOCAL_RANK" "$SLURM_NODEID" "$(hostname)" "$AMPA_OVERLAP_BUDGET" "$AMPA_OVERLAP_STEPS" "$CUDA_DEVICE_MAX_CONNECTIONS"
    exec "$TRAIN_PY" "$ROOT_DIR/profiling/run_pretrain_ep_overlap_budget.py" \
        --run-dir "$AMPA_OVERLAP_RUN" --case uccl_ordinary --capture off \
        --communication-sms "$AMPA_OVERLAP_BUDGET" --execution-steps "$AMPA_OVERLAP_STEPS" \
        --pretrained-checkpoint "$AMPA_C0" --hf-revision "$AMPA_HF_REVISION" \
        --data-mode indexed --dataset-data-path "$AMPA_DATA_PATH" --dataset-manifest "$AMPA_OVERLAP_RUN/config/dataset_manifest.json" \
        --topology-variant cp1_ep16 --lightweight-profiling off --recompute-mode none \
        --sequence-length 4096 --micro-batch-size 1 --global-batch-size 64 \
        --warmup-steps "$warmup" --measured-steps "$measured" --log-interval 1 --dataset-sampler-seed 1234 \
        --sample-fingerprints --routing-audit --audit-num-groups 4 --audit-received-experts \
        --launch-policy torchrun_node4 --communication-backend uccl_ep --uccl-build-dir "$AMPA_UCCL_BUILD_ROOT"
elif [[ ${1:-} == --torchrun-parent ]]; then
    [[ $# == 1 && ${SLURM_LOCALID:-} == 0 && ${SLURM_NTASKS:-} == 4 && ${SLURM_NODEID:-} =~ ^[0-3]$ ]] || exit 2
    log_dir="$AMPA_OVERLAP_RUN/logs/torchrun-node-$SLURM_NODEID"
    command=("$TRAIN_PY" -m torch.distributed.run --nnodes=4 --nproc_per_node=4
        --node_rank="$SLURM_NODEID" --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT"
        --max_restarts=0 --log_dir="$log_dir" --redirects=3 --no-python
        bash "$ROOT_DIR/jobs/training/ep_overlap_budget_worker.sh" --worker)
    exec "$TRAIN_PY" - "$$" "$ROOT_DIR/jobs/training/ep_overlap_budget_worker.sh" "${command[@]}" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import sys

parent_pid = int(sys.argv[1])
if parent_pid <= 0 or os.getpid() != parent_pid:
    raise ValueError('torchrun launcher PID does not name this recorder process')
source = Path(sys.argv[2])
node = int(os.environ['SLURM_NODEID'])
# Keep the established parent-file schema, also used by earlier OFF cases.
# No Nsight collector or CUDA context is created by this stdlib recorder.
record = {'schema_version': 1, 'kind': 'ampa_nsight_torchrun_parent',
          'pid': parent_pid, 'job': os.environ['SLURM_JOB_ID'], 'node_rank': node,
          'role': 'torchrun_launcher', 'source_path': str(source),
          'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
          'capture': 'off', 'collector_started': False,
          'experiment_protocol': 'ep_overlap_ordinary_off_budget_5_plus_5_v1',
          'pair_id': os.environ['AMPA_BUDGET_PAIR_ID'], 'pair_leg': os.environ['AMPA_BUDGET_LEG'],
          'communication_sms': int(os.environ['AMPA_OVERLAP_BUDGET']),
          'cuda_device_max_connections': os.environ['CUDA_DEVICE_MAX_CONNECTIONS'],
          'command': sys.argv[3:]}
destination = Path(os.environ['AMPA_OVERLAP_RUN']) / 'nsight' / f'node-{node}-torchrun-parent.json'
with destination.open('x') as stream:
    json.dump(record, stream, indent=2, sort_keys=True)
    stream.write('\n')
os.execv(sys.argv[3], sys.argv[3:])
PY
elif [[ ${1:-} == --node ]]; then
    [[ $# == 1 && ${SLURM_LOCALID:-} == 0 && ${SLURM_NTASKS:-} == 4 && ${SLURM_NODEID:-} =~ ^[0-3]$ ]] || exit 2
    if [[ $SLURM_NODEID == 0 ]]; then
        "$TRAIN_PY" "$ROOT_DIR/profiling/run_pretrain_ep_overlap_budget.py" --prepare-inputs \
            --orchestration-dir "$AMPA_OVERLAP_ROOT" --run-dir "$AMPA_OVERLAP_RUN" \
            --case uccl_ordinary --capture off --communication-sms "$AMPA_OVERLAP_BUDGET" \
            --execution-steps "$AMPA_OVERLAP_STEPS" --warmup-steps "$warmup" --measured-steps "$measured"
        "$TRAIN_PY" - <<'PY'
import os
from pathlib import Path
import subprocess

root = Path(os.environ['SLURM_SUBMIT_DIR'])
out = Path(os.environ['AMPA_OVERLAP_ROOT']) / 'source'
for label, directory in (('megatron_lm', root / 'Megatron-LM'),
                         ('megatron_bridge', root / 'Megatron-Bridge')):
    for kind, arguments in (('revision', ['rev-parse', 'HEAD']),
                            ('status', ['status', '--porcelain=v1']),
                            ('diff', ['diff', '--no-ext-diff', '--binary', 'HEAD'])):
        with (out / f'launch_{label}_{kind}.txt').open('x') as stream:
            subprocess.run(['git', '-C', str(directory), *arguments], stdout=stream,
                           stderr=subprocess.STDOUT, check=True, timeout=30)
PY
        printf 'prepared_by_node=0\njob=%s\n' "$SLURM_JOB_ID" > "$AMPA_OVERLAP_ROOT/prepared.tmp"
        mv "$AMPA_OVERLAP_ROOT/prepared.tmp" "$AMPA_OVERLAP_ROOT/prepared"
    else
        deadline=$((SECONDS + 180))
        until [[ -f $AMPA_OVERLAP_ROOT/prepared ]]; do
            (( SECONDS < deadline )) || exit 3
            sleep 1
        done
    fi
    uv pip install --python "$TRAIN_PY" --no-deps --no-build-isolation \
        --editable "$ROOT_DIR/Megatron-LM" --editable "$ROOT_DIR/Megatron-Bridge"
    printf 'hostname=%s Slurm_node=%s Slurm_local=%s visible=%s\nLD_LIBRARY_PATH=%s\nEFA_HOME=%s\nCUDA_DEVICE_MAX_CONNECTIONS=%s\n' \
        "$(hostname)" "$SLURM_NODEID" "$SLURM_LOCALID" "${CUDA_VISIBLE_DEVICES:-}" "$LD_LIBRARY_PATH" "$EFA_HOME" "$CUDA_DEVICE_MAX_CONNECTIONS"
    cat /proc/net/route > "$AMPA_OVERLAP_ROOT/logs/network-node-$SLURM_NODEID.txt"
    log_dir="$AMPA_OVERLAP_RUN/logs/torchrun-node-$SLURM_NODEID"
    bash "$ROOT_DIR/jobs/training/ep_overlap_budget_worker.sh" --torchrun-parent
    if ! "$TRAIN_PY" "$ROOT_DIR/tools/uccl_ep_log_proof.py" --log-dir "$log_dir" --node-rank "$SLURM_NODEID" \
        --output "$AMPA_OVERLAP_RUN/raw/native-cxi-node-$SLURM_NODEID.json" \
        > "$AMPA_OVERLAP_RUN/logs/native-proof-node-$SLURM_NODEID.out" \
        2> "$AMPA_OVERLAP_RUN/logs/native-proof-node-$SLURM_NODEID.err"; then
        printf 'native_log_proof_status=pending_posthoc_analysis\n' > "$AMPA_OVERLAP_RUN/raw/native-cxi-node-$SLURM_NODEID-pending.txt"
    fi
    shopt -s nullglob globstar
    for local_rank in {0..3}; do
        stdout=("$log_dir"/**/attempt_0/"$local_rank"/stdout.log)
        stderr=("$log_dir"/**/attempt_0/"$local_rank"/stderr.log)
        [[ ${#stdout[@]} == 1 && ${#stderr[@]} == 1 ]] || exit 2
        global_rank=$((SLURM_NODEID * 4 + local_rank))
        [[ ! -e $AMPA_OVERLAP_RUN/logs/rank-$global_rank.out ]] || exit 2
        cat "${stdout[0]}" "${stderr[0]}" > "$AMPA_OVERLAP_RUN/logs/rank-$global_rank.out"
        sha256sum "${stdout[0]}" "${stderr[0]}" "$AMPA_OVERLAP_RUN/logs/rank-$global_rank.out" \
            >> "$AMPA_OVERLAP_RUN/logs/log-map-node-$SLURM_NODEID.sha256"
    done
elif [[ ${1:-} == --validate ]]; then
    [[ $# == 1 && ${AMPA_OVERLAP_WORKERS_SUCCEEDED:-} == 1 ]] || exit 2
    "$TRAIN_PY" - <<'PY'
import json
import os
from pathlib import Path
from analysis.summarize_training_evidence import summarize

run = Path(os.environ['AMPA_OVERLAP_RUN'])
record = summarize(run, steps=int(os.environ['AMPA_OVERLAP_STEPS']), interval=1)
with (run / 'analysis/real_training_evidence.json').open('x') as stream:
    json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
    stream.write('\n')
PY
    "$TRAIN_PY" "$ROOT_DIR/analysis/extract_step_timing.py" --run-dir "$AMPA_OVERLAP_RUN" \
        --warmup-steps "$warmup" --measured-steps "$measured" --log-interval 1 --global-batch-size 64 --sequence-length 4096
    "$TRAIN_PY" - <<'PY'
import json
import os
from pathlib import Path

out = Path(os.environ['AMPA_OVERLAP_ROOT'])
acquired = json.loads((out / 'acquisition.json').read_text())
steps = int(os.environ['AMPA_OVERLAP_STEPS'])
completed = {**acquired, 'success': True, 'workers_srun_returned_zero': True,
             'completed_updates': steps, 'training_health_and_timer_summary_complete': True,
             'cuda_device_max_connections': os.environ['CUDA_DEVICE_MAX_CONNECTIONS'],
             'qualification_only': steps == 2,
             'timing_window': {'warmup_steps': 1 if steps == 2 else 5,
                               'measured_steps': 1 if steps == 2 else 5},
             'native_nsys_reports': 0, 'fresh_checkpoint_payload_verified': False,
             'fresh_dataset_payload_verified': False, 'allocation_scope': 'shared_pair',
             'pair_id': os.environ['AMPA_BUDGET_PAIR_ID'], 'pair_leg': os.environ['AMPA_BUDGET_LEG'],
             'routing_acceptance': 'pending_posthoc_analysis', 'backend_acceptance': 'pending_posthoc_analysis',
             'trace_acceptance': 'not_collected', 'gpu_metrics_acceptance': 'not_collected',
             'comparison_accepted': False}
with (out / 'acquisition_completed.json').open('x') as stream:
    json.dump(completed, stream, indent=2, sort_keys=True, allow_nan=False)
    stream.write('\n')
PY
else
    exit 2
fi
