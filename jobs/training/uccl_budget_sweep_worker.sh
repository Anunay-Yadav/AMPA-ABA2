#!/usr/bin/env bash
set -euo pipefail
[[ ${SLURM_JOB_PARTITION:-} == debug && ${SLURM_JOB_NUM_NODES:-} == 4 ]] || exit 2
readonly ROOT_DIR=${SLURM_SUBMIT_DIR:?}
readonly TRAIN_PY=/opt/venvs/megatron-bridge/bin/python
cd "$ROOT_DIR"
[[ ${AMPA_UCCL_SWEEP_SMS:-} =~ ^(20|12|32|40)$ && ${AMPA_UCCL_DEBUG_CASE:-} == uccl_ordinary && ${AMPA_UCCL_DEBUG_LEG:-} == control ]] || exit 2
export AMPA_UCCL_REUSE_CONTEXT="$AMPA_UCCL_COMPARISON_ROOT/input_reuse_context.json"
case ${LD_LIBRARY_PATH:-} in
    /opt/amazon/efa/lib|/opt/amazon/efa/lib:*) ;;
    *) export LD_LIBRARY_PATH="/opt/amazon/efa/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
esac
export FI_MR_CACHE_MONITOR=disabled PYTHONDONTWRITEBYTECODE=1

if [[ ${1:-} == --worker ]]; then
    [[ $# == 1 && ${WORLD_SIZE:-} == 16 && ${LOCAL_WORLD_SIZE:-} == 4
        && ${LOCAL_RANK:-} =~ ^[0-3]$ && ${RANK:-} =~ ^([0-9]|1[0-5])$ && ${SLURM_LOCALID:-} == 0 ]] || exit 2
    printf 'AMPA_WORKER_ID rank=%s local_rank=%s node_rank=%s hostname=%s budget=%s\n' "$RANK" "$LOCAL_RANK" "$SLURM_NODEID" "$(hostname)" "$AMPA_UCCL_SWEEP_SMS"
    exec "$TRAIN_PY" "$ROOT_DIR/profiling/run_pretrain_uccl_budget.py" \
        --run-dir "$AMPA_UCCL_SWEEP_RUN" --pretrained-checkpoint "$AMPA_C0" --hf-revision "$AMPA_HF_REVISION" \
        --data-mode indexed --dataset-data-path "$AMPA_DATA_PATH" --dataset-manifest "$AMPA_UCCL_SWEEP_RUN/config/dataset_manifest.json" \
        --topology-variant cp1_ep16 --lightweight-profiling off --recompute-mode none \
        --sequence-length 4096 --micro-batch-size 1 --global-batch-size 64 \
        --warmup-steps 5 --measured-steps 5 --log-interval 1 --dataset-sampler-seed 1234 \
        --sample-fingerprints --routing-audit --audit-num-groups 4 --launch-policy torchrun_node4 \
        --audit-received-experts --communication-backend uccl_ep --uccl-build-dir "$AMPA_UCCL_BUILD_ROOT" \
        --communication-sms "$AMPA_UCCL_SWEEP_SMS"
elif [[ ${1:-} == --torchrun-parent ]]; then
    [[ $# == 1 && ${SLURM_LOCALID:-} == 0 && ${SLURM_NTASKS:-} == 4 && ${SLURM_NODEID:-} =~ ^[0-3]$ ]] || exit 2
    log_dir="$AMPA_UCCL_SWEEP_RUN/logs/torchrun-node-$SLURM_NODEID"
    torchrun_command=("$TRAIN_PY" -m torch.distributed.run --nnodes=4 --nproc_per_node=4
        --node_rank="$SLURM_NODEID" --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT"
        --max_restarts=0 --log_dir="$log_dir" --redirects=3 --no-python
        bash "$ROOT_DIR/jobs/training/uccl_budget_sweep_worker.sh" --worker)
    exec "$TRAIN_PY" - "$$" "$ROOT_DIR/jobs/training/uccl_budget_sweep_worker.sh" "${torchrun_command[@]}" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import sys

parent_pid = int(sys.argv[1])
if parent_pid <= 0 or os.getpid() != parent_pid:
    raise ValueError('torchrun launcher PID does not name this recorder process')
source = Path(sys.argv[2])
if not source.is_absolute() or not source.is_file():
    raise ValueError('torchrun launcher source must be an absolute regular file')
node = int(os.environ['SLURM_NODEID'])
record = {'schema_version': 1, 'kind': 'ampa_nsight_torchrun_parent',
    'pid': parent_pid, 'job': os.environ['SLURM_JOB_ID'], 'node_rank': node,
    'role': 'torchrun_launcher', 'source_path': str(source),
    'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(), 'command': sys.argv[3:]}
destination = Path(os.environ['AMPA_UCCL_SWEEP_RUN']) / 'nsight' / f'node-{node}-torchrun-parent.json'
with destination.open('x') as stream:
    json.dump(record, stream, indent=2, sort_keys=True)
    stream.write('\n')
os.execv(sys.argv[3], sys.argv[3:])
PY
elif [[ ${1:-} == --node ]]; then
    [[ $# == 1 && ${SLURM_LOCALID:-} == 0 && ${SLURM_NTASKS:-} == 4 && ${SLURM_NODEID:-} =~ ^[0-3]$ ]] || exit 2
    if [[ $SLURM_NODEID == 0 ]]; then
        "$TRAIN_PY" "$ROOT_DIR/tools/uccl_ep_verified_inputs_reuse.py" --prepare \
            --orchestration-dir "$AMPA_UCCL_COMPARISON_ROOT" --run-dir "$AMPA_UCCL_SWEEP_RUN" --case uccl_ordinary --leg control
        "$TRAIN_PY" - <<'PY'
import json
import os
from pathlib import Path
import subprocess

out = Path(os.environ['AMPA_UCCL_COMPARISON_ROOT'])
run = Path(os.environ['AMPA_UCCL_SWEEP_RUN'])
cap = Path(os.environ['AMPA_UCCL_NSYS_CAP_ROOT'])
proof = json.loads((cap / 'capabilities.json').read_text())
if proof.get('native_report_generated') is not True or not proof.get('nsys_path'):
    raise ValueError('completed native Nsight capability proof is required')
contract = {'schema_version': 1, 'experiment_protocol': 'uccl_ordinary_budget_nsys_5_plus_5_v1',
    'job': os.environ['SLURM_JOB_ID'], 'partition': 'debug', 'requested_time': '00:21:00',
    'communication_sms': int(os.environ['AMPA_UCCL_SWEEP_SMS']), 'case': 'uccl_ordinary',
    'kineto_profiler_enabled': False, 'nsight_systems_enabled': True,
    'torch_profile_all_steps': False, 'torch_profile_steps': 0,
    'capture_step_indices': list(range(5, 10)), 'capture_ranks': list(range(16)),
    'nsight_capture': 'cudaProfilerApi; measured updates6..10; all16ranks; oneownerpernode',
    'nsight_capability_root': str(cap), 'gpu_metrics_requested': proof.get('gpu_metrics_collected') is True,
    'gpu_metrics_device_option': proof.get('gpu_metrics_device_option'),
    'gpu_metrics_acceptance': 'pending_actual_report_analysis',
    'warmup_steps': 5, 'measured_steps': 5, 'completed_updates': 10,
    'train_iters': 25, 'cache_budget_samples': 1600, 'exit_interval': 10,
    'run_dir': str(run), 'timing_kind': 'Nsight_instrumented_diagnostic',
    'input_verification': 'reused_historical_immutable_payload_proof',
    'physical_placement': 'record_actual_runtime; no_assumed_match',
    'source_policy': 'record_current_native_revisions_and_diff; no_historical_source_equivalence_claim'}
for dest in (out / 'nsight_budget_contract.json', run / 'config/nsight_budget_contract.json'):
    with dest.open('x') as f:
        json.dump(contract, f, indent=2, sort_keys=True)
        f.write('\n')
acquisition_path = out / 'acquisition.json'
acquisition = json.loads(acquisition_path.read_text())
acquisition['legacy_input_reuse_protocol'] = acquisition.get('experiment_protocol')
acquisition.update(contract)
acquisition['comparison_accepted'] = False
acquisition_path.write_text(json.dumps(acquisition, indent=2, sort_keys=True) + '\n')
for label, directory in (('megatron_lm', Path(os.environ['SLURM_SUBMIT_DIR']) / 'Megatron-LM'),
                         ('megatron_bridge', Path(os.environ['SLURM_SUBMIT_DIR']) / 'Megatron-Bridge')):
    for kind, args in (('revision', ['rev-parse', 'HEAD']), ('status', ['status', '--porcelain=v1']),
                       ('diff', ['diff', '--no-ext-diff', '--binary', 'HEAD'])):
        with (out / 'source' / f'{label}_{kind}.txt').open('x') as f:
            subprocess.run(['git', '-C', str(directory), *args], stdout=f, stderr=subprocess.STDOUT,
                           check=True, timeout=30)
PY
        printf 'job=%s\n' "$SLURM_JOB_ID" > "$AMPA_UCCL_COMPARISON_ROOT/prepared.tmp"
        mv "$AMPA_UCCL_COMPARISON_ROOT/prepared.tmp" "$AMPA_UCCL_COMPARISON_ROOT/prepared"
    else
        until [[ -f "$AMPA_UCCL_COMPARISON_ROOT/prepared" ]]; do sleep 1; done
    fi
    uv pip install --python "$TRAIN_PY" --no-deps --no-build-isolation --editable "$ROOT_DIR/Megatron-LM" --editable "$ROOT_DIR/Megatron-Bridge"
    cat /proc/net/route > "$AMPA_UCCL_COMPARISON_ROOT/logs/network-node-$SLURM_NODEID.txt"
    # Build an argv array from the capability-proven local CLI. This collector
    # owns all four torchrun children on this node; workers never run Kineto.
    mapfile -d '' -t nsys_command < <("$TRAIN_PY" - <<'PY'
import json
import os
from pathlib import Path
import sys

cap = Path(os.environ['AMPA_UCCL_NSYS_CAP_ROOT'])
proof = json.loads((cap / 'capabilities.json').read_text())
commands = json.loads((cap / 'commands.json').read_text())
native = next(x for x in commands if x['name'] == 'capture_native')
argv = native['argv']
output_index = next(i for i, x in enumerate(argv) if x.startswith('--output='))
args = [proof['nsys_path'], 'profile', *argv[2:output_index]]
if proof.get('gpu_metrics_collected') is True:
    recorded = next(x for x in commands if x['name'] == 'capture_gpu_metrics')['argv']
    args += [x for x in recorded if x.startswith('--gpu-metrics-')]
run = Path(os.environ['AMPA_UCCL_SWEEP_RUN'])
node = os.environ['SLURM_NODEID']
args.append(f'--output={run / "nsight" / ("node-" + node + "-native")}')
record = {'job': os.environ['SLURM_JOB_ID'], 'node_rank': int(node),
          'collector_argv_prefix': args, 'capability_root': str(cap),
          'gpu_metrics_requested': proof.get('gpu_metrics_collected') is True}
with (run / 'nsight' / f'node-{node}-launch.json').open('x') as f:
    json.dump(record, f, indent=2, sort_keys=True)
    f.write('\n')
sys.stdout.buffer.write(b''.join(os.fsencode(x) + b'\0' for x in args))
PY
    )
    [[ ${#nsys_command[@]} -ge 5 ]] || exit 2
    log_dir="$AMPA_UCCL_SWEEP_RUN/logs/torchrun-node-$SLURM_NODEID"
    set +e
    "${nsys_command[@]}" bash "$ROOT_DIR/jobs/training/uccl_budget_sweep_worker.sh" --torchrun-parent \
        > "$AMPA_UCCL_SWEEP_RUN/nsight/node-$SLURM_NODEID-collector.out" \
        2> "$AMPA_UCCL_SWEEP_RUN/nsight/node-$SLURM_NODEID-collector.err"
    capture_status=$?
    set -e
    printf 'exit_code=%s\njob=%s\nnode=%s\n' "$capture_status" "$SLURM_JOB_ID" "$SLURM_NODEID" > "$AMPA_UCCL_SWEEP_RUN/nsight/node-$SLURM_NODEID-exit.txt"
    (( capture_status == 0 )) || exit "$capture_status"
    [[ -s "$AMPA_UCCL_SWEEP_RUN/nsight/node-$SLURM_NODEID-native.nsys-rep" ]] || exit 3
    "$TRAIN_PY" "$ROOT_DIR/tools/uccl_ep_log_proof.py" --log-dir "$log_dir" --node-rank "$SLURM_NODEID" \
        --output "$AMPA_UCCL_SWEEP_RUN/raw/native-cxi-node-$SLURM_NODEID.json" \
        > "$AMPA_UCCL_SWEEP_RUN/logs/native-proof-node-$SLURM_NODEID.out" \
        2> "$AMPA_UCCL_SWEEP_RUN/logs/native-proof-node-$SLURM_NODEID.err" \
        || printf 'pending_posthoc_analysis\n' > "$AMPA_UCCL_SWEEP_RUN/raw/native-cxi-node-$SLURM_NODEID-pending.txt"
    shopt -s nullglob globstar
    for local_rank in {0..3}; do
        stdout=("$log_dir"/**/attempt_0/"$local_rank"/stdout.log)
        stderr=("$log_dir"/**/attempt_0/"$local_rank"/stderr.log)
        [[ ${#stdout[@]} == 1 && ${#stderr[@]} == 1 ]] || exit 2
        global_rank=$((SLURM_NODEID * 4 + local_rank))
        [[ ! -e "$AMPA_UCCL_SWEEP_RUN/logs/rank-$global_rank.out" ]] || exit 2
        cat "${stdout[0]}" "${stderr[0]}" > "$AMPA_UCCL_SWEEP_RUN/logs/rank-$global_rank.out"
    done
elif [[ ${1:-} == --validate ]]; then
    [[ $# == 1 && ${AMPA_UCCL_WORKERS_SUCCEEDED:-} == 1 ]] || exit 2
    "$TRAIN_PY" - <<'PY'
import json
import os
from pathlib import Path
from analysis.summarize_training_evidence import summarize

run = Path(os.environ['AMPA_UCCL_SWEEP_RUN'])
for node in range(4):
    path = run / 'nsight' / f'node-{node}-native.nsys-rep'
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f'expected four native Nsight reports; missing {path}')
record = summarize(run, steps=10, interval=1)
with (run / 'analysis/real_training_evidence.json').open('x') as f:
    json.dump(record, f, indent=2, sort_keys=True)
    f.write('\n')
PY
    "$TRAIN_PY" "$ROOT_DIR/analysis/extract_step_timing.py" --run-dir "$AMPA_UCCL_SWEEP_RUN" \
        --warmup-steps 5 --measured-steps 5 --log-interval 1 --global-batch-size 64 --sequence-length 4096
    "$TRAIN_PY" - <<'PY'
import json
import os
from pathlib import Path

out = Path(os.environ['AMPA_UCCL_COMPARISON_ROOT'])
contract = json.loads((out / 'nsight_budget_contract.json').read_text())
completed = {**contract, 'success': True, 'workers_srun_returned_zero': True,
    'training_health_and_timer_summary_complete': True, 'native_nsys_reports': 4,
    'gpu_metrics_acceptance': 'pending_posthoc_analysis',
    'routing_acceptance': 'pending_posthoc_analysis', 'trace_acceptance': 'pending_posthoc_analysis',
    'backend_acceptance': 'pending_posthoc_analysis', 'comparison_accepted': False,
    'fresh_checkpoint_payload_verified': False, 'fresh_dataset_payload_verified': False}
with (out / 'acquisition_completed.json').open('x') as f:
    json.dump(completed, f, indent=2, sort_keys=True)
    f.write('\n')
PY
else
    exit 2
fi
