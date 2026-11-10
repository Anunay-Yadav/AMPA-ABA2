#!/usr/bin/env bash
# Inventory the installed Nsight CLI and exercise only a tiny CUDA workload.
#SBATCH --job-name=ampa-uccl-nsys-cap
#SBATCH --account=infra01
#SBATCH --partition=debug
#SBATCH --qos=normal
#SBATCH --constraint=gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --exclusive
#SBATCH --mem=64G
#SBATCH --time=00:05:00
#SBATCH --no-requeue
#SBATCH --output=.runtime/slurms/slurm-%x-%j.out
#SBATCH --error=.runtime/slurms/slurm-%x-%j.err
set -euo pipefail
[[ ${SLURM_JOB_PARTITION:-} == debug && ${SLURM_JOB_NUM_NODES:-} == 1 ]] || exit 2
readonly ROOT_DIR=${SLURM_SUBMIT_DIR:?}
[[ ${AMPA_UCCL_NSYS_CAP_ID:?} =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*$ ]] || exit 2
export AMPA_UCCL_NSYS_CAP_ROOT="$ROOT_DIR/runs/$AMPA_UCCL_NSYS_CAP_ID"
export PYTHONDONTWRITEBYTECODE=1

if [[ ${1:-} == --container ]]; then
    [[ $# == 1 && ${SLURM_PROCID:-} == 0 && ${WORKSPACE:-} == "$ROOT_DIR" ]] || exit 2
    [[ -f "$AMPA_UCCL_NSYS_CAP_ROOT/job-id" && $(cat "$AMPA_UCCL_NSYS_CAP_ROOT/job-id") == "$SLURM_JOB_ID" ]] || exit 2
    cd "$ROOT_DIR"
    /opt/venvs/megatron-bridge/bin/python - <<'PY'
import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import sqlite3
import subprocess
import sys
import time

root = Path(os.environ['WORKSPACE'])
out = Path(os.environ['AMPA_UCCL_NSYS_CAP_ROOT'])
deadline = time.monotonic() + 225
commands = []

def dump(name, value):
    target = out / name
    target.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')

def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def run(name, argv, timeout=12):
    budget = min(timeout, max(0, deadline - time.monotonic() - 3))
    record = {'name': name, 'argv': [str(x) for x in argv],
              'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'timeout_seconds': budget, 'stdout': f'logs/{name}.out',
              'stderr': f'logs/{name}.err'}
    started = time.monotonic()
    with (out / record['stdout']).open('w') as stdout, (out / record['stderr']).open('w') as stderr:
        if budget <= 0:
            record.update(exit_code=124, timed_out=True, skipped='probe time budget exhausted')
        else:
            try:
                proc = subprocess.Popen(record['argv'], cwd=root, stdout=stdout, stderr=stderr,
                                        start_new_session=True)
                try:
                    record['exit_code'] = proc.wait(timeout=budget)
                    record['timed_out'] = False
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        proc.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                        proc.wait()
                    record.update(exit_code=124, process_exit_code=proc.returncode, timed_out=True)
            except OSError as exc:
                stderr.write(f'{type(exc).__name__}: {exc}\n')
                record.update(exit_code=127, timed_out=False)
    record['elapsed_seconds'] = time.monotonic() - started
    commands.append(record)
    dump('commands.json', commands)
    return record

def read_log(record):
    return (out / record['stdout']).read_text(errors='replace') + '\n' + (out / record['stderr']).read_text(errors='replace')

def sqlite_inventory(path):
    result = {'path': str(path), 'exists': path.is_file(), 'metric_rows': 0}
    if not path.is_file():
        return result
    try:
        with sqlite3.connect(f'file:{path}?mode=ro', uri=True) as db:
            tables = [r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
            result['tables'] = tables
            result['gpu_metric_tables'] = {}
            for name in tables:
                if 'GPU_METRIC' not in name.upper():
                    continue
                quoted = '"' + name.replace('"', '""') + '"'
                rows = db.execute(f'SELECT COUNT(*) FROM {quoted}').fetchone()[0]
                columns = [r[1] for r in db.execute(f'PRAGMA table_info({quoted})')]
                result['gpu_metric_tables'][name] = {'rows': rows, 'columns': columns,
                    'first_rows': db.execute(f'SELECT * FROM {quoted} LIMIT 8').fetchall()}
                if name.upper() == 'GPU_METRICS':
                    result['metric_rows'] = rows
    except (sqlite3.Error, OSError) as exc:
        result['error'] = str(exc)
    return result

provenance = {
    'kind': 'uccl_nsight_capabilities', 'schema_version': 1,
    'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'hostname': socket.gethostname(), 'execution_location': 'allocated_compute_configured_container',
    'python': sys.executable, 'python_version': sys.version, 'sys_path': sys.path,
    'job_id': os.environ['SLURM_JOB_ID'], 'partition': os.environ['SLURM_JOB_PARTITION'],
    'node_list': os.environ.get('SLURM_JOB_NODELIST'),
    'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
    'cpu_affinity': sorted(os.sched_getaffinity(0)), 'container_config': str(root / 'setup/gh200-cuda130-runtime.toml'),
    'training_executed': False, 'kineto_profiler_requested': False, 'packages': {}}
for name in ('torch', 'megatron-bridge', 'megatron-core', 'transformer-engine', 'cuda-python', 'numpy', 'PyYAML', 'nvtx'):
    try:
        dist = importlib.metadata.distribution(name)
        provenance['packages'][name] = {'version': dist.version, 'location': str(dist.locate_file(''))}
    except importlib.metadata.PackageNotFoundError:
        provenance['packages'][name] = None
dump('provenance.json', provenance)

files = ['jobs/smoke/uccl_nsight_capabilities.sbatch', 'setup/gh200-cuda130-runtime.toml',
         'profiling/run_pretrain.py', 'profiling/run_pretrain_debug.py',
         'profiling/run_pretrain_debug_reuse.py', 'profiling/ampa_profile/uccl_identity.py',
         'Megatron-LM/megatron/core/transformer/moe/fused_a2a.py',
         'Megatron-LM/megatron/core/transformer/moe/token_dispatcher.py',
         'Megatron-LM/megatron/core/transformer/moe/moe_layer.py',
         'Megatron-Bridge/src/megatron/bridge/training/pretrain.py']
identities = {}
for relative in files:
    path = root / relative
    if path.is_file():
        identities[relative] = {'sha256': digest(path), 'bytes': path.stat().st_size}
        dest = out / 'source' / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
    else:
        identities[relative] = {'missing': True}
dump('source_identity.json', identities)
for label, directory in (('ampa', root), ('megatron_lm', root / 'Megatron-LM'), ('megatron_bridge', root / 'Megatron-Bridge')):
    run(f'{label}_revision', ['git', '-C', str(directory), 'rev-parse', 'HEAD'], 5)
    run(f'{label}_status', ['git', '-C', str(directory), 'status', '--porcelain=v1', '--untracked-files=normal'], 8)
    if label != 'ampa':
        run(f'{label}_diff', ['git', '-C', str(directory), 'diff', '--no-ext-diff', '--binary', 'HEAD'], 8)

run('nvidia_smi', ['nvidia-smi', '--query-gpu=index,uuid,name,driver_version,memory.total', '--format=csv,noheader'], 8)
run('driver_parameters', ['cat', '/proc/driver/nvidia/params'], 4)
run('kernel_perf_permissions', ['cat', '/proc/sys/kernel/perf_event_paranoid'], 4)

app = out / 'tiny_cuda.py'
app.write_text('''import json, os, time
import torch
torch.cuda.set_device(0)
p = torch.cuda.get_device_properties(0)
print(json.dumps({"torch":torch.__version__, "torch_file":torch.__file__, "cuda":torch.version.cuda,
 "visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES"), "device_count":torch.cuda.device_count(),
 "device_name":p.name, "sm_count":p.multi_processor_count, "major":p.major, "minor":p.minor,
 "device_uuid":str(getattr(p,"uuid",None)), "training":False}), flush=True)
a=torch.full((2048,2048),0.015625,device="cuda",dtype=torch.bfloat16)
b=torch.full_like(a,0.015625)
c=torch.empty_like(a)
for _ in range(4): torch.mm(a,b,out=c)
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStart()
torch.cuda.nvtx.range_push("AMPA_NSYS_CAPABILITY_TINY_GPU")
started=time.monotonic()
iterations=0
while time.monotonic()-started < 3.0:
 for _ in range(16):
  torch.mm(a,b,out=c)
  c.relu_()
 torch.cuda.synchronize()
 iterations+=16
torch.cuda.nvtx.range_pop()
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
print(json.dumps({"iterations":iterations,"elapsed_s":time.monotonic()-started,
 "sample":float(c[0,0].item()),"finite":bool(torch.isfinite(c[0,0]).item())}),flush=True)
''')
dump('tiny_application.json', {'path': str(app), 'sha256': digest(app),
                             'duration_seconds': 3, 'training': False, 'selected_cuda_device': 0})

nsys = shutil.which('nsys')
if nsys is None:
    for pattern in ('/opt/nvidia/nsight-systems/*/bin/nsys', '/opt/nvidia/nsight-systems-cli/*/bin/nsys',
                    '/usr/local/cuda/bin/nsys', '/opt/nsight-systems/bin/nsys'):
        import glob
        candidates = [x for x in sorted(glob.glob(pattern)) if os.access(x, os.X_OK)]
        if candidates:
            nsys = candidates[-1]
            break
result = {'schema_version': 1, 'job_id': os.environ['SLURM_JOB_ID'], 'hostname': socket.gethostname(),
          'nsys_path': nsys, 'native_report_generated': False, 'gpu_metrics_collected': False,
          'gpu_metrics_status': 'unavailable', 'training_executed': False,
          'cuda_profiler_api_capture_requested': True, 'gpu_metrics_device_option': None,
          'claim_limit': 'Tiny capture tests tool/report/metric availability only. It does not profile Qwen or prove every desired counter is available.'}
if nsys:
    provenance['nsys'] = {'path': nsys, 'sha256': digest(Path(nsys).resolve())}
    dump('provenance.json', provenance)
    run('nsys_version', [nsys, '--version'], 6)
    run('nsys_help', [nsys, '--help'], 6)
    help_record = run('nsys_profile_help', [nsys, 'profile', '--help'], 8)
    help_text = read_log(help_record)
    device_option = next((flag for flag in ('--gpu-metrics-devices', '--gpu-metrics-device')
                          if re.search(re.escape(flag) + r'(?=[=\s,]|$)', help_text)), None)
    result['gpu_metrics_device_option'] = device_option
    run('nsys_status', [nsys, 'status', '--environment'], 10)
    export_help = run('nsys_export_help', [nsys, 'export', '--help'], 6)
    if device_option:
        run('gpu_metrics_devices', [nsys, 'profile', f'{device_option}=help'], 10)
    if '--gpu-metrics-set' in help_text:
        run('gpu_metrics_sets', [nsys, 'profile', '--gpu-metrics-set=help'], 10)
    capture_args = ['--trace=cuda,nvtx', '--sample=none',
                    '--capture-range=cudaProfilerApi', '--capture-range-end=stop']
    if '--cpuctxsw' in help_text:
        capture_args.append('--cpuctxsw=none')
    baseline = run('capture_native', [nsys, 'profile', *capture_args,
                    f'--output={out / "captures/native"}', sys.executable, str(app)], 50)
    native_report = out / 'captures/native.nsys-rep'
    result['native_capture_exit_code'] = baseline['exit_code']
    result['native_report_generated'] = baseline['exit_code'] == 0 and native_report.is_file() and native_report.stat().st_size > 0
    if native_report.is_file():
        result['native_report'] = {'path': str(native_report), 'bytes': native_report.stat().st_size,
                                   'sha256': digest(native_report)}
    if result['native_report_generated'] and device_option:
        metrics_args = [*capture_args, f'{device_option}=all']
        if '--gpu-metrics-frequency' in help_text:
            metrics_args.append('--gpu-metrics-frequency=1000')
        metric_capture = run('capture_gpu_metrics', [nsys, 'profile', *metrics_args,
                            f'--output={out / "captures/gpu_metrics"}', sys.executable, str(app)], 60)
        result['gpu_metrics_capture_exit_code'] = metric_capture['exit_code']
        metric_report = out / 'captures/gpu_metrics.nsys-rep'
        if metric_report.is_file():
            result['gpu_metrics_report'] = {'path': str(metric_report), 'bytes': metric_report.stat().st_size,
                                           'sha256': digest(metric_report)}
        if metric_capture['exit_code'] == 0 and metric_report.is_file():
            exported = run('export_gpu_metrics', [nsys, 'export', '--type=sqlite',
                            f'--output={out / "captures/gpu_metrics.sqlite"}', str(metric_report)], 30)
            inventory = sqlite_inventory(out / 'captures/gpu_metrics.sqlite')
            dump('gpu_metrics_sqlite_inventory.json', inventory)
            result['gpu_metrics_collected'] = exported['exit_code'] == 0 and inventory['metric_rows'] > 0
            result['gpu_metrics_status'] = 'sample_rows_observed' if result['gpu_metrics_collected'] else 'no_metric_rows_proven'
        else:
            result['gpu_metrics_status'] = 'capture_failed_or_no_report'
    elif device_option is None:
        result['gpu_metrics_status'] = 'installed_cli_has_no_gpu_metrics_device_option'
else:
    result['failure'] = 'nsys executable unavailable in configured container and standard tool locations'
result['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
dump('capabilities.json', result)
print(json.dumps(result, indent=2, sort_keys=True), flush=True)
raise SystemExit(0 if result['native_report_generated'] else 1)
PY
    exit $?
fi

[[ $# == 0 ]] || exit 2
[[ ! -e "$AMPA_UCCL_NSYS_CAP_ROOT" && ! -L "$AMPA_UCCL_NSYS_CAP_ROOT" ]] || exit 2
mkdir "$AMPA_UCCL_NSYS_CAP_ROOT"
mkdir -p "$AMPA_UCCL_NSYS_CAP_ROOT"/{logs,captures,source}
printf '%s\n' "$SLURM_JOB_ID" > "$AMPA_UCCL_NSYS_CAP_ROOT/job-id"
cp "$ROOT_DIR/jobs/smoke/uccl_nsight_capabilities.sbatch" "$AMPA_UCCL_NSYS_CAP_ROOT/source/uccl_nsight_capabilities.sbatch"
sha256sum "$ROOT_DIR/jobs/smoke/uccl_nsight_capabilities.sbatch" > "$AMPA_UCCL_NSYS_CAP_ROOT/source.sha256"
printf 'job=%s\npartition=%s\nnodes=%s\nstarted_utc=%s\n' "$SLURM_JOB_ID" "$SLURM_JOB_PARTITION" "$SLURM_JOB_NODELIST" "$(date -u +%FT%TZ)" > "$AMPA_UCCL_NSYS_CAP_ROOT/allocation.txt"
squeue -u "$USER"
set +e
srun --mpi=pmix --cpu-bind=none --container-remap-root --environment="$ROOT_DIR/setup/gh200-cuda130-runtime.toml" \
    --nodes=1 --ntasks=1 --gpus-per-task=1 \
    --output="$AMPA_UCCL_NSYS_CAP_ROOT/logs/container.out" --error="$AMPA_UCCL_NSYS_CAP_ROOT/logs/container.err" \
    bash "$ROOT_DIR/jobs/smoke/uccl_nsight_capabilities.sbatch" --container
probe_status=$?
set -e
printf 'exit_code=%s\nfinished_utc=%s\n' "$probe_status" "$(date -u +%FT%TZ)" > "$AMPA_UCCL_NSYS_CAP_ROOT/exit.txt"
exit "$probe_status"
