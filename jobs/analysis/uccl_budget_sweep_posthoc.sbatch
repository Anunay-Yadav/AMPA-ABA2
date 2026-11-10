#!/usr/bin/env bash
# Export/validate the four acquired budgets, then assemble their delivery index.
#SBATCH --job-name=ampa-uccl-budget-posthoc
#SBATCH --account=infra01
#SBATCH --partition=debug
#SBATCH --qos=normal
#SBATCH --constraint=gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --exclusive
#SBATCH --time=00:21:00
#SBATCH --no-requeue
#SBATCH --output=.runtime/slurms/slurm-%x-%j.out
#SBATCH --error=.runtime/slurms/slurm-%x-%j.err
set -euo pipefail
[[ ${SLURM_JOB_PARTITION:-} == debug && ${SLURM_JOB_NUM_NODES:-} == 1 ]] || exit 2
readonly ROOT_DIR=${SLURM_SUBMIT_DIR:?}
[[ ${AMPA_UCCL_SWEEP_CONTROL:?} == "$ROOT_DIR"/runs/* && -d $AMPA_UCCL_SWEEP_CONTROL ]] || exit 2
[[ ${AMPA_UCCL_SWEEP_POSTHOC_ID:?} =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*$ ]] || exit 2
[[ ${AMPA_UCCL_NSYS_CAP_ROOT:?} == "$ROOT_DIR"/runs/* && -f $AMPA_UCCL_NSYS_CAP_ROOT/capabilities.json ]] || exit 2
export AMPA_UCCL_SWEEP_POSTHOC_OUT="$ROOT_DIR/runs/$AMPA_UCCL_SWEEP_POSTHOC_ID"
export AMPA_UCCL_SWEEP_CONTROL AMPA_UCCL_SWEEP_POSTHOC_ID AMPA_UCCL_NSYS_CAP_ROOT

if [[ ${1:-} == --container ]]; then
    [[ $# == 1 && ${WORKSPACE:-} == "$ROOT_DIR" && ${SLURM_PROCID:-} == 0 ]] || exit 2
    cd "$ROOT_DIR"
    export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
    export AMPA_UCCL_BUILD_ROOT="$ROOT_DIR/runs/uccl-ep-owned-build-20261101T1713Z"
    export EFA_HOME="$AMPA_UCCL_BUILD_ROOT/not-efa" FI_MR_CACHE_MONITOR=disabled UCCL_EP_TRANSPORT=cxi
    export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/profiling:$ROOT_DIR/analysis:$AMPA_UCCL_BUILD_ROOT/site:$AMPA_UCCL_BUILD_ROOT/deps:$AMPA_UCCL_BUILD_ROOT/uccl/ep/deep_ep_wrapper:$ROOT_DIR/Megatron-LM:$ROOT_DIR/Megatron-Bridge/src${PYTHONPATH:+:$PYTHONPATH}"
    case ${LD_LIBRARY_PATH:-} in
        /opt/amazon/efa/lib|/opt/amazon/efa/lib:*) ;;
        *) export LD_LIBRARY_PATH="/opt/amazon/efa/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
    esac
    readonly POSTHOC_PY=/opt/venvs/megatron-bridge/bin/python
    readonly OUT=$AMPA_UCCL_SWEEP_POSTHOC_OUT
    sha256sum --check "$OUT/source.sha256"
    sha256sum --check "$OUT/control-input.sha256"
    cmp "$AMPA_UCCL_SWEEP_CONTROL/acquisitions.tsv" "$OUT/acquisitions.tsv"
    bash -n "$ROOT_DIR/jobs/analysis/uccl_budget_sweep_posthoc.sbatch"
    "$POSTHOC_PY" - <<'PY'
import ast
import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import sys

root = Path(os.environ['WORKSPACE'])
out = Path(os.environ['AMPA_UCCL_SWEEP_POSTHOC_OUT'])
rows = [row.split('\t') for row in (out / 'acquisitions.tsv').read_text().splitlines()]
if (len(rows) != 4 or any(len(row) != 4 for row in rows)
        or sorted(row[0] for row in rows) != ['12', '20', '32', '40']
        or rows[0][0] != '20' or len({row[1] for row in rows}) != 4):
    raise ValueError('expected budgets 20,12,32,40 with baseline 20 first and four distinct acquisition jobs')
for budget, job, orchestration, run in rows:
    if not job.isdigit():
        raise ValueError('acquisition job ID is not numeric')
    for item in (orchestration, run):
        path = Path(item)
        if path.parent != root / 'runs' or not path.is_dir() or path.is_symlink():
            raise ValueError('acquisition paths must be existing direct project runs')
for relative in ('tools/uccl_budget_sweep_posthoc.py', 'analysis/uccl_budget_sweep_delivery.py'):
    ast.parse((root / relative).read_text(), filename=relative)
record = {'schema_version': 1, 'kind': 'uccl_budget_posthoc_execution',
    'slurm_job_id': os.environ['SLURM_JOB_ID'], 'partition': os.environ['SLURM_JOB_PARTITION'],
    'hostname': socket.gethostname(), 'python': sys.executable, 'python_version': sys.version,
    'execution_location': 'allocated_configured_container_CPU_export',
    'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'), 'gpu_execution': False,
    'parallel_case_validators': 4, 'export_timeout_seconds_per_report': 240,
    'acquisition_order': [int(row[0]) for row in rows], 'baseline_budget': 20,
    'control_dir': os.environ['AMPA_UCCL_SWEEP_CONTROL'], 'capability_root': os.environ['AMPA_UCCL_NSYS_CAP_ROOT'],
    'source_manifest_sha256': hashlib.sha256((out / 'source.sha256').read_bytes()).hexdigest(),
    'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
with (out / 'execution_provenance.json').open('x') as stream:
    json.dump(record, stream, indent=2, sort_keys=True)
    stream.write('\n')
PY
    declare -a validator_pids=() validator_budgets=()
    printf 'budget\tpid\n' > "$OUT/case-pids.tsv"
    while IFS=$'\t' read -r budget acquisition_job orchestration actual_run; do
        remaining=$((AMPA_UCCL_SWEEP_POSTHOC_DEADLINE - $(date +%s)))
        (( remaining > 5 )) || exit 124
        printf 'budget=%s job=%s orchestration=%s actual_run=%s\n' "$budget" "$acquisition_job" "$orchestration" "$actual_run"
        timeout --signal=TERM --kill-after=5s "${remaining}s" "$POSTHOC_PY" \
            "$ROOT_DIR/tools/uccl_budget_sweep_posthoc.py" --orchestration-dir "$orchestration" \
            --output-dir "$OUT/cases/sms$budget" --capability-root "$AMPA_UCCL_NSYS_CAP_ROOT" \
            --export-timeout-seconds 240 \
            > "$OUT/logs/sms$budget.out" 2> "$OUT/logs/sms$budget.err" &
        validator_pids+=("$!")
        validator_budgets+=("$budget")
        printf '%s\t%s\n' "$budget" "${validator_pids[-1]}" >> "$OUT/case-pids.tsv"
    done < "$OUT/acquisitions.tsv"
    failed=0
    printf 'budget\tpid\texit_code\n' > "$OUT/case-exits.tsv"
    for index in "${!validator_pids[@]}"; do
        if wait "${validator_pids[index]}"; then
            validator_status=0
        else
            validator_status=$?
            failed=1
        fi
        printf '%s\t%s\t%s\n' "${validator_budgets[index]}" "${validator_pids[index]}" "$validator_status" >> "$OUT/case-exits.tsv"
    done
    sha256sum --check "$OUT/source.sha256"
    sha256sum --check "$OUT/control-input.sha256"
    cmp "$AMPA_UCCL_SWEEP_CONTROL/acquisitions.tsv" "$OUT/acquisitions.tsv"
    (( failed == 0 )) || exit 1
    remaining=$((AMPA_UCCL_SWEEP_POSTHOC_DEADLINE - $(date +%s)))
    (( remaining > 5 )) || exit 124
    timeout --signal=TERM --kill-after=5s "${remaining}s" "$POSTHOC_PY" \
        "$ROOT_DIR/analysis/uccl_budget_sweep_delivery.py" --control-dir "$AMPA_UCCL_SWEEP_CONTROL" \
        --posthoc-dir "$OUT" --capability-root "$AMPA_UCCL_NSYS_CAP_ROOT" \
        > "$OUT/logs/delivery.out" 2> "$OUT/logs/delivery.err"
    "$POSTHOC_PY" - <<'PY'
import json
import os
from pathlib import Path

out = Path(os.environ['AMPA_UCCL_SWEEP_POSTHOC_OUT'])
if json.loads((out / 'DELIVERY.json').read_text()).get('valid_delivery') is not True:
    raise ValueError('delivery builder did not accept all four budget runs')
if not (out / 'README.md').is_file():
    raise ValueError('delivery opening instructions are missing')
PY
    sha256sum --check "$OUT/source.sha256"
    sha256sum --check "$OUT/control-input.sha256"
    cmp "$AMPA_UCCL_SWEEP_CONTROL/acquisitions.tsv" "$OUT/acquisitions.tsv"
    exit 0
fi

[[ $# == 0 && -f $AMPA_UCCL_SWEEP_CONTROL/acquisitions.tsv ]] || exit 2
[[ ! -e $AMPA_UCCL_SWEEP_POSTHOC_OUT && ! -L $AMPA_UCCL_SWEEP_POSTHOC_OUT ]] || exit 2
export AMPA_UCCL_SWEEP_POSTHOC_DEADLINE=$(( $(date +%s) + 1190 ))
mkdir "$AMPA_UCCL_SWEEP_POSTHOC_OUT"
mkdir -p "$AMPA_UCCL_SWEEP_POSTHOC_OUT"/{logs,source,cases}
printf '%s\n' "$SLURM_JOB_ID" > "$AMPA_UCCL_SWEEP_POSTHOC_OUT/job-id"
cp "$AMPA_UCCL_SWEEP_CONTROL/acquisitions.tsv" "$AMPA_UCCL_SWEEP_POSTHOC_OUT/acquisitions.tsv"
sha256sum "$AMPA_UCCL_SWEEP_CONTROL/acquisitions.tsv" > "$AMPA_UCCL_SWEEP_POSTHOC_OUT/control-input.sha256"
for relative in jobs/analysis/uccl_budget_sweep_posthoc.sbatch tools/uccl_budget_sweep_posthoc.py analysis/uccl_budget_sweep_delivery.py; do
    mkdir -p "$AMPA_UCCL_SWEEP_POSTHOC_OUT/source/$(dirname "$relative")"
    cp "$ROOT_DIR/$relative" "$AMPA_UCCL_SWEEP_POSTHOC_OUT/source/$relative"
    sha256sum "$ROOT_DIR/$relative" >> "$AMPA_UCCL_SWEEP_POSTHOC_OUT/source.sha256"
done
record() {
    local note="$(date -u '+%Y-%m-%d %H:%M:%S UTC') budget_posthoc job=$SLURM_JOB_ID: $*"
    printf '%s\n' "$note" >> "$AMPA_UCCL_SWEEP_POSTHOC_OUT/timeline.md"
    printf '%s\n' "$note" > "$AMPA_UCCL_SWEEP_POSTHOC_OUT/CURRENT.md"
    for target in progress/sessions/2026-11-01-uccl-ep-comparison/timeline.md \
        progress/sessions/2026-11-01-uccl-ep-comparison/session_state.md \
        ai/qwen-uccl-block-sweep/iteration-01/STATUS.md; do
        printf '\n- %s\n' "$note" >> "$ROOT_DIR/$target"
    done
}
complete=0
on_exit() {
    local status=$?
    trap - EXIT
    printf 'exit_code=%s\ncomplete=%s\n' "$status" "$complete" > "$AMPA_UCCL_SWEEP_POSTHOC_OUT/exit.txt"
    record "AFTER compute status=$status complete=$complete; preserve all exports/partial evidence; output $AMPA_UCCL_SWEEP_POSTHOC_OUT"
    exit "$status"
}
trap on_exit EXIT
record "BEFORE compute four concurrent CPU-only budget validators/sixteen Nsight exports then delivery index; one exclusive debugGHnode/$SLURM_JOB_NODELIST, configuredcontainer, CUDAmasked, perexport240s/totaldeadline1190s+5skillguard; no training"
squeue -u "$USER"
remaining=$((AMPA_UCCL_SWEEP_POSTHOC_DEADLINE - $(date +%s)))
(( remaining > 5 )) || exit 124
timeout --signal=TERM --kill-after=5s "${remaining}s" \
    srun --mpi=pmix --cpu-bind=none --container-remap-root --environment="$ROOT_DIR/setup/gh200-cuda130-runtime.toml" \
        --nodes=1 --ntasks=1 --gpus-per-task=1 \
        --output="$AMPA_UCCL_SWEEP_POSTHOC_OUT/logs/container.out" --error="$AMPA_UCCL_SWEEP_POSTHOC_OUT/logs/container.err" \
        bash "$ROOT_DIR/jobs/analysis/uccl_budget_sweep_posthoc.sbatch" --container
complete=1
printf 'job=%s\nvalid_delivery=true\noutput=%s\n' "$SLURM_JOB_ID" "$AMPA_UCCL_SWEEP_POSTHOC_OUT" > "$AMPA_UCCL_SWEEP_POSTHOC_OUT/PASS.txt"
