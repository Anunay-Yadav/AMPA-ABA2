#!/usr/bin/env bash
# Qualified posthoc aggregation. The root agent alone submits allocations.
#SBATCH --job-name=ampa-overlap-summary
#SBATCH --account=infra01
#SBATCH --partition=debug
#SBATCH --qos=normal
#SBATCH --constraint=gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --no-requeue
#SBATCH --output=.runtime/slurms/slurm-%x-%j.out
#SBATCH --error=.runtime/slurms/slurm-%x-%j.err
set -euo pipefail
[[ ${SLURM_JOB_PARTITION:-} == debug && ${SLURM_JOB_NUM_NODES:-} == 1
    && ${SLURM_JOB_ID:-} =~ ^[0-9]+$ ]] || exit 2
readonly ROOT_DIR=${SLURM_SUBMIT_DIR:?}
cd "$ROOT_DIR"
readonly SELF="$ROOT_DIR/jobs/analysis/ep_overlap_campaign_summary.sbatch"
readonly TRAIN_PY=/opt/venvs/megatron-bridge/bin/python
export PYTHONDONTWRITEBYTECODE=1 CUDA_DEVICE_MAX_CONNECTIONS=32 FI_MR_CACHE_MONITOR=disabled
export AMPA_UCCL_BUILD_ROOT="$ROOT_DIR/runs/uccl-ep-owned-build-20261101T1713Z"
export EFA_HOME="$AMPA_UCCL_BUILD_ROOT/not-efa" UCCL_EP_TRANSPORT=cxi
export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/profiling:$ROOT_DIR/analysis:$ROOT_DIR/Megatron-LM:$ROOT_DIR/Megatron-Bridge/src:$AMPA_UCCL_BUILD_ROOT/site:$AMPA_UCCL_BUILD_ROOT/deps:$AMPA_UCCL_BUILD_ROOT/uccl/ep/deep_ep_wrapper${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="/opt/amazon/efa/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
[[ ${AMPA_OVERLAP_SUMMARY_MODE:?} =~ ^(qa|summarize)$
    && ${AMPA_OVERLAP_SUMMARY_ID:?} =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$
    && ${AMPA_OVERLAP_SUMMARY_SOURCE_MANIFEST:?} == "$ROOT_DIR"/runs/*
    && -f $AMPA_OVERLAP_SUMMARY_SOURCE_MANIFEST && ! -L $AMPA_OVERLAP_SUMMARY_SOURCE_MANIFEST
    && ${AMPA_OVERLAP_SUMMARY_SOURCE_SHA256:?} =~ ^[0-9a-f]{64}$ ]] || exit 2
read -r digest _ < <(sha256sum "$AMPA_OVERLAP_SUMMARY_SOURCE_MANIFEST")
[[ $digest == "$AMPA_OVERLAP_SUMMARY_SOURCE_SHA256" ]] || exit 2
sha256sum --check --strict "$AMPA_OVERLAP_SUMMARY_SOURCE_MANIFEST"

if [[ ${1:-} == --inside-container ]]; then
    [[ $# == 1 && ${AMPA_OVERLAP_SUMMARY_JOB_ROOT:?} == "$ROOT_DIR/runs/$AMPA_OVERLAP_SUMMARY_ID-job-$SLURM_JOB_ID" ]] || exit 2
    if [[ $AMPA_OVERLAP_SUMMARY_MODE == qa ]]; then
        bash -n "$SELF"
        exec timeout --signal=TERM --kill-after=5s 540s "$TRAIN_PY" - <<'PY'
import hashlib
import json
import os
from pathlib import Path
import sys
import unittest

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot

from tools import ep_overlap_campaign_summary as summary

root = Path(os.environ['SLURM_SUBMIT_DIR'])
output = Path(os.environ['AMPA_OVERLAP_SUMMARY_JOB_ROOT'])
if Path(sys.prefix) != Path('/opt/venvs/megatron-bridge'):
    raise ValueError('configured container Python is required')
inventory = summary.adapter.verify_source_manifest(
    Path(os.environ['AMPA_OVERLAP_SUMMARY_SOURCE_MANIFEST']),
    os.environ['AMPA_OVERLAP_SUMMARY_SOURCE_SHA256'])
closure = set(summary.import_closure(root, ('tools/ep_overlap_campaign_summary.py',))) | set(summary.NEW_SOURCES)
missing = sorted(closure - set(inventory['files']))
summary.write_new(output / 'closure_paths.json', {
    'all_paths': sorted(closure), 'missing_paths': missing,
    'summary_source_sha256': inventory['sha256']})
if missing:
    raise ValueError('qualified source omits the complete summary import closure; missing: ' + ', '.join(missing))
for name in summary.NEW_SOURCES:
    if name.endswith('.py'):
        compile((root / name).read_text(), name, 'exec')
suite = unittest.defaultTestLoader.discover(str(root / 'tests'), pattern='test_ep_overlap_campaign_summary.py')
result = unittest.TextTestRunner(verbosity=2).run(suite)
if not result.wasSuccessful() or result.testsRun < 1:
    raise SystemExit(1)
summary.adapter.verify_source_manifest(Path(inventory['path']), inventory['sha256'])
proof = {'qa_passed': True, 'training_executed': False, 'partition': 'debug',
         'job': os.environ['SLURM_JOB_ID'], 'tests': result.testsRun,
         'summary_source_sha256': inventory['sha256'], 'source_files': inventory['files'],
         'closure_paths_sha256': hashlib.sha256((output / 'closure_paths.json').read_bytes()).hexdigest()}
summary.write_new(output / 'qa.json', proof)
with (output / 'PASS.txt').open('x') as stream:
    stream.write(f"job={proof['job']}\nsummary_source_sha256={inventory['sha256']}\n"
                 f"qa_sha256={hashlib.sha256((output / 'qa.json').read_bytes()).hexdigest()}\n")
print(json.dumps({'qa_passed': True, 'tests': result.testsRun, 'job': proof['job']}))
PY
    fi
    [[ ${AMPA_OVERLAP_SUMMARY_OUTPUT:?} == "$ROOT_DIR/runs/$AMPA_OVERLAP_SUMMARY_ID" ]] || exit 2
    timeout --signal=TERM --kill-after=10s 5100s "$TRAIN_PY" "$ROOT_DIR/tools/ep_overlap_campaign_summary.py" \
        --coordinator-dir "$AMPA_OVERLAP_SUMMARY_COORDINATOR" \
        --accounting-file "$AMPA_OVERLAP_SUMMARY_JOB_ROOT/accounting_utc.psv" \
        --summary-source-manifest "$AMPA_OVERLAP_SUMMARY_SOURCE_MANIFEST" \
        --summary-source-sha256 "$AMPA_OVERLAP_SUMMARY_SOURCE_SHA256" \
        --output-dir "$AMPA_OVERLAP_SUMMARY_OUTPUT"
    timeout --signal=TERM --kill-after=5s 240s "$TRAIN_PY" \
        "$ROOT_DIR/ai/ep-overlap-summary-review/iteration-01/numerical_audit.py" \
        --summary-dir "$AMPA_OVERLAP_SUMMARY_OUTPUT" \
        --output "$AMPA_OVERLAP_SUMMARY_OUTPUT/independent_numerical_audit.json"
    jq -e '.valid_independent_numeric_audit == true and .cases == 12 and .optimizer_updates == 120
           and .torch_captures == 640 and .hardware_windows == 320' \
        "$AMPA_OVERLAP_SUMMARY_OUTPUT/independent_numerical_audit.json" > /dev/null
    exit 0
fi

[[ $# == 0 ]] || exit 2
export AMPA_OVERLAP_SUMMARY_JOB_ROOT="$ROOT_DIR/runs/$AMPA_OVERLAP_SUMMARY_ID-job-$SLURM_JOB_ID"
export AMPA_OVERLAP_SUMMARY_OUTPUT="$ROOT_DIR/runs/$AMPA_OVERLAP_SUMMARY_ID"
[[ ! -e $AMPA_OVERLAP_SUMMARY_JOB_ROOT && ! -L $AMPA_OVERLAP_SUMMARY_JOB_ROOT
    && ! -e $AMPA_OVERLAP_SUMMARY_OUTPUT && ! -L $AMPA_OVERLAP_SUMMARY_OUTPUT ]] || exit 2
export AMPA_OVERLAP_SESSION_DIR=${AMPA_OVERLAP_SESSION_DIR:-$ROOT_DIR/progress/sessions/2026-11-10-qwen-ep-overlap}
[[ $AMPA_OVERLAP_SESSION_DIR == "$ROOT_DIR"/progress/sessions/* && -d $AMPA_OVERLAP_SESSION_DIR
    && -f $AMPA_OVERLAP_SESSION_DIR/session_state.md ]] || exit 2
mkdir "$AMPA_OVERLAP_SUMMARY_JOB_ROOT"
mkdir "$AMPA_OVERLAP_SUMMARY_JOB_ROOT/logs"
cp "$AMPA_OVERLAP_SUMMARY_SOURCE_MANIFEST" "$AMPA_OVERLAP_SUMMARY_JOB_ROOT/summary-source.sha256"
record() {
    local note="$(date -u '+%Y-%m-%d %H:%M:%S UTC') summary_job=$SLURM_JOB_ID: $*"
    printf '%s\n' "$note" >> "$AMPA_OVERLAP_SUMMARY_JOB_ROOT/timeline.md"
    printf '\n- %s\n' "$note" >> "$AMPA_OVERLAP_SESSION_DIR/timeline.md"
    printf '\nSummary milestone: %s\n' "$note" >> "$AMPA_OVERLAP_SESSION_DIR/session_state.md"
}
complete=0
on_exit() {
    local status=$?
    trap - EXIT
    if (( status == 0 && complete != 1 )); then status=1; fi
    record "AFTER mode=$AMPA_OVERLAP_SUMMARY_MODE status=$status complete=$complete; preserve this attempt and all source acquisitions"
    printf 'exit_code=%s\ncomplete=%s\nmode=%s\n' "$status" "$complete" "$AMPA_OVERLAP_SUMMARY_MODE" > "$AMPA_OVERLAP_SUMMARY_JOB_ROOT/exit.txt"
    exit "$status"
}
trap on_exit EXIT
if [[ $AMPA_OVERLAP_SUMMARY_MODE == summarize ]]; then
    [[ ${AMPA_OVERLAP_SUMMARY_COORDINATOR:?} == "$ROOT_DIR"/runs/*
        && -d $AMPA_OVERLAP_SUMMARY_COORDINATOR && ! -L $AMPA_OVERLAP_SUMMARY_COORDINATOR
        && -f $AMPA_OVERLAP_SUMMARY_COORDINATOR/ACQUISITIONS_COMPLETE.txt
        && ${AMPA_OVERLAP_SUMMARY_QA_ROOT:?} == "$ROOT_DIR"/runs/*
        && -f $AMPA_OVERLAP_SUMMARY_QA_ROOT/qa.json && -f $AMPA_OVERLAP_SUMMARY_QA_ROOT/PASS.txt ]] || exit 2
    read -r qa_digest _ < <(sha256sum "$AMPA_OVERLAP_SUMMARY_QA_ROOT/qa.json")
    grep --fixed-strings --line-regexp --quiet -- "qa_sha256=$qa_digest" "$AMPA_OVERLAP_SUMMARY_QA_ROOT/PASS.txt"
    grep --fixed-strings --line-regexp --quiet -- "summary_source_sha256=$AMPA_OVERLAP_SUMMARY_SOURCE_SHA256" "$AMPA_OVERLAP_SUMMARY_QA_ROOT/PASS.txt"
    jq -e --arg source "$AMPA_OVERLAP_SUMMARY_SOURCE_SHA256" \
        '.qa_passed == true and .training_executed == false and .partition == "debug"
         and .summary_source_sha256 == $source and .tests > 0' "$AMPA_OVERLAP_SUMMARY_QA_ROOT/qa.json" > /dev/null
    jobs=$(awk -F '\t' '$1 !~ /^pilot-/ {if (NF != 5 || $3 !~ /^[0-9]+$/ || $4 !~ /^[0-9]+$/) exit 2; print $3; print $4}' \
        "$AMPA_OVERLAP_SUMMARY_COORDINATOR/accepted.tsv" | sort -nu | paste -sd, -)
    [[ $(tr ',' '\n' <<< "$jobs" | wc -l) == 24 ]] || exit 2
    export AMPA_OVERLAP_ACCOUNTING_START=${AMPA_OVERLAP_ACCOUNTING_START:-2026-11-10}
    [[ $AMPA_OVERLAP_ACCOUNTING_START =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]] || exit 2
    TZ=UTC SLURM_TIME_FORMAT=standard sacct -X --jobs="$jobs" --starttime="$AMPA_OVERLAP_ACCOUNTING_START" \
        --noheader --parsable2 --format=JobIDRaw,State,ExitCode,Start,End,ElapsedRaw,NodeList,Partition,NNodes \
        > "$AMPA_OVERLAP_SUMMARY_JOB_ROOT/accounting_utc.psv"
fi
record "BEFORE compute mode=$AMPA_OVERLAP_SUMMARY_MODE node=$SLURM_JOB_NODELIST container=setup/gh200-cuda130-runtime.toml; one debug node, no training or trace reparsing"
queue=$(squeue --array -u "$USER" --noheader --format='%i %P %T %D %R')
printf '%s\n' "$queue"
requested=$(awk '{total += $4} END {print total+0}' <<< "$queue")
(( requested <= 4 )) || exit 2
srun --mpi=pmix --cpu-bind=none --container-remap-root --environment="$ROOT_DIR/setup/gh200-cuda130-runtime.toml" \
    --nodes=1 --ntasks=1 --gpus-per-task=1 --kill-on-bad-exit=1 \
    --output="$AMPA_OVERLAP_SUMMARY_JOB_ROOT/logs/summary.out" --error="$AMPA_OVERLAP_SUMMARY_JOB_ROOT/logs/summary.err" \
    bash "$SELF" --inside-container
complete=1
record "AFTER allocated mode=$AMPA_OVERLAP_SUMMARY_MODE returned zero; inspect QA or summary proof before accepting conclusions"
