#!/usr/bin/env bash
# Reconcile the completed adaptive campaign. Only the root agent submits.
#SBATCH --job-name=ampa-budget-summary
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
readonly SELF="$ROOT_DIR/jobs/analysis/ep_overlap_budget_summary.sbatch"
readonly TRAIN_PY=/opt/venvs/megatron-bridge/bin/python
[[ ${AMPA_BUDGET_SUMMARY_MODE:?} =~ ^(qa|summarize)$
    && ${AMPA_BUDGET_SUMMARY_ID:?} =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$
    && ${AMPA_BUDGET_SUMMARY_SOURCE_MANIFEST:?} == "$ROOT_DIR"/runs/*
    && -f $AMPA_BUDGET_SUMMARY_SOURCE_MANIFEST && ! -L $AMPA_BUDGET_SUMMARY_SOURCE_MANIFEST
    && ${AMPA_BUDGET_SUMMARY_SOURCE_SHA256:?} =~ ^[0-9a-f]{64}$ ]] || exit 2
read -r digest _ < <(sha256sum "$AMPA_BUDGET_SUMMARY_SOURCE_MANIFEST")
[[ $digest == "$AMPA_BUDGET_SUMMARY_SOURCE_SHA256" ]] || exit 2
sha256sum --check --strict "$AMPA_BUDGET_SUMMARY_SOURCE_MANIFEST"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/profiling:$ROOT_DIR/analysis${PYTHONPATH:+:$PYTHONPATH}"

if [[ ${1:-} == --inside-container ]]; then
    [[ $# == 1 && ${AMPA_BUDGET_SUMMARY_JOB_ROOT:?} == "$ROOT_DIR/runs/$AMPA_BUDGET_SUMMARY_ID-job-$SLURM_JOB_ID" ]] || exit 2
    export MPLCONFIGDIR="$AMPA_BUDGET_SUMMARY_JOB_ROOT/mpl"
    if [[ $AMPA_BUDGET_SUMMARY_MODE == qa ]]; then
        bash -n "$SELF"
        exec timeout --signal=TERM --kill-after=5s 540s "$TRAIN_PY" - <<'PY'
import os
from pathlib import Path
import sys
import unittest

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot

from tools import ep_overlap_budget_summary as summary

if Path(sys.prefix) != Path('/opt/venvs/megatron-bridge'):
    raise ValueError('configured Python is required')
root = Path(os.environ['SLURM_SUBMIT_DIR'])
out = Path(os.environ['AMPA_BUDGET_SUMMARY_JOB_ROOT'])
manifest = Path(os.environ['AMPA_BUDGET_SUMMARY_SOURCE_MANIFEST'])
digest = os.environ['AMPA_BUDGET_SUMMARY_SOURCE_SHA256']
source = summary.adapter.verify_source_manifest(manifest, digest)
required = set(summary.import_closure(root, summary.NEW_SOURCES[:2])) | set(summary.NEW_SOURCES)
missing = sorted(required - set(source['files']))
summary.write_new(out / 'closure_paths.json', {'all_paths': sorted(required), 'missing_paths': missing})
if missing:
    raise ValueError('summary source omits: ' + ', '.join(missing))
suite = unittest.defaultTestLoader.discover(str(root / 'tests'), pattern='test_ep_overlap_budget_summary.py')
result = unittest.TextTestRunner(verbosity=2).run(suite)
if not result.wasSuccessful() or result.testsRun != 7 or result.skipped or result.expectedFailures:
    raise SystemExit(1)
summary.adapter.verify_source_manifest(manifest, digest)
summary.write_new(out / 'qa.json', {'qa_passed': True, 'tests': result.testsRun,
    'summary_source_sha256': digest, 'training_executed': False, 'partition': 'debug', 'job': os.environ['SLURM_JOB_ID']})
with (out / 'PASS.txt').open('x') as stream:
    stream.write(f"summary_source_sha256={digest}\nqa_sha256={summary.adapter.sha(out / 'qa.json')}\njob={os.environ['SLURM_JOB_ID']}\n")
PY
    fi
    [[ ${AMPA_BUDGET_SUMMARY_OUTPUT:?} == "$ROOT_DIR/runs/$AMPA_BUDGET_SUMMARY_ID" ]] || exit 2
    exec timeout --signal=TERM --kill-after=10s 5250s "$TRAIN_PY" "$ROOT_DIR/tools/ep_overlap_budget_summary.py" \
        --campaign-dir "$AMPA_BUDGET_SUMMARY_CAMPAIGN" \
        --accounting-file "$AMPA_BUDGET_SUMMARY_JOB_ROOT/accounting_utc.psv" \
        --summary-source-manifest "$AMPA_BUDGET_SUMMARY_SOURCE_MANIFEST" \
        --summary-source-sha256 "$AMPA_BUDGET_SUMMARY_SOURCE_SHA256" \
        --output-dir "$AMPA_BUDGET_SUMMARY_OUTPUT"
fi

[[ $# == 0 ]] || exit 2
export AMPA_BUDGET_SUMMARY_JOB_ROOT="$ROOT_DIR/runs/$AMPA_BUDGET_SUMMARY_ID-job-$SLURM_JOB_ID"
export AMPA_BUDGET_SUMMARY_OUTPUT="$ROOT_DIR/runs/$AMPA_BUDGET_SUMMARY_ID"
export AMPA_BUDGET_SESSION_DIR=${AMPA_BUDGET_SESSION_DIR:-$ROOT_DIR/progress/sessions/2026-11-10-qwen-ep-overlap}
[[ ! -e $AMPA_BUDGET_SUMMARY_JOB_ROOT && ! -L $AMPA_BUDGET_SUMMARY_JOB_ROOT
    && ! -e $AMPA_BUDGET_SUMMARY_OUTPUT && ! -L $AMPA_BUDGET_SUMMARY_OUTPUT
    && $AMPA_BUDGET_SESSION_DIR == "$ROOT_DIR"/progress/sessions/*
    && -f $AMPA_BUDGET_SESSION_DIR/timeline.md && -f $AMPA_BUDGET_SESSION_DIR/session_state.md ]] || exit 2
mkdir "$AMPA_BUDGET_SUMMARY_JOB_ROOT"
mkdir "$AMPA_BUDGET_SUMMARY_JOB_ROOT/logs"
cp "$AMPA_BUDGET_SUMMARY_SOURCE_MANIFEST" "$AMPA_BUDGET_SUMMARY_JOB_ROOT/summary-source.sha256"
record() {
    local note="$(date -u '+%Y-%m-%d %H:%M:%S UTC') budget_summary_job=$SLURM_JOB_ID: $*"
    printf '%s\n' "$note" >> "$AMPA_BUDGET_SUMMARY_JOB_ROOT/timeline.md"
    printf '\n- %s\n' "$note" >> "$AMPA_BUDGET_SESSION_DIR/timeline.md"
    printf '\nBudget summary milestone: %s\n' "$note" >> "$AMPA_BUDGET_SESSION_DIR/session_state.md"
}
complete=0
on_exit() {
    local status=$?
    trap - EXIT
    if (( status == 0 && complete != 1 )); then status=1; fi
    record "AFTER mode=$AMPA_BUDGET_SUMMARY_MODE exit=$status complete=$complete; no training executed; preserve all source runs"
    printf 'exit_code=%s\ncomplete=%s\nmode=%s\n' "$status" "$complete" "$AMPA_BUDGET_SUMMARY_MODE" > "$AMPA_BUDGET_SUMMARY_JOB_ROOT/exit.txt"
    exit "$status"
}
trap on_exit EXIT
if [[ $AMPA_BUDGET_SUMMARY_MODE == summarize ]]; then
    [[ ${AMPA_BUDGET_SUMMARY_CAMPAIGN:?} == "$ROOT_DIR"/runs/*
        && -d $AMPA_BUDGET_SUMMARY_CAMPAIGN && ! -L $AMPA_BUDGET_SUMMARY_CAMPAIGN
        && -f $AMPA_BUDGET_SUMMARY_CAMPAIGN/ACQUISITIONS_COMPLETE.txt
        && ${AMPA_BUDGET_SUMMARY_QA_ROOT:?} == "$ROOT_DIR"/runs/*
        && -f $AMPA_BUDGET_SUMMARY_QA_ROOT/qa.json && ! -L $AMPA_BUDGET_SUMMARY_QA_ROOT/qa.json
        && -f $AMPA_BUDGET_SUMMARY_QA_ROOT/PASS.txt && ! -L $AMPA_BUDGET_SUMMARY_QA_ROOT/PASS.txt ]] || exit 2
    read -r qa_digest _ < <(sha256sum "$AMPA_BUDGET_SUMMARY_QA_ROOT/qa.json")
    grep --fixed-strings --line-regexp --quiet -- "qa_sha256=$qa_digest" "$AMPA_BUDGET_SUMMARY_QA_ROOT/PASS.txt"
    grep --fixed-strings --line-regexp --quiet -- "summary_source_sha256=$digest" "$AMPA_BUDGET_SUMMARY_QA_ROOT/PASS.txt"
    jq -e --arg source "$digest" '.qa_passed == true and .tests == 7 and .training_executed == false
        and .partition == "debug" and .summary_source_sha256 == $source
        and (.job | type == "string" and test("^[0-9]+$"))' "$AMPA_BUDGET_SUMMARY_QA_ROOT/qa.json" > /dev/null
    qa_job=$(jq -er '.job' "$AMPA_BUDGET_SUMMARY_QA_ROOT/qa.json")
    grep --fixed-strings --line-regexp --quiet -- "job=$qa_job" "$AMPA_BUDGET_SUMMARY_QA_ROOT/PASS.txt"
    jobs=$(awk -F '\t' '{if(NF!=7 || $5 !~ /^[0-9]+$/ || $6 !~ /^[0-9]+$/) exit 2; print $5; print $6}' \
        "$AMPA_BUDGET_SUMMARY_CAMPAIGN/accepted.tsv" | sort -nu | paste -sd, -)
    accepted=$(wc -l < "$AMPA_BUDGET_SUMMARY_CAMPAIGN/accepted.tsv")
    (( accepted >= 6 && $(tr ',' '\n' <<< "$jobs" | wc -l) == 2 * accepted )) || exit 2
    accounting_start=${AMPA_BUDGET_ACCOUNTING_START:-2026-11-10}
    [[ $accounting_start =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]] || exit 2
    TZ=UTC SLURM_TIME_FORMAT=standard sacct -X --jobs="$jobs" --starttime="$accounting_start" \
        --noheader --parsable2 --format=JobIDRaw,State,ExitCode,Start,End,ElapsedRaw,NodeList,Partition,NNodes \
        > "$AMPA_BUDGET_SUMMARY_JOB_ROOT/accounting_utc.psv"
fi
record "BEFORE allocated mode=$AMPA_BUDGET_SUMMARY_MODE node=$SLURM_JOB_NODELIST container=setup/gh200-cuda130-runtime.toml; no training"
queue=$(squeue --array -u "$USER" --noheader --format='%i %P %T %D %R')
printf '%s\n' "$queue" > "$AMPA_BUDGET_SUMMARY_JOB_ROOT/queue-before-srun.txt"
requested=$(awk '{total += $4} END {print total+0}' <<< "$queue")
(( requested <= 4 )) || exit 2
srun --input=/dev/null --mpi=pmix --cpu-bind=none --container-remap-root --environment="$ROOT_DIR/setup/gh200-cuda130-runtime.toml" \
    --nodes=1 --ntasks=1 --gpus-per-task=1 --kill-on-bad-exit=1 \
    --output="$AMPA_BUDGET_SUMMARY_JOB_ROOT/logs/summary.out" --error="$AMPA_BUDGET_SUMMARY_JOB_ROOT/logs/summary.err" \
    bash "$SELF" --inside-container
complete=1
record "AFTER allocated mode=$AMPA_BUDGET_SUMMARY_MODE returned zero; inspect QA or metrics proof and independent interpretation review"
