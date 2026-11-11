#!/usr/bin/env bash
# Consume matching source QA, then acquire one case. Analysis uses another job.
#SBATCH --job-name=ampa-overlap-checked
#SBATCH --account=infra01
#SBATCH --partition=debug
#SBATCH --qos=normal
#SBATCH --constraint=gh
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --exclusive
#SBATCH --time=00:21:00
#SBATCH --no-requeue
#SBATCH --output=.runtime/slurms/slurm-%x-%j.out
#SBATCH --error=.runtime/slurms/slurm-%x-%j.err
set -euo pipefail
[[ $# == 0 && ${SLURM_JOB_PARTITION:-} == debug && ${SLURM_JOB_NUM_NODES:-} == 4 ]] || exit 2
cd "${SLURM_SUBMIT_DIR:?}"
readonly ROOT_DIR=$PWD
export AMPA_OVERLAP_STEPS=${AMPA_OVERLAP_STEPS:-10}
export AMPA_OVERLAP_BUDGET=${AMPA_OVERLAP_BUDGET:-20}
[[ $AMPA_OVERLAP_STEPS =~ ^(2|10)$ && $AMPA_OVERLAP_BUDGET =~ ^(12|20|32|40)$ ]] || exit 2
export AMPA_OVERLAP_ROOT="$ROOT_DIR/runs/${AMPA_OVERLAP_ID:?}"
export AMPA_OVERLAP_RUN="$ROOT_DIR/runs/$AMPA_OVERLAP_ID-training"
export AMPA_OVERLAP_SESSION_DIR=${AMPA_OVERLAP_SESSION_DIR:-$ROOT_DIR/progress/sessions/2026-11-10-qwen-ep-overlap}
record() {
    local note="$(date -u '+%Y-%m-%d %H:%M:%S UTC') job $SLURM_JOB_ID: $*"
    printf '\n- %s\n' "$note" >> "$AMPA_OVERLAP_SESSION_DIR/timeline.md"
    printf '\nCompute milestone: %s\n' "$note" >> "$AMPA_OVERLAP_SESSION_DIR/session_state.md"
}
phase=preflight_proof
complete=0
on_exit() {
    local status=$?
    trap - EXIT
    if (( status == 0 && complete != 1 )); then status=1; fi
    record "AFTER checked pipeline phase=$phase exit=$status complete=$complete. Preserve original outputs; no acceptance implied by an incomplete phase."
    exit "$status"
}
trap on_exit EXIT
[[ ${AMPA_OVERLAP_PREFLIGHT_ROOT:?} == "$ROOT_DIR"/runs/*
    && -f $AMPA_OVERLAP_PREFLIGHT_ROOT/preflight.json && ! -L $AMPA_OVERLAP_PREFLIGHT_ROOT/preflight.json
    && -f $AMPA_OVERLAP_PREFLIGHT_ROOT/PASS.txt && ! -L $AMPA_OVERLAP_PREFLIGHT_ROOT/PASS.txt ]] || exit 2
read -r preflight_digest _ < <(sha256sum "$AMPA_OVERLAP_PREFLIGHT_ROOT/preflight.json")
expected_preflight_digest=$(sed -n 's/^preflight_sha256=//p' "$AMPA_OVERLAP_PREFLIGHT_ROOT/PASS.txt")
[[ $expected_preflight_digest =~ ^[0-9a-f]{64}$ && $preflight_digest == "$expected_preflight_digest" ]] || exit 2
jq -e --arg source "${AMPA_OVERLAP_SOURCE_SHA256:?}" \
    '.valid_preflight == true and .training_executed == false and .partition == "debug"
     and .source_manifest_sha256 == $source and (.job | type == "string" and test("^[0-9]+$"))
     and (.focused_tests | length == 6)
     and ([.focused_tests[].file] | sort) == (["test_ep_overlap_entrypoint.py", "test_overlap_trace.py",
       "test_ep_overlap_capture.py", "test_ep_overlap_validate.py", "test_ep_overlap_gpu_activity.py",
       "test_ep_overlap_nsys_posthoc.py"] | sort)
     and all(.focused_tests[]; .tests | type == "number" and . > 0 and . == floor)' \
    "$AMPA_OVERLAP_PREFLIGHT_ROOT/preflight.json" > /dev/null
preflight_job=$(jq -er '.job' "$AMPA_OVERLAP_PREFLIGHT_ROOT/preflight.json")
grep --fixed-strings --line-regexp --quiet -- "job=$preflight_job" "$AMPA_OVERLAP_PREFLIGHT_ROOT/PASS.txt"
grep --fixed-strings --line-regexp --quiet -- "source_sha256=$AMPA_OVERLAP_SOURCE_SHA256" "$AMPA_OVERLAP_PREFLIGHT_ROOT/PASS.txt"
record "Matching debug preflight=$AMPA_OVERLAP_PREFLIGHT_ROOT job=$preflight_job proof=$preflight_digest consumed without repeated tests; BEFORE independent C0 acquisition on4debugnodes/21minutes"
phase=acquisition
bash "$ROOT_DIR/jobs/training/ep_overlap_case.sbatch"
record "AFTER acquisition worker/health summaries. Four-node job ends here; strict execution/routing/capture analysis remains pending a separate1node debug allocation."
complete=1
phase=acquisition_complete
