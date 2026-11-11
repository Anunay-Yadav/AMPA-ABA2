#!/usr/bin/env bash
# Analyze one completed acquisition after its four-node allocation ends.
#SBATCH --job-name=ampa-overlap-posthoc
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
[[ $# == 0 && ${SLURM_JOB_PARTITION:-} == debug && ${SLURM_JOB_NUM_NODES:-} == 1
    && ${SLURM_JOB_ID:-} =~ ^[0-9]+$ ]] || exit 2
readonly ROOT_DIR=${SLURM_SUBMIT_DIR:?}
cd "$ROOT_DIR"
[[ ${AMPA_OVERLAP_ID:?} =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*$
    && ${AMPA_OVERLAP_ROOT:?} == "$ROOT_DIR/runs/$AMPA_OVERLAP_ID"
    && ${AMPA_OVERLAP_RUN:?} == "$ROOT_DIR/runs/$AMPA_OVERLAP_ID-training"
    && ${AMPA_OVERLAP_CASE:?} =~ ^(ordinary|grouped|uccl_ordinary|uccl)$
    && ${AMPA_OVERLAP_CAPTURE:?} =~ ^(off|torch|nsys)$
    && ${AMPA_OVERLAP_STEPS:?} =~ ^(2|10)$ && ${AMPA_OVERLAP_BUDGET:?} =~ ^(12|20|32|40)$ ]] || exit 2
[[ -d $AMPA_OVERLAP_ROOT && ! -L $AMPA_OVERLAP_ROOT
    && -d $AMPA_OVERLAP_RUN && ! -L $AMPA_OVERLAP_RUN ]] || exit 2
if [[ $AMPA_OVERLAP_CAPTURE == nsys ]]; then
    [[ $AMPA_OVERLAP_CASE == uccl_ordinary ]] || exit 2
else
    [[ $AMPA_OVERLAP_BUDGET == 20 ]] || exit 2
fi
: "${AMPA_OVERLAP_SOURCE_MANIFEST:?}" "${AMPA_OVERLAP_SOURCE_SHA256:?}"
[[ $AMPA_OVERLAP_SOURCE_MANIFEST == "$ROOT_DIR"/runs/* && -f $AMPA_OVERLAP_SOURCE_MANIFEST
    && ! -L $AMPA_OVERLAP_SOURCE_MANIFEST && $AMPA_OVERLAP_SOURCE_SHA256 =~ ^[0-9a-f]{64}$ ]] || exit 2
export AMPA_OVERLAP_SESSION_DIR=${AMPA_OVERLAP_SESSION_DIR:-$ROOT_DIR/progress/sessions/2026-11-10-qwen-ep-overlap}
export AMPA_OVERLAP_STATUS_FILE=${AMPA_OVERLAP_STATUS_FILE:-$ROOT_DIR/ai/qwen-ep-overlap-campaign/iteration-01/STATUS.md}
[[ $AMPA_OVERLAP_SESSION_DIR == "$ROOT_DIR"/progress/sessions/* && -d $AMPA_OVERLAP_SESSION_DIR
    && -f $AMPA_OVERLAP_SESSION_DIR/session_state.md
    && $AMPA_OVERLAP_STATUS_FILE == "$ROOT_DIR"/ai/*/iteration-*/STATUS.md
    && -f $AMPA_OVERLAP_STATUS_FILE ]] || exit 2
training_job=$(jq -er '.job | select(type == "string" and test("^[0-9]+$"))' "$AMPA_OVERLAP_ROOT/acquisition.json")
expected_budget=null
if [[ $AMPA_OVERLAP_CASE == uccl || $AMPA_OVERLAP_CASE == uccl_ordinary ]]; then expected_budget=$AMPA_OVERLAP_BUDGET; fi
jq -e --arg job "$training_job" --arg run "$AMPA_OVERLAP_RUN" --arg case_name "$AMPA_OVERLAP_CASE" \
    --arg capture "$AMPA_OVERLAP_CAPTURE" --arg source "$AMPA_OVERLAP_SOURCE_SHA256" \
    --argjson steps "$AMPA_OVERLAP_STEPS" --argjson budget "$expected_budget" \
    '.success == true and .workers_srun_returned_zero == true and .job == $job
     and .partition == "debug" and .run_dir == $run and .case == $case_name and .capture == $capture
     and .completed_updates == $steps and .communication_sms == $budget
     and .campaign_source_sha256 == $source' "$AMPA_OVERLAP_ROOT/acquisition_completed.json" > /dev/null
export AMPA_OVERLAP_POSTHOC_ROOT="$ROOT_DIR/runs/$AMPA_OVERLAP_ID-posthoc-job-$SLURM_JOB_ID"
[[ ! -e $AMPA_OVERLAP_POSTHOC_ROOT && ! -L $AMPA_OVERLAP_POSTHOC_ROOT ]] || exit 2
mkdir "$AMPA_OVERLAP_POSTHOC_ROOT"
mkdir "$AMPA_OVERLAP_POSTHOC_ROOT/logs"
printf '%s\n' "$SLURM_JOB_ID" > "$AMPA_OVERLAP_POSTHOC_ROOT/job-id"
printf '%s\n' "$training_job" > "$AMPA_OVERLAP_POSTHOC_ROOT/training-job-id"
record() {
    local note="$(date -u '+%Y-%m-%d %H:%M:%S UTC') posthoc_job=$SLURM_JOB_ID training_job=$training_job: $*"
    printf '%s\n' "$note" >> "$AMPA_OVERLAP_POSTHOC_ROOT/timeline.md"
    printf '%s\n' "$note" > "$AMPA_OVERLAP_POSTHOC_ROOT/CURRENT.md"
    printf '\n- %s\n' "$note" >> "$AMPA_OVERLAP_SESSION_DIR/timeline.md"
    printf '\nCompute milestone: %s\n' "$note" >> "$AMPA_OVERLAP_SESSION_DIR/session_state.md"
    printf '\nCompute milestone: %s\n' "$note" >> "$AMPA_OVERLAP_STATUS_FILE"
}
complete=0
on_exit() {
    local status=$?
    trap - EXIT
    if (( status == 0 && complete != 1 )); then status=1; fi
    record "AFTER posthoc status=$status complete=$complete; preserve all original and analysis files; scoped gate results determine acceptance"
    printf 'exit_code=%s\nposthoc_job=%s\ntraining_job=%s\n' "$status" "$SLURM_JOB_ID" "$training_job" > "$AMPA_OVERLAP_POSTHOC_ROOT/exit.txt"
    exit "$status"
}
trap on_exit EXIT
record "BEFORE compute strict execution/accounting and capture analysis case=$AMPA_OVERLAP_CASE capture=$AMPA_OVERLAP_CAPTURE steps=$AMPA_OVERLAP_STEPS budget=$AMPA_OVERLAP_BUDGET node=$SLURM_JOB_NODELIST container=setup/gh200-cuda130-runtime.toml;1debugnode/90minutes"
queue=$(squeue --array -u "$USER" --noheader --format='%i %P %T %D %R')
printf '%s\n' "$queue"
requested=$(awk '{total += $4} END {print total+0}' <<< "$queue")
(( requested <= 4 )) || exit 2
srun --mpi=pmix --cpu-bind=none --container-remap-root --environment="$ROOT_DIR/setup/gh200-cuda130-runtime.toml" \
    --nodes=1 --ntasks=1 --gpus-per-task=1 --kill-on-bad-exit=1 \
    --output="$AMPA_OVERLAP_POSTHOC_ROOT/logs/posthoc.out" --error="$AMPA_OVERLAP_POSTHOC_ROOT/logs/posthoc.err" \
    bash "$ROOT_DIR/jobs/training/ep_overlap_posthoc.sh" --analyze
jq -n --arg id "$AMPA_OVERLAP_ID" --arg training "$training_job" --arg posthoc "$SLURM_JOB_ID" \
    --arg source "$AMPA_OVERLAP_SOURCE_SHA256" --arg root "$AMPA_OVERLAP_ROOT" \
    '{success:true, experiment_id:$id, training_slurm_job_id:$training, posthoc_slurm_job_id:$posthoc,
      source_manifest_sha256:$source, orchestration_dir:$root, partition:"debug", requested_nodes:1}' \
    > "$AMPA_OVERLAP_POSTHOC_ROOT/completed.json"
complete=1
record "AFTER allocated posthoc returned0. Inspect execution_accounting and mode-specific scoped acceptance before requesting the next4nodes."
