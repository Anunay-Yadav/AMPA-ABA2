#!/usr/bin/env bash
# Scheduler-only coordinator. Training, tests and analysis stay on compute.
set -euo pipefail
readonly ROOT_DIR=${AMPA_OVERLAP_PROJECT_ROOT:?}
cd "$ROOT_DIR"
# A second controller must not race the global queue/capacity check.
exec 9> "$ROOT_DIR/.runtime/ep-overlap-coordinator.lock"
flock --nonblock 9 || exit 2
[[ ${AMPA_OVERLAP_CAMPAIGN_ID:?} =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*$
    && ${AMPA_OVERLAP_EXISTING_PILOT_JOB:?} =~ ^[0-9]+$
    && ${AMPA_OVERLAP_EXISTING_PILOT_ID:?} =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*$ ]] || exit 2
readonly OUT="$ROOT_DIR/runs/$AMPA_OVERLAP_CAMPAIGN_ID-coordinator"
[[ ! -e $OUT && ! -L $OUT ]] || exit 2
mkdir "$OUT"
printf '%s\n' "$$" > "$OUT/controller.pid"
cp "$ROOT_DIR/jobs/coordination/ep_overlap_campaign.sh" "$OUT/controller.sh"
sha256sum "$OUT/controller.sh" > "$OUT/controller.sha256"
readonly SESSION="$ROOT_DIR/progress/sessions/2026-11-10-qwen-ep-overlap"
readonly STATUS="$ROOT_DIR/ai/qwen-ep-overlap-campaign/iteration-01/STATUS.md"
export AMPA_OVERLAP_SESSION_DIR="$SESSION" AMPA_OVERLAP_STATUS_FILE="$STATUS"
export AMPA_OVERLAP_STEPS=10 AMPA_OVERLAP_BUDGET=20
phase=initializing
active_job=
complete=0
record() {
    local note="$(date -u '+%Y-%m-%d %H:%M:%S UTC'): $*"
    printf '%s\n' "$note" | tee -a "$OUT/timeline.md"
    printf '%s\n' "$note" > "$OUT/CURRENT.md"
    printf '\n- %s\n' "$note" >> "$SESSION/timeline.md"
    printf '\nCoordinator milestone: %s\n' "$note" >> "$SESSION/session_state.md"
    printf '\nCoordinator milestone: %s\n' "$note" >> "$STATUS"
}
on_exit() {
    local status=$?
    trap - EXIT
    if (( status == 0 && complete != 1 )); then status=1; fi
    record "COORDINATOR EXIT status=$status complete=$complete phase=$phase active_job=${active_job:-none}. Preserve all outputs. No failed acquisition is retried automatically."
    printf 'exit_code=%s\ncomplete=%s\nphase=%s\nactive_job=%s\n' "$status" "$complete" "$phase" "${active_job:-}" > "$OUT/exit.txt"
    exit "$status"
}
trap on_exit EXIT
check_source() {
    local digest
    read -r digest _ < <(sha256sum "${AMPA_OVERLAP_SOURCE_MANIFEST:?}")
    [[ $digest == "${AMPA_OVERLAP_SOURCE_SHA256:?}" ]] || return 2
    sha256sum --check --strict "$AMPA_OVERLAP_SOURCE_MANIFEST" > "$OUT/latest-source-check.txt"
}
check_source
[[ -f ${AMPA_OVERLAP_PREFLIGHT_ROOT:?}/PASS.txt ]]
: "${AMPA_OVERLAP_NSYS_CAP_ROOT:?}"
[[ $AMPA_OVERLAP_NSYS_CAP_ROOT == "$ROOT_DIR"/runs/*
    && -f $AMPA_OVERLAP_NSYS_CAP_ROOT/capabilities.json && -f $AMPA_OVERLAP_NSYS_CAP_ROOT/commands.json ]]
export AMPA_OVERLAP_NSYS_CAP_ROOT
jq -e --arg source "$AMPA_OVERLAP_SOURCE_SHA256" --arg cap "$AMPA_OVERLAP_NSYS_CAP_ROOT" \
    '.valid_preflight == true and .training_executed == false and .partition == "debug" and .source_manifest_sha256 == $source
     and .nsys_capability.root == $cap' \
    "$AMPA_OVERLAP_PREFLIGHT_ROOT/preflight.json" > /dev/null
read -r preflight_digest _ < <(sha256sum "$AMPA_OVERLAP_PREFLIGHT_ROOT/preflight.json")
expected_preflight_digest=$(sed -n 's/^preflight_sha256=//p' "$AMPA_OVERLAP_PREFLIGHT_ROOT/PASS.txt")
[[ $expected_preflight_digest =~ ^[0-9a-f]{64}$ && $preflight_digest == "$expected_preflight_digest" ]] || exit 2
printf '%s\n' "$AMPA_OVERLAP_SOURCE_MANIFEST" "$AMPA_OVERLAP_SOURCE_SHA256" "$AMPA_OVERLAP_PREFLIGHT_ROOT" > "$OUT/configuration.txt"

wait_job() {
    local job=$1 state previous= queue final
    while true; do
        if ! queue=$(squeue -h -j "$job" -o '%i %P %T %D %R'); then
            record "WATCH job=$job queue query returned nonzero; terminal accounting must independently confirm completion"
            break
        fi
        [[ -n $queue ]] || break
        state=$(awk '{print $3}' <<< "$queue")
        if [[ $state != "$previous" ]]; then
            record "WATCH job=$job $queue"
            previous=$state
        fi
        sleep 30
    done
    for attempt in {1..10}; do
        final=$(sacct -X -j "$job" --noheader --parsable2 --format=JobIDRaw,State,ExitCode,ElapsedRaw,NodeList)
        printf '%s\n' "$final" > "$OUT/job-$job-accounting.txt"
        if [[ $final == "$job|COMPLETED|0:0|"* ]]; then return 0; fi
        if [[ -n $final && $final != *'|RUNNING|'* && $final != *'|PENDING|'* && $final != *'|COMPLETING|'* ]]; then
            record "AFTER job=$job FAILED scheduler accounting: $final"
            return 1
        fi
        sleep 30
    done
    record "AFTER job=$job terminal accounting unavailable; stop for review"
    return 1
}

accepted() {
    local id=$1 job=$2 case_name=$3 capture=$4 steps=$5 budget=$6 posthoc_job=$7 expected_budget=null proof="$ROOT_DIR/runs/$1-posthoc/execution_accounting.json"
    if [[ $case_name == uccl || $case_name == uccl_ordinary ]]; then expected_budget=$budget; fi
    jq -e --arg job "$job" --arg case_name "$case_name" --arg capture "$capture" \
        --arg source "$AMPA_OVERLAP_SOURCE_SHA256" --arg posthoc "$posthoc_job" --argjson steps "$steps" --argjson budget "$expected_budget" \
        '.valid_execution_and_accounting == true and .valid_training_health == true
         and .valid_send_receive_audit == true and .valid_all_rank_startup_and_completion == true
         and .training_slurm_job_id == $job and .case == $case_name and .capture == $capture
         and .execution.slurm_job_id == $posthoc and .execution.partition == "debug"
         and .source_manifest_sha256 == $source and .completed_updates == $steps
         and .communication_sms == $budget
         and .qualification_only == ($steps == 2)' "$proof" > /dev/null
    if [[ $capture == torch ]]; then
        jq -e --arg source "$AMPA_OVERLAP_SOURCE_SHA256" --argjson count "$((steps * 16))" \
            '.capture_count == $count and (.captures | length) == $count
             and .positive_concurrency_capture_count > 0 and .source_manifest_sha256 == $source' \
            "$ROOT_DIR/runs/$id-posthoc/gpu_activity/summary.json" > /dev/null
    elif [[ $capture == nsys ]]; then
        jq -e --arg job "$job" --arg posthoc "$posthoc_job" --arg source "$AMPA_OVERLAP_SOURCE_SHA256" --argjson budget "$budget" \
            '.valid_overlap_nsys_run == true and .valid_nsight_timeline == true
               and .hardware_metrics.valid_bound_metric_analysis == true and .communication_sms == $budget
               and .training_slurm_job_id == $job and .source_manifest_sha256 == $source
               and .execution.slurm_job_id == $posthoc and .execution.partition == "debug"' \
            "$ROOT_DIR/runs/$id-nsys-posthoc/acceptance.json" > /dev/null
        jq -e --arg job "$job" --argjson budget "$budget" \
            '.valid_bound_metric_analysis == true and (.rank_update_windows | length) == 80
               and .communication_sms == $budget and .training_slurm_job_id == $job
               and .pooled_sample_summaries.during_any_payload.all_80_windows_available == true' \
            "$ROOT_DIR/runs/$id-nsys-posthoc/metrics.json" > /dev/null
    fi
    jq -e --arg id "$id" --arg training "$job" --arg posthoc "$posthoc_job" --arg source "$AMPA_OVERLAP_SOURCE_SHA256" \
        '.success == true and .experiment_id == $id and .training_slurm_job_id == $training
         and .posthoc_slurm_job_id == $posthoc and .source_manifest_sha256 == $source
         and .partition == "debug" and .requested_nodes == 1' \
        "$ROOT_DIR/runs/$id-posthoc-job-$posthoc_job/completed.json" > /dev/null
}

wait_capacity() {
    local needed=$1 queue requested previous=-1
    while true; do
        queue=$(squeue --array -u "$USER" --noheader --format='%i %P %T %D %R')
        printf '%s\n' "$queue" > "$OUT/latest-queue.txt"
        requested=$(awk '{sum += $4} END {print sum+0}' <<< "$queue")
        if (( requested + needed <= 4 )); then return 0; fi
        if [[ $requested != "$previous" ]]; then
            record "WAIT total requested/allocated nodes=$requested; need=$needed; global cap=4"
            previous=$requested
        fi
        sleep 30
        check_source
    done
}

# label|case|capture|budget|steps. The first pilot is already submitted by root.
rows=(
    'pilot-native-ordinary-off|ordinary|off|20|2'
    'pilot-native-grouped-torch|grouped|torch|20|2'
    'pilot-uccl-ordinary-torch|uccl_ordinary|torch|20|2'
    'pilot-uccl-grouped-off|uccl|off|20|2'
    'native-ordinary-off|ordinary|off|20|10'
    'native-ordinary-torch|ordinary|torch|20|10'
    'native-grouped-off|grouped|off|20|10'
    'native-grouped-torch|grouped|torch|20|10'
    'uccl-ordinary-off|uccl_ordinary|off|20|10'
    'uccl-ordinary-torch|uccl_ordinary|torch|20|10'
    'uccl-grouped-off|uccl|off|20|10'
    'uccl-grouped-torch|uccl|torch|20|10'
    'uccl-ordinary-nsys20|uccl_ordinary|nsys|20|10'
    'uccl-ordinary-nsys12|uccl_ordinary|nsys|12|10'
    'uccl-ordinary-nsys32|uccl_ordinary|nsys|32|10'
    'uccl-ordinary-nsys40|uccl_ordinary|nsys|40|10'
)
for index in "${!rows[@]}"; do
    IFS='|' read -r label case_name capture budget steps <<< "${rows[$index]}"
    phase=$label
    if (( index == 0 )); then
        id=$AMPA_OVERLAP_EXISTING_PILOT_ID
        active_job=$AMPA_OVERLAP_EXISTING_PILOT_JOB
        record "RESUME root-submitted pilot=$id job=$active_job"
    else
        id="$AMPA_OVERLAP_CAMPAIGN_ID-$label"
        export AMPA_OVERLAP_ID="$id" AMPA_OVERLAP_CASE="$case_name" AMPA_OVERLAP_CAPTURE="$capture"
        export AMPA_OVERLAP_BUDGET="$budget" AMPA_OVERLAP_STEPS="$steps"
        check_source
        record "BEFORE submit $label case=$case_name capture=$capture budget=$budget steps=$steps; four exclusive debug GH200 nodes/21minutes, acquisition only"
        wait_capacity 4
        active_job=$(sbatch --parsable --exclude=nid006657 "$ROOT_DIR/jobs/training/ep_overlap_checked.sbatch")
        [[ $active_job =~ ^[0-9]+$ ]] || exit 2
        record "AFTER submit $label job=$active_job; acceptance pending"
    fi
    training_job=$active_job
    export AMPA_OVERLAP_ID="$id" AMPA_OVERLAP_CASE="$case_name" AMPA_OVERLAP_CAPTURE="$capture"
    export AMPA_OVERLAP_BUDGET="$budget" AMPA_OVERLAP_STEPS="$steps"
    export AMPA_OVERLAP_ROOT="$ROOT_DIR/runs/$id" AMPA_OVERLAP_RUN="$ROOT_DIR/runs/$id-training"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$label" "$case_name" "$capture" "$budget" "$steps" "$active_job" "$ROOT_DIR/runs/$id" >> "$OUT/acquisitions.tsv"
    wait_job "$active_job"
    jq -e --arg job "$training_job" --arg source "$AMPA_OVERLAP_SOURCE_SHA256" \
        --arg case_name "$case_name" --arg capture "$capture" --argjson steps "$steps" \
        '.success == true and .workers_srun_returned_zero == true and .job == $job
         and .partition == "debug" and .case == $case_name and .capture == $capture
         and .completed_updates == $steps and .campaign_source_sha256 == $source' \
        "$AMPA_OVERLAP_ROOT/acquisition_completed.json" > /dev/null
    phase="$label-posthoc"
    check_source
    record "AFTER acquisition $label training_job=$training_job COMPLETED0:0; BEFORE separate1debugnode/90minute posthoc submission; no4node request remains"
    wait_capacity 1
    active_job=$(sbatch --parsable --exclude=nid006657 "$ROOT_DIR/jobs/analysis/ep_overlap_posthoc.sbatch")
    [[ $active_job =~ ^[0-9]+$ ]] || exit 2
    posthoc_job=$active_job
    record "AFTER submit posthoc=$posthoc_job training_job=$training_job; strict scoped acceptance pending"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$label" "$case_name" "$capture" "$budget" "$steps" \
        "$training_job" "$posthoc_job" "$AMPA_OVERLAP_ROOT" >> "$OUT/posthoc-jobs.tsv"
    wait_job "$posthoc_job"
    accepted "$id" "$training_job" "$case_name" "$capture" "$steps" "$budget" "$posthoc_job"
    record "AFTER $label training_job=$training_job posthoc_job=$posthoc_job both COMPLETED0:0 and scoped execution/route/health/capture gates passed; diagnostic limits retained"
    active_job=
done
complete=1
phase=acquisitions_complete
printf 'qualification_pilots=4\nfull_acquisitions=12\nfull_optimizer_updates=120\ncomparison_accepted=false\n' > "$OUT/ACQUISITIONS_COMPLETE.txt"
record "All4 qualification pilots and12 full acquisitions completed with scoped gates. Final cross-case synthesis remains separate; no causal block-budget or free-SM conclusion follows automatically."
