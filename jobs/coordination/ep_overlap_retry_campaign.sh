#!/usr/bin/env bash
# Scheduler-only continuation. Keep the validated 144 compute files unchanged.
set -euo pipefail
readonly ROOT_DIR=${AMPA_OVERLAP_PROJECT_ROOT:?}
cd "$ROOT_DIR"
exec 9> "$ROOT_DIR/.runtime/ep-overlap-coordinator.lock"
flock --nonblock 9 || exit 2
[[ ${AMPA_OVERLAP_CAMPAIGN_ID:?} =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*$
    && ${AMPA_OVERLAP_EXISTING_PILOT_JOB:?} =~ ^[0-9]+$
    && ${AMPA_OVERLAP_EXISTING_PILOT_ID:?} =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*$ ]] || exit 2
readonly APPROVED_SOURCE=ed74f4058652bb6f96dfa0496f70f7a888a150a2b2ee639df2d7edd499c8620a
readonly MAX_ATTEMPTS=3
readonly OUT="$ROOT_DIR/runs/$AMPA_OVERLAP_CAMPAIGN_ID-retry-coordinator"
readonly SELF="$ROOT_DIR/jobs/coordination/ep_overlap_retry_campaign.sh"
[[ ! -e $OUT && ! -L $OUT && -f $SELF && ! -L $SELF
    && ${AMPA_OVERLAP_SOURCE_SHA256:?} == "$APPROVED_SOURCE"
    && ${AMPA_OVERLAP_SOURCE_MANIFEST:?} == "$ROOT_DIR"/runs/*
    && -f $AMPA_OVERLAP_SOURCE_MANIFEST && ! -L $AMPA_OVERLAP_SOURCE_MANIFEST ]] || exit 2
mkdir "$OUT"
mkdir "$OUT/retained" "$OUT/failures"
printf '%s\n' "$$" > "$OUT/controller.pid"
cp "$SELF" "$OUT/controller.sh"
sha256sum "$SELF" "$OUT/controller.sh" > "$OUT/controller.sha256"
readonly SESSION="$ROOT_DIR/progress/sessions/2026-11-10-qwen-ep-overlap"
readonly STATUS="$ROOT_DIR/ai/qwen-ep-overlap-campaign/iteration-01/STATUS.md"
export AMPA_OVERLAP_SESSION_DIR="$SESSION" AMPA_OVERLAP_STATUS_FILE="$STATUS"
phase=initializing
active_job=
complete=0
record() {
    local note="$(date -u '+%Y-%m-%d %H:%M:%S UTC'): $*"
    printf '%s\n' "$note" | tee -a "$OUT/timeline.md" || return 2
    printf '%s\n' "$note" > "$OUT/CURRENT.md" || return 2
    printf '\n- %s\n' "$note" >> "$SESSION/timeline.md" || return 2
    printf '\nRetry coordinator milestone: %s\n' "$note" >> "$SESSION/session_state.md" || return 2
    printf '\nRetry coordinator milestone: %s\n' "$note" >> "$STATUS" || return 2
}
on_exit() {
    local status=$?
    trap - EXIT
    if (( status == 0 && complete != 1 )); then status=1; fi
    record "RETRY COORDINATOR EXIT status=$status complete=$complete phase=$phase active_job=${active_job:-none}. All failed outputs remain preserved."
    printf 'exit_code=%s\ncomplete=%s\nphase=%s\nactive_job=%s\n' "$status" "$complete" "$phase" "${active_job:-}" > "$OUT/exit.txt"
    exit "$status"
}
trap on_exit EXIT
check_source() {
    local digest
    read -r digest _ < <(sha256sum "$AMPA_OVERLAP_SOURCE_MANIFEST") || return 2
    [[ $digest == "$APPROVED_SOURCE" ]] || return 2
    sha256sum --check --strict "$AMPA_OVERLAP_SOURCE_MANIFEST" > "$OUT/latest-source-check.txt" || return 2
    sha256sum --check --strict "$OUT/controller.sha256" > "$OUT/latest-driver-check.txt" || return 2
    if [[ -f $OUT/retained/proof-files.sha256 ]]; then
        sha256sum --check --strict "$OUT/retained/proof-files.sha256" > "$OUT/latest-retained-check.txt" || return 2
    fi
}
check_source
[[ ${AMPA_OVERLAP_PREFLIGHT_ROOT:?} == "$ROOT_DIR"/runs/*
    && -f $AMPA_OVERLAP_PREFLIGHT_ROOT/PASS.txt && ! -L $AMPA_OVERLAP_PREFLIGHT_ROOT/PASS.txt
    && -f $AMPA_OVERLAP_PREFLIGHT_ROOT/preflight.json && ! -L $AMPA_OVERLAP_PREFLIGHT_ROOT/preflight.json
    && ${AMPA_OVERLAP_NSYS_CAP_ROOT:?} == "$ROOT_DIR"/runs/*
    && -f $AMPA_OVERLAP_NSYS_CAP_ROOT/capabilities.json && -f $AMPA_OVERLAP_NSYS_CAP_ROOT/commands.json ]] || exit 2
export AMPA_OVERLAP_NSYS_CAP_ROOT
jq -e --arg source "$APPROVED_SOURCE" --arg cap "$AMPA_OVERLAP_NSYS_CAP_ROOT" \
    '.valid_preflight == true and .training_executed == false and .partition == "debug"
     and .source_manifest_sha256 == $source and .nsys_capability.root == $cap' \
    "$AMPA_OVERLAP_PREFLIGHT_ROOT/preflight.json" > /dev/null
read -r preflight_digest _ < <(sha256sum "$AMPA_OVERLAP_PREFLIGHT_ROOT/preflight.json")
expected_preflight_digest=$(sed -n 's/^preflight_sha256=//p' "$AMPA_OVERLAP_PREFLIGHT_ROOT/PASS.txt")
[[ $expected_preflight_digest =~ ^[0-9a-f]{64}$ && $preflight_digest == "$expected_preflight_digest" ]] || exit 2
printf '%s\n' "$AMPA_OVERLAP_SOURCE_MANIFEST" "$APPROVED_SOURCE" "$AMPA_OVERLAP_PREFLIGHT_ROOT" > "$OUT/configuration.txt"
record "Resume after3909844 failed before Torch startup: root diagnosed node0 read-only git status exceeding30seconds on shared storage. Retry fresh independent cases without changing validated compute source. Limit=$MAX_ATTEMPTS attempts per stage including the root-submitted retry."

wait_job() {
    # 0=verified success, 1=verified terminal failure, 2=unknown state: stop.
    local job=$1 queue state previous= final row exit_code
    while true; do
        if ! queue=$(squeue -h -j "$job" -o '%i %P %T %D %R'); then
            record "WATCH job=$job queue query failed; consult accounting before any retry" || return 2
            break
        fi
        [[ -n $queue ]] || break
        state=$(awk '{print $3}' <<< "$queue") || return 2
        if [[ $state != "$previous" ]]; then
            record "WATCH job=$job $queue" || return 2
            previous=$state
        fi
        sleep 30 || return 2
    done
    for poll in {1..10}; do
        if ! final=$(sacct -X -j "$job" --noheader --parsable2 --format=JobIDRaw,State,ExitCode,ElapsedRaw,NodeList); then
            sleep 30 || return 2
            continue
        fi
        printf '%s\n' "$final" > "$OUT/job-$job-accounting.txt" || return 2
        row=$(awk -F'|' -v job="$job" '$1 == job {print}' <<< "$final") || return 2
        if [[ $row == "$job|COMPLETED|0:0|"* ]]; then return 0; fi
        IFS='|' read -r _ state exit_code _ <<< "$row"
        if [[ $state =~ ^(FAILED|CANCELLED|TIMEOUT|NODE_FAIL|PREEMPTED|OUT_OF_MEMORY|BOOT_FAIL|DEADLINE|REVOKED|SPECIAL_EXIT|COMPLETED) ]]; then
            record "VERIFIED FAILURE job=$job state=$state exit=$exit_code accounting=$row" || return 2
            return 1
        fi
        sleep 30 || return 2
    done
    record "STOP job=$job terminal accounting unknown; no automatic retry can safely assume this job ended" || return 2
    return 2
}

wait_capacity() {
    local needed=$1 queue requested previous=-1
    while true; do
        check_source || return 2
        queue=$(squeue --array -u "$USER" --noheader --format='%i %P %T %D %R') || return 2
        printf '%s\n' "$queue" > "$OUT/latest-queue.txt" || return 2
        requested=$(awk '{sum += $4} END {print sum+0}' <<< "$queue") || return 2
        if (( requested + needed <= 4 )); then return 0; fi
        if [[ $requested != "$previous" ]]; then
            record "WAIT requested/allocated=$requested need=$needed global cap=4" || return 2
            previous=$requested
        fi
        sleep 30 || return 2
    done
}

accepted() {
    # Every gate propagates failure explicitly, including under an `if`.
    local id=$1 training=$2 case_name=$3 capture=$4 steps=$5 budget=$6 posthoc=$7 expected_budget=null
    if [[ $case_name == uccl || $case_name == uccl_ordinary ]]; then expected_budget=$budget; fi
    jq -e --arg training "$training" --arg posthoc "$posthoc" --arg case_name "$case_name" --arg capture "$capture" \
        --arg source "$APPROVED_SOURCE" --argjson steps "$steps" --argjson budget "$expected_budget" \
        '.valid_execution_and_accounting == true and .valid_training_health == true
         and .valid_send_receive_audit == true and .valid_all_rank_startup_and_completion == true
         and .training_slurm_job_id == $training and .case == $case_name and .capture == $capture
         and .execution.slurm_job_id == $posthoc and .execution.partition == "debug"
         and .source_manifest_sha256 == $source and .completed_updates == $steps
         and .communication_sms == $budget and .qualification_only == ($steps == 2)
         and .execution_contract.ddp_delay_wgrad_compute == true' \
        "$ROOT_DIR/runs/$id-posthoc/execution_accounting.json" > /dev/null || return 1
    if [[ $capture == torch ]]; then
        jq -e --arg source "$APPROVED_SOURCE" --argjson count "$((steps * 16))" \
            '.capture_count == $count and (.captures | length) == $count
             and .positive_concurrency_capture_count > 0 and .source_manifest_sha256 == $source' \
            "$ROOT_DIR/runs/$id-posthoc/gpu_activity/summary.json" > /dev/null || return 1
    elif [[ $capture == nsys ]]; then
        jq -e --arg training "$training" --arg posthoc "$posthoc" --arg source "$APPROVED_SOURCE" --argjson budget "$budget" \
            '.valid_overlap_nsys_run == true and .valid_nsight_timeline == true
             and .hardware_metrics.valid_bound_metric_analysis == true and .communication_sms == $budget
             and .training_slurm_job_id == $training and .source_manifest_sha256 == $source
             and .execution.slurm_job_id == $posthoc and .execution.partition == "debug"' \
            "$ROOT_DIR/runs/$id-nsys-posthoc/acceptance.json" > /dev/null || return 1
        jq -e --arg training "$training" --argjson budget "$budget" \
            '.valid_bound_metric_analysis == true and (.rank_update_windows | length) == 80
             and .communication_sms == $budget and .training_slurm_job_id == $training
             and .pooled_sample_summaries.during_any_payload.all_80_windows_available == true' \
            "$ROOT_DIR/runs/$id-nsys-posthoc/metrics.json" > /dev/null || return 1
    fi
    jq -e --arg id "$id" --arg training "$training" --arg posthoc "$posthoc" --arg source "$APPROVED_SOURCE" \
        '.success == true and .experiment_id == $id and .training_slurm_job_id == $training
         and .posthoc_slurm_job_id == $posthoc and .source_manifest_sha256 == $source
         and .partition == "debug" and .requested_nodes == 1' \
        "$ROOT_DIR/runs/$id-posthoc-job-$posthoc/completed.json" > /dev/null || return 1
    return 0
}

retained=(
    'native-ordinary-off|ep-overlap-debug-ddp-pilot-native-ordinary-off-20261110T111819Z|3908166|3909025|ordinary|off'
    'native-grouped-torch|ep-overlap-debug-ddp-enabled-20261110T111937Z-pilot-native-grouped-torch|3909134|3909367|grouped|torch'
)
for retained_row in "${retained[@]}"; do
    IFS='|' read -r label id training posthoc case_name capture <<< "$retained_row"
    phase="retained-$label"
    proof="$ROOT_DIR/runs/$id-posthoc/execution_accounting.json"
    receipt="$ROOT_DIR/runs/$id-posthoc-job-$posthoc/completed.json"
    sha256sum "$proof" "$receipt" >> "$OUT/retained/proof-files.sha256"
    if [[ $capture == torch ]]; then
        sha256sum "$ROOT_DIR/runs/$id-posthoc/gpu_activity/summary.json" >> "$OUT/retained/proof-files.sha256"
    fi
    accepted "$id" "$training" "$case_name" "$capture" 2 20 "$posthoc"
    check_source
    jq -n --arg id "$id" --arg training "$training" --arg posthoc "$posthoc" --arg proof "$proof" --arg source "$APPROVED_SOURCE" \
        '{retained_qualification:true, experiment_id:$id, training_slurm_job_id:$training,
          posthoc_slurm_job_id:$posthoc, proof_path:$proof, source_manifest_sha256:$source}' \
        > "$OUT/retained/$label.json"
    printf 'retained\t%s\t%s\t%s\t%s\t%s\n' "$label" "$training" "$posthoc" "$id" "$proof" >> "$OUT/qualifications.tsv"
    record "RETAINED $label training=$training posthoc=$posthoc: unchanged source-bound2update qualification and mode gates passed"
done
check_source

failure_details() {
    local id=$1 attempt=$2 stage=$3 training=$4 posthoc=$5 path destination="$OUT/failures/$1-attempt$2.log"
    printf 'stage=%s\ntraining_job=%s\nposthoc_job=%s\nrun_id=%s\n' "$stage" "$training" "$posthoc" "$id" > "$destination" || return 2
    for path in "$ROOT_DIR/runs/$id/logs/node-0.err" "$ROOT_DIR/runs/$id/logs/node-0.out" \
        "$ROOT_DIR/.runtime/slurms/slurm-ampa-overlap-checked-$training.err" \
        "$ROOT_DIR/runs/$id-posthoc-job-$posthoc/logs/posthoc.err" \
        "$ROOT_DIR/runs/$id-posthoc-job-$posthoc/logs/posthoc.out"; do
        if [[ -f $path && ! -L $path ]]; then
            printf '\n--- %s ---\n' "$path" >> "$destination" || return 2
            tail -n 35 "$path" >> "$destination" || return 2
        fi
    done
}

# Two native qualifications are retained above. Finish the other two, then12 full cases.
rows=(
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
    stage_accepted=0
    for (( attempt=1; attempt<=MAX_ATTEMPTS; attempt++ )); do
        phase="$label-attempt$attempt-training"
        failure=
        posthoc=
        if (( index == 0 && attempt == 1 )); then
            id=$AMPA_OVERLAP_EXISTING_PILOT_ID
            training=$AMPA_OVERLAP_EXISTING_PILOT_JOB
            active_job=$training
            record "RESUME fresh root-submitted UCCL qualification id=$id training=$training attempt=$attempt"
        else
            id="$AMPA_OVERLAP_CAMPAIGN_ID-$label-attempt$attempt"
            [[ ! -e $ROOT_DIR/runs/$id && ! -L $ROOT_DIR/runs/$id
                && ! -e $ROOT_DIR/runs/$id-training && ! -L $ROOT_DIR/runs/$id-training ]] || exit 2
            export AMPA_OVERLAP_ID="$id" AMPA_OVERLAP_CASE="$case_name" AMPA_OVERLAP_CAPTURE="$capture"
            export AMPA_OVERLAP_BUDGET="$budget" AMPA_OVERLAP_STEPS="$steps"
            record "BEFORE training submit stage=$label attempt=$attempt id=$id;4debugnodes/21minutes, independent C0, unchanged source"
            wait_capacity 4
            if ! active_job=$(sbatch --parsable --exclude=nid006657 "$ROOT_DIR/jobs/training/ep_overlap_checked.sbatch"); then
                record "STOP training submission failed or identity unknown; root must inspect queue before any retry"
                exit 2
            fi
            [[ $active_job =~ ^[0-9]+$ ]] || exit 2
            training=$active_job
            record "AFTER training submit stage=$label attempt=$attempt training=$training id=$id"
        fi
        export AMPA_OVERLAP_ID="$id" AMPA_OVERLAP_CASE="$case_name" AMPA_OVERLAP_CAPTURE="$capture"
        export AMPA_OVERLAP_BUDGET="$budget" AMPA_OVERLAP_STEPS="$steps"
        export AMPA_OVERLAP_ROOT="$ROOT_DIR/runs/$id" AMPA_OVERLAP_RUN="$ROOT_DIR/runs/$id-training"
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$label" "$attempt" "$case_name" "$capture" "$budget" "$steps" "$training" "$id" >> "$OUT/acquisitions.tsv"
        if wait_job "$training"; then
            if ! jq -e --arg job "$training" --arg source "$APPROVED_SOURCE" --arg case_name "$case_name" \
                --arg capture "$capture" --argjson steps "$steps" \
                '.success == true and .workers_srun_returned_zero == true and .job == $job
                 and .partition == "debug" and .case == $case_name and .capture == $capture
                 and .completed_updates == $steps and .campaign_source_sha256 == $source' \
                "$AMPA_OVERLAP_ROOT/acquisition_completed.json" > /dev/null; then failure=acquisition_gate; fi
        else
            status=$?
            if (( status != 1 )); then exit "$status"; fi
            failure=training_scheduler
        fi
        if [[ -z $failure ]]; then
            phase="$label-attempt$attempt-posthoc"
            record "BEFORE posthoc submit training=$training id=$id; training released;1debugnode/90minutes"
            wait_capacity 1
            if ! active_job=$(sbatch --parsable --exclude=nid006657 "$ROOT_DIR/jobs/analysis/ep_overlap_posthoc.sbatch"); then
                record "STOP posthoc submission failed or identity unknown; root must inspect queue before any retry"
                exit 2
            fi
            [[ $active_job =~ ^[0-9]+$ ]] || exit 2
            posthoc=$active_job
            record "AFTER posthoc submit training=$training posthoc=$posthoc id=$id attempt=$attempt"
            printf '%s\t%s\t%s\t%s\t%s\n' "$label" "$attempt" "$training" "$posthoc" "$id" >> "$OUT/posthoc-jobs.tsv"
            if wait_job "$posthoc"; then
                if accepted "$id" "$training" "$case_name" "$capture" "$steps" "$budget" "$posthoc"; then
                    stage_accepted=1
                else
                    failure=scoped_acceptance_gate
                fi
            else
                status=$?
                if (( status != 1 )); then exit "$status"; fi
                failure=posthoc_scheduler
            fi
        fi
        if (( stage_accepted == 1 )); then
            record "ACCEPTED stage=$label attempt=$attempt training=$training posthoc=$posthoc id=$id; all scoped gates passed"
            printf '%s\t%s\t%s\t%s\t%s\n' "$label" "$attempt" "$training" "$posthoc" "$id" >> "$OUT/accepted.tsv"
            if [[ $steps == 2 ]]; then
                printf 'new\t%s\t%s\t%s\t%s\t%s\n' "$label" "$training" "$posthoc" "$id" "$ROOT_DIR/runs/$id-posthoc/execution_accounting.json" >> "$OUT/qualifications.tsv"
            fi
            active_job=
            break
        fi
        failure_details "$id" "$attempt" "$failure" "$training" "$posthoc"
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$label" "$attempt" "$failure" "$training" "$posthoc" "$id" >> "$OUT/failed-attempts.tsv"
        record "FAILED stage=$label attempt=$attempt reason=$failure training=$training posthoc=${posthoc:-none} id=$id. Preserve all outputs; diagnostics=$OUT/failures/$id-attempt$attempt.log"
        active_job=
        if (( attempt == MAX_ATTEMPTS )); then
            printf 'stage=%s\nattempts=%s\nlast_failure=%s\nlast_training_job=%s\nlast_posthoc_job=%s\nnext_action=root must diagnose preserved failure logs, repair or qualify a new source if necessary, and resume with a new coordinator ID\n' \
                "$label" "$MAX_ATTEMPTS" "$failure" "$training" "$posthoc" > "$OUT/RETRIES_EXHAUSTED.txt"
            record "STOP stage=$label reached$MAX_ATTEMPTS failed attempts; root diagnosis required; no false acceptance or automatic gate/source changes"
            exit 1
        fi
        record "RETRY stage=$label next_attempt=$((attempt + 1)); rerun the whole case with a unique ID because partial fixed posthoc directories cannot be overwritten safely"
    done
    (( stage_accepted == 1 )) || exit 1
done
complete=1
phase=remaining_acquisitions_complete
printf 'retained_qualification_pilots=2\nnew_qualification_pilots=2\nqualification_pilots=4\nfull_acquisitions=12\nfull_optimizer_updates=120\ncomparison_accepted=false\n' > "$OUT/ACQUISITIONS_COMPLETE.txt"
record "Complete:2retained+2new qualifications and12 full acquisitions passed scoped gates. Final cross-case synthesis and research conclusions remain separate."
