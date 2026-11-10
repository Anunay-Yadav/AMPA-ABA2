#!/usr/bin/env bash
# Scheduler-only owner: no Python, training or trace parsing on the login node.
set -euo pipefail
readonly ROOT_DIR=${AMPA_UCCL_SWEEP_ROOT:?}
readonly BASE=${AMPA_UCCL_SWEEP_BASE:?}
readonly CAP_JOB=${AMPA_UCCL_NSYS_CAP_JOB:?}
readonly CAP_ROOT=${AMPA_UCCL_NSYS_CAP_ROOT:?}
readonly QA_JOB=${AMPA_UCCL_BUDGET_QA_JOB:?}
readonly QA_ROOT=${AMPA_UCCL_BUDGET_QA_ROOT:?}
readonly MANIFEST=${AMPA_UCCL_SWEEP_EXPECTED_SOURCE_MANIFEST:?}
readonly MANIFEST_SHA=${AMPA_UCCL_SWEEP_EXPECTED_SOURCE_SHA256:?}
readonly CONTROL="$ROOT_DIR/runs/$BASE-coordinator"
readonly SESSION="$ROOT_DIR/progress/sessions/2026-11-01-uccl-ep-comparison"
readonly STATUS="$ROOT_DIR/ai/qwen-uccl-block-sweep/iteration-01/STATUS.md"
[[ $BASE =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*$ && $CAP_JOB =~ ^[0-9]+$ && $QA_JOB =~ ^[0-9]+$ ]] || exit 2
[[ -d $ROOT_DIR && $CAP_ROOT == "$ROOT_DIR"/runs/* && $QA_ROOT == "$ROOT_DIR"/runs/*
   && $MANIFEST == "$ROOT_DIR"/runs/* && $MANIFEST_SHA =~ ^[a-f0-9]{64}$ ]] || exit 2
cd "$ROOT_DIR"
mkdir -p "$CONTROL"
mkdir "$CONTROL/owner.lock" || { printf 'An owner lock already exists: %s\n' "$CONTROL"; exit 2; }
printf '%s\n' "$$" > "$CONTROL/owner.pid"
hostname > "$CONTROL/owner.host"
cp "$ROOT_DIR/jobs/training/uccl_budget_sweep_coordinator.sh" "$CONTROL/coordinator-source.sh"

stage=initializing
current_job=$CAP_JOB
done_all=0
record() {
    local now note
    now=$(date -u +%FT%TZ)
    note="$now stage=$stage job=$current_job: $*"
    printf '%s\n' "$note" | tee -a "$CONTROL/timeline.md"
    printf '%s\n' "$note" > "$CONTROL/CURRENT.tmp"
    mv "$CONTROL/CURRENT.tmp" "$CONTROL/CURRENT.md"
    printf '\n%s\n' "$note" >> "$SESSION/timeline.md"
    printf '\n%s\n' "$note" >> "$SESSION/session_state.md"
    printf '\nPipeline milestone: %s\n' "$note" >> "$STATUS"
}
on_exit() {
    local code=$?
    trap - EXIT
    if (( done_all != 1 )); then
        record "STOP status=$code; preserve all outputs. No further jobs are submitted."
    fi
    printf 'exit_code=%s\nall_stages_complete=%s\n' "$code" "$done_all" > "$CONTROL/exit.txt"
    rmdir "$CONTROL/owner.lock" || true
    exit "$code"
}
trap on_exit EXIT

source_guard() {
    [[ $(sha256sum "$MANIFEST" | awk '{print $1}') == "$MANIFEST_SHA" ]]
    sha256sum --quiet --check "$MANIFEST" >> "$CONTROL/source-checks.log" 2>&1
}

wait_job() {
    local job=$1 row state exit_code nodes previous=''
    while :; do
        row=$(sacct -j "$job" --noheader --parsable2 --format=JobIDRaw,State,ExitCode,NodeList \
            | awk -F'|' -v wanted="$job" '$1 == wanted {print; exit}' || true)
        if [[ -n $row ]]; then
            IFS='|' read -r _ state exit_code nodes <<< "$row"
            if [[ $state != "$previous" ]]; then
                record "Job $job is $state exit=$exit_code nodes=$nodes"
                previous=$state
            fi
            case $state in
                COMPLETED)
                    [[ $exit_code == 0:0 ]] || return 1
                    printf '%s\n' "$nodes" > "$CONTROL/job-$job.nodes"
                    return 0 ;;
                FAILED*|CANCELLED*|TIMEOUT*|NODE_FAIL*|OUT_OF_MEMORY*|PREEMPTED*|BOOT_FAIL*|DEADLINE*)
                    return 1 ;;
            esac
        fi
        sleep 20
    done
}

wait_debug_capacity() {
    local needed=$1 requested previous=-1
    while :; do
        requested=$(squeue -u "$USER" --noheader --format='%P %D' \
            | awk '$1 ~ /^debug\*?$/ {sum += $2} END {print sum+0}')
        if (( requested + needed <= 4 )); then return 0; fi
        if [[ $requested != "$previous" ]]; then
            record "Waiting for debug capacity: $requested nodes requested/allocated; need $needed, limit4. Other partitions are untouched."
            previous=$requested
        fi
        sleep 20
    done
}

stage=capability
record "Owner started; wait for existing one-node native-report/counter capability job."
wait_job "$CAP_JOB"
jq -e '.native_report_generated == true and (.nsys_path | type == "string")' "$CAP_ROOT/capabilities.json" >/dev/null
stage=focused_checks
current_job=$QA_JOB
record "Wait for configured adapter/native-budget/multi-child Nsight checks."
wait_job "$QA_JOB"
[[ -f "$QA_ROOT/PASS.txt" && -f "$QA_ROOT/checks.json" ]]
[[ $(awk -F= '$1 == "job" {print $2}' "$QA_ROOT/PASS.txt") == "$QA_JOB" ]]
[[ $(awk -F= '$1 == "source_sha256" {print $2}' "$QA_ROOT/PASS.txt") == "$MANIFEST_SHA" ]]
[[ $(awk -F= '$1 == "checks_sha256" {print $2}' "$QA_ROOT/PASS.txt") == "$(sha256sum "$QA_ROOT/checks.json" | awk '{print $1}')" ]]
jq -e --arg job "$QA_JOB" --arg source "$MANIFEST_SHA" \
    '.valid_runtime_checks == true and .job == $job and .expected_manifest_sha256 == $source' "$QA_ROOT/checks.json" >/dev/null
source_guard
placement=''

for budget in 20 12 32 40; do
    stage="acquire_budget_$budget"
    current_job=not_submitted
    wait_debug_capacity 4
    source_guard
    export AMPA_UCCL_SWEEP_ID="$BASE-sms$budget"
    export AMPA_UCCL_SWEEP_SMS=$budget
    export AMPA_UCCL_NSYS_CAP_ROOT=$CAP_ROOT
    export AMPA_UCCL_VERIFIED_FIRST_ROOT="$ROOT_DIR/runs/uccl-debug5x5-20261104T1415Z-grouped-profile"
    local_args=()
    if [[ -n $placement ]]; then local_args+=(--nodelist="$placement"); fi
    record "BEFORE submit budget$budget; four exclusive debug nodes/21min, same-node request=${placement:-first_allocation}, ordinary top8,5+5, native Nsight reports."
    squeue -u "$USER"
    current_job=$(sbatch --parsable --job-name="ampa-uccl-b$budget" "${local_args[@]}" \
        "$ROOT_DIR/jobs/training/uccl_budget_sweep.sbatch")
    current_job=${current_job%%;*}
    [[ $current_job =~ ^[0-9]+$ ]]
    orchestration="$ROOT_DIR/runs/$AMPA_UCCL_SWEEP_ID"
    actual_run="$ROOT_DIR/runs/$AMPA_UCCL_SWEEP_ID-uccl_ordinary-control"
    printf '%s\t%s\t%s\t%s\n' "$budget" "$current_job" "$orchestration" "$actual_run" >> "$CONTROL/acquisitions.tsv"
    record "SUBMITTED budget$budget; output $actual_run"
    wait_job "$current_job"
    jq -e --argjson sms "$budget" '.success == true and .communication_sms == $sms and .completed_updates == 10 and .native_nsys_reports == 4' \
        "$orchestration/acquisition_completed.json" >/dev/null
    for node in 0 1 2 3; do [[ -s "$actual_run/nsight/node-$node-native.nsys-rep" ]]; done
    if [[ -z $placement ]]; then
        placement=$(cat "$CONTROL/job-$current_job.nodes")
        mapfile -t placement_hosts < <(scontrol show hostnames "$placement")
        [[ ${#placement_hosts[@]} == 4 ]]
        host_pattern='^[[:alnum:]_.-]+$'
        for host in "${placement_hosts[@]}"; do [[ $host =~ $host_pattern ]]; done
        printf '%s\n' "$placement" > "$CONTROL/requested-placement.txt"
    fi
    record "ACQUISITION COMPLETE budget$budget; four native reports exist. Full acceptance remains posthoc."
done

stage=posthoc
current_job=not_submitted
wait_debug_capacity 1
source_guard
[[ -f "$ROOT_DIR/jobs/analysis/uccl_budget_sweep_posthoc.sbatch" ]]
export AMPA_UCCL_SWEEP_CONTROL=$CONTROL
export AMPA_UCCL_SWEEP_POSTHOC_ID="$BASE-posthoc"
record "BEFORE one-node configured CPU-only Nsight export, training/route/batch/timeline/metric validation and portable report index."
squeue -u "$USER"
current_job=$(sbatch --parsable "$ROOT_DIR/jobs/analysis/uccl_budget_sweep_posthoc.sbatch")
current_job=${current_job%%;*}
[[ $current_job =~ ^[0-9]+$ ]]
printf '%s\n' "$current_job" > "$CONTROL/posthoc-job.txt"
record "SUBMITTED posthoc job; original Nsight reports are preserved."
wait_job "$current_job"
[[ -f "$ROOT_DIR/runs/$AMPA_UCCL_SWEEP_POSTHOC_ID/DELIVERY.json" ]]
jq -e '.valid_delivery == true' "$ROOT_DIR/runs/$AMPA_UCCL_SWEEP_POSTHOC_ID/DELIVERY.json" >/dev/null
stage=complete
record "COMPLETE four budgets and sixteen PC-openable .nsys-rep files; validated delivery runs/$AMPA_UCCL_SWEEP_POSTHOC_ID/README.md. Actual metric availability is recorded per report."
done_all=1
