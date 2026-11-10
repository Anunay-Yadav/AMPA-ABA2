#!/usr/bin/env bash
# Scheduler/file orchestration only. All workloads run on configured compute.
set -euo pipefail
readonly ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
cd "$ROOT_DIR"
[[ ${AMPA_UCCL_DEBUG_PIPELINE_ID:?} =~ ^[a-zA-Z0-9_-]+$ \
    && ${AMPA_UCCL_DEBUG_QA_JOB:?} =~ ^[0-9]+$ \
    && ${AMPA_UCCL_DEBUG_QA_DIR:?} == "$ROOT_DIR"/runs/* ]] || exit 2
: "${AMPA_UCCL_DEBUG_EXPECTED_SOURCE_MANIFEST:?}" "${AMPA_UCCL_DEBUG_EXPECTED_SOURCE_SHA256:?}"
readonly PIPELINE_ID=$AMPA_UCCL_DEBUG_PIPELINE_ID
readonly PIPELINE_OUT="$ROOT_DIR/runs/$PIPELINE_ID-coordinator"
readonly SESSION_DIR="$ROOT_DIR/progress/sessions/2026-11-01-uccl-ep-comparison"
readonly TASK_STATUS="$ROOT_DIR/ai/qwen-debug-5plus5/iteration-01/STATUS.md"
[[ ! -e "$PIPELINE_OUT" && ! -L "$PIPELINE_OUT" ]] || exit 2
[[ "$AMPA_UCCL_DEBUG_EXPECTED_SOURCE_MANIFEST" == "$ROOT_DIR"/runs/* \
    && -f "$AMPA_UCCL_DEBUG_EXPECTED_SOURCE_MANIFEST" && ! -L "$AMPA_UCCL_DEBUG_EXPECTED_SOURCE_MANIFEST" \
    && "$AMPA_UCCL_DEBUG_EXPECTED_SOURCE_SHA256" =~ ^[0-9a-f]{64}$ ]] || exit 2
mkdir -p "$PIPELINE_OUT"
printf '%s\n' "$$" > "$PIPELINE_OUT/pid"
hostname > "$PIPELINE_OUT/hostname"
cp "$AMPA_UCCL_DEBUG_EXPECTED_SOURCE_MANIFEST" "$PIPELINE_OUT/source.sha256"
current_stage=focused_checks
current_job=$AMPA_UCCL_DEBUG_QA_JOB
finished=0
record() {
    local note="$(date -u '+%Y-%m-%d %H:%M:%S UTC') debug_pipeline=$PIPELINE_ID stage=$current_stage job=$current_job: $*"
    printf '%s\n' "$note" >> "$PIPELINE_OUT/timeline.md"
    printf '%s\n' "$note" > "$PIPELINE_OUT/CURRENT.md"
    printf '\n%s\n' "$note" >> "$SESSION_DIR/timeline.md"
    printf '\n%s\n' "$note" >> "$SESSION_DIR/session_state.md"
    printf '\n%s\n' "$note" >> "$TASK_STATUS"
}
finish() {
    local code=$?
    trap - EXIT TERM INT
    if (( finished == 0 )); then
        record "STOP exit=$code; no later submissions. Current allocation is preserved."
    fi
    exit "$code"
}
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
check_source() {
    [[ $(sha256sum "$AMPA_UCCL_DEBUG_EXPECTED_SOURCE_MANIFEST" | cut -d ' ' -f 1) == "$AMPA_UCCL_DEBUG_EXPECTED_SOURCE_SHA256" ]]
    [[ $(sha256sum "$PIPELINE_OUT/source.sha256" | cut -d ' ' -f 1) == "$AMPA_UCCL_DEBUG_EXPECTED_SOURCE_SHA256" ]]
    sha256sum --check "$PIPELINE_OUT/source.sha256" > "$PIPELINE_OUT/source-last-check.log"
}
wait_job() {
    local entry state code
    while true; do
        entry=$(sacct -n -X -j "$current_job" --format=State,ExitCode -P | head -n 1) || { sleep 30; continue; }
        state=${entry%%|*}
        code=${entry#*|}
        case "$state" in
            COMPLETED)
                [[ "$code" == 0:0 ]] || return 1
                record "AFTER compute COMPLETED exit0:0"
                return 0 ;;
            FAILED*|CANCELLED*|TIMEOUT*|OUT_OF_MEMORY*|NODE_FAIL*|PREEMPTED*|BOOT_FAIL*|DEADLINE*|REVOKED*)
                record "AFTER compute failure state=$state exit=$code"
                return 1 ;;
            *) sleep 30 ;;
        esac
    done
}
wait_capacity() {
    local nodes
    while true; do
        nodes=$(squeue -h -u "$USER" -o '%D' | awk '{total += $1} END {print total+0}')
        if [[ "$nodes" == 0 ]]; then
            check_source
            squeue -u "$USER" -o '%.18i %.12P %.10T %.6D %.10l %R'
            return 0
        fi
        sleep 30
    done
}
submitted() {
    current_job=${1%%;*}
    [[ "$current_job" =~ ^[0-9]+$ ]]
    printf '%s\t%s\t%s\n' "$current_stage" "$current_job" "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" >> "$PIPELINE_OUT/jobs.tsv"
    record "SUBMITTED; partition=debug; one allocation at a time; configured container"
    wait_job
}
record "Waiting for focused configured checks; eight cases use5warmup+5measured and separate allocation placement"
wait_job
[[ -f "$AMPA_UCCL_DEBUG_QA_DIR/PASS.txt" ]]
[[ $(cat "$AMPA_UCCL_DEBUG_QA_DIR/job-id") == "$AMPA_UCCL_DEBUG_QA_JOB" ]]
[[ $(sha256sum "$AMPA_UCCL_DEBUG_QA_DIR/source.sha256" | cut -d ' ' -f 1) == "$AMPA_UCCL_DEBUG_EXPECTED_SOURCE_SHA256" ]]
check_source
# The slowest native profile runs first and establishes cold-start headroom.
cases=(grouped:profile grouped:control ordinary:control ordinary:profile uccl:control uccl:profile uccl_ordinary:control uccl_ordinary:profile)
for item in "${cases[@]}"; do
    experiment=${item%:*}
    leg=${item#*:}
    run_id="$PIPELINE_ID-$experiment-$leg"
    current_stage="acquire-$experiment-$leg"
    record "BEFORE compute four nodes/16GPUs/22m30s; independent C0 start,10realupdates, all-rank profile when enabled"
    wait_capacity
    job=$(env AMPA_UCCL_DEBUG_ID="$run_id" AMPA_UCCL_DEBUG_CASE="$experiment" AMPA_UCCL_DEBUG_LEG="$leg" \
        AMPA_UCCL_DEBUG_STATUS_FILE="$TASK_STATUS" AMPA_UCCL_DEBUG_SESSION_DIR="$SESSION_DIR" \
        sbatch --parsable --time=00:22:30 --mem=450G --exclude=nid007499,nid007534 jobs/training/uccl_ep_debug_case.sbatch)
    submitted "$job"
    [[ -f "$ROOT_DIR/runs/$run_id/acquisition_completed.json" ]]
    check_source
    current_stage="validate-$experiment-$leg"
    record "BEFORE compute one-node input/health/fingerprint/routing/source/trace acceptance; no training allocation overlaps"
    wait_capacity
    job=$(env AMPA_UCCL_DEBUG_VALIDATION_ID="$run_id-validation" \
        AMPA_UCCL_DEBUG_ORCHESTRATION_DIR="$ROOT_DIR/runs/$run_id" \
        sbatch --parsable jobs/analysis/uccl_debug_validate.sbatch)
    submitted "$job"
    [[ -f "$ROOT_DIR/runs/$run_id-validation/acceptance.json" ]]
    printf '%s\t%s\t%s\n' "$experiment" "$leg" "$ROOT_DIR/runs/$run_id-validation/acceptance.json" >> "$PIPELINE_OUT/accepted-cases.tsv"
    check_source
done
current_stage=awaiting_cross_allocation_report
record "All eight cases completed and passed individual gates. Cross-allocation sample/placement comparison, numerical audit and final report/presentation review remain separate."
finished=1
