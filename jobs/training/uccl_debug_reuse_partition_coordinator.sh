#!/usr/bin/env bash
# Scheduler-only success chain. User authorizes four debug nodes alongside preemptable work.
# Capture first; defer all analysis gates.
set -euo pipefail
readonly ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
cd "$ROOT_DIR"
[[ ${AMPA_UCCL_REUSE_PIPELINE_ID:?} =~ ^[a-zA-Z0-9_-]+$ \
    && ${AMPA_UCCL_REUSE_QA_JOB:?} =~ ^[0-9]+$ \
    && ${AMPA_UCCL_REUSE_QA_DIR:?} == "$ROOT_DIR"/runs/* \
    && ${AMPA_UCCL_VERIFIED_FIRST_ROOT:?} == "$ROOT_DIR"/runs/* ]] || exit 2
readonly PIPELINE_ID=$AMPA_UCCL_REUSE_PIPELINE_ID
readonly PIPELINE_OUT="$ROOT_DIR/runs/$PIPELINE_ID-partition-coordinator"
readonly SESSION_DIR="$ROOT_DIR/progress/sessions/2026-11-01-uccl-ep-comparison"
readonly TASK_STATUS="$ROOT_DIR/ai/qwen-debug-5plus5/iteration-01/STATUS.md"
[[ ! -e "$PIPELINE_OUT" && ! -L "$PIPELINE_OUT" ]] || exit 2
mkdir -p "$PIPELINE_OUT"
printf '%s\n' "$$" > "$PIPELINE_OUT/pid"
hostname > "$PIPELINE_OUT/hostname"
sha256sum "$ROOT_DIR/jobs/training/uccl_debug_reuse_partition_coordinator.sh" > "$PIPELINE_OUT/coordinator-source.sha256"
current_stage=one_time_launcher_check
current_job=$AMPA_UCCL_REUSE_QA_JOB
finished=0
record() {
    local note="$(date -u '+%Y-%m-%d %H:%M:%S UTC') reuse_pipeline=$PIPELINE_ID stage=$current_stage job=$current_job: $*"
    printf '%s\n' "$note" >> "$PIPELINE_OUT/timeline.md"
    printf '%s\n' "$note" > "$PIPELINE_OUT/CURRENT.md"
    printf '\n%s\n' "$note" >> "$SESSION_DIR/timeline.md"
    printf '\n%s\n' "$note" >> "$SESSION_DIR/session_state.md"
    printf '\n%s\n' "$note" >> "$TASK_STATUS"
}
finish() {
    local code=$?
    trap - EXIT TERM INT
    if (( finished == 0 )); then record "STOP exit=$code; later cases are not submitted and current allocation is preserved"; fi
    exit "$code"
}
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
wait_job() {
    local entry state code
    while true; do
        entry=$(sacct -n -X -j "$current_job" --format=State,ExitCode -P | head -n 1) || { sleep 15; continue; }
        state=${entry%%|*}
        code=${entry#*|}
        case "$state" in
            COMPLETED) [[ "$code" == 0:0 ]] || return 1; record "AFTER compute COMPLETED exit0:0"; return 0 ;;
            FAILED*|CANCELLED*|TIMEOUT*|OUT_OF_MEMORY*|NODE_FAIL*|PREEMPTED*|BOOT_FAIL*|DEADLINE*|REVOKED*)
                record "AFTER compute failure state=$state exit=$code"; return 1 ;;
            *) sleep 15 ;;
        esac
    done
}
wait_capacity() {
    local nodes
    while true; do
        nodes=$(squeue -h -u "$USER" -p debug -o '%D' | awk '{total += $1} END {print total+0}')
        if [[ "$nodes" == 0 ]]; then
            squeue -u "$USER" -o '%.18i %.12P %.10T %.6D %.10l %R'
            return 0
        fi
        sleep 15
    done
}
record "One-time focused verification of the changed reuse launcher; no per-case file-verification or analysis jobs"
wait_job
[[ -f "$AMPA_UCCL_REUSE_QA_DIR/PASS.txt" && $(cat "$AMPA_UCCL_REUSE_QA_DIR/job-id") == "$AMPA_UCCL_REUSE_QA_JOB" ]]
# Reuse only the successful first execution, without rerunning its gates.
previous_job=3851782
current_stage=first_completed_case
current_job=$previous_job
wait_job
[[ -f "$AMPA_UCCL_VERIFIED_FIRST_ROOT/acquisition_completed.json" \
    && -f "$AMPA_UCCL_VERIFIED_FIRST_ROOT/reviewed_inputs.json" ]]
printf 'grouped\tprofile\t%s\t%s\n' "$previous_job" "$AMPA_UCCL_VERIFIED_FIRST_ROOT" > "$PIPELINE_OUT/acquisitions.tsv"
cases=(grouped:control ordinary:control ordinary:profile uccl:control uccl:profile uccl_ordinary:control uccl_ordinary:profile)
for item in "${cases[@]}"; do
    experiment=${item%:*}
    leg=${item#*:}
    run_id="$PIPELINE_ID-$experiment-$leg"
    current_stage="acquire-$experiment-$leg"
    record "BEFORE compute four debug nodes/22minutes/5+5; same measurement records, reuse input verification, analysis deferred"
    wait_capacity
    dependency=()
    # Slurm purges completed job IDs after five minutes. Use native afterok
    # while the parent exists; successful accounting is checked in all cases.
    if scontrol show job "$previous_job" >/dev/null 2>&1; then
        dependency=(--dependency="afterok:$previous_job")
    fi
    squeue -u "$USER" -o '%.18i %.12P %.10T %.6D %.10l %R'
    job=$(env AMPA_UCCL_DEBUG_ID="$run_id" AMPA_UCCL_DEBUG_CASE="$experiment" AMPA_UCCL_DEBUG_LEG="$leg" \
        AMPA_UCCL_DEBUG_STATUS_FILE="$TASK_STATUS" AMPA_UCCL_DEBUG_SESSION_DIR="$SESSION_DIR" \
        sbatch --parsable --time=00:22:00 --mem=450G --exclude=nid007499,nid007534,nid007650 \
        "${dependency[@]}" jobs/training/uccl_ep_debug_reuse.sbatch)
    current_job=${job%%;*}
    [[ "$current_job" =~ ^[0-9]+$ ]]
    printf '%s\t%s\t%s\t%s\n' "$experiment" "$leg" "$current_job" "$ROOT_DIR/runs/$run_id" >> "$PIPELINE_OUT/acquisitions.tsv"
    record "SUBMITTED after successful job$previous_job; one four-node request at a time"
    wait_job
    [[ -f "$ROOT_DIR/runs/$run_id/acquisition_completed.json" ]]
    previous_job=$current_job
done
current_stage=all_acquisition_complete_analysis_deferred
record "All eight acquisitions completed. Raw measurement records are preserved; detailed routing/trace acceptance and final comparisons have not run."
finished=1
