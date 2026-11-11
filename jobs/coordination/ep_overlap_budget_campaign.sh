#!/usr/bin/env bash
# Root-owned scheduler only. Statistics run inside posthoc allocations.
set -euo pipefail
readonly ROOT_DIR=${AMPA_OVERLAP_PROJECT_ROOT:-$PWD}
cd "$ROOT_DIR"
readonly SELF="$ROOT_DIR/jobs/coordination/ep_overlap_budget_campaign.sh"
readonly PROTOCOL=ep_overlap_ordinary_off_budget_5_plus_5_v1
readonly MAX_ATTEMPTS=3
[[ ${AMPA_BUDGET_CAMPAIGN_ID:?} =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$
    && ${AMPA_OVERLAP_SOURCE_SHA256:?} =~ ^[0-9a-f]{64}$
    && ${AMPA_OVERLAP_SOURCE_MANIFEST:?} == "$ROOT_DIR"/runs/*
    && ${AMPA_BUDGET_PREFLIGHT_ROOT:?} == "$ROOT_DIR"/runs/*
    && ${AMPA_BUDGET_QUALIFICATION_PROOF:?} == "$ROOT_DIR"/runs/* ]] || exit 2
[[ -z ${AMPA_BUDGET_SOURCE_MANIFEST:-} || $AMPA_BUDGET_SOURCE_MANIFEST == "$AMPA_OVERLAP_SOURCE_MANIFEST" ]] || exit 2
[[ -z ${AMPA_BUDGET_SOURCE_SHA256:-} || $AMPA_BUDGET_SOURCE_SHA256 == "$AMPA_OVERLAP_SOURCE_SHA256" ]] || exit 2
readonly SOURCE_SHA=$AMPA_OVERLAP_SOURCE_SHA256
readonly OUT="$ROOT_DIR/runs/$AMPA_BUDGET_CAMPAIGN_ID-coordinator"
readonly PREFLIGHT="$AMPA_BUDGET_PREFLIGHT_ROOT/preflight.json"
readonly QUALIFICATION=$AMPA_BUDGET_QUALIFICATION_PROOF
readonly ANALYZER="$ROOT_DIR/analysis/ep_overlap_budget_compare.py"
readonly VALIDATOR="$ROOT_DIR/tools/ep_overlap_budget_validate.py"
readonly FABRIC_CANONICAL=/opt/amazon/efa/lib/libfabric.so.1.25.0
readonly FABRIC_SHA=998956619429c918c9dba07f8a68faf5d9e8c630b799496e261e54622797db91
readonly FABRIC_COPY="$ROOT_DIR/runs/qwen-debug-results-20261105T0905Z-local-work/container-libraries/recorded-libfabric.so.1"
readonly FABRIC_MAP="$ROOT_DIR/runs/qwen-debug-results-20261105T0905Z-local-work/container-libraries/library-map.json"
readonly FABRIC_MAP_SHA=b9b81b6be0e0af682fc2004fe0a1f0963f23d345a89951fbabd55a4c8cdf96a9
export AMPA_BUDGET_SESSION_DIR=${AMPA_BUDGET_SESSION_DIR:-$ROOT_DIR/progress/sessions/2026-11-10-qwen-ep-overlap}
export AMPA_BUDGET_STATUS_FILE=${AMPA_BUDGET_STATUS_FILE:-$ROOT_DIR/ai/ep-overlap-budget-coordinator/iteration-01/STATUS.md}
[[ -d $ROOT_DIR/.runtime && -f $SELF && ! -L $SELF
    && -f $AMPA_OVERLAP_SOURCE_MANIFEST && ! -L $AMPA_OVERLAP_SOURCE_MANIFEST
    && -f $PREFLIGHT && ! -L $PREFLIGHT && -f $QUALIFICATION && ! -L $QUALIFICATION
    && $AMPA_BUDGET_SESSION_DIR == "$ROOT_DIR"/progress/sessions/*
    && -f $AMPA_BUDGET_SESSION_DIR/timeline.md && -f $AMPA_BUDGET_SESSION_DIR/session_state.md
    && $AMPA_BUDGET_STATUS_FILE == "$ROOT_DIR"/ai/*/iteration-*/STATUS.md
    && -f $AMPA_BUDGET_STATUS_FILE && ! -e $OUT && ! -L $OUT ]] || exit 2
exec 9> "$ROOT_DIR/.runtime/ep-overlap-budget-coordinator.lock"
flock --nonblock 9 || exit 2
mkdir "$OUT"
mkdir "$OUT/failures" "$OUT/decisions"
printf '%s\n' "$$" > "$OUT/controller.pid"
hostname > "$OUT/controller.hostname"
cp "$SELF" "$OUT/controller.sh"
sha256sum "$SELF" "$OUT/controller.sh" > "$OUT/controller.sha256"
cp "$AMPA_OVERLAP_SOURCE_MANIFEST" "$OUT/campaign-source.sha256"
touch "$OUT/accepted.tsv" "$OUT/accepted-proofs.sha256"
read -r ledger_sha _ < <(sha256sum "$OUT/accepted.tsv")
phase=initializing
active_job=
complete=0
record() {
    local note="$(date -u '+%Y-%m-%d %H:%M:%S UTC'): $*"
    printf '%s\n' "$note" | tee -a "$OUT/timeline.md" || return 2
    printf '%s\n' "$note" > "$OUT/CURRENT.md" || return 2
    printf '\n- %s\n' "$note" >> "$AMPA_BUDGET_SESSION_DIR/timeline.md" || return 2
    printf '\nBudget coordinator milestone: %s\n' "$note" >> "$AMPA_BUDGET_SESSION_DIR/session_state.md" || return 2
    printf '\nBudget coordinator milestone: %s\n' "$note" >> "$AMPA_BUDGET_STATUS_FILE" || return 2
}
on_exit() {
    local status=$?
    trap - EXIT
    if (( status == 0 && complete != 1 )); then status=1; fi
    record "EXIT status=$status complete=$complete phase=$phase active_job=${active_job:-none}; preserve every attempt"
    printf 'exit_code=%s\ncomplete=%s\nphase=%s\nactive_job=%s\n' "$status" "$complete" "$phase" "$active_job" > "$OUT/exit.txt"
    exit "$status"
}
trap on_exit EXIT
regular() {
    [[ $1 == "$ROOT_DIR"/* && $1 != */../* && $1 != */..
        && $1 != *$'\n'* && $1 != *$'\r'* && $1 != *$'\t'*
        && -f $1 && ! -L $1 ]]
}
verify_binding() {
    local path=$1 expected=$2 digest map_digest
    # Preserve the container identity. Verify only this approved external
    # binding through its existing exact-byte extraction on the shared store.
    if [[ $path == "$FABRIC_CANONICAL" && $expected == "$FABRIC_SHA" ]]; then
        regular "$FABRIC_COPY" && regular "$FABRIC_MAP" || return 2
        read -r digest _ < <(sha256sum "$FABRIC_COPY") || return 2
        read -r map_digest _ < <(sha256sum "$FABRIC_MAP") || return 2
        [[ $digest == "$FABRIC_SHA" && $map_digest == "$FABRIC_MAP_SHA" ]] || return 2
        if [[ ! -f $OUT/container-library-binding.json ]]; then
            jq -n --arg canonical "$FABRIC_CANONICAL" --arg copy "$FABRIC_COPY" --arg digest "$FABRIC_SHA" \
                --arg map "$FABRIC_MAP" --arg map_sha "$FABRIC_MAP_SHA" \
                '{original_container_path:$canonical,verified_local_copy:$copy,sha256:$digest,
                  existing_extraction_mapping:$map,mapping_sha256:$map_sha,
                  scope:"Scheduler-host byte equivalence only; allocated validators check the actual container library."}' \
                > "$OUT/container-library-binding.json" || return 2
            sha256sum "$FABRIC_COPY" "$FABRIC_MAP" "$OUT/container-library-binding.json" >> "$OUT/accepted-proofs.sha256" || return 2
        fi
        return 0
    fi
    regular "$path" || return 2
    read -r digest _ < <(sha256sum "$path") || return 2
    [[ $digest == "$expected" ]] || return 2
}
source_entry() {
    local value
    value=$(awk -v path="$1" 'substr($0,67)==path {print substr($0,1,64)}' "$AMPA_OVERLAP_SOURCE_MANIFEST") || return 2
    [[ $value =~ ^[0-9a-f]{64}$ ]] || return 2
    printf '%s\n' "$value"
}
check_source() {
    local digest
    read -r digest _ < <(sha256sum "$AMPA_OVERLAP_SOURCE_MANIFEST") || return 2
    [[ $digest == "$SOURCE_SHA" ]] || return 2
    cmp -s "$AMPA_OVERLAP_SOURCE_MANIFEST" "$OUT/campaign-source.sha256" || return 2
    sha256sum --check --strict "$AMPA_OVERLAP_SOURCE_MANIFEST" > "$OUT/latest-source-check.txt" || return 2
    sha256sum --check --strict "$OUT/controller.sha256" > "$OUT/latest-controller-check.txt" || return 2
    if [[ -f $OUT/qualification-inputs.sha256 ]]; then
        sha256sum --check --strict "$OUT/qualification-inputs.sha256" > "$OUT/latest-qualification-check.txt" || return 2
    fi
    if [[ -s $OUT/accepted-proofs.sha256 ]]; then
        sha256sum --check --strict "$OUT/accepted-proofs.sha256" > "$OUT/latest-accepted-check.txt" || return 2
    fi
    read -r digest _ < <(sha256sum "$OUT/accepted.tsv") || return 2
    [[ $digest == "$ledger_sha" ]] || return 2
}
check_source
readonly ANALYZER_SHA=$(source_entry "$ANALYZER")
readonly VALIDATOR_SHA=$(source_entry "$VALIDATOR")
[[ $ANALYZER_SHA =~ ^[0-9a-f]{64}$ && $VALIDATOR_SHA =~ ^[0-9a-f]{64}$ ]] || exit 2
regular "$AMPA_BUDGET_PREFLIGHT_ROOT/PASS.txt"
read -r preflight_sha _ < <(sha256sum "$PREFLIGHT")
grep --fixed-strings --line-regexp --quiet -- "preflight_sha256=$preflight_sha" "$AMPA_BUDGET_PREFLIGHT_ROOT/PASS.txt"
grep --fixed-strings --line-regexp --quiet -- "source_sha256=$SOURCE_SHA" "$AMPA_BUDGET_PREFLIGHT_ROOT/PASS.txt"
jq -e --arg source "$SOURCE_SHA" --arg protocol "$PROTOCOL" \
    '.valid_preflight == true and .training_executed == false and .partition == "debug"
     and .source_manifest_sha256 == $source and .experiment_protocol == $protocol
     and (.job | type == "string" and test("^[0-9]+$"))
     and (.focused_tests | type == "array" and length > 0)
     and all(.focused_tests[]; .tests | type == "number" and . > 0 and . == floor)
     and (.allowed_budgets | type == "array" and length > 0)
     and all(.allowed_budgets[]; type == "number" and . >= 4 and . <= 64 and . % 2 == 0)
     and ([12,20,32,40] - .allowed_budgets | length) == 0' "$PREFLIGHT" > /dev/null
preflight_job=$(jq -er '.job' "$PREFLIGHT")
grep --fixed-strings --line-regexp --quiet -- "job=$preflight_job" "$AMPA_BUDGET_PREFLIGHT_ROOT/PASS.txt"
allowed() {
    [[ $1 =~ ^[1-9][0-9]*$ && $1 != 20 ]] || return 2
    jq -e --argjson budget "$1" '.allowed_budgets | index($budget) != null' "$PREFLIGHT" > /dev/null || return 2
}
wait_job() {
    # 0: completed0:0; 1: verified terminal failure; 2: unknown, stop.
    local job=$1 queue state previous= final row code
    while true; do
        if ! queue=$(squeue -h -j "$job" -o '%i %P %T %D %R'); then
            record "WATCH job=$job query failed; inspect accounting before any retry" || return 2
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
        row=$(awk -F'|' -v job="$job" '$1==job {print}' <<< "$final") || return 2
        [[ $row != "$job|COMPLETED|0:0|"* ]] || return 0
        IFS='|' read -r _ state code _ <<< "$row"
        if [[ $state =~ ^(FAILED|CANCELLED|TIMEOUT|NODE_FAIL|PREEMPTED|OUT_OF_MEMORY|BOOT_FAIL|DEADLINE|REVOKED|SPECIAL_EXIT|COMPLETED) ]]; then
            record "VERIFIED FAILURE job=$job state=$state exit=$code accounting=$row" || return 2
            return 1
        fi
        sleep 30 || return 2
    done
    record "STOP job=$job terminal identity unknown; no retry follows an observation timeout" || return 2
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
validate_pair() {
    # Propagate every failure explicitly when called conditionally.
    local id=$1 candidate=$2 order=$3 steps=$4 training=$5 posthoc=$6
    local root="$ROOT_DIR/runs/$1" proof="$ROOT_DIR/runs/$1-posthoc/pair_comparison.json"
    local receipt="$ROOT_DIR/runs/$1-posthoc-job-$6/completed.json" role index budget leg_root leg_proof digest current prefix
    [[ $training =~ ^[0-9]+$ && $posthoc =~ ^[0-9]+$ && $training != "$posthoc" ]] || return 1
    for current in "$proof" "$receipt" "$root/pair.json" "$root/pair_completed.json"; do regular "$current" || return 1; done
    jq -e --arg id "$id" --arg source "$SOURCE_SHA" --arg protocol "$PROTOCOL" --arg training "$training" \
        --arg order "$order" --argjson candidate "$candidate" --argjson steps "$steps" \
        '.kind == "ampa_ep_overlap_budget_pair" and .pair_id == $id and .training_slurm_job_id == $training
         and .source_manifest_sha256 == $source and .experiment_protocol == $protocol
         and .partition == "debug" and .requested_nodes == 4 and .requested_time_limit == "00:21:00"
         and .candidate_budget == $candidate and .reference_budget == 20 and .order == $order
         and .capture == "off" and .case == "uccl_ordinary" and .execution_steps == $steps
         and .qualification_only == ($steps == 2) and (.legs | length) == 2' "$root/pair.json" > /dev/null || return 1
    jq -e --arg id "$id" --arg source "$SOURCE_SHA" --arg training "$training" --argjson steps "$steps" \
        '.success == true and .both_workers_and_health_srun_returned_zero == true and .completed_legs == 2
         and .pair_id == $id and .training_slurm_job_id == $training and .source_manifest_sha256 == $source
         and .execution_steps == $steps and .qualification_only == ($steps == 2)' "$root/pair_completed.json" > /dev/null || return 1
    cmp -s "$root/campaign-source.sha256" "$AMPA_OVERLAP_SOURCE_MANIFEST" || return 1
    jq -e --arg id "$id" --arg source "$SOURCE_SHA" --arg protocol "$PROTOCOL" --arg training "$training" \
        --arg posthoc "$posthoc" --arg analyzer "$ANALYZER_SHA" --arg order "$order" \
        --argjson candidate "$candidate" --argjson steps "$steps" \
        '.kind == "ampa_ep_overlap_budget_pair_comparison" and .valid_paired_budget_comparison == true
         and .matched_actual_rank_placement == true and .pair_id == $id and .candidate_budget == $candidate
         and .reference_budget == 20 and .order == $order and .execution_steps == $steps
         and .completed_updates_per_case == $steps and .qualification_only == ($steps == 2)
         and .training_slurm_job_id == $training and .execution.slurm_job_id == $posthoc
         and .execution.partition == "debug" and .source_manifest_sha256 == $source
         and .experiment_protocol == $protocol and .analyzer_sha256 == $analyzer
         and (.consumed_input_fingerprint_sha256 | type == "string" and test("^[0-9a-f]{64}$"))
         and (.legs | length) == 2 and ([.legs[].role] | sort) == ["candidate","reference"]
         and ([.legs[].index] | sort) == [0,1]' "$proof" > /dev/null || return 1
    jq -e --arg id "$id" --arg source "$SOURCE_SHA" --arg protocol "$PROTOCOL" --arg training "$training" --arg posthoc "$posthoc" \
        '.success == true and .pair_id == $id and .source_manifest_sha256 == $source
         and .experiment_protocol == $protocol and .training_slurm_job_id == $training
         and .posthoc_slurm_job_id == $posthoc and .partition == "debug" and .requested_nodes == 1' "$receipt" > /dev/null || return 1
    prefix=$(jq -er '.consumed_input_fingerprint_sha256' "$proof") || return 1
    jq -r '.legs[] | [.role,.index,.communication_sms,.orchestration_dir,.proof_path,.proof_sha256] | @tsv' \
        "$proof" > "$OUT/latest-leg-proofs.tsv" || return 1
    while IFS=$'\t' read -r role index budget leg_root leg_proof digest; do
        [[ $leg_root == "$root-"* && $leg_proof == "$leg_root-posthoc/execution_accounting.json"
            && $digest =~ ^[0-9a-f]{64}$ ]] || return 1
        regular "$leg_proof" || return 1
        read -r current _ < <(sha256sum "$leg_proof") || return 1
        [[ $current == "$digest" ]] || return 1
        if [[ $role == candidate ]]; then [[ $budget == "$candidate" ]] || return 1
        else [[ $budget == 20 ]] || return 1; fi
        jq -e --arg root "$leg_root" --arg source "$SOURCE_SHA" --arg protocol "$PROTOCOL" \
            --arg training "$training" --arg posthoc "$posthoc" --arg validator "$VALIDATOR_SHA" --arg prefix "$prefix" \
            --argjson budget "$budget" --argjson steps "$steps" \
            '.kind == "ampa_ep_overlap_unprofiled_budget_execution_accounting_validation"
             and .valid_execution_and_accounting == true and .valid_training_health == true
             and .valid_send_receive_audit == true and .valid_all_rank_startup_and_completion == true
             and .valid_actual_budget_runtime_guards == true and .validator_sha256 == $validator
             and .backend.valid_backend_identity_and_cleanup == true
             and .case == "uccl_ordinary" and .capture == "off" and .communication_backend == "uccl_ep"
             and .communication_sms == $budget and .completed_updates == $steps
             and .qualification_only == ($steps == 2) and .training_slurm_job_id == $training
             and .execution.slurm_job_id == $posthoc and .execution.partition == "debug"
             and .source_manifest_sha256 == $source and .orchestration_dir == $root and .run_dir == ($root + "-training")
             and .consumed_input_fingerprint_sha256 == $prefix and (.rank_placement | length) == 16
             and .execution_contract.experiment_protocol == $protocol and .execution_contract.ddp_delay_wgrad_compute == true
             and .execution_contract.capture == "off" and .execution_contract.execution_steps == $steps
             and .execution_contract.communication_sms == $budget
             and .execution_contract.warmup_steps == (if $steps == 2 then 1 else 5 end)
             and .execution_contract.measured_steps == (if $steps == 2 then 1 else 5 end)
             and .execution_contract.CUDA_DEVICE_MAX_CONNECTIONS == "32"
             and .execution_contract.overlap_flags.overlap_moe_expert_parallel_comm == true
             and .execution_contract.overlap_flags.delay_wgrad_compute == true
             and .execution_contract.overlap_flags.ep_overlap_early_attn_memory_release == true
             and .execution_contract.overlap_flags.overlap_dispatch_backward_with_experts_wgrad == false
             and .execution_contract.overlap_flags.moe_shared_expert_overlap == false
             and .execution_contract.overlap_flags.moe_shortcut_parallel == false
             and (.budget_runtime_guards | length) == 16
             and all(.budget_runtime_guards[]; .valid_actual_budget == true
                 and .requested_budget == $budget and .buffer_num_sms == $budget and .buffer_class_num_sms == $budget)' \
            "$leg_proof" > /dev/null || return 1
    done < "$OUT/latest-leg-proofs.tsv"
    return 0
}
# Qualify and pin the two-update pilot before submitting full work.
qualification_id=$(jq -er '.pair_id' "$QUALIFICATION")
qualification_candidate=$(jq -er '.candidate_budget' "$QUALIFICATION")
qualification_order=$(jq -er '.order' "$QUALIFICATION")
qualification_training=$(jq -er '.training_slurm_job_id | select(test("^[0-9]+$"))' "$QUALIFICATION")
qualification_posthoc=$(jq -er '.execution.slurm_job_id | select(test("^[0-9]+$"))' "$QUALIFICATION")
[[ $qualification_id =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$
    && $qualification_order =~ ^(candidate-first|reference-first)$
    && $QUALIFICATION == "$ROOT_DIR/runs/$qualification_id-posthoc/pair_comparison.json" ]] || exit 2
allowed "$qualification_candidate"
phase=qualification
wait_job "$qualification_training"
wait_job "$qualification_posthoc"
validate_pair "$qualification_id" "$qualification_candidate" "$qualification_order" 2 "$qualification_training" "$qualification_posthoc"
sha256sum "$PREFLIGHT" "$AMPA_BUDGET_PREFLIGHT_ROOT/PASS.txt" "$QUALIFICATION" \
    "$ROOT_DIR/runs/$qualification_id/pair.json" "$ROOT_DIR/runs/$qualification_id/pair_completed.json" \
    "$ROOT_DIR/runs/$qualification_id-posthoc-job-$qualification_posthoc/completed.json" > "$OUT/qualification-inputs.sha256"
while IFS=$'\t' read -r _ _ _ _ leg_proof _; do sha256sum "$leg_proof" >> "$OUT/qualification-inputs.sha256"; done < "$OUT/latest-leg-proofs.tsv"
cp "$PREFLIGHT" "$OUT/preflight.json"
cp "$QUALIFICATION" "$OUT/qualification_pair_comparison.json"
record "Qualified source=$SOURCE_SHA preflight=$preflight_job two-update pair=$qualification_id; qualification excluded from accepted.tsv"
export AMPA_BUDGET_CAMPAIGN_ROOT="$OUT"
export AMPA_BUDGET_PAIR_STEPS=10
initial=('12|reference-first|1' '32|candidate-first|1' '40|reference-first|1'
         '12|candidate-first|2' '32|reference-first|2' '40|candidate-first|2')
candidate=12
order=reference-first
replicate=1
accepted_count=0
verify_decision() {
    local path=$1 id=$2 proof=$3 expected snapshot snapshot_sha digest binding_path binding_sha
    regular "$path" || return 2
    expected=$(jq -Rn --arg id "$id" '[inputs | split("\t")[3]] + [$id] | sort' < "$OUT/accepted.tsv") || return 2
    jq -e --arg source "$SOURCE_SHA" --arg protocol "$PROTOCOL" --arg analyzer "$ANALYZER_SHA" --argjson expected "$expected" \
        '.valid_campaign_decision == true and .source_manifest_sha256 == $source and .experiment_protocol == $protocol
         and .analyzer_sha256 == $analyzer and (.accepted_pair_ids | sort) == $expected
         and .accepted_pair_count == ($expected | length)
         and (.input_files | type == "array" and length > 0)
         and all(.input_files[]; (.path | type == "string") and (.sha256 | type == "string" and test("^[0-9a-f]{64}$")))
         and ((.action == "complete" and .next_pair == null) or
              (.action == "run_pair" and (.next_pair | type == "object")
               and (.next_pair.candidate_budget | type == "number" and . >= 4 and . <= 64 and . % 2 == 0)
               and (.next_pair.order == "candidate-first" or .next_pair.order == "reference-first")
               and (.next_pair.replicate | type == "number" and . >= 1 and . <= 3 and . == floor)))' "$path" > /dev/null || return 2
    snapshot=$(jq -er '.accepted_ledger_snapshot.path' "$path") || return 2
    snapshot_sha=$(jq -er '.accepted_ledger_snapshot.sha256' "$path") || return 2
    [[ $snapshot == "$ROOT_DIR/runs/$id-posthoc/campaign_accepted_snapshot.tsv" && $snapshot_sha == "$ledger_sha" ]] || return 2
    regular "$snapshot" || return 2
    cmp -s "$snapshot" "$OUT/accepted.tsv" || return 2
    jq -r '.input_files[] | [.path,.sha256] | @tsv' "$path" > "$OUT/latest-decision-bindings.tsv" || return 2
    while IFS=$'\t' read -r binding_path binding_sha; do
        [[ $binding_path != "$OUT/accepted.tsv" ]] || return 2
        verify_binding "$binding_path" "$binding_sha" || return 2
    done < "$OUT/latest-decision-bindings.tsv"
    read -r digest _ < <(sha256sum "$proof") || return 2
    jq -e --arg proof "$proof" --arg digest "$digest" --arg preflight "$PREFLIGHT" --arg preflight_sha "$preflight_sha" \
        --arg analyzer "$ANALYZER" --arg analyzer_sha "$ANALYZER_SHA" --arg snapshot "$snapshot" --arg snapshot_sha "$snapshot_sha" \
        'any(.input_files[]; .path == $proof and .sha256 == $digest)
         and any(.input_files[]; .path == $preflight and .sha256 == $preflight_sha)
         and any(.input_files[]; .path == $analyzer and .sha256 == $analyzer_sha)
         and any(.input_files[]; .path == $snapshot and .sha256 == $snapshot_sha)' "$path" > /dev/null || return 2
}
while true; do
    allowed "$candidate"
    [[ $order =~ ^(candidate-first|reference-first)$ && $replicate =~ ^[123]$ ]] || exit 2
    [[ -z $(awk -F'\t' -v b="$candidate" -v o="$order" -v r="$replicate" '$1==b&&$2==o&&$3==r {print}' "$OUT/accepted.tsv") ]] || exit 2
    if (( accepted_count < 6 )); then
        [[ "$candidate|$order|$replicate" == "${initial[$accepted_count]}" ]] || exit 2
    fi
    stage_accepted=0
    for (( attempt=1; attempt<=MAX_ATTEMPTS; attempt++ )); do
        id="$AMPA_BUDGET_CAMPAIGN_ID-b$candidate-$order-r$replicate-attempt$attempt"
        pair_root="$ROOT_DIR/runs/$id"
        [[ ! -e $pair_root && ! -L $pair_root && ! -e $pair_root-posthoc && ! -L $pair_root-posthoc ]] || exit 2
        export AMPA_BUDGET_PAIR_ID="$id" AMPA_BUDGET_PAIR_CANDIDATE="$candidate" AMPA_BUDGET_PAIR_ORDER="$order"
        export AMPA_BUDGET_PAIR_ROOT="$pair_root"
        phase="b$candidate-$order-r$replicate-attempt$attempt-training"
        record "BEFORE training candidate=$candidate order=$order replicate=$replicate attempt=$attempt; four debug nodes/21minutes; independent C0 per leg"
        wait_capacity 4
        if ! active_job=$(sbatch --parsable "$ROOT_DIR/jobs/training/ep_overlap_budget_pair.sbatch"); then
            record "STOP training submission failed or identity unknown; inspect queue before any restart"
            exit 2
        fi
        [[ $active_job =~ ^[0-9]+$ ]] || exit 2
        training=$active_job
        posthoc=
        failure=
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$candidate" "$order" "$replicate" "$attempt" "$id" "$training" >> "$OUT/acquisitions.tsv"
        record "AFTER training submit job=$training pair=$id; strict acceptance pending"
        if wait_job "$training"; then
            if ! jq -e --arg id "$id" --arg source "$SOURCE_SHA" --arg training "$training" \
                '.success == true and .completed_legs == 2 and .both_workers_and_health_srun_returned_zero == true
                 and .pair_id == $id and .training_slurm_job_id == $training and .source_manifest_sha256 == $source
                 and .execution_steps == 10 and .qualification_only == false' "$pair_root/pair_completed.json" > /dev/null; then failure=training_completion_gate; fi
        else
            status=$?
            if (( status != 1 )); then exit "$status"; fi
            failure=training_scheduler
        fi
        if [[ -z $failure ]]; then
            phase="b$candidate-$order-r$replicate-attempt$attempt-posthoc"
            record "BEFORE posthoc pair=$id training=$training released; one debug node/up to90minutes"
            wait_capacity 1
            if ! active_job=$(sbatch --parsable "$ROOT_DIR/jobs/analysis/ep_overlap_budget_posthoc.sbatch"); then
                record "STOP posthoc submission failed or identity unknown; inspect queue before any restart"
                exit 2
            fi
            [[ $active_job =~ ^[0-9]+$ ]] || exit 2
            posthoc=$active_job
            printf '%s\t%s\t%s\n' "$id" "$training" "$posthoc" >> "$OUT/posthoc-jobs.tsv"
            record "AFTER posthoc submit job=$posthoc training=$training pair=$id; paired and decision gates pending"
            if wait_job "$posthoc"; then
                check_source
                if validate_pair "$id" "$candidate" "$order" 10 "$training" "$posthoc"; then
                    proof="$pair_root-posthoc/pair_comparison.json"
                    decision="$pair_root-posthoc/next_decision.json"
                    verify_decision "$decision" "$id" "$proof"
                    check_source
                    sha256sum "$proof" "$decision" "$pair_root-posthoc/campaign_accepted_snapshot.tsv" \
                        "$ROOT_DIR/runs/$id-posthoc-job-$posthoc/completed.json" >> "$OUT/accepted-proofs.sha256"
                    while IFS=$'\t' read -r _ _ _ _ leg_proof _; do sha256sum "$leg_proof" >> "$OUT/accepted-proofs.sha256"; done < "$OUT/latest-leg-proofs.tsv"
                    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$candidate" "$order" "$replicate" "$id" "$training" "$posthoc" "$proof" >> "$OUT/accepted.tsv"
                    read -r ledger_sha _ < <(sha256sum "$OUT/accepted.tsv")
                    accepted_count=$((accepted_count + 1))
                    cp "$decision" "$OUT/decisions/$id.json"
                    stage_accepted=1
                else failure=paired_acceptance_gate; fi
            else
                status=$?
                if (( status != 1 )); then exit "$status"; fi
                failure=posthoc_scheduler
            fi
        fi
        if (( stage_accepted == 1 )); then
            record "ACCEPTED pair=$id candidate=$candidate order=$order replicate=$replicate training=$training posthoc=$posthoc; total=$accepted_count"
            active_job=
            break
        fi
        printf 'reason=%s\npair=%s\ntraining=%s\nposthoc=%s\n' "$failure" "$id" "$training" "$posthoc" > "$OUT/failures/$id.txt"
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$id" "$attempt" "$failure" "$training" "$posthoc" "$pair_root" >> "$OUT/failed-attempts.tsv"
        record "FAILED pair=$id reason=$failure; preserved original legs and analysis; retry only with a fresh whole-pair ID"
        active_job=
        if (( attempt == MAX_ATTEMPTS )); then
            printf 'candidate=%s\norder=%s\nreplicate=%s\nattempts=%s\nlast_pair=%s\nnext_action=root diagnosis and fresh qualified continuation\n' \
                "$candidate" "$order" "$replicate" "$MAX_ATTEMPTS" "$id" > "$OUT/RETRIES_EXHAUSTED.txt"
            exit 1
        fi
    done
    (( stage_accepted == 1 )) || exit 1
    action=$(jq -er '.action' "$decision")
    if [[ $action == complete ]]; then
        (( accepted_count >= 6 )) || exit 2
        phase=bounded_acquisitions_complete
        complete=1
        printf 'bounded_paired_sweep_complete=true\naccepted_pairs=%s\nfull_cases=%s\noptimizer_updates=%s\nfinal_report_accepted=false\n' \
            "$accepted_count" "$((accepted_count * 2))" "$((accepted_count * 20))" > "$OUT/ACQUISITIONS_COMPLETE.txt"
        record "COMPLETE bounded paired sweep: $accepted_count accepted allocations; analyzer stopping/boundary result retained; final report review remains separate"
        break
    fi
    candidate=$(jq -er '.next_pair.candidate_budget' "$decision")
    order=$(jq -er '.next_pair.order' "$decision")
    replicate=$(jq -er '.next_pair.replicate' "$decision")
done
