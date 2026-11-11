#!/usr/bin/env bash
# Two independent C0 starts on the same allocation. Root alone submits.
#SBATCH --job-name=ampa-overlap-budget-pair
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
[[ $# == 0 && ${SLURM_JOB_PARTITION:-} == debug && ${SLURM_JOB_NUM_NODES:-} == 4
    && ${SLURM_JOB_ID:-} =~ ^[0-9]+$ ]] || exit 2
readonly ROOT_DIR=${SLURM_SUBMIT_DIR:?}
readonly PROTOCOL=ep_overlap_ordinary_off_budget_5_plus_5_v1
cd "$ROOT_DIR"
if [[ -n ${AMPA_BUDGET_SOURCE_MANIFEST:-} ]]; then
    [[ -z ${AMPA_OVERLAP_SOURCE_MANIFEST:-} || $AMPA_OVERLAP_SOURCE_MANIFEST == "$AMPA_BUDGET_SOURCE_MANIFEST" ]] || exit 2
    export AMPA_OVERLAP_SOURCE_MANIFEST=$AMPA_BUDGET_SOURCE_MANIFEST
fi
if [[ -n ${AMPA_BUDGET_SOURCE_SHA256:-} ]]; then
    [[ -z ${AMPA_OVERLAP_SOURCE_SHA256:-} || $AMPA_OVERLAP_SOURCE_SHA256 == "$AMPA_BUDGET_SOURCE_SHA256" ]] || exit 2
    export AMPA_OVERLAP_SOURCE_SHA256=$AMPA_BUDGET_SOURCE_SHA256
fi
[[ ${AMPA_BUDGET_PAIR_ID:?} =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$
    && ${AMPA_BUDGET_PAIR_CANDIDATE:?} =~ ^[1-9][0-9]*$
    && ${AMPA_BUDGET_PAIR_ORDER:?} =~ ^(candidate-first|reference-first)$ ]] || exit 2
# The static thread ceiling is 64. The bound QA proof sets the smaller
# actual shared-memory limit, and its allowed_budgets gate remains mandatory.
(( AMPA_BUDGET_PAIR_CANDIDATE >= 4 && AMPA_BUDGET_PAIR_CANDIDATE <= 64 && AMPA_BUDGET_PAIR_CANDIDATE != 20
   && AMPA_BUDGET_PAIR_CANDIDATE % 2 == 0 )) || exit 2
export AMPA_BUDGET_PAIR_STEPS=${AMPA_BUDGET_PAIR_STEPS:-10}
[[ $AMPA_BUDGET_PAIR_STEPS =~ ^(2|10)$ ]] || exit 2
default_reserve=480
if [[ $AMPA_BUDGET_PAIR_STEPS == 2 ]]; then default_reserve=360; fi
export AMPA_BUDGET_CANDIDATE_RESERVE_SECONDS=${AMPA_BUDGET_CANDIDATE_RESERVE_SECONDS:-$default_reserve}
export AMPA_BUDGET_REFERENCE_RESERVE_SECONDS=${AMPA_BUDGET_REFERENCE_RESERVE_SECONDS:-$default_reserve}
for value in "$AMPA_BUDGET_CANDIDATE_RESERVE_SECONDS" "$AMPA_BUDGET_REFERENCE_RESERVE_SECONDS"; do
    [[ $value =~ ^[1-9][0-9]*$ ]] && (( value >= 60 && value <= 1200 )) || exit 2
done
[[ ${AMPA_OVERLAP_SOURCE_MANIFEST:?} == "$ROOT_DIR"/runs/*
    && -f $AMPA_OVERLAP_SOURCE_MANIFEST && ! -L $AMPA_OVERLAP_SOURCE_MANIFEST
    && ${AMPA_OVERLAP_SOURCE_SHA256:?} =~ ^[0-9a-f]{64}$
    && ${AMPA_BUDGET_PREFLIGHT_ROOT:?} == "$ROOT_DIR"/runs/*
    && -f $AMPA_BUDGET_PREFLIGHT_ROOT/preflight.json && ! -L $AMPA_BUDGET_PREFLIGHT_ROOT/preflight.json
    && -f $AMPA_BUDGET_PREFLIGHT_ROOT/PASS.txt && ! -L $AMPA_BUDGET_PREFLIGHT_ROOT/PASS.txt ]] || exit 2
read -r source_digest _ < <(sha256sum "$AMPA_OVERLAP_SOURCE_MANIFEST")
[[ $source_digest == "$AMPA_OVERLAP_SOURCE_SHA256" ]] || exit 2
sha256sum --check --strict "$AMPA_OVERLAP_SOURCE_MANIFEST"
read -r preflight_digest _ < <(sha256sum "$AMPA_BUDGET_PREFLIGHT_ROOT/preflight.json")
grep --fixed-strings --line-regexp --quiet -- "preflight_sha256=$preflight_digest" "$AMPA_BUDGET_PREFLIGHT_ROOT/PASS.txt"
grep --fixed-strings --line-regexp --quiet -- "source_sha256=$source_digest" "$AMPA_BUDGET_PREFLIGHT_ROOT/PASS.txt"
jq -e --arg source "$source_digest" --arg protocol "$PROTOCOL" --argjson candidate "$AMPA_BUDGET_PAIR_CANDIDATE" \
    '.valid_preflight == true and .training_executed == false and .partition == "debug"
     and .source_manifest_sha256 == $source and .experiment_protocol == $protocol
     and (.job | type == "string" and test("^[0-9]+$"))
     and (.allowed_budgets | index($candidate)) != null and (.allowed_budgets | index(20)) != null
     and (.focused_tests | type == "array" and length > 0)
     and all(.focused_tests[]; .tests | type == "number" and . > 0 and . == floor)' \
    "$AMPA_BUDGET_PREFLIGHT_ROOT/preflight.json" > /dev/null
preflight_job=$(jq -er '.job' "$AMPA_BUDGET_PREFLIGHT_ROOT/preflight.json")
grep --fixed-strings --line-regexp --quiet -- "job=$preflight_job" "$AMPA_BUDGET_PREFLIGHT_ROOT/PASS.txt"
export AMPA_BUDGET_PAIR_ROOT="$ROOT_DIR/runs/$AMPA_BUDGET_PAIR_ID"
export AMPA_BUDGET_SESSION_DIR=${AMPA_BUDGET_SESSION_DIR:-$ROOT_DIR/progress/sessions/2026-11-10-qwen-ep-overlap}
export AMPA_BUDGET_STATUS_FILE=${AMPA_BUDGET_STATUS_FILE:-$ROOT_DIR/ai/ep-overlap-budget-launch/iteration-01/STATUS.md}
[[ $AMPA_BUDGET_SESSION_DIR == "$ROOT_DIR"/progress/sessions/*
    && -f $AMPA_BUDGET_SESSION_DIR/timeline.md && -f $AMPA_BUDGET_SESSION_DIR/session_state.md
    && $AMPA_BUDGET_STATUS_FILE == "$ROOT_DIR"/ai/*/iteration-*/STATUS.md && -f $AMPA_BUDGET_STATUS_FILE
    && ! -e $AMPA_BUDGET_PAIR_ROOT && ! -L $AMPA_BUDGET_PAIR_ROOT ]] || exit 2
mkdir "$AMPA_BUDGET_PAIR_ROOT"
cp "$AMPA_OVERLAP_SOURCE_MANIFEST" "$AMPA_BUDGET_PAIR_ROOT/campaign-source.sha256"
cp "$AMPA_BUDGET_PREFLIGHT_ROOT/preflight.json" "$AMPA_BUDGET_PAIR_ROOT/preflight.json"
cp "$AMPA_BUDGET_PREFLIGHT_ROOT/PASS.txt" "$AMPA_BUDGET_PAIR_ROOT/preflight-PASS.txt"
jq -n --arg id "$AMPA_BUDGET_PAIR_ID" --arg root "$ROOT_DIR" --arg job "$SLURM_JOB_ID" \
    --arg order "$AMPA_BUDGET_PAIR_ORDER" --arg source "$source_digest" --arg protocol "$PROTOCOL" \
    --arg preflight "$AMPA_BUDGET_PREFLIGHT_ROOT" --arg preflight_sha "$preflight_digest" \
    --argjson candidate "$AMPA_BUDGET_PAIR_CANDIDATE" --argjson steps "$AMPA_BUDGET_PAIR_STEPS" \
    --argjson candidate_reserve "$AMPA_BUDGET_CANDIDATE_RESERVE_SECONDS" \
    --argjson reference_reserve "$AMPA_BUDGET_REFERENCE_RESERVE_SECONDS" \
    '{schema_version:1,kind:"ampa_ep_overlap_budget_pair",experiment_protocol:$protocol,
      pair_id:$id,training_slurm_job_id:$job,partition:"debug",requested_nodes:4,
      requested_time_limit:"00:21:00",candidate_budget:$candidate,reference_budget:20,order:$order,
      execution_steps:$steps,qualification_only:($steps==2),capture:"off",case:"uccl_ordinary",
      source_manifest_sha256:$source,preflight_root:$preflight,preflight_sha256:$preflight_sha,
      scheduling_reserve_seconds:{candidate:$candidate_reserve,reference:$reference_reserve},
      reserve_claim:"Declared scheduling estimates, not guaranteed case durations or performance evidence.",
      allocation_scope:"one Slurm allocation shared by both independent C0 cases",
      per_case_slurm_elapsed_available:false,
      legs:((if $order=="candidate-first" then ["candidate","reference"] else ["reference","candidate"] end)
        | to_entries | map(.key as $index | .value as $role |
          (if $role=="candidate" then $candidate else 20 end) as $budget |
          ($id+"-"+$role+"-b"+($budget|tostring)) as $caseid |
          {index:$index,role:$role,budget:$budget,experiment_id:$caseid,
           orchestration_dir:($root+"/runs/"+$caseid),run_dir:($root+"/runs/"+$caseid+"-training")}))}' \
    > "$AMPA_BUDGET_PAIR_ROOT/pair.json"
jq -r '.legs[] | [.index,.role,.budget,.experiment_id,.orchestration_dir,.run_dir] | @tsv' \
    "$AMPA_BUDGET_PAIR_ROOT/pair.json" > "$AMPA_BUDGET_PAIR_ROOT/cases.tsv"
record() {
    local note="$(date -u '+%Y-%m-%d %H:%M:%S UTC') pair_job=$SLURM_JOB_ID pair=$AMPA_BUDGET_PAIR_ID: $*"
    printf '%s\n' "$note" | tee -a "$AMPA_BUDGET_PAIR_ROOT/timeline.md"
    printf '%s\n' "$note" > "$AMPA_BUDGET_PAIR_ROOT/CURRENT.md"
    printf '\n- %s\n' "$note" >> "$AMPA_BUDGET_SESSION_DIR/timeline.md"
    printf '\nCompute milestone: %s\n' "$note" >> "$AMPA_BUDGET_SESSION_DIR/session_state.md"
    printf '\nCompute milestone: %s\n' "$note" >> "$AMPA_BUDGET_STATUS_FILE"
}
time_left_seconds() {
    local value=$1 days=0 hours=0 minutes seconds
    if [[ $value == *-* ]]; then days=${value%%-*}; value=${value#*-}; fi
    if [[ $value =~ ^([0-9]+):([0-9][0-9]):([0-9][0-9])$ ]]; then
        hours=${BASH_REMATCH[1]}; minutes=${BASH_REMATCH[2]}; seconds=${BASH_REMATCH[3]}
    elif [[ $value =~ ^([0-9]+):([0-9][0-9])$ ]]; then
        minutes=${BASH_REMATCH[1]}; seconds=${BASH_REMATCH[2]}
    else return 2; fi
    printf '%s\n' "$((10#$days*86400 + 10#$hours*3600 + 10#$minutes*60 + 10#$seconds))"
}
check_queue() {
    local needed=${1:-0} queue requested remaining
    queue=$(squeue --array -u "$USER" --noheader --format='%i %P %T %D %L %R')
    printf '%s\n' "$queue" > "$AMPA_BUDGET_PAIR_ROOT/latest-queue.txt"
    requested=$(awk '{sum += $4} END {print sum+0}' <<< "$queue")
    (( requested <= 4 )) || { record "STOP global requested nodes=$requested exceeds4"; return 2; }
    remaining=$(awk -v job="$SLURM_JOB_ID" '$1==job {print $5}' <<< "$queue")
    remaining=$(time_left_seconds "$remaining")
    if (( remaining < needed )); then
        record "INCOMPLETE resource/time reserve: remaining=$remaining seconds need=$needed; preserve partial pair; not a performance decline"
        return 3
    fi
}
complete=0
phase=preparation
on_exit() {
    local status=$?
    trap - EXIT
    if (( status == 0 && complete != 1 )); then status=1; fi
    record "AFTER pair phase=$phase exit=$status complete=$complete; preserve every case; partial pairs are not accepted"
    printf 'exit_code=%s\ncomplete=%s\nphase=%s\n' "$status" "$complete" "$phase" > "$AMPA_BUDGET_PAIR_ROOT/exit.txt"
    exit "$status"
}
trap on_exit EXIT
export AMPA_UCCL_BUILD_ROOT="$ROOT_DIR/runs/uccl-ep-owned-build-20261101T1713Z"
export AMPA_UCCL_VERIFIED_FIRST_ROOT="$ROOT_DIR/runs/uccl-debug5x5-20261104T1415Z-grouped-profile"
export AMPA_C0="$ROOT_DIR/checkpoint/megatron/qwen3-30b-a3b-base-c0"
export AMPA_HF_REVISION=1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9
export AMPA_DATA_MANIFEST="$ROOT_DIR/dataset/manifests/ep16-cache-20261011T1757Z-cp1-ep16.json"
export AMPA_DATA_PATH="0.55 $ROOT_DIR/dataset/indexed/qwen3-30b-a3b-base-realistic-mix-v2/web/qwen3_30b_a3b_base_realistic_mix_web 0.15 $ROOT_DIR/dataset/indexed/qwen3-30b-a3b-base-realistic-mix-v2/multilingual/qwen3_30b_a3b_base_realistic_mix_multilingual 0.15 $ROOT_DIR/dataset/indexed/qwen3-30b-a3b-base-realistic-mix-v2/code/qwen3_30b_a3b_base_realistic_mix_code 0.15 $ROOT_DIR/dataset/indexed/qwen3-30b-a3b-base-realistic-mix-v2/math/qwen3_30b_a3b_base_realistic_mix_math"
export EFA_HOME="$AMPA_UCCL_BUILD_ROOT/not-efa" UCCL_EP_TRANSPORT=cxi CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_DEVICE_MAX_CONNECTIONS=32 FI_MR_CACHE_MONITOR=disabled PYTHONDONTWRITEBYTECODE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$AMPA_UCCL_BUILD_ROOT/site:$AMPA_UCCL_BUILD_ROOT/deps:$AMPA_UCCL_BUILD_ROOT/uccl/ep/deep_ep_wrapper:$ROOT_DIR:$ROOT_DIR/profiling:$ROOT_DIR/analysis${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="/opt/amazon/efa/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
[[ -n $MASTER_ADDR ]] || exit 2
export MASTER_ADDR
record "BEFORE independent paired C0 acquisition candidate=$AMPA_BUDGET_PAIR_CANDIDATE order=$AMPA_BUDGET_PAIR_ORDER steps=$AMPA_BUDGET_PAIR_STEPS nodes=$SLURM_JOB_NODELIST container=setup/gh200-cuda130-runtime.toml source=$source_digest"
check_queue "$((AMPA_BUDGET_CANDIDATE_RESERVE_SECONDS + AMPA_BUDGET_REFERENCE_RESERVE_SECONDS + 60))"
check_case_completion() {
    local case_root=$1 run=$2 id=$3 role=$4 budget=$5
    [[ -f $case_root/acquisition_completed.json && ! -L $case_root/acquisition_completed.json
        && -f $case_root/case_host_wall.json && ! -L $case_root/case_host_wall.json
        && -f $case_root/source-after.txt && ! -L $case_root/source-after.txt ]] || return 2
    jq -e --arg job "$SLURM_JOB_ID" --arg run "$run" --arg source "$source_digest" \
        --arg pair "$AMPA_BUDGET_PAIR_ID" --arg role "$role" \
        --argjson budget "$budget" --argjson steps "$AMPA_BUDGET_PAIR_STEPS" \
        '.success == true and .workers_srun_returned_zero == true and .training_health_and_timer_summary_complete == true
         and .job == $job and .run_dir == $run and .case == "uccl_ordinary" and .capture == "off"
         and .communication_sms == $budget and .completed_updates == $steps
         and .qualification_only == ($steps==2) and .campaign_source_sha256 == $source
         and .pair_id == $pair and .pair_leg == $role' "$case_root/acquisition_completed.json" > /dev/null || return 2
    jq -e --arg job "$SLURM_JOB_ID" --arg id "$id" --arg source "$source_digest" \
        --arg pair "$AMPA_BUDGET_PAIR_ID" --arg role "$role" --argjson budget "$budget" \
        '.workers_srun_returned_zero == true and .health_timing_srun_returned_zero == true
         and .training_slurm_job_id == $job and .experiment_id == $id
         and .source_manifest_sha256 == $source and .pair_id == $pair and .pair_leg == $role
         and .communication_sms == $budget and .allocation_scope == "shared_pair"
         and .per_case_slurm_elapsed_available == false and (.host_elapsed_seconds | type=="number" and .>0)' \
        "$case_root/case_host_wall.json" > /dev/null || return 2
}
completed_legs=0
completed_candidate_id=
completed_reference_id=
while IFS=$'\t' read -r leg_index leg budget id case_root run; do
    phase="leg-$leg_index-$leg"
    export AMPA_BUDGET_LEG_INDEX="$leg_index" AMPA_BUDGET_LEG="$leg"
    export AMPA_OVERLAP_ID="$id" AMPA_OVERLAP_ROOT="$case_root" AMPA_OVERLAP_RUN="$run"
    export AMPA_OVERLAP_CASE=uccl_ordinary AMPA_OVERLAP_CAPTURE=off
    export AMPA_OVERLAP_STEPS="$AMPA_BUDGET_PAIR_STEPS" AMPA_OVERLAP_BUDGET="$budget"
    export AMPA_UCCL_REUSE_CONTEXT="$case_root/input_reuse_context.json"
    export AMPA_REVIEWED_INPUTS_PROOF="$case_root/reviewed_inputs.json"
    unset AMPA_OVERLAP_WORKERS_SUCCEEDED
    [[ ! -e $case_root && ! -L $case_root && ! -e $run && ! -L $run ]] || exit 2
    mkdir "$case_root" "$run"
    mkdir -p "$case_root"/{source,logs} "$run"/{config,raw,metrics,analysis,plots,traces,nsight,logs/wandb}
    cp "$AMPA_OVERLAP_SOURCE_MANIFEST" "$case_root/campaign-source.sha256"
    while IFS= read -r line; do
        source_path=${line:66}
        [[ ${line:0:64} =~ ^[0-9a-f]{64}$ && ${line:64:2} == '  '
            && $source_path == "$ROOT_DIR"/* && -f $source_path && ! -L $source_path ]] || exit 2
        relative=${source_path#"$ROOT_DIR/"}
        [[ $relative != ../* && $relative != */../* ]] || exit 2
        mkdir -p "$case_root/source/$(dirname "$relative")"
        cp "$source_path" "$case_root/source/$relative"
    done < "$AMPA_OVERLAP_SOURCE_MANIFEST"
    printf '%s\n' "$SLURM_JOB_ID" > "$case_root/job-id"
    printf 'uccl_ordinary\toff\t%s\n' "$run" > "$case_root/cases.tsv"
    reserve=$AMPA_BUDGET_CANDIDATE_RESERVE_SECONDS
    if [[ $leg == reference ]]; then reserve=$AMPA_BUDGET_REFERENCE_RESERVE_SECONDS; fi
    sha256sum --check --strict "$AMPA_OVERLAP_SOURCE_MANIFEST" > "$case_root/source-before.txt"
    check_queue "$((reserve + 30))"
    start_utc=$(date -u '+%Y-%m-%dT%H:%M:%S.%NZ')
    start_ns=$(date -u '+%s%N')
    read -r start_uptime _ < /proc/uptime
    jq -n --arg pair "$AMPA_BUDGET_PAIR_ID" --arg id "$id" --arg leg "$leg" --arg job "$SLURM_JOB_ID" \
        --arg host "$(hostname)" --arg utc "$start_utc" --arg ns "$start_ns" --arg uptime "$start_uptime" \
        --arg source "$source_digest" --argjson budget "$budget" --argjson index "$leg_index" \
        '{schema_version:1,kind:"ampa_budget_case_host_wall_start",pair_id:$pair,experiment_id:$id,
          pair_leg:$leg,leg_index:$index,communication_sms:$budget,training_slurm_job_id:$job,
          hostname:$host,start_utc:$utc,start_epoch_ns:$ns,start_host_uptime_seconds:$uptime,
          monotonic_clock_source:"/proc/uptime on the same allocated batch host",
          source_manifest_sha256:$source,allocation_scope:"shared_pair",per_case_slurm_elapsed_available:false,
          interval:"training srun launch through health/timing srun return; excludes shared setup and source archival"}' \
        > "$case_root/case_host_wall_start.json"
    export MASTER_PORT="$((21000 + (SLURM_JOB_ID % 10000)*2 + leg_index))"
    record "BEFORE leg=$leg_index role=$leg budget=$budget case=$id fresh C0; declared reserve=$reserve seconds"
    srun --input=/dev/null --mpi=pmix --cpu-bind=none --container-remap-root --environment="$ROOT_DIR/setup/gh200-cuda130-runtime.toml" \
        --nodes=4 --ntasks=4 --ntasks-per-node=1 --gpus-per-task=4 --kill-on-bad-exit=1 \
        --output="$case_root/logs/node-%t.out" --error="$case_root/logs/node-%t.err" \
        bash "$ROOT_DIR/jobs/training/ep_overlap_budget_worker.sh" --node
    export AMPA_OVERLAP_WORKERS_SUCCEEDED=1
    record "AFTER all16 workers returned0 for leg=$leg; BEFORE small health/all-step timing summaries"
    check_queue
    srun --input=/dev/null --mpi=pmix --cpu-bind=none --container-remap-root --environment="$ROOT_DIR/setup/gh200-cuda130-runtime.toml" \
        --nodes=1 --ntasks=1 --gpus-per-task=1 --kill-on-bad-exit=1 \
        --output="$case_root/logs/health-timing.out" --error="$case_root/logs/health-timing.err" \
        bash "$ROOT_DIR/jobs/training/ep_overlap_budget_worker.sh" --validate
    end_utc=$(date -u '+%Y-%m-%dT%H:%M:%S.%NZ')
    end_ns=$(date -u '+%s%N')
    read -r end_uptime _ < /proc/uptime
    elapsed_ns=$((end_ns - start_ns))
    (( elapsed_ns > 0 )) || exit 2
    jq --arg utc "$end_utc" --arg ns "$end_ns" --arg uptime "$end_uptime" --arg elapsed "$elapsed_ns" \
        '. + {kind:"ampa_budget_case_host_wall",end_utc:$utc,end_epoch_ns:$ns,end_host_uptime_seconds:$uptime,
          elapsed_nanoseconds:$elapsed,host_elapsed_seconds:(($elapsed|tonumber)/1e9),
          monotonic_host_elapsed_seconds:(($uptime|tonumber)-(.start_host_uptime_seconds|tonumber)),
          workers_srun_returned_zero:true,health_timing_srun_returned_zero:true}' \
        "$case_root/case_host_wall_start.json" > "$case_root/case_host_wall.json"
    sha256sum --check --strict "$AMPA_OVERLAP_SOURCE_MANIFEST" > "$case_root/source-after.txt"
    check_case_completion "$case_root" "$run" "$id" "$leg" "$budget"
    if [[ $leg == candidate ]]; then
        [[ -z $completed_candidate_id && $id == "$AMPA_BUDGET_PAIR_ID-candidate-b$AMPA_BUDGET_PAIR_CANDIDATE"
            && $budget == "$AMPA_BUDGET_PAIR_CANDIDATE" ]] || exit 2
        completed_candidate_id=$id
    elif [[ $leg == reference ]]; then
        [[ -z $completed_reference_id && $id == "$AMPA_BUDGET_PAIR_ID-reference-b20" && $budget == 20 ]] || exit 2
        completed_reference_id=$id
    else exit 2; fi
    completed_legs=$((completed_legs + 1))
    record "AFTER leg=$leg_index role=$leg budget=$budget completed; case host wall recorded; strict paired acceptance pending posthoc"
done < "$AMPA_BUDGET_PAIR_ROOT/cases.tsv"
[[ $completed_legs == 2 && $completed_candidate_id == "$AMPA_BUDGET_PAIR_ID-candidate-b$AMPA_BUDGET_PAIR_CANDIDATE"
    && $completed_reference_id == "$AMPA_BUDGET_PAIR_ID-reference-b20" ]] || exit 2
for role in candidate reference; do
    budget=$AMPA_BUDGET_PAIR_CANDIDATE
    if [[ $role == reference ]]; then budget=20; fi
    id="$AMPA_BUDGET_PAIR_ID-$role-b$budget"
    check_case_completion "$ROOT_DIR/runs/$id" "$ROOT_DIR/runs/$id-training" "$id" "$role" "$budget"
done
jq --argjson completed_legs "$completed_legs" --arg candidate_id "$completed_candidate_id" --arg reference_id "$completed_reference_id" \
    '. + {success:true,completed_legs:$completed_legs,completed_case_ids:[$candidate_id,$reference_id],both_workers_and_health_srun_returned_zero:true,
          comparison_accepted:false,routing_acceptance:"pending_posthoc_analysis",
          actual_same_gpu_placement:"pending_posthoc_analysis"}' \
    "$AMPA_BUDGET_PAIR_ROOT/pair.json" > "$AMPA_BUDGET_PAIR_ROOT/pair_completed.json"
complete=1
phase=acquisition_complete
record "AFTER both independent cases acquired; release4nodes before1node posthoc; shared allocation elapsed is not a per-case time"
