#!/usr/bin/env bash
# Analyze both legs only after their shared four-node allocation ends.
#SBATCH --job-name=ampa-overlap-budget-posthoc
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
readonly TRAIN_PY=/opt/venvs/megatron-bridge/bin/python
readonly PROTOCOL=ep_overlap_ordinary_off_budget_5_plus_5_v1
readonly SELF="$ROOT_DIR/jobs/analysis/ep_overlap_budget_posthoc.sbatch"
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
    && ${AMPA_OVERLAP_SOURCE_MANIFEST:?} == "$ROOT_DIR"/runs/*
    && -f $AMPA_OVERLAP_SOURCE_MANIFEST && ! -L $AMPA_OVERLAP_SOURCE_MANIFEST
    && ${AMPA_OVERLAP_SOURCE_SHA256:?} =~ ^[0-9a-f]{64}$ ]] || exit 2
export AMPA_BUDGET_PAIR_ROOT="$ROOT_DIR/runs/$AMPA_BUDGET_PAIR_ID"
[[ -d $AMPA_BUDGET_PAIR_ROOT && ! -L $AMPA_BUDGET_PAIR_ROOT
    && -f $AMPA_BUDGET_PAIR_ROOT/pair.json && ! -L $AMPA_BUDGET_PAIR_ROOT/pair.json
    && -f $AMPA_BUDGET_PAIR_ROOT/pair_completed.json && ! -L $AMPA_BUDGET_PAIR_ROOT/pair_completed.json ]] || exit 2
read -r source_digest _ < <(sha256sum "$AMPA_OVERLAP_SOURCE_MANIFEST")
[[ $source_digest == "$AMPA_OVERLAP_SOURCE_SHA256" ]] || exit 2
cmp -s "$AMPA_OVERLAP_SOURCE_MANIFEST" "$AMPA_BUDGET_PAIR_ROOT/campaign-source.sha256"
sha256sum --check --strict "$AMPA_OVERLAP_SOURCE_MANIFEST"
training_job=$(jq -er '.training_slurm_job_id | select(type=="string" and test("^[0-9]+$"))' "$AMPA_BUDGET_PAIR_ROOT/pair.json")
[[ $training_job != "$SLURM_JOB_ID" ]] || exit 2
jq -e --arg id "$AMPA_BUDGET_PAIR_ID" --arg source "$source_digest" --arg protocol "$PROTOCOL" --arg training "$training_job" \
    '.success == true and .completed_legs == 2 and .both_workers_and_health_srun_returned_zero == true
     and .pair_id == $id and .training_slurm_job_id == $training and .experiment_protocol == $protocol
     and .partition == "debug" and .capture == "off" and .case == "uccl_ordinary"
     and .source_manifest_sha256 == $source and .reference_budget == 20 and (.legs | length)==2
     and (.execution_steps==2 or .execution_steps==10) and .qualification_only==(.execution_steps==2)' \
    "$AMPA_BUDGET_PAIR_ROOT/pair_completed.json" > /dev/null
export CUDA_DEVICE_MAX_CONNECTIONS=32 FI_MR_CACHE_MONITOR=disabled PYTHONDONTWRITEBYTECODE=1
export AMPA_UCCL_BUILD_ROOT="$ROOT_DIR/runs/uccl-ep-owned-build-20261101T1713Z"
export EFA_HOME="$AMPA_UCCL_BUILD_ROOT/not-efa" UCCL_EP_TRANSPORT=cxi
export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/profiling:$ROOT_DIR/analysis:$ROOT_DIR/Megatron-LM:$ROOT_DIR/Megatron-Bridge/src:$AMPA_UCCL_BUILD_ROOT/site:$AMPA_UCCL_BUILD_ROOT/deps:$AMPA_UCCL_BUILD_ROOT/uccl/ep/deep_ep_wrapper${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="/opt/amazon/efa/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

if [[ ${1:-} == --inside-container ]]; then
    [[ $# == 1 && ${AMPA_BUDGET_POSTHOC_JOB_ROOT:?} == "$ROOT_DIR/runs/$AMPA_BUDGET_PAIR_ID-posthoc-job-$SLURM_JOB_ID" ]] || exit 2
    leg_roots_text=$(jq -er '.legs[].orchestration_dir' "$AMPA_BUDGET_PAIR_ROOT/pair.json")
    mapfile -t leg_roots <<< "$leg_roots_text"
    [[ ${#leg_roots[@]} == 2 && ${leg_roots[0]} != "${leg_roots[1]}" ]] || exit 2
    for leg_root in "${leg_roots[@]}"; do
        [[ $leg_root == "$ROOT_DIR/runs/$AMPA_BUDGET_PAIR_ID-"* && -d $leg_root && ! -L $leg_root
            && $(dirname "$leg_root") == "$ROOT_DIR/runs" ]] || exit 2
        export AMPA_OVERLAP_ID=${leg_root##*/} AMPA_OVERLAP_ROOT="$leg_root" AMPA_OVERLAP_RUN="$leg_root-training"
        export AMPA_OVERLAP_CASE=uccl_ordinary AMPA_OVERLAP_CAPTURE=off
        AMPA_OVERLAP_STEPS=$(jq -er '.execution_steps' "$AMPA_BUDGET_PAIR_ROOT/pair.json")
        AMPA_OVERLAP_BUDGET=$(jq -er --arg leg "$leg_root" '.legs[] | select(.orchestration_dir==$leg) | .budget' "$AMPA_BUDGET_PAIR_ROOT/pair.json")
        export AMPA_OVERLAP_STEPS AMPA_OVERLAP_BUDGET
        "$TRAIN_PY" "$ROOT_DIR/tools/ep_overlap_budget_validate.py" \
            --orchestration-dir "$leg_root" --output-dir "$leg_root-posthoc"
        jq -e --arg source "$source_digest" --arg training "$training_job" \
            '.valid_execution_and_accounting == true and .valid_training_health == true
             and .valid_send_receive_audit == true and .valid_all_rank_startup_and_completion == true
             and .capture == "off" and .case == "uccl_ordinary" and .source_manifest_sha256 == $source
             and .training_slurm_job_id == $training' "$leg_root-posthoc/execution_accounting.json" > /dev/null
    done
    "$TRAIN_PY" "$ROOT_DIR/tools/ep_overlap_budget_validate.py" \
        --pair-dir "$AMPA_BUDGET_PAIR_ROOT" --output-dir "$AMPA_BUDGET_PAIR_ROOT-posthoc"
    candidate=$(jq -er '.candidate_budget' "$AMPA_BUDGET_PAIR_ROOT/pair.json")
    qualification=$(jq -r '.execution_steps==2' "$AMPA_BUDGET_PAIR_ROOT/pair.json")
    jq -e --arg source "$source_digest" --argjson candidate "$candidate" --argjson qualification "$qualification" \
        '.valid_paired_budget_comparison == true and .reference_budget == 20
         and .candidate_budget == $candidate and .qualification_only == $qualification
         and .matched_actual_rank_placement == true and .source_manifest_sha256 == $source' \
        "$AMPA_BUDGET_PAIR_ROOT-posthoc/pair_comparison.json" > /dev/null
    if [[ -n ${AMPA_BUDGET_CAMPAIGN_ROOT:-} ]]; then
        [[ $qualification == false && $AMPA_BUDGET_CAMPAIGN_ROOT == "$ROOT_DIR"/runs/*
            && -d $AMPA_BUDGET_CAMPAIGN_ROOT && ! -L $AMPA_BUDGET_CAMPAIGN_ROOT
            && $(dirname "$AMPA_BUDGET_CAMPAIGN_ROOT") == "$ROOT_DIR/runs"
            && ${AMPA_BUDGET_PREFLIGHT_ROOT:?} == "$ROOT_DIR"/runs/*
            && -f $AMPA_BUDGET_PREFLIGHT_ROOT/preflight.json && ! -L $AMPA_BUDGET_PREFLIGHT_ROOT/preflight.json ]] || exit 2
        "$TRAIN_PY" "$ROOT_DIR/analysis/ep_overlap_budget_compare.py" \
            --campaign-dir "$AMPA_BUDGET_CAMPAIGN_ROOT" \
            --include-pair "$AMPA_BUDGET_PAIR_ROOT-posthoc/pair_comparison.json" \
            --preflight-proof "$AMPA_BUDGET_PREFLIGHT_ROOT/preflight.json" \
            --decision-output "$AMPA_BUDGET_PAIR_ROOT-posthoc/next_decision.json"
        jq -e --arg source "$source_digest" --arg protocol "$PROTOCOL" --arg pair "$AMPA_BUDGET_PAIR_ID" \
            '.valid_campaign_decision == true and .source_manifest_sha256 == $source
             and .experiment_protocol == $protocol and (.action | type=="string" and length>0)
             and (.accepted_pair_ids | type=="array" and index($pair)!=null)
             and (.next_pair==null or (.next_pair | type=="object"))' \
            "$AMPA_BUDGET_PAIR_ROOT-posthoc/next_decision.json" > /dev/null
    fi
    exit 0
fi
[[ $# == 0 ]] || exit 2
training_queue=$(squeue -h -j "$training_job" -o '%i')
[[ -z $training_queue ]] || exit 2
accounting=$(TZ=UTC SLURM_TIME_FORMAT=standard sacct -X -j "$training_job" --noheader --parsable2 \
    --format=JobIDRaw,State,ExitCode,Start,End,ElapsedRaw,NodeList,Partition,NNodes)
[[ $(awk -F'|' -v job="$training_job" '$1==job {print $2"|"$3"|"$8"|"$9}' <<< "$accounting") == 'COMPLETED|0:0|debug|4' ]] || exit 2
export AMPA_BUDGET_SESSION_DIR=${AMPA_BUDGET_SESSION_DIR:-$ROOT_DIR/progress/sessions/2026-11-10-qwen-ep-overlap}
export AMPA_BUDGET_STATUS_FILE=${AMPA_BUDGET_STATUS_FILE:-$ROOT_DIR/ai/ep-overlap-budget-launch/iteration-01/STATUS.md}
[[ $AMPA_BUDGET_SESSION_DIR == "$ROOT_DIR"/progress/sessions/*
    && -f $AMPA_BUDGET_SESSION_DIR/timeline.md && -f $AMPA_BUDGET_SESSION_DIR/session_state.md
    && $AMPA_BUDGET_STATUS_FILE == "$ROOT_DIR"/ai/*/iteration-*/STATUS.md && -f $AMPA_BUDGET_STATUS_FILE ]] || exit 2
export AMPA_BUDGET_POSTHOC_JOB_ROOT="$ROOT_DIR/runs/$AMPA_BUDGET_PAIR_ID-posthoc-job-$SLURM_JOB_ID"
[[ ! -e $AMPA_BUDGET_POSTHOC_JOB_ROOT && ! -L $AMPA_BUDGET_POSTHOC_JOB_ROOT ]] || exit 2
mkdir "$AMPA_BUDGET_POSTHOC_JOB_ROOT"
mkdir "$AMPA_BUDGET_POSTHOC_JOB_ROOT/logs"
printf '%s\n' "$accounting" > "$AMPA_BUDGET_POSTHOC_JOB_ROOT/training_accounting_utc.psv"
record() {
    local note="$(date -u '+%Y-%m-%d %H:%M:%S UTC') budget_posthoc=$SLURM_JOB_ID pair_training=$training_job: $*"
    printf '%s\n' "$note" | tee -a "$AMPA_BUDGET_POSTHOC_JOB_ROOT/timeline.md"
    printf '%s\n' "$note" > "$AMPA_BUDGET_POSTHOC_JOB_ROOT/CURRENT.md"
    printf '\n- %s\n' "$note" >> "$AMPA_BUDGET_SESSION_DIR/timeline.md"
    printf '\nCompute milestone: %s\n' "$note" >> "$AMPA_BUDGET_SESSION_DIR/session_state.md"
    printf '\nCompute milestone: %s\n' "$note" >> "$AMPA_BUDGET_STATUS_FILE"
}
complete=0
on_exit() {
    local status=$?
    trap - EXIT
    if (( status == 0 && complete != 1 )); then status=1; fi
    record "AFTER paired posthoc exit=$status complete=$complete; preserve original cases and partial analysis"
    printf 'exit_code=%s\ncomplete=%s\ntraining_job=%s\nposthoc_job=%s\n' "$status" "$complete" "$training_job" "$SLURM_JOB_ID" \
        > "$AMPA_BUDGET_POSTHOC_JOB_ROOT/exit.txt"
    exit "$status"
}
trap on_exit EXIT
record "BEFORE separate one-node debug analysis; pair allocation released; strict per-leg and paired placement/source/input gates; container=setup/gh200-cuda130-runtime.toml"
queue=$(squeue --array -u "$USER" --noheader --format='%i %P %T %D %R')
printf '%s\n' "$queue" > "$AMPA_BUDGET_POSTHOC_JOB_ROOT/queue-before-srun.txt"
requested=$(awk '{sum += $4} END {print sum+0}' <<< "$queue")
(( requested <= 4 )) || exit 2
srun --mpi=pmix --cpu-bind=none --container-remap-root --environment="$ROOT_DIR/setup/gh200-cuda130-runtime.toml" \
    --nodes=1 --ntasks=1 --gpus-per-task=1 --kill-on-bad-exit=1 \
    --output="$AMPA_BUDGET_POSTHOC_JOB_ROOT/logs/posthoc.out" --error="$AMPA_BUDGET_POSTHOC_JOB_ROOT/logs/posthoc.err" \
    bash "$SELF" --inside-container
jq -n --arg pair "$AMPA_BUDGET_PAIR_ID" --arg training "$training_job" --arg posthoc "$SLURM_JOB_ID" \
    --arg source "$source_digest" --arg protocol "$PROTOCOL" \
    '{success:true,pair_id:$pair,training_slurm_job_id:$training,posthoc_slurm_job_id:$posthoc,
      experiment_protocol:$protocol,source_manifest_sha256:$source,partition:"debug",requested_nodes:1,
      result:"both independent cases and their paired comparison passed; interpretation remains separate"}' \
    > "$AMPA_BUDGET_POSTHOC_JOB_ROOT/completed.json"
complete=1
record "AFTER both leg validations and paired comparison returned0; actual16GPU equality, inputs and timing distinction checked"
