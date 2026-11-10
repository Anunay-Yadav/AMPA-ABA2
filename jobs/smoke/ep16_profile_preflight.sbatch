#!/usr/bin/env bash
#SBATCH --job-name=ampa-ep16-preflight
#SBATCH --account=infra01
#SBATCH --partition=preemptable
#SBATCH --qos=normal
#SBATCH --constraint=gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --no-requeue
#SBATCH --output=.runtime/slurms/slurm-%x-%j.out
#SBATCH --error=.runtime/slurms/slurm-%x-%j.err

set -euo pipefail
[[ "${SLURM_JOB_PARTITION:-}" == preemptable ]] || exit 2
: "${AMPA_DATA_PREFIX:?}"
readonly ROOT_DIR="${SLURM_SUBMIT_DIR}"
squeue -u "$USER"
srun --mpi=pmix --container-remap-root --environment="${ROOT_DIR}/setup/gh200-cuda130-runtime.toml" \
    --nodes=1 --ntasks=1 --gpus-per-task=1 \
    bash -c 'set -euo pipefail
        cd "$WORKSPACE"
        export PYTHONPATH="$WORKSPACE:$WORKSPACE/profiling:$WORKSPACE/analysis${PYTHONPATH:+:$PYTHONPATH}"
        uv pip install --python /opt/venvs/megatron-bridge/bin/python --no-deps --no-build-isolation --editable "$WORKSPACE/Megatron-LM" --editable "$WORKSPACE/Megatron-Bridge"
        /opt/venvs/megatron-bridge/bin/python -m py_compile profiling/run_pretrain.py tools/ep16_indexed_manifest.py tools/qwen_workload.py profiling/ampa_profile/deep_trace.py profiling/ampa_profile/routing_audit.py profiling/ampa_profile/provenance.py profiling/ampa_profile/sample_fingerprint.py analysis/validate_torch_trace.py analysis/analyze_all_step_trace.py analysis/compare_all_step_profiles.py analysis/compare_deep_trace_overhead.py analysis/compare_profile_overhead.py analysis/check_ep16_smoke.py
        bash -n jobs/training/qwen3_ep16_all_steps.sbatch jobs/training/ep16_all_steps_group_routing.sbatch jobs/smoke/ep16_profile_preflight.sbatch
        /opt/venvs/megatron-bridge/bin/python -m unittest tests.test_ep16_runtime tests.test_ep16_observers tests.test_ep16_analysis tests.test_all_step_profiling tests.test_deep_trace tests.test_validate_torch_trace tests.test_indexed_run_provenance
        for cp in 1 2; do
            /opt/venvs/megatron-bridge/bin/python -m tools.ep16_indexed_manifest --variant "cp${cp}_ep16" --output "$WORKSPACE/dataset/manifests/${AMPA_DATA_PREFIX}-cp${cp}-ep16.json"
        done'
