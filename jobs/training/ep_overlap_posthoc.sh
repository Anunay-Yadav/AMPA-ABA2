#!/usr/bin/env bash
set -euo pipefail
[[ $# == 1 && ${SLURM_JOB_PARTITION:-} == debug && ${SLURM_JOB_NUM_NODES:-} == 1 ]] || exit 2
cd "${SLURM_SUBMIT_DIR:?}"
readonly ROOT_DIR=$PWD
readonly TRAIN_PY=/opt/venvs/megatron-bridge/bin/python
export CUDA_DEVICE_MAX_CONNECTIONS=32 PYTHONDONTWRITEBYTECODE=1 FI_MR_CACHE_MONITOR=disabled
export AMPA_UCCL_BUILD_ROOT="$ROOT_DIR/runs/uccl-ep-owned-build-20261101T1713Z"
export EFA_HOME="$AMPA_UCCL_BUILD_ROOT/not-efa" UCCL_EP_TRANSPORT=cxi
export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/profiling:$ROOT_DIR/analysis:$ROOT_DIR/Megatron-LM:$ROOT_DIR/Megatron-Bridge/src:$AMPA_UCCL_BUILD_ROOT/site:$AMPA_UCCL_BUILD_ROOT/deps:$AMPA_UCCL_BUILD_ROOT/uccl/ep/deep_ep_wrapper${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="/opt/amazon/efa/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
[[ ${AMPA_OVERLAP_SOURCE_MANIFEST:?} == "$ROOT_DIR"/runs/* ]] || exit 2
read -r digest _ < <(sha256sum "$AMPA_OVERLAP_SOURCE_MANIFEST")
[[ $digest == "${AMPA_OVERLAP_SOURCE_SHA256:?}" ]] || exit 2
sha256sum --check --strict "$AMPA_OVERLAP_SOURCE_MANIFEST"
for required in jobs/training/ep_overlap_checked.sbatch jobs/training/ep_overlap_posthoc.sh \
    jobs/analysis/ep_overlap_posthoc.sbatch jobs/coordination/ep_overlap_campaign.sh \
    analysis/ep_overlap_gpu_activity.py tools/ep_overlap_validate.py tools/ep_overlap_nsys_posthoc.py \
    tests/test_ep_overlap_validate.py tests/test_ep_overlap_gpu_activity.py tests/test_ep_overlap_nsys_posthoc.py; do
    binding=$(sha256sum "$ROOT_DIR/$required")
    grep --fixed-strings --line-regexp --quiet -- "$binding" "$AMPA_OVERLAP_SOURCE_MANIFEST"
done
if [[ $1 == --qa ]]; then
    for file in ep_overlap_checked.sbatch ep_overlap_posthoc.sh ep_overlap_case.sbatch ep_overlap_worker.sh; do
        bash -n "$ROOT_DIR/jobs/training/$file"
    done
    "$TRAIN_PY" - <<'PY'
import json
import hashlib
import os
from pathlib import Path

proof_root = Path(os.environ['AMPA_OVERLAP_PREFLIGHT_ROOT'])
root = Path(os.environ['SLURM_SUBMIT_DIR'])
if proof_root.parent != root / 'runs' or proof_root.is_symlink():
    raise ValueError('invalid preflight evidence directory')
proof = json.loads((proof_root / 'preflight.json').read_text())
marker = dict(line.split('=', 1) for line in (proof_root / 'PASS.txt').read_text().splitlines())
if (proof.get('valid_preflight') is not True or proof.get('training_executed') is not False
        or proof.get('partition') != 'debug'
        or proof.get('source_manifest_sha256') != os.environ['AMPA_OVERLAP_SOURCE_SHA256']
        or marker.get('preflight_sha256') != hashlib.sha256((proof_root / 'preflight.json').read_bytes()).hexdigest()
        or marker.get('source_sha256') != os.environ['AMPA_OVERLAP_SOURCE_SHA256']
        or marker.get('job') != proof.get('job')):
    raise ValueError('matching allocated source/configuration preflight must pass first')
tests = {row['file']: row['tests'] for row in proof['focused_tests']}
required = {'test_ep_overlap_entrypoint.py', 'test_overlap_trace.py', 'test_ep_overlap_capture.py',
            'test_ep_overlap_validate.py', 'test_ep_overlap_gpu_activity.py', 'test_ep_overlap_nsys_posthoc.py'}
if set(tests) != required or not all(type(count) is int and count > 0 for count in tests.values()):
    raise ValueError('preflight omits required successful focused tests')
print(json.dumps({'matching_preflight': str(proof_root), 'focused_tests': tests, 'tests_repeated': False}))
PY
elif [[ $1 == --analyze ]]; then
    [[ ${AMPA_OVERLAP_ROOT:?} == "$ROOT_DIR/runs/$AMPA_OVERLAP_ID" ]] || exit 2
    "$TRAIN_PY" "$ROOT_DIR/tools/ep_overlap_validate.py" \
        --orchestration-dir "$AMPA_OVERLAP_ROOT" --output-dir "$AMPA_OVERLAP_ROOT-posthoc"
    if [[ ${AMPA_OVERLAP_CAPTURE:?} == torch ]]; then
        "$TRAIN_PY" - <<'PY'
import gc
import gzip
import json
import os
from pathlib import Path
from analysis.ep_overlap_gpu_activity import inspect_events
from analysis.inspect_uccl_trace import sha256

root = Path(os.environ['AMPA_OVERLAP_ROOT'])
run = Path(os.environ['AMPA_OVERLAP_RUN'])
out = Path(str(root) + '-posthoc') / 'gpu_activity'
out.mkdir()
rows = []
source_manifest = Path(os.environ['AMPA_OVERLAP_SOURCE_MANIFEST'])
source_sha = sha256(source_manifest)
if source_sha != os.environ['AMPA_OVERLAP_SOURCE_SHA256']:
    raise ValueError('analysis source manifest differs')
programs = [Path('analysis/ep_overlap_gpu_activity.py'), Path('analysis/inspect_uccl_trace.py'),
            Path('profiling/ampa_profile/overlap_trace.py'), Path('profiling/ampa_profile/deep_trace.py'),
            Path('profiling/ampa_profile/uccl_trace.py'), Path('jobs/training/ep_overlap_posthoc.sh')]
program_inputs = {str(path.resolve()): sha256(path) for path in programs}
for step in range(int(os.environ['AMPA_OVERLAP_STEPS'])):
    for rank in range(16):
        trace_path = run / 'traces/torch_profile' / f'step-{step:05d}' / f'rank-{rank}.json.gz'
        record_path = run / 'raw/overlap_calls' / f'step-{step:05d}' / f'rank-{rank}.json'
        bindings = {str(path): sha256(path) for path in (trace_path, record_path)}
        with gzip.open(trace_path, 'rt') as stream:
            trace = json.load(stream)
        result = inspect_events(trace, json.loads(record_path.read_text()))
        if bindings != {str(path): sha256(path) for path in (trace_path, record_path)}:
            raise ValueError('trace or journal changed during GPU analysis')
        result['inputs'] = bindings
        result['program_inputs'] = program_inputs
        result['source_manifest_sha256'] = source_sha
        with (out / f'step-{step:05d}-rank-{rank}.json').open('x') as stream:
            json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n')
        rows.append({key: result[key] for key in ('rank', 'step', 'communication_backend',
                     'known_payload_kernels', 'correlated_compute_kernels', 'intersection_ms',
                     'observed_positive_gpu_concurrency', 'whole_session_event_completeness')})
        del trace, result
        gc.collect()
if program_inputs != {str(path.resolve()): sha256(path) for path in programs} or sha256(source_manifest) != source_sha:
    raise ValueError('analysis program identity changed')
with (out / 'summary.json').open('x') as stream:
    json.dump({'captures': rows, 'capture_count': len(rows),
               'program_inputs': program_inputs, 'source_manifest_sha256': source_sha,
               'positive_concurrency_capture_count': sum(row['observed_positive_gpu_concurrency'] for row in rows),
               'claim_limit': 'Recorded known payload activity and selected computation only; no exposed-latency or free-SM claim.'},
              stream, indent=2, sort_keys=True, allow_nan=False)
    stream.write('\n')
PY
    elif [[ ${AMPA_OVERLAP_CAPTURE:?} == nsys && ${AMPA_OVERLAP_STEPS:?} == 10 ]]; then
        timeout --signal=TERM --kill-after=30s 4800s "$TRAIN_PY" "$ROOT_DIR/tools/ep_overlap_nsys_posthoc.py" \
            --orchestration-dir "$AMPA_OVERLAP_ROOT" \
            --execution-accounting "$AMPA_OVERLAP_ROOT-posthoc/execution_accounting.json" \
            --output-dir "$AMPA_OVERLAP_ROOT-nsys-posthoc" --hardware-metrics --max-seconds 4500
    fi
else
    exit 2
fi
