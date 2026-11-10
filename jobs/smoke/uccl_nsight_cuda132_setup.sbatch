#!/usr/bin/env bash
# Isolated verified Nsight installation, followed by the unchanged capability probe.
#SBATCH --job-name=ampa-uccl-nsys132
#SBATCH --account=infra01
#SBATCH --partition=debug
#SBATCH --qos=normal
#SBATCH --constraint=gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --exclusive
#SBATCH --time=00:10:00
#SBATCH --no-requeue
#SBATCH --output=.runtime/slurms/slurm-%x-%j.out
#SBATCH --error=.runtime/slurms/slurm-%x-%j.err
set -euo pipefail
[[ ${SLURM_JOB_PARTITION:-} == debug && ${SLURM_JOB_NUM_NODES:-} == 1 ]] || exit 2
readonly ROOT_DIR=${SLURM_SUBMIT_DIR:?}
[[ ${AMPA_UCCL_NSYS_SETUP_ID:?} =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*$ ]] || exit 2
export AMPA_UCCL_NSYS_SETUP_ROOT="$ROOT_DIR/runs/$AMPA_UCCL_NSYS_SETUP_ID"
export AMPA_UCCL_NSYS_CAP_ID="$AMPA_UCCL_NSYS_SETUP_ID-cap"
export AMPA_UCCL_NSYS_CAP_ROOT="$ROOT_DIR/runs/$AMPA_UCCL_NSYS_CAP_ID"
export PYTHONDONTWRITEBYTECODE=1
readonly OUT=$AMPA_UCCL_NSYS_SETUP_ROOT
readonly CAP=$AMPA_UCCL_NSYS_CAP_ROOT
readonly PROBE="$ROOT_DIR/jobs/smoke/uccl_nsight_capabilities.sbatch"
readonly ARCHIVE_URL=https://developer.download.nvidia.com/compute/cuda/redist/nsight_systems/linux-sbsa/nsight_systems-linux-sbsa-2025.6.3.343-archive.tar.xz
readonly METADATA_URL=https://developer.download.nvidia.com/compute/cuda/redist/redistrib_13.2.0.json
readonly ARCHIVE_SHA=0ea6098fd201eecae071461dda7a4f5358de383109fdaa9ebb5221b06af550c9
readonly ARCHIVE_BYTES=1120017636
readonly PROBE_SHA=e63ce79e2cdf51c7566e649545510c17356e4560f3aaad422f72205ebad12f28

if [[ ${1:-} == --setup-container ]]; then
    [[ $# == 1 && ${WORKSPACE:-} == "$ROOT_DIR" && ${SLURM_PROCID:-} == 0 ]] || exit 2
    [[ $(cat "$OUT/job-id") == "$SLURM_JOB_ID" ]] || exit 2
    cd "$ROOT_DIR"
    sha256sum --check "$OUT/source.sha256" > "$OUT/logs/source-before-setup.txt"
    for tool in curl tar sha256sum stat find readlink timeout; do command -v "$tool"; done
    curl --version > "$OUT/logs/curl-version.txt"
    partial="$OUT/nsight_systems-linux-sbsa-2025.6.3.343-archive.tar.xz.part"
    archive="${partial%.part}"
    [[ ! -e $partial && ! -e $archive && ! -L $partial && ! -L $archive ]] || exit 2
    printf 'archive_url=%s\nmetadata_url=%s\nexpected_sha256=%s\nexpected_bytes=%s\n' \
        "$ARCHIVE_URL" "$METADATA_URL" "$ARCHIVE_SHA" "$ARCHIVE_BYTES" > "$OUT/download.txt"
    resume_args=()
    export AMPA_UCCL_NSYS_RESUMED_BYTES=0
    if [[ -n ${AMPA_UCCL_NSYS_RESUME_ROOT:-} ]]; then
        resume_root=$AMPA_UCCL_NSYS_RESUME_ROOT
        [[ $resume_root == "$ROOT_DIR"/runs/* && -d $resume_root && ! -L $resume_root ]] || exit 2
        resume_source="$resume_root/$(basename "$partial")"
        [[ -f $resume_source && ! -L $resume_source ]] || exit 2
        resume_bytes=$(stat -c %s "$resume_source")
        (( resume_bytes > 0 && resume_bytes < ARCHIVE_BYTES )) || exit 2
        sha256sum "$resume_source" > "$OUT/resume-source.sha256"
        cp --reflink=auto -- "$resume_source" "$partial"
        export AMPA_UCCL_NSYS_RESUMED_BYTES=$resume_bytes
        printf 'resume_source=%s\nresumed_bytes=%s\n' "$resume_source" "$resume_bytes" >> "$OUT/download.txt"
        resume_args=(--continue-at -)
    fi
    curl --proto '=https' --proto-redir '=https' --tlsv1.2 --location --fail --silent --show-error \
        --connect-timeout 20 --max-time 300 --retry 0 "${resume_args[@]}" \
        --dump-header "$OUT/logs/download-headers.txt" --output "$partial" \
        --write-out 'http_code=%{http_code}\nsize_download=%{size_download}\ntime_total=%{time_total}\nurl_effective=%{url_effective}\n' \
        "$ARCHIVE_URL" > "$OUT/logs/download.out" 2> "$OUT/logs/download.err"
    actual_bytes=$(stat -c %s "$partial")
    printf 'actual_bytes=%s\n' "$actual_bytes" >> "$OUT/download.txt"
    [[ $actual_bytes == "$ARCHIVE_BYTES" ]] || { printf 'Archive byte count differs.\n' >&2; exit 3; }
    printf '%s  %s\n' "$ARCHIVE_SHA" "$partial" | sha256sum --check - > "$OUT/logs/archive-verification.txt"
    mv "$partial" "$archive"
    printf '%s  %s\n' "$ARCHIVE_SHA" "$archive" > "$OUT/archive.sha256"
    printf 'archive_verified=true\n' >> "$OUT/download.txt"
    mkdir "$OUT/tool"
    tar --extract --xz --file "$archive" --directory "$OUT/tool" --no-same-owner --no-same-permissions \
        > "$OUT/logs/extract.out" 2> "$OUT/logs/extract.err"
    mapfile -t candidates < <(find "$OUT/tool" -type f -path '*/target-linux-sbsa-armv8/nsys' -perm -u+x -print)
    if (( ${#candidates[@]} == 0 )); then
        mapfile -t candidates < <(find "$OUT/tool" -type f -name nsys -perm -u+x -print)
    fi
    printf '%s\n' "${candidates[@]}" > "$OUT/nsys-candidates.txt"
    [[ ${#candidates[@]} == 1 ]] || { printf 'Expected one executable Nsight CLI.\n' >&2; exit 3; }
    nsys=$(readlink -f "${candidates[0]}")
    [[ $nsys == "$OUT/tool/"* && -f $nsys && -x $nsys ]] || exit 3
    timeout --signal=TERM --kill-after=2s 15s "$nsys" --version > "$OUT/logs/nsys-version.out" 2> "$OUT/logs/nsys-version.err"
    grep -F '2025.6.3.343' "$OUT/logs/nsys-version.out"
    printf '%s\n' "$nsys" > "$OUT/nsys-path.txt"
    sha256sum "$nsys" > "$OUT/nsys.sha256"
    /opt/venvs/megatron-bridge/bin/python - "$OUT" "$nsys" "$archive" <<'PY'
import datetime
import json
import os
from pathlib import Path
import socket
import sys

out, tool, archive = map(Path, sys.argv[1:])
record = {'schema_version': 1, 'setup_verified': True, 'job': os.environ['SLURM_JOB_ID'],
    'hostname': socket.gethostname(), 'execution_location': 'allocated_configured_container',
    'metadata_url': 'https://developer.download.nvidia.com/compute/cuda/redist/redistrib_13.2.0.json',
    'archive_url': 'https://developer.download.nvidia.com/compute/cuda/redist/nsight_systems/linux-sbsa/nsight_systems-linux-sbsa-2025.6.3.343-archive.tar.xz',
    'archive_path': str(archive), 'archive_sha256': '0ea6098fd201eecae071461dda7a4f5358de383109fdaa9ebb5221b06af550c9',
    'expected_bytes': 1120017636, 'actual_bytes': archive.stat().st_size,
    'resume_root': os.environ.get('AMPA_UCCL_NSYS_RESUME_ROOT'),
    'resumed_bytes': int(os.environ['AMPA_UCCL_NSYS_RESUMED_BYTES']),
    'nsys_path': str(tool), 'version': (out / 'logs/nsys-version.out').read_text().strip(),
    'system_installation_modified': False, 'native_training_sources_modified': False,
    'verified_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
with (out / 'setup.json').open('x') as stream:
    json.dump(record, stream, indent=2, sort_keys=True)
    stream.write('\n')
PY
    sha256sum --check "$OUT/source.sha256" > "$OUT/logs/source-after-setup.txt"
    exit 0
elif [[ ${1:-} == --capability-container ]]; then
    [[ $# == 1 && ${WORKSPACE:-} == "$ROOT_DIR" && ${SLURM_PROCID:-} == 0 ]] || exit 2
    [[ $(cat "$OUT/job-id") == "$SLURM_JOB_ID" && $(cat "$CAP/job-id") == "$SLURM_JOB_ID" ]] || exit 2
    cd "$ROOT_DIR"
    sha256sum --check "$OUT/source.sha256" > "$OUT/logs/source-before-capability.txt"
    nsys=$(cat "$OUT/nsys-path.txt")
    [[ $nsys == "$OUT/tool/"* && -f $nsys && -x $nsys ]] || exit 3
    printf 'nsys_path=%s\nPATH_prefix=%s\n' "$nsys" "$(dirname "$nsys")" > "$CAP/tool-selection.txt"
    set +e
    env PATH="$(dirname "$nsys"):$PATH" \
        AMPA_UCCL_NSYS_CAP_ID="$AMPA_UCCL_NSYS_CAP_ID" \
        bash "$PROBE" --container
    capability_status=$?
    set -e
    printf 'exit_code=%s\nfinished_utc=%s\n' "$capability_status" "$(date -u +%FT%TZ)" > "$CAP/exit.txt"
    sha256sum --check "$OUT/source.sha256" > "$OUT/logs/source-after-capability.txt"
    (( capability_status == 0 )) || exit "$capability_status"
    /opt/venvs/megatron-bridge/bin/python - "$CAP" "$nsys" <<'PY'
import json
from pathlib import Path
import sys

cap, selected = Path(sys.argv[1]), Path(sys.argv[2]).resolve()
record = json.loads((cap / 'capabilities.json').read_text())
if record.get('native_report_generated') is not True or Path(record.get('nsys_path', '')).resolve() != selected:
    raise ValueError('capability probe did not exercise the selected new Nsight CLI')
PY
    exit 0
fi

[[ $# == 0 && ! -e $OUT && ! -L $OUT && ! -e $CAP && ! -L $CAP ]] || exit 2
export AMPA_UCCL_NSYS_SETUP_DEADLINE=$(( $(date +%s) + 580 ))
mkdir "$OUT" "$CAP"
mkdir -p "$OUT"/{logs,source/jobs/smoke} "$CAP"/{logs,captures,source}
printf '%s\n' "$SLURM_JOB_ID" | tee "$OUT/job-id" > "$CAP/job-id"
cp "$ROOT_DIR/jobs/smoke/uccl_nsight_cuda132_setup.sbatch" "$OUT/source/jobs/smoke/"
cp "$PROBE" "$OUT/source/jobs/smoke/"
sha256sum "$ROOT_DIR/jobs/smoke/uccl_nsight_cuda132_setup.sbatch" "$PROBE" > "$OUT/source.sha256"
printf '%s  %s\n' "$PROBE_SHA" "$PROBE" | sha256sum --check - > "$OUT/logs/immutable-probe-source.txt"
cp "$PROBE" "$CAP/source/uccl_nsight_capabilities.sbatch"
printf '%s  %s\n' "$PROBE_SHA" "$PROBE" > "$CAP/source.sha256"
printf 'job=%s\npartition=%s\nnodes=%s\nstarted_utc=%s\n' "$SLURM_JOB_ID" "$SLURM_JOB_PARTITION" "$SLURM_JOB_NODELIST" "$(date -u +%FT%TZ)" > "$OUT/allocation.txt"
cp "$OUT/allocation.txt" "$CAP/allocation.txt"
if [[ -f "$ROOT_DIR/ai/qwen-uccl-block-sweep/iteration-01/NSIGHT_CUDA132_PACKAGE.json" ]]; then
    cp "$ROOT_DIR/ai/qwen-uccl-block-sweep/iteration-01/NSIGHT_CUDA132_PACKAGE.json" "$OUT/package-provenance.json"
fi
record() {
    local note="$(date -u '+%Y-%m-%d %H:%M:%S UTC') nsight132_setup job=$SLURM_JOB_ID: $*"
    printf '%s\n' "$note" >> "$OUT/timeline.md"
    printf '%s\n' "$note" > "$OUT/CURRENT.md"
    for target in progress/sessions/2026-11-01-uccl-ep-comparison/timeline.md \
        progress/sessions/2026-11-01-uccl-ep-comparison/session_state.md \
        ai/qwen-uccl-block-sweep/iteration-01/STATUS.md; do
        printf '\n- %s\n' "$note" >> "$ROOT_DIR/$target"
    done
}
complete=0
on_exit() {
    local status=$?
    trap - EXIT
    printf 'exit_code=%s\ncomplete=%s\n' "$status" "$complete" > "$OUT/exit.txt"
    record "AFTER compute status=$status complete=$complete; preserve all downloaded/setup/capability artifacts at $OUT and $CAP"
    exit "$status"
}
trap on_exit EXIT
record "BEFORE compute isolated official Nsight2025.6.3.343 download/SHA verification/extraction; one exclusive debugGHnode/$SLURM_JOB_NODELIST, configuredcontainer, ten-minute allocation; no training/system/native-source installation"
remaining=$((AMPA_UCCL_NSYS_SETUP_DEADLINE - $(date +%s)))
(( remaining > 5 )) || exit 124
(( remaining > 380 )) && remaining=380
squeue -u "$USER"
timeout --signal=TERM --kill-after=5s "${remaining}s" \
    srun --mpi=pmix --cpu-bind=none --container-remap-root --environment="$ROOT_DIR/setup/gh200-cuda130-runtime.toml" \
        --nodes=1 --ntasks=1 --gpus-per-task=1 \
        --output="$OUT/logs/setup-container.out" --error="$OUT/logs/setup-container.err" \
        bash "$ROOT_DIR/jobs/smoke/uccl_nsight_cuda132_setup.sbatch" --setup-container
record "AFTER isolated setup; BEFORE fresh configured srun of unchanged native-report/metric capability probe with the new tool directory first on PATH"
remaining=$((AMPA_UCCL_NSYS_SETUP_DEADLINE - $(date +%s)))
(( remaining > 5 )) || exit 124
(( remaining > 270 )) && remaining=270
squeue -u "$USER"
timeout --signal=TERM --kill-after=5s "${remaining}s" \
    srun --mpi=pmix --cpu-bind=none --container-remap-root --environment="$ROOT_DIR/setup/gh200-cuda130-runtime.toml" \
        --nodes=1 --ntasks=1 --gpus-per-task=1 \
        --output="$CAP/logs/container.out" --error="$CAP/logs/container.err" \
        bash "$ROOT_DIR/jobs/smoke/uccl_nsight_cuda132_setup.sbatch" --capability-container
complete=1
printf 'setup_and_native_capability_passed=true\njob=%s\ncapability_root=%s\n' "$SLURM_JOB_ID" "$CAP" > "$OUT/PASS.txt"
