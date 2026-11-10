#!/usr/bin/env bash
#SBATCH --job-name=ampa-uccl-build-verify
#SBATCH --account=infra01
#SBATCH --partition=preemptable
#SBATCH --qos=normal
#SBATCH --constraint=gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:03:00
#SBATCH --no-requeue
#SBATCH --output=.runtime/slurms/slurm-%x-%j.out
#SBATCH --error=.runtime/slurms/slurm-%x-%j.err
set -euo pipefail
[[ "${SLURM_JOB_PARTITION:-}" == preemptable ]] || exit 2
readonly ROOT_DIR="$SLURM_SUBMIT_DIR"
[[ "${AMPA_UCCL_VERIFY_ID:?}" =~ ^[a-zA-Z0-9._-]+$ ]] || exit 2
export AMPA_UCCL_VERIFY_ROOT="$ROOT_DIR/runs/$AMPA_UCCL_VERIFY_ID"
export AMPA_UCCL_BUILD_ROOT="$ROOT_DIR/runs/uccl-ep-build-20261101T1137Z-v7"
[[ ! -e "$AMPA_UCCL_VERIFY_ROOT" && ! -L "$AMPA_UCCL_VERIFY_ROOT" ]] || exit 2
squeue -u "$USER"
srun --mpi=pmix --container-remap-root --environment="$ROOT_DIR/setup/gh200-cuda130-runtime.toml" \
    --nodes=1 --ntasks=1 --gpus-per-task=1 bash -c 'set -euo pipefail
    cd "$WORKSPACE"
    mkdir -p "$AMPA_UCCL_VERIFY_ROOT/source"
    files=(jobs/smoke/uccl_ep_verify_build.sbatch profiling/ampa_profile/uccl_identity.py)
    cp "${files[@]}" "$AMPA_UCCL_VERIFY_ROOT/source/"
    sha256sum "${files[@]}" > "$AMPA_UCCL_VERIFY_ROOT/source.sha256"
    readonly VERIFY_PY=/opt/venvs/megatron-bridge/bin/python
    export LD_LIBRARY_PATH="/opt/amazon/efa/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export EFA_HOME="$AMPA_UCCL_BUILD_ROOT/not-efa"
    [[ ! -e "$EFA_HOME" ]] || exit 2
    export PYTHONPATH="$AMPA_UCCL_BUILD_ROOT/site:$AMPA_UCCL_BUILD_ROOT/deps:$AMPA_UCCL_BUILD_ROOT/uccl/ep/deep_ep_wrapper:$WORKSPACE${PYTHONPATH:+:$PYTHONPATH}"
    sha256sum -c "$AMPA_UCCL_BUILD_ROOT/binary.sha256" "$AMPA_UCCL_BUILD_ROOT/libfabric_runtime.sha256" \
        "$AMPA_UCCL_BUILD_ROOT/libibverbs_runtime.sha256"
    git -C "$AMPA_UCCL_BUILD_ROOT/uccl" diff --exit-code
    [[ -z "$(git -C "$AMPA_UCCL_BUILD_ROOT/uccl" status --porcelain)" ]] || exit 2
    "$AMPA_UCCL_BUILD_ROOT/fabric_probe" > "$AMPA_UCCL_VERIFY_ROOT/fabric_probe.txt"
    ldd "$AMPA_UCCL_BUILD_ROOT"/site/uccl/*.so > "$AMPA_UCCL_VERIFY_ROOT/extension_libraries.txt"
    "$VERIFY_PY" - <<"PY"
import json, os
from pathlib import Path
import torch
import uccl.ep as ep
import deep_ep.buffer as wrapper
from profiling.ampa_profile.uccl_identity import _manifest_file, _loaded_fabric
build = Path(os.environ["AMPA_UCCL_BUILD_ROOT"])
binary = _manifest_file(build / "binary.sha256", ep.__file__)
assert binary["sha256"] == "9cc779f30da88773476f7daf44abe65487332a5d06994bc15abc9be38ad56c50"
assert ep.num_max_nvl_peers() == 4 and wrapper.ep is ep
fabric = _loaded_fabric()
assert fabric == _manifest_file(build / "libfabric_runtime.sha256", fabric["path"])
record = {"valid_build_verification": True, "build_dir": str(build), "binary": binary,
          "libfabric": fabric, "num_max_nvl_peers": ep.num_max_nvl_peers(),
          "torch_version": torch.__version__, "job": os.environ["SLURM_JOB_ID"],
          "LD_LIBRARY_PATH": os.environ["LD_LIBRARY_PATH"],
          "original_build_job": "3825735", "original_build_job_state": "FAILED",
          "claim_limit": "Compilation/import/runtime binding only. Distributed training remains untested."}
with (Path(os.environ["AMPA_UCCL_VERIFY_ROOT"]) / "verification.json").open("x") as handle:
    json.dump(record, handle, indent=2, sort_keys=True)
print(json.dumps(record), flush=True)
PY
    sha256sum -c "$AMPA_UCCL_VERIFY_ROOT/source.sha256"
    printf "UCCL_BUILD_VERIFICATION_COMPLETE\n"
    '
