#!/usr/bin/env bash
#SBATCH --job-name=ampa-uccl-build
#SBATCH --account=infra01
#SBATCH --partition=preemptable
#SBATCH --qos=normal
#SBATCH --constraint=gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --output=.runtime/slurms/slurm-%x-%j.out
#SBATCH --error=.runtime/slurms/slurm-%x-%j.err
set -euo pipefail
[[ "${SLURM_JOB_PARTITION:-}" == preemptable ]] || exit 2
readonly ROOT_DIR="${SLURM_SUBMIT_DIR}"
[[ "${AMPA_UCCL_BUILD_ID:?}" =~ ^[a-zA-Z0-9._-]+$ ]] || exit 2
export AMPA_UCCL_BUILD_ROOT="${ROOT_DIR}/runs/${AMPA_UCCL_BUILD_ID}"
[[ ! -e "$AMPA_UCCL_BUILD_ROOT" && ! -L "$AMPA_UCCL_BUILD_ROOT" ]] || exit 2
squeue -u "$USER"
srun --mpi=pmix --container-remap-root --environment="${ROOT_DIR}/setup/gh200-cuda130-runtime.toml" \
    --nodes=1 --ntasks=1 --gpus-per-task=1 bash -c 'set -euo pipefail
    readonly BUILD_PY=/opt/venvs/megatron-bridge/bin/python
    [[ -x "$BUILD_PY" ]] || exit 2
    for executable in git grep nvcc ninja g++ fi_info ldconfig readelf; do command -v "$executable"; done
    readonly BASE_REVISION=c8fd14e93f2106b69a0d30e0f78c12a52e07a181
    readonly APPROVED_PATCH_SHA=749a28bdb82d25ba92bace55ad6d99bafc1075280bb5466e12f378f5a53a21b2
    mkdir -p "$AMPA_UCCL_BUILD_ROOT"/{source,deps,site/uccl,objects}
    files=(jobs/smoke/uccl_ep_build.sbatch tools/uccl_ep_source_identity.py
        tests/test_uccl_ep_source_identity.py profiling/ampa_profile/uccl_identity.py tests/test_uccl_identity.py
        ai/qwen-uccl-ep-comparison/iteration-01/02-implement/uccl-borrowed-scratch.patch)
    cd "$WORKSPACE"
    sha256sum "${files[@]/#/$WORKSPACE/}" > "$AMPA_UCCL_BUILD_ROOT/build-source.sha256"
    cp "${files[@]}" "$AMPA_UCCL_BUILD_ROOT/source/"
    readonly PATCH_ARCHIVE="$AMPA_UCCL_BUILD_ROOT/source/uccl-borrowed-scratch.patch"
    printf "%s  %s\n" "$APPROVED_PATCH_SHA" "$PATCH_ARCHIVE" | sha256sum -c -
    git clone --depth=1 --filter=blob:none --no-checkout https://github.com/uccl-project/uccl.git "$AMPA_UCCL_BUILD_ROOT/uccl"
    git -C "$AMPA_UCCL_BUILD_ROOT/uccl" fetch --depth=1 origin "$BASE_REVISION"
    git -C "$AMPA_UCCL_BUILD_ROOT/uccl" sparse-checkout init --cone
    git -C "$AMPA_UCCL_BUILD_ROOT/uccl" sparse-checkout set ep include uccl
    git -C "$AMPA_UCCL_BUILD_ROOT/uccl" checkout --detach "$BASE_REVISION"
    readonly BUILD_SOURCE="$AMPA_UCCL_BUILD_ROOT/uccl"
    [[ "$(git -C "$BUILD_SOURCE" rev-parse HEAD)" == "$BASE_REVISION" && -z "$(git -C "$BUILD_SOURCE" status --porcelain)" ]] || exit 2
    git -C "$BUILD_SOURCE" apply --check "$PATCH_ARCHIVE"
    git -C "$BUILD_SOURCE" apply "$PATCH_ARCHIVE"
    export PYTHONPATH="$WORKSPACE:$WORKSPACE/profiling${PYTHONPATH:+:$PYTHONPATH}"
    "$BUILD_PY" "$WORKSPACE/tools/uccl_ep_source_identity.py" --build-dir "$AMPA_UCCL_BUILD_ROOT" --write \
        > "$AMPA_UCCL_BUILD_ROOT/source_identity_before.json"
    sha256sum "$AMPA_UCCL_BUILD_ROOT/source_identity.json" > "$AMPA_UCCL_BUILD_ROOT/source_identity.sha256"
    "$BUILD_PY" -m py_compile "$WORKSPACE/tools/uccl_ep_source_identity.py"
    for pattern in test_uccl_ep_source_identity.py test_uccl_identity.py; do
        CUDA_VISIBLE_DEVICES="" "$BUILD_PY" -m unittest discover -s "$WORKSPACE/tests" -p "$pattern" -v \
            >> "$AMPA_UCCL_BUILD_ROOT/source_identity_tests.log" 2>&1
    done
    cat "$AMPA_UCCL_BUILD_ROOT/source_identity_tests.log"
    cp -R --no-preserve=all "$BUILD_SOURCE/uccl/." "$AMPA_UCCL_BUILD_ROOT/site/uccl/"
    cd "$BUILD_SOURCE/ep"
    export USE_LIBFABRIC_CXI=1 NUM_MAX_NVL_PEERS=4 TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=8
    export EFA_HOME="$AMPA_UCCL_BUILD_ROOT/not-efa"
    [[ ! -e "$EFA_HOME" ]] || exit 2
    export PYTHONPATH="$AMPA_UCCL_BUILD_ROOT/deps:$AMPA_UCCL_BUILD_ROOT/site:$BUILD_SOURCE/ep/deep_ep_wrapper:$BUILD_SOURCE:$WORKSPACE:$WORKSPACE/profiling${PYTHONPATH:+:$PYTHONPATH}"
    "$BUILD_PY" -m pip install --no-deps --target "$AMPA_UCCL_BUILD_ROOT/deps" \
        --report "$AMPA_UCCL_BUILD_ROOT/nanobind-install.json" nanobind==3.1.0
    for header in /usr/include/infiniband/verbs.h /usr/include/numa.h /usr/include/numaif.h; do
        [[ -f "$header" ]] || { printf "Required build header unavailable: %s\n" "$header"; exit 2; }
    done
    command -v fi_info
    fi_info --version
    ldd "$(command -v fi_info)" > "$AMPA_UCCL_BUILD_ROOT/fi_info_libraries.txt"
    [[ "$(fi_info --version)" == *"libfabric: 2.6.0"* ]] || exit 2
    fabric_runtime=""
    while read -r library arrow location rest; do
        if [[ "$library" == libfabric.so* ]]; then fabric_runtime="$location"; break; fi
    done < "$AMPA_UCCL_BUILD_ROOT/fi_info_libraries.txt"
    [[ -f "$fabric_runtime" ]] || exit 2
    export LD_LIBRARY_PATH="$(dirname "$fabric_runtime")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    git clone --depth=1 --no-checkout --branch=v2.6.0 https://github.com/ofiwg/libfabric.git "$AMPA_UCCL_BUILD_ROOT/libfabric-source"
    git -C "$AMPA_UCCL_BUILD_ROOT/libfabric-source" sparse-checkout init --cone
    git -C "$AMPA_UCCL_BUILD_ROOT/libfabric-source" sparse-checkout set include
    git -C "$AMPA_UCCL_BUILD_ROOT/libfabric-source" checkout --detach HEAD
    git -C "$AMPA_UCCL_BUILD_ROOT/libfabric-source" rev-parse HEAD > "$AMPA_UCCL_BUILD_ROOT/libfabric_headers_commit.txt"
    export LIBFABRIC_HOME="$AMPA_UCCL_BUILD_ROOT/libfabric-dev"
    mkdir -p "$LIBFABRIC_HOME"/{include,lib}
    cp -R --no-preserve=all "$AMPA_UCCL_BUILD_ROOT/libfabric-source/include/rdma" "$LIBFABRIC_HOME/include/"
    ln -s "$fabric_runtime" "$LIBFABRIC_HOME/lib/libfabric.so"
    sha256sum "$fabric_runtime" > "$AMPA_UCCL_BUILD_ROOT/libfabric_runtime.sha256"
    ldconfig -p > "$AMPA_UCCL_BUILD_ROOT/runtime_libraries.txt"
    verbs_runtime=""
    while read -r library remainder; do
        if [[ "$library" == libibverbs.so.1 ]]; then
            verbs_runtime="${remainder##*=> }"
            break
        fi
    done < "$AMPA_UCCL_BUILD_ROOT/runtime_libraries.txt"
    [[ -f "$verbs_runtime" ]] || exit 2
    verbs_runtime="$(readlink -f "$verbs_runtime")"
    ln -s "$verbs_runtime" "$LIBFABRIC_HOME/lib/libibverbs.so"
    sha256sum "$verbs_runtime" > "$AMPA_UCCL_BUILD_ROOT/libibverbs_runtime.sha256"
    readelf -h "$verbs_runtime" > "$AMPA_UCCL_BUILD_ROOT/libibverbs_elf.txt"
    ldd "$verbs_runtime" > "$AMPA_UCCL_BUILD_ROOT/libibverbs_libraries.txt"
    if grep -q "not found" "$AMPA_UCCL_BUILD_ROOT/libibverbs_libraries.txt"; then exit 2; fi
    cat > "$AMPA_UCCL_BUILD_ROOT/fabric_probe.cc" <<"CPP"
#include <rdma/fabric.h>
#include <rdma/fi_domain.h>
#include <cstdio>
#include <cstring>
int main() {
  std::printf("headers=%d.%d runtime=%u.%u\n", FI_MAJOR_VERSION, FI_MINOR_VERSION,
              FI_MAJOR(fi_version()), FI_MINOR(fi_version()));
  auto *hints = fi_allocinfo();
  if (!hints) return 2;
  hints->caps = FI_RMA | FI_WRITE | FI_ATOMIC | FI_FENCE | FI_REMOTE_WRITE |
                FI_REMOTE_READ | FI_HMEM | FI_LOCAL_COMM | FI_REMOTE_COMM;
  hints->mode = FI_CONTEXT;
  hints->fabric_attr->prov_name = strdup("cxi");
  hints->ep_attr->type = FI_EP_RDM;
  hints->domain_attr->mr_mode = FI_MR_ENDPOINT | FI_MR_ALLOCATED | FI_MR_PROV_KEY;
  hints->domain_attr->threading = FI_THREAD_DOMAIN;
  fi_info *info = nullptr;
  int rc = fi_getinfo(FI_VERSION(1, 15), nullptr, nullptr, 0, hints, &info);
  std::printf("CXI discovery rc=%d\n", rc);
  if (!rc) for (auto *entry = info; entry; entry = entry->next)
    std::printf("domain=%s\n", entry->domain_attr->name);
  if (info) fi_freeinfo(info);
  fi_freeinfo(hints);
  return rc ? 1 : 0;
}
CPP
    g++ -I "$LIBFABRIC_HOME/include" "$AMPA_UCCL_BUILD_ROOT/fabric_probe.cc" \
        -L "$LIBFABRIC_HOME/lib" -Wl,-rpath,"$(dirname "$fabric_runtime")" -lfabric \
        -libverbs -lnuma -lnl-3 -lnl-route-3 -o "$AMPA_UCCL_BUILD_ROOT/fabric_probe"
    if ! "$AMPA_UCCL_BUILD_ROOT/fabric_probe" > "$AMPA_UCCL_BUILD_ROOT/fabric_probe.txt"; then
        FI_LOG_LEVEL=warn fi_info -p cxi -t FI_EP_RDM > "$AMPA_UCCL_BUILD_ROOT/fabric_diagnostic.txt" 2>&1 || true
        cat "$AMPA_UCCL_BUILD_ROOT/fabric_probe.txt"
        exit 2
    fi
    cat "$AMPA_UCCL_BUILD_ROOT/fabric_probe.txt"
    git -C "$BUILD_SOURCE" rev-parse HEAD > "$AMPA_UCCL_BUILD_ROOT/uccl_commit.txt"
    env | sort | grep -E "^(USE_LIBFABRIC_CXI|NUM_MAX_NVL_PEERS|TORCH_CUDA_ARCH_LIST|MAX_JOBS|LIBFABRIC_HOME|EFA_HOME)=" > "$AMPA_UCCL_BUILD_ROOT/build_flags.txt"
    "$BUILD_PY" setup.py build_ext --build-lib "$AMPA_UCCL_BUILD_ROOT/site/uccl" \
        --build-temp "$AMPA_UCCL_BUILD_ROOT/objects" > "$AMPA_UCCL_BUILD_ROOT/build.log" 2>&1
    "$BUILD_PY" -c "import torch, uccl.ep, deep_ep; print(uccl.ep.__file__); print(deep_ep.__file__); print(uccl.ep.num_max_nvl_peers()); assert uccl.ep.num_max_nvl_peers() == 4" \
        > "$AMPA_UCCL_BUILD_ROOT/import-check.txt"
    cat "$AMPA_UCCL_BUILD_ROOT/import-check.txt"
    sha256sum "$AMPA_UCCL_BUILD_ROOT"/site/uccl/*.so > "$AMPA_UCCL_BUILD_ROOT/binary.sha256"
    ldd "$AMPA_UCCL_BUILD_ROOT"/site/uccl/*.so > "$AMPA_UCCL_BUILD_ROOT/extension_libraries.txt"
    extension_fabric=""
    while read -r library arrow location rest; do
        if [[ "$library" == libfabric.so* ]]; then extension_fabric="$location"; break; fi
    done < "$AMPA_UCCL_BUILD_ROOT/extension_libraries.txt"
    [[ -f "$extension_fabric" && "$(readlink -f "$extension_fabric")" == "$(readlink -f "$fabric_runtime")" ]] || exit 2
    while read -r library arrow location rest; do
        if [[ "$library" == libibverbs.so* ]]; then
            [[ -f "$location" && "$(readlink -f "$location")" == "$verbs_runtime" ]] || exit 2
        fi
    done < "$AMPA_UCCL_BUILD_ROOT/extension_libraries.txt"
    sha256sum -c "$AMPA_UCCL_BUILD_ROOT/libibverbs_runtime.sha256"
    sha256sum -c "$AMPA_UCCL_BUILD_ROOT/libfabric_runtime.sha256"
    "$BUILD_PY" "$WORKSPACE/tools/uccl_ep_source_identity.py" --build-dir "$AMPA_UCCL_BUILD_ROOT" \
        > "$AMPA_UCCL_BUILD_ROOT/source_identity_after.json"
    sha256sum -c "$AMPA_UCCL_BUILD_ROOT/source_identity.sha256" "$AMPA_UCCL_BUILD_ROOT/build-source.sha256"
    "$BUILD_PY" - <<"PY"
import hashlib
import json
import os
from pathlib import Path
import torch
import uccl.ep as ep
import deep_ep.buffer as wrapper
from tools.uccl_ep_source_identity import verify_uccl_source

build = Path(os.environ["AMPA_UCCL_BUILD_ROOT"])
def bound_file(manifest, actual):
    lines = manifest.read_text().splitlines()
    assert len(lines) == 1
    digest, name = lines[0].split(maxsplit=1)
    path = Path(name.lstrip(" *")).resolve()
    assert path == Path(actual).resolve() and path.is_file()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    return {"path": str(path), "sha256": digest}

binary = bound_file(build / "binary.sha256", ep.__file__)
assert Path(binary["path"]).parent == build / "site/uccl"
assert ep.num_max_nvl_peers() == 4 and wrapper.ep is ep
assert "free_gpu_buffer" in (ep.Proxy.stop.__doc__ or "")
paths = {Path(line.split()[-1]).resolve() for line in Path("/proc/self/maps").read_text().splitlines()
         if len(line.split()) >= 6 and "/libfabric.so" in line.split()[-1]}
assert len(paths) == 1
fabric = bound_file(build / "libfabric_runtime.sha256", paths.pop())
record = {"valid_build_verification": True, "build_stage": "compiled_imported_source_and_runtime_verified",
          "build_dir": str(build), "job": os.environ["SLURM_JOB_ID"], "binary": binary,
          "libfabric": fabric, "source_identity": verify_uccl_source(build),
          "source_identity_manifest_sha256": hashlib.sha256((build / "source_identity.json").read_bytes()).hexdigest(),
          "num_max_nvl_peers": ep.num_max_nvl_peers(), "owning_default_stop_api_present": True,
          "torch_version": str(torch.__version__), "LD_LIBRARY_PATH": os.environ["LD_LIBRARY_PATH"],
          "claim_limit": "Fresh approved-patch build/import/runtime binding. Distributed cleanup and training remain untested."}
with (build / "verification.json").open("x") as handle:
    json.dump(record, handle, indent=2, sort_keys=True)
print(json.dumps(record), flush=True)
PY
    sha256sum -c "$AMPA_UCCL_BUILD_ROOT/source_identity.sha256" "$AMPA_UCCL_BUILD_ROOT/build-source.sha256"
    printf "UCCL_BUILD_COMPLETE\n"
    '
