#!/usr/bin/env python3
"""Emit non-invasive GPU-placement facts for one Slurm task.

This diagnostic is deliberately separate from model loading.  It is run only
inside the project Slurm container to establish which physical-GPU identity
source remains valid after task-level CUDA remapping.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
from typing import Any
import uuid


def _command(*args: str) -> dict[str, object]:
    """Return a bounded, JSON-safe subprocess result without interpreting it."""
    try:
        completed = subprocess.run(args, check=False, capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError) as error:
        return {"command": list(args), "error": f"{type(error).__name__}: {error}"}
    return {
        "command": list(args),
        "returncode": completed.returncode,
        "stdout": completed.stdout.splitlines(),
        "stderr": completed.stderr.splitlines(),
    }


def _property_value(properties: Any, name: str) -> object:
    """Read a potential device-property field without relying on one PyTorch version."""
    try:
        value = getattr(properties, name)
    except (AttributeError, RuntimeError):
        return None
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _cuda_driver_identity() -> dict[str, object]:
    """Read the UUID of CUDA's already-selected task device, or report why not."""
    try:
        import cuda.bindings.driver as cuda_driver
        import cuda.bindings.runtime as cuda_runtime
    except ImportError:
        try:
            import cuda.cuda as cuda_driver
            import cuda.cudart as cuda_runtime
        except ImportError as error:
            return {"error": f"cuda-python bindings unavailable: {error}"}
    try:
        runtime_status, device_id = cuda_runtime.cudaGetDevice()
        if runtime_status != cuda_runtime.cudaError_t.cudaSuccess:
            return {"error": f"cudaGetDevice failed: {runtime_status!r}"}
        driver_status, device_uuid = cuda_driver.cuDeviceGetUuid(device_id)
        if driver_status != cuda_driver.CUresult.CUDA_SUCCESS:
            return {"device_id": device_id, "error": f"cuDeviceGetUuid failed: {driver_status!r}"}
        return {"device_id": device_id, "uuid": f"GPU-{uuid.UUID(bytes=device_uuid.bytes)}"}
    except (AttributeError, RuntimeError, TypeError, ValueError) as error:
        return {"error": f"CUDA driver identity failed: {type(error).__name__}: {error}"}


def main() -> None:
    """Initialize only the assigned CUDA device and print one JSON document."""
    import torch

    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    slurm_keys = (
        "SLURM_JOB_ID",
        "SLURM_PROCID",
        "SLURM_LOCALID",
        "SLURM_NODEID",
        "SLURM_STEP_GPUS",
        "SLURM_JOB_GPUS",
        "CUDA_VISIBLE_DEVICES",
        "LOCAL_RANK",
    )
    record = {
        "schema_version": 1,
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "environment": {key: os.environ.get(key) for key in slurm_keys},
        "torch": {
            "version": torch.__version__,
            "device_count": torch.cuda.device_count(),
            "current_device": device,
            "name": torch.cuda.get_device_name(device),
            "properties": {
                field: _property_value(properties, field)
                for field in ("uuid", "_uuid", "pci_bus_id", "pci_device_id", "name", "total_memory")
            },
        },
        "cuda_driver": _cuda_driver_identity(),
        "nvidia_smi": {
            "all_gpus": _command(
                "nvidia-smi", "--query-gpu=index,uuid,pci.bus_id", "--format=csv,noheader"
            ),
            "compute_apps": _command(
                "nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"
            ),
        },
    }
    print(json.dumps(record, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
