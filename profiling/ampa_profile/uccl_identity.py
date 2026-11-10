"""Bind the opt-in native UCCL backend to its reviewed local build.

Startup and shutdown evidence stays outside optimizer-step timing. Native
endpoint logs must separately prove transport selection. These records do not
turn requested CXI settings or logical delivery counts into wire measurements.
"""
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import socket

from tools.uccl_ep_source_identity import verify_uccl_source

_DEEPEP_TRANSIENT_FIELDS = (
    "token_indices", "token_probs", "dispatched_indices", "dispatched_probs",
    "dispatched_routing_map", "reversed_mapping_for_combine", "pad_offsets", "tokens_per_expert",
)
_TRANSPORT_ENVIRONMENT_KEYS = (
    "FI_MR_CACHE_MONITOR", "FI_MR_CUDA_CACHE_MONITOR_ENABLED",
    "FI_MR_CACHE_MAX_COUNT", "FI_MR_CACHE_MAX_SIZE",
    "FI_LOG_LEVEL", "FI_LOG_SUBSYS",
)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _manifest_file(manifest, actual):
    """Require one exact regular binary and its recorded hash."""
    lines = Path(manifest).read_text().splitlines()
    _require(len(lines) == 1, f"expected one build binary: {manifest}")
    fields = lines[0].split(maxsplit=1)
    _require(len(fields) == 2 and len(fields[0]) == 64, "malformed build hash manifest")
    path = Path(fields[1].lstrip(" *"))
    _require(path.is_absolute() and path.resolve() == Path(actual).resolve(),
             "loaded backend library differs from reviewed build path")
    _require(path.is_file() and _sha256(actual) == fields[0], "backend library hash mismatch")
    return {"path": str(path.resolve()), "sha256": fields[0]}


def _loaded_fabric():
    paths = {Path(line.split()[-1]).resolve()
             for line in Path("/proc/self/maps").read_text().splitlines()
             if len(line.split()) >= 6 and "/libfabric.so" in line.split()[-1]}
    _require(len(paths) == 1, "expected exactly one mapped libfabric runtime")
    path = paths.pop()
    return {"path": str(path), "sha256": _sha256(path)}


def _deepep_managers(model_chunks):
    """Validate all final native managers before dropping any transient state."""
    from megatron.core.transformer.moe.token_dispatcher import MoEFlexTokenDispatcher, _DeepepManager

    managers = {}
    for chunk in model_chunks:
        for module in chunk.modules():
            if not (hasattr(module, "local_expert_indices") and hasattr(module, "token_dispatcher")):
                continue
            layer = module.layer_number
            dispatcher = module.token_dispatcher
            manager = getattr(dispatcher, "_comm_manager", None)
            _require(type(layer) is int and layer in range(1, 49) and layer not in managers,
                     "UCCL cleanup requires48 distinct MoE layer identities")
            _require(type(dispatcher) is MoEFlexTokenDispatcher and type(manager) is _DeepepManager,
                     "UCCL cleanup requires the existing native Flex/DeepEP managers")
            _require(manager.handle is None, "UCCL cleanup cannot release a live forward handle")
            _require(all(hasattr(manager, name) for name in _DEEPEP_TRANSIENT_FIELDS),
                     "UCCL cleanup lacks a required final native transient field")
            managers[layer] = manager
    _require(set(managers) == set(range(1, 49)), "UCCL cleanup did not reach all48 MoE layers")
    return managers


def _release_deepep_transients(managers):
    """Drop only known final-step caches; keep model and optimizer state intact."""
    cleared = 0
    for manager in managers.values():
        for name in _DEEPEP_TRANSIENT_FIELDS:
            setattr(manager, name, None)
            cleared += 1
    return cleared


def make_uccl_identity_callback(*, run_dir, build_dir):
    """Use only with the existing native Flex/DeepEP training adapter."""
    from megatron.bridge.training.callbacks import Callback

    run_dir, build_dir = Path(run_dir), Path(build_dir).resolve()

    class UcclIdentity(Callback):
        def __init__(self):
            self.rank = None
            self.started = False
            self.used = False
            self.cleanup_completed = False

        def _write(self, suffix, record):
            destination = run_dir / "raw" / f"uccl_backend_rank-{self.rank}_{suffix}.json"
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("x") as handle:
                json.dump(record, handle, indent=2, sort_keys=True)

        def on_train_start(self, context):
            import torch
            import torch.distributed as dist
            import uccl.ep as ep
            import deep_ep.buffer as wrapper
            from megatron.core.transformer.moe import fused_a2a

            self.rank = dist.get_rank()
            _require(dist.get_world_size() == 16 and ep.num_max_nvl_peers() == 4,
                     "UCCL training identity requires EP16 with four local peers")
            _require(wrapper.ep is ep and fused_a2a.HAVE_DEEP_EP
                     and fused_a2a.Buffer is wrapper.Buffer,
                     "native Flex adapter did not bind the UCCL compatibility Buffer")
            _require(os.environ.get("UCCL_EP_TRANSPORT") == "cxi"
                     and os.environ.get("CUDA_DEVICE_ORDER") == "PCI_BUS_ID",
                     "UCCL launch requires explicit CXI and PCI_BUS_ID ordering")
            current = torch.cuda.current_device()
            local_rank = int(os.environ["LOCAL_RANK"])
            physical_rank = ep.get_physical_gpu_rank(current)
            _require(current == local_rank == physical_rank and local_rank in range(4),
                     "UCCL local worker, selected CUDA ordinal and physical GPU ordinal differ")
            _require(os.environ.get("EFA_HOME") == str(build_dir / "not-efa")
                     and not (build_dir / "not-efa").exists(), "EFA autodetection must remain disabled")
            binary = Path(ep.__file__).resolve()
            _require(binary.suffix == ".so" and binary.parent == build_dir / "site/uccl",
                     "UCCL extension is outside the isolated reviewed build")
            binary_record = _manifest_file(build_dir / "binary.sha256", binary)
            source = build_dir / "uccl"
            source_identity = verify_uccl_source(build_dir)
            wrappers = {}
            for name in ("deep_ep", "deep_ep.buffer", "deep_ep.utils"):
                path = Path(inspect.getfile(importlib.import_module(name))).resolve()
                leaf = "__init__.py" if name == "deep_ep" else name.rsplit(".", 1)[1] + ".py"
                expected = source / "ep/deep_ep_wrapper/deep_ep" / leaf
                _require(_sha256(path) == _sha256(expected), "official UCCL wrapper bytes differ")
                wrappers[name] = {"path": str(path), "sha256": _sha256(path)}
            fabric = _loaded_fabric()
            expected_fabric = _manifest_file(build_dir / "libfabric_runtime.sha256", fabric["path"])
            _require(fabric == expected_fabric, "loaded fabric runtime differs from build")
            native = Path(inspect.getfile(fused_a2a)).resolve()
            self._write("startup", {"valid_identity": True, "rank": self.rank,
                "hostname": socket.gethostname(), "job": os.environ.get("SLURM_JOB_ID"),
                "backend": "uccl_ep", "source_revision": source_identity["base_revision"],
                "source_identity": source_identity, "binary": binary_record,
                "wrappers": wrappers, "libfabric": fabric,
                "requested_transport_environment": {
                    name: os.environ.get(name) for name in _TRANSPORT_ENVIRONMENT_KEYS},
                "native_adapter": {"path": str(native), "sha256": _sha256(native)},
                "gpu_ordinals": {"local_rank": local_rank, "cuda_current_device": current,
                                 "uccl_physical_gpu_rank": physical_rank},
                "mode": "normal_high_throughput", "deduplication": "token_per_destination_rank_and_node",
                "low_latency_dedup_api_exercised": False, "requested_transport": "cxi",
                "actual_transport_evidence": "requires native endpoint logs", "build_dir": str(build_dir)})
            self.started = True

        def on_train_step_end(self, context):
            if self.used:
                return
            import uccl.ep as ep
            from megatron.core.transformer.moe import fused_a2a

            buffer = fused_a2a._buffer
            _require(self.started and isinstance(buffer, fused_a2a.Buffer)
                     and isinstance(buffer.runtime, ep.Buffer), "native UCCL Buffer was not exercised")
            _require(buffer.low_latency_mode is False and buffer.group_size == 16
                     and buffer.num_nvl_bytes > 0 and buffer.num_rdma_bytes > 0,
                     "runtime Buffer differs from normal internode EP16 mode")
            self._write("runtime", {"valid_identity": True, "rank": self.rank,
                "first_completed_step": int(context.state.train_state.step),
                "buffer_type": f"{type(buffer).__module__}.{type(buffer).__name__}",
                "requested_transport_environment": {
                    name: os.environ.get(name) for name in _TRANSPORT_ENVIRONMENT_KEYS},
                "num_nvl_bytes": buffer.num_nvl_bytes, "num_rdma_bytes": buffer.num_rdma_bytes,
                "num_sms": buffer.num_sms, "low_latency_mode": buffer.low_latency_mode,
                "scratch_device": str(buffer.scratch.device),
                "scratch_is_cuda": bool(buffer.scratch.is_cuda),
                "claim_limit": "Scratch placement alone does not measure physical traffic or proxy time."})
            self.used = True

        def on_train_end(self, context):
            _require(self.started and self.used, "UCCL runtime identity is incomplete")
            import gc
            import torch
            import torch.distributed as dist
            from deep_ep.utils import destroy_uccl
            from megatron.core.transformer.moe import fused_a2a

            # Bridge fires this callback before it destroys the process groups.
            # The reviewed experiment has no post-training model evaluation.
            _require(context.state.cfg.validation.eval_iters == 0,
                     "UCCL cleanup requires no post-training MoE evaluation")
            buffer = fused_a2a._buffer
            _require(buffer is not None and not buffer.explicitly_destroy,
                     "expected the native destructor-managed UCCL Buffer")
            managers = _deepep_managers(context.model)
            torch.cuda.synchronize()
            dist.barrier()
            cleared = _release_deepep_transients(managers)
            gc.collect()
            # record_stream() can defer allocator events until tensor release.
            # Drain those caches while the native communication stream lives.
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            destroy_uccl(buffer.proxies, buffer.workers)
            runtime = buffer.runtime
            buffer.runtime = None
            del runtime
            gc.collect()
            # Keep backing scratch alive until every peer closes native IPC.
            dist.barrier()
            fused_a2a._buffer = None
            del buffer
            gc.collect()
            self._write("cleanup", {"valid_cleanup": True, "rank": self.rank,
                "official_destroy_uccl_returned": True, "native_runtime_released_before_scratch": True,
                "transient_metadata_layers": len(managers), "transient_attributes_cleared": cleared,
                "known_transient_fields": list(_DEEPEP_TRANSIENT_FIELDS),
                "allocator_drained_before_native_stream_release": True,
                "cleanup_limit": "The official helper suppresses some stop/unregister exceptions.",
                "peer_closure_barrier_completed": True, "process_groups_still_alive": dist.is_initialized()})
            self.cleanup_completed = True

    return UcclIdentity()
