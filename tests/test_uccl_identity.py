"""CPU-only tests for UCCL build identity and first native Buffer evidence."""

from contextlib import contextmanager, nullcontext
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import types
import unittest
from unittest.mock import Mock, patch

from profiling.ampa_profile.uccl_identity import (
    _loaded_fabric, _manifest_file, make_uccl_identity_callback,
)


def _write(path, data=b"reviewed bytes"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _manifest(path, binary):
    digest = hashlib.sha256(binary.read_bytes()).hexdigest()
    path.write_text(f"{digest}  {binary}\n")
    return {"path": str(binary.resolve()), "sha256": digest}


class UcclIdentityHelpersTest(unittest.TestCase):
    def test_manifest_matches_absolute_path_hash_and_canonical_symlink(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            binary = _write(root / "native.abi3.so")
            expected = _manifest(root / "binary.sha256", binary)
            alias = root / "alias.so"
            alias.symlink_to(binary)
            self.assertEqual(_manifest_file(root / "binary.sha256", alias), expected)

    def test_manifest_rejects_wrong_path_changed_bytes_and_ambiguous_records(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            binary = _write(root / "native.so")
            other = _write(root / "other.so")
            manifest = root / "binary.sha256"
            expected = _manifest(manifest, binary)
            with self.assertRaisesRegex(RuntimeError, "build path"):
                _manifest_file(manifest, other)
            binary.write_bytes(b"changed")
            with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
                _manifest_file(manifest, binary)
            for content in ("", f"{expected['sha256']}  {binary}\n" * 2,
                            f"{expected['sha256']}  native.so\n", "invalid hash record\n"):
                with self.subTest(content=content):
                    manifest.write_text(content)
                    with self.assertRaises(RuntimeError):
                        _manifest_file(manifest, binary)

    def test_loaded_fabric_deduplicates_mappings_and_hashes_canonical_file(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            fabric = _write(root / "libfabric.so.1.0")
            alias = root / "libfabric.so.1"
            alias.symlink_to(fabric)
            maps = (f"0000-1000 r--p 0000 00:00 1 {alias}\n"
                    f"1000-2000 r-xp 1000 00:00 1 {fabric}\n"
                    "2000-3000 rw-p 0000 00:00 2 /other/libunrelated.so\n")
            with patch.object(Path, "read_text", return_value=maps):
                self.assertEqual(_loaded_fabric(), {"path": str(fabric.resolve()),
                    "sha256": hashlib.sha256(fabric.read_bytes()).hexdigest()})

    def test_loaded_fabric_rejects_absent_and_multiple_runtimes(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = _write(root / "first/libfabric.so.1")
            second = _write(root / "second/libfabric.so.1")
            maps = f"0-1 r-xp 0 00:00 1 {first}\n1-2 r-xp 0 00:00 2 {second}\n"
            for content in ("", maps):
                with self.subTest(maps=content), patch.object(Path, "read_text", return_value=content):
                    with self.assertRaisesRegex(RuntimeError, "exactly one"):
                        _loaded_fabric()


class UcclIdentityCallbackTest(unittest.TestCase):
    @contextmanager
    def fixture(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir, build = root / "run", root / "build"
            events = []
            native_state = {"stream_alive": True}
            binary = _write(build / "site/uccl/ep.abi3.so", b"extension")
            _manifest(build / "binary.sha256", binary)
            fabric = _write(root / "runtime/libfabric.so.1", b"fabric runtime")
            fabric_record = _manifest(build / "libfabric_runtime.sha256", fabric)
            names = (
                "torch", "torch.distributed", "uccl", "uccl.ep", "deep_ep",
                "deep_ep.buffer", "deep_ep.utils", "megatron", "megatron.core",
                "megatron.core.transformer", "megatron.core.transformer.moe",
                "megatron.core.transformer.moe.fused_a2a", "megatron.core.transformer.moe.token_dispatcher", "megatron.bridge",
                "megatron.bridge.training", "megatron.bridge.training.callbacks",
            )
            modules = {name: types.ModuleType(name) for name in names}
            for name, module in modules.items():
                parent, _, child = name.rpartition(".")
                if parent in modules:
                    setattr(modules[parent], child, module)
            modules["megatron.bridge.training.callbacks"].Callback = object
            dist = modules["torch.distributed"]
            dist.get_rank, dist.get_world_size = Mock(return_value=3), Mock(return_value=16)
            dist.barrier = Mock(side_effect=lambda: events.append("barrier"))
            dist.is_initialized = Mock(return_value=True)
            modules["torch"].cuda = types.SimpleNamespace(
                synchronize=Mock(side_effect=lambda: events.append("synchronize")),
                empty_cache=Mock(side_effect=lambda: events.append("empty_cache")),
                current_device=Mock(return_value=3))
            ep, wrapper = modules["uccl.ep"], modules["deep_ep.buffer"]
            ep.__file__ = str(binary)
            ep.num_max_nvl_peers = Mock(return_value=4)
            ep.get_physical_gpu_rank = Mock(return_value=3)

            class NativeRuntime:
                def __del__(self):
                    native_state["stream_alive"] = False
                    events.append("runtime_released")

            ep.Buffer = NativeRuntime
            wrapper.ep = ep
            for name in ("deep_ep", "deep_ep.buffer", "deep_ep.utils"):
                leaf = "__init__.py" if name == "deep_ep" else name.rsplit(".", 1)[1] + ".py"
                expected = build / "uccl/ep/deep_ep_wrapper/deep_ep" / leaf
                actual = root / "loaded_wrappers" / leaf
                _write(expected, name.encode())
                modules[name].__file__ = str(_write(actual, name.encode()))

            class Scratch:
                device, is_cuda = "cuda:3", True

                def __del__(self):
                    events.append("scratch_released")

            class Buffer:
                def __init__(self):
                    self.runtime = ep.Buffer()
                    self.low_latency_mode, self.group_size = False, 16
                    self.num_nvl_bytes, self.num_rdma_bytes, self.num_sms = 1024, 2048, 20
                    self.scratch = Scratch()
                    self.explicitly_destroy = False
                    self.proxies, self.workers = object(), object()

            wrapper.Buffer = Buffer
            fused = modules["megatron.core.transformer.moe.fused_a2a"]
            fused.__file__ = str(_write(root / "native/fused_a2a.py", b"unchanged native adapter"))
            fused.Buffer, fused.HAVE_DEEP_EP, fused._buffer = Buffer, True, None
            destroy = modules["deep_ep.utils"].destroy_uccl = Mock(
                side_effect=lambda *args: events.append("destroy_uccl"))

            class RecordedTransient:
                def __del__(self):
                    events.append("transient_released" if native_state["stream_alive"] else "late_transient_release")

            class _DeepepManager:
                def __init__(self):
                    self.handle = None
                    self.token_indices, self.token_probs = RecordedTransient(), RecordedTransient()
                    self.dispatched_indices = self.dispatched_probs = None
                    self.dispatched_routing_map, self.reversed_mapping_for_combine = RecordedTransient(), RecordedTransient()
                    self.pad_offsets, self.tokens_per_expert = None, object()
                    self.config, self.group = object(), object()

            class MoEFlexTokenDispatcher:
                def __init__(self):
                    self._comm_manager = _DeepepManager()

            token_dispatcher = modules["megatron.core.transformer.moe.token_dispatcher"]
            token_dispatcher.MoEFlexTokenDispatcher, token_dispatcher._DeepepManager = MoEFlexTokenDispatcher, _DeepepManager
            layers = [types.SimpleNamespace(layer_number=layer, local_expert_indices=list(range(8)),
                token_dispatcher=MoEFlexTokenDispatcher(), weight=object()) for layer in range(1, 49)]
            managers = [layer.token_dispatcher._comm_manager for layer in layers]
            context = types.SimpleNamespace(
                model=[types.SimpleNamespace(modules=lambda: layers)], optimizer=types.SimpleNamespace(state=object()),
                state=types.SimpleNamespace(train_state=types.SimpleNamespace(step=1),
                    cfg=types.SimpleNamespace(validation=types.SimpleNamespace(eval_iters=0))))
            environment = {"UCCL_EP_TRANSPORT": "cxi", "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                           "EFA_HOME": str(build / "not-efa"), "SLURM_JOB_ID": "mock-job",
                           "LOCAL_RANK": "3"}
            source_identity = {"base_revision": "c8fd14e93f2106b69a0d30e0f78c12a52e07a181",
                "patch_sha256": "749a28bdb82d25ba92bace55ad6d99bafc1075280bb5466e12f378f5a53a21b2",
                "source_diff_sha256": "d" * 64,
                "modified_files": {name: "a" * 64 for name in (
                    "ep/include/uccl_proxy.hpp", "ep/src/uccl_proxy.cpp", "ep/src/uccl_ep.cc", "ep/bench/utils.py")}}
            source_verifier = Mock(return_value=source_identity)
            with patch.dict("sys.modules", modules), patch.dict("os.environ", environment), \
                    patch("profiling.ampa_profile.uccl_identity.verify_uccl_source", source_verifier), \
                    patch("profiling.ampa_profile.uccl_identity._loaded_fabric", return_value=fabric_record), \
                    patch("gc.collect", side_effect=lambda: events.append("collect")):
                callback = make_uccl_identity_callback(run_dir=run_dir, build_dir=build)
                self.assertIs(callback.cleanup_completed, False)
                yield types.SimpleNamespace(root=root, run_dir=run_dir, build=build, ep=ep,
                    wrapper=wrapper, fused=fused, modules=modules, source_verifier=source_verifier,
                    source_identity=source_identity, callback=callback,
                    context=context, Buffer=Buffer, fabric=fabric, fabric_record=fabric_record,
                    events=events, destroy=destroy, dist=dist, torch=modules["torch"],
                    layers=layers, managers=managers, native_state=native_state)

    def assert_no_cleanup(self, fixture):
        fixture.torch.cuda.synchronize.assert_not_called()
        fixture.torch.cuda.empty_cache.assert_not_called()
        fixture.dist.barrier.assert_not_called()
        fixture.destroy.assert_not_called()
        self.assertIs(fixture.callback.cleanup_completed, False)
        self.assertFalse((fixture.run_dir / "raw/uccl_backend_rank-3_cleanup.json").exists())

    def test_startup_and_first_buffer_record_truthful_identity_once(self):
        with self.fixture() as fixture:
            fixture.callback.on_train_start(fixture.context)
            startup = json.loads((fixture.run_dir / "raw/uccl_backend_rank-3_startup.json").read_text())
            self.assertIs(startup["valid_identity"], True)
            self.assertEqual(startup["source_revision"], fixture.source_identity["base_revision"])
            self.assertEqual(startup["source_identity"], fixture.source_identity)
            self.assertEqual(startup["mode"], "normal_high_throughput")
            self.assertEqual(startup["gpu_ordinals"], {"local_rank": 3,
                "cuda_current_device": 3, "uccl_physical_gpu_rank": 3})
            fixture.ep.get_physical_gpu_rank.assert_called_once_with(3)
            self.assertIs(startup["low_latency_dedup_api_exercised"], False)
            self.assertEqual(startup["actual_transport_evidence"], "requires native endpoint logs")
            fixture.fused._buffer = fixture.Buffer()
            fixture.callback.on_train_step_end(fixture.context)
            runtime_path = fixture.run_dir / "raw/uccl_backend_rank-3_runtime.json"
            runtime = json.loads(runtime_path.read_text())
            self.assertEqual(runtime["first_completed_step"], 1)
            self.assertIs(runtime["low_latency_mode"], False)
            self.assertEqual((runtime["num_nvl_bytes"], runtime["num_rdma_bytes"]), (1024, 2048))
            self.assertIs(runtime["scratch_is_cuda"], True)
            fixture.context.state.train_state.step = 2
            fixture.callback.on_train_step_end(fixture.context)
            self.assertEqual(json.loads(runtime_path.read_text()), runtime)
            fixture.callback.on_train_end(fixture.context)
            self.assertIs(fixture.callback.cleanup_completed, True)
            fixture.source_verifier.assert_called_once_with(fixture.build)

    def test_wrong_extension_path_or_hash_rejects_startup_and_incomplete_end(self):
        for fault in ("path", "hash"):
            with self.subTest(fault=fault), self.fixture() as fixture:
                if fault == "path":
                    fixture.ep.__file__ = str(_write(fixture.root / "unreviewed/ep.so", b"extension"))
                else:
                    Path(fixture.ep.__file__).write_bytes(b"changed extension")
                with self.assertRaises(RuntimeError):
                    fixture.callback.on_train_start(fixture.context)
                with self.assertRaisesRegex(RuntimeError, "incomplete"):
                    fixture.callback.on_train_end(fixture.context)
                self.assert_no_cleanup(fixture)
                self.assertFalse((fixture.run_dir / "raw/uccl_backend_rank-3_startup.json").exists())

    def test_wrong_wrapper_binding_or_bytes_rejects_startup(self):
        for fault in ("ep", "buffer", "available", "bytes"):
            with self.subTest(fault=fault), self.fixture() as fixture:
                if fault == "ep":
                    fixture.wrapper.ep = object()
                elif fault == "buffer":
                    fixture.fused.Buffer = object()
                elif fault == "available":
                    fixture.fused.HAVE_DEEP_EP = False
                else:
                    Path(fixture.wrapper.__file__).write_bytes(b"changed wrapper")
                with self.assertRaises(RuntimeError):
                    fixture.callback.on_train_start(fixture.context)
                with self.assertRaisesRegex(RuntimeError, "incomplete"):
                    fixture.callback.on_train_end(fixture.context)
                self.assert_no_cleanup(fixture)

    def test_wrong_transport_order_or_loaded_fabric_rejects_startup(self):
        for fault in ("transport", "ordering", "fabric"):
            with self.subTest(fault=fault), self.fixture() as fixture:
                if fault == "fabric":
                    other = _write(fixture.root / "other/libfabric.so.1", b"fabric runtime")
                    value = {"path": str(other), "sha256": fixture.fabric_record["sha256"]}
                    failure_patch = patch("profiling.ampa_profile.uccl_identity._loaded_fabric", return_value=value)
                else:
                    key = "UCCL_EP_TRANSPORT" if fault == "transport" else "CUDA_DEVICE_ORDER"
                    failure_patch = patch.dict("os.environ", {key: "unexpected"})
                with failure_patch, self.assertRaises(RuntimeError):
                    fixture.callback.on_train_start(fixture.context)
                with self.assertRaisesRegex(RuntimeError, "incomplete"):
                    fixture.callback.on_train_end(fixture.context)
                self.assert_no_cleanup(fixture)

    def test_shared_source_verifier_rejection_propagates_without_startup_or_cleanup(self):
        for reason in ("unapproved base", "unapproved source change", "wrong approved patch"):
            with self.subTest(reason=reason), self.fixture() as fixture:
                error = RuntimeError(reason)
                fixture.source_verifier.side_effect = error
                with self.assertRaises(RuntimeError) as raised:
                    fixture.callback.on_train_start(fixture.context)
                self.assertIs(raised.exception, error)
                fixture.source_verifier.assert_called_once_with(fixture.build)
                self.assertFalse((fixture.run_dir / "raw/uccl_backend_rank-3_startup.json").exists())
                self.assert_no_cleanup(fixture)

    def test_worker_cuda_and_physical_ordinals_must_match_inside_four_peers(self):
        for fault in ("worker", "cuda", "physical", "range"):
            with self.subTest(fault=fault), self.fixture() as fixture:
                environment = {}
                if fault == "worker":
                    environment["LOCAL_RANK"] = "2"
                elif fault == "cuda":
                    fixture.torch.cuda.current_device.return_value = 2
                elif fault == "physical":
                    fixture.ep.get_physical_gpu_rank.return_value = 2
                else:
                    environment["LOCAL_RANK"] = "4"
                    fixture.torch.cuda.current_device.return_value = 4
                    fixture.ep.get_physical_gpu_rank.return_value = 4
                with patch.dict("os.environ", environment), self.assertRaisesRegex(RuntimeError, "ordinal"):
                    fixture.callback.on_train_start(fixture.context)
                with self.assertRaisesRegex(RuntimeError, "incomplete"):
                    fixture.callback.on_train_end(fixture.context)
                self.assert_no_cleanup(fixture)

    def test_invalid_first_buffer_or_mode_rejects_completion_and_end(self):
        for fault in ("missing", "runtime", "low_latency_mode", "group_size", "num_nvl_bytes", "num_rdma_bytes"):
            with self.subTest(fault=fault), self.fixture() as fixture:
                fixture.callback.on_train_start(fixture.context)
                buffer = fixture.Buffer()
                if fault == "runtime":
                    buffer.runtime = object()
                elif fault == "low_latency_mode":
                    buffer.low_latency_mode = True
                elif fault == "group_size":
                    buffer.group_size = 8
                elif fault in ("num_nvl_bytes", "num_rdma_bytes"):
                    setattr(buffer, fault, 0)
                fixture.fused._buffer = None if fault == "missing" else buffer
                with self.assertRaises(RuntimeError):
                    fixture.callback.on_train_step_end(fixture.context)
                with self.assertRaisesRegex(RuntimeError, "incomplete"):
                    fixture.callback.on_train_end(fixture.context)
                self.assert_no_cleanup(fixture)
                self.assertFalse((fixture.run_dir / "raw/uccl_backend_rank-3_runtime.json").exists())

    def test_success_cleanup_releases_runtime_before_peer_barrier_and_scratch(self):
        with self.fixture() as fixture:
            fixture.callback.on_train_start(fixture.context)
            fixture.fused._buffer = fixture.Buffer()
            proxies, workers = fixture.fused._buffer.proxies, fixture.fused._buffer.workers
            parameters = [layer.weight for layer in fixture.layers]
            persistent_manager_fields = [(manager.config, manager.group) for manager in fixture.managers]
            optimizer_state = fixture.context.optimizer.state
            fixture.callback.on_train_step_end(fixture.context)
            self.assertIs(fixture.callback.cleanup_completed, False)
            fixture.events.clear()
            original_write = fixture.callback._write

            def checked_write(suffix, record):
                self.assertEqual(suffix, "cleanup")
                self.assertIs(fixture.callback.cleanup_completed, False)
                original_write(suffix, record)
                self.assertIs(fixture.callback.cleanup_completed, False)

            with patch.object(fixture.callback, "_write", side_effect=checked_write):
                fixture.callback.on_train_end(fixture.context)
            self.assertIs(fixture.callback.cleanup_completed, True)
            self.assertEqual([event for event in fixture.events if event != "transient_released"],
                ["synchronize", "barrier", "collect", "synchronize", "empty_cache", "destroy_uccl",
                "runtime_released", "collect", "barrier", "scratch_released", "collect"])
            self.assertEqual(fixture.events.count("transient_released"), 48 * 4)
            self.assertNotIn("late_transient_release", fixture.events)
            self.assertLess(max(index for index, event in enumerate(fixture.events) if event == "transient_released"),
                            fixture.events.index("empty_cache"))
            self.assertEqual([layer.weight for layer in fixture.layers], parameters)
            self.assertEqual([(manager.config, manager.group) for manager in fixture.managers], persistent_manager_fields)
            self.assertIs(fixture.context.optimizer.state, optimizer_state)
            for manager in fixture.managers:
                for name in ("token_indices", "token_probs", "dispatched_indices", "dispatched_probs",
                             "dispatched_routing_map", "reversed_mapping_for_combine", "pad_offsets", "tokens_per_expert"):
                    self.assertIsNone(getattr(manager, name))
            fixture.destroy.assert_called_once_with(proxies, workers)
            self.assertIsNone(fixture.fused._buffer)
            record = json.loads((fixture.run_dir / "raw/uccl_backend_rank-3_cleanup.json").read_text())
            for name in ("valid_cleanup", "official_destroy_uccl_returned", "native_runtime_released_before_scratch",
                         "peer_closure_barrier_completed", "process_groups_still_alive", "allocator_drained_before_native_stream_release"):
                self.assertIs(record[name], True)
            self.assertEqual(record["transient_metadata_layers"], 48)
            self.assertEqual(record["transient_attributes_cleared"], 48 * 8)

    def test_cleanup_completion_stays_false_when_shutdown_or_final_evidence_fails(self):
        for fault in ("synchronize", "proxy_stop", "peer_barrier", "evidence_write"):
            with self.subTest(fault=fault), self.fixture() as fixture:
                fixture.callback.on_train_start(fixture.context)
                fixture.fused._buffer = fixture.Buffer()
                fixture.callback.on_train_step_end(fixture.context)
                error = RuntimeError("cleanup did not finish")
                if fault == "synchronize":
                    fixture.torch.cuda.synchronize.side_effect = error
                elif fault == "proxy_stop":
                    fixture.destroy.side_effect = error
                elif fault == "peer_barrier":
                    fixture.dist.barrier.side_effect = [None, error]
                writer = patch.object(fixture.callback, "_write", side_effect=error) if fault == "evidence_write" else nullcontext()
                with writer:
                    with self.assertRaises(RuntimeError) as raised:
                        fixture.callback.on_train_end(fixture.context)
                self.assertIs(raised.exception, error)
                self.assertIs(fixture.callback.cleanup_completed, False)
                self.assertFalse((fixture.run_dir / "raw/uccl_backend_rank-3_cleanup.json").exists())

    def test_cleanup_validates_every_manager_and_handle_before_any_transient_or_cuda_mutation(self):
        for fault in ("coverage", "live_handle", "missing_field", "wrong_type"):
            with self.subTest(fault=fault), self.fixture() as fixture:
                fixture.callback.on_train_start(fixture.context)
                fixture.fused._buffer = fixture.Buffer()
                fixture.callback.on_train_step_end(fixture.context)
                if fault == "coverage":
                    fixture.context.model[0].modules = lambda: fixture.layers[:-1]
                elif fault == "live_handle":
                    fixture.managers[-1].handle = object()
                elif fault == "missing_field":
                    del fixture.managers[-1].token_probs
                else:
                    fixture.layers[-1].token_dispatcher._comm_manager = object()
                identities = [id(manager.token_indices) for manager in fixture.managers]
                fixture.events.clear()
                with self.assertRaises(RuntimeError):
                    fixture.callback.on_train_end(fixture.context)
                self.assertEqual([id(manager.token_indices) for manager in fixture.managers], identities)
                self.assertNotIn("transient_released", fixture.events)
                self.assert_no_cleanup(fixture)

    def test_cleanup_rejects_later_evaluation_or_explicit_destroy_before_cuda_work(self):
        for fault in ("evaluation", "explicit_destroy"):
            with self.subTest(fault=fault), self.fixture() as fixture:
                fixture.callback.on_train_start(fixture.context)
                fixture.fused._buffer = fixture.Buffer()
                fixture.callback.on_train_step_end(fixture.context)
                if fault == "evaluation":
                    fixture.context.state.cfg.validation.eval_iters = 1
                else:
                    fixture.fused._buffer.explicitly_destroy = True
                with self.assertRaises(RuntimeError):
                    fixture.callback.on_train_end(fixture.context)
                self.assert_no_cleanup(fixture)


if __name__ == "__main__":
    unittest.main()
