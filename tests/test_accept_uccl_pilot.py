"""CPU consumer fixtures for accepted upstream pilot-validation records.

Trace byte fixtures deliberately need no trace parser. These tests exercise
the proof consumer and its bindings, not native training or trace validity.
"""
import copy
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import uuid
import types

import yaml

from analysis import accept_uccl_pilot as pilot
from analysis.summarize_training_evidence import summarize
from analysis.validate_ep16_received_routes import validate_run as validate_received_run
from profiling.ampa_profile import provenance
from test_ep16_received_routes import fixture as routing_fixture
from tools.uccl_ep_log_proof import inspect_node
from tools.uccl_ep_source_identity import BASE_REVISION, PATCH_FILES, PATCH_SHA256


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def runtime_topology_fixture():
    """Exercise the real producer with bounded CP1 parallel-state answers."""
    megatron, core = types.ModuleType("megatron"), types.ModuleType("megatron.core")
    core.parallel_state = types.SimpleNamespace(
        get_tensor_model_parallel_world_size=lambda: 1,
        get_pipeline_model_parallel_world_size=lambda: 1,
        get_context_parallel_world_size=lambda: 1,
        get_expert_model_parallel_world_size=lambda: 16,
        get_expert_tensor_parallel_world_size=lambda: 1,
        get_data_parallel_world_size=lambda with_context_parallel=False: 16,
        get_expert_data_parallel_world_size=lambda: 1,
    )
    megatron.core = core
    with patch.dict(sys.modules, {"megatron": megatron, "megatron.core": core}):
        return provenance.validate_runtime_topology(
            provenance._runtime_topology(types.SimpleNamespace(get_world_size=lambda: 16)),
            expected=provenance._EP16_RUNTIME_TOPOLOGIES["cp1_ep16"])


@contextmanager
def evidence(directory):
    root = Path(directory)
    sources = sorted(set(provenance._AMPA_RUNTIME_SOURCE_PATHS) | set(pilot.ANALYSIS_SOURCES) | {
        "jobs/training/uccl_ep_comparison.sbatch", "jobs/training/qwen3_uccl_ep_worker.sh"})
    for name in sources:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"reviewed source fixture: {name}\n")
    adapter = root / "Megatron-LM/megatron/core/transformer/moe/fused_a2a.py"
    adapter.parent.mkdir(parents=True, exist_ok=True)
    adapter.write_text("native adapter fixture\n")
    with patch.object(provenance, "_AMPA_RUNTIME_SOURCE_PATHS", tuple(sources)):
        runtime_sources = provenance._ampa_runtime_sources(root)
        build, orchestration, analysis = root / "runs/build", root / "runs/pilot", root / "analysis-results"
        binary, fabric = build / "site/uccl/ep.so", root / "libraries/libfabric.so.2"
        for path in (binary, fabric):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(path.name.encode())
        fabric_alias = fabric.with_name("libfabric.so.1")
        fabric_alias.symlink_to(fabric)
        (build / "binary.sha256").write_text(f"{digest(binary)}  {binary}\n")
        (build / "libfabric_runtime.sha256").write_text(f"{digest(fabric)}  {fabric_alias}\n")
        for relative in PATCH_FILES:
            path = build / "uccl" / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"verified patched source fixture: {relative}\n")
        wrappers = {}
        for name, leaf in (("deep_ep", "__init__.py"), ("deep_ep.buffer", "buffer.py"), ("deep_ep.utils", "utils.py")):
            path = build / "uccl/ep/deep_ep_wrapper/deep_ep" / leaf
            path.parent.mkdir(parents=True, exist_ok=True)
            if name == "deep_ep":
                path.write_text(name)
            else:
                target = build / "uccl/ep/bench" / leaf
                if not target.exists():
                    target.write_text(name)
                path.symlink_to(Path("../../bench") / leaf)
            wrappers[name] = {"path": str(path.resolve()), "sha256": digest(path)}
        archive = build / "source/uccl-borrowed-scratch.patch"
        archive.parent.mkdir(parents=True)
        archive.write_bytes((pilot.ROOT / "ai/qwen-uccl-ep-comparison/iteration-01/02-implement/uccl-borrowed-scratch.patch").read_bytes())
        source_identity = {"base_revision": BASE_REVISION, "patch_sha256": PATCH_SHA256,
            "source_diff_sha256": "d" * 64,
            "modified_files": {name: digest(build / "uccl" / name) for name in PATCH_FILES}}
        write(build / "source_identity.json", source_identity)
        c0 = root / "checkpoint/megatron/qwen3-30b-a3b-base-c0"
        c0.mkdir(parents=True)
        write(c0 / "ampa_c0_manifest.json", {"fixture": "immutable C0"})
        c0_hash = digest(c0 / "ampa_c0_manifest.json")
        dataset_bytes = json.dumps({"fixture": "immutable indexed data"}).encode()
        dataset_hash = hashlib.sha256(dataset_bytes).hexdigest()
        native = {"revision": "reviewed-native", "status": "dirty", "diff_sha256": "a" * 64}
        reviewed = {"valid_reviewed_inputs": True, "slurm_job_id": "training-job", "ampa_runtime_sources": runtime_sources,
            "checkpoint_payload": {"verified": True, "path": str(c0)}, "c0_identity_manifest_sha256": c0_hash,
            "dataset_manifest_sha256": dataset_hash, "native_checkouts": {"megatron_lm": native, "megatron_bridge": native}}
        write(orchestration / "reviewed_inputs.json", reviewed)
        write(orchestration / "acquisition.json", {"mode": "pilot", "topology": "cp1_ep16", "launch_policy": "torchrun_node4",
            "runtime_source_sha256": runtime_sources["sha256"], "accepted_pilot_proof": None, "job": "training-job",
            "binary_sha256": digest(binary), "uccl_source_identity": source_identity,
            "uccl_transport_policy": copy.deepcopy(pilot.UCCL_TRANSPORT_POLICY)})
        _, _, _, routing_environment = routing_fixture()
        placement = [{"rank": rank, "hostname": f"node-{rank // 4}", "slurm_node_id": str(rank // 4), "slurm_local_id": "0",
            "launch_policy": "torchrun_node4", "cuda_visible_devices": "0,1,2,3", "cuda_visible_device_count": 4,
            "local_rank": rank % 4, "local_world_size": 4, "cuda_current_device": rank % 4, "cpu_affinity": [0, 1],
            "gpu_name": "GH200", "physical_gpu_uuid": "GPU-" + str(uuid.UUID(int=rank + 1)),
            "physical_gpu_identity_source": provenance._GPU_UUID_SOURCE,
            "local_expert_indices": routing_environment["rank_placement"][rank]["local_expert_indices"],
            "process_groups": {"cp": [rank], "ep": routing_environment["rank_placement"][rank]["process_groups"]["ep"],
                               "dense_dp": list(range(16)), "dense_dp_cp": list(range(16))}}
            for rank in range(16)]
        runs = {case: root / "runs" / f"pilot-{case}-profile" for case in pilot.CASES}
        (orchestration / "cases.tsv").write_text("".join(f"{case}\tprofile\t{run}\n" for case, run in runs.items()))
        write(orchestration / "acquisition_completed.json", {"success": True, "workers_srun_returned_zero": True,
            "mode": "pilot", "job": "training-job", "runtime_source_sha256": runtime_sources["sha256"],
            "binary_sha256": digest(binary), "cases_tsv_sha256": digest(orchestration / "cases.tsv")})
        for case, run in runs.items():
            uccl, grouped = case in pilot.UCCL_CASES, case in pilot.GROUPED_CASES
            _, routing_ownership, routing_audits, _ = routing_fixture(
                "uccl_ep" if uccl else "alltoall", grouped)
            for name, payload in (("c0_manifest.json", (c0 / "ampa_c0_manifest.json").read_bytes()),
                                  ("dataset_manifest.json", dataset_bytes), ("model_manifest.json", b"{}")):
                path = run / "config" / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload)
            contract = {"communication_backend": "uccl_ep" if uccl else "alltoall", "launch_policy": "torchrun_node4",
                "sample_fingerprints": True, "routing_audit": True, "audit_received_experts": True,
                "routing": {"topk": 8, "num_groups": 4 if grouped else None, "group_topk": 2 if grouped else None},
                "baseline": {"dispatcher": "flex" if uccl else "alltoall", "capacity_drop": False},
                "parallelism": {"tp": 1, "pp": 1, "cp": 1, "ep": 16, "etp": 1, "dense_dp": 16},
                "model": {"precision": "bf16"}, "initialization": {"mode": "pretrained", "checkpoint": str(c0)},
                "c0_identity_manifest_sha256": c0_hash, "dataset": {"manifest_sha256": dataset_hash},
                "training": {"sequence_length": 4096, "micro_batch_size": 1, "global_batch_size": 64, "gradient_accumulation": 4,
                    "warmup_steps": 5, "measured_steps": 20, "completed_steps": 2, "log_interval": 1, "recompute_mode": "none"}}
            if uccl:
                contract["uccl_build_dir"] = str(build)
            write(run / "config/comparison_contract.json", contract)
            model = {"moe_router_num_groups": contract["routing"]["num_groups"], "moe_router_group_topk": contract["routing"]["group_topk"],
                "moe_token_dispatcher_type": contract["baseline"]["dispatcher"], "moe_flex_dispatcher_backend": "deepep" if uccl else None,
                "tensor_model_parallel_size": 1, "pipeline_model_parallel_size": 1, "context_parallel_size": 1,
                "expert_model_parallel_size": 16, "expert_tensor_parallel_size": 1, "seq_length": 4096, "bf16": True,
                "fp16": False, "moe_router_dtype": None, "moe_enable_deepep": False}
            configuration = {"model": model, "train": {"train_iters": 25, "exit_interval": 2, "global_batch_size": 64, "micro_batch_size": 1},
                "profiling": {"use_pytorch_profiler": False, "use_nsys_profiler": False},
                "optimizer": {"lr": 1e-6, "weight_decay": 0.1, "grad_norm_skip_threshold": math.inf},
                "logger": {"tensorboard_dir": str(run / "logs/tensorboard"), "wandb_exp_name": run.name, "wandb_save_dir": str(run / "logs/wandb")}}
            (run / "config/resolved.yaml").write_text(yaml.safe_dump(configuration))
            environment = {"slurm_job_id": "training-job", "launch_policy": "torchrun_node4", "ampa_runtime_sources": runtime_sources,
                "rank_placement": copy.deepcopy(placement), "pytorch_version": "reviewed", "cuda_version": "13.0", "nccl_version": [2, 0, 0],
                "cuda_allocator_conf": "expandable_segments:True", "container_sqsh_sha256": "f" * 64,
                "megatron_lm": native, "megatron_bridge": native, "checkpoint": str(c0), "dataset_blend_per_split": [["fixed"]],
                "parallelism": contract["parallelism"], "runtime_topology": runtime_topology_fixture(),
                "activation_recomputation": {"mode": "none"}, "hf_model_id": "Qwen", "hf_model_revision": "fixed",
                "dataset_consumer": {"fast_cache_load": True}, "reviewed_inputs_proof": {"path": str(orchestration / "reviewed_inputs.json"),
                    "sha256": digest(orchestration / "reviewed_inputs.json")}}
            write(run / "environment.json", environment)
            write(run / "config/routing_ownership.json", routing_ownership)
            write(run / "config/observation_mode.json", {"data_mode": "indexed", "lightweight_profiling": "off", "nsys_trace": False,
                "torch_trace": True, "sample_fingerprints": True, "router_observer_attached": False,
                "route_wrapper_installed": False, "dispatcher_preprocess_wrapper_installed": False})
            write(run / "config/all_step_capture.json", {"enabled": True, "steps": [0, 1], "ranks": list(range(16))})
            for rank in range(16):
                node, local = divmod(rank, 4)
                stdout = f"worker {rank}\n"
                if rank == 15:
                    for step in (1, 2):
                        stdout += (f"iteration {step} / 25 | elapsed time per iteration (ms): 100.0 | consumed samples: {step * 64} | "
                            "learning rate: 1e-6 | global batch size: 64 | lm loss: 2.0 | load_balancing_loss: 2.0 | grad norm: 0.5 | "
                            "number of skipped iterations: 0 | number of nan iterations: 0 |\n")
                stderr = ""
                if uccl:
                    stderr = f"Selected CXI domain cxi{local} for local_rank {local}\n[CXI] libfabric/CXI endpoint initialized\n"
                    for name, text in (("stdout", stdout), ("stderr", stderr)):
                        path = run / "logs" / f"torchrun-node-{node}" / "job/attempt_0" / str(local) / f"{name}.log"
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_text(text)
                path = run / "logs" / f"rank-{rank}.out"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(stdout + stderr)
                batches = [{"step": step, "batches": [{"tokens": {"sha256": hashlib.sha256(f"{rank}:{step}:{micro}".encode()).hexdigest(),
                    "shape": [1, 4096], "dtype": "torch.int64"}} for micro in range(4)]} for step in (0, 1)]
                write(run / "raw" / f"sample_fingerprint_rank-{rank}.json", {"schema_version": 1, "kind": "consumed_cpu_token_batch_sha256",
                    "rank": rank, "steps": batches})
                audit = run / "raw" / f"routing_audit_rank-{rank}.json"
                write(audit, routing_audits[rank])
                validations, aggregates = [], []
                for step in (0, 1):
                    trace, call = pilot._trace_paths(run, rank, step)
                    trace.parent.mkdir(parents=True, exist_ok=True)
                    trace.write_bytes(f"compressed trace fixture {case}:{rank}:{step}".encode())
                    write(call, {"rank": rank, "schema_version": 2 if uccl else 1,
                        "steps": [{"step": step, "skipped_iter": False}]})
                    if uccl:
                        write(analysis / case / f"rank-{rank}-step-{step:05d}.json", {"valid_uccl_trace_capture": True,
                            "synthetic": False, "pair_count": 384, "paired_direction_count": 768,
                            "inputs": {str(trace): digest(trace), str(call): digest(call)},
                            "analysis_source_sha256": digest(root / "analysis/inspect_uccl_trace.py")})
                    else:
                        validations.append({"rank": rank, "steps": [step], "trace_path": str(trace), "trace_sha256": digest(trace),
                            "call_record_path": str(call), "call_record_sha256": digest(call)})
                        aggregate = run / "analysis/all_steps" / f"rank-{rank}" / f"step-{step:05d}.json"
                        write(aggregate, {"upstream_aggregation_fixture": True})
                        aggregates.append({"step": step, "path": str(aggregate), "sha256": digest(aggregate)})
                if not uccl:
                    write(run / "analysis/all_steps" / f"rank-{rank}" / "summary.json", {"valid_capture": True,
                        "rank": rank, "steps": [0, 1], "world_size": 16, "cp": 1, "gradient_accumulation": 4,
                        "routing_audit_path": str(audit), "routing_audit_sha256": digest(audit), "validation": validations,
                        "step_summaries": aggregates, "analysis_source_sha256": {name: digest(root / "analysis" / name) for name in (
                            "analyze_all_step_trace.py", "validate_torch_trace.py", "aggregate_deep_trace.py")}})
                if uccl:
                    transport = {key: "disabled" if key == "FI_MR_CACHE_MONITOR" else None
                                 for key in pilot.TRANSPORT_ENVIRONMENT_KEYS}
                    write(run / "raw" / f"uccl_backend_rank-{rank}_startup.json", {"valid_identity": True, "rank": rank,
                        "hostname": placement[rank]["hostname"], "job": "training-job", "backend": "uccl_ep",
                        "source_revision": BASE_REVISION, "source_identity": source_identity,
                        "binary": {"path": str(binary), "sha256": digest(binary)},
                        "libfabric": {"path": str(fabric), "sha256": digest(fabric)}, "mode": "normal_high_throughput",
                        "requested_transport": "cxi", "low_latency_dedup_api_exercised": False, "build_dir": str(build),
                        "requested_transport_environment": transport,
                        "gpu_ordinals": {"local_rank": local, "cuda_current_device": local, "uccl_physical_gpu_rank": local},
                        "native_adapter": {"path": str(adapter), "sha256": digest(adapter)}, "wrappers": wrappers})
                    write(run / "raw" / f"uccl_backend_rank-{rank}_runtime.json", {"valid_identity": True, "rank": rank,
                        "first_completed_step": 0, "low_latency_mode": False, "scratch_is_cuda": True,
                        "requested_transport_environment": transport,
                        "scratch_device": f"cuda:{local}", "buffer_type": "deep_ep.buffer.Buffer",
                        "num_nvl_bytes": 256, "num_rdma_bytes": 256, "num_sms": 20})
                    write(run / "raw" / f"uccl_backend_rank-{rank}_cleanup.json", {"valid_cleanup": True, "rank": rank,
                        "official_destroy_uccl_returned": True, "native_runtime_released_before_scratch": True,
                        "peer_closure_barrier_completed": True, "process_groups_still_alive": True,
                        "allocator_drained_before_native_stream_release": True,
                        "transient_metadata_layers": 48, "transient_attributes_cleared": 384,
                        "known_transient_fields": ["token_indices", "token_probs", "dispatched_indices", "dispatched_probs",
                            "dispatched_routing_map", "reversed_mapping_for_combine", "pad_offsets", "tokens_per_expert"]})
            if uccl:
                for node in range(4):
                    write(run / "raw" / f"native-cxi-node-{node}.json", inspect_node(run / "logs" / f"torchrun-node-{node}", node))
            write(run / "analysis/smoke_health.json", {"valid": True, "cp": 1, "ranks": 16, "steps": 2,
                "health": summarize(run, steps=2, interval=1)})
            write(analysis / case / "received_routes.json", validate_received_run(run, 2))
        with patch.object(pilot, "verify_uccl_source", return_value=source_identity):
            yield root, orchestration, analysis, runs


class AcceptPilotTests(unittest.TestCase):
    def test_resolved_comparison_allows_only_documented_positive_infinity(self):
        reference = {"optimizer": {"grad_norm_skip_threshold": math.inf, "lr": 1e-6},
                     "model": {"count": 1}, "order": [1, 2], "zero": 0.0}
        pilot.equal_resolved_configuration(reference, copy.deepcopy(reference))
        for fault in ("negative_inf", "nan_threshold", "other_inf", "other_nan", "typed_inf",
                      "finite_threshold", "bool_alias", "list_order", "signed_zero"):
            with self.subTest(fault=fault):
                actual, expected = copy.deepcopy(reference), copy.deepcopy(reference)
                if fault in ("negative_inf", "nan_threshold"):
                    actual["optimizer"]["grad_norm_skip_threshold"] = -math.inf if fault == "negative_inf" else math.nan
                    expected = copy.deepcopy(actual)
                elif fault in ("other_inf", "other_nan"):
                    actual["model"]["count"] = math.inf if fault == "other_inf" else math.nan
                    expected = copy.deepcopy(actual)
                elif fault == "typed_inf":
                    actual["optimizer"]["grad_norm_skip_threshold"] = "inf"
                elif fault == "finite_threshold":
                    actual["optimizer"]["grad_norm_skip_threshold"] = 10.0
                elif fault == "bool_alias":
                    actual["model"]["count"] = True
                elif fault == "list_order":
                    actual["order"].reverse()
                else:
                    actual["zero"] = -0.0
                with self.assertRaises(ValueError):
                    pilot.equal_resolved_configuration(actual, expected)

    def test_actual_topology_field_is_required_and_typed(self):
        for value in (None, 8, True, "16"):
            with self.subTest(value=value), TemporaryDirectory() as directory, evidence(directory) as (root, orchestration, analysis, runs):
                path = runs["ordinary"] / "environment.json"
                record = json.loads(path.read_text())
                if value is None:
                    del record["runtime_topology"]["dense_dp"]
                else:
                    record["runtime_topology"]["dense_dp"] = value
                write(path, record)
                with self.assertRaisesRegex(ValueError, "actual pilot runtime topology"):
                    pilot.accept_pilot(orchestration, analysis, root=root)

    def test_complete_bound_validator_outputs_accept_without_parsing_trace_bytes(self):
        with TemporaryDirectory() as directory, evidence(directory) as (root, orchestration, analysis, runs):
            result = pilot.accept_pilot(orchestration, analysis, root=root)
            self.assertIs(result["valid_pilot"], True)
            self.assertEqual(result["cases"], ["ordinary", "grouped", "uccl_ordinary", "uccl"])
            self.assertEqual(result["case_runs"], {case: str(path) for case, path in runs.items()})
            self.assertEqual(result["matched_cpu_batches_per_case"], 128)
            self.assertGreater(len(result["input_files"]), 300)
            self.assertEqual(set(result["all_analysis_source_sha256"]), set(pilot.ANALYSIS_SOURCES))
            pilot.verify_uccl_source.assert_called_once_with(root / "runs/build")
            self.assertEqual(result["uccl_source_identity"], json.loads((root / "runs/build/source_identity.json").read_text()))
            self.assertTrue(result["initial_ordinary_uccl_layer0_histograms_equal"])
            self.assertEqual(result["uccl_transport_policy"], pilot.UCCL_TRANSPORT_POLICY)
            topology = json.loads((runs["ordinary"] / "environment.json").read_text())["runtime_topology"]
            self.assertEqual(topology, provenance._EP16_RUNTIME_TOPOLOGIES["cp1_ep16"])
            self.assertNotIn("dense_dp_cp", topology)

    def test_ordinary_uccl_requires_its_own_identity_cleanup_and_disabled_monitor(self):
        for fault in ("cleanup", "monitor", "runtime_monitor", "policy", "build"):
            with self.subTest(fault=fault), TemporaryDirectory() as directory, evidence(directory) as (root, orchestration, analysis, runs):
                run = runs["uccl_ordinary"]
                if fault == "cleanup":
                    (run / "raw/uccl_backend_rank-7_cleanup.json").unlink()
                elif fault == "policy":
                    path = orchestration / "acquisition.json"
                    record = json.loads(path.read_text())
                    record["uccl_transport_policy"]["FI_MR_CACHE_MONITOR"] = "memhooks"
                    write(path, record)
                elif fault == "build":
                    path = run / "config/comparison_contract.json"
                    record = json.loads(path.read_text())
                    record["uccl_build_dir"] = str(root / "runs/another-build")
                    write(path, record)
                else:
                    phase = "startup" if fault == "monitor" else "runtime"
                    path = run / "raw" / f"uccl_backend_rank-7_{phase}.json"
                    record = json.loads(path.read_text())
                    record["requested_transport_environment"]["FI_MR_CACHE_MONITOR"] = "memhooks"
                    write(path, record)
                with self.assertRaises(ValueError):
                    pilot.accept_pilot(orchestration, analysis, root=root)

    def test_wrong_approved_base_or_patch_metadata_is_rejected(self):
        for kind in ("base", "patch"):
            with self.subTest(kind=kind), TemporaryDirectory() as directory, evidence(directory) as (root, orchestration, analysis, runs):
                if kind == "base":
                    path = runs["uccl"] / "raw/uccl_backend_rank-3_startup.json"
                    record = json.loads(path.read_text())
                    record["source_identity"]["base_revision"] = "0" * 40
                else:
                    path = orchestration / "acquisition.json"
                    record = json.loads(path.read_text())
                    record["uccl_source_identity"]["patch_sha256"] = "0" * 64
                write(path, record)
                with self.assertRaises(ValueError):
                    pilot.accept_pilot(orchestration, analysis, root=root)

    def test_official_wrapper_symlink_cannot_escape_verified_source_tree(self):
        with TemporaryDirectory() as directory, evidence(directory) as (root, orchestration, analysis, runs):
            link = root / "runs/build/uccl/ep/deep_ep_wrapper/deep_ep/buffer.py"
            outside = root / "outside-buffer.py"
            outside.write_bytes(link.read_bytes())
            link.unlink()
            link.symlink_to(outside)
            # Even byte-identical imported records must not authorize an
            # official source symlink that points outside the verified tree.
            for rank in range(16):
                path = runs["uccl"] / "raw" / f"uccl_backend_rank-{rank}_startup.json"
                record = json.loads(path.read_text())
                record["wrappers"]["deep_ep.buffer"] = {"path": str(outside), "sha256": digest(outside)}
                write(path, record)
            with self.assertRaisesRegex(ValueError, "escapes verified source"):
                pilot.accept_pilot(orchestration, analysis, root=root)

    def test_unexpected_optimizer_or_precision_change_is_not_normalized_away(self):
        for kind in ("optimizer", "precision"):
            with self.subTest(kind=kind), TemporaryDirectory() as directory, evidence(directory) as (root, orchestration, analysis, runs):
                path = runs["uccl"] / "config/resolved.yaml"
                configuration = yaml.safe_load(path.read_text())
                if kind == "optimizer":
                    configuration["optimizer"]["lr"] = 2e-6
                else:
                    configuration["model"]["moe_router_dtype"] = "fp32"
                path.write_text(yaml.safe_dump(configuration))
                with self.assertRaises(ValueError):
                    pilot.accept_pilot(orchestration, analysis, root=root)

    def test_wrong_batches_placement_and_source_identity_are_rejected(self):
        for kind in ("batches", "placement", "source"):
            with self.subTest(kind=kind), TemporaryDirectory() as directory, evidence(directory) as (root, orchestration, analysis, runs):
                if kind == "batches":
                    path = runs["grouped"] / "raw/sample_fingerprint_rank-4.json"
                    record = json.loads(path.read_text())
                    record["steps"][1]["batches"][2]["tokens"]["sha256"] = "f" * 64
                    write(path, record)
                elif kind == "placement":
                    path = runs["grouped"] / "environment.json"
                    record = json.loads(path.read_text())
                    record["rank_placement"][0]["cuda_current_device"] = 1
                    write(path, record)
                else:
                    (root / "analysis/inspect_uccl_trace.py").write_text("changed analysis implementation")
                expected_error = provenance.ProvenanceInvariantError if kind == "placement" else ValueError
                with self.assertRaises(expected_error):
                    pilot.accept_pilot(orchestration, analysis, root=root)

    def test_missing_cleanup_or_false_domain_cannot_use_valid_boolean_alone(self):
        for kind in ("cleanup", "domain"):
            with self.subTest(kind=kind), TemporaryDirectory() as directory, evidence(directory) as (root, orchestration, analysis, runs):
                if kind == "cleanup":
                    (runs["uccl"] / "raw/uccl_backend_rank-7_cleanup.json").unlink()
                else:
                    path = runs["uccl"] / "raw/native-cxi-node-2.json"
                    record = json.loads(path.read_text())
                    record["workers"][1]["native_cxi_domain"] = "cxi0"
                    write(path, record)
                with self.assertRaises(ValueError):
                    pilot.accept_pilot(orchestration, analysis, root=root)

    def test_incomplete_or_mistyped_stream_lifetime_cleanup_is_rejected(self):
        for key, value in (
            ("allocator_drained_before_native_stream_release", None),
            ("allocator_drained_before_native_stream_release", False),
            ("allocator_drained_before_native_stream_release", 1),
            ("transient_metadata_layers", 47),
            ("transient_metadata_layers", "48"),
            ("transient_attributes_cleared", 383),
            ("known_transient_fields", ["token_indices", "token_probs"]),
        ):
            with self.subTest(key=key, value=value), TemporaryDirectory() as directory, evidence(directory) as (root, orchestration, analysis, runs):
                path = runs["uccl"] / "raw/uccl_backend_rank-7_cleanup.json"
                record = json.loads(path.read_text())
                if value is None:
                    del record[key]
                else:
                    record[key] = value
                write(path, record)
                with self.assertRaisesRegex(ValueError, "UCCL cleanup"):
                    pilot.accept_pilot(orchestration, analysis, root=root)

    def test_matching_proof_hashes_cannot_hide_tampered_actual_cxi_messages(self):
        with TemporaryDirectory() as directory, evidence(directory) as (root, orchestration, analysis, runs):
            run = runs["uccl"]
            proof_path = run / "raw/native-cxi-node-0.json"
            proof = json.loads(proof_path.read_text())
            worker = proof["workers"][0]
            stderr = Path(worker["logs"]["stderr"]["path"])
            stdout = Path(worker["logs"]["stdout"]["path"])
            stderr.write_text(stderr.read_text().replace("domain cxi0", "domain cxi1"))
            worker["logs"]["stderr"]["sha256"] = digest(stderr)
            write(proof_path, proof)
            (run / "logs/rank-0.out").write_bytes(stdout.read_bytes() + stderr.read_bytes())
            self.assertIs(proof["valid_native_cxi_log_binding"], True)
            self.assertEqual(worker["native_cxi_domain"], "cxi0")
            with self.assertRaisesRegex(ValueError, "native CXI domain"):
                pilot.accept_pilot(orchestration, analysis, root=root)

    def test_acquisition_completion_must_exist_be_true_and_bind_current_inputs(self):
        for kind in ("missing", "false", "bool_alias", "worker_exit", "case_hash", "source_hash", "job"):
            with self.subTest(kind=kind), TemporaryDirectory() as directory, evidence(directory) as (root, orchestration, analysis, runs):
                path = orchestration / "acquisition_completed.json"
                record = json.loads(path.read_text())
                if kind == "missing":
                    path.unlink()
                else:
                    key, value = {"false": ("success", False), "bool_alias": ("success", 1),
                        "worker_exit": ("workers_srun_returned_zero", False), "case_hash": ("cases_tsv_sha256", "0" * 64),
                        "source_hash": ("runtime_source_sha256", "0" * 64), "job": ("job", "other")}[kind]
                    record[key] = value
                    write(path, record)
                with self.assertRaises(ValueError):
                    pilot.accept_pilot(orchestration, analysis, root=root)

    def test_stale_synthetic_or_missing_trace_validation_is_rejected(self):
        for kind in ("hash", "synthetic", "source", "missing"):
            with self.subTest(kind=kind), TemporaryDirectory() as directory, evidence(directory) as (root, orchestration, analysis, runs):
                path = analysis / "uccl/rank-3-step-00001.json"
                if kind == "hash":
                    trace, _ = pilot._trace_paths(runs["ordinary"], 0, 0)
                    trace.write_bytes(b"changed raw compressed bytes")
                elif kind == "missing":
                    path.unlink()
                else:
                    record = json.loads(path.read_text())
                    record["synthetic" if kind == "synthetic" else "analysis_source_sha256"] = True if kind == "synthetic" else "0" * 64
                    write(path, record)
                with self.assertRaises(ValueError):
                    pilot.accept_pilot(orchestration, analysis, root=root)

    def test_receive_input_hashes_and_actual_health_logs_are_required(self):
        for kind in ("receive", "health"):
            with self.subTest(kind=kind), TemporaryDirectory() as directory, evidence(directory) as (root, orchestration, analysis, runs):
                if kind == "receive":
                    path = analysis / "ordinary/received_routes.json"
                    record = json.loads(path.read_text())
                    record["input_files"][0]["sha256"] = "0" * 64
                    write(path, record)
                else:
                    path = runs["ordinary"] / "logs/rank-15.out"
                    path.write_text(path.read_text().replace("number of skipped iterations: 0", "number of skipped iterations: 1"))
                with self.assertRaises(ValueError):
                    pilot.accept_pilot(orchestration, analysis, root=root)

    def test_initial_grouped_uccl_histogram_gate_does_not_require_later_route_equality(self):
        with TemporaryDirectory() as directory, evidence(directory) as (root, orchestration, analysis, runs):
            path = runs["uccl"] / "raw/routing_audit_rank-0.json"
            record = json.loads(path.read_text())
            record["steps"][0]["layers"][0]["expert_counts"][0] -= 1
            record["steps"][0]["layers"][0]["expert_counts"][2] += 1
            write(path, record)
            ownership = json.loads((runs["uccl"] / "config/routing_ownership.json").read_text())
            owner = ownership["expert_owners"]["0"][0]
            destination = runs["uccl"] / "raw" / f"routing_audit_rank-{owner}.json"
            received = json.loads(destination.read_text())
            row = received["steps"][0]["layers"][0]
            row["received_expert_counts"][row["received_expert_ids"].index(0)] -= 1
            row["received_expert_counts"][row["received_expert_ids"].index(2)] += 1
            write(destination, received)
            # Both routing records and the independently recomputed receive
            # report remain valid; only the initial cross-case gate differs.
            write(analysis / "uccl/received_routes.json", validate_received_run(runs["uccl"], 2))
            with self.assertRaisesRegex(ValueError, "initial grouped/UCCL"):
                pilot.accept_pilot(orchestration, analysis, root=root)

    def test_initial_ordinary_uccl_histograms_are_compared_separately_from_grouped(self):
        with TemporaryDirectory() as directory, evidence(directory) as (root, orchestration, analysis, runs):
            run = runs["uccl_ordinary"]
            path = run / "raw/routing_audit_rank-0.json"
            record = json.loads(path.read_text())
            record["steps"][0]["layers"][0]["expert_counts"][0] -= 1
            record["steps"][0]["layers"][0]["expert_counts"][2] += 1
            write(path, record)
            ownership = json.loads((run / "config/routing_ownership.json").read_text())
            owner = ownership["expert_owners"]["0"][0]
            destination = run / "raw" / f"routing_audit_rank-{owner}.json"
            received = json.loads(destination.read_text())
            row = received["steps"][0]["layers"][0]
            row["received_expert_counts"][row["received_expert_ids"].index(0)] -= 1
            row["received_expert_counts"][row["received_expert_ids"].index(2)] += 1
            write(destination, received)
            write(analysis / "uccl_ordinary/received_routes.json", validate_received_run(run, 2))
            with self.assertRaisesRegex(ValueError, "initial ordinary/UCCL"):
                pilot.accept_pilot(orchestration, analysis, root=root)

    def test_matching_hashes_and_valid_flags_cannot_hide_invalid_actual_receive_records(self):
        for kind in ("counts", "ownership"):
            with self.subTest(kind=kind), TemporaryDirectory() as directory, evidence(directory) as (root, orchestration, analysis, runs):
                if kind == "counts":
                    path = runs["uccl"] / "raw/routing_audit_rank-0.json"
                    record = json.loads(path.read_text())
                    record["steps"][0]["layers"][0]["received_expert_counts"][0] += 1
                    record["steps"][0]["layers"][0]["received_assignments"] += 1
                else:
                    path = runs["uccl"] / "config/routing_ownership.json"
                    record = {"upstream_validated_fixture": True}
                write(path, record)
                proof_path = analysis / "uccl/received_routes.json"
                proof = json.loads(proof_path.read_text())
                next(row for row in proof["input_files"] if row["path"] == str(path))["sha256"] = digest(path)
                write(proof_path, proof)
                self.assertIs(proof["valid_send_receive_audit"], True)
                with self.assertRaises(ValueError):
                    pilot.accept_pilot(orchestration, analysis, root=root)

    def test_final_rehash_detects_mutation_and_cli_rejects_output_inside_case(self):
        with TemporaryDirectory() as directory, evidence(directory) as (root, orchestration, analysis, runs):
            actual = pilot._uccl_identity
            def changed(*args):
                result = actual(*args)
                path = runs["ordinary"] / "logs/rank-0.out"
                path.write_text(path.read_text() + "late change\n")
                return result
            with patch.object(pilot, "_uccl_identity", side_effect=changed), self.assertRaises(ValueError):
                pilot.accept_pilot(orchestration, analysis, root=root)
            argv = ["accept_uccl_pilot.py", "--orchestration-dir", str(orchestration), "--analysis-root", str(analysis),
                    "--output", str(runs["ordinary"] / "accepted.json")]
            with patch.object(sys, "argv", argv), patch.dict("os.environ", {"SLURM_JOB_ID": "analysis", "SLURM_JOB_PARTITION": "preemptable"}), \
                 patch.object(pilot, "accept_pilot", return_value={"case_runs": {case: str(run) for case, run in runs.items()}}), \
                 self.assertRaisesRegex(ValueError, "outside every case"):
                pilot.main()


if __name__ == "__main__":
    unittest.main()
