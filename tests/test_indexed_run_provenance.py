"""CPU-only contracts for common indexed-run provenance."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from profiling.ampa_profile.provenance import (
    ProvenanceInvariantError,
    _atomic_write_json,
    _ampa_runtime_sources,
    _read_optional,
    build_environment_record,
    build_observation_mode,
    validate_rank_placement,
    validate_recomputation_runtime,
    validate_runtime_topology,
    validate_cp2_groups,
)


def _moe_act_fixture() -> tuple[SimpleNamespace, list[object]]:
    config = SimpleNamespace(
        recompute_granularity="selective",
        recompute_modules=["moe_act"],
        moe_grouped_gemm=True,
        moe_layer_recompute=False,
        recompute_method=None,
        recompute_num_layers=None,
        cuda_graph_impl="none",
        bf16=True,
        fp8=None,
        fp4=None,
        use_transformer_engine_op_fuser=False,
        moe_token_dispatcher_type="alltoall",
        num_moe_experts=128,
        num_layers=48,
        moe_layer_freq=1,
    )
    expert_type = type("TEGroupedMLP", (), {})
    dispatcher_type = type("MoEAlltoAllTokenDispatcher", (), {})
    attention_type = type("SelfAttention", (), {})
    modules = []
    for layer in range(48):
        attention = attention_type()
        attention.checkpoint_core_attention = False
        attention.layer_number = layer + 1
        modules.append((f"layer-{layer}.attention", attention))
        modules.append((f"layer-{layer}.moe", SimpleNamespace(
            local_expert_indices=tuple(range(16)),
            token_dispatcher=dispatcher_type(),
            layer_number=layer + 1,
            experts=expert_type(),
            moe_layer_recompute=False,
            shared_experts_recompute=False,
        )))
        modules[-1][1].experts.activation_recompute = True
        modules[-1][1].experts._with_fused_impl = False
    return config, [SimpleNamespace(named_modules=lambda: iter(modules))]


def _placement() -> list[dict[str, object]]:
    return [
        {
            "rank": rank,
            "hostname": "node-a" if rank < 4 else "node-b",
            "slurm_node_id": "0" if rank < 4 else "1",
            "slurm_local_id": str(rank % 4),
            "cuda_visible_device_count": 1,
            "physical_gpu_uuid": f"GPU-00000000-0000-0000-0000-{rank:012d}",
            "physical_gpu_identity_source": "cuda_driver.cuDeviceGetUuid(cuda_runtime.cudaGetDevice())",
        }
        for rank in range(8)
    ]


def _topology() -> dict[str, int]:
    return {"world_size": 8, "tp": 1, "pp": 1, "cp": 1, "ep": 8, "etp": 1, "dense_dp": 8, "expert_dp": 1}


def _cp2_placement() -> list[dict[str, object]]:
    placement = _placement()
    for rank, record in enumerate(placement):
        record["process_groups"] = {
            "cp": [rank // 2 * 2, rank // 2 * 2 + 1],
            "dense_dp": [rank % 2 + offset for offset in (0, 2, 4, 6)],
            "dense_dp_cp": list(range(8)),
            "ep": list(range(8)),
        }
        record["local_expert_indices"] = list(range(rank * 16, (rank + 1) * 16))
    return placement


def _no_recompute_fixture() -> tuple[SimpleNamespace, list[object]]:
    config = SimpleNamespace(
        recompute_granularity=None, recompute_modules=None, recompute_method=None,
        recompute_num_layers=None, moe_layer_recompute=False,
    )
    attention_type = type("SelfAttention", (), {})
    modules = []
    for _ in range(48):
        attention = attention_type()
        attention.checkpoint_core_attention = False
        modules.append(attention)
        modules.append(SimpleNamespace(
            local_expert_indices=tuple(range(16)), token_dispatcher=object(),
            moe_layer_recompute=False, shared_experts_recompute=False,
            experts=SimpleNamespace(activation_recompute=False),
        ))
    return config, [SimpleNamespace(modules=lambda: iter(modules))]


class IndexedRunProvenanceTest(unittest.TestCase):
    def test_cp2_no_recompute_checks_all_constructed_modules(self) -> None:
        config, chunks = _no_recompute_fixture()
        record = validate_recomputation_runtime(
            config, chunks, expected_mode="none", strict_no_recompute=True
        )
        self.assertEqual(record["verified_moe_layers"], 48)
        self.assertEqual(record["verified_attention_layers"], 48)
        self.assertEqual(record["local_expert_indices"], list(range(16)))
        # MCore finalize normalizes an inert None to ["core_attn"] even when
        # recompute_granularity is None; constructed flags remain authoritative.
        config.recompute_modules = ["core_attn"]
        self.assertEqual(
            validate_recomputation_runtime(config, chunks, expected_mode="none", strict_no_recompute=True)["mode"],
            "none",
        )
        modules = list(chunks[0].modules())
        modules[0].checkpoint_core_attention = True
        with self.assertRaisesRegex(ProvenanceInvariantError, "attention checkpointing"):
            validate_recomputation_runtime(config, chunks, expected_mode="none", strict_no_recompute=True)
        modules[0].checkpoint_core_attention = False
        modules[1].experts.activation_recompute = True
        with self.assertRaisesRegex(ProvenanceInvariantError, "MoE checkpointing"):
            validate_recomputation_runtime(config, chunks, expected_mode="none", strict_no_recompute=True)
        modules[1].experts.activation_recompute = False
        config.recompute_modules = ["moe_act"]
        with self.assertRaisesRegex(ProvenanceInvariantError, "checkpoint settings"):
            validate_recomputation_runtime(config, chunks, expected_mode="none", strict_no_recompute=True)

    def test_accepts_only_constructed_moe_act_checkpoint_modules(self) -> None:
        config, chunks = _moe_act_fixture()
        inventory = validate_recomputation_runtime(config, chunks, expected_mode="moe_act")
        self.assertEqual(inventory["grouped_expert_activation_checkpoint_layers"], 48)
        self.assertEqual(inventory["full_moe_checkpoint_layers"], 0)
        self.assertEqual(inventory["attention_checkpoint_layers"], 0)

    def test_accepts_additional_transformer_block_attention_flag(self) -> None:
        config, chunks = _moe_act_fixture()
        modules = list(chunks[0].named_modules())
        modules.insert(0, ("decoder", SimpleNamespace(checkpoint_core_attention=False)))
        chunks = [SimpleNamespace(named_modules=lambda: iter(modules))]
        inventory = validate_recomputation_runtime(config, chunks, expected_mode="moe_act")
        self.assertEqual(inventory["attention_flagged_modules"], 49)

    def test_rejects_full_moe_or_attention_recomputation(self) -> None:
        config, chunks = _moe_act_fixture()
        modules = list(chunks[0].named_modules())
        modules[1][1].moe_layer_recompute = True
        with self.assertRaisesRegex(ProvenanceInvariantError, "isolated TEGroupedMLP"):
            validate_recomputation_runtime(config, chunks, expected_mode="moe_act")
        modules[1][1].moe_layer_recompute = False
        modules[0][1].checkpoint_core_attention = True
        with self.assertRaisesRegex(ProvenanceInvariantError, "attention recomputation"):
            validate_recomputation_runtime(config, chunks, expected_mode="moe_act")

    def test_rejects_missing_moe_layer(self) -> None:
        config, chunks = _moe_act_fixture()
        modules = list(chunks[0].named_modules())[:-1]
        chunks = [SimpleNamespace(named_modules=lambda: iter(modules))]
        with self.assertRaisesRegex(ProvenanceInvariantError, "expected 48"):
            validate_recomputation_runtime(config, chunks, expected_mode="moe_act")

    def test_rejects_duplicate_attention_layer_identity(self) -> None:
        config, chunks = _moe_act_fixture()
        modules = list(chunks[0].named_modules())
        modules[2][1].layer_number = 1
        chunks = [SimpleNamespace(named_modules=lambda: iter(modules))]
        with self.assertRaisesRegex(ProvenanceInvariantError, "duplicate attention layer identity"):
            validate_recomputation_runtime(config, chunks, expected_mode="moe_act")

    def test_accepts_actual_ep8_placement_and_topology(self) -> None:
        placement = validate_rank_placement(_placement())
        self.assertEqual([record["rank"] for record in placement], list(range(8)))
        self.assertEqual(validate_runtime_topology(_topology()), _topology())

    def test_cp2_requires_intra_node_cp_and_global_ep(self) -> None:
        placement = _cp2_placement()
        validate_cp2_groups(placement)
        placement[0]["process_groups"]["cp"] = [0, 4]
        with self.assertRaisesRegex(ProvenanceInvariantError, "CP2 peers cross"):
            validate_cp2_groups(placement)

    def test_rejects_duplicate_physical_gpu(self) -> None:
        placement = _placement()
        placement[1]["physical_gpu_uuid"] = placement[0]["physical_gpu_uuid"]
        with self.assertRaisesRegex(ProvenanceInvariantError, "not unique"):
            validate_rank_placement(placement)

    def test_rejects_non_four_by_four_or_bad_local_ranks(self) -> None:
        placement = _placement()
        placement[4]["hostname"] = "node-a"
        with self.assertRaisesRegex(ProvenanceInvariantError, "two hosts with four ranks"):
            validate_rank_placement(placement)
        placement = _placement()
        placement[3]["slurm_local_id"] = "4"
        with self.assertRaisesRegex(ProvenanceInvariantError, "local ranks"):
            validate_rank_placement(placement)

    def test_rejects_wrong_actual_dense_dp(self) -> None:
        topology = _topology()
        topology["dense_dp"] = 1
        with self.assertRaisesRegex(ProvenanceInvariantError, "dense_dp"):
            validate_runtime_topology(topology)

    def test_observation_mode_fails_closed_for_wrapper_claims(self) -> None:
        self.assertFalse(build_observation_mode(lightweight_profiling="off", router_observer_attached=False)["route_wrapper_installed"])
        self.assertTrue(build_observation_mode(lightweight_profiling="on", router_observer_attached=True)["dispatcher_preprocess_wrapper_installed"])
        with self.assertRaisesRegex(ProvenanceInvariantError, "requires an attached"):
            build_observation_mode(lightweight_profiling="on", router_observer_attached=False)
        with self.assertRaisesRegex(ProvenanceInvariantError, "forbids"):
            build_observation_mode(lightweight_profiling="off", router_observer_attached=True)
        self.assertTrue(build_observation_mode(
            lightweight_profiling="off", router_observer_attached=False, nsys_trace=True,
        )["nsys_trace"])
        self.assertTrue(build_observation_mode(
            lightweight_profiling="off", router_observer_attached=False, torch_trace=True,
        )["torch_trace"])
        with self.assertRaisesRegex(ProvenanceInvariantError, "Nsight diagnostic requires"):
            build_observation_mode(
                lightweight_profiling="on", router_observer_attached=True, nsys_trace=True,
            )

    def test_environment_records_finalized_blend_and_atomic_json(self) -> None:
        record = build_environment_record(
            slurm_job_id="123",
            pytorch_version="2.x",
            cuda_version="13.x",
            nccl_version=[2, 0],
            gpu_name="GH200",
            container_sqsh_sha256="a" * 64,
            megatron_lm={"revision": "lm"},
            megatron_bridge={"revision": "bridge"},
            ampa_runtime_sources={"sha256": "source", "files": {}},
            checkpoint="/checkpoint/c0",
            dataset_data_path=None,
            dataset_blend_per_split=[(["/data/web", "/data/code"], [0.5, 0.5]), None, None],
            parallelism={"tp": 1, "pp": 1, "cp": 1, "ep": 8, "etp": 1},
            runtime_topology=_topology(),
            rank_placement=_placement(),
            hf_model_id="Qwen/Qwen3-30B-A3B-Base",
            hf_model_revision="pinned",
        )
        self.assertEqual(record["dataset_data_path"], None)
        self.assertEqual(record["dataset_blend_per_split"][0][0], ["/data/web", "/data/code"])
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "environment.json"
            _atomic_write_json(destination, record)
            self.assertEqual(json.loads(destination.read_text(encoding="utf-8"))["slurm_job_id"], "123")
            self.assertFalse(destination.with_suffix(".json.tmp").exists())
        with self.assertRaisesRegex(ProvenanceInvariantError, "blend_per_split"):
            build_environment_record(
                slurm_job_id=None,
                pytorch_version="2.x",
                cuda_version=None,
                nccl_version=None,
                gpu_name="GH200",
                container_sqsh_sha256=None,
                megatron_lm={},
                megatron_bridge={},
                ampa_runtime_sources={},
                checkpoint=None,
                dataset_data_path=None,
                dataset_blend_per_split=None,
                parallelism={},
                runtime_topology={},
                rank_placement=[],
                hf_model_id=None,
                hf_model_revision=None,
            )

    def test_container_identity_uses_only_sha256_field(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.sqsh.sha256"
            path.write_text("a" * 64 + "  /absolute/image.sqsh\n", encoding="utf-8")
            self.assertEqual(_read_optional(path), "a" * 64)
            path.write_text("invalid  /absolute/image.sqsh\n", encoding="utf-8")
            with self.assertRaisesRegex(ProvenanceInvariantError, "invalid container SHA256"):
                _read_optional(path)

    def test_ampa_runtime_source_identity_changes_with_source(self) -> None:
        from profiling.ampa_profile.provenance import _AMPA_RUNTIME_SOURCE_PATHS

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for relative in _AMPA_RUNTIME_SOURCE_PATHS:
                source = root / relative
                source.parent.mkdir(parents=True, exist_ok=True)
                source.write_text(relative, encoding="utf-8")
            before = _ampa_runtime_sources(root)
            source = root / _AMPA_RUNTIME_SOURCE_PATHS[0]
            source.write_text("changed", encoding="utf-8")
            after = _ampa_runtime_sources(root)
            self.assertNotEqual(before["sha256"], after["sha256"])
            source.unlink()
            with self.assertRaisesRegex(ProvenanceInvariantError, "missing or symlinked"):
                _ampa_runtime_sources(root)


if __name__ == "__main__":
    unittest.main()
