"""Adversarial overlap accounting checks; execute on allocated compute only."""
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from analysis import accept_uccl_pilot as evidence
from profiling import run_pretrain_ep_overlap as adapter
from profiling.ampa_profile import overlap_trace
from tools import ep_overlap_validate as validator
from test_ep16_received_routes import fixture as routing_fixture


def write(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record))


def execution(case="ordinary", capture="off", steps=2, budget=20):
    return adapter.execution_contract(SimpleNamespace(
        execution_steps=steps, overlap_capture=capture, case=case,
        communication_backend="uccl_ep" if case.startswith("uccl") else "alltoall",
        communication_sms=budget, overlap_source_sha256="a" * 64))


def training(case="ordinary", capture="off", steps=2):
    selected = execution(case, capture, steps)
    contract, ownership, audits, environment = routing_fixture(
        selected["communication_backend"], case in ("grouped", "uccl"), steps)
    contract["training"].update(warmup_steps=selected["warmup_steps"], measured_steps=selected["measured_steps"],
                                learning_rate=1e-6, recompute_mode="none")
    contract["model"].update(hf_model_id="Qwen/Qwen3-30B-A3B-Base",
        hf_revision="1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9")
    contract.update(ep_overlap=selected, run_kind="ep16_overlap_experiment", sample_fingerprints=True,
                    dataset={"mode": "indexed", "sampler_seed": 1234})
    contract["baseline"] = dict(ep_a2a_overlap=True, delay_expert_wgrad=True,
        ep_overlap_early_attn_memory_release=True, dispatch_backward_expert_wgrad_overlap=False,
        bridge_comm_overlap_configured=False, dense_dp_overlap_grad_reduce=True,
        dense_dp_overlap_param_gather=True, cuda_device_max_connections="32", capacity_drop=False,
        router_padding=False, cuda_graphs=False, activation_recomputation=False,
        dispatcher="flex" if case.startswith("uccl") else "alltoall")
    if case.startswith("uccl"):
        contract["uccl_build_dir"] = "/isolated/build"
    return selected, contract, ownership, audits, environment


def call(execution_record, rank=0, step=0):
    backend = execution_record["communication_backend"]
    observer = overlap_trace.OverlapMethodLabels(None, [], communication_backend=backend)
    record = dict(schema_version=1, kind="ampa_ep_overlap_calls", rank=rank, step=step,
        capture_mode=execution_record["capture"], skipped_iter=False, layers=list(range(48)),
        counts=dict(observer.expected_counts(4)), observation_contract=observer.contract_record(4),
        payload_bytes={}, payload_dtypes={}, uccl_autograd_pairs=[])
    if backend == "uccl_ep":
        for layer in range(48):
            for family in ("dispatch", "combine"):
                for microbatch in range(4):
                    record["uccl_autograd_pairs"].append(dict(call_id=len(record["uccl_autograd_pairs"]),
                        layer=layer, family=family, context_id=1 + layer * 8 + microbatch,
                        forward_returned=True, backward_returned=True, epoch=1))
    if execution_record["capture"] == "torch":
        record.update(profiler_stop_seconds=0.01, export_seconds=0.01)
    return record


def lifecycle(directory, mode="off"):
    run = Path(directory)
    selected = execution(capture=mode)
    placement = []
    for rank in range(16):
        slot = dict(rank=rank, hostname=f"node-{rank // 4}", local_rank=rank % 4, slurm_node_id=str(rank // 4))
        placement.append(slot)
        start = dict(schema_version=1, contract=overlap_trace.CONTRACT, valid_overlap_startup=True,
            rank=rank, hostname=slot["hostname"], slurm_job_id="123", local_rank=str(slot["local_rank"]),
            world_size=16, all_rank_configuration_agreement=True, configuration_ranks_checked=16,
            versions={"torch": "2.8.0", "transformer-engine": "2.19.0", "megatron-core": "0.15.0", "megatron-bridge": "0.1.0"},
            selected_flags=dict(overlap_trace.SELECTED_FLAGS), capture_mode=mode, completed_steps=2,
            ddp_delay_wgrad_compute=True,
            communication_backend="alltoall", communication_sms=20, cuda_device_max_connections="32",
            geometry=dict(pipeline_model_parallel_size=1, tensor_model_parallel_size=1,
                context_parallel_size=1, expert_model_parallel_size=16, expert_tensor_parallel_size=1,
                num_layers=48, cuda_graph_impl="none", recompute_granularity=None, moe_router_topk=8))
        done = dict(schema_version=1, contract=overlap_trace.CONTRACT, valid_step_execution=True,
            rank=rank, capture_mode=mode, completed_updates=2, ended_steps=[0, 1],
            capture_steps=selected["capture_step_indices"], boundaries=[], qualification_only=True)
        write(run / "raw" / f"ep_overlap_rank-{rank}_startup.json", start)
        write(run / "raw" / f"ep_overlap_rank-{rank}_capture.json", done)
        for step in selected["capture_step_indices"]:
            write(run / "raw/overlap_calls" / f"step-{step:05d}" / f"rank-{rank}.json", call(selected, rank, step))
            trace = run / "traces/torch_profile" / f"step-{step:05d}" / f"rank-{rank}.json.gz"
            trace.parent.mkdir(parents=True, exist_ok=True)
            # Deliberately not a valid event trace. This gate must never turn
            # artifact existence into correlated GPU event acceptance.
            trace.write_bytes(b"unparsed trace artifact")
    return run, selected, placement, {"pytorch_version": "2.8.0"}


class OverlapValidationTests(unittest.TestCase):
    def test_all_approved_cases_and_actual_windows_keep_overlap_flags(self):
        for case in adapter.CASES:
            for steps in (2, 10):
                with self.subTest(case=case, steps=steps):
                    selected, contract, _, _, _ = training(case, steps=steps)
                    before = copy.deepcopy(contract)
                    validator.checked_execution_contract(selected, case=case, capture="off", steps=steps, source_sha256="a" * 64)
                    validator.checked_training_contract(contract, selected, case)
                    self.assertEqual(contract, before)
                    self.assertTrue(contract["baseline"]["ep_a2a_overlap"])

    def test_flags_budget_and_bool_aliases_cannot_be_normalized_into_acceptance(self):
        for key, value in (("overlap_flags", {**adapter.OVERLAP_FLAGS, "delay_wgrad_compute": False}),
                           ("execution_steps", True), ("CUDA_DEVICE_MAX_CONNECTIONS", "1"),
                           ("communication_sms", 20)):
            selected = execution()
            selected[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validator.checked_execution_contract(selected, case="ordinary", capture="off", steps=2, source_sha256="a" * 64)
        selected, contract, _, _, _ = training()
        contract["baseline"]["delay_expert_wgrad"] = False
        with self.assertRaises(ValueError):
            validator.checked_training_contract(contract, selected, "ordinary")

    def test_resolved_flags_are_checked_without_rewriting_actual_configuration(self):
        run = Path("/project/runs/real-training")
        selected = execution()
        model = dict(adapter.OVERLAP_FLAGS)
        model.update(moe_router_num_groups=None, moe_router_group_topk=None,
            moe_token_dispatcher_type="alltoall", moe_flex_dispatcher_backend=None,
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1,
            context_parallel_size=1, expert_model_parallel_size=16,
            expert_tensor_parallel_size=1, seq_length=4096, bf16=True, fp16=False,
            moe_router_dtype=None, moe_enable_deepep=False)
        configuration = dict(model=model, train=dict(train_iters=25, exit_interval=2,
            global_batch_size=64, micro_batch_size=1), optimizer={"lr": 1e-6},
            comm_overlap=None, ddp=dict(overlap_grad_reduce=True, overlap_param_gather=True, delay_wgrad_compute=True),
            profiling=dict(use_pytorch_profiler=False, use_nsys_profiler=False),
            logger=dict(tensorboard_dir=str(run / "logs/tensorboard"), wandb_exp_name=run.name,
                        wandb_save_dir=str(run / "logs/wandb")))
        before = copy.deepcopy(configuration)
        validator.checked_resolved(configuration, selected, "ordinary", run)
        self.assertEqual(configuration, before)
        for key, value in adapter.OVERLAP_FLAGS.items():
            changed = copy.deepcopy(configuration)
            changed["model"][key] = not value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validator.checked_resolved(changed, selected, "ordinary", run)
        for value in (False, None):
            changed = copy.deepcopy(configuration)
            if value is None:
                changed["ddp"].pop("delay_wgrad_compute")
            else:
                changed["ddp"]["delay_wgrad_compute"] = value
            with self.subTest(ddp_delay_wgrad_compute=value), self.assertRaises(ValueError):
                validator.checked_resolved(changed, selected, "ordinary", run)

    def test_integer_gate_preserves_actual_flags_and_catches_last_chunk_wrong_owner(self):
        for steps in (2, 10):
            selected, contract, ownership, audits, environment = training("uccl", steps=steps)
            before = copy.deepcopy(contract)
            result = validator.checked_receives(contract, ownership, audits, environment, steps)
            self.assertEqual(contract, before)
            self.assertEqual(result["total_source_expert_assignments"], steps * 16 * 48 * 131072)
            self.assertEqual(result["checked_expert_equalities"], steps * 48 * 128)
            audits[0]["steps"][-1]["layers"][-1]["rank_traffic"][0] += 1
            with self.assertRaises(ValueError):
                validator.checked_receives(contract, ownership, audits, environment, steps)

    def test_source_receive_corruptions_and_missing_final_step_fail(self):
        for mutate in (lambda audits: audits[0]["steps"].pop(),
                       lambda audits: audits[-1]["steps"][-1]["layers"][-1].update(receive_calls=True),
                       lambda audits: audits[0].update(rank=True)):
            _, contract, ownership, audits, environment = training()
            mutate(audits)
            with self.assertRaises(ValueError):
                validator.checked_receives(contract, ownership, audits, environment, 2)

    def test_self_declared_call_counts_and_missing_delayed_wgrad_fail(self):
        selected = execution(capture="torch")
        record = call(selected)
        validator.checked_call(record, rank=0, step=0, execution=selected)
        label = next(key for key in record["counts"] if "/WGRAD/" in key)
        record["counts"][label] = 0
        record["observation_contract"]["expected_counts"][label] = 0
        with self.assertRaises(ValueError):
            validator.checked_call(record, rank=0, step=0, execution=selected)

    def test_uccl_journal_requires_all_completed_pairs_without_unique_context_assumption(self):
        selected = execution("uccl_ordinary", "nsys", budget=32)
        record = call(selected)
        validator.checked_call(record, rank=0, step=0, execution=selected)
        for field, value in (("backward_returned", False), ("epoch", 2), ("call_id", True)):
            bad = copy.deepcopy(record)
            bad["uccl_autograd_pairs"][-1][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validator.checked_call(bad, rank=0, step=0, execution=selected)

    def test_off_lifecycle_is_unwrapped_and_missing_rank_or_skipped_step_fails(self):
        with TemporaryDirectory() as directory:
            run, selected, placement, environment = lifecycle(directory)
            result = validator.checked_lifecycle(run, selected, placement, "123", environment, evidence.Inputs())
            self.assertFalse(result["host_call_observation_enabled"])
            self.assertIsNone(result["valid_recorded_host_call_grid"])
            self.assertEqual(result["gpu_event_correlation_acceptance"], "pending_separate_analysis")
            path = run / "raw/ep_overlap_rank-15_capture.json"
            record = json.loads(path.read_text())
            record["ended_steps"] = [0]
            write(path, record)
            with self.assertRaises(ValueError):
                validator.checked_lifecycle(run, selected, placement, "123", environment, evidence.Inputs())
            path.unlink()
            with self.assertRaises(ValueError):
                validator.checked_lifecycle(run, selected, placement, "123", environment, evidence.Inputs())

    def test_finalized_worker_flags_and_cuda_environment_are_checked_on_every_rank(self):
        for field, value in (("selected_flags", {**overlap_trace.SELECTED_FLAGS, "delay_wgrad_compute": False}),
                             ("cuda_device_max_connections", "1"), ("all_rank_configuration_agreement", 1),
                             ("ddp_delay_wgrad_compute", False), ("ddp_delay_wgrad_compute", None)):
            with self.subTest(field=field), TemporaryDirectory() as directory:
                run, selected, placement, environment = lifecycle(directory)
                path = run / "raw/ep_overlap_rank-15_startup.json"
                record = json.loads(path.read_text())
                if value is None:
                    record.pop(field)
                else:
                    record[field] = value
                write(path, record)
                with self.assertRaises(ValueError):
                    validator.checked_lifecycle(run, selected, placement, "123", environment, evidence.Inputs())

    def test_nonempty_torch_artifacts_leave_event_and_hardware_acceptance_pending(self):
        with TemporaryDirectory() as directory:
            run, selected, placement, environment = lifecycle(directory, "torch")
            result = validator.checked_lifecycle(run, selected, placement, "123", environment, evidence.Inputs())
            self.assertEqual(result["call_records"], 32)
            self.assertEqual(result["torch_artifacts_bound"], 32)
            self.assertEqual(result["gpu_event_correlation_acceptance"], "pending_separate_analysis")
            self.assertEqual(result["hardware_metrics_acceptance"], "pending_separate_analysis")
            (run / "traces/torch_profile/step-00001/rank-15.json.gz").unlink()
            with self.assertRaises(ValueError):
                validator.checked_lifecycle(run, selected, placement, "123", environment, evidence.Inputs())

    def test_native_flattened_logs_must_match_original_worker_streams(self):
        with TemporaryDirectory() as directory:
            run = Path(directory)
            slot = {"rank": 7, "slurm_node_id": "2", "local_rank": 1}
            original = run / "logs/torchrun-node-2/uuid/attempt_0/1"
            original.mkdir(parents=True)
            (original / "stdout.log").write_text("actual training output\n")
            (original / "stderr.log").write_text("actual stderr\n")
            flat = run / "logs/rank-7.out"
            flat.write_bytes((original / "stdout.log").read_bytes() + (original / "stderr.log").read_bytes())
            inputs = evidence.Inputs()
            records = [(path, inputs.bind(path)) for path in (original / "stdout.log", original / "stderr.log", flat)]
            (run / "logs/log-map-node-2.sha256").write_text("".join(f"{digest}  {path}\n" for path, digest in records))
            validator.checked_worker_logs(run, [slot], evidence.Inputs())
            flat.write_text("invented successful training output\n")
            with self.assertRaises(ValueError):
                validator.checked_worker_logs(run, [slot], evidence.Inputs())


if __name__ == "__main__":
    unittest.main()
