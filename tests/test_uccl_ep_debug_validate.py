"""Adversarial tests for the ten-update adapter and individual-case evidence.

Run only inside an allocated configured container. Small fixtures exercise
the strict gates without claiming to reproduce training or native traces.
"""
import copy
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import yaml

from tools import uccl_ep_debug_validate as debug
from test_accept_uccl_pilot import evidence as pilot_fixture, digest, write
from test_ep16_received_routes import fixture as routing_fixture


def log_text(steps=10, *, skipped=0, timer="100.0"):
    return "".join(
        f"iteration {step} / 25 | elapsed time per iteration (ms): {timer} | consumed samples: {step * 64} | "
        "learning rate: 1e-6 | global batch size: 64 | lm loss: 2.0 | load_balancing_loss: 2.0 | grad norm: 0.5 | "
        f"number of skipped iterations: {skipped} | number of nan iterations: 0 |\n"
        for step in range(1, steps + 1))


@contextmanager
def control_fixture(directory):
    with pilot_fixture(directory) as (root, orchestration, _, runs):
        case, leg, run = "ordinary", "control", runs["ordinary"]
        for relative in ("traces/torch_profile", "raw/deep_trace", "analysis/all_steps"):
            shutil.rmtree(run / relative)
        source_paths = [root / name for name in ("jobs/training/uccl_ep_debug_case.sbatch",
            "jobs/training/qwen3_uccl_debug_worker.sh", "profiling/run_pretrain_debug.py", "tools/uccl_ep_debug_validate.py")]
        for path in source_paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"bound new debug source fixture {path.name}\n")
        campaign = orchestration / "campaign-source.sha256"
        campaign.write_text("".join(f"{digest(path)}  {path}\n" for path in source_paths))
        (orchestration / "source.sha256").write_text(campaign.read_text() + f"{digest(campaign)}  {campaign}\n")
        acquisition = json.loads((orchestration / "acquisition.json").read_text())
        acquisition.update(mode="full", experiment_protocol=debug.PROTOCOL, partition="debug", case=case, leg=leg,
            warmup_steps=5, measured_steps=5, completed_updates=10, accepted_pilot_proof=None,
            cache_budget_updates=25, cache_budget_samples=1600, campaign_source_sha256=digest(campaign),
            uccl_transport_policy=copy.deepcopy(debug.DEBUG_TRANSPORT_POLICY))
        write(orchestration / "acquisition.json", acquisition)
        sources = json.loads((orchestration / "reviewed_inputs.json").read_text())["ampa_runtime_sources"]
        (orchestration / "runtime-source.sha256").write_text("".join(
            f"{value}  {root / name}\n" for name, value in sorted(sources["files"].items())))
        (orchestration / "cases.tsv").write_text(f"{case}\t{leg}\t{run}\n")
        completed = json.loads((orchestration / "acquisition_completed.json").read_text())
        completed.update(mode="full", cases_tsv_sha256=digest(orchestration / "cases.tsv"), warmup_steps=5,
            measured_steps=5, completed_updates=10)
        write(orchestration / "acquisition_completed.json", completed)
        contract = json.loads((run / "config/comparison_contract.json").read_text())
        contract["training"].update(measured_steps=5, completed_steps=10, learning_rate=1e-6)
        contract["model"].update(hf_model_id="Qwen/Qwen3-30B-A3B-Base",
            hf_revision="1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9")
        contract["dataset"].update(mode="indexed", sampler_seed=1234)
        write(run / "config/comparison_contract.json", contract)
        write(run / "config/debug_execution_contract.json", debug.execution_contract())
        configuration = yaml.safe_load((run / "config/resolved.yaml").read_text())
        configuration["train"]["exit_interval"] = 10
        (run / "config/resolved.yaml").write_text(yaml.safe_dump(configuration))
        mode = json.loads((run / "config/observation_mode.json").read_text())
        mode["torch_trace"] = False
        write(run / "config/observation_mode.json", mode)
        write(run / "config/all_step_capture.json", {"enabled": False, "steps": list(range(10)), "ranks": [0]})
        _, ownership, audits, _ = routing_fixture("alltoall", False, 10)
        write(run / "config/routing_ownership.json", ownership)
        for rank in range(16):
            (run / "logs" / f"rank-{rank}.out").write_text(log_text() if rank == 15 else f"worker {rank}\n")
            write(run / "raw" / f"routing_audit_rank-{rank}.json", audits[rank])
            write(run / "raw" / f"sample_fingerprint_rank-{rank}.json", {
                "schema_version": 1, "kind": "consumed_cpu_token_batch_sha256", "rank": rank,
                "steps": [{"step": step, "batches": [{"tokens": {
                    "sha256": hashlib.sha256(f"{rank}:{step}:{micro}".encode()).hexdigest(),
                    "shape": [1, 4096], "dtype": "torch.int64"}} for micro in range(4)]} for step in range(10)]})
        yield root, orchestration, run


class DebugValidationTests(unittest.TestCase):
    def test_real_integer_gate_covers_final_chunk_without_mutating_records(self):
        contract, ownership, audits, environment = routing_fixture("uccl_ep", True, 10)
        contract["training"]["measured_steps"] = 5
        before = copy.deepcopy((contract, audits))
        result = debug.checked_receives(contract, ownership, audits, environment)
        self.assertTrue(result["valid_send_receive_audit"])
        self.assertEqual(result["steps"], list(range(10)))
        self.assertEqual(result["total_source_expert_assignments"], 10 * 16 * 48 * 16384 * 8)
        self.assertEqual(result["checked_expert_equalities"], 10 * 48 * 128)
        self.assertEqual((contract, audits), before)
        # Preserve the source total while moving an assignment to the wrong
        # actual owner, late in the final chunk. The integer gate must fail.
        row = audits[0]["steps"][9]["layers"][47]
        row["rank_traffic"][0] += 1
        with self.assertRaises(ValueError):
            debug.checked_receives(contract, ownership, audits, environment)

    def test_original_step_grid_is_required_before_chunk_renumbering(self):
        contract, ownership, audits, environment = routing_fixture("alltoall", False, 10)
        contract["training"]["measured_steps"] = 5
        audits[8]["steps"][8]["step"] = 9
        with self.assertRaisesRegex(ValueError, "original step grid"):
            debug.checked_receives(contract, ownership, audits, environment)

    def test_health_rejects_incomplete_and_failed_updates(self):
        with TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "logs").mkdir()
            path = run / "logs/rank-15.out"
            path.write_text(log_text())
            health, timing = debug.checked_timing(run)
            self.assertEqual(health["completed_logged_optimizer_steps"], 10)
            self.assertEqual(timing["count"], 5)
            self.assertEqual(timing["sample_sd"], 0)
            self.assertEqual([row["first_step"] for row in timing["by_update"]], list(range(5, 10)))
            for text in (log_text(9), log_text(skipped=1), log_text(timer="0.0"),
                         log_text().replace("grad norm: 0.5", "grad norm: 0.0")):
                path.write_text(text)
                with self.assertRaises(ValueError):
                    debug.checked_timing(run)

    def test_bound_complete_control_accepts_as_individual_allocation(self):
        with TemporaryDirectory() as directory, control_fixture(directory) as (root, orchestration, run):
            result = debug.validate_case(orchestration, root / "runs/accepted", root=root)
            self.assertTrue(result["valid_debug_case"])
            self.assertEqual(result["completed_updates"], 10)
            self.assertEqual(result["consumed_cpu_batches"], 640)
            self.assertFalse(result["same_actual_rank_placement_across_allocations_claimed"])
            self.assertFalse(result["pure_profiler_overhead_available"])
            self.assertIsNone(result["valid_capture"])
            self.assertEqual(len(result["sample_prefix_sha256"]), 64)
            self.assertTrue((root / "runs/accepted/acceptance.json").is_file())

    def test_incomplete_input_and_forged_source_never_emit_acceptance(self):
        for fault in ("sample_tail", "source", "job", "length", "resolved_exit", "duplicate_case",
                      "cache_budget", "campaign_digest", "transport_policy", "execution_contract"):
            with self.subTest(fault=fault), TemporaryDirectory() as directory, control_fixture(directory) as (root, orchestration, run):
                output = root / "runs/accepted"
                if fault == "sample_tail":
                    path = run / "raw/sample_fingerprint_rank-7.json"
                    record = json.loads(path.read_text())
                    record["steps"].pop()
                    write(path, record)
                elif fault == "source":
                    (root / "profiling/run_pretrain_debug.py").write_text("unreviewed change\n")
                elif fault == "job":
                    path = run / "environment.json"
                    record = json.loads(path.read_text())
                    record["slurm_job_id"] = "other-allocation"
                    write(path, record)
                elif fault == "length":
                    path = run / "config/comparison_contract.json"
                    record = json.loads(path.read_text())
                    record["training"]["measured_steps"] = 20
                    write(path, record)
                elif fault == "resolved_exit":
                    path = run / "config/resolved.yaml"
                    record = yaml.safe_load(path.read_text())
                    record["train"]["exit_interval"] = 25
                    path.write_text(yaml.safe_dump(record))
                elif fault == "execution_contract":
                    path = run / "config/debug_execution_contract.json"
                    record = json.loads(path.read_text())
                    record["cache_budget_steps"] = 10
                    write(path, record)
                elif fault in ("cache_budget", "campaign_digest", "transport_policy"):
                    path = orchestration / "acquisition.json"
                    record = json.loads(path.read_text())
                    if fault == "cache_budget":
                        record["cache_budget_samples"] = 640
                    elif fault == "campaign_digest":
                        record["campaign_source_sha256"] = "0" * 64
                    else:
                        record["uccl_transport_policy"] = copy.deepcopy(debug.evidence.UCCL_TRANSPORT_POLICY)
                    write(path, record)
                else:
                    path = orchestration / "cases.tsv"
                    path.write_text(path.read_text() * 2)
                with self.assertRaises(ValueError):
                    debug.validate_case(orchestration, output, root=root)
                self.assertFalse((output / "acceptance.json").exists())

    def test_profile_commands_cover_all160_including_warmup_without_synthetic(self):
        run, output = Path("/runs/case"), Path("/runs/result")
        uccl = debug.profile_commands("uccl", run, output)
        self.assertEqual(len(uccl), 160)
        self.assertIn("rank-15-step-00009", [name for name, _ in uccl])
        self.assertIn("rank-0-step-00000", [name for name, _ in uccl])
        self.assertTrue(all("--synthetic" not in command for _, command in uccl))
        native = debug.profile_commands("grouped", run, output)
        self.assertEqual(len(native), 16)
        self.assertEqual([int(command[command.index("--native-rank") + 1]) for _, command in native], list(range(16)))

    def test_missing_profile_result_cannot_be_accepted(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(ValueError):
                debug.checked_profile("uccl", root / "run", root / "output", root, debug.evidence.Inputs())


if __name__ == "__main__":
    unittest.main()
