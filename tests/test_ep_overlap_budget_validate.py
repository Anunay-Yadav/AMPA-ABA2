"""Failure-oriented budget, pairing and adaptive-selection checks for compute QA."""
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest

from analysis import ep_overlap_budget_compare as compare
from profiling import run_pretrain_ep_overlap_budget as adapter
from tools import ep_overlap_budget_validate as validator

QUALIFIED_BUDGETS = tuple(range(4, 57, 2))


def execution(budget=48, steps=10):
    return adapter.execution_contract(SimpleNamespace(case="uccl_ordinary", overlap_capture="off",
        communication_backend="uccl_ep", communication_sms=budget, execution_steps=steps,
        warmup_steps=1 if steps == 2 else 5, measured_steps=1 if steps == 2 else 5,
        overlap_source_sha256="a" * 64))


def runtime_record(budget=48):
    limits = {"ordinal": 2, "major": 9, "multi_processor_count": 132, "warp_size": 32,
        "max_threads_per_block": 1024, "shared_memory_per_block_optin": 232448,
        "supported_budgets": list(QUALIFIED_BUDGETS)}
    return {"schema_version": 1, "valid_actual_budget": True, "experiment_protocol": adapter.PROTOCOL,
        "rank": 6, "job": "5000", "first_completed_update": 1, "requested_budget": budget,
        "buffer_num_sms": budget, "buffer_class_num_sms": budget, "low_latency_mode": False,
        "group_size": 16, "device_launch_limits": limits}


def observations(budgets=(12, 32, 40), ratios=None):
    ratios = ratios or {12: .90, 32: .95, 40: .80, 48: .85, 54: .79, 8: .91, 4: .92}
    result = []
    for budget in budgets:
        for order in compare.ORDERS:
            result.append({"pair_id": f"b{budget}-{order}", "training_slurm_job_id": f"job-{budget}-{order}",
                "candidate_budget": budget, "order": order, "candidate_over_reference_mean_ratio": ratios[budget],
                "execution_steps": 10, "qualification_only": False,
                "legs": [{"mean_step_ms": 10000.0, "sample_sd_ms": 50.0}] * 2})
    return result


class UnprofiledBudgetValidationTests(unittest.TestCase):
    def test_completion_ids_cover_both_cases_independently_of_execution_order(self):
        for ordered in (["candidate", "reference"], ["reference", "candidate"]):
            compare.checked_completed_case_ids(["candidate", "reference"], ordered)
        for observed in (["candidate"], ["candidate", "candidate"], ["candidate", "other"]):
            with self.subTest(observed=observed), self.assertRaises(ValueError):
                compare.checked_completed_case_ids(observed, ["candidate", "reference"])

    def test_truthful_non20_off_contract_is_accepted(self):
        for budget in (4, 12, 20, 32, 40, 54, 56):
            record = execution(budget)
            self.assertEqual(validator.checked_execution_contract(record, case="uccl_ordinary", capture="off",
                steps=10, source_sha256="a" * 64)["communication_sms"], budget)

    def test_profiled_or_grouped_case_cannot_enter_off_comparison(self):
        for case, mode in (("uccl", "off"), ("uccl_ordinary", "nsys"), ("uccl_ordinary", "torch")):
            with self.subTest(case=case, mode=mode), self.assertRaises(ValueError):
                validator.checked_execution_contract(execution(), case=case, capture=mode, steps=10, source_sha256="a" * 64)

    def test_contract_cannot_hide_budget_or_source_drift(self):
        for field, value in (("communication_sms", 20), ("ddp_delay_wgrad_compute", False),
                             ("source_manifest_sha256", "b" * 64)):
            record = execution()
            record[field] = value
            # A budget change alone remains a valid different contract. Its
            # native launch requirements still identify48, so exact checking fails.
            with self.subTest(field=field), self.assertRaises(ValueError):
                validator.checked_execution_contract(record, case="uccl_ordinary", capture="off", steps=10,
                                                     source_sha256="a" * 64)

    def test_exercised_budget_guard_checks_actual_rank_job_and_device(self):
        slot = {"rank": 6, "cuda_current_device": 2}
        validator.checked_budget_runtime_record(runtime_record(), execution(), slot, "5000")
        for field, value in (("buffer_num_sms", 20), ("rank", 7), ("job", "4999"), ("low_latency_mode", True)):
            record = runtime_record()
            record[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validator.checked_budget_runtime_record(record, execution(), slot, "5000")

    def test_supported_list_cannot_override_actual_shared_memory_limit(self):
        record = runtime_record(54)
        record["device_launch_limits"]["shared_memory_per_block_optin"] = 200000
        with self.assertRaises(ValueError):
            validator.checked_budget_runtime_record(record, execution(54), {"rank": 6, "cuda_current_device": 2}, "5000")

    def test_initial_plan_preserves_candidate_and_reverse_order(self):
        records = []
        complete = observations()
        for budget, order in compare.INITIAL_PLAN:
            decision = compare.plan_next_pair(records, QUALIFIED_BUDGETS)
            self.assertEqual((decision["next_pair"]["candidate_budget"], decision["next_pair"]["order"]), (budget, order))
            records.append(next(row for row in complete if row["candidate_budget"] == budget and row["order"] == order))

    def test_qualification_or_duplicate_allocation_is_not_a_replicate(self):
        record = observations()[0]
        bad = {**record, "execution_steps": 2, "qualification_only": True}
        for records in ([bad], [record, copy.deepcopy(record)]):
            with self.subTest(records=len(records)), self.assertRaises(ValueError):
                compare.budget_statistics(records)

    def test_competitive_upper_edge_extends_after_balanced_initial_pairs(self):
        decision = compare.plan_next_pair(observations(), QUALIFIED_BUDGETS)
        self.assertEqual(decision["next_pair"]["candidate_budget"], 48)

    def test_loss_is_relative_to_incumbent_even_if_better_than20(self):
        records = observations((12, 32, 40, 48))
        decision = compare.plan_next_pair(records, QUALIFIED_BUDGETS)
        self.assertEqual(decision["action"], "complete")
        self.assertEqual(decision["best_tested_budget"], 40)
        self.assertEqual(decision["boundary_status"]["upper"]["status"], "repeatably_worse")

    def test_overlapping_bands_do_not_manufacture_a_slowdown(self):
        records = observations((12, 32, 40, 48), {12: .90, 32: .95, 40: .80, 48: .805})
        decision = compare.plan_next_pair(records, QUALIFIED_BUDGETS)
        self.assertEqual(decision["action"], "run_pair")
        self.assertEqual(decision["next_pair"]["candidate_budget"], 56)

    def test_noise_adds_only_a_third_pair_before_extension(self):
        records = observations()
        records[0]["legs"][0]["sample_sd_ms"] = 500.0
        decision = compare.plan_next_pair(records, QUALIFIED_BUDGETS)
        self.assertEqual(decision["next_pair"]["candidate_budget"], 12)
        self.assertEqual(decision["next_pair"]["replicate"], 3)

    def test_hardware_limit_is_not_called_performance_degradation(self):
        records = observations((4, 8, 12, 32, 40, 48, 56),
            {4: .94, 8: .92, 12: .90, 32: .85, 40: .80, 48: .77, 56: .74})
        decision = compare.plan_next_pair(records, QUALIFIED_BUDGETS)
        self.assertEqual(decision["action"], "complete")
        self.assertEqual(decision["best_tested_budget"], 56)
        self.assertEqual(decision["boundary_status"]["upper"]["status"], "supported_limit_without_clear_degradation")

    def test_configuration_normalization_preserves_unrelated_workload_changes(self):
        left = Path("/tmp/case-left")
        config = {"model": {"moe_flex_dispatcher_num_sms": 12, "seed": 1234},
            "optimizer": {"lr": 1e-6}, "logger": {"tensorboard_dir": str(left / "logs/tensorboard"),
            "wandb_exp_name": left.name, "wandb_save_dir": str(left / "logs/wandb")}}
        normalized = compare.normalized_config(config, left, 12)
        drift = copy.deepcopy(config)
        drift["optimizer"]["lr"] = 2e-6
        self.assertNotEqual(compare.normalized_config(drift, left, 12), normalized)


if __name__ == "__main__":
    unittest.main()
