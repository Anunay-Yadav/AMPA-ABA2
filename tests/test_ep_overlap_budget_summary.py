"""Failure-oriented adaptive summary checks. Execute on allocated compute."""
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from tools import ep_overlap_budget_summary as summary


def ledger_rows(runs):
    repeats = {}
    rows = []
    for index, (budget, order) in enumerate(summary.compare.INITIAL_PLAN):
        repeats[budget] = repeats.get(budget, 0) + 1
        pair = f"fixture-pair-{index}"
        rows.append([str(budget), order, str(repeats[budget]), pair, str(100 + 2 * index), str(101 + 2 * index),
                     str(runs / f"{pair}-posthoc/pair_comparison.json")])
    return rows


def bytes_for(rows):
    return ("\n".join("\t".join(row) for row in rows) + "\n").encode()


def timers():
    return [{"first_step": step, "last_step_exclusive": step + 1, "optimizer_steps": 1,
             "mean_optimizer_step_ms": float(1000 * (step + 1))} for step in range(10)]


class AdaptiveSummaryTests(unittest.TestCase):
    def test_ledger_requires_full_initial_allocation_set(self):
        runs = Path("/fixture/runs")
        rows = ledger_rows(runs)
        self.assertEqual(len(summary.read_ledger(bytes_for(rows), runs=runs)), 6)
        with self.assertRaises(ValueError):
            summary.read_ledger(bytes_for(rows[:-1]), runs=runs)

    def test_reused_job_or_foreign_proof_does_not_supply_an_independent_pair(self):
        runs = Path("/fixture/runs")
        rows = ledger_rows(runs)
        rows[1][4] = rows[0][5]
        with self.assertRaises(ValueError):
            summary.read_ledger(bytes_for(rows), runs=runs)
        rows = ledger_rows(runs)
        rows[1][6] = rows[0][6]
        with self.assertRaises(ValueError):
            summary.read_ledger(bytes_for(rows), runs=runs)

    def test_every_optimizer_timer_is_retained_but_only_last_five_supply_statistics(self):
        value = summary.check_timer_windows(timers())
        self.assertEqual(value["sum_optimizer_seconds"], 55.0)
        self.assertEqual(value["all_ten"]["count"], 10)
        self.assertEqual(value["measured_five"]["count"], 5)
        self.assertEqual(value["measured_five"]["mean_ms"], 8000.0)
        self.assertEqual(value["measured_five"]["aggregate_tokens_per_second"], 5 * summary.TOKENS_PER_STEP / 40)

    def test_duplicate_or_missing_update_never_forms_a_complete_leg(self):
        duplicate = timers()
        duplicate[4] = duplicate[3]
        for rows in (duplicate, timers()[:-1]):
            with self.subTest(rows=len(rows)), self.assertRaises(ValueError):
                summary.check_timer_windows(rows)

    def test_completed_driver_can_report_no_active_job_but_cannot_repeat_a_marker_key(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "exit.txt"
            path.write_text("exit_code=0\ncomplete=1\nactive_job=\n")
            self.assertEqual(summary.read_keys(path)["active_job"], "")
            path.write_text("exit_code=0\ncomplete=1\ncomplete=0\n")
            with self.assertRaises(ValueError):
                summary.read_keys(path)

    def test_saved_route_totals_must_match_raw_reconciliation(self):
        with TemporaryDirectory() as directory:
            run = Path(directory)
            ownership = {"kind": "fixture ownership"}
            audits = [{"rank": rank} for rank in range(16)]
            contract, environment = {"kind": "fixture contract"}, {"kind": "fixture environment"}
            summary.write_new(run / "config/routing_ownership.json", ownership)
            for audit in audits:
                summary.write_new(run / "raw" / f"routing_audit_rank-{audit['rank']}.json", audit)
            stored_path = run / "received_routes.json"
            received = {"valid_send_receive_audit": True, "total_source_expert_assignments": 160,
                        "total_received_expert_assignments": 160, "checked_expert_equalities": 16}
            summary.write_new(stored_path, {**received, "checked_expert_equalities": 15})
            with patch.object(summary.validator, "checked_receives", return_value=received) as validate:
                with self.assertRaisesRegex(ValueError, "fresh integer send/receive reconciliation"):
                    summary.reconcile_received_routes(run, stored_path, contract, environment, summary.evidence.Inputs())
                validate.assert_called_once_with(contract, ownership, audits, environment, 10)
                stored_path.unlink()
                summary.write_new(stored_path, received)
                self.assertEqual(summary.reconcile_received_routes(run, stored_path, contract, environment,
                                 summary.evidence.Inputs()), received)

    def test_stopping_proof_must_bind_exact_immutable_earlier_ledger(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / "campaign_accepted_snapshot.tsv"
            previous = [{"pair_id": "earlier", "ledger_line": "earlier ledger row"}]
            current = {"pair_id": "new", "proof_path": str(root / "pair_comparison.json")}
            payload = b"earlier ledger row\n"
            digest = hashlib.sha256(payload).hexdigest()
            snapshot.write_bytes(payload)
            plan = {"action": "complete", "next_pair": None, "global_optimum_claimed": False}
            source, analyzer, supported = "a" * 64, "b" * 64, [12, 20, 32, 40]
            record = {**plan, "valid_campaign_decision": True, "experiment_protocol": summary.adapter.PROTOCOL,
                "source_manifest_sha256": source, "analyzer_sha256": analyzer, "supported_budgets": supported,
                "accepted_pair_ids": ["earlier", "new"], "accepted_pair_count": 2,
                "tolerance_fraction": summary.compare.TOLERANCE, "prospective_pair_path": current["proof_path"],
                "accepted_ledger_snapshot": {"path": str(snapshot), "sha256": digest},
                "input_files": [{"path": str(snapshot), "sha256": digest}]}
            summary.check_decision(record, plan, current, previous, source, analyzer, supported, summary.evidence.Inputs())
            snapshot.write_bytes(b"different earlier ledger row\n")
            with self.assertRaises(ValueError):
                summary.check_decision(record, plan, current, previous, source, analyzer, supported, summary.evidence.Inputs())


if __name__ == "__main__":
    unittest.main()
