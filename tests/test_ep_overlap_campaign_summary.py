"""Failure-oriented summary checks. Run only on allocated compute."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from tools import ep_overlap_campaign_summary as summary


def ledgers():
    accepted, acquired = [], []
    for index, (label, (case, capture, budget)) in enumerate(summary.EXPECTED.items()):
        train, posthoc, run = str(100 + index * 2), str(101 + index * 2), "run-" + label
        acquired.append([label, "1", case, capture, str(budget), "10", train, run])
        accepted.append([label, "1", train, posthoc, run])
    # An unaccepted later attempt must never replace the accepted run.
    acquired.append(["native-ordinary-off", "2", "ordinary", "off", "20", "10", "900", "failed-attempt"])
    acquired.append(["pilot-uccl-ordinary-torch", "1", "uccl_ordinary", "torch", "20", "2", "950", "qualification"])
    accepted.append(["pilot-uccl-ordinary-torch", "1", "950", "951", "qualification"])
    return accepted, acquired


def timers():
    return [{"first_step": step, "last_step_exclusive": step + 1, "optimizer_steps": 1,
             "mean_optimizer_step_ms": (step + 1) * 1000.0} for step in range(10)]


class CampaignSummaryTests(unittest.TestCase):
    def test_selects_only_exact_accepted_full_attempts(self):
        selected, qualifications = summary.select_full_cases(*ledgers())
        self.assertEqual(len(selected), 12)
        self.assertEqual(selected[0]["id"], "run-native-ordinary-off")
        self.assertEqual(len(qualifications), 1)
        self.assertTrue(all(row["steps"] == 10 for row in selected))

    def test_missing_full_cell_cannot_be_replaced_by_pilot(self):
        accepted, acquired = ledgers()
        del accepted[0]
        with self.assertRaises(ValueError):
            summary.select_full_cases(accepted, acquired)

    def test_duplicate_full_cell_cannot_become_an_extra_replicate(self):
        accepted, acquired = ledgers()
        acquired.append(["native-ordinary-off", "3", "ordinary", "off", "20", "10", "980", "duplicate-cell"])
        accepted.append(["native-ordinary-off", "3", "980", "981", "duplicate-cell"])
        with self.assertRaises(ValueError):
            summary.select_full_cases(accepted, acquired)

    def test_reused_posthoc_job_cannot_bind_two_cases(self):
        accepted, acquired = ledgers()
        accepted[1][3] = accepted[0][3]
        with self.assertRaises(ValueError):
            summary.select_full_cases(accepted, acquired)

    def test_allocation_and_update_times_remain_separate(self):
        result = summary.separate_timing(timers(), {"elapsed_seconds": 500}, {"elapsed_seconds": 200})
        self.assertEqual(result["optimizer_update_sum_seconds"], 55.0)
        self.assertEqual(result["training_allocation_elapsed_seconds"], 500)
        self.assertEqual(result["posthoc_allocation_elapsed_seconds"], 200)
        measured = result["measured_five_statistics"]
        self.assertEqual(measured["count"], 5)
        self.assertEqual(measured["mean_ms"], 8000.0)
        self.assertEqual(measured["aggregate_tokens_per_second"], 5 * summary.TOKENS_PER_STEP / 40.0)

    def test_missing_or_duplicate_optimizer_window_is_rejected(self):
        duplicate = timers()
        duplicate[1] = duplicate[0]
        for rows in (timers()[:-1], duplicate):
            with self.subTest(rows=len(rows)), self.assertRaises(ValueError):
                summary.separate_timing(rows, {"elapsed_seconds": 500}, {"elapsed_seconds": 200})

    def test_accounting_binds_utc_elapsed_nodes_and_success(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "accounting.psv"
            path.write_text("100|COMPLETED|0:0|2026-11-10T20:00:00|2026-11-10T20:10:00|600|nid[000001-000003,000005]|debug|4\n")
            record = summary.read_accounting(path, {"100"})["100"]
            self.assertEqual(record["nodes"], ["nid000001", "nid000002", "nid000003", "nid000005"])
            self.assertEqual(record["utc_span_seconds"], 600.0)
            self.assertEqual(record["timezone"], "UTC")
            valid = path.read_text()
            for changed in (valid.replace("|600|", "|599|"), valid.replace("|debug|4", "|debug|3"),
                            valid.replace("|COMPLETED|", "|FAILED|")):
                path.write_text(changed)
                with self.assertRaises(ValueError):
                    summary.read_accounting(path, {"100"})

    def test_accounting_requires_every_selected_allocation(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "empty.psv"
            path.write_text("")
            with self.assertRaises(ValueError):
                summary.read_accounting(path, {"100", "101"})


if __name__ == "__main__":
    unittest.main()
