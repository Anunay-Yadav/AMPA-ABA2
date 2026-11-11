# Allocated adaptive summary commands

Run from the AMPA project root. Only the lead submits jobs. Keep the full queue within four requested nodes. Prefer to run this final QA and synthesis after the training coordinator finishes.

Freeze a unique summary manifest from the exact acquisition source used by the completed campaign. The current acquisition inventory has159 files. Add only the three summary program, test and launcher files.

```bash
budget_campaign="$PWD/runs/ep-overlap-budget-off-adaptive-20261110T223433Z-coordinator"
budget_summary_source="$PWD/runs/ep-overlap-budget-summary-source-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir "$budget_summary_source"
cat "$budget_campaign/campaign-source.sha256" > "$budget_summary_source/source.sha256"
sha256sum "$PWD/tools/ep_overlap_budget_summary.py" \
  "$PWD/tests/test_ep_overlap_budget_summary.py" \
  "$PWD/jobs/analysis/ep_overlap_budget_summary.sbatch" \
  >> "$budget_summary_source/source.sha256"
wc -l "$budget_summary_source/source.sha256"
export AMPA_BUDGET_SUMMARY_SOURCE_MANIFEST="$budget_summary_source/source.sha256"
export AMPA_BUDGET_SUMMARY_SOURCE_SHA256="$(sha256sum "$AMPA_BUDGET_SUMMARY_SOURCE_MANIFEST" | awk '{print $1}')"
```

The expected inventory has162 files. Run the focused QA on one debug node for ten minutes. Confirm the expanded queue has capacity immediately before submission.

```bash
export AMPA_BUDGET_SUMMARY_MODE=qa
export AMPA_BUDGET_SUMMARY_ID=ep-overlap-budget-summary-qa-$(date -u +%Y%m%dT%H%M%SZ)
squeue --array -u "$USER"
sbatch --parsable --time=00:10:00 --exclude=nid006657 jobs/analysis/ep_overlap_budget_summary.sbatch
```

The QA job runs seven tests and checks shell syntax, import closure and plotting imports inside setup/gh200-cuda130-runtime.toml. Require terminal Slurm success and the matching qa.json/PASS.txt digest. Set the actual QA job directory below.

```bash
export AMPA_BUDGET_SUMMARY_MODE=summarize
export AMPA_BUDGET_SUMMARY_ID=ep-overlap-budget-summary-$(date -u +%Y%m%dT%H%M%SZ)
export AMPA_BUDGET_SUMMARY_QA_ROOT="$PWD/runs/<QA_ID>-job-<QA_JOB>"
export AMPA_BUDGET_SUMMARY_CAMPAIGN="$budget_campaign"
export AMPA_BUDGET_ACCOUNTING_START=2026-11-10
squeue --array -u "$USER"
sbatch --parsable --exclude=nid006657 jobs/analysis/ep_overlap_budget_summary.sbatch
```

The synthesis job uses one debug node for at most90 minutes. It reads fresh UTC accounting for every accepted training/posthoc job. The output includes metrics.json, optimizer_steps.csv, legs.csv, pairs.csv, budgets.csv, paired_budget_comparison.png/.svg, evidence copies, input references and the generated audit. Require successful terminal accounting and the metrics digest in VALIDATED.txt.

The generated QWEN_UCCL_UNPROFILED_BUDGET_AUDIT.md remains an immutable report draft inside the new run. Review numbers, chart and interpretation independently. Then promote the reviewed report to docs/QWEN_UCCL_UNPROFILED_BUDGET_AUDIT.md. Keep the generated draft and prior audits unchanged.
