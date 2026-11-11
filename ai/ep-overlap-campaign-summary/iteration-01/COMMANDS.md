# Allocated summary execution

Run these commands from the AMPA project root. Only the root agent submits jobs. The user authorizes debug for this campaign.

Wait until the reviewer marks the numerical audit source stable. Make a unique directory under runs/ for the new source manifest.

```bash
summary_source_dir="$PWD/runs/ep-overlap-summary-source-<UTC_TIMESTAMP>"
mkdir "$summary_source_dir"
cat runs/ep-overlap-debug-ddp-source-20261110T111031Z/source.sha256 > "$summary_source_dir/source.sha256"
sha256sum "$PWD/tools/ep_overlap_campaign_summary.py" \
  "$PWD/jobs/analysis/ep_overlap_campaign_summary.sbatch" \
  "$PWD/tests/test_ep_overlap_campaign_summary.py" \
  "$PWD/ai/ep-overlap-summary-review/iteration-01/numerical_audit.py" \
  "$PWD/profiling/ampa_profile/c0_mock_sanity.py" \
  >> "$summary_source_dir/source.sha256"
sha256sum "$summary_source_dir/source.sha256"
```

The manifest contains 149 files if these are the complete dependencies. Static closure includes the existing mock-sanity module through a conditional import. It does not run that branch. QA writes all_paths and missing_paths to closure_paths.json before its strict source gate. Add any additional named dependencies to a fresh manifest without changing the original acquisition inventory. Set a unique QA ID. Inspect the full user queue immediately before submission. Do not exceed four requested nodes across running and pending jobs.

```bash
export AMPA_OVERLAP_SUMMARY_SOURCE_MANIFEST="$summary_source_dir/source.sha256"
export AMPA_OVERLAP_SUMMARY_SOURCE_SHA256=<MANIFEST_SHA256>
export AMPA_OVERLAP_SUMMARY_MODE=qa
export AMPA_OVERLAP_SUMMARY_ID=ep-overlap-summary-qa-<UTC_TIMESTAMP>
squeue --array -u "$USER"
sbatch --parsable --time=00:10:00 --exclude=nid006657 jobs/analysis/ep_overlap_campaign_summary.sbatch
```

The QA proof is runs/QA_ID-job-QA_JOB/qa.json. Require a successful terminal Slurm state, exit code 0:0, and the matching PASS.txt digest. This job runs eight tests and a shell syntax check inside setup/gh200-cuda130-runtime.toml. It starts no training.

After all twelve full cases pass, set the matching QA directory and a unique synthesis ID. The next job reads the 24 accepted allocation records and computes the summary inside the configured container.

```bash
export AMPA_OVERLAP_SUMMARY_MODE=summarize
export AMPA_OVERLAP_SUMMARY_ID=ep-overlap-summary-<UTC_TIMESTAMP>
export AMPA_OVERLAP_SUMMARY_QA_ROOT="$PWD/runs/<QA_ID>-job-<QA_JOB>"
export AMPA_OVERLAP_SUMMARY_COORDINATOR="$PWD/runs/ep-overlap-auto-retry-20261110T165819Z-retry-coordinator"
export AMPA_OVERLAP_ACCOUNTING_START=2026-11-10
squeue --array -u "$USER"
sbatch --parsable --exclude=nid006657 jobs/analysis/ep_overlap_campaign_summary.sbatch
```

Inspect campaign_summary.json, optimizer_steps.csv, eight_cases.csv, sm_budgets.csv, the PNG/SVG chart, VALIDATED.txt, and independent_numerical_audit.json. The wrapper logs and exit receipt are in runs/SUMMARY_ID-job-SUMMARY_JOB/. The summary directory must be new. Every failed attempt stays intact.

The summary reuses only analysis.uccl_budget_hardware_compare.plot. Its old-schema main and load_case gates are not applicable to these overlap records. The independent audit checks numerical results after the summary succeeds. Final research conclusions still require the lead review.
