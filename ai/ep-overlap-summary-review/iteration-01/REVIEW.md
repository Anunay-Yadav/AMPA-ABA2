# Independent review of the overlap summary

Final independent review passes. Corrected QA3911359 passed eight tests with no missing imports. Summary and independent arithmetic audit passed in allocated job3911397. All twelve full cases are accepted. The final report and six numerical/document artifacts are bound by [FINAL_REVIEWED_ARTIFACTS.sha256](FINAL_REVIEWED_ARTIFACTS.sha256).

## Confirmed source behavior

The summary selects accepted attempts by their exact ledger identities. It requires the eight backend/routing/profiler cases and four ordinary-UCCL Nsight budgets. Qualifications do not enter the twelve timing rows. Ten ordered update windows are required per case. Updates 6–10 supply the measured statistics. Training allocation time and posthoc allocation time remain separate from optimizer timers.

The tool rechecks optimizer health, actual consumed batches, send/receive totals and capture lifecycle records. It also requires the source inventory, input bindings, separate posthoc receipt and actual placement. All new full input prefixes must agree before the aggregate can pass.

The inspected native Torch artifact agrees with the per-capture filename and field assumptions. Native attribution covers forward communication only. The inspected Nsight20 inventory contains four node records with 15,360 known payload directions per node. Its metrics contain all 80 rank/update windows and the three expected sampled hardware fields. Nsight12 uses the same schema. Device-wide percentages include concurrent computation. The tool preserves missing views and does not infer free SMs.

## Source-inventory gap

The initial summary draft archives the imported repository source closure but only requires the new files in the declared summary inventory. The author added closure membership and declared-hash checks. The source identity now covers the executed dependencies directly. Allocated QA imports matplotlib with the Agg backend and compiles all new Python sources before synthesis.

The first allocated QA3911281 correctly failed before tests because source148 omitted a static dependency. Corrected source149 adds unchanged c0_mock_sanity.py and retains both strict gates. The import exists only under the mock-data branch at run_pretrain.py:739. Its additional imports are standard-library or framework imports. The indexed campaign does not execute this branch. The manifest diff changes only diagnostics in the new summary/launcher and adds that dependency. The frozen acquisition source144, tests and independent audit remain unchanged. The corrected source requires fresh allocated QA.

## Original input comparison

The original campaign acceptance and its UCCL ordinary control acceptance use prefix `6ce98c488cce7b047566c233fe93567d3d49905432e9b1709d6b07a4fd0514cb`. The new full UCCL ordinary OFF execution proof uses the same prefix and records ten updates. The execution validator compares each rank against the fixed original ten-update reference. A successful aggregate recheck can establish this equality for all twelve new cases. Two-update qualifications cannot establish the full prefix.

## Final result review

The launcher uses one debug node for at most 90 minutes. It requires a matching passed source-qualified QA record and all twelve completed acquisitions before synthesis. It obtains the accepted job accounting with UTC explicitly selected. A fresh expanded queue check precedes its container srun. The summary and independent audit execute sequentially and fail the allocation if either fails. The focused tests reject missing cells, duplicate cells, reused posthoc jobs, incomplete timer windows and invalid accounting. They also preserve separation of optimizer and allocation timing.

The separate [numerical audit](numerical_audit.py) passed all120 raw/CSV timers, measured-five statistics,24 allocation elapsed records,640 Torch captures and320 hardware windows. It also passed hardware mean/count copies and independently reconstructed sample-count-weighted means. All twelve full input prefixes match the original640-batch prefix. It ran on allocated compute inside job3911397.

Manual final review reconciled all eight original timing means to original accepted case records. All twelve new timing means, ten-update sums, job IDs and allocation elapsed values agree with the final CSV/accounting. The four sampled hardware metric triples agree with their original metrics.json records. Every reported start/stop API maximum agrees with the raw per-rank lifecycle evidence, including budget40. All24 case allocations completed0:0. The coordinator exited0 with twelve full acquisitions and four separate qualifications.

Every final per-case proof has ten updates, no qualification flag, and successful training-health, send/receive and all-rank lifecycle gates. Every case retains frozen sourceed74, the intended overlap bundle, actual DDP delay and the original input prefix. The full matrix contains exactly eight backend/routing/profiler cases and four Nsight budgets. Qualifying pilots, rejected pilots and failed attempts do not enter the statistics.

The final QWEN_EP_OVERLAP_AUDIT.md retains capture-boundary costs, physical-placement differences, one allocation per case and unresolved whole-session event completeness. Its four Torch positive-concurrency counts match the actual summaries:116,107,160,160 out of160 each. The artifact wording correctly places availability flags in JSON. The report contains no stale pending statement and makes no optimal-budget, free-SM, achieved-occupancy or isolated causal-speedup claim. Budget20 is the only unprofiled budget control in this matrix. The final factual interpretation passes review.

Reviewed hashes:

| Artifact | SHA-256 |
| --- | --- |
| Final report | 8d9f32f9757640c47dfeebefea51736f5969bb967783ca19cbbd4f7b80d29cd7 |
| campaign_summary.json | d322cc8e695141bb75b222cecd57d3d6457cf874d67dccca69816cacaf602445 |
| independent_numerical_audit.json | c4a12b026e1cdef873becb266c45c06837c5d8ea223e04975d9a938c6e2b61ed |
| optimizer_steps.csv | 4b9228c38756021bfe56a08942ec584e7f808deb6acfe01c5f3c76edb269d7e6 |

This reviewer submitted no jobs and ran no Python, tests or trace parser on the login node. The numerical source is independent of the summary implementation and executed only on allocated compute. The audit addresses arithmetic and copied evidence. Existing producer validations establish raw GPU event attribution. No review blocker remains.
