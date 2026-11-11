# Independent adaptive summary review

## Resolved finding

The first summary version exported `logical_expert_assignments` from `received_routes.json` after checking its success flag and equal totals. The actual qualification artifacts confirm that neither the leg proof nor the pair proof binds this derivative. An altered derivative could therefore preserve equal totals while changing the exported count.

The corrected `reconcile_received_routes` reads the bound ownership record and all sixteen raw rank audits. It calls `validator.checked_receives` for all ten updates and requires exact equality with the saved derivative before exporting counts. This reuses the original integer send/receive validation. The seventh test checks rejection of an altered expert-check count and acceptance of the matching record. Both launcher gates and the commands document expect seven tests. The targeted review passes. [Route reconciliation](../../../tools/ep_overlap_budget_summary.py:139)

## Ledger and allocation evidence

The seven-column accepted ledger matches the coordinator interface. The summary rejects duplicate allocations and replicate slots, requires canonical proof paths, and checks the training and posthoc submission ledgers. Every decision replays against the earlier accepted pairs. Its immutable snapshot matches the exact earlier ledger bytes. The final completion marker, terminal coordinator state and both boundary decisions must agree. Qualification records and failed attempts remain outside the numerical rows.

Fresh UTC Slurm accounting uses the same parser as the pair producer. The summary compares accepted and fresh training accounting and requires separate completed posthoc allocations. It checks actual node identities, both host-wall intervals and leg order. It keeps the shared allocation elapsed, each case wall interval, the ten optimizer timers and the separate posthoc elapsed distinct. The accepted qualification pair confirms the expected proof and accounting schema. [Qualification pair](../../../runs/ep-overlap-budget-off-pilot12-20-20261110T221816Z-posthoc/pair_comparison.json)

## Timing, placement and interpretation

The summary reads all ten raw optimizer timers and requires contiguous update indices. It reconstructs the measured-five mean, median, sample standard deviation and throughput from updates 6–10. It joins each timer to its matching logged health update. It rechecks actual input fingerprints, the original full prefix, all-rank runtime budget guards and paired physical GPU identity. The common workload comparison preserves the overlap settings and removes only the selected budget and per-case output destinations from the resolved comparison.

The planner and chart use one ratio per allocation pair. The chart labels the observed range and the operational 2% band. The report states the sampled allocation and update units, the incumbent-relative stopping rule and the qualified device boundary. The actual QA record supports 4–56 even blocks and shows the shared-memory exclusion at 58. The report limits hardware claims to the configured payload budget. It presents the chosen value as the best observed median under this protocol. [Actual device qualification](../../../runs/ep-overlap-budget-off-qa-20261110T221142Z/preflight.json)

## Execution

The launcher checks the summary manifest, source closure, plotting imports, seven focused tests and matching QA proof. The commands use one debug node and the configured container for QA and synthesis. The summary reads container-side library bindings in that environment. This avoids the scheduler-host namespace mismatch handled separately by the coordinator. No source-review blocker remains. Actual completed-campaign synthesis and independent numerical and visual review are still required before promoting the report. [Allocated commands](../../../ai/ep-overlap-budget-summary/iteration-01/COMMANDS.md)
