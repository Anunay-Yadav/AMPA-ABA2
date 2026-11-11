# Final independent adaptive sweep review

**PASS.** The reviewed report, numerical artifacts and chart support the stated bounded, short-run conclusion. No blocking finding remains. This review changes only its own task records. It does not change the frozen acquisition source, summary source, generated evidence or promoted report.

## Scope and verification

The [accepted ledger](../../../runs/ep-overlap-budget-off-adaptive-20261110T223433Z-coordinator/accepted.tsv) contains 20 full paired allocations and 40 fresh C0 starts. The final coordinator marker and terminal exit agree on 20 pairs, 40 cases and 400 updates. The seven-column ledger, all 20 small pair proofs, final decision, generated report, four CSV tables, metrics and fresh accounting were read independently. The CSV files contain 400 optimizer rows, 40 case rows, 20 pair rows and eight budget rows. Exactly 200 updates are measured: updates 6–10 in each case. Qualification is excluded. The failed training allocation 3912016 is preserved in the failure ledger; its replacement is accepted, and the failed attempt contributes no numerical rows.

Every pair has the same actual sixteen-rank physical GPU placement in both legs. Every case has sixteen valid native runtime guards, with requested budget equal to the actual Buffer instance and class budgets, ten ordered complete-step timers, 640 consumed CPU batches and ten finite logged health updates with zero skipped or NaN iterations. Each case reconciles 1,006,632,960 logical expert assignments. The summary source rederives routing from bound ownership and all sixteen raw rank audits, rather than trusting the saved derivative. The [resolved source-review finding](REVIEW.md) is fixed and covered by the seventh focused test.

The common metadata preserves the actual Qwen model revision, C0, indexed dataset, BF16, ordinary top-eight routing, EP16/CP1/TP1/PP1/ETP1, sequence length 4096, global batch 64, micro batch one and accumulation four. Main expert-parallel overlap, model and actual DDP delayed weight gradients, and early attention memory release are enabled. The alternate dispatch-backward flag remains disabled. Source-bound validation rechecks the original complete consumed prefix `6ce98c488cce7b047566c233fe93567d3d49905432e9b1709d6b07a4fd0514cb` in every case. Immutable payload verification reuse is stated accurately.

The independently reviewed summary implementation reconstructs all timers, measured-five mean, median, sample standard deviation, sum and aggregate throughput from raw logs on allocated compute. This final review reconciled the published pair means and ratios with all twenty original pair proofs, inspected small timing and health records, checked table identities and counts, and compared the promoted report with the immutable generated draft. No Python, test suite, trace parser or training workload ran on the login node for this review.

There are twenty distinct four-node training allocations and twenty distinct one-node posthoc allocations. Their forty fresh accounting rows all show `COMPLETED`, `0:0`, and the debug partition. The report keeps shared allocation elapsed time, forty case host intervals, ten-timer sums and separate posthoc time distinct. It does not invent a per-case Slurm allocation time. Direct independent scheduler inspection also confirmed QA 3913091 completed in 18 seconds and synthesis 3913094 in 112 seconds, each on one debug node. QA passed seven tests with no missing import-closure paths. The successful synthesis marker binds the exact metrics digest below.

## Interpretation and visual review

The final replayed decision selects 40 by its lowest observed median ratio, `0.9784990549120511`. Its two opposite-order allocation pairs have ratios `0.9770443682859735` and `0.9799537415381286`: 2.00–2.30% shorter measured step time, with median reduction 2.15%. The report explicitly limits this result to two short allocation replicates.

Budget 56 has ratios `0.9228724880464979`, `1.022693331141836` and `1.02871626086267`. Its first large reduction did not repeat, including in another reference-first allocation. The promoted report accurately states the 7.71% shorter to 2.87% longer range, two slower pairs and 2.27% slower median. It does not call 56 a reliable winner or assign the variation to a proven cause.

The actual final lower boundary is four, repeatably worse relative to the 40-block incumbent under the declared rule. The upper boundary is 56, a supported limit without clear degradation. The 58-block notify request needs 237,568 dynamic shared-memory bytes, beyond the actual 232,448-byte opt-in limit. Resource qualification of even budgets 4–56 is distinguished from full training at every intermediate value. Neither boundary establishes a global optimum.

The PNG was independently opened and inspected. Every pair appears with its execution-order marker. The median, observed ranges, paired 20-block reference and operational 2% band are clear; the large 56-block spread is visible. Whiskers are described as observed ranges. The five sequential measured updates are subsamples, not independent allocation replicates. Observers remain enabled and their cost is included. Profiler OFF means sampled SM activity, achieved occupancy and free SMs are unavailable. The report makes no statistical-significance, global-optimum, physical-traffic or long-run-quality claim.

## Approved artifact hashes

All paths below are relative to the project root. Generated artifacts remain immutable; the promoted documentation adds interpretation and reproduction links without changing generated numerical tables.

| Artifact | SHA-256 |
| --- | --- |
| `docs/QWEN_UCCL_UNPROFILED_BUDGET_AUDIT.md` | `f610ca9e0d4dd253aeb2f82f9d4ab7504148b98c2ee81ef866bda4dd53e0b858` |
| `runs/ep-overlap-budget-off-source-20261110T221142Z/source.sha256` (159 files) | `252ef8a09ddb96efef52ceebf339a52287e2dc48422f0a9fa472f5d41fc76927` |
| `runs/ep-overlap-budget-off-final-source-20261110T224818Z/source.sha256` (162 files) | `2b170f107aabc07c1da917f37ff6ed8eb5c97fb2da03850e64f62037a4b2c5f6` |
| `runs/ep-overlap-budget-summary-qa-20261111T041120Z-job-3913091/qa.json` | `544dd0cba3145bdaed21edd05affb32f81c78ddca08a2fef2e94e7768749fc87` |
| `runs/ep-overlap-budget-summary-final-20261111T041250Z/metrics.json` | `c8aa699baa1f73be1e40c5da7fe585100cccea23ccbf984ab8eeb2bbb6f59e7a` |
| `runs/ep-overlap-budget-summary-final-20261111T041250Z/optimizer_steps.csv` | `ee1983f3f666bd3106c0f506f9bfd7ff268f1148eca8d7966fa862e664be6704` |
| `runs/ep-overlap-budget-summary-final-20261111T041250Z/legs.csv` | `b3822b74dc20b33c18c5248ae00a01a079582189cca28230a2f8a4db8479f982` |
| `runs/ep-overlap-budget-summary-final-20261111T041250Z/pairs.csv` | `fa1d76d66b7ca0329613abafe7401a0eecaa9defeb158d2f1d89449222f07dfc` |
| `runs/ep-overlap-budget-summary-final-20261111T041250Z/budgets.csv` | `4af03009d28deef9a66c395a8eadd8e6837acfb4bc6bf531690ecb62ae52c90a` |
| `runs/ep-overlap-budget-summary-final-20261111T041250Z/paired_budget_comparison.png` | `f6a8b4226d5115e099daa4e14908b4c4613480506d514c8f8384f990758035c5` |
| `runs/ep-overlap-budget-summary-final-20261111T041250Z/paired_budget_comparison.svg` | `c8190d2116e7428149be1a8bad55f1646f73e03afb7031a50c586cb11c128e43` |
| `runs/ep-overlap-budget-summary-final-20261111T041250Z/QWEN_UCCL_UNPROFILED_BUDGET_AUDIT.md` | `ba174f2123f536f76c0857554b268dd33d1b92e66ec173492c478304f08c5874` |
| `runs/ep-overlap-budget-summary-final-20261111T041250Z/input_references.json` | `fd74cf31b748e6deb0e163245319dc447a149d927114b234ad6c36727038b09b` |
| `runs/ep-overlap-budget-summary-final-20261111T041250Z/evidence/accounting_utc.psv` | `f457ed507c31c01d625031a1cb9e75e0af4a5d15f595df99be7cc5e1a6b755fc` |

Prior audit `docs/QWEN_EP_OVERLAP_AUDIT.md` is unchanged at SHA-256 `8d9f32f9757640c47dfeebefea51736f5969bb967783ca19cbbd4f7b80d29cd7`. The lead owns the separate final acceptance receipt.
