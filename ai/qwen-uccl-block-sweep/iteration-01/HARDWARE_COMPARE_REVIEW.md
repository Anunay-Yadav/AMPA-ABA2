# Independent four-budget comparison review

State: Final independent review passed. HARDWARE_COMPARE_SOURCE_GO=true. FOUR_CASE_HARDWARE_OUTPUT_GO=true. NATIVE_TIMELINE_DELIVERY_GO=true. START_HERE_GO=true. Both source findings are resolved. No material blocker remains within the declared scope.

## Scope

Review the standalone [comparison tool](../../../analysis/uccl_budget_hardware_compare.py), its small saved summaries, chart, CSV and final delivery. Training and trace acquisition are outside the edit scope. The reviewer makes no implementation changes, runs no GPU work or jobs, and does not scan the large SQLite files.

The [hardware metric review](HARDWARE_METRIC_REVIEW.md) accepts analyzer `f43fb3fc` and its sampled-device method. The comparison must bind each case to its own accepted timeline and that analyzer. Percentages describe all-device samples during known payload intervals. They do not isolate communication-only work or establish achieved occupancy. Whole-session event completeness remains unresolved.

## Source findings and resolutions

1. Original source081932f6 checks only the GPU UUID set. A coherent rank/UUID permutation can pass, and metric SQLite identities lack accepted export bindings. Resolved in final source00ad2ca9. Each stream binds to its accepted node, full process ID, rank, physical UUID, VM and device ID. Each node requires four distinct streams and processes. Metric SQLite paths and hashes match accepted exports. The author tests coherent UUID swaps, process/node/VM changes, duplicate streams and SQLite identity drift.
2. Original source081932f6 lacks cross-case workload equivalence. Valid analyses from different workloads can appear together. Resolved in final source00ad2ca9. The comparison binds accepted checkpoint, data, input prefix, runtime, binary, adapter and worker identities. It checks archived native revisions, status and diffs against runtime records. It also binds the environment, model manifest and resolved configuration. Normalization removes only the checked budget and three output locations. The sole allowed nonfinite configuration value is positive infinity at the documented gradient-skip default. All four normalized configuration hashes match.

The final chart explicitly states that these percentages are not achieved occupancy. The author log records 14 record, two matrix, ten mapping, five configuration and nine cross-case rejections. The synthetic PNG/SVG render passes. This reviewer read those checks and their source, but did not rerun the author's Python tests.

## Actual output and independent arithmetic

The final [comparison](../../../runs/uccl-budget4-hardware-comparison-20261109T233200Z/comparison.json) uses budgets12,20,32,40 in ascending order. Its 54 bound inputs include the accepted per-case metadata and the reviewed analyzer identity. All cases have 80 unique rank/update windows, 16 physical GPU UUIDs, 61,440 payload kernels and available selected views.

I independently reconstructed every pooled mean and count from the 80 window summaries in each saved metric output. The calculation uses JavaScript, not the comparison implementation. For each metric and view it computes `sum(count × window_mean) / sum(count)`. All 48 means and counts match exactly. The [independent summary checks](../../qwen-budget-compare-review/iteration-01/independent-summary-checks.json) bind the four source metric JSON files and record their populations and coverage.

I compared all 48 comparison JSON means and all 48 CSV rows against those independent results. Differences are zero. I visually inspected the actual [PNG](../../../runs/uccl-budget4-hardware-comparison-20261109T233200Z/hardware_budget_comparison.png). Its three panels contain the correct twelve rounded labels, ascending budgets, zero-based percent axes and visible different-range and interpretation notes. There is no clipped label or misleading occupancy claim. The SVG comes from the same reviewed figure save loop, and its generated hash matches the output manifest. I did not separately render SVG in a desktop application.

| Blocks | Payload sample bundles | SM activity (%) | SM issue (%) | Compute-warp throughput (%) |
| --- | ---: | ---: | ---: | ---: |
| 12 | 873372 | 10.51607448 | 1.62245641 | 2.42528270 |
| 20 | 733521 | 15.78961339 | 2.10168080 | 3.55813808 |
| 32 | 937877 | 21.71446789 | 2.23594885 | 4.05282356 |
| 40 | 783943 | 25.72375160 | 2.60469958 | 4.63513291 |

These decimal means summarize samples with one-percentage-point source resolution. They do not imply equivalent hardware accuracy or independent repeated experiments.

## Native delivery and timing interpretation

The [delivery proof](../../../runs/uccl-budget4-tracked-20261109T165740Z-posthoc-cpu-v2/DELIVERY.json) passes under unchanged source7a8e6423. It records the same workload and input prefix, four accepted cases and 16 reports. The report index has exactly one report per budget/node pair. All report hashes match accepted export proofs and the delivery bindings. Current nonempty sizes match the index and total 3,371,535,669 bytes. The lead's delivery command rehashed the report files. This reviewer checked the metadata joins and file sizes without repeating that multigigabyte read.

The timing CSV has 20 records. The payload CSV has 640 records plus its header. I independently reconstructed the timing means, sample deviations and coefficients of variation. They match the delivery values. These values include capture overhead and do not measure intrinsic training noise or profiler-free throughput.

The [opening guide](../../../runs/uccl-budget4-hardware-comparison-20261109T233200Z/START_HERE.md) states that the final logged update includes profiler-stop overhead. I rehashed the boundary input manifest and independently recomputed maxima from all 64 rank capture records. Stop calls reach 35.1912,34.5601,35.5223,34.1326 seconds for budgets12,20,32,40. The adapter closes the AMPA host scope before its stop boundary. Accepted payload intervals stay inside their checked windows. This does not establish complete unenumerated optimizer GPU activity. The guide correctly prohibits subtracting an API duration as a performance correction.

I checked the guide's local links and the four Slurm records. All four jobs completed0:0 with ten healthy updates and four reports each. Every allocation provides 1,152 CPUs across four nodes. Report-clock, placement, capture scope, immutable-input reuse and occupancy limits remain explicit.

## Final identities

| Artifact | SHA256 |
| --- | --- |
| Final comparison source | `00ad2ca9f097f3adb8f959f340e0c7a96345dbcc292d7b90111480902bf17616` |
| Hardware comparison JSON | `3e0ea000033266da720187ab6c1a8f542a15bc5b47c833c23340b15239b77b46` |
| PNG | `9805ca502cd325ed3238cb8d2760a580215b4350417351d17369a80030a0df20` |
| SVG | `f271bd0e14b3b2ca2f9ddbf1dd06349337067a197626339766dd1942f069d882` |
| Hardware CSV | `f09cf959148d3e2906e76eee1d20f0955271e88a2dcd7ef96ba7a1a44cc9f87d` |
| Native delivery JSON | `71cb81ad338a3b870b7d56d956db31bf0a5f600933365109e5145708f386893e` |
| Opening guide | `8a1523875a94ff9ce04d735814dcef6e68703fdc2d96d2479d12c4b5727c4937` |
| Boundary summary | `2a12d44ff1c9cba30004653124838813e0de9425c0a49563ebc1b1f98590934d` |

The [output checks](../../qwen-budget-compare-review/iteration-01/independent-output-checks.json) record the independent numeric and report-index checks. All named artifact hashes were checked directly. The reviewer changed only review/task records. No implementation, raw acquisition, native trainer or accepted metric output changed.

## Limits

Acceptance covers known source-bound AMPA ranges and UCCL payload launches. Whole-session CUDA/NVTX completeness remains unresolved. Hardware means describe sampled all-device activity during selected intervals. They do not establish communication-only work, achieved occupancy, free SM reservations, NIC saturation or exposed delay. Physical placement differs, and one diagnostic allocation per budget does not establish repeatability or a causal best budget. No further work remains for this bounded review.
