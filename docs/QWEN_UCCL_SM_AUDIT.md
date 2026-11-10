# Qwen UCCL SM experiment audit

These four acquisitions measure the original configuration with dedicated MoE expert-parallel overlap disabled. They are separate from the new overlap-enabled campaign.

Each allocation uses four GH200 nodes and 16 GPUs. Every run independently loads the same C0 and consumes the same deterministic input prefix. The workload uses ordinary top-eight routing, EP16/CP1/TP1/PP1, BF16, sequence length 4096, global batch 64, and gradient accumulation four. All ten optimizer updates passed the recorded training-health and routing gates. Updates 1–5 are warmup. Nsight captures updates 6–10 through one collector per node.

The configured communication budget is a payload block count. It does not reserve that number of SMs. The original [configuration and input audit](QWEN_UCCL_EXPERIMENT_AUDIT.md) describes the common inputs. The [hardware comparison](../runs/uccl-budget4-hardware-comparison-20261109T233200Z/hardware_budget_comparison.md) describes the sampled counters and acceptance limits.

## Whole allocation and step totals

The Slurm elapsed time includes preparation, model construction, checkpoint loading, all ten training updates, capture finalization, and the launcher's health/timing summaries. It excludes queue waiting and later postprocessing. The timer sum only adds the ten printed complete optimizer-step timers. Their difference includes work outside those timers and timer-resolution effects. No subtraction estimates pure communication or profiling overhead.

| Payload blocks | Slurm job | Start UTC | End UTC | Allocation seconds | Ten-step timer sum (s) | Updates 6–10 mean (s) |
| ---: | ---: | --- | --- | ---: | ---: | ---: |
| 12 | 3902094 | 2026-11-09T17:54:18Z | 2026-11-09T18:01:31Z | 433 | 224.4051 | 22.92854 |
| 20 | 3901926 | 2026-11-09T17:02:09Z | 2026-11-09T17:08:57Z | 408 | 208.3214 | 21.26288 |
| 32 | 3902744 | 2026-11-09T18:26:13Z | 2026-11-09T18:33:40Z | 447 | 239.4935 | 24.50054 |
| 40 | 3903070 | 2026-11-09T18:52:01Z | 2026-11-09T18:59:11Z | 430 | 219.4447 | 22.13304 |

All four jobs completed with exit code 0:0. Accounting reports four nodes and 1152 CPUs per allocation. Physical node placements differ between runs. These diagnostic timings do not establish a causal budget effect or profiler-free throughput.

## Every optimizer step

Values below are seconds converted from the original millisecond timers in each run's rank-15 log. Update numbers are one-based. The [step CSV](data/qwen_uccl_sm_steps.csv) retains the original millisecond precision and local evidence paths.

| Update | Window | 12 blocks | 20 blocks | 32 blocks | 40 blocks |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | Warmup | 49.4008 | 47.8661 | 51.0180 | 49.3788 |
| 2 | Warmup | 15.7533 | 14.1863 | 17.2040 | 16.2047 |
| 3 | Warmup | 14.8812 | 13.4746 | 16.2953 | 14.3465 |
| 4 | Warmup | 14.8209 | 13.4274 | 16.3141 | 14.5247 |
| 5 | Warmup | 14.9062 | 13.0526 | 16.1594 | 14.3248 |
| 6 | Nsight capture | 17.8458 | 16.4329 | 19.3863 | 17.1661 |
| 7 | Nsight capture | 15.6084 | 14.0758 | 17.1369 | 14.9352 |
| 8 | Nsight capture | 15.3540 | 13.6802 | 16.7384 | 14.7180 |
| 9 | Nsight capture | 15.2403 | 13.6597 | 16.8314 | 14.8676 |
| 10 | Nsight capture | 50.5942 | 48.4658 | 52.4097 | 48.9783 |

The start boundary affects update 6. The stop boundary strongly affects update 10. Recorded maximum stop API times are 35.1912, 34.5601, 35.5223, and 34.1326 seconds for budgets 12, 20, 32, and 40. These costs explain why the last timer is much larger. The audit preserves all raw values. It does not subtract boundary costs or report the diagnostic mean as normal training throughput.

## Published metric records

These are the original four metrics.json files with unchanged bytes. They include source bindings, all rank/update windows, coverage statistics, acceptance paths, and unavailable metrics. Referenced raw traces and local postprocessing artifacts remain outside this source/results repository.

- Budget 12: [metrics.json](../runs/uccl-budget12-hardware-final-20261109T181400Z/metrics.json), SHA-256 `29e4c27105c27bfa000c8fc2bd65842b9bf8adb9c3453f7e0cdf758276d73be4`.
- Budget 20: [metrics.json](../runs/uccl-budget20-hardware-final-20261109T175146Z/metrics.json), SHA-256 `889c09b26d22c59519140482f51e4f4337715f5cefbd8a5c9fc1c72610c13196`.
- Budget 32: [metrics.json](../runs/uccl-budget32-hardware-final-20261109T232451Z/metrics.json), SHA-256 `3fb1ce4cafe3d18f73a7c16288d477c1775b6ab8b183445484be61da3cf26d67`.
- Budget 40: [metrics.json](../runs/uccl-budget40-hardware-final-20261109T232946Z/metrics.json), SHA-256 `86077d0697f486c0501bbf0418f4a8ab9fff2557bda6f3bfc66a1e28d1460169`.

Sampled SM activity describes all device work during the selected UCCL payload intervals. It includes concurrent computation. It is not a communication-only counter, an instantaneous count of busy SMs, or a count of free SMs. Accepted scope covers 80 rank/update windows and 61,440 payload kernels per budget. Whole-session event completeness remains unresolved. Compute-warp activity is not achieved occupancy. Nested and overlapping intervals are not additive time partitions.

## Evidence and regeneration

The acquisition map is `runs/uccl-budget4-tracked-20261109T165740Z-coordinator/acquisitions.tsv`. Each run has its original rank logs, finalized configuration, routing and received-count records, input fingerprints, and four native Nsight reports. The accepted postprocessing root is `runs/uccl-budget4-tracked-20261109T165740Z-posthoc-cpu-v2/cases/sms<BUDGET>/`. Each case contains acceptance.json, timing.json, training_health.json, and native SQLite proofs. Capture-boundary evidence is `ai/qwen-uccl-block-sweep/iteration-01/CAPTURE_BOUNDARY_OVERHEAD.json` with its input manifest. The [measurement guide](../ai/qwen-debug-reproducibility/iteration-01/MEASUREMENTS.md) lists the acquisition and analysis entrypoints.

The [saved Slurm accounting](data/qwen_uccl_sm_slurm_accounting.txt) preserves the exact output. Slurm accounting was queried with UTC explicitly selected:

```bash
TZ=UTC SLURM_TIME_FORMAT=standard sacct -X \
  -j 3901926,3902094,3902744,3903070 \
  --starttime=2026-11-09 --endtime=2026-11-11 \
  --noheader --parsable2 \
  --format=JobIDRaw,State,ExitCode,Start,End,ElapsedRaw,Elapsed,NNodes,NCPUS,NodeList
```

Run training and analysis only on allocated compute in the configured container. The published metric records are sufficient to inspect the accepted numerical results. Recomputing them requires the original or newly acquired native reports and the recorded source, environment, inputs, and validation gates.
