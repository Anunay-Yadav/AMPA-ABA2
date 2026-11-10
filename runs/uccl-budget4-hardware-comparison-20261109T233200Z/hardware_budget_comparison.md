# Four-budget sampled hardware comparison

These are sampled whole-device means during any UCCL payload interval. Every case passes all 80 selected coverage windows and its accepted 61,440-payload event witness.

| Payload blocks | Job | SM activity (%) | Instruction issue (%) | Compute-warp activity (%) | Selected samples |
| --- | --- | ---: | ---: | ---: | ---: |
| 12 | 3902094 | 10.516 | 1.622 | 2.425 | 873372 |
| 20 | 3901926 | 15.790 | 2.102 | 3.558 | 733521 |
| 32 | 3902744 | 21.714 | 2.236 | 4.053 | 937877 |
| 40 | 3903070 | 25.724 | 2.605 | 4.635 | 783943 |

Means pool samples across all rank/update windows. Longer selected windows contribute more samples. They are not an equal-weight average of the 80 window means.

The CSV also includes dispatch, combine, and optimizer-step selections. Their intervals overlap. No pooled quantile is reconstructed from window quantiles.

- Values are pooled arithmetic means of selected sample points, weighted by sample count, without time integration.
- Samples describe all device work during payload intervals, including concurrent work; communication-only activity is unavailable.
- Compute-warp throughput percentage is not a conversion to achieved occupancy or resident warps per SM.
- Payload block budgets do not denote reserved SMs or free SM capacity.
- Physical placement differs across independent allocations. These Nsight diagnostics establish no causal effect or speedup.
- Dispatch/combine selections overlap and are not additive partitions. Unsampled short intervals remain recorded.
- Whole-session event completeness remains unresolved; acceptance covers the known payload/call event set.

Input records:

- Budget 12: [uccl-budget12-hardware-final-20261109T181400Z](/iopsstor/scratch/cscs/anunay/swissai/post-training/ampa/runs/uccl-budget12-hardware-final-20261109T181400Z/metrics.json), SHA-256 `29e4c27105c27bfa000c8fc2bd65842b9bf8adb9c3453f7e0cdf758276d73be4`.
- Budget 20: [uccl-budget20-hardware-final-20261109T175146Z](/iopsstor/scratch/cscs/anunay/swissai/post-training/ampa/runs/uccl-budget20-hardware-final-20261109T175146Z/metrics.json), SHA-256 `889c09b26d22c59519140482f51e4f4337715f5cefbd8a5c9fc1c72610c13196`.
- Budget 32: [uccl-budget32-hardware-final-20261109T232451Z](/iopsstor/scratch/cscs/anunay/swissai/post-training/ampa/runs/uccl-budget32-hardware-final-20261109T232451Z/metrics.json), SHA-256 `3fb1ce4cafe3d18f73a7c16288d477c1775b6ab8b183445484be61da3cf26d67`.
- Budget 40: [uccl-budget40-hardware-final-20261109T232946Z](/iopsstor/scratch/cscs/anunay/swissai/post-training/ampa/runs/uccl-budget40-hardware-final-20261109T232946Z/metrics.json), SHA-256 `86077d0697f486c0501bbf0418f4a8ab9fff2557bda6f3bfc66a1e28d1460169`.
