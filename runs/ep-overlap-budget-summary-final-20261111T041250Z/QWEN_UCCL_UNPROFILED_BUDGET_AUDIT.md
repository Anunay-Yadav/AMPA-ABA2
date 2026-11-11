# Qwen UCCL unprofiled payload-budget audit

The sweep reconciles 20 full paired allocations and 40 independent C0 starts. The planner selected budget 40 by the lowest observed median ratio. The recorded ranges and stopping rules define the supported conclusion. They do not establish a global optimum.

Every case completed five warmup and five measured optimizer updates. Each candidate shares the same 16 physical GPUs with its own 20-block reference. Opposite execution orders and repeated allocations reduce placement and order effects. Each allocation is one replicate. The five measured updates within a leg are subsamples.

The workload uses Qwen3-30B-A3B-Base, ordinary top-eight routing, BF16 and EP16/CP1/TP1/PP1/ETP1. Each update consumes 64 samples of length 4096. Each run starts from the same C0 and consumes the same 640-sample prefix. The main expert-parallel overlap, model and DDP delayed weight gradients, and early attention memory release stay enabled. Immutable payload verification is reused. The actual consumed prefix is rechecked.

PyTorch and Nsight profilers are off. Common routing, receive, fingerprint and health observers remain enabled. Their costs are in the recorded timings. No SM activity, occupancy or free-SM measurement is available from this sweep.

The full-precision outputs live in `runs/ep-overlap-budget-summary-final-20261111T041250Z`. `optimizer_steps.csv` preserves every step and its health record. `legs.csv` contains per-case timing and wall intervals. `pairs.csv` contains the allocation times and paired ratios. `metrics.json` binds actual placement, source and input identities, accepted evidence and the planner decisions.

## Budget comparisons

A ratio below one means the candidate took less time than its same-allocation reference. The reference row defines ratio one; its individual 20-block timings remain in the leg and pair tables.

| Blocks | Candidate allocation replicates | Reference legs | Median ratio | Observed ratio range | Maximum within-leg step CV |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 3 | 0 | 1.04011 | 1.02426–1.06159 | 1.22% |
| 8 | 3 | 0 | 1.02053 | 0.99877–1.03927 | 0.93% |
| 12 | 3 | 0 | 0.99958 | 0.99691–1.02475 | 0.96% |
| 20 | 0 | 20 | 1.00000 | 1.00000–1.00000 | n/a |
| 32 | 3 | 0 | 1.00011 | 0.99513–1.00484 | 1.10% |
| 40 | 2 | 0 | 0.97850 | 0.97704–0.97995 | 1.97% |
| 48 | 3 | 0 | 0.99018 | 0.98600–0.99951 | 1.48% |
| 56 | 3 | 0 | 1.02269 | 0.92287–1.02872 | 0.79% |

## Allocation and case timings

The training elapsed time covers the single allocation shared by both cases. The case wall interval starts before its training launch and ends after its health/timing summary. It excludes shared setup and source archival. The optimizer timers cover the complete updates. The posthoc time is a separate allocation. There is no per-case Slurm elapsed time.

| Pair | Budget | Order | Train job | Posthoc job | Candidate mean (s) | Reference mean (s) | Ratio | Shared allocation (s) | Posthoc (s) |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ep-overlap-budget-off-adaptive-20261110T223433Z-b12-reference-first-r1-attempt1` | 12 | reference-first | 3911753 | 3911822 | 15.4721 | 15.5200 | 0.99691 | 758 | 26 |
| `ep-overlap-budget-off-adaptive-20261110T223433Z-b32-candidate-first-r1-attempt1` | 32 | candidate-first | 3911828 | 3911891 | 15.3749 | 15.3009 | 1.00484 | 748 | 22 |
| `ep-overlap-budget-off-adaptive-20261110T223433Z-b40-reference-first-r1-attempt1` | 40 | reference-first | 3911907 | 3911947 | 12.7151 | 13.0138 | 0.97704 | 704 | 28 |
| `ep-overlap-budget-off-adaptive-20261110T223433Z-b12-candidate-first-r2-attempt1` | 12 | candidate-first | 3911951 | 3912015 | 15.0092 | 14.6467 | 1.02475 | 743 | 29 |
| `ep-overlap-budget-off-adaptive-20261110T223433Z-b32-reference-first-r2-attempt2` | 32 | reference-first | 3912077 | 3912091 | 13.3368 | 13.3353 | 1.00011 | 712 | 32 |
| `ep-overlap-budget-off-adaptive-20261110T223433Z-b40-candidate-first-r2-attempt1` | 40 | candidate-first | 3912096 | 3912187 | 14.1257 | 14.4147 | 0.97995 | 738 | 32 |
| `ep-overlap-budget-off-adaptive-20261110T223433Z-b12-reference-first-r3-attempt1` | 12 | reference-first | 3912190 | 3912210 | 14.7701 | 14.7763 | 0.99958 | 738 | 36 |
| `ep-overlap-budget-off-adaptive-20261110T223433Z-b32-reference-first-r3-attempt1` | 32 | reference-first | 3912211 | 3912229 | 14.5826 | 14.6540 | 0.99513 | 746 | 37 |
| `ep-overlap-budget-off-adaptive-20261110T223433Z-b8-reference-first-r1-attempt1` | 8 | reference-first | 3912231 | 3912414 | 14.3113 | 14.0234 | 1.02053 | 731 | 39 |
| `ep-overlap-budget-off-adaptive-20261110T223433Z-b8-candidate-first-r2-attempt1` | 8 | candidate-first | 3912419 | 3912444 | 14.4944 | 14.5123 | 0.99877 | 744 | 39 |
| `ep-overlap-budget-off-adaptive-20261110T223433Z-b8-reference-first-r3-attempt1` | 8 | reference-first | 3912445 | 3912467 | 15.6819 | 15.0894 | 1.03927 | 756 | 42 |
| `ep-overlap-budget-off-adaptive-20261110T223433Z-b4-reference-first-r1-attempt1` | 4 | reference-first | 3912476 | 3912494 | 14.0800 | 13.2631 | 1.06159 | 703 | 44 |
| `ep-overlap-budget-off-adaptive-20261110T223433Z-b4-candidate-first-r2-attempt1` | 4 | candidate-first | 3912495 | 3912528 | 14.9434 | 14.5896 | 1.02426 | 755 | 48 |
| `ep-overlap-budget-off-adaptive-20261110T223433Z-b4-reference-first-r3-attempt1` | 4 | reference-first | 3912544 | 3912624 | 15.1099 | 14.5272 | 1.04011 | 745 | 46 |
| `ep-overlap-budget-off-adaptive-20261110T223433Z-b48-reference-first-r1-attempt1` | 48 | reference-first | 3912805 | 3912930 | 13.7345 | 13.9295 | 0.98600 | 714 | 48 |
| `ep-overlap-budget-off-adaptive-20261110T223433Z-b48-candidate-first-r2-attempt1` | 48 | candidate-first | 3912931 | 3912968 | 14.5902 | 14.7350 | 0.99018 | 750 | 50 |
| `ep-overlap-budget-off-adaptive-20261110T223433Z-b48-reference-first-r3-attempt1` | 48 | reference-first | 3912973 | 3913001 | 13.1704 | 13.1768 | 0.99951 | 712 | 54 |
| `ep-overlap-budget-off-adaptive-20261110T223433Z-b56-reference-first-r1-attempt1` | 56 | reference-first | 3913007 | 3913038 | 13.9664 | 15.1336 | 0.92287 | 729 | 57 |
| `ep-overlap-budget-off-adaptive-20261110T223433Z-b56-candidate-first-r2-attempt1` | 56 | candidate-first | 3913039 | 3913057 | 14.4355 | 14.1152 | 1.02269 | 751 | 58 |
| `ep-overlap-budget-off-adaptive-20261110T223433Z-b56-reference-first-r3-attempt1` | 56 | reference-first | 3913062 | 3913087 | 14.7901 | 14.3772 | 1.02872 | 726 | 59 |

| Case | Blocks | Case wall (s) | Ten timers sum (s) | Measured mean (s) | Measured median (s) | Measured SD (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ep-overlap-budget-off-adaptive-20261110T223433Z-b12-reference-first-r1-attempt1-reference-b20` | 20 | 382.219 | 188.7441 | 15.5200 | 15.4721 | 0.0986 |
| `ep-overlap-budget-off-adaptive-20261110T223433Z-b12-reference-first-r1-attempt1-candidate-b12` | 12 | 365.501 | 186.6499 | 15.4721 | 15.5005 | 0.0613 |
| `ep-overlap-budget-off-adaptive-20261110T223433Z-b32-candidate-first-r1-attempt1-candidate-b32` | 32 | 374.539 | 186.3826 | 15.3749 | 15.3808 | 0.0322 |
| `ep-overlap-budget-off-adaptive-20261110T223433Z-b32-candidate-first-r1-attempt1-reference-b20` | 20 | 363.420 | 185.2658 | 15.3009 | 15.2590 | 0.1684 |
| `ep-overlap-budget-off-adaptive-20261110T223433Z-b40-reference-first-r1-attempt1-reference-b20` | 20 | 355.109 | 162.7856 | 13.0138 | 12.9612 | 0.2570 |
| `ep-overlap-budget-off-adaptive-20261110T223433Z-b40-reference-first-r1-attempt1-candidate-b40` | 40 | 337.799 | 159.7683 | 12.7151 | 12.7898 | 0.1564 |
| `ep-overlap-budget-off-adaptive-20261110T223433Z-b12-candidate-first-r2-attempt1-candidate-b12` | 12 | 380.762 | 182.2652 | 15.0092 | 14.9915 | 0.0647 |
| `ep-overlap-budget-off-adaptive-20261110T223433Z-b12-candidate-first-r2-attempt1-reference-b20` | 20 | 352.014 | 177.5387 | 14.6467 | 14.6436 | 0.0527 |
| `ep-overlap-budget-off-adaptive-20261110T223433Z-b32-reference-first-r2-attempt2-reference-b20` | 20 | 356.513 | 165.0811 | 13.3353 | 13.3258 | 0.0608 |
| `ep-overlap-budget-off-adaptive-20261110T223433Z-b32-reference-first-r2-attempt2-candidate-b32` | 32 | 340.425 | 164.0872 | 13.3368 | 13.2871 | 0.1138 |
| `ep-overlap-budget-off-adaptive-20261110T223433Z-b40-candidate-first-r2-attempt1-candidate-b40` | 40 | 368.287 | 174.7531 | 14.1257 | 14.1048 | 0.1035 |
| `ep-overlap-budget-off-adaptive-20261110T223433Z-b40-candidate-first-r2-attempt1-reference-b20` | 20 | 354.520 | 175.7919 | 14.4147 | 14.3941 | 0.1045 |
| `ep-overlap-budget-off-adaptive-20261110T223433Z-b12-reference-first-r3-attempt1-reference-b20` | 20 | 373.382 | 180.7069 | 14.7763 | 14.7583 | 0.1424 |
| `ep-overlap-budget-off-adaptive-20261110T223433Z-b12-reference-first-r3-attempt1-candidate-b12` | 12 | 353.764 | 179.4473 | 14.7701 | 14.7052 | 0.1244 |
| `ep-overlap-budget-off-adaptive-20261110T223433Z-b32-reference-first-r3-attempt1-reference-b20` | 20 | 379.236 | 180.1929 | 14.6540 | 14.6421 | 0.0767 |
| `ep-overlap-budget-off-adaptive-20261110T223433Z-b32-reference-first-r3-attempt1-candidate-b32` | 32 | 355.718 | 177.5217 | 14.5826 | 14.5838 | 0.1275 |
| `ep-overlap-budget-off-adaptive-20261110T223433Z-b8-reference-first-r1-attempt1-reference-b20` | 20 | 364.003 | 173.2621 | 14.0234 | 14.0127 | 0.1054 |
| `ep-overlap-budget-off-adaptive-20261110T223433Z-b8-reference-first-r1-attempt1-candidate-b8` | 8 | 353.462 | 175.5378 | 14.3113 | 14.3472 | 0.0826 |
| `ep-overlap-budget-off-adaptive-20261110T223433Z-b8-candidate-first-r2-attempt1-candidate-b8` | 8 | 370.354 | 176.0986 | 14.4944 | 14.4756 | 0.0749 |
| `ep-overlap-budget-off-adaptive-20261110T223433Z-b8-candidate-first-r2-attempt1-reference-b20` | 20 | 357.503 | 175.8431 | 14.5123 | 14.5908 | 0.1354 |
| `ep-overlap-budget-off-adaptive-20261110T223433Z-b8-reference-first-r3-attempt1-reference-b20` | 20 | 375.122 | 184.1637 | 15.0894 | 15.0966 | 0.0549 |
| `ep-overlap-budget-off-adaptive-20261110T223433Z-b8-reference-first-r3-attempt1-candidate-b8` | 8 | 367.473 | 189.4359 | 15.6819 | 15.7182 | 0.0787 |
| `ep-overlap-budget-off-adaptive-20261110T223433Z-b4-reference-first-r1-attempt1-reference-b20` | 20 | 348.645 | 162.9764 | 13.2631 | 13.2528 | 0.0930 |
| `ep-overlap-budget-off-adaptive-20261110T223433Z-b4-reference-first-r1-attempt1-candidate-b4` | 4 | 343.641 | 169.7844 | 14.0800 | 14.0531 | 0.1531 |
| `ep-overlap-budget-off-adaptive-20261110T223433Z-b4-candidate-first-r2-attempt1-candidate-b4` | 4 | 383.553 | 181.8695 | 14.9434 | 14.9460 | 0.0848 |
| `ep-overlap-budget-off-adaptive-20261110T223433Z-b4-candidate-first-r2-attempt1-reference-b20` | 20 | 355.325 | 177.2844 | 14.5896 | 14.5883 | 0.0707 |
| `ep-overlap-budget-off-adaptive-20261110T223433Z-b4-reference-first-r3-attempt1-reference-b20` | 20 | 368.163 | 180.4952 | 14.5272 | 14.4831 | 0.1518 |
| `ep-overlap-budget-off-adaptive-20261110T223433Z-b4-reference-first-r3-attempt1-candidate-b4` | 4 | 362.953 | 184.0978 | 15.1099 | 15.0570 | 0.1839 |
| `ep-overlap-budget-off-adaptive-20261110T223433Z-b48-reference-first-r1-attempt1-reference-b20` | 20 | 358.676 | 172.6700 | 13.9295 | 13.8747 | 0.2055 |
| `ep-overlap-budget-off-adaptive-20261110T223433Z-b48-reference-first-r1-attempt1-candidate-b48` | 48 | 340.926 | 168.0162 | 13.7345 | 13.7447 | 0.0787 |
| `ep-overlap-budget-off-adaptive-20261110T223433Z-b48-candidate-first-r2-attempt1-candidate-b48` | 48 | 370.664 | 180.9141 | 14.5902 | 14.5813 | 0.0449 |
| `ep-overlap-budget-off-adaptive-20261110T223433Z-b48-candidate-first-r2-attempt1-reference-b20` | 20 | 366.437 | 180.7583 | 14.7350 | 14.7285 | 0.0566 |
| `ep-overlap-budget-off-adaptive-20261110T223433Z-b48-reference-first-r3-attempt1-reference-b20` | 20 | 356.940 | 163.3194 | 13.1768 | 13.1875 | 0.1426 |
| `ep-overlap-budget-off-adaptive-20261110T223433Z-b48-reference-first-r3-attempt1-candidate-b48` | 48 | 341.988 | 162.0822 | 13.1704 | 13.2003 | 0.1000 |
| `ep-overlap-budget-off-adaptive-20261110T223433Z-b56-reference-first-r1-attempt1-reference-b20` | 20 | 373.154 | 183.8266 | 15.1336 | 15.1261 | 0.0620 |
| `ep-overlap-budget-off-adaptive-20261110T223433Z-b56-reference-first-r1-attempt1-candidate-b56` | 56 | 345.441 | 170.9118 | 13.9664 | 13.9513 | 0.0549 |
| `ep-overlap-budget-off-adaptive-20261110T223433Z-b56-candidate-first-r2-attempt1-candidate-b56` | 56 | 371.889 | 178.3930 | 14.4355 | 14.4735 | 0.1144 |
| `ep-overlap-budget-off-adaptive-20261110T223433Z-b56-candidate-first-r2-attempt1-reference-b20` | 20 | 364.408 | 175.5411 | 14.1152 | 14.1431 | 0.1092 |
| `ep-overlap-budget-off-adaptive-20261110T223433Z-b56-reference-first-r3-attempt1-reference-b20` | 20 | 364.128 | 176.2619 | 14.3772 | 14.3589 | 0.1098 |
| `ep-overlap-budget-off-adaptive-20261110T223433Z-b56-reference-first-r3-attempt1-candidate-b56` | 56 | 352.327 | 178.9271 | 14.7901 | 14.7832 | 0.0672 |

## Boundary decisions and limits

The predeclared 2% threshold is an operational rule. It does not give a significance test or a confidence interval. The planner uses the observed replicate ranges, order coverage and within-case variation. It adds a third pair when the rule requires it.

- Lower boundary: budget 4, compared with 40. The minimum ratio 1.02426 exceeds the incumbent's upper ratio times 1.02 (0.99955).
- Upper boundary: budget 56, compared with 40. The tested boundary reaches the qualified limit. The minimum ratio 0.92287 does not exceed 0.99955.

The qualified range is 4–56 even blocks. The allocated device reports 232,448 bytes of opt-in shared memory per block. These resource checks do not measure the number of free SMs or prove successful execution at every untested point.

The native cached notify path requires more than three blocks. The budget must be even, so four is the lower limit. The bound follows from `num_channels * 2 > 3` in `uccl/ep/src/internode.cu` and `num_channels = budget / 2`.

Budget 58 would request 237,568 dynamic shared bytes per notify block. This exceeds the measured 232,448-byte device limit.

The sweep excludes its two-update qualification and 1 failed full-pair attempts. Failures and incomplete acquisitions do not count as measured degradation. The input, health, source, routing and native budget checks support the stated short-run timings. They do not prove long-run quality or remove network, cache and clock variation.

Training source SHA-256: `252ef8a09ddb96efef52ceebf339a52287e2dc48422f0a9fa472f5d41fc76927`. Consumed input prefix SHA-256: `6ce98c488cce7b047566c233fe93567d3d49905432e9b1709d6b07a4fd0514cb`.

The generated report requires independent interpretation review before promotion to the project documentation.
