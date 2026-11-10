# GPU metrics during Qwen communication

The saved runs reveal communication launch geometry, resource requirements and kernel durations. They do not reveal measured active-SM counts or achieved warp occupancy. The main UCCL dispatch and combine kernels launch 20 blocks on a GPU with 132 SMs. Native expert AllToAll kernels in the sampled updates launch four or eight blocks.

An SM is a GPU streaming multiprocessor. A warp contains 32 threads. Occupancy describes resident warps relative to capacity. A resident warp can wait or poll instead of issuing useful data-copy instructions. These quantities need separate labels.

## Evidence scope

The extraction reads one accepted profiler-ON trace for each configuration: native ordinary, native grouped, UCCL ordinary and UCCL grouped. Each trace covers rank 0 and human update 6, the first measured update. It does not sample all ranks or all measured updates.

Each raw trace hash matches its case acceptance. Each case proof matches the campaign proof. A separate read checks all 64 UCCL runtime records: two routing modes, OFF/ON and sixteen ranks. All record `Buffer.num_sms=20`, normal mode and valid identity.

| Artifact | Contents |
| --- | --- |
| [launch_metrics.json](launch_metrics.json) | Four bound trace identities, device properties, field coverage, estimator histograms, geometry/resource rows and 64 runtime records. |
| [communication_kernel_geometry.csv](communication_kernel_geometry.csv) | All 34 extracted communication kernel-name/geometry/resource combinations, counts and durations. |
| [Trace-field review](TRACE_FIELDS_REVIEW.md) | Independent inspection of two accepted captures and exact field meanings. |
| [UCCL launch evidence](UCCL_LAUNCH_EVIDENCE.md) | Pinned source, warp roles, clustered launches, helper kernels and budget qualifications. |
| [Metric semantics review](METRIC_SEMANTICS_REVIEW.md) | Primary-source definitions and a suitable future hardware-counter diagnostic. |
| [Extraction source](extract_launch_metrics.py) | Reproducible CPU-only extraction. It changes no acquisition or training code. |

The four traces contain 49,289, 51,401, 68,121 and 70,233 positive-duration kernel events respectively. Geometry and resource metadata also cover computation and audit kernels. The exported communication subset uses native process-group/collective labels and explicit UCCL symbols. It does not classify every NCCL kernel as expert communication.

## Recorded GPU capacity

The sampled device headers report `NVIDIA GH200 120GB`, compute capability 9.0, 132 SMs and 32 threads per warp. `maxThreadsPerMultiprocessor=2048` gives capacity for 64 warps per SM. The full GPU therefore has 8448 warp slots as a capacity calculation.

The headers report 65,536 registers per SM, 233,472 bytes of shared memory per SM and 232,448 opt-in shared bytes per block. The default per-block shared limit is 49,152 bytes. The recorded visible global memory is 102,005,473,280 bytes, or 95 GiB. That field is distinct from the product name.

These are capacity properties. They do not measure runtime clocks, power, temperature, utilization or memory traffic.

## Main communication kernels

The following geometry and resource values agree between the two routing modes for each backend in these sampled captures. A block contains the listed number of threads, grouped into warps. Shared memory is the trace's combined static-plus-dynamic resource value.

| Kernel | Blocks per launch | Threads per block | Launched warps per block | Registers per thread | Shared bytes per block |
| --- | ---: | ---: | ---: | ---: | ---: |
| Native EP AllToAll / SendRecv, variant A | 8 | 640 | 20 | 96 | 103,808 |
| Native EP AllToAll / SendRecv, variant B | 4 | 640 | 20 | 96 | 103,808 |
| UCCL dispatch | 20 | 384 | 12 | 96 | 82,944 |
| UCCL combine | 20 | 544 | 17 | 87 | 148,992 |

Each native sampled update has 768 eight-block and 384 four-block expert AllToAll kernels. Each UCCL sampled update has 384 dispatch and 384 combine payload kernels. Dispatch has two template variants with the same listed geometry. These counts include the forward and backward communication paths. A kernel name alone does not identify training direction.

The backends therefore use different launch budgets in these captures. That is another difference in the configurations as launched. It does not prove that the native configuration is wrong or that the block budget causes the timing gap. A separate budget sweep can test communication delay against GPU resource use while preserving correctness and transport identity.

For one kernel with N blocks, at most N distinct SMs can execute those blocks at the same instant. Thus the individual UCCL payload kernel has an upper bound of 20/132, or 15.15% of SMs. The native variants have upper bounds of 6.06% and 3.03%. These are bounds from launch geometry, not observations of active SMs.

The UCCL source requests cooperative launches and two-block clusters on SM90. Cluster attributes and residency affect scheduling. They do not turn the grid size into a measured SM count. Concurrent kernels can occupy other SMs, so none of these bounds describes all communication or whole-GPU activity.

The UCCL main-kernel budget is corroborated by the actual sampled grids. The 64 runtime JSON records alone describe the Buffer setting. Per-call configuration can override that setting, so a budget record does not establish every kernel launch throughout the campaign.

## UCCL helper kernels also use SMs

The 20-block payload setting is not a cap on all UCCL work. The same sampled traces also contain these helper launches:

| Helper | Blocks | Threads per block | Launched warps per block | Registers per thread | Shared bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| Routing layout | 34 | 256 | 8 | 40 | 14,336 |
| Initial dispatch notification | 5 | 512 | 16 | 54 | 0 |
| Cached notification | 20 | 320 | 10 | 34 | 81,920 |

Each UCCL sampled update contains 192 layout, 192 initial-notification and 576 cached-notification kernels. The layout's individual-kernel bound is 34 SMs, or 25.76% of SM count. It is metadata work rather than the main payload transfer.

Input fingerprints, routing/layout audits, permutations, expert computation and other communication can add their own kernels. Counting only the main dispatch grid does not describe every GPU operation during a communication phase.

## What the recorded warp values mean

The trace contains fields named `blocks per SM`, `warps per SM` and `est. achieved occupancy %`. The first two follow these launch-normalization formulas:

```text
blocks_per_SM = launched_blocks / 132
warps_per_SM = launched_blocks * launched_warps_per_block / 132
```

| Main kernel | Trace blocks/SM | Trace warps/SM | Launched warps / whole-GPU slot capacity |
| --- | ---: | ---: | ---: |
| Native eight-block variant | 0.060606 | 1.212121 | 1.89394% |
| Native four-block variant | 0.030303 | 0.606061 | 0.94697% |
| UCCL dispatch | 0.151515 | 1.818182 | 2.84091% |
| UCCL combine | 0.151515 | 2.575758 | 4.02462% |

The last column divides all launched warps by 8448 capacity slots. It is arithmetic about the launch. It assumes simultaneous residency to interpret it as a capacity fraction. It does not establish achieved occupancy or useful work. Neither do the fractional `warps per SM` values mean every SM runs that fraction of a warp.

UCCL also assigns different roles to different warps. In one dispatch block role, only five of twelve launched warps continue after the role branch. Seven return. Other blocks contain sender, coordinator and NVLink receiver roles. Combine uses sender, receiver, forwarder and coordinator roles. Resident and issuing warp counts can therefore change during a kernel. The pinned [launch investigation](UCCL_LAUNCH_EVIDENCE.md) gives exact source anchors.

All extracted main payload and NCCL kernel variants report an estimated occupancy value of zero. Layout reports 3% and initial notification reports 1%. Many computation kernels also have nonzero estimates. These values are estimates from launch/resource calculations, not hardware measurements. The zero values cannot support a claim of zero actual warp occupancy.

The upstream [Kineto implementation](https://github.com/pytorch/kineto/blob/20f652846f651fcae287f667d34bcf164c99f383/libkineto/src/DeviceProperties.cpp#L132-L203) computes the grid normalization and uses a static occupancy calculator. Its exporter rounds that estimate. The exact Kineto source revision inside the training image remains unidentified. A cause for these zero estimates is not established. An ordinary rounding explanation alone is insufficient for the 20-block examples.

## Available and missing GPU metrics

| Quantity | Existing evidence |
| --- | --- |
| GPU model, SM/warp capacity and memory limits | Recorded device properties. |
| Communication kernel count, duration, stream and host correlation | Recorded. Durations include waiting inside kernels and can overlap. |
| Grid, block, registers/thread and shared memory | Recorded resource and launch metadata. |
| UCCL configured budget and actual sampled payload grids | Recorded and corroborated. |
| Actual active SM count over time | Not recorded. |
| Measured active, eligible, issuing or stalled warps | Not recorded. |
| Achieved warp occupancy | Not recorded. Static estimates do not supply it. |
| Useful copying/reduction versus polling/waiting | Not separated. |
| SM instruction throughput or Tensor Core activity | Not recorded as hardware counters. |
| HBM/L2 traffic or achieved bandwidth | Not recorded as hardware counters. |
| Physical NVLink/NIC bandwidth and bytes | Not supplied by these captures. Logical payload counts are a different quantity. |
| Time-resolved clock, power, temperature or throttling | Not supplied by the sampled trace schema. |

All four traces contain zero Chrome counter events with `ph=C` and no separate hardware metric table. This is consistent with the capture configuration, which requests PyTorch activity tracing rather than hardware counters. The conclusion concerns these captures. It is not a claim that every possible external telemetry file was audited.

## What would measure the missing quantities

A short separate Nsight Systems capture with periodic GPU metrics can measure SM-active and instruction-issue activity over time, with supported warp metrics selected for the installed tool/device. Keep the original validated training data and sources. Use a new diagnostic run identity and retain clean multi-rank completion checks.

Device-wide counters need alignment with communication intervals. Concurrent compute or audit kernels make a sample mixed activity. A percentage of SM-active cycles is not an instantaneous list of physical SM IDs. Polling can count as active. NVIDIA explains these meanings in the [Nsight Systems GPU Metrics guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#gpu-metrics).

Kernel-level achieved occupancy and stall analysis can require Nsight Compute. Default replay and serialization can stop progress in distributed communication. The installed mode must support the required concurrent ranks. NVIDIA documents this in its [mandatory-concurrent-kernel guidance](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html#mandatory-concurrent-kernels). A default `ncu --set full` wrapper is not an appropriate first measurement for these jobs.

The [independent metric review](METRIC_SEMANTICS_REVIEW.md) gives the bounded design and compatibility requirements. It also distinguishes active-cycle from elapsed-cycle counter denominators. No new GPU capture or Slurm job ran in this investigation.

Few SMs occupied by communication does not establish a small communication delay. A kernel can wait for remote progress while later training depends on its completion. Conversely, high residency can reflect polling. The missing hardware counters and critical-path attribution prevent either conclusion from occupancy alone.
