# Native timing interpretation

The accepted data do not show a new large native slowdown caused by enabling the overlap bundle. Native updates near 65 seconds already appear in the original overlap-OFF baseline. This review found no blocking correctness defect in the new OFF capture path. It does not identify the physical cause of the native/UCCL timing gap.

## Recorded timing evidence

These are the recorded means for updates 6–10, in milliseconds. The old values come from the [original accepted cases](../../../runs/qwen-debug-results-20261105T0925Z-cpu-validation/cases/). The new values come from the corresponding accepted `<case>-attempt1-posthoc/timing.json` files under `runs/ep-overlap-auto-retry-20261110T165819Z-*`. Both campaigns retain one allocation per cell. These are descriptive values, pending the complete old/new input comparison by the lead.

| Backend and routing | Capture | Original overlap OFF | New overlap ON |
| --- | --- | ---: | ---: |
| Native ordinary | OFF | 64843.64 | 62468.30 |
| Native ordinary | Torch | 71281.04 | 72978.24 |
| Native grouped | OFF | 65422.08 | 65579.06 |
| Native grouped | Torch | 74355.22 | 75185.60 |
| UCCL ordinary | OFF | 15553.56 | 14186.34 |
| UCCL ordinary | Torch | 30496.58 | 30291.98 |
| UCCL grouped | OFF | 9859.58 | 10250.18 |
| UCCL grouped | Torch | 25007.30 | 25957.84 |

The new native ordinary qualification also records 61679.9 ms for its second update. Its [timing record](../../../runs/ep-overlap-debug-ddp-pilot-native-ordinary-off-20261110T111819Z-posthoc/timing.json) uses a different allocation from the full run. The full run records 62468.3 ms across its five measured updates. This agreement argues against a new isolated node incident as the sole explanation. It does not prove identical network conditions.

## OFF capture and execution path

The [capture callback](../../../profiling/ampa_profile/overlap_capture.py:141) returns before constructing a labeler or profiler when the step is outside `capture_steps`. OFF sets that list to empty. The [actual rank15 OFF capture record](../../../runs/ep-overlap-auto-retry-20261110T165819Z-native-ordinary-off-attempt1-training/raw/ep_overlap_rank-15_capture.json) records ten completed updates, `capture_steps=[]`, and `boundaries=[]`. The [entrypoint](../../../profiling/run_pretrain_ep_overlap.py:362) suppresses the historical deep-trace callback and inserts this capture owner. I found no OFF branch that attaches the new trace wrappers, starts Kineto, exports traces, or calls the Nsight synchronization boundary.

OFF still includes the common routing, receive and input-fingerprint recorders. The [routing audit](../../../profiling/ampa_profile/routing_audit.py:365) reads GPU accounting tensors at step end and performs host validation. Its native receive hook reads CPU counts that the dispatcher already produced. The [fingerprint callback](../../../profiling/ampa_profile/sample_fingerprint.py:54) hashes CPU batches before transfer and adds no CUDA copy or synchronization. These observations preserve execution semantics but have costs. OFF is not observer-free production training. The [original audit](../../../docs/QWEN_UCCL_EXPERIMENT_AUDIT.md:269) already states this limit, including backend-specific audit work.

The actual native [execution proof](../../../runs/ep-overlap-auto-retry-20261110T165819Z-native-ordinary-off-attempt1-posthoc/execution_accounting.json) records backend `alltoall`, ten healthy updates and accepted send/receive accounting. The resolved configuration selects `moe_token_dispatcher_type=alltoall`. Startup records actual model overlap, model delay, DDP delay and CUDA connections32. The pinned [scheduler](../../../Megatron-LM/megatron/core/pipeline_parallel/schedules.py:779) selects the combined forward/backward schedule when main EP overlap is true. Its [implementation](../../../Megatron-LM/megatron/core/pipeline_parallel/combined_1f1b.py:38) schedules F0, F1+B0, F2+B1, F3+B2 and B3. The accepted Torch call record contains all five returned passes. This is evidence of the selected native schedule, with no observed fallback to the ordinary schedule or UCCL.

The representative original native [environment](../../../runs/uccl-debug5x5-reuse-20261104T1615Z-ordinary-control-ordinary-control/environment.json) and new native [environment](../../../runs/ep-overlap-auto-retry-20261110T165819Z-native-ordinary-off-attempt1-training/environment.json) agree on the following identities. Their common runtime-source hash is `409c26f60171efc0ea54aba24d9a544fc1a3a0672b4556b3ab69edee76def400`. Their container hash, NCCL 2.27.7, framework revisions and recorded dirty-source identities also agree. The complete new measurement program adds separately bound adapter and observer sources. The intended overlap flags and CUDA connection limit also change, so the complete programs are not identical.

## Diagnostic activity and remaining uncertainty

The accepted native Torch [step5/rank15 GPU result](../../../runs/ep-overlap-auto-retry-20261110T165819Z-native-ordinary-torch-attempt1-posthoc/gpu_activity/step-00005-rank-15.json) records 32132.0855 ms of forward payload interval union. Step index 5 is human update 6. The record contains 576 known forward payload kernels, 204 selected computation kernels, and 0.141819 ms of their interval intersection. The full native ordinary summary records positive selected concurrency in 116 of 160 captures. This supports some actual same-device concurrency. It does not measure all computation, backward native payloads, exposed communication latency or a partition of the optimizer timer. Autograd worker-thread compute without exact ancestry remains excluded. Small selected intersections therefore do not prove that all actual overlap is small.

The same representative call record records 1.3311 seconds in profiler stop and 7.7679 seconds in trace export. The [callback](../../../profiling/ampa_profile/overlap_capture.py:183) performs both before the later Bridge [timing/logging path](../../../Megatron-Bridge/src/megatron/bridge/training/train.py:518). These recorded costs explain why Torch timers include diagnostic work. They do not permit subtraction to recover unprofiled throughput. Allocation differences prevent assigning the complete OFF/ON difference to these two fields.

The native dispatcher calls the pinned [AllToAll mapping](../../../Megatron-LM/megatron/core/tensor_parallel/mappings.py:428) through the existing NCCL process group. The environment records NCCL version, not proof of the selected native network provider or a tuned baseline. This review therefore cannot attribute the long native payload activity to TCP fallback, a CXI defect, poor channels, SM contention, or a physical bandwidth limit. It found no runtime transport evidence that establishes any of those causes.

The final report must retain separate allocations, the full configuration bundle, backend-specific audit costs, partial GPU attribution and profiler boundary/export costs. It must use OFF complete-step timings as primary descriptive timing evidence. It must not infer a tuned-backend speedup, free SM capacity, achieved occupancy, physical traffic or a causal overlap benefit from these records. Final campaign acceptance and complete old/new input equality remain the lead's responsibility.

This review used file inspection and small JSON metadata reads. It ran no Python, tests, raw-trace parsing, workload or scheduler command. All original and frozen acquisition sources remain unchanged.
