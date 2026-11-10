# Reproducing the Qwen measurements

This note covers input preparation, real training, measurement collection, validation, and numerical extraction.
The user narrowed the scope to obtaining the measurements on 2026-11-09.
The accepted experiment contains eight independent starts from the same initial checkpoint, called C0.

The [complete source list](REQUIRED_FILES.txt) now covers the original measurements and the later SM/Nsight experiment.
It contains 123 existing files: 107 source/setup/schema files and 16 tests required by the included source checks or build/preflight jobs.
The [annotated file guide](FILE_GUIDE.md) explains what every file does and why it is included.
Four optional tests from the earlier list are excluded, and eight directly required test dependencies are added.
It excludes presentation generators and generated binaries or measurement data.
The additional SM sources are listed separately below.

## Workload and experiment matrix

The workload uses Qwen/Qwen3-30B-A3B-Base at revision `1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9`.
It runs on four GH200 nodes with four GPUs each.
Expert parallelism spans sixteen ranks, with context, tensor, and pipeline parallelism each set to one.
The model has 48 MoE layers, 128 experts per layer, and top-8 routing.

| Setting | Value |
| --- | --- |
| Model precision | BF16 |
| Sequence length | 4096 |
| Microbatch size | 1 |
| Gradient accumulation | 4 |
| Global batch size | 64 |
| Input tokens per optimizer update | 262,144 |
| Seed | 1234 |
| Executed updates | 10 complete forward/backward/optimizer updates |
| Timing window | Updates 6–10, after five warmup updates |
| Dataset/cache budget | Original 25-update / 1600-sample cache, consuming its first 640 samples |
| Activation recomputation | Disabled |
| MoE AllToAll overlap | Disabled |
| Dense data-parallel overlap | Enabled |

Run native Megatron and UCCL with ordinary and grouped routing.
For each combination, run once with the PyTorch trace callback OFF and once with it ON.
Grouped routing selects two of four groups before top-8 expert selection.
Every run starts independently from C0 and replays the same input prefix.

The accepted case/job map is [acquisitions.tsv](../../../runs/uccl-debug5x5-reordered-20261104T2221Z-reordered-coordinator/acquisitions.tsv).
It names orchestration directories whose `cases.tsv` files name the actual training directories.
The complete saved configuration and source/data identities are in [QWEN_UCCL_EXPERIMENT_AUDIT.md](../../../docs/QWEN_UCCL_EXPERIMENT_AUDIT.md).
That audit also records effective scheduled weight decay `0.033`, despite the optimizer field declaring `0.1`.

## Input and environment files

Prepare each input once and preserve it across comparable runs.
The image contains dependencies, while launchers overlay the mounted Bridge and Megatron source trees.
Retain the input manifests and hashes alongside every acquisition.

| Purpose | Files |
| --- | --- |
| Training image and container definition | `scripts/build_gh200_image.sbatch`, `setup/Dockerfile.gh200-cuda130`, `setup/gh200-cuda130-runtime.toml` |
| Model architecture and runtime workload | `configs/model/qwen3_30b_a3b_base.json`, `tools/qwen_workload.py`, `profiling/run_pretrain.py` |
| Pinned HF snapshot | `jobs/checkpoint/materialize_qwen3_30b_a3b_base_hf.sbatch`, `tools/materialize_hf_snapshot.py` |
| One-time Megatron C0 import and inventory | `jobs/checkpoint/import_qwen3_30b_a3b_base_c0.sbatch`, `tools/write_c0_manifest.py`, `tools/c0_validation.py` |
| Select text and tokenize indexed components | `jobs/dataset/materialize_qwen3_30b_a3b_base_mix.sbatch`, `tools/materialize_indexed_mixture.py` |
| Check indexed components | `jobs/dataset/validate_qwen3_30b_a3b_base_mix.sbatch`, `tools/validate_indexed_mixture.py` |
| Prepare and check the sample cache | `jobs/dataset/prepare_qwen3_30b_a3b_base_mix_cache.sbatch`, `validate_qwen3_30b_a3b_base_mix_cache.sbatch`, `probe_qwen3_30b_a3b_base_mix_cache_hit.sbatch`, `tools/megatron_cache_stage.py`, `tools/megatron_cache_contract.py` |
| Promote reviewed cache manifests | `jobs/dataset/promote_qwen3_30b_a3b_base_mix_manifest.sbatch`, `tools/promote_indexed_manifest.py`, `tools/promote_cp2_feasibility_manifest.py` |
| Produce the EP16 runtime derivative | `tools/ep16_indexed_manifest.py`, `jobs/smoke/ep16_profile_preflight.sbatch` |
| Build the pinned UCCL extension and wrapper | `jobs/smoke/uccl_ep_build.sbatch`, `tools/uccl_ep_source_identity.py`, `ai/qwen-uccl-ep-comparison/iteration-01/02-implement/uccl-borrowed-scratch.patch` |

The data mixture is 55% FineWeb, 15% multilingual FineWeb-2, 15% code, and 15% OpenWebMath.
The preparation selects deterministic subsets and tokenizes them once with the pinned Qwen tokenizer.
It appends EOD ID 151643 after documents and writes int32 indexed tokens.
The materialized subset contains 14,217 documents and 13,146,588 tokens including EOD.
Its selection ledgers and manifests preserve the exact upstream revisions, extraction rules, hashes, and order.

The checkpoint is `checkpoint/megatron/qwen3-30b-a3b-base-c0/`.
The indexed component prefixes are under `dataset/indexed/qwen3-30b-a3b-base-realistic-mix-v2/`.
The accepted runtime manifest is `dataset/manifests/ep16-cache-20261011T1757Z-cp1-ep16.json`.
The historical cache chain includes a CP2/EP8 parent, then a checked CP1/EP16 derivative.
Do not rebuild a shorter cache for the ten-update run, because its sample order can differ.

The pinned Bridge revision is `3a459fe4ab5e5e6b6c489434c687cbc4a01a55b3`.
The pinned Megatron-LM revision is `e2c2e55bfe819e7d7b74306a042e30289e5ddfd3`.
The pinned UCCL revision is `c8fd14e93f2106b69a0d30e0f78c12a52e07a181` plus the recorded ownership patch.
The recorded runtime uses PyTorch `2.8.0a0+34c6371d24.nv25.08`, CUDA 13.0, and NCCL 2.27.7.

## Launch and training files

The first acquisition verifies C0, data, cache, source, and native build before training.
The seven later accepted acquisitions explicitly reuse that proof and retain all measurement observers.
The reuse path records that it did not repeat bulk payload verification in each allocation.

| File | Responsibility |
| --- | --- |
| `jobs/training/uccl_ep_debug_case.sbatch` | First four-node allocation with fresh input verification. |
| `jobs/training/qwen3_uccl_debug_worker.sh` | Starts per-node torchrun and the first-case training entrypoint. |
| `jobs/training/uccl_ep_debug_reuse.sbatch` | Later four-node allocations using the recorded first-case proof. |
| `jobs/training/qwen3_uccl_debug_reuse_worker.sh` | Starts the later workers and chooses backend, routing, and profiler flags. |
| `tools/uccl_ep_verified_inputs_reuse.py` | Binds historical input verification to the current job and run. |
| `profiling/run_pretrain_debug.py` | Keeps `train_iters=25` and sets native `exit_interval=10`. |
| `profiling/run_pretrain_debug_reuse.py` | Reuses immutable-input metadata during parsing, then runs the same short training path. |
| `profiling/run_pretrain.py` | Builds the model/training configuration, attaches observers, and invokes native Bridge pretrain. |

Each allocation runs one Slurm task per node with four visible GPUs.
Its `srun` uses the configured container, `--mpi=pmix`, and `--cpu-bind=none`.
Each node runs `torch.distributed.run --nnodes=4 --nproc_per_node=4`.
The resulting sixteen workers execute the same checkpoint/data/batch configuration.

The worker selects these measurement and case flags:

```text
All cases:
  --sample-fingerprints --routing-audit --audit-num-groups 4
  --audit-received-experts --launch-policy torchrun_node4
  --topology-variant cp1_ep16 --lightweight-profiling off
  --recompute-mode none --sequence-length 4096
  --micro-batch-size 1 --global-batch-size 64
  --warmup-steps 5 --measured-steps 5 --log-interval 1
  --dataset-sampler-seed 1234

Grouped routing:
  --router-num-groups 4 --router-group-topk 2

UCCL:
  --communication-backend uccl_ep --uccl-build-dir BUILD

PyTorch profiler ON:
  --torch-profile-all-steps
  --torch-profile-ranks 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
```

All UCCL cases set `FI_MR_CACHE_MONITOR=disabled` before native initialization.
They select CXI transport and use the normal high-throughput Buffer through the existing Flex/DeepEP interface.
The build uses `USE_LIBFABRIC_CXI=1`, `NUM_MAX_NVL_PEERS=4`, and `TORCH_CUDA_ARCH_LIST=9.0`.

Historical scheduler coordinators are bookkeeping helpers for this campaign.
They contain fixed job IDs and historical proof paths.
The acquisition workers and entrypoints contain the actual training and observation path.

## Collectors, outputs, and extraction

Complete optimizer-step time comes from the native interval timer.
Trace phase durations provide separate diagnostic measurements.
Logical routing counters describe assignments and destinations.

| Measurement | Files that collect or extract it | Output |
| --- | --- | --- |
| Complete update time and throughput | Normal Bridge logger, `analysis/extract_step_timing.py` | Unique timer-bearing `logs/rank-*.out`, `metrics/timing_windows.parquet`, `analysis/optimizer_step_breakdown.json` |
| Loss, LR, gradient norm, skipped/NaN updates | Normal logger, `analysis/summarize_training_evidence.py` | `analysis/real_training_evidence.json` |
| Input-batch equality | `profiling/ampa_profile/sample_fingerprint.py` | `raw/sample_fingerprint_rank-R.json` |
| Expert/rank assignments, locality, received counts, unique UCCL destinations | `profiling/ampa_profile/routing_audit.py` | `raw/routing_audit_rank-R.json`, `config/routing_ownership.json` |
| CPU scopes, correlated GPU activity, forward/backward call journals | `profiling/ampa_profile/deep_trace.py`, `uccl_trace.py` | `traces/torch_profile/step-SSSSS/rank-R.json.gz`, `raw/deep_trace/step-SSSSS/rank-R.json` |
| Actual placement, topology, source, checkpoint, and dataset identity | `profiling/ampa_profile/provenance.py` | `environment.json` and recorded configuration/proofs |
| UCCL binary/fabric, transport, and cleanup | `profiling/ampa_profile/uccl_identity.py`, `tools/uccl_ep_log_proof.py` | Backend startup/cleanup records and native log proofs |

Both profiler OFF and ON retain fingerprint, routing, and received-count observers.
The controls therefore measure training with those observers attached.
Profiler ON captures all ten updates on all sixteen ranks.
That produces 160 rank-update captures per ON case and 640 across the experiment.

Use `tools/uccl_debug_campaign_validate.py` to drive the eight-case checks.
It uses `tools/uccl_ep_debug_validate.py`, `analysis/validate_torch_trace.py`, and `analysis/aggregate_deep_trace.py` for native captures.
It uses `analysis/inspect_uccl_trace.py` for UCCL captures.
Assignment and receive checks use `analysis/validate_ep16_received_routes.py`.
The dependency chain also includes `accept_uccl_pilot.py`, `compare_uccl_profiles.py`, `compare_deep_trace_overhead.py`, and `compare_profile_overhead.py`.

For a combined numerical JSON/CSV, use `analysis/qwen_debug_results.py` after campaign acceptance.
Its helper `analysis/qwen_debug_profile_phases.py` averages router, dispatch, expert, and combine activity over ranks 0–15 and updates 6–10.
`analysis/qwen_debug_independent_audit.py` independently reconstructs the numerical claims from the underlying records.

The posthoc payload is:

```bash
# Run inside a fresh allocated node using setup/gh200-cuda130-runtime.toml.
# CASE_LIST names the eight acquisitions. VALIDATION and RESULTS are new absolute runs/ paths.
export PYTHONPATH="$PWD:$PWD/profiling:$PWD/analysis${PYTHONPATH:+:$PYTHONPATH}"
TRAIN_PY=/opt/venvs/megatron-bridge/bin/python
"$TRAIN_PY" tools/uccl_debug_campaign_validate.py \
  --case-list "$CASE_LIST" --output-dir "$VALIDATION" --workers 4 --metadata-only
"$TRAIN_PY" tools/uccl_debug_campaign_validate.py \
  --case-list "$CASE_LIST" --output-dir "$VALIDATION" --workers 4
"$TRAIN_PY" analysis/qwen_debug_results.py \
  --case-list "$CASE_LIST" --validation-root "$VALIDATION" --output-dir "$RESULTS"
```

The result is `RESULTS/report.json` and `RESULTS/timing.csv`.
The full timing-extraction CLI requires `pyarrow` for its Parquet output.
Require finite health, no failed/skipped update, identical input prefixes, conserved assignments, and complete correlated captures before accepting results.

## Additional SM/Nsight experiment files

The later experiment tests UCCL communication block budgets 12, 20, 32, and 40 with ordinary top-8 routing.
It preserves the same C0, data prefix, batch, topology, and five-warmup/five-measured protocol.
Nsight owns CUDA tracing, and Kineto is disabled.
Each run captures updates 6–10 with one collector per node and produces four native reports.

```text
profiling/run_pretrain_uccl_budget.py

jobs/training/uccl_budget_sweep.sbatch
jobs/training/uccl_budget_sweep_worker.sh
jobs/training/uccl_budget_sweep_coordinator.sh

jobs/smoke/uccl_nsight_cuda132_setup.sbatch
jobs/smoke/uccl_nsight_capabilities.sbatch
jobs/smoke/uccl_budget_runtime_checks.sbatch
jobs/analysis/uccl_budget_sweep_posthoc.sbatch

tools/check_uccl_budget_runtime.py
tools/qualify_uccl_budget_capture.py
tools/uccl_budget_sweep_posthoc.py

analysis/uccl_budget_hardware_metrics.py
analysis/uccl_budget_sweep_delivery.py
analysis/uccl_budget_hardware_compare.py

tests/test_uccl_budget_entrypoint.py
ai/qwen-uccl-block-sweep/iteration-01/HARDWARE_METRIC_SCHEMA.md
```

The adapter sets the budget before native Buffer allocation and records the effective dispatch/combine configuration and size hints.
It adds NVTX labels and CUDA profiler start/stop boundaries around the five measured updates.
The worker wraps four torchrun children per node in a single Nsight collector.
The posthoc helper checks health, batches, routing, native budgets, callback records, and exact label/call/kernel correlations.

`analysis/uccl_budget_hardware_metrics.py` extracts sampled SM activity, SM issue, and compute-warps-in-flight throughput percentages.
The delivery and comparison helpers also check workload/source/input/native identities and accepted report bindings.
Their validation role remains relevant even if chart outputs are unnecessary.
The older `ai/qwen-gpu-communication-metrics/iteration-01/extract_launch_metrics.py` extracts kernel geometry from the original PyTorch traces instead.
That older extractor is also included in REQUIRED_FILES.txt.

The completed jobs were 3901926, 3902094, 3902744, and 3903070, in budget order 20/12/32/40.
Their acquisition map is `runs/uccl-budget4-tracked-20261109T165740Z-coordinator/acquisitions.tsv`.
The four jobs produced sixteen native reports.
The frozen 74-file sweep manifest is `runs/uccl-budget4-source-20261109T173624Z/expected-source.sha256`.
The 123-file list includes those entries plus later hardware extraction/comparison and the original campaign/preparation dependencies.

Two inventory-bound wrappers need historical qualification.
The sweep coordinator still requires strict QA success, while the original QA job 3901242 remains failed.
The actual four acquisitions used lead-tracked submissions with a separately qualified capture proof.
The posthoc batch is an allocated wrapper, while accepted posthoc analysis called the helpers directly under historical CPU authorization.
Do not treat the coordinator as a successful automatic launcher for this final campaign.

The hardware extractor requires Nsight Systems 2025.6.3.343 and binds its executable and recipe source hashes.
Its installed dependencies include `target-linux-sbsa-armv8/python/packages/nsys_recipe/lib/data_utils.py` and `gpu_metrics.py`.
It also requires the tracked metric-schema document listed above.
Its current CPU-analysis interface requires masked CUDA and the exact `.runtime/venvs/qwen-posthoc-cpu` prefix.
These paths and execution-policy guards need deliberate adaptation for another environment.

Exact replay of the historical capture qualification also needs these archived artifacts:

```text
runs/uccl-scoped-capture-20261109T164619Z/acquired-source/tools/check_uccl_budget_runtime.py
runs/uccl-scoped-capture-20261109T164619Z/acquired-source.sha256
```

The qualifier compares the original captured helper against current source and preserves its original failed QA state.
A fresh current capture uses its own source-bound QA evidence rather than relabeling that historical failure.
Native UCCL libraries, Nsight installation, original C0/data/cache, manifests, and accepted raw records remain separate runtime artifacts.

The hardware values describe all-device samples during known UCCL payload intervals.
They do not measure communication-only activity or achieved warp occupancy.
Whole-session event completeness remains unresolved, while acceptance covers the exact required payload/call event set.

## Measurement-specific upstream gaps

Several current files assume the original site and campaign.
These assumptions need explicit configuration or supplied artifacts before fresh-clone measurement works.
The investigation did not change those sources.

1. Include the required `jobs/` and `analysis/` source, which `.gitignore` currently excludes.
2. Supply the missing source plan `dataset/manifests/qwen3-30b-a3b-base-realistic-mix-v2.json` before fresh text materialization.
3. Replace fixed source/build/input paths, account settings, historical proof roots, and job IDs with supported configuration.
4. Align launchers and Python guards with current `preemptable` policy. The historical acquisitions used an authorized `debug` exception, and overriding the batch directive alone fails their guards.
5. Include every entry in `_AMPA_RUNTIME_SOURCE_PATHS` in `profiling/ampa_profile/provenance.py`, or deliberately revise that inventory for a new campaign. Its 47 required files include older launchers and analyzers.
6. Supply or recreate the pinned C0, indexed text/cache, UCCL extension/wrapper/patch, and new input/source/runtime proofs. Preserve the original sample-cache budget.

Fresh measurements can differ because allocations use different nodes and network conditions.
The five measured updates do not establish isolated backend causality or profiler overhead.
Logical destinations do not measure physical network bytes.
Nested CPU scopes and concurrent GPU activity do not partition complete-step time.
