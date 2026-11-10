# Measurement files: purpose and reason for inclusion

This guide describes every file in [REQUIRED_FILES.txt](REQUIRED_FILES.txt).
The corrected list contains 123 files: 107 source/setup/schema files and 16 tests required by the included checks.
The user requests tests only when their absence breaks an included run or prerequisite.

The trainer does not import tests, but six are bound by the frozen sweep source manifest.
Other retained tests are invoked by the included build/preflight jobs or provide imported fixtures.
Removing those files without changing the checks causes failure.
Four optional tests from the earlier list are excluded.
Eight directly required test dependencies omitted from that list are added.
No source, launcher, test behavior, staging, or commit changed in this documentation task.

An older file marked as retained by an inventory is not an instruction to run that older experiment.
The current verification code still hashes its presence and bytes.
This is a compatibility inventory, rather than a minimal redesigned upstream package.
Optional branches of older recovery CLIs need additional historical files and are outside this inventory.

## Special source and schema files

| File | What it does and why it is included |
| --- | --- |
| [ai/qwen-gpu-communication-metrics/iteration-01/extract_launch_metrics.py](../../../ai/qwen-gpu-communication-metrics/iteration-01/extract_launch_metrics.py) | Extracts kernel geometry and resources from four accepted PyTorch traces. Reproduces the original communication launch measurements. |
| [ai/qwen-uccl-block-sweep/iteration-01/HARDWARE_METRIC_SCHEMA.md](../../../ai/qwen-uccl-block-sweep/iteration-01/HARDWARE_METRIC_SCHEMA.md) | Defines the SM metric names, units, sampling rules, and limits. The hardware extractor requires this document. |
| [ai/qwen-uccl-ep-comparison/iteration-01/02-implement/uccl-borrowed-scratch.patch](../../../ai/qwen-uccl-ep-comparison/iteration-01/02-implement/uccl-borrowed-scratch.patch) | Fixes ownership of borrowed UCCL scratch during shutdown. The pinned native build requires this exact patch. |

## Numerical analysis and evidence checks

| File | What it does and why it is included |
| --- | --- |
| [analysis/accept_uccl_pilot.py](../../../analysis/accept_uccl_pilot.py) | Provides hash-bound evidence readers and identity checks. Current validators import these helpers from the older pilot workflow. |
| [analysis/aggregate_deep_trace.py](../../../analysis/aggregate_deep_trace.py) | Aggregates correlated CPU/GPU activity by phase. The native trace validator uses it for numerical phase summaries. |
| [analysis/aggregate_run.py](../../../analysis/aggregate_run.py) | Consolidates earlier lightweight routing records into metrics. Retained because the runtime source inventory requires it. |
| [analysis/analyze_all_step_trace.py](../../../analysis/analyze_all_step_trace.py) | Drives older two-update and 25-update trace analysis. Retained by source inventories rather than used for ten-update acquisition. |
| [analysis/check_ep16_smoke.py](../../../analysis/check_ep16_smoke.py) | Checks two healthy EP16 updates and required records on sixteen ranks. Retained from the earlier pilot workflow. |
| [analysis/compare_all_step_profiles.py](../../../analysis/compare_all_step_profiles.py) | Compares earlier ordinary/grouped full captures with strict matching gates. Retained by the runtime source inventory. |
| [analysis/compare_deep_trace_overhead.py](../../../analysis/compare_deep_trace_overhead.py) | Checks consumed batches and matched trace/control evidence. Current acceptance and reporting import its fingerprint helpers. |
| [analysis/compare_ep16_recovery.py](../../../analysis/compare_ep16_recovery.py) | Builds bounded reports for the earlier EP16 recovery experiment. Retained by the runtime source inventory. |
| [analysis/compare_profile_overhead.py](../../../analysis/compare_profile_overhead.py) | Supplies source, environment, configuration, and placement comparisons. Current shared analysis imports these helpers. |
| [analysis/compare_uccl_profiles.py](../../../analysis/compare_uccl_profiles.py) | Defines backend cases, routing summaries, and normalized phase rows. Current debug reporting and validation reuse these functions. |
| [analysis/extract_step_timing.py](../../../analysis/extract_step_timing.py) | Extracts complete-update times from the authoritative Bridge log. Produces timing summaries and Parquet output. |
| [analysis/inspect_uccl_trace.py](../../../analysis/inspect_uccl_trace.py) | Checks UCCL forward/backward journals against launches and GPU events. Rejects incomplete or incorrectly attributed captures. |
| [analysis/qwen_debug_independent_audit.py](../../../analysis/qwen_debug_independent_audit.py) | Independently reconstructs timing, routing, fingerprints, and phase values. Supplies a separate check of the numerical report. |
| [analysis/qwen_debug_profile_phases.py](../../../analysis/qwen_debug_profile_phases.py) | Averages accepted CPU/GPU phase records over sixteen ranks and updates 6–10. Supplies the common phase measurements. |
| [analysis/qwen_debug_results.py](../../../analysis/qwen_debug_results.py) | Combines accepted cases into numerical JSON and CSV. Supplies timing, routing, phase summaries, and descriptive comparisons. |
| [analysis/summarize_training_evidence.py](../../../analysis/summarize_training_evidence.py) | Reads loss, LR, gradient norm, consumed samples, and failed-update signals. Establishes whether training completed healthily. |
| [analysis/uccl_budget_hardware_compare.py](../../../analysis/uccl_budget_hardware_compare.py) | Checks cross-budget workload and metric bindings, then compares sampled hardware values. Also produces tables and plots. |
| [analysis/uccl_budget_hardware_metrics.py](../../../analysis/uccl_budget_hardware_metrics.py) | Extracts device-wide SM activity, SM issue, and compute-warp samples during accepted intervals. Produces the new hardware measurements. |
| [analysis/uccl_budget_sweep_delivery.py](../../../analysis/uccl_budget_sweep_delivery.py) | Checks four accepted budgets, common identities, and sixteen native reports. Writes numerical summaries and the report inventory. |
| [analysis/validate_ep16_received_routes.py](../../../analysis/validate_ep16_received_routes.py) | Reconciles source expert assignments with actual receivers and ownership. Detects traffic or locality accounting errors. |
| [analysis/validate_torch_trace.py](../../../analysis/validate_torch_trace.py) | Checks native trace scopes, call counts, and CPU-to-GPU correlations. Establishes whether captured activity supports quantitative analysis. |

## Model configuration

| File | What it does and why it is included |
| --- | --- |
| [configs/model/qwen3_30b_a3b_base.json](../../../configs/model/qwen3_30b_a3b_base.json) | Pins model/tokenizer identity and architecture assertions. Preparation and training use it to identify the measured model. |

## Analysis launchers

| File | What it does and why it is included |
| --- | --- |
| [jobs/analysis/uccl_budget_sweep_posthoc.sbatch](../../../jobs/analysis/uccl_budget_sweep_posthoc.sbatch) | Runs budget validation and delivery in the configured compute container. Included as the allocated posthoc wrapper. |
| [jobs/analysis/uccl_debug_validate.sbatch](../../../jobs/analysis/uccl_debug_validate.sbatch) | Runs the ten-update single-case validator in a compute allocation. Also retained by the original source inventory. |
| [jobs/analysis/uccl_ep_validate_profiles.sbatch](../../../jobs/analysis/uccl_ep_validate_profiles.sbatch) | Wraps the earlier full-profile validation coordinator on compute. The runtime source inventory requires this launcher. |

## Checkpoint launchers

| File | What it does and why it is included |
| --- | --- |
| [jobs/checkpoint/import_qwen3_30b_a3b_base_c0.sbatch](../../../jobs/checkpoint/import_qwen3_30b_a3b_base_c0.sbatch) | Imports the pinned local HF snapshot into immutable Megatron C0. Creates the common starting checkpoint. |
| [jobs/checkpoint/materialize_qwen3_30b_a3b_base_hf.sbatch](../../../jobs/checkpoint/materialize_qwen3_30b_a3b_base_hf.sbatch) | Downloads and inventories the pinned HF snapshot once on compute. Supplies the source for C0 import. |

## Dataset and cache launchers

| File | What it does and why it is included |
| --- | --- |
| [jobs/dataset/materialize_qwen3_30b_a3b_base_mix.sbatch](../../../jobs/dataset/materialize_qwen3_30b_a3b_base_mix.sbatch) | Runs deterministic text selection and indexed tokenization. Creates the four reusable dataset components. |
| [jobs/dataset/prepare_qwen3_30b_a3b_base_mix_cache.sbatch](../../../jobs/dataset/prepare_qwen3_30b_a3b_base_mix_cache.sbatch) | Builds the native Megatron sample cache before training. Keeps cache construction outside measurement. |
| [jobs/dataset/probe_qwen3_30b_a3b_base_mix_cache_hit.sbatch](../../../jobs/dataset/probe_qwen3_30b_a3b_base_mix_cache_hit.sbatch) | Runs a real Bridge cache-hit probe. Confirms that the training consumer reuses the prepared arrays. |
| [jobs/dataset/promote_qwen3_30b_a3b_base_mix_manifest.sbatch](../../../jobs/dataset/promote_qwen3_30b_a3b_base_mix_manifest.sbatch) | Rechecks reviewed data/cache evidence and writes a final manifest. Supports the original and CP2 parent variants. |
| [jobs/dataset/validate_qwen3_30b_a3b_base_mix.sbatch](../../../jobs/dataset/validate_qwen3_30b_a3b_base_mix.sbatch) | Checks indexed token files and selection ledgers on compute. Establishes data integrity before cache preparation. |
| [jobs/dataset/validate_qwen3_30b_a3b_base_mix_cache.sbatch](../../../jobs/dataset/validate_qwen3_30b_a3b_base_mix_cache.sbatch) | Checks the prepared cache arrays and their input bindings. Establishes cache integrity before measurement. |

## Build and preflight launchers

| File | What it does and why it is included |
| --- | --- |
| [jobs/smoke/ep16_profile_preflight.sbatch](../../../jobs/smoke/ep16_profile_preflight.sbatch) | Runs EP16 preflight checks and creates runtime dataset derivatives. Required when preparing the EP16 input path. |
| [jobs/smoke/uccl_budget_runtime_checks.sbatch](../../../jobs/smoke/uccl_budget_runtime_checks.sbatch) | Runs native budget and multi-process capture checks before the sweep. Invokes the budget test module directly. |
| [jobs/smoke/uccl_debug_checks.sbatch](../../../jobs/smoke/uccl_debug_checks.sbatch) | Checks the ten-update adapter and single-case validator. Invokes their test files before the original short campaign. |
| [jobs/smoke/uccl_debug_reuse_checks.sbatch](../../../jobs/smoke/uccl_debug_reuse_checks.sbatch) | Checks the reused-input adapter and provenance helper. Requires its explicit reuse test file. |
| [jobs/smoke/uccl_ep_build.sbatch](../../../jobs/smoke/uccl_ep_build.sbatch) | Builds the pinned patched UCCL extension and wrapper. Requires source-identity and loaded-identity tests during the build. |
| [jobs/smoke/uccl_ep_dispatch_smoke.sbatch](../../../jobs/smoke/uccl_ep_dispatch_smoke.sbatch) | Runs distributed forward/backward correctness and cleanup checks. Requires identity and all-step profiling tests. |
| [jobs/smoke/uccl_ep_verify_build.sbatch](../../../jobs/smoke/uccl_ep_verify_build.sbatch) | Checks the built extension, imported wrapper, loaded fabric, and source proof. Establishes the native build used by training. |
| [jobs/smoke/uccl_nsight_capabilities.sbatch](../../../jobs/smoke/uccl_nsight_capabilities.sbatch) | Probes Nsight tracing, report export, and GPU metric availability. Supplies the capture command and capability proof. |
| [jobs/smoke/uccl_nsight_cuda132_setup.sbatch](../../../jobs/smoke/uccl_nsight_cuda132_setup.sbatch) | Installs the reviewed compatible Nsight package in an isolated location. Supplies the collector used by the SM sweep. |

## Training and scheduling launchers

| File | What it does and why it is included |
| --- | --- |
| [jobs/training/cp2_all_steps_group_routing.sbatch](../../../jobs/training/cp2_all_steps_group_routing.sbatch) | Earlier CP2/EP8 ordinary/grouped capture-and-control suite. Retained because the runtime source inventory requires it. |
| [jobs/training/cp2_torchprof_pair.sbatch](../../../jobs/training/cp2_torchprof_pair.sbatch) | Earlier matched CP2/EP8 control and short PyTorch trace pair. Retained by the runtime source inventory. |
| [jobs/training/ep16_all_steps_group_routing.sbatch](../../../jobs/training/ep16_all_steps_group_routing.sbatch) | Earlier EP16 pilot and four-leg full-profile suite. Retained by the runtime source inventory. |
| [jobs/training/ep16_all_steps_reviewed.sbatch](../../../jobs/training/ep16_all_steps_reviewed.sbatch) | Earlier reviewed EP16 acquisition and bounded recovery wrapper. Retained by the runtime source inventory. |
| [jobs/training/qwen3_30b_a3b_base_cp2_paired_overhead.sbatch](../../../jobs/training/qwen3_30b_a3b_base_cp2_paired_overhead.sbatch) | Earlier CP2/EP8 lightweight-profiler OFF/ON launcher. Retained by the runtime source inventory. |
| [jobs/training/qwen3_30b_a3b_base_ep8_profile.sbatch](../../../jobs/training/qwen3_30b_a3b_base_ep8_profile.sbatch) | Earlier general EP8 training/profiling launcher. Retained by the runtime source inventory. |
| [jobs/training/qwen3_all_steps_profile.sbatch](../../../jobs/training/qwen3_all_steps_profile.sbatch) | Earlier CP2/EP8 all-update capture or control launcher. Retained by the runtime source inventory. |
| [jobs/training/qwen3_ep16_all_steps.sbatch](../../../jobs/training/qwen3_ep16_all_steps.sbatch) | Earlier EP16 two-update or 25-update launcher. Retained by the runtime source inventory. |
| [jobs/training/qwen3_uccl_debug_reuse_worker.sh](../../../jobs/training/qwen3_uccl_debug_reuse_worker.sh) | Starts four workers per node for later short cases. Selects backend, routing, profiling, and unchanged measurement flags. |
| [jobs/training/qwen3_uccl_debug_worker.sh](../../../jobs/training/qwen3_uccl_debug_worker.sh) | Starts the first short case after fresh input verification. Runs training and records worker logs and timing. |
| [jobs/training/qwen3_uccl_ep_worker.sh](../../../jobs/training/qwen3_uccl_ep_worker.sh) | Earlier worker for pilots and the 25-update campaign. Retained by the runtime source inventory. |
| [jobs/training/uccl_budget_sweep.sbatch](../../../jobs/training/uccl_budget_sweep.sbatch) | Allocates four nodes for one communication budget. Creates source/input records and starts the Nsight worker. |
| [jobs/training/uccl_budget_sweep_coordinator.sh](../../../jobs/training/uccl_budget_sweep_coordinator.sh) | Schedules the original automatic sweep and requires strict QA success. Inventory-bound, but superseded for the final accepted sequence. |
| [jobs/training/uccl_budget_sweep_worker.sh](../../../jobs/training/uccl_budget_sweep_worker.sh) | Runs one Nsight collector around four training processes per node. Produces native reports, journals, and training summaries. |
| [jobs/training/uccl_debug_coordinator.sh](../../../jobs/training/uccl_debug_coordinator.sh) | Original eight-case scheduler with validation after each acquisition. Retained by the original frozen source inventory. |
| [jobs/training/uccl_debug_reordered_coordinator.sh](../../../jobs/training/uccl_debug_reordered_coordinator.sh) | Preserves three successful cases, schedules four UCCL cases, and retries native profiling last. Records the accepted acquisition map. |
| [jobs/training/uccl_debug_reuse_partition_coordinator.sh](../../../jobs/training/uccl_debug_reuse_partition_coordinator.sh) | Earlier continuation that reused input verification and deferred analysis. Part of the historical acquisition sequence. |
| [jobs/training/uccl_ep_comparison.sbatch](../../../jobs/training/uccl_ep_comparison.sbatch) | Earlier pilot/full comparison orchestrator using four nodes. Retained by the runtime source inventory. |
| [jobs/training/uccl_ep_debug_case.sbatch](../../../jobs/training/uccl_ep_debug_case.sbatch) | Allocates the first short case and freshly verifies inputs/build/source. Establishes the proof reused by later acquisitions. |
| [jobs/training/uccl_ep_debug_reuse.sbatch](../../../jobs/training/uccl_ep_debug_reuse.sbatch) | Allocates later short cases using the first verified inputs. Retains all collectors and writes health/timing records. |

## Collectors and runtime records

| File | What it does and why it is included |
| --- | --- |
| [profiling/ampa_profile/__init__.py](../../../profiling/ampa_profile/__init__.py) | Defines the observer package and exports RoutingObserver. Required when Python imports this package. |
| [profiling/ampa_profile/callback.py](../../../profiling/ampa_profile/callback.py) | Connects the earlier lightweight observer to Bridge callbacks. Retained by package dependencies and the runtime source inventory. |
| [profiling/ampa_profile/deep_trace.py](../../../profiling/ampa_profile/deep_trace.py) | Labels native training phases and controls PyTorch capture/export. Also supplies method-label infrastructure reused by UCCL/Nsight. |
| [profiling/ampa_profile/observer.py](../../../profiling/ampa_profile/observer.py) | Implements the earlier lightweight dispatcher routing observer. Exported by the package and retained by source checks. |
| [profiling/ampa_profile/provenance.py](../../../profiling/ampa_profile/provenance.py) | Records actual topology, physical GPUs, source hashes, and input identity. Enforces the mandatory runtime source-file inventory. |
| [profiling/ampa_profile/routing_audit.py](../../../profiling/ampa_profile/routing_audit.py) | Records selected experts, destination counts, locality, and receiver ownership. Supplies logical communication accounting. |
| [profiling/ampa_profile/sample_fingerprint.py](../../../profiling/ampa_profile/sample_fingerprint.py) | Hashes actual consumed CPU batches with schema and order. Establishes that comparable runs use identical inputs. |
| [profiling/ampa_profile/uccl_identity.py](../../../profiling/ampa_profile/uccl_identity.py) | Checks loaded native binary, wrapper, fabric, and runtime state. Records backend identity and bounded cleanup. |
| [profiling/ampa_profile/uccl_trace.py](../../../profiling/ampa_profile/uccl_trace.py) | Labels UCCL layout, dispatch/combine, and backward adjoints. Supplies the call identities needed to correlate communication. |

## Training entrypoints

| File | What it does and why it is included |
| --- | --- |
| [profiling/run_pretrain.py](../../../profiling/run_pretrain.py) | Builds the common configuration, attaches collectors, and invokes native Bridge training. It is the shared training entrypoint. |
| [profiling/run_pretrain_debug.py](../../../profiling/run_pretrain_debug.py) | Executes five warmup plus five measured updates. Preserves the original 25-update cache and native ten-update exit. |
| [profiling/run_pretrain_debug_reuse.py](../../../profiling/run_pretrain_debug_reuse.py) | Reuses bound input-verification metadata during argument parsing. Executes the same short training and measurement path. |
| [profiling/run_pretrain_uccl_budget.py](../../../profiling/run_pretrain_uccl_budget.py) | Sets the selected UCCL budget before Buffer creation and adds NVTX/CUDA capture boundaries. Drives the new SM experiment. |

## Container setup

| File | What it does and why it is included |
| --- | --- |
| [scripts/build_gh200_image.sbatch](../../../scripts/build_gh200_image.sbatch) | Builds the pinned dependency image on a GH200 compute node. Produces the reusable container image. |
| [setup/Dockerfile.gh200-cuda130](../../../setup/Dockerfile.gh200-cuda130) | Defines CUDA/PyTorch and Bridge dependency installation. Recreates the measurement software environment. |
| [setup/gh200-cuda130-runtime.toml](../../../setup/gh200-cuda130-runtime.toml) | Defines the container image, mounts, paths, and environment for Slurm. Used by compute launchers. |

## Tests required by included checks

| File | What it does and why it is included |
| --- | --- |
| [tests/test_accept_uccl_pilot.py](../../../tests/test_accept_uccl_pilot.py) | Tests evidence acceptance and supplies fixtures imported by the debug validator tests. Also required by the frozen sweep source manifest. |
| [tests/test_all_step_profiling.py](../../../tests/test_all_step_profiling.py) | Tests per-update export and grouped routing. Required by EP16 preparation and native dispatch preflights. |
| [tests/test_deep_trace.py](../../../tests/test_deep_trace.py) | Tests transparent method wrapping and restoration. Directly invoked by the included EP16 preparation preflight. |
| [tests/test_ep16_analysis.py](../../../tests/test_ep16_analysis.py) | Tests EP16 analysis, batch identity, and placement gates. Directly invoked by the included EP16 preparation preflight. |
| [tests/test_ep16_observers.py](../../../tests/test_ep16_observers.py) | Tests EP16 topology and routing-observer contracts. Directly invoked by the included EP16 preparation preflight. |
| [tests/test_ep16_received_routes.py](../../../tests/test_ep16_received_routes.py) | Tests integer send/receive reconciliation and supplies acceptance-test fixtures. Required by the frozen sweep source manifest. |
| [tests/test_ep16_runtime.py](../../../tests/test_ep16_runtime.py) | Tests EP16 runtime and immutable-cache guards. Directly invoked by the included EP16 preparation preflight. |
| [tests/test_indexed_run_provenance.py](../../../tests/test_indexed_run_provenance.py) | Tests source, topology, placement, and observation records. Directly invoked by the included EP16 preparation preflight. |
| [tests/test_uccl_budget_entrypoint.py](../../../tests/test_uccl_budget_entrypoint.py) | Tests budget setup and capture boundaries. Directly invoked by runtime checks and required by the frozen sweep manifest. |
| [tests/test_uccl_debug_entrypoint.py](../../../tests/test_uccl_debug_entrypoint.py) | Tests the ten-update adapter and unchanged cache budget. Required by the debug preflight and frozen sweep manifest. |
| [tests/test_uccl_debug_reuse.py](../../../tests/test_uccl_debug_reuse.py) | Tests truthful verification reuse and parser restoration. Directly invoked by the included reuse preflight. |
| [tests/test_uccl_ep_debug_validate.py](../../../tests/test_uccl_ep_debug_validate.py) | Tests ten-update evidence and capture coverage. Required by the debug preflight and frozen sweep manifest. |
| [tests/test_uccl_ep_source_identity.py](../../../tests/test_uccl_ep_source_identity.py) | Tests rejection of unapproved native source changes. The included UCCL build launcher requires and executes it. |
| [tests/test_uccl_identity.py](../../../tests/test_uccl_identity.py) | Tests loaded binary, fabric, wrapper, and cleanup evidence. Required by the native build and dispatch preflight. |
| [tests/test_uccl_pretrain_config.py](../../../tests/test_uccl_pretrain_config.py) | Tests backend/configuration wiring and supplies fixtures to debug tests. Required by the frozen sweep source manifest. |
| [tests/test_validate_torch_trace.py](../../../tests/test_validate_torch_trace.py) | Tests rejection of inconsistent scopes and CUDA correlations. Directly invoked by the included EP16 preparation preflight. |

## Shared preparation and validation tools

| File | What it does and why it is included |
| --- | --- |
| [tools/__init__.py](../../../tools/__init__.py) | Defines the local tools package. Ensures imports resolve to project helpers. |
| [tools/bridge_indexed_consumer.py](../../../tools/bridge_indexed_consumer.py) | Applies the reviewed tokenizer and indexed-dataset configuration. Shared by training and cache-hit probes. |
| [tools/c0_validation.py](../../../tools/c0_validation.py) | Checks C0 identity, source snapshot bindings, and payload inventories. Shared by preparation and training acceptance. |
| [tools/check_uccl_budget_runtime.py](../../../tools/check_uccl_budget_runtime.py) | Checks effective native budgets and one-collector/multiple-child tracing. Required by the sweep's runtime preflight and posthoc checks. |
| [tools/ep16_indexed_manifest.py](../../../tools/ep16_indexed_manifest.py) | Creates a checked EP16 derivative of the existing cache manifest. Preserves indexed tokens and sample-cache identity. |
| [tools/ep16_recovery_placement.py](../../../tools/ep16_recovery_placement.py) | Provides reference-run, hashing, and write-once helpers used by reviewed-input preparation. Also contains historical placement-recovery logic. |
| [tools/ep16_reviewed_inputs.py](../../../tools/ep16_reviewed_inputs.py) | Checks checkpoint/data/source against the reviewed reference and archives source identity. Used by first-case verification. |
| [tools/inspect_gpu_identity.py](../../../tools/inspect_gpu_identity.py) | Inspects the physical GPU selected for a Slurm task. Retained by the runtime source inventory. |
| [tools/materialize_hf_snapshot.py](../../../tools/materialize_hf_snapshot.py) | Fetches the exact HF revision and writes its inventory. Supplies reproducible local model/tokenizer inputs. |
| [tools/materialize_indexed_mixture.py](../../../tools/materialize_indexed_mixture.py) | Selects deterministic source documents and writes indexed token components. Implements one-time dataset preparation. |
| [tools/megatron_cache_contract.py](../../../tools/megatron_cache_contract.py) | Defines and checks cache/component/sample-budget bindings. Prevents silent use or creation of a different training cache. |
| [tools/megatron_cache_stage.py](../../../tools/megatron_cache_stage.py) | Wraps native Megatron cache preparation, validation, and cache-hit probing. Keeps indexing work outside timed training. |
| [tools/promote_cp2_feasibility_manifest.py](../../../tools/promote_cp2_feasibility_manifest.py) | Promotes independently reviewed CP2/EP8 cache evidence. Supplies the historical parent used by the EP16 derivative. |
| [tools/promote_indexed_manifest.py](../../../tools/promote_indexed_manifest.py) | Writes a final data manifest after reviewing and rechecking data/cache evidence. Required for the prepared-input lifecycle. |
| [tools/qualify_uccl_budget_capture.py](../../../tools/qualify_uccl_budget_capture.py) | Checks the original saved capture against the scoped event requirements. Reproduces the accepted historical qualification without relabeling failed QA. |
| [tools/qwen_workload.py](../../../tools/qwen_workload.py) | Centralizes topology, accumulation, and token-count expectations. Training, collectors, and validators share these values. |
| [tools/uccl_budget_sweep_posthoc.py](../../../tools/uccl_budget_sweep_posthoc.py) | Validates SM-sweep training, inputs, budgets, journals, and native report contents. Exports SQLite and writes per-case acceptance. |
| [tools/uccl_debug_campaign_validate.py](../../../tools/uccl_debug_campaign_validate.py) | Validates the complete eight-case short campaign and input-prefix equality. Writes campaign acceptance used by numerical reporting. |
| [tools/uccl_ep_debug_validate.py](../../../tools/uccl_ep_debug_validate.py) | Adapts strict single-case checks to ten updates. Reconciles health, routes, identity, and native/UCCL trace coverage. |
| [tools/uccl_ep_dispatch_smoke.py](../../../tools/uccl_ep_dispatch_smoke.py) | Exercises real native UCCL forward/backward and cleanup. Supplies distributed correctness evidence before full-model measurements. |
| [tools/uccl_ep_log_proof.py](../../../tools/uccl_ep_log_proof.py) | Binds native CXI endpoint messages to worker logs. Establishes the actual transport rather than only requested environment variables. |
| [tools/uccl_ep_source_identity.py](../../../tools/uccl_ep_source_identity.py) | Checks the pinned UCCL source and exact approved patch. Prevents an unrelated native change from entering the measured build. |
| [tools/uccl_ep_validate_profiles.py](../../../tools/uccl_ep_validate_profiles.py) | Coordinates strict validators for the earlier full campaign. Retained by the runtime source inventory. |
| [tools/uccl_ep_verified_inputs_reuse.py](../../../tools/uccl_ep_verified_inputs_reuse.py) | Binds first-case input verification to each later job. Records that bulk verification was reused rather than repeated. |
| [tools/validate_indexed_mixture.py](../../../tools/validate_indexed_mixture.py) | Checks indexed files, document/token accounting, and selection-ledger bindings. Supplies independent prepared-data evidence. |
| [tools/write_c0_manifest.py](../../../tools/write_c0_manifest.py) | Writes identity and payload hashes after successful C0 import. Supplies the immutable checkpoint provenance used by later runs. |

## Results documents and reproduction notes

These documents are additional candidates for the proposed commit.
They are outside the 123 source-file count.
The results documents retain their original evidence links, some of which point to local raw artifacts.

| File | What it records |
| --- | --- |
| [docs/QWEN_UCCL_EXPERIMENT_AUDIT.md](../../../docs/QWEN_UCCL_EXPERIMENT_AUDIT.md) | Original eight-case timing, throughput, routing, and phase tables. Includes full configuration, input provenance, health records, and limitations. |
| [runs/uccl-budget4-hardware-comparison-20261109T233200Z/hardware_budget_comparison.md](../../../runs/uccl-budget4-hardware-comparison-20261109T233200Z/hardware_budget_comparison.md) | Final 12/20/32/40 sampled SM, issue, and compute-warp values, sample counts, and interpretation limits. This is the concise SM results table. |
| [ai/qwen-gpu-communication-metrics/iteration-01/FINDINGS.md](../../qwen-gpu-communication-metrics/iteration-01/FINDINGS.md) | Earlier kernel launch geometry and resource results. Distinguishes grid-based bounds from measured hardware utilization. |
| [ai/qwen-uccl-block-sweep/iteration-01/HARDWARE_COMPARE_REVIEW.md](../../qwen-uccl-block-sweep/iteration-01/HARDWARE_COMPARE_REVIEW.md) | Independent verification of hardware results, exact means, native report bindings, timing limits, and source hashes. |
| [MEASUREMENTS.md](MEASUREMENTS.md) | Preparation, launch flow, metric outputs, and known packaging gaps for both measurement campaigns. |
| [REQUIRED_FILES.txt](REQUIRED_FILES.txt) | One repository-relative source path per line for the proposed file scope. |
| [FILE_GUIDE.md](FILE_GUIDE.md) | This per-file explanation for reviewing the proposed scope before any commit. |

## Existing staged setup and dependency references

Eight entries were already staged before this request.
Three of them, the image builder, Dockerfile, and EDF, already appear in the 123-file list.
The following five additional entries still need the user's scope confirmation.
They remain staged exactly as before.

| Entry | What it does and why it can be included |
| --- | --- |
| [.gitmodules](../../../.gitmodules) | Records the Bridge and Megatron-LM submodule locations. Required when cloning the project with those dependency references. |
| Megatron-Bridge | A Git submodule reference to the training framework. Includes a pinned source revision, rather than a copy of every framework file. |
| Megatron-LM | A Git submodule reference to model/distributed-training implementations. Supplies the pinned implementation observed by AMPA. |
| [scripts/refresh_bridge_lock.sbatch](../../../scripts/refresh_bridge_lock.sbatch) | Refreshes dependencies after deliberately changing Bridge's Megatron revision. Maintenance helper, not needed to repeat the pinned experiment. |
| [scripts/run_qwen3_30b_a3b_smoke.sbatch](../../../scripts/run_qwen3_30b_a3b_smoke.sbatch) | Earlier two-node mock-data training smoke. Useful setup diagnostic, not the real-data measurement launcher. |

## Required runtime artifacts outside the source list

The source list does not include the large image, C0, indexed data/cache, UCCL binary, Nsight installation, or raw captures.
Their pinned identities and paths are documented in MEASUREMENTS.md.
The original v2 dataset source plan is missing and must be restored for fresh materialization.
Exact historical capture qualification also needs the archived original helper and inventory recorded in that note.
Some current launchers enforce historical partitions, paths, source manifests, and CPU environments.
These are concrete portability limits rather than reasons to add optional tests.

Assumptions: Preserve the existing launch/check behavior while choosing files for a source-and-results commit.

Checked: Every selected path, direct test invocations, imported fixtures, frozen sweep test entries, and the described source roles.

Changed: Inventory and documentation only. No Git staging or commit.

Verification: Source inspection and inventory coverage checks. No test suite, workload, or training execution.

Risk: The selected source set preserves the current supported paths. It does not make all historical CLI branches or machine-specific paths portable.

