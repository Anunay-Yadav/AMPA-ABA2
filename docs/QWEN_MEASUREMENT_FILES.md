# Qwen measurement sources and results

This repository contains the measurement code and compact results for the native Megatron and UCCL comparisons. It also contains the adaptive unprofiled UCCL block-budget sweep. PDF and presentation generation is outside this package.

## Results

The [original configuration and comparison audit](QWEN_UCCL_EXPERIMENT_AUDIT.md) describes the first native/UCCL experiments. The [overlap-enabled comparison](QWEN_EP_OVERLAP_AUDIT.md) records native and UCCL results with ordinary and grouped routing, profiler OFF/ON and four Nsight budgets. The [unprofiled block-budget audit](QWEN_UCCL_UNPROFILED_BUDGET_AUDIT.md) records the later 4–56 block sweep with repeated same-allocation comparisons.

The reports explain complete optimizer-step timings, separate allocation costs, input identity, training health and measurement limits. Their links lead to compact JSON, CSV and PNG/SVG outputs. The original raw trace files are external.

## Container and immutable inputs

The tracked container sources are [Dockerfile.gh200-cuda130](../setup/Dockerfile.gh200-cuda130), [the runtime EDF](../setup/gh200-cuda130-runtime.toml) and [the image-build job](../scripts/build_gh200_image.sbatch). The build job pins Bridge and builds the dependency image. Training overlays the mounted framework checkouts. The small [SQSH checksum](../setup/gh200-cuda130-megatron.sqsh.sha256) records the image used for these results. The built image is not committed.

The framework revisions are the Git submodule pins in this repository. The UCCL source revision, native build and ownership patch are recorded in the [measurement guide](../ai/qwen-debug-reproducibility/iteration-01/MEASUREMENTS.md). That guide also lists the one-time checkpoint import, deterministic indexed data and sample-cache preparation.

Before a fresh run, adapt the workspace paths, Slurm account, partition and container mounts for the target cluster. Prepare the pinned model checkpoint, tokenizer, indexed mixture, cache and native UCCL build. Create fresh qualifications and run IDs. Historical job IDs, proof files and machine-specific paths do not qualify another machine automatically.

## Measurement entrypoints

The overlap comparison uses `profiling/run_pretrain_ep_overlap.py`. The `ep_overlap_preflight`, `ep_overlap_checked`, `ep_overlap_worker` and `ep_overlap_posthoc` launchers under `jobs/` perform qualification, training and validation. `tools/ep_overlap_campaign_summary.py` generates its numerical results. The [summary commands](../ai/ep-overlap-campaign-summary/iteration-01/COMMANDS.md) describe its allocated QA and replay.

The unprofiled sweep uses `profiling/run_pretrain_ep_overlap_budget.py` with `jobs/training/ep_overlap_budget_pair.sbatch` and `ep_overlap_budget_worker.sh`. Each allocation runs an independent candidate and 20-block reference on the same GPUs. `tools/ep_overlap_budget_validate.py` checks both cases. `analysis/ep_overlap_budget_compare.py` computes the paired ratio and the next budget. `tools/ep_overlap_budget_summary.py` reconciles the complete campaign. Its [allocated commands](../ai/ep-overlap-budget-summary/iteration-01/COMMANDS.md) describe final QA and replay.

The two scheduler continuations are `jobs/coordination/ep_overlap_retry_campaign.sh` and `ep_overlap_budget_campaign.sh`. They contain the recorded campaign's identity and qualification checks. A fresh campaign needs fresh inputs and proofs. Both preserve failed attempts and keep analysis separate from training.

The required source lists are preserved in the [overlap acquisition manifest](../runs/ep-overlap-debug-ddp-source-20261110T111031Z/source.sha256), [overlap summary manifest](../runs/ep-overlap-summary-source-20261110T2046Z/source.sha256), [unprofiled acquisition manifest](../runs/ep-overlap-budget-off-source-20261110T221142Z/source.sha256) and [unprofiled summary manifest](../runs/ep-overlap-budget-off-final-source-20261110T224818Z/source.sha256). Only tests required by the included build, preflight and summary commands belong to this source package. Those commands fail if their required test modules are missing.

## Included evidence and limits

The compact release includes the accepted summaries, all 120 overlap and 400 unprofiled optimizer-step timers, charts, four overlap-enabled Nsight `metrics.json` records, source manifests, small QA records and accepted job ledgers. It also includes independent numerical and interpretation reviews.

Checkpoints, indexed datasets, caches, SQSH images, native binaries, raw Torch/Nsight traces and bulk per-rank records are external. The input-reference JSON files preserve their original paths and hashes. Their presence does not mean the referenced payloads are included in Git. Full raw-evidence replay needs those original artifacts or a newly qualified measurement campaign.

The unprofiled sweep keeps common routing, receive, input-fingerprint and health observers enabled. It does not measure free SMs. Its two or three allocation replicates per candidate do not establish statistical significance or a global optimum. Use the reports' stated limits when comparing results.
