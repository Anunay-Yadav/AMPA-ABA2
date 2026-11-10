# Qwen/UCCL experiment: configuration, code and evidence audit

Prepared on 2026-11-05 for the completed Clariden GH200 experiment. Paths are relative to this repository. This document describes recorded execution, including its limits. It does not change training code or accept failed runs.

## 1. What the data can support

The eight accepted runs support a short descriptive comparison of complete optimizer-update time and forward routing behavior. Every run completes ten real updates. The comparison uses updates 6–10. All 640 recorded rank-update profiler captures pass the completeness and correlation gates.

The evidence also supports finite logged training health, identical consumed input batches, assignment conservation, receiver ownership and the recorded UCCL runtime identity. It does not establish whole-model numerical equivalence, training quality, a tuned native baseline, physical network traffic or an isolated cause for the speed differences.

The learning rate is `1e-6` for every recorded update in all eight runs. Five warmup updates means timing warmup. It does not mean learning-rate warmup. These five updates execute forward, backward and the optimizer update, and change the training state.

The new audit finds a configuration discrepancy: `optimizer.weight_decay=0.1`, but the constant scheduler sets a base weight decay of `0.033`. The source applies the scheduler value to parameter groups. Treat `0.033 × wd_mult` as the effective value, where `wd_mult` is the parameter-group multiplier. This is a source-derived conclusion. Per-update weight decay was not logged. All eight cases share this configuration, so it does not create a configuration difference between them.

There are active runtime guards for nonfinite gradient buckets and language-model losses. Router audits also test routing probabilities. There is no saved, complete scan of hidden activations or per-parameter gradients. Earlier synthetic tests compare exact outputs and gradients for a bounded communication fixture. They do not establish end-to-end Qwen equivalence.

Start with sections 2–7 to review the experiment. Sections 8–13 explain the checks, history, results and remaining uncertainty. The appendices contain every logged update and the complete representative resolved configuration.

## 2. Authoritative evidence

Use the final acceptance graph and its bound raw files. An old status paragraph, a file name or a successful process exit is insufficient by itself. The report was written before its independent audit. Its immutable `independent_numerical_audit: pending` field is superseded by the separate audit proof, not edited afterward.

| Evidence | Purpose |
| --- | --- |
| [Acquisition map][acquisitions] | Accepted case, profiler setting, Slurm job and orchestration directory. |
| [Campaign acceptance][campaign] | Eight accepted cases, ten updates each, five measured updates and 640 accepted captures. |
| [Per-case acceptance directory][case-proofs] | Configuration, health, consumed batches, routing, receives and native/trace evidence for each case. |
| [Final report][report] | Accepted timing samples, routing counts, phase activity, contrasts and claim limits. |
| [Independent numerical audit][numeric-audit] | Independent reconstruction of raw timing, health, routing, receives, input fingerprints and phase values. |
| [First immutable-input proof][first-proof] | Fresh C0/data/source checks in job 3851782. Seven later cases explicitly reuse this proof. |
| [New documentary snapshot][audit-snapshot] | All 80 LR/loss/gradient records cross-checked against raw logs, and eight timing means/SDs reproduced. |
| [Configuration/data investigation][config-evidence] | Exact configuration values, effective scheduler behavior, dataset provenance and case differences. |
| [Check inventory][checks-evidence] | Source anchors, actual proof paths, test coverage and missing signals. |
| [History/reliability investigation][history-evidence] | Accepted history, rejected attempts, assumptions and boundaries on interpretation. |
| [Decisions][decisions] and [session timeline][timeline] | User choices, technical progression and job outcomes. Read the newest state first. |
| [Final deck verification][deck-verification] and [independent presentation review][deck-review] | Numerical and presentation coverage for the delivered nine-slide deck. |

Selected SHA-256 identities allow direct comparisons with the saved evidence:

| Object | SHA-256 |
| --- | --- |
| Final report | `78d562bc123aac967af63e2c122d425afc669c14be71ab00d8815aefe4e2917b` |
| Independent numerical audit | `498ed6f347482faa7681a36a628e7ad065ea5e410a6aa1fd735bb8a41486ea4f` |
| C0 identity manifest | `27edfbc3ff917890e34fbad5f91fab9e79312c7adbb91829e0fcd18eb2244eba` |
| Selected runtime dataset manifest | `03a0edd8e04dbc4083cc408b8268b5d6cada7fb3cd1f21ba872be7ea17bdbed7` |
| Common AMPA runtime-source identity | `409c26f60171efc0ea54aba24d9a544fc1a3a0672b4556b3ab69edee76def400` |
| UCCL native binary | `110d27746db1c326c0e6216f413c5c45cf7d26eca83037b3874241c73abb8d36` |
| Loaded libfabric | `998956619429c918c9dba07f8a68faf5d9e8c630b799496e261e54622797db91` |
| Approved native ownership patch | `749a28bdb82d25ba92bace55ad6d99bafc1075280bb5466e12f378f5a53a21b2` |

A hash identifies bytes. It does not prove that the code, experimental design or interpretation is correct. The gates and independent reviews supply separate evidence.

## 3. Exact case matrix and completed jobs

`control` means PyTorch profiler OFF. `profile` means the AMPA PyTorch callback is ON. Both retain routing, receive and input-fingerprint records. `uccl` is the historical identifier for UCCL grouped routing.

| Case | Communication path | Router groups / selected groups | OFF job | ON job |
| --- | --- | --- | ---: | ---: |
| `ordinary` | Native Megatron AllToAll | `None / None` | 3854430 | 3861954 |
| `grouped` | Native Megatron AllToAll | `4 / 2` | 3854037 | 3851782 |
| `uccl_ordinary` | Existing Flex dispatcher, DeepEP manager, UCCL normal Buffer | `None / None` | 3861721 | 3861781 |
| `uccl` | Existing Flex dispatcher, DeepEP manager, UCCL normal Buffer | `4 / 2` | 3858901 | 3861400 |

All eight jobs complete successfully and pass later acceptance. Each starts independently from C0. They do not continue training from the preceding case. Grouped routing changes which experts a token can select. It is a deliberate model-workload variation, so identical losses or gradients across routing modes are not expected.

| Acquisition order | Case / profiler | Job | Allocation elapsed | Recorded nodes |
| ---: | --- | ---: | --- | --- |
| 1 | Grouped / ON | 3851782 | 17m47s | `nid[007646,007652,007657,007660]` |
| 2 | Grouped / OFF | 3854037 | 15m38s | `nid[007646,007648,007653,007660]` |
| 3 | Ordinary / OFF | 3854430 | 14m27s | `nid[007655,007658-007660]` |
| 4 | UCCL grouped / OFF | 3858901 | 6m54s | `nid[007649,007658,007660,007666]` |
| 5 | UCCL grouped / ON | 3861400 | 8m48s | `nid[006058,006077,006843,006888]` |
| 6 | UCCL ordinary / OFF | 3861721 | 6m37s | `nid[005397,007483,007496,007552]` |
| 7 | UCCL ordinary / ON | 3861781 | 10m25s | `nid[007647,007649,007660,007666]` |
| 8 | Ordinary / ON, retry | 3861954 | 15m41s | `nid[007653,007655,007658-007659]` |

Allocation elapsed time includes startup, checkpoint loading, profiler export and shutdown. Use complete-update timers for throughput comparisons. Different node lists prevent a claim of identical physical placement.

The excluded ordinary/ON attempt, job 3856098, times out after 22m29s. Its first update takes 468.9539s. It logs ten finite updates, but lacks clean completion and final acceptance. The accepted ordinary/ON result comes from job 3861954. The original timeout evidence remains intact. Its initiating cause is unproved.

Evidence: [full acquisition record][acquisition-notes], [accepted campaign][campaign], and [warmup/window analysis][warmup-analysis].

## 4. Workload and hyperparameters

### Model, topology and batch

EP means expert parallelism. CP means context parallelism. TP and PP mean tensor and pipeline parallelism. Dense data parallelism replicates dense model work across ranks. Gradient accumulation combines several microbatches before one optimizer update.

| Setting | Executed value / interpretation |
| --- | --- |
| Model | `Qwen/Qwen3-30B-A3B-Base` |
| HF model/tokenizer revision | `1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9` |
| Initial weights | Local immutable Megatron C0, not a fresh HF load per run. |
| C0 location | `checkpoint/megatron/qwen3-30b-a3b-base-c0` |
| Model architecture | 48 layers, hidden size 2048, 128 experts per MoE layer, top-8 routing, MoE intermediate size 768. |
| Precision | BF16 model parameters, FP32 main parameters/gradients and Adam moments. No FP8/FP4 workload. |
| Hardware | Four GH200 nodes, four GPUs per node, sixteen GPUs/ranks. |
| CPU allocation / affinity | Exclusive nodes, 288 CPUs available per node, no launcher CPU-count cap and `--cpu-bind=none`. |
| OpenMP threads | Actual torchrun logs set `OMP_NUM_THREADS=1` by default. Full CPU availability does not mean unrestricted OpenMP threading or full CPU use. |
| Parallelism | EP16, CP1, TP1, PP1, expert TP1, dense DP16. |
| Expert ownership | Eight local experts per rank. Actual ownership comes from runtime modules and process groups. |
| Sequence length | 4096 input tokens. |
| Microbatch size | 1 sequence per rank. |
| Gradient accumulation | 4 microbatches per optimizer update. |
| Global batch | `1 × 16 × 4 = 64` sequences per update. |
| Input tokens per update | `64 × 4096 = 262,144`. |
| Executed updates | 10, with updates 1–5 excluded from timing statistics. |
| Measured input tokens | `5 × 262,144 = 1,310,720` per case. |
| Total consumed input tokens | `10 × 262,144 = 2,621,440` per case. |
| Cache/training budget | `train_iters=25`, cache for 1600 samples, native `exit_interval=10`. |
| RNG / dataset seed | 1234. No new shuffle or reservoir selection per case. |
| Checkpoint writes / evaluation | Disabled for these measurements. |

The C0 manifest records its original EP8 write topology. Distributed checkpoint loading reshares that same checkpoint into the executed EP16 topology. This does not mean that these runs use EP8. The C0 contains 17 inventory files totaling 61,129,914,201 bytes.

Three vocabulary counts require separate names. The HF architecture manifest declares 151936. The tokenizer has 151669 entries. The runtime pads that tokenizer vocabulary with 11 dummy rows to 151680. Do not substitute one number for another when reviewing the model mapping. The [configuration investigation][config-evidence] links the manifest and actual padding log.

| Architecture / implementation detail | Saved value |
| --- | --- |
| Attention heads / query groups / explicit head dimension | `32 / 4 / 128`. Use the explicit head dimension, not `hidden_size / num_heads`. |
| Norm and epsilon | RMSNorm, `1e-6`, Q/K layer normalization enabled. |
| Activation | SiLU with a gated linear unit. |
| Position encoding | RoPE, base `1,000,000`, rotary fraction `1.0`. |
| Hidden / attention dropout | `0.0 / 0.0`. |
| Shared embedding and output weights | False. |
| Shared-expert intermediate size | Null. |
| Grouped expert GEMM | Enabled in all cases. This differs from grouped routing. |
| Layer implementation / attention backend | Transformer Engine / `auto`. This does not identify the actual selected attention kernel. |
| Gradient accumulation / masked-softmax / cross-entropy fusion | Enabled. Cross-entropy implementation is native. |
| MoE permutation fusion | Enabled. |
| Router / auxiliary-loss fusion | Disabled. |
| Bias-activation / bias-dropout / RoPE fusion | Disabled. |

The source/configuration investigation explains these fields and their enabling conditions. The full YAML also retains a dense FFN width of 6144. Every runtime transformer layer is an MoE layer with expert intermediate size 768. An unused provider field must not be counted as extra executed model work.

### Optimizer and schedule

AdamW behavior uses Adam with decoupled weight decay. The LR is constant throughout the recorded run. The framework scheduler counts consumed samples, so `lr_decay_steps=1600` corresponds to the retained 25-update budget.

| Setting | Value |
| --- | --- |
| Optimizer | Native distributed Adam / Transformer Engine fused optimizer path. |
| Learning rate / minimum LR | `1e-6 / 1e-6`. |
| LR schedule | Constant. |
| LR warmup | `lr_warmup_iters=0`, `lr_warmup_steps=0`, `lr_warmup_samples=0`. |
| Adam beta1 / beta2 | `0.9 / 0.95`. |
| Adam epsilon | `1e-8`. |
| Gradient clipping | Global norm threshold `1.0`. The logged norm is measured before clipping. |
| Optimizer weight-decay field | `0.1`. |
| Effective scheduled base weight decay | `0.033`, constant, multiplied by each parameter group's `wd_mult`. |
| Gradient-norm skip threshold | Positive infinity. This does not add a finite large-gradient rejection threshold. |
| Explicit large-gradient / spiky-loss checks | Disabled. |
| Loss scale | Logged as `1.0` for BF16. Do not infer FP16 dynamic scaling from inactive scaler defaults. |
| Log interval | 1 update. Every executed update supplies a health record. |

The scheduler discrepancy is a review finding, not a reason to relabel the historical runs as weight decay 0.1. Any future correction changes the workload and needs a new experiment identity. No correction is applied by this documentation task.

### Router, memory and overlap settings

Ordinary routing selects eight of 128 experts without the group restriction. Grouped routing first selects two of four groups, then selects eight experts within the allowed groups. It does not require four experts from each selected group. The runtime ownership map is needed to connect groups with node destinations.

| Setting | Value / qualification |
| --- | --- |
| Router top-k | 8 in all cases. |
| Router score / load balancing | Softmax, `moe_router_pre_softmax=false`, auxiliary loss coefficient `0.001`, load-balancing type `aux_loss`. |
| Router group restriction | Ordinary `None/None`, grouped `4/2`. |
| Forced uniform load balancing | Disabled. Actual trained router outputs remain in use. |
| Expert capacity / token dropping | Capacity factor `None`, no token dropping. |
| Sequence parallelism | Disabled. |
| Activation recomputation | Disabled. `recompute_granularity=None` and runtime checkpointed-expert-layer count is zero. |
| Residual `recompute_modules` entry | Resolved YAML lists `core_attn`. It is inactive without recompute granularity. |
| MoE communication overlap | Disabled, including expert backward/weight-gradient overlap and shared-expert overlap. |
| Dense DP gradient-reduce / parameter-gather overlap | Enabled through the existing framework configuration. |
| CUDA graphs | Disabled. |
| Quantization padding | Disabled. |
| Custom lightweight AMPA profiler | OFF in every case. The ON cases use the separate per-update PyTorch callback. |
| Built-in Bridge profiler flags | Both false. AMPA owns the separate capture lifecycle. |

Only seven flattened leaves differ between saved resolved YAML files: four intended router/backend fields and three logger paths/names. The profiler ON/OFF selection lives in AMPA callback declarations. It is not represented by a change to the built-in Bridge profiler flag.

The [configuration investigation][config-evidence] lists additional architecture, attention, normalization, fusion, loader and logger values with exact source anchors. Appendix B reproduces every field of the representative resolved YAML. Many entries are inactive framework defaults. A present key does not prove that its feature executes.

## 5. Data mixture, preparation and consumed samples

The selected workload uses real indexed text, not mock tokens. It performs continued pretraining through the existing language-model forward and loss path. The run does not use the chat template to turn the data into a chat task.

| Component | Blend weight | Indexed prefix under `dataset/indexed/qwen3-30b-a3b-base-realistic-mix-v2/` |
| --- | ---: | --- |
| Web | 55% | `web/qwen3_30b_a3b_base_realistic_mix_web` |
| Multilingual | 15% | `multilingual/qwen3_30b_a3b_base_realistic_mix_multilingual` |
| Code | 15% | `code/qwen3_30b_a3b_base_realistic_mix_code` |
| Math | 15% | `math/qwen3_30b_a3b_base_realistic_mix_math` |

These percentages specify blend weights. They are not a measurement of exact source proportions within the short consumed prefix. Document counts, token counts and sampled sequence proportions are different quantities.

The selected runtime manifest is [ep16-cache-20261011T1757Z-cp1-ep16.json][dataset-manifest]. It pins tokenizer identity, component artifacts, preparation provenance, the indexed consumer and the existing cache. The [data investigation][config-evidence] gives upstream datasets/revisions, document and token counts, extraction rules and component hashes. These details matter more than the four category names.

| Component | Pinned upstream source / revision | Selection | Materialized documents | Tokens including EOD |
| --- | --- | --- | ---: | ---: |
| Web | `HuggingFaceFW/fineweb` / `9bb295ddab0e05d785b879661af7260fed5140fc` | `sample-10BT`, train, text field `text`. | 10,538 | 7,209,408 |
| Multilingual | `HuggingFaceFW/fineweb-2` / `af9c13333eb981300149d5ca60a8e9d659b276b9` | Equal subweights for Arabic, German, French, Russian and Spanish configs. | 1,801 | 1,990,942 |
| Code | `codeparrot/github-code-clean` / `c48d40f9e70f0196f8236901ee35807f7d6c44c0` | Pinned parquet, `.py` paths, `license=apache-2.0`, field `code`. | 836 | 1,971,694 |
| Math | `open-web-math/open-web-math` / `fde8ef8de2300f5e778f56261843dab89f230815` | Default train split, field `text`, URL/text-hash identity. | 1,042 | 1,974,544 |

Preparation selects deterministic stream order until each token budget is reached. Complete documents can overshoot a target budget. Selection ledgers retain source identity, text hash, token count and provenance, without copying reservoir text. The materialized reservoir contains 13,146,588 tokens including EOD. It is larger than this campaign's consumed prefix.

Tokenization uses the pinned tokenizer once, int32 indexed tokens and EOD ID 151643. It appends EOD after documents. Training uses normal next-token prediction, `add_extra_token_to_sequence=true`, `eod_mask_loss=false`, and does not reset positions or attention at document boundaries. Actual loader configuration has two workers, persistent workers and pinned memory. The saved indexed/cache hashes and all consumed-batch fingerprints bind the result.

The runtime derivative intentionally retains historical preparation metadata from the CP2/EP8 cache. Its `validated_workload` names CP1/EP16/GA4. The parser accepts it through the reviewed runtime-derivative path. Its `canonical_training_allowed=false` flag must not be read alone as evidence that these executions bypass validation.

The cache retains the original 1600-sample budget. The short run consumes its first 640 samples, not a regenerated ten-update cache. Every rank records four actual CPU batches per update. The campaign compares all sixteen ranks across all ten updates, including batch field schema, dtype, shape and bytes through fingerprints.

There are 640 consumed batch records per case: `16 ranks × 10 updates × 4 microbatches`. This is also 640 global sequences because the microbatch size is one. Measured updates consume 320 sequences. Across all eight cases, 5120 batch records support the equality comparison.

Identical fingerprints establish equality of consumed input bytes at the recorded fields. They do not establish equal router choices, activations, gradients or final weights. Grouped routing changes the trajectory deliberately. BF16 accumulation and different backend paths can also change later routes.

## 6. Entrypoints and source changes to review

The training loop remains the native Bridge `pretrain` path with its native GPT forward step. AMPA adds configuration, callbacks, launch policy and posthoc analysis. The saved runtime identity records a clean Megatron-LM checkout. Bridge records a deleted test `.gitkeep` and an untracked core dump. Those recorded dirty entries are not model-math edits.

| Layer | Files / evidence | What changes and why |
| --- | --- | --- |
| Canonical configuration | [profiling/run_pretrain.py](../profiling/run_pretrain.py), especially `build_config`, `comparison_contract`, `main` | Loads C0, chooses topology/router/dispatcher, fixes batch/schedule and registers observers. Review actual resolved files as well as declarations. |
| Short-run adapter | [run_pretrain_debug.py](../profiling/run_pretrain_debug.py) | Preserves 25-update cache validation and `train_iters`, then uses native exit after update 10. Actual timing window is indices 5–9. |
| Reused-input adapter | [run_pretrain_debug_reuse.py](../profiling/run_pretrain_debug_reuse.py) | Temporarily replaces only C0/data payload-validation handlers during parsing, uses bound historical metadata, and restores the original handlers. |
| Reuse provenance | [uccl_ep_verified_inputs_reuse.py](../tools/uccl_ep_verified_inputs_reuse.py) | Binds the current job/run to first-case verification. Explicitly records that current-allocation payload checks did not run. |
| Launch / workers | [uccl_ep_debug_reuse.sbatch](../jobs/training/uccl_ep_debug_reuse.sbatch), [qwen3_uccl_debug_reuse_worker.sh](../jobs/training/qwen3_uccl_debug_reuse_worker.sh) | Four exclusive debug nodes, per-node torchrun, no CPU-binding restriction, same source/data and backend-specific flags. |
| Runtime records | [provenance.py](../profiling/ampa_profile/provenance.py), [sample_fingerprint.py](../profiling/ampa_profile/sample_fingerprint.py), [routing_audit.py](../profiling/ampa_profile/routing_audit.py) | Record actual topology, device identity, batches, routes and receives. The audits add work to both OFF and ON runs. |
| Diagnostic scopes | [deep_trace.py](../profiling/ampa_profile/deep_trace.py), [uccl_trace.py](../profiling/ampa_profile/uccl_trace.py) | Add scoped labels, native call counters and UCCL forward/backward journals. Return the original results and restore wrapped methods. |
| UCCL identity / cleanup | [uccl_identity.py](../profiling/ampa_profile/uccl_identity.py) | Binds the loaded binary/fabric/wrapper/device and safely releases final transient caches before native-stream destruction. |
| Native UCCL ownership fix | [archived patch][ownership-patch] and [source identity][uccl-source] | Four-file teardown fix prevents the proxy from freeing scratch owned by a tensor. It changes ownership/shutdown, not in-step routing or message math. |
| Host-memory monitor workaround | Worker `FI_MR_CACHE_MONITOR=disabled` | Avoids the captured libfabric/CUPTI reentrant mutex deadlock. Applied before initialization to all four UCCL legs. It can change registration cost. |
| Acceptance | [uccl_ep_debug_validate.py](../tools/uccl_ep_debug_validate.py), [uccl_debug_campaign_validate.py](../tools/uccl_debug_campaign_validate.py) | Reuses strict native/UCCL checks, validates every case and capture, binds raw evidence hashes and assembles campaign acceptance. |
| Report / independent arithmetic | [qwen_debug_results.py](../analysis/qwen_debug_results.py), [qwen_debug_profile_phases.py](../analysis/qwen_debug_profile_phases.py), [qwen_debug_independent_audit.py](../analysis/qwen_debug_independent_audit.py) | Restricts the report to accepted inputs/window and independently reproduces numerical claims. |

The runtime-source digest does not replace the short-run adapter inventory. The [first source manifest][source-manifest], per-acquisition `source/` archives and their SHA files bind those additional files. Current checkout files are convenient for review. Historical conclusions depend on the recorded and archived versions.

The seven reused acquisitions omit fresh bulk input verification and a fresh postflight source comparison. They retain source records, runtime identity callbacks and later evidence checks. They assume that the previously verified immutable payloads and reviewed execution sources remain unchanged. Final posthoc acceptance does not turn that historical reuse into a new per-allocation payload or postflight check.

Native pins are Bridge `3a459fe4ab5e5e6b6c489434c687cbc4a01a55b3`, Megatron-LM `e2c2e55bfe819e7d7b74306a042e30289e5ddfd3` and UCCL `c8fd14e93f2106b69a0d30e0f78c12a52e07a181` plus the recorded ownership patch. Training records PyTorch `2.8.0a0+34c6371d24.nv25.08`, CUDA 13.0 and NCCL 2.27.7.

The UCCL path uses the existing Flex/DeepEP API with UCCL's normal high-throughput `Buffer`. It does not substitute the newer low-latency dedup API. Flex converts transport probabilities to FP32. This backend bundle and its arithmetic order are part of the compared implementation.

The approved ownership change adds `stop(free_gpu_buffer=True)` and calls it with `False` for caller-owned scratch. After the last update, cleanup checks all 48 managers and clears eight known transient fields per manager. It preserves weights, optimizer state and process groups, and drains the known transients before destroying the native stream and scratch.

The official cleanup helper suppresses some stop/unregister exceptions. Clean worker exits and cleanup records therefore support this bounded lifecycle, not a claim that every internal cleanup operation is exception-free. The C++ signature change requires a matching rebuilt extension. Its public helper behavior is not a universal fix for unknown external callers.

Unused `3x5` adapters and launchers are drafts. No accepted run uses them. The user keeps five warmup plus five measured updates. This audit adds documentation only and does not edit the frozen training sources, `scripts/`, `setup/`, C0 or data.

## 7. Exactly what profiling adds

The ON legs capture every executed update, including warmup: indices 0–9 on ranks 0–15. There are `10 × 16 = 160` captures per ON case and 640 across four ON cases. The common phase report later selects only indices 5–9, or 80 rank-update observations per case.

Each selected update starts an AMPA-owned `torch.profiler.profile` context with `record_shapes=False` and `with_stack=False`. AMPA attaches method labels, enters `AMPA/STEP/<index>`, executes the native training update and validates call counts. It restores the wrapped methods before stopping the profiler and exporting a compressed Chrome trace. It also writes a per-update call journal.

| Scope family | Recorded coverage |
| --- | --- |
| Full step | Four microbatch forwards, four microbatch backwards, one gradient finalization and one optimizer step. |
| Router | Router total/gate/routing, main top-k, auxiliary scores and auxiliary loss for each layer and microbatch. |
| Native dispatch | Metadata, first permutation, activation/probability AllToAll, expert-local sort and dispatch total. |
| Expert computation | Expert total, FC1 and FC2. |
| Native combine | Inverse local sort, expert-output AllToAll, final unpermutation and combine total. |
| UCCL-specific communication | Layout, dispatch, combine, input/probability-gradient combine and output-gradient dispatch. |
| UCCL autograd journal | Forward/backward pair identity, completion, call epoch and payload dtype/bytes. Both adjoint communication paths must return. |
| Other model boundaries | Pre-MLP norm, MoE total and residual/bias/dropout/add scopes. |
| Device linkage | Recorded CUDA activities must correlate with the relevant CPU launches and scopes. Missing/orphaned required activity rejects quantitative acceptance. |

A CPU scope is a host interval. Correlated GPU activity is device work linked to that interval. Neither is automatically an exposed delay on the full-step critical path. Nested scopes overlap. Concurrent kernels overlap. Summing them does not partition the update timer.

The OFF timer still includes the actual routing/receive/input observers and native health work. UCCL also computes membership/layout checks that differ from native checks. The measured control is therefore the backend with these observers. Its time is not a measurement of observer-free training.

Profiler stop/export, acquisition gates and startup also affect job wall time. Moving posthoc gates outside the allocation removes those gates from the allocation budget. It does not remove the measurement observers or all diagnostic perturbation from the update.

## 8. Learning rate, gradients and activation coverage

### Learning rate for every update

All eight raw timer logs agree with this table. `Consumed samples` is the cumulative global count. The scalar audit cross-checks each of the 80 raw lines against the accepted report. Appendix A preserves every LR/loss/gradient value and timer.

| Human update | Internal index | Timing role | Cumulative samples | LR |
| ---: | ---: | --- | ---: | ---: |
| 1 | 0 | Warmup | 64 | `1e-6` |
| 2 | 1 | Warmup | 128 | `1e-6` |
| 3 | 2 | Warmup | 192 | `1e-6` |
| 4 | 3 | Warmup | 256 | `1e-6` |
| 5 | 4 | Warmup | 320 | `1e-6` |
| 6 | 5 | Measured | 384 | `1e-6` |
| 7 | 6 | Measured | 448 | `1e-6` |
| 8 | 7 | Measured | 512 | `1e-6` |
| 9 | 8 | Measured | 576 | `1e-6` |
| 10 | 9 | Measured | 640 | `1e-6` |

### Recorded health for all ten updates

Every case has ten finite LM-loss, load-balancing-loss, LR and gradient-norm records. All learning rates and gradient norms are positive. All skipped-update and NaN-update counters are zero. Logged gradients above one are compatible with the clipping threshold because the norm is recorded before clipping.

| Case / profiler | LM loss range | Logged gradient-norm range |
| --- | --- | --- |
| Ordinary / OFF | 1.822699–1.899306 | 0.441–0.543 |
| Ordinary / ON | 1.822906–1.899374 | 0.446–0.551 |
| Grouped / OFF | 2.014337–2.121368 | 1.049–1.374 |
| Grouped / ON | 2.014759–2.120836 | 1.042–1.382 |
| UCCL ordinary / OFF | 1.822799–1.899357 | 0.442–0.542 |
| UCCL ordinary / ON | 1.822821–1.899390 | 0.444–0.540 |
| UCCL grouped / OFF | 2.014922–2.120995 | 1.042–1.393 |
| UCCL grouped / ON | 2.015128–2.120940 | 1.050–1.395 |

The parser also requires the exact update sequence and `consumed_samples = update × global_batch_size`. It rejects missing, repeated or skipped update boundaries. See [summarize_training_evidence.py](../analysis/summarize_training_evidence.py) and each case's `training_health.json` under [case acceptance][case-proofs].

### Runtime guards and recorded evidence are different

| Signal / check | Actual coverage | Evidence and limit |
| --- | --- | --- |
| Nonfinite gradient guard | DDP gradient buckets test norms for NaN/Inf when `check_for_nan_in_grad=true`. | Enabled in saved configuration and bound native source. No separate per-bucket pass log is saved. |
| Nonfinite loss guard | Native loss code tests loss values for NaN/Inf. | `check_for_nan_in_loss=true`. Disabled rerun mode still executes fatal rejection logic. No full loss-tensor archive is saved. |
| Global gradient health | Positive finite logged norm on every one of 80 updates. | Direct raw-log evidence. It does not prove that every gradient element or parameter delta is correct. |
| Gradient clipping | Native threshold 1.0. | Source/configuration evidence. No separate post-clipping tensor/norm dump is saved. |
| Router probabilities | Each audited router call tests finite probabilities and exact top-8 membership. | Direct routing-audit counters. This concerns router outputs, not every hidden activation. |
| UCCL payload metadata | Selected IDs/weights and layout counts agree with the reference. | Actual route/layout counters and payload dtype/byte journals. These do not archive full activation values. |
| Bounded synthetic outputs/gradients | Exact synthetic outputs, input gradients, routing-probability gradients and expert-scale gradients across 16 ranks. | Historical accepted communication test, job 3827788. This is not an end-to-end Qwen parity test. |
| Every hidden activation finite | Not recorded as a complete scan. | No activation min/max/RMS/NaN tensor census. A profiler scope is not an activation-value check. |
| Every parameter gradient / weight delta | Not recorded. | No saved per-parameter gradient norms, gradient reference or before/after weight comparison. |
| Large-gradient / spiky-loss rejection | Disabled. | Do not describe finite gradient health as an outlier detector. |
| HF/native whole-model numerical equivalence | Not accepted. | Explicitly removed as a gate by D-014. Failed historical parity diagnostics remain preserved. |

The [check investigation][checks-evidence] names the exact native code paths and line numbers. Successful execution gives evidence that active guards did not raise. It does not replace a saved value-by-value correctness comparison.

Historical activation diagnostics also matter. Job 3484940 found exact equality at the sampled layer-0 pre-attention residual. It then found 17,560 of 32,768 BF16 residual elements different after attention, and two of 128 selected expert IDs different. That localized a numerical difference without proving its cause. The [recorded diagnostic](../progress/02_model_checkpoint/evidence/layer0-pre-attention-raw-residual-diagnostic-job-3484940-completed-2026-09-23.md) remains failed parity evidence. D-014 permits the narrower Megatron-native measurement claim.

The accepted UCCL synthetic test uses 32 saved rank/mode records, two cycles and four output/gradient comparisons per cycle. All 256 recorded comparisons have maximum absolute error zero. Nonuniform weights, expert-dependent gains, distinct token identities, zero-receive owners and reverse-order backward test more than an identity round trip. The fixture is bounded and does not exhaust all real BF16 inputs or uniquely encode every expert through its gains alone.

## 9. Additional reliability checks and what each proves

The checks operate at different levels. Source review establishes intended semantics. Runtime records establish actual execution facts. Posthoc joins test relationships that one rank cannot prove alone. Independent arithmetic reduces the chance that the report repeats its producer's calculation error.

| Check | Population / pass condition | What it protects against |
| --- | --- | --- |
| Model/checkpoint/tokenizer identity | Exact pinned identifiers, manifests and C0 inventory binding. | Accidental model version or starting-weight changes. |
| Runtime source identity | Common runtime digest, native checkout revision/diff identity, adapter archives and per-file SHA records. | Silent source drift or comparison of different code. |
| Actual topology and expert ownership | All 16 ranks, 48 layers, actual process groups/local expert IDs. | Wrong rank arithmetic, EP/DP/CP confusion or fabricated ownership. |
| GPU selection | CUDA current device, local rank, physical UUID and hostname uniqueness. | Multiple workers selecting the same GPU or using an unrelated GPU enumeration. |
| Batch identity | All 16 ranks, ten updates, four microbatches per update in every case. | Equal seeds hiding different consumed inputs or batch shapes. |
| Training health | Ten sequential boundaries, finite positive LR/gradient norm, finite losses, zero skip/NaN, final 640 samples. | Inference-only, incomplete or failed update evidence. |
| Actual top-k | Every audited layer/microbatch has eight expert choices per token. | Dropped/extra assignments or a changed top-k workload. |
| Group restriction | Grouped choices remain within two of four groups. Ordinary mode has the ordinary bound. | Treating an ordinary route as grouped or imposing grouped bounds on ordinary routes. |
| Assignment conservation | `sum(expert_counts) = sum(rank_traffic) = routed_tokens × 8`. | Incorrect counting, missing expert assignments or duplicated accounting. |
| Locality conservation | Local + intra-node + inter-node assignments equal the same assignment total. | Inconsistent locality definitions or an incorrect rank-to-node join. |
| Received expert ownership/counts | Join all source counts to each actual receiver's local expert IDs for every layer/update. | A plausible global total masking delivery to the wrong expert/rank. |
| UCCL metadata/layout | Selected IDs/weights and unique-rank/node counts agree with an independent membership construction. | Mislabeling assignment counts as deduplicated deliveries. |
| Complete training call grid | Four forwards/backwards plus one gradient finalization/optimizer call per captured update. | Partial forward-only traces labeled as full updates. |
| Layer/phase completeness | Every required phase for 48 layers and four microbatches per captured rank-update. | Missing scopes, wrong layer identity or incomplete capture windows. |
| UCCL autograd pairing | Both native forward/backward communication pairs complete with unique journals. | Forward-only communication evidence that misses reverse gradient paths. |
| CUDA correlation | All 640 captures pass backend-aware launch/activity/correlation gates. | Treating an empty or orphaned trace as quantitative GPU evidence. |
| Native backend identity | All UCCL ranks bind exact wrapper/binary/fabric bytes, normal Buffer and actual CUDA scratch. | Importing the wrong shared library or substituting a different API. |
| Actual transport initialization | Native endpoint/CXI-domain evidence, not requested environment alone. | Treating `UCCL_EP_TRANSPORT=cxi` as proof of transport use. |
| Complete shutdown | Successful worker exits plus UCCL cleanup records and source-bound acquisition markers. | Accepting numerical results from a process that later fails during teardown. |
| Input reuse honesty | First-case proof retained, later jobs explicitly marked as historical reuse. | Claiming checks ran again when the user removed them from the allocation. |
| Acceptance caches | Input/source hashes bind reused posthoc proofs. Changed/corrupt inputs invalidate caches. | Stale successful validation attached to changed raw evidence. |
| Report acceptance graph | Exact eight-case matrix, window, case proof hashes, raw input hashes and accepted phase summaries. | A valid report built from an excluded run, stale proof or wrong window. |
| Independent numerical reconstruction | Raw timers, all health records, fingerprint joins, route/receive conservation, contrasts and phase means. | Shared producer/report arithmetic or unit errors. |
| Targeted tests | Thirteen final adapter tests pass. Earlier frozen matrix QA and synthetic trace fixtures remain separate evidence. | Regressions in the new adapter, historical-proof handling, cache binding and CPU library mapping. |
| Independent exported-deck review | All 9 slides, 55 claims, 70 chart points and 354 text objects. | Correct JSON displayed with wrong values, units, labels or scope. |

For each measured case, the forward routing total is `1,310,720 input tokens × 48 layers × 8 = 503,316,480` expert assignments. The `62,914,560` routed-token count counts token-layer observations. It is not the number of unique input tokens.

The strict received-route checks cover all ten updates, including warmup. Reported routing summaries use only the five measured updates. The synthetic test also exercises two live forward handles, reverse-order backward, zero-receive owners and plain/observed modes. These adversarial cases test errors that an identity round trip can miss.

The final routing observer sees 30,720 router calls per case, or 245,760 across eight cases. Each case's all-source receive join tests `10 updates × 48 layers × 128 experts = 61,440` exact expert equalities. All eight cases pass 491,520 such equalities. Each ten-update case sends and receives 1,006,632,960 logical assignments.

Native trace acceptance requires at least 95% linked event count and duration, together with stricter linkage of eligible launches and required communication. It does not require every background event to link. Across accepted native captures, the minimum linked-event fractions are 0.9999917685 ordinary and 0.9999824009 grouped. Minimum linked-activity fractions are 0.9999984305 and 0.9999989509. This residual unlinked activity is explicit in the saved summaries.

Earlier check populations remain separate. Frozen four-case matrix QA passes 60 methods. Synthetic trace QA passes 14 plus eight methods and checks sixteen synthetic traces. The later four-case Qwen pilot contributes two real updates and 128 rank-update captures. The final CPU adapter passes thirteen methods before validating the eight final acquisitions. These tests do not add independent timing samples to the final five-update windows.

The checks do not make the eight runs independent statistical replicates. They improve evidence integrity and execution confidence. They do not supply missing network counters, occupancy data or long-run quality results.

Capture acceptance concerns the required call/scope grid and recorded host/device correlations. It cannot detect every event that disappears from both sides of a trace. It does not prove exhaustive capture of persistent kernels, CPU proxies or every copy/fill operation. “Complete” in this audit means that the defined completeness gates pass.

## 10. Assumptions and progression

The project changes scope as evidence and allocation limits become clear. This table records decisions that affect interpretation. It does not convert an earlier failed run into accepted data. Detailed job/source evidence appears in the [history investigation][history-evidence].

| Stage / decision | Assumption or finding | Resulting action and remaining limit |
| --- | --- | --- |
| Initial baseline, D-001–D-005 | Measure real continued pretraining from immutable C0 with reproducible text. | Keep model math, tokenizer/data identity and full optimizer updates. Fix the 55/15/15/15 mixture. |
| C0 numerical investigation, D-013/D-014 | HF and Megatron BF16 forwards do not select identical experts for every token. | User accepts the imported Megatron-native C0 without HF numerical parity. Identity and native training remain mandatory. |
| Memory/topology progression, D-017/D-018 | Earlier CP1 workload hits memory limits. Later four-node EP16 changes memory per rank. | Historical CP2/EP8 work remains separate. Current experiment is CP1/EP16 and uses its validated cache derivative. |
| Native measurement corrections, D-019–D-021 | Routing ownership, backward completeness, event correlation and plot scope require explicit gates. | Corrected native profiles establish accepted historical evidence. ON/OFF route differences block a pure-overhead claim. |
| UCCL integration, D-022 | Native transport and autograd correctness require direct tests. | Pin CXI/four-peer build, fix duplicate scratch ownership, test exact synthetic output/gradients and clean lifetime. |
| Profiler failure investigation | Timeouts alone do not identify a protocol bug. A debugger captures a reentrant libfabric memory-cache mutex deadlock. | Set supported `FI_MR_CACHE_MONITOR=disabled`. Do not apply an unproved transport/ring repair or raise timeouts as a fix. |
| Eight-case expansion, D-023 | Backend, routing and profiler each require their own controls. | Plan eight fresh C0 starts with 5+20 updates on one allocation. Validate a fresh four-case pilot first. |
| Short debug campaign, D-024 | Preemption and short debug allocations prevent the full campaign from completing. | User chooses 5+5, separate four-node allocations. Retain the original 25-update cache budget and disclose placement changes. |
| CPU/time request correction | A 22m30s request rounds to 23 whole minutes for the debug QoS budget. | Use 22 whole minutes, exclusive nodes and `--cpu-bind=none`. Four pending CPUs are not a four-CPU execution cap. |
| Deferred gates, D-025 | Repeating immutable checks consumes the short allocation. | Reuse successful first-case payload verification. Keep all measurement records and run detailed acceptance afterward. |
| Partition clarification, D-026 | An unrelated preemptable job should not hold the debug chain. | User permits coexistence across those partitions. Keep one four-node experiment at a time in this chain. |
| Timeout and reorder, D-027 | Original ordinary/ON first update takes 469s. Reducing warmup is considered. | User retains 5+5, runs four UCCL legs next and retries ordinary/ON last. No 3+5 job is submitted. |
| CPU posthoc, D-028/D-029 | Saved-trace analysis needs CPU, not GPU. | User authorizes an isolated CPU venv on the login node. Cancel duplicate queued posthoc jobs and keep actual execution provenance. |
| Final acceptance and audit | Eight jobs complete. All case/capture gates and independent arithmetic pass. | Deliver the descriptive report and independently reviewed deck. Preserve statistical, causal and numerical limits. |
| This configuration audit | Declared optimizer weight decay and applied scheduler weight decay differ. | Record effective base 0.033 and leave historical code/data unchanged. Separate inactive defaults from active behavior. |

The removed overhead includes the repeated immutable-payload preflight and detailed post-acquisition analysis gates. The first fresh preflight takes about 83 seconds. That single duration is not a measurement of every removable overhead component. Worker initialization, checkpoint loading, measurement observers, trace export and successful shutdown remain.

Warmup does not use a special statistical algorithm. It uses the same training path. Early update duration can include initialization effects. Earlier logs show warmup means and variability dominated by the first update. The 469s outlier is preserved, and its cause remains unknown.

The [five-versus-twenty study][five-v-twenty] reads six historical 25-update logs. First-five measured means differ from the full twenty by at most 1.077% in those records. Those five updates are a subset of the twenty, not an independent validation set. One grouped-native profile has first-five SD 0.29660s versus full-twenty SD 0.78066s. Five samples miss some later variability.

Therefore 5+5 is a defensible bounded diagnostic protocol. It is not evidence that five updates always reproduce twenty-update conclusions. Five consecutive updates do not measure variance across allocations. Small effects need repeated allocations and longer coverage before a strong conclusion.

## 11. Results supported by the accepted data

### Complete optimizer-update timing

The headline mean uses the five OFF update timers. Sample SD uses denominator `n−1`. CV is `100 × SD / mean`. Throughput is aggregate global input tokens per second, `262144 / mean_seconds`. It is not the arithmetic mean of individual reciprocal rates.

| Configuration | OFF mean (s) | OFF sample SD (s) | OFF CV | Global input tokens/s | ON mean (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Native ordinary | 64.84364 | 1.44739 | 2.232% | 4042.71 | 71.28104 |
| Native grouped | 65.42208 | 0.34075 | 0.521% | 4006.97 | 74.35522 |
| UCCL ordinary | 15.55356 | 0.06075 | 0.391% | 16854.28 | 30.49658 |
| UCCL grouped | 9.85958 | 0.11128 | 1.129% | 26587.75 | 25.00730 |

Observed native/UCCL mean-time ratios are 4.16905 for ordinary and 6.63538 for grouped. Within UCCL, grouped duration is 36.60885% lower than ordinary. Within native, grouped duration is 0.89205% higher. These are descriptive ratios of this setup, including its observers.

The large ratios motivate controlled follow-up. They do not isolate deduplication, prove a communication bottleneck or show that the native provider was tuned. The native transport provider/plugin was not recorded. Backend path, arithmetic, audit cost, allocation, contention and clocks remain possible contributors.

The observed ON/OFF duration changes are +9.92757%, +13.65463%, +96.07460% and +153.63454% in the table's order. The ON runs are slower in these observed comparisons. Different placements and later routes prevent attribution of those differences to profiling alone.

### Forward logical routing and load

Local means the source rank itself. Intra-node means another rank on the same node. Inter-node means a rank on another node. The percentages below count expert assignments, including multiple experts for one token at the same destination.

| OFF configuration | Local | Intra-node | Inter-node | Unique destination GPUs per token/layer | Unique destination nodes per token/layer |
| --- | ---: | ---: | ---: | ---: | ---: |
| Native ordinary | 6.184% | 18.724% | 75.092% | Unavailable | Unavailable |
| Native grouped | 6.178% | 18.654% | 75.168% | Unavailable | Unavailable |
| UCCL ordinary | 6.183% | 18.724% | 75.092% | 6.60223 | 3.64164 |
| UCCL grouped | 6.178% | 18.655% | 75.167% | 5.51710 | 1.99995 |

The UCCL destination averages include local destinations. Remote-node averages are 2.73167 ordinary and 1.50268 grouped. Native marginal expert/rank counts cannot recover per-token unique destination membership, so those cells stay unavailable.

Grouped UCCL lowers unique destination fanout while leaving roughly 75% of expert assignments inter-node. Both facts can hold at once. Fewer distinct nodes per token is different from fewer expert assignments to remote nodes. Logical destination counts do not measure transmitted NIC bytes.

Load spread also changes. For each layer/update, node-load CV uses population SD across four incoming-node assignment totals, divided by their mean. The report then averages those CVs over 240 measured layer/update observations. This differs from timing CV, which uses sample SD across five update times.

Mean incoming node-load CV is 0.09405 for native ordinary and 0.17971 for native grouped. UCCL values are 0.09406 and 0.17973. These CV values are fractions, not percentages. Grouping concentrates destinations and increases this node-load spread in the observed workload.

### Inclusive GPU phase activity

These values come from the ON runs. Each cell averages inclusive correlated GPU seconds across 16 ranks and five measured updates. The 80 rank-update observations are not 80 independent experiments. The phase definitions remain the same across cases where the native interfaces permit them.

| Forward phase | Native ordinary | Native grouped | UCCL ordinary | UCCL grouped |
| --- | ---: | ---: | ---: | ---: |
| Router | 0.04328 | 0.05619 | 0.04350 | 0.05669 |
| Dispatch | 14.29078 | 14.63476 | 3.36832 | 2.16911 |
| Expert computation | 0.17459 | 0.17532 | 0.15882 | 0.16021 |
| Combine | 14.67647 | 15.20534 | 4.19280 | 2.33390 |

These are forward-phase inclusive activity measurements. They do not include a complete additive accounting of backward, attention, gradient communication or the optimizer. The final report also retains the corresponding CPU-scope values. Neither set is an OFF phase-cost estimate.

Do not make a step-time pie chart from this table. Overlap, nesting and waiting make that interpretation invalid. The delivered routing donuts partition mutually exclusive assignment categories, so those charts are valid partitions of their stated quantity.

## 12. Communication mechanism and missing metrics

SMs execute GPU kernels. The NIC transfers data through registered GPU buffers. UCCL's CPU proxy submits network requests and processes completions. The pinned normal-mode code gives SMs packing, NVLink forwarding, receive/reordering, combine reductions and queue/credit polling work.

The software buffer is registered GPU memory, not an assumed buffer in NIC onboard memory. Cross-node NIC DMA moves its payload. GPU kernels also wait for data and credits. Their elapsed activity does not equal useful copying time or measured SM occupancy.

Recorded UCCL runtime metadata reports a 20-SM launch budget, normal Buffer mode, CUDA scratch and scratch capacities of 70,872,832 NVLink bytes and 69,084,800 RDMA bytes for the documented runtime instance. Those are capacities. They are not transferred bytes. See [the source-based SM explanation][sm-evidence] and each UCCL rank's runtime record.

| Quantity | Availability in this campaign |
| --- | --- |
| Complete optimizer-update time | Available from native timers. |
| Forward expert assignments and locality | Available and reconciled. |
| UCCL per-token unique rank/node destinations | Available and checked. |
| Required phase scopes and correlated GPU activity | Available for all ON captures. |
| UCCL autograd pair completion and payload metadata | Available in call journals. |
| Physical NIC transmit/receive bytes or bandwidth | Unavailable. |
| Exposed communication latency / step critical path | Unavailable as a separated quantity. |
| CPU proxy execution time and scheduling delay | Unavailable as a separated quantity. |
| SM occupancy / useful-copy versus polling fraction | Unavailable. `num_sms=20` does not supply it. |
| Pure profiler overhead / pure observer overhead | Unavailable. |
| Observer-free whole-step throughput | Unavailable. |
| End-to-end activation/gradient/weight numerical parity | Unavailable. |
| Long-run convergence, validation quality or perplexity | Unavailable. |

## 13. Evidence-supported research directions

These are follow-up questions derived from the current data. They are not claims that a proposed optimization works. The first experiment for each direction must preserve workload identity and add only the controls needed to answer that question.

| Direction | Existing evidence | Inference / next measurement |
| --- | --- | --- |
| Reproduce backend timing on controlled placements | Large OFF native/UCCL ratios, five successive within-run samples, independent allocations. | Repeat matched node/GPU placements and record actual native provider, clocks and contention. Test the setup effect before attributing its cause. |
| Destination fanout versus physical bytes | UCCL grouped unique nodes/token fall 3.64164→1.99995 while assignment locality stays near 75% inter-node. | Measure physical counters with matched routes to test whether fewer destinations reduce transferred bytes or network cost. |
| Grouping versus load balance | Grouped node-load CV rises about 0.094→0.180. UCCL grouped duration falls, while native duration changes little. | Study fanout, receiver load and critical-path ranks jointly. A logical reduction can coexist with concentrated work. |
| Diagnostic perturbation | Observed ON/OFF duration changes reach 153.6%. | Use matched placement, order/replicates and input/route controls. Separate profiler lifecycle cost from routing/audit and runtime perturbation. |
| Communication GPU work versus waiting | Dispatch/combine inclusive activity is large in native and smaller in UCCL. Source assigns polling and forwarding to GPU kernels. | Add occupancy/instruction/stall and stream/proxy evidence. Do not infer useful SM copying time from kernel duration. |
| Numerical behavior of dispatcher bundles | Exact bounded synthetic gradients pass. Real grouped losses and gradients differ from ordinary, and backend trajectories are close but unequal. | Compare selected end-to-end activations/gradients/updates with declared BF16 tolerances if numerical equivalence becomes a research goal. |
| Statistical stability and startup | Historical prefix means often approximate twenty updates, but short windows understate some variability. One first update takes 469s. | Repeat independent allocations and extend observation when budget permits. Keep startup and steady-state questions separate. |
| Configuration clarity | Scheduler applies 0.033 despite the optimizer's 0.1 field. Inactive defaults can look active in YAML. | Make future intended values explicit and add effective runtime scalar records. Treat corrected runs as a new experiment. |

A broad claim such as “communication-bound” needs evidence of exposed communication on the step's critical path. Summed NCCL or UCCL kernel durations do not supply that proof. A broad claim such as “same model quality” needs longer training and evaluation. Finite health over ten updates does not supply it.

## 14. How to review or reproduce the documentary checks

Run inspection commands from the repository root. The commands below do not launch training or modify an acquisition. Hashing the final report and audit is small. Revalidating all saved traces is a separate CPU task over roughly 10 GB of compressed data.

```bash
sha256sum \
  runs/qwen-debug-results-20261105T0936Z-report/report.json \
  runs/qwen-debug-results-20261105T0936Z-report/independent-audit.json

jq '{valid_debug_campaign, rank_update_captures, warmup_steps, measured_steps}' \
  runs/qwen-debug-results-20261105T0925Z-cpu-validation/campaign_acceptance.json

jq '{valid_independent_audit, report_sha256}' \
  runs/qwen-debug-results-20261105T0936Z-report/independent-audit.json

.runtime/venvs/qwen-posthoc-cpu/bin/python \
  ai/qwen-experiment-audit/iteration-01/build_audit_snapshot.py
```

The last command reads small report/log files, reproduces the tables and checks all 80 scalar records. It does not repeat bulk C0/data hashing or trace acceptance. The user explicitly permits this CPU-only posthoc work on the login node. Training and GPU workloads still require compute allocations.

For a code review, follow the native trainer caller first, then each added callback and its restoration/cleanup path. Compare resolved runtime configuration with the declared contract. Read the exact per-case proof before accepting a plotted value. Preserve failed evidence and historical reused-input labels.

Posthoc analysis runs in `.runtime/venvs/qwen-posthoc-cpu` with honest hostname and no invented Slurm job ID. Final trace validation uses eight CPU workers after an initial two-worker attempt. The earlier D-029 registry names two workers. The timeline records the later change and preserved partial attempts. No capture gate is removed.

The container's recorded libfabric alias is absent on the login node. The validator uses an explicit scoped map to the exact recorded bytes recovered from the CSCS public network stack. It rejects the different stock-image library. The [library map][library-map] binds the original alias/canonical path, copy SHA and host-hook evidence. It does not load the native GPU backend or simulate a GPU run.

The final adapter passes thirteen targeted tests, including corrupt evidence/cache rejection, historical/current proof semantics, exact library mapping and restoration of the scoped adapter. The [source review][posthoc-review] and [independent numerical audit][numeric-audit] remain separate checks. The audit does not rehash the full C0 or raw dataset payloads for seven later acquisitions.

## 15. Findings to retain when using these results

The code and accepted evidence describe real optimizer updates with reproducible input, strong routing/receive accounting and captures that pass the defined completeness gates. The central descriptive measurements are independently reproducible from raw records. Failed attempts are excluded rather than silently recovered into the matrix.

The main source/configuration finding is effective scheduled weight decay 0.033 instead of the optimizer field's 0.1. The launcher grants full CPU affinity, but torchrun sets OpenMP threads to one by default. Other apparent inconsistencies have documented explanations: 25 declared updates versus ten executed, inactive recompute defaults, AMPA-owned profiling with built-in flags false, historical cache topology and runtime vocabulary padding.

The main trust limits are separate allocations, five consecutive measured updates, different observer costs, unknown native provider, historical input-check reuse and missing whole-model numerical/activation evidence. Use the large observed timing and fanout differences to choose controlled follow-up experiments. Do not convert those observations into unmeasured causal, wire-traffic or model-quality claims.

## Appendix A. Every logged update and measured timer

The following tables come from [the documentary extractor][audit-extractor]. Its [JSON snapshot][audit-snapshot] records source paths and raw-log hashes. Values match the accepted report and all 80 raw log lines. The complete table appears below so that this document remains useful without opening another summary.

Read-only extraction from the accepted report, cross-checked against all 80 raw Bridge update log lines.

`control` means profiler OFF. `profile` means profiler ON. Every row is a complete optimizer update.

Updates 1–5 are timing warmup. Updates 6–10 supply the measured timing samples. Every update trains with LR `1e-6`.

### Five measured timing samples

| Case | Profiler | Update 6 | Update 7 | Update 8 | Update 9 | Update 10 | Mean (s) | Sample SD (s) | CV |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ordinary | control | 66.5139 | 65.3907 | 65.6137 | 63.6931 | 63.0068 | 64.84364 | 1.44739 | 2.232% |
| ordinary | profile | 70.3536 | 72.0682 | 71.1622 | 71.3163 | 71.5049 | 71.28104 | 0.62154 | 0.872% |
| grouped | control | 65.4210 | 65.6913 | 65.4657 | 65.6794 | 64.8530 | 65.42208 | 0.34075 | 0.521% |
| grouped | profile | 74.6553 | 74.5540 | 74.3743 | 74.5731 | 73.6194 | 74.35522 | 0.42392 | 0.570% |
| uccl_ordinary | control | 15.5326 | 15.6375 | 15.4745 | 15.5404 | 15.5828 | 15.55356 | 0.06075 | 0.391% |
| uccl_ordinary | profile | 30.1046 | 30.6413 | 30.6986 | 30.9355 | 30.1029 | 30.49658 | 0.37518 | 1.230% |
| uccl | control | 9.7913 | 10.0482 | 9.7858 | 9.7991 | 9.8735 | 9.85958 | 0.11128 | 1.129% |
| uccl | profile | 24.7157 | 24.7533 | 25.1324 | 25.6586 | 24.7765 | 25.00730 | 0.40083 | 1.603% |

### Every logged learning rate, loss and gradient norm

These values are Bridge scalars. They do not contain layer activation tensors, per-parameter gradients or parameter deltas.

All 80 rows record zero skipped updates and zero NaN updates. The gradient norm is the logged global norm before clipping.

#### ordinary/control, job 3854430

Source: [runs/uccl-debug5x5-reuse-20261104T1615Z-ordinary-control-ordinary-control/logs/rank-15.out](../runs/uccl-debug5x5-reuse-20261104T1615Z-ordinary-control-ordinary-control/logs/rank-15.out).

| Update | Window | Consumed samples | LR | LM loss | Load-balancing loss | Gradient norm | Time (s) |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Warmup | 64 | 1e-06 | 1.869538 | 2.390183 | 0.468 | 91.0669 |
| 2 | Warmup | 128 | 1e-06 | 1.889644 | 2.390579 | 0.541 | 63.2392 |
| 3 | Warmup | 192 | 1e-06 | 1.886157 | 2.298309 | 0.461 | 64.5269 |
| 4 | Warmup | 256 | 1e-06 | 1.822699 | 2.250946 | 0.494 | 64.4842 |
| 5 | Warmup | 320 | 1e-06 | 1.878913 | 2.366684 | 0.480 | 63.2489 |
| 6 | Measured | 384 | 1e-06 | 1.856200 | 2.212511 | 0.543 | 66.5139 |
| 7 | Measured | 448 | 1e-06 | 1.899306 | 2.320472 | 0.458 | 65.3907 |
| 8 | Measured | 512 | 1e-06 | 1.885904 | 2.186601 | 0.512 | 65.6137 |
| 9 | Measured | 576 | 1e-06 | 1.869123 | 2.224016 | 0.488 | 63.6931 |
| 10 | Measured | 640 | 1e-06 | 1.888500 | 2.318421 | 0.441 | 63.0068 |

#### ordinary/profile, job 3861954

Source: [runs/uccl-debug5x5-reordered-20261104T2221Z-ordinary-profile-ordinary-profile/logs/rank-15.out](../runs/uccl-debug5x5-reordered-20261104T2221Z-ordinary-profile-ordinary-profile/logs/rank-15.out).

| Update | Window | Consumed samples | LR | LM loss | Load-balancing loss | Gradient norm | Time (s) |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Warmup | 64 | 1e-06 | 1.869538 | 2.390183 | 0.468 | 105.1414 |
| 2 | Warmup | 128 | 1e-06 | 1.889700 | 2.390532 | 0.551 | 71.8065 |
| 3 | Warmup | 192 | 1e-06 | 1.886320 | 2.298216 | 0.469 | 71.2861 |
| 4 | Warmup | 256 | 1e-06 | 1.822906 | 2.250948 | 0.493 | 71.7660 |
| 5 | Warmup | 320 | 1e-06 | 1.878909 | 2.366621 | 0.480 | 70.1599 |
| 6 | Measured | 384 | 1e-06 | 1.856365 | 2.212535 | 0.483 | 70.3536 |
| 7 | Measured | 448 | 1e-06 | 1.899374 | 2.320367 | 0.491 | 72.0682 |
| 8 | Measured | 512 | 1e-06 | 1.886035 | 2.186631 | 0.514 | 71.1622 |
| 9 | Measured | 576 | 1e-06 | 1.869317 | 2.223944 | 0.491 | 71.3163 |
| 10 | Measured | 640 | 1e-06 | 1.888638 | 2.318458 | 0.446 | 71.5049 |

#### grouped/control, job 3854037

Source: [runs/uccl-debug5x5-reuse-20261104T1615Z-grouped-control-grouped-control/logs/rank-15.out](../runs/uccl-debug5x5-reuse-20261104T1615Z-grouped-control-grouped-control/logs/rank-15.out).

| Update | Window | Consumed samples | LR | LM loss | Load-balancing loss | Gradient norm | Time (s) |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Warmup | 64 | 1e-06 | 2.088584 | 2.383717 | 1.254 | 92.0986 |
| 2 | Warmup | 128 | 1e-06 | 2.121368 | 2.379564 | 1.374 | 65.9051 |
| 3 | Warmup | 192 | 1e-06 | 2.079575 | 2.290487 | 1.049 | 65.2215 |
| 4 | Warmup | 256 | 1e-06 | 2.014337 | 2.243939 | 1.093 | 65.8167 |
| 5 | Warmup | 320 | 1e-06 | 2.092687 | 2.358983 | 1.200 | 65.4650 |
| 6 | Measured | 384 | 1e-06 | 2.053761 | 2.206581 | 1.081 | 65.4210 |
| 7 | Measured | 448 | 1e-06 | 2.106805 | 2.312592 | 1.084 | 65.6913 |
| 8 | Measured | 512 | 1e-06 | 2.113169 | 2.182171 | 1.286 | 65.4657 |
| 9 | Measured | 576 | 1e-06 | 2.092481 | 2.217801 | 1.126 | 65.6794 |
| 10 | Measured | 640 | 1e-06 | 2.091924 | 2.310391 | 1.055 | 64.8530 |

#### grouped/profile, job 3851782

Source: [runs/uccl-debug5x5-20261104T1415Z-grouped-profile-grouped-profile/logs/rank-15.out](../runs/uccl-debug5x5-20261104T1415Z-grouped-profile-grouped-profile/logs/rank-15.out).

| Update | Window | Consumed samples | LR | LM loss | Load-balancing loss | Gradient norm | Time (s) |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Warmup | 64 | 1e-06 | 2.088584 | 2.383717 | 1.254 | 105.0262 |
| 2 | Warmup | 128 | 1e-06 | 2.120836 | 2.379757 | 1.382 | 74.7370 |
| 3 | Warmup | 192 | 1e-06 | 2.079804 | 2.290529 | 1.042 | 74.8519 |
| 4 | Warmup | 256 | 1e-06 | 2.014759 | 2.243979 | 1.095 | 74.4586 |
| 5 | Warmup | 320 | 1e-06 | 2.093672 | 2.358689 | 1.186 | 73.9267 |
| 6 | Measured | 384 | 1e-06 | 2.053917 | 2.206751 | 1.082 | 74.6553 |
| 7 | Measured | 448 | 1e-06 | 2.106990 | 2.312716 | 1.112 | 74.5540 |
| 8 | Measured | 512 | 1e-06 | 2.113136 | 2.182094 | 1.284 | 74.3743 |
| 9 | Measured | 576 | 1e-06 | 2.092531 | 2.217853 | 1.098 | 74.5731 |
| 10 | Measured | 640 | 1e-06 | 2.090756 | 2.310698 | 1.048 | 73.6194 |

#### uccl_ordinary/control, job 3861721

Source: [runs/uccl-debug5x5-reordered-20261104T2221Z-uccl_ordinary-control-uccl_ordinary-control/logs/rank-15.out](../runs/uccl-debug5x5-reordered-20261104T2221Z-uccl_ordinary-control-uccl_ordinary-control/logs/rank-15.out).

| Update | Window | Consumed samples | LR | LM loss | Load-balancing loss | Gradient norm | Time (s) |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Warmup | 64 | 1e-06 | 1.869477 | 2.390214 | 0.473 | 51.0792 |
| 2 | Warmup | 128 | 1e-06 | 1.889719 | 2.390630 | 0.542 | 16.5666 |
| 3 | Warmup | 192 | 1e-06 | 1.886345 | 2.298288 | 0.462 | 15.6601 |
| 4 | Warmup | 256 | 1e-06 | 1.822799 | 2.250924 | 0.487 | 15.5954 |
| 5 | Warmup | 320 | 1e-06 | 1.878860 | 2.366644 | 0.480 | 15.6513 |
| 6 | Measured | 384 | 1e-06 | 1.856153 | 2.212475 | 0.480 | 15.5326 |
| 7 | Measured | 448 | 1e-06 | 1.899357 | 2.320406 | 0.467 | 15.6375 |
| 8 | Measured | 512 | 1e-06 | 1.885955 | 2.186547 | 0.525 | 15.4745 |
| 9 | Measured | 576 | 1e-06 | 1.869356 | 2.224000 | 0.493 | 15.5404 |
| 10 | Measured | 640 | 1e-06 | 1.888419 | 2.318457 | 0.442 | 15.5828 |

#### uccl_ordinary/profile, job 3861781

Source: [runs/uccl-debug5x5-reordered-20261104T2221Z-uccl_ordinary-profile-uccl_ordinary-profile/logs/rank-15.out](../runs/uccl-debug5x5-reordered-20261104T2221Z-uccl_ordinary-profile-uccl_ordinary-profile/logs/rank-15.out).

| Update | Window | Consumed samples | LR | LM loss | Load-balancing loss | Gradient norm | Time (s) |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Warmup | 64 | 1e-06 | 1.869477 | 2.390214 | 0.473 | 125.0201 |
| 2 | Warmup | 128 | 1e-06 | 1.889686 | 2.390558 | 0.540 | 31.1685 |
| 3 | Warmup | 192 | 1e-06 | 1.886304 | 2.298231 | 0.460 | 29.8513 |
| 4 | Warmup | 256 | 1e-06 | 1.822821 | 2.250894 | 0.488 | 30.3142 |
| 5 | Warmup | 320 | 1e-06 | 1.878788 | 2.366623 | 0.482 | 29.9683 |
| 6 | Measured | 384 | 1e-06 | 1.856450 | 2.212462 | 0.479 | 30.1046 |
| 7 | Measured | 448 | 1e-06 | 1.899390 | 2.320409 | 0.464 | 30.6413 |
| 8 | Measured | 512 | 1e-06 | 1.886036 | 2.186634 | 0.516 | 30.6986 |
| 9 | Measured | 576 | 1e-06 | 1.869124 | 2.223990 | 0.493 | 30.9355 |
| 10 | Measured | 640 | 1e-06 | 1.888502 | 2.318361 | 0.444 | 30.1029 |

#### uccl/control, job 3858901

Source: [runs/uccl-debug5x5-reordered-20261104T2221Z-uccl-control-uccl-control/logs/rank-15.out](../runs/uccl-debug5x5-reordered-20261104T2221Z-uccl-control-uccl-control/logs/rank-15.out).

| Update | Window | Consumed samples | LR | LM loss | Load-balancing loss | Gradient norm | Time (s) |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Warmup | 64 | 1e-06 | 2.088175 | 2.383531 | 1.266 | 61.2329 |
| 2 | Warmup | 128 | 1e-06 | 2.120995 | 2.379835 | 1.393 | 11.4460 |
| 3 | Warmup | 192 | 1e-06 | 2.079920 | 2.290610 | 1.042 | 10.1249 |
| 4 | Warmup | 256 | 1e-06 | 2.014922 | 2.244010 | 1.092 | 9.9648 |
| 5 | Warmup | 320 | 1e-06 | 2.093330 | 2.358792 | 1.198 | 9.9004 |
| 6 | Measured | 384 | 1e-06 | 2.053872 | 2.206590 | 1.087 | 9.7913 |
| 7 | Measured | 448 | 1e-06 | 2.107025 | 2.312603 | 1.218 | 10.0482 |
| 8 | Measured | 512 | 1e-06 | 2.113746 | 2.182153 | 1.308 | 9.7858 |
| 9 | Measured | 576 | 1e-06 | 2.092149 | 2.217969 | 1.098 | 9.7991 |
| 10 | Measured | 640 | 1e-06 | 2.091453 | 2.310824 | 1.047 | 9.8735 |

#### uccl/profile, job 3861400

Source: [runs/uccl-debug5x5-reordered-20261104T2221Z-uccl-profile-uccl-profile/logs/rank-15.out](../runs/uccl-debug5x5-reordered-20261104T2221Z-uccl-profile-uccl-profile/logs/rank-15.out).

| Update | Window | Consumed samples | LR | LM loss | Load-balancing loss | Gradient norm | Time (s) |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Warmup | 64 | 1e-06 | 2.088175 | 2.383531 | 1.265 | 75.8020 |
| 2 | Warmup | 128 | 1e-06 | 2.120940 | 2.379705 | 1.395 | 26.6554 |
| 3 | Warmup | 192 | 1e-06 | 2.079693 | 2.290606 | 1.061 | 25.5617 |
| 4 | Warmup | 256 | 1e-06 | 2.015128 | 2.244040 | 1.105 | 25.2277 |
| 5 | Warmup | 320 | 1e-06 | 2.093050 | 2.358762 | 1.189 | 25.1452 |
| 6 | Measured | 384 | 1e-06 | 2.054678 | 2.206704 | 1.107 | 24.7157 |
| 7 | Measured | 448 | 1e-06 | 2.105819 | 2.312721 | 1.077 | 24.7533 |
| 8 | Measured | 512 | 1e-06 | 2.113221 | 2.182338 | 1.300 | 25.1324 |
| 9 | Measured | 576 | 1e-06 | 2.092517 | 2.217936 | 1.134 | 25.6586 |
| 10 | Measured | 640 | 1e-06 | 2.091489 | 2.310871 | 1.050 | 24.7765 |

## Appendix B. Complete representative resolved configuration

This is the saved `ordinary/control` resolved YAML from job 3854430, reproduced verbatim. It contains 957 lines. The secret-field inspection finds no nonempty API key, password or credential value. `comet_api_key` is null and `env_vars` is empty. All eight original resolved files remain under their run directories in [campaign case_runs][campaign].

The complete YAML contains inactive defaults for inference, alternate optimizers, precision modes, offloading and profiling. Interpret a field with its enabling conditions and actual runtime records. Only seven flattened leaves differ across the eight YAML files: four intended router/backend leaves and three logger leaves. The ON/OFF callback configuration is recorded separately in `all_step_capture.json` and `observation_mode.json`.

<details>
<summary>Show all resolved YAML fields for native ordinary / profiler OFF</summary>

```yaml
_target_: megatron.bridge.training.config.ConfigContainer
checkpoint:
  _target_: megatron.bridge.training.config.CheckpointConfig
  also_save_hf_checkpoint: false
  async_ckpt_cpu_priority: 10
  async_ckpt_io_priority: 3
  async_ckpt_use_cpu_shm: false
  async_save: false
  async_strategy: nvrx
  async_write_results_mp_mode: fork
  auto_detect_ckpt_format: false
  ckpt_assume_constant_structure: false
  ckpt_convert_format: null
  ckpt_convert_save: null
  ckpt_convert_update_legacy_dist_opt_format: false
  ckpt_format: torch_dist
  ckpt_fully_parallel_load_exchange_algo: broadcast
  ckpt_fully_parallel_load_per_rank_objects: false
  ckpt_fully_parallel_load_process_group: dp
  ckpt_fully_parallel_save_process_group: dp
  ckpt_load_validate_sharding_integrity: true
  ckpt_pg_tensors_cache_create: false
  ckpt_pg_tensors_cache_path: null
  ckpt_step: null
  custom_manager_class: null
  dist_ckpt_optim_fully_reshardable: false
  dist_ckpt_save_pre_mcore_014: false
  dist_ckpt_strictness: assume_ok_unexpected
  dist_ckpt_workers: 1
  distrib_optim_fully_reshardable_mem_efficient: false
  exit_on_missing_checkpoint: false
  finetune: true
  fully_parallel_load: false
  fully_parallel_save: true
  hf_distributed_save: false
  hf_save_every_n_ranks: 1
  hf_source_path: null
  hf_trust_remote_code: false
  load: null
  load_main_params_from_ckpt: false
  load_optim: true
  load_rng: true
  most_recent_k: -1
  non_persistent_ckpt_type: null
  non_persistent_global_ckpt_dir: null
  non_persistent_local_ckpt_algo: fully_parallel
  non_persistent_local_ckpt_dir: null
  non_persistent_save_interval: null
  pretrained_checkpoint: /iopsstor/scratch/cscs/anunay/swissai/post-training/ampa/checkpoint/megatron/qwen3-30b-a3b-base-c0
  replication: false
  replication_factor: 2
  replication_jump: null
  save: null
  save_activations_interval: null
  save_dgrads_interval: null
  save_interval: 500
  save_optim: true
  save_params_interval: null
  save_retain_interval: null
  save_rng: true
  save_rng_state_per_dp_rank: false
  save_tokenizer_assets: true
  save_tokens_per_expert_interval: null
  save_wgrads_interval: null
  stage_precision_aware_optimizer_state_on_cpu: false
  storage_writers_per_rank: 1
  strict_fsdp_dtensor_load: false
  use_checkpoint_args: false
  use_mp_args_from_checkpoint_args: false
  use_persistent_ckpt_worker: true
  use_tokenizer_model_from_checkpoint_args: true
  verify_integrity: false
comm_overlap: null
dataset:
  _target_: megatron.bridge.training.config.GPTDatasetConfig
  add_extra_token_to_sequence: true
  allow_ambiguous_pad_tokens: false
  blend: null
  blend_per_split:
  - - - /iopsstor/scratch/cscs/anunay/swissai/post-training/ampa/dataset/indexed/qwen3-30b-a3b-base-realistic-mix-v2/web/qwen3_30b_a3b_base_realistic_mix_web
      - /iopsstor/scratch/cscs/anunay/swissai/post-training/ampa/dataset/indexed/qwen3-30b-a3b-base-realistic-mix-v2/multilingual/qwen3_30b_a3b_base_realistic_mix_multilingual
      - /iopsstor/scratch/cscs/anunay/swissai/post-training/ampa/dataset/indexed/qwen3-30b-a3b-base-realistic-mix-v2/code/qwen3_30b_a3b_base_realistic_mix_code
      - /iopsstor/scratch/cscs/anunay/swissai/post-training/ampa/dataset/indexed/qwen3-30b-a3b-base-realistic-mix-v2/math/qwen3_30b_a3b_base_realistic_mix_math
    - - 0.55
      - 0.15
      - 0.15
      - 0.15
  - null
  - null
  context_parallel_size: null
  create_attention_mask: true
  data_parallel_size: 16
  data_path: null
  data_sharding: true
  dataloader_load: null
  dataloader_save: null
  dataloader_type: single
  defer_npy_index_mmap: false
  drop_last: true
  drop_last_partial_validation_sequence: true
  eod_mask_loss: false
  fast_cache_load: true
  full_validation: null
  hybrid_context_parallel: false
  inter_document_masking: false
  mid_level_dataset_surplus: 0.005
  mmap_bin_files: true
  mock: false
  multiple_validation_sets: null
  num_dataset_builder_threads: 1
  num_workers: 2
  object_storage_cache_path: null
  path_to_cache: /iopsstor/scratch/cscs/anunay/swissai/post-training/ampa/dataset/metadata/cache/qwen3-30b-a3b-base-realistic-mix-v2/90ba67d7f50ec2da19005e5e939bc1fe8fb2f0091f4430bf02a85dcff93f7849
  per_dataset_sequences_path: /iopsstor/scratch/cscs/anunay/swissai/post-training/ampa/dataset/metadata/cache/qwen3-30b-a3b-base-realistic-mix-v2/90ba67d7f50ec2da19005e5e939bc1fe8fb2f0091f4430bf02a85dcff93f7849/per_dataset_sequences.json
  persistent_workers: true
  pin_memory: true
  random_seed: 1234
  reset_attention_mask: false
  reset_position_ids: false
  seq_length: 4096
  sequence_parallel_size: 0
  sequences_per_dataset:
    ? /iopsstor/scratch/cscs/anunay/swissai/post-training/ampa/dataset/indexed/qwen3-30b-a3b-base-realistic-mix-v2/code/qwen3_30b_a3b_base_realistic_mix_code
    : - 836
      - 837
    ? /iopsstor/scratch/cscs/anunay/swissai/post-training/ampa/dataset/indexed/qwen3-30b-a3b-base-realistic-mix-v2/math/qwen3_30b_a3b_base_realistic_mix_math
    : - 1042
      - 1043
    ? /iopsstor/scratch/cscs/anunay/swissai/post-training/ampa/dataset/indexed/qwen3-30b-a3b-base-realistic-mix-v2/multilingual/qwen3_30b_a3b_base_realistic_mix_multilingual
    : - 1801
      - 1802
    ? /iopsstor/scratch/cscs/anunay/swissai/post-training/ampa/dataset/indexed/qwen3-30b-a3b-base-realistic-mix-v2/web/qwen3_30b_a3b_base_realistic_mix_web
    : - 10538
      - 10539
  sft_mock_dataset_config_json: null
  split: null
  split_matrix: null
  token_dtype_code: 4
  tokenizer:
    _call_: true
    _target_: megatron.core.tokenizers.text.text_tokenizer.MegatronTokenizerText
  trust_remote_code: null
  varlen_mock_dataset_config_json: null
  varlen_sbhd_validation: false
ddp:
  _target_: megatron.core.distributed.distributed_data_parallel_config.DistributedDataParallelConfig
  align_param_gather: false
  average_in_collective: true
  bucket_size: 40000000
  check_for_large_grads: false
  check_for_nan_in_grad: true
  data_parallel_sharding_strategy: optim_grads_params
  delay_wgrad_compute: false
  disable_symmetric_registration: false
  expert_data_parallel_sharding_strategy: optim_grads_params
  expert_outer_dp_sharding_strategy: no_shard
  fp4_param_gather: false
  fp8_param_gather: false
  fsdp_all_gather_in_start_param_sync: true
  fsdp_db_use_persist_buf_on_alloc_fail: false
  fsdp_double_buffer: false
  fsdp_manual_registration: false
  grad_reduce_in_fp32: true
  gradient_reduce_div_fusion: true
  hfsdp_param_gather_overlap: false
  keep_fp8_transpose_cache: false
  megatron_fsdp_cuda_graph_mode: false
  megatron_fsdp_enable_fine_grained_param_gather: false
  megatron_fsdp_grad_comm_dtype: null
  megatron_fsdp_main_grads_dtype: null
  megatron_fsdp_main_params_dtype:
    _call_: false
    _target_: torch.float32
  megatron_fsdp_max_pool_double_buffer: false
  megatron_fsdp_use_decoupled_grad: false
  megatron_fsdp_version: 1
  nccl_ub: false
  num_buckets: null
  num_distributed_optimizer_instances: 1
  outer_dp_sharding_strategy: no_shard
  overlap_grad_reduce: true
  overlap_param_gather: true
  pad_buckets_for_high_nccl_busbw: false
  param_name_patterns_for_fp32_local_accumulation: []
  reduce_scatter_with_fp32_accumulation: false
  reuse_grad_buf_for_mxfp8_param_ag: false
  suggested_communication_unit_size: null
  use_custom_fsdp: false
  use_distributed_optimizer: true
  use_megatron_fsdp: false
dist:
  _target_: megatron.bridge.training.config.DistributedInitConfig
  align_grad_reduce: true
  disable_jit_fuser: false
  distributed_backend: nccl
  distributed_timeout_minutes: 10
  distributed_timeout_seconds_after_init: null
  enable_megatron_core_experimental: false
  eval_context_parallel_size: null
  external_gpu_device_mapping: false
  flight_recorder_dump_on_timeout: true
  flight_recorder_dump_path: null
  flight_recorder_extra_dump_on_exec: true
  flight_recorder_include_only_active: true
  flight_recorder_include_stack_trace: false
  flight_recorder_trace_buffer_size: 2000
  high_priority_stream_groups: []
  lazy_mpu_init: false
  local_rank: 0
  nccl_communicator_config_path: null
  sharp_enabled_group: null
  use_decentralized_pg: false
  use_gloo_process_groups: true
  use_megatron_fsdp: false
  use_sharp: false
  use_torch_fsdp2: false
  use_tp_pp_dp_mapping: false
env_vars: {}
ft: null
inprocess_restart: null
logger:
  _target_: megatron.bridge.training.config.LoggerConfig
  barrier_with_L1_time: true
  comet_api_key: null
  comet_experiment_name: null
  comet_project: null
  comet_tags: null
  comet_workspace: null
  filter_warnings: true
  log_device_memory_used: false
  log_energy: false
  log_interval: 1
  log_l2_norm_grad_to_tensorboard: false
  log_loss_scale_to_tensorboard: true
  log_max_attention_logit: false
  log_memory_interval: null
  log_memory_to_tensorboard: false
  log_num_zeros_in_grad: false
  log_params_norm: false
  log_progress: false
  log_runtime_to_tensorboard: false
  log_throughput: false
  log_throughput_to_tensorboard: true
  log_timers_to_tensorboard: false
  log_validation_ppl_to_tensorboard: false
  log_world_size_to_tensorboard: false
  logging_level: null
  memory_keys: null
  mlflow_description: null
  mlflow_experiment: null
  mlflow_log_artifacts: true
  mlflow_run_name: null
  mlflow_tags: null
  mlflow_tracking_uri: null
  modules_to_filter: null
  moe_routing_trace_capture_hidden_states: false
  moe_routing_trace_capture_logits: false
  moe_routing_trace_dump_weights: false
  moe_routing_trace_max_training_iters: null
  moe_routing_trace_path: null
  runtime_time_unit: hours
  save_config_filepath: null
  set_level_for_all_loggers: false
  skip_train_metrics_log: false
  tensorboard_dir: /iopsstor/scratch/cscs/anunay/swissai/post-training/ampa/runs/uccl-debug5x5-reuse-20261104T1615Z-ordinary-control-ordinary-control/logs/tensorboard
  tensorboard_log_interval: 1
  tensorboard_queue_size: 1000
  throughput_window_size: 5
  timing_log_level: 0
  timing_log_option: minmax
  wandb_entity: anunay-yadav-epfl
  wandb_exp_name: uccl-debug5x5-reuse-20261104T1615Z-ordinary-control-ordinary-control
  wandb_project: moe-profile
  wandb_save_dir: /iopsstor/scratch/cscs/anunay/swissai/post-training/ampa/runs/uccl-debug5x5-reuse-20261104T1615Z-ordinary-control-ordinary-control/logs/wandb
mixed_precision:
  _target_: megatron.bridge.training.mixed_precision.MixedPrecisionConfig
  autocast_dtype: null
  autocast_enabled: false
  bf16: true
  first_last_layers_bf16: false
  fp16: false
  fp32: false
  fp4: null
  fp4_param: false
  fp4_param_gather: false
  fp4_quantizer_factory: null
  fp4_recipe: nvfp4
  fp4_recipe_attrs: null
  fp8: null
  fp8_amax_compute_algo: most_recent
  fp8_amax_history_len: 1
  fp8_dot_product_attention: false
  fp8_margin: 0
  fp8_multi_head_attention: false
  fp8_param: false
  fp8_param_gather: false
  fp8_quantizer_factory: null
  fp8_recipe: tensorwise
  fp8_recipe_attrs: null
  fp8_wgrad: true
  grad_reduce_in_fp32: true
  hysteresis: 2
  initial_loss_scale: 4294967296
  loss_scale: null
  loss_scale_window: 1000
  min_loss_scale: 1.0
  num_layers_at_end_in_bf16: 0
  num_layers_at_start_in_bf16: 0
  params_dtype:
    _call_: false
    _target_: torch.bfloat16
  pipeline_dtype:
    _call_: false
    _target_: torch.bfloat16
  reuse_grad_buf_for_mxfp8_param_ag: false
model:
  _target_: megatron.bridge.models.gpt_provider.GPTModelProvider
  account_for_embedding_in_pipeline_split: false
  account_for_loss_in_pipeline_split: false
  activation_func:
    _call_: false
    _target_: torch.nn.functional.silu
  activation_func_clamp_value: null
  activation_func_fp8_input_store: false
  activation_func_tanh_clamp_scale: null
  activation_func_tanh_clamp_scale_linear: null
  activation_offload_fraction: 1.0
  add_bias_linear: false
  add_qkv_bias: false
  apply_query_key_layer_scaling: false
  apply_residual_connection_post_layernorm: false
  apply_rope_fusion: false
  attention_backend:
    _args_:
    - 5
    _call_: true
    _name_: auto
    _target_: megatron.core.transformer.enums.AttnBackend
  attention_cp_layout: zigzag
  attention_dropout: 0.0
  attention_output_gate: false
  attention_softmax_in_fp32: false
  attn_logit_softcapping: null
  autocast_dtype:
    _call_: false
    _target_: torch.bfloat16
  barrier_with_L1_time: true
  batch_invariant_backend: te_native
  batch_invariant_collective: ordered
  batch_invariant_mode: false
  batch_p2p_comm: true
  batch_p2p_sync: true
  bf16: true
  bias_activation_fusion: false
  bias_dropout_fusion: false
  calculate_per_token_loss: false
  clone_scatter_output_in_embedding: true
  config_logger_dir: ''
  context_parallel_size: 1
  cp_comm_type: null
  cpu_offloading: false
  cpu_offloading_activations: true
  cpu_offloading_double_buffering: false
  cpu_offloading_num_layers: 0
  cpu_offloading_retain_pinned_cpu_buffers: false
  cpu_offloading_weights: false
  cross_entropy_fusion_impl: native
  cross_entropy_loss_fusion: true
  csa_compress_ratios: null
  csa_compress_rotary_base: 40000.0
  csa_dense_mode: false
  csa_window_size: 128
  cuda_graph_impl: none
  cuda_graph_modules: []
  cuda_graph_retain_backward_graph: false
  cuda_graph_scope: null
  cuda_graph_use_single_mempool: true
  cuda_graph_warmup_steps: 3
  deallocate_pipeline_outputs: true
  defer_embedding_wgrad_compute: false
  delay_offload_until_cuda_graph: false
  delay_wgrad_compute: false
  delta_offload_bytes_across_pp_ranks: 0
  dense_grouped_gemm: false
  deterministic_mode: false
  disable_bf16_reduced_precision_matmul: false
  disable_parameter_transpose_cache: false
  distribute_saved_activations: false
  dsa_indexer_head_dim: null
  dsa_indexer_k_norm_epsilon: null
  dsa_indexer_k_norm_fp32: false
  dsa_indexer_loss_coeff: null
  dsa_indexer_n_heads: null
  dsa_indexer_rope_interleaved: false
  dsa_indexer_rotate_activation: true
  dsa_indexer_scoring_relu: true
  dsa_indexer_skip_topk_offset: 0
  dsa_indexer_topk: null
  dsa_indexer_topk_freq: 1
  dsa_indexer_use_sparse_loss: false
  dsa_kernel_backend: none
  embedding_init_method:
    _args_: []
    _partial_: true
    _target_: torch.nn.init.normal_
    mean: 0.0
    std: 0.02
  embedding_init_method_std: 0.02
  enable_autocast: false
  enable_cuda_graph: false
  enable_mhc_connections: false
  ep_overlap_early_attn_memory_release: false
  experimental_attention_variant: null
  experimental_attention_variant_loss_scale_func: null
  expert_gtp_weight_remat_size: 1
  expert_model_parallel_size: 16
  expert_tensor_parallel_num_weight_shards: null
  expert_tensor_parallel_size: 1
  external_cuda_graph: false
  ffn_hidden_size: 6144
  finalize_model_grads_func:
    _args_: []
    _partial_: true
    _target_: megatron.core.distributed.finalize_model_grads.finalize_model_grads
    pg_collection:
      _call_: true
      _target_: megatron.core.process_groups_config.ProcessGroupCollection
  fine_grained_activation_offloading: false
  fine_grained_offloading_max_inflight_offloads: null
  first_last_layers_bf16: false
  flash_attention_version: null
  flash_decode: false
  fp16: false
  fp16_lm_cross_entropy: false
  fp32_residual_connection: false
  fp4: null
  fp4_param: false
  fp4_quantizer_factory: null
  fp4_recipe: nvfp4
  fp8: null
  fp8_amax_compute_algo: most_recent
  fp8_amax_history_len: 1
  fp8_dot_product_attention: false
  fp8_interval: 1
  fp8_margin: 0
  fp8_multi_head_attention: false
  fp8_output_proj: false
  fp8_param: false
  fp8_quantizer_factory: null
  fp8_recipe: tensorwise
  fp8_wgrad: true
  freeze_base_model_for_mtp: false
  fused_residual_rmsnorm: false
  fused_single_qkv_rope: false
  gated_linear_unit: true
  gdp_cutedsl_kernel: false
  gdp_num_chunk_states_to_recompute: 2
  gdp_num_householder: 3
  glu_linear_offset: 0.0
  grad_scale_func:
    _call_: false
    _target_: megatron.core.optimizer.optimizer.MegatronOptimizer.scale_loss
  grad_sync_func:
    _call_: false
    _target_: megatron.core.distributed.distributed_data_parallel.DistributedDataParallel.start_grad_sync
  gradient_accumulation_fusion: true
  gtp_remat_opt_in_modules: []
  gtp_weight_remat_size: 1
  hetereogenous_dist_checkpoint: false
  heterogeneous_block_specs: false
  hf_model_id: Qwen/Qwen3-30B-A3B-Base
  hf_model_revision: 1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9
  hidden_dropout: 0.0
  hidden_size: 2048
  hierarchical_context_parallel_sizes: null
  high_priority_a2a_comm_stream: false
  hybrid_context_parallel: false
  inference_cuda_graph_scope:
    _args_:
    - 1
    _call_: true
    _name_: none
    _target_: megatron.core.transformer.enums.InferenceCudaGraphScope
  inference_disable_triton_nvls_kernels: false
  inference_flashinfer_mxfp8_token_capacity: null
  inference_fuse_tp_communication: false
  inference_grouped_gemm_backend: vllm
  inference_moe_disable_fused_quant_kernels: false
  inference_moe_token_dispatcher_type: nvls
  inference_rng_tracker: false
  inference_sampling_seed: 42
  init_method:
    _args_: []
    _partial_: true
    _target_: torch.nn.init.normal_
    mean: 0.0
    std: 0.02
  init_method_std: 0.02
  init_model_with_meta_device: false
  is_hybrid_model: false
  keep_mtp_in_bf16: false
  kitchen_attention_backend: sdpa
  kv_channels: 128
  layernorm_epsilon: 1.0e-06
  layernorm_zero_centered_gamma: false
  linear_attention_freq: null
  linear_conv_kernel_dim: 4
  linear_cp_layout: zigzag
  linear_cp_mode: headwise
  linear_key_head_dim: 128
  linear_num_key_heads: 16
  linear_num_value_heads: 32
  linear_value_head_dim: 128
  log_max_attention_logit: false
  logit_dtype: null
  make_vocab_size_divisible_by: 128
  mamba_head_dim: 64
  mamba_num_groups: 8
  mamba_num_heads: null
  mamba_state_dim: 128
  mamba_training_ssm_states_dtype: null
  masked_softmax_fusion: true
  max_seqlen_per_dp_cp_rank: null
  memory_efficient_layer_norm: false
  mhc_fused_backend: auto
  mhc_init_gating_factor: 0.01
  mhc_num_residual_streams: 4
  mhc_recompute_layer_num: null
  mhc_sinkhorn_iterations: 20
  microbatch_group_size_per_vp_stage: 1
  min_offloaded_tensor_size: 1048576
  mlp_chunks_for_prefill: 1
  mlp_chunks_for_training: 1
  moe_apply_probs_on_input: false
  moe_aux_loss_coeff: 0.001
  moe_combine_bwd_dtype: bf16
  moe_deepep_num_sms: null
  moe_dispatch_fwd_dtype: bf16
  moe_enable_deepep: false
  moe_enable_routing_replay: false
  moe_expert_capacity_factor: null
  moe_expert_rank_capacity_factor: null
  moe_ffn_hidden_size: 768
  moe_flex_dispatcher_backend: null
  moe_flex_dispatcher_num_sms: null
  moe_grad_scale_func: null
  moe_grouped_gemm: true
  moe_hybridep_num_blocks_permute: null
  moe_hybridep_num_blocks_unpermute: null
  moe_hybridep_num_sms: null
  moe_hybridep_num_sms_preprocessing: 108
  moe_hybridep_pad_uneven_dispatch_inputs: false
  moe_input_jitter_eps: null
  moe_latent_size: null
  moe_layer_freq: 1
  moe_layer_recompute: false
  moe_mlp_glu_interleave_size: null
  moe_ncclep_zero_copy: false
  moe_pad_expert_input_to_capacity: false
  moe_pad_experts_for_cuda_graph_inference: false
  moe_paged_stash: false
  moe_paged_stash_buffer_size_factor_cpu: 0.0
  moe_paged_stash_buffer_size_factor_cuda: 1.1
  moe_paged_stash_page_size: 64
  moe_per_layer_logging: false
  moe_permute_fusion: true
  moe_permute_fusion_into_hybridep: false
  moe_router_aux_loss_fusion: false
  moe_router_bias_update_rate: 0.001
  moe_router_dtype: null
  moe_router_enable_expert_bias: false
  moe_router_force_biased: null
  moe_router_force_load_balancing: false
  moe_router_fusion: false
  moe_router_group_topk: null
  moe_router_load_balancing_type: aux_loss
  moe_router_num_groups: null
  moe_router_padding_for_fp8: false
  moe_router_padding_for_quantization: false
  moe_router_pre_softmax: false
  moe_router_quantile_balancing_ema: 0.0
  moe_router_score_function: softmax
  moe_router_skip_muon: false
  moe_router_topk: 8
  moe_router_topk_limited_devices: null
  moe_router_topk_scaling_factor: null
  moe_shared_expert_gate: false
  moe_shared_expert_glu_interleave_size: null
  moe_shared_expert_intermediate_size: null
  moe_shared_expert_overlap: false
  moe_shortcut_connection: false
  moe_shortcut_parallel: false
  moe_shortcut_post_norm: false
  moe_single_grouped_bias: false
  moe_single_grouped_weight: false
  moe_token_dispatcher_type: alltoall
  moe_token_drop_policy: probs
  moe_token_dropping: false
  moe_use_grouped_tensor: false
  moe_use_norm_before_up_proj: false
  moe_z_loss_coeff: null
  mrope_section: null
  mtp_detach_heads: false
  mtp_enabled: false
  mtp_grad_scale_func: null
  mtp_hsm: false
  mtp_hybrid_override_pattern: null
  mtp_layer_spec_transform: null
  mtp_loss_scaling_factor: 0.1
  mtp_num_layers: null
  mtp_standalone: false
  mtp_use_repeated_layer: false
  multi_latent_attention: false
  mup_attn_scale_power: 1.0
  mup_base_head_dim: null
  mup_base_hidden_size: null
  mup_embedding_mult: 1.0
  mup_output_mult: 1.0
  mup_width_mult: 1.0
  nccl_all_reduce_for_prefill: false
  no_rope_freq: null
  no_sync_func:
    _call_: false
    _target_: megatron.core.distributed.distributed_data_parallel.DistributedDataParallel.no_sync
  normalization: RMSNorm
  num_attention_heads: 32
  num_layers: 48
  num_layers_at_end_in_bf16: 0
  num_layers_at_start_in_bf16: 0
  num_layers_in_first_pipeline_stage: null
  num_layers_in_last_pipeline_stage: null
  num_microbatches_with_partial_activation_checkpoints: null
  num_moe_experts: 128
  num_query_groups: 4
  offload_modules: []
  output_layer_init_method:
    _args_: []
    _partial_: true
    _target_: torch.nn.init.normal_
    mean: 0.0
    std: 0.0020412414523193153
  overlap_dispatch_backward_with_experts_wgrad: false
  overlap_moe_expert_parallel_comm: false
  overlap_p2p_comm: false
  overlap_p2p_comm_warmup_flush: false
  parallel_output: true
  param_sync_func: null
  params_dtype:
    _call_: false
    _target_: torch.bfloat16
  perform_initialization: true
  persist_layer_norm: false
  pipeline_dtype:
    _call_: false
    _target_: torch.bfloat16
  pipeline_model_parallel_comm_backend: null
  pipeline_model_parallel_layout: null
  pipeline_model_parallel_size: 1
  position_embedding_type: rope
  qk_clip: false
  qk_clip_alpha: 0.5
  qk_clip_threshold: 100
  qk_l2_norm: false
  qk_layernorm: true
  quant_recipe: null
  recompute_granularity: null
  recompute_method: null
  recompute_modules:
  - core_attn
  recompute_num_layers: null
  restore_modelopt_state: false
  rope_scaling: false
  rope_scaling_factor: 1.0
  rotary_base: 1000000.0
  rotary_interleaved: false
  rotary_percent: 1.0
  rotary_scaling_factor: null
  scatter_embedding_sequence_parallel: true
  seq_len_interpolation_factor: null
  seq_length: 4096
  sequence_packing_scheduler: null
  sequence_parallel: false
  share_embeddings_and_output_weights: false
  should_pad_vocab: true
  softmax_scale: null
  softmax_type: vanilla
  symmetric_ar_type: null
  tensor_model_parallel_size: 1
  tensor_parallel_num_weight_shards: null
  test_mode: false
  timers:
    _call_: true
    _target_: megatron.core.timers.Timers
  tp_comm_atomic_ag: false
  tp_comm_atomic_rs: false
  tp_comm_bootstrap_backend: nccl
  tp_comm_bulk_dgrad: true
  tp_comm_bulk_wgrad: true
  tp_comm_overlap: false
  tp_comm_overlap_ag: true
  tp_comm_overlap_cfg: null
  tp_comm_overlap_disable_fc1: false
  tp_comm_overlap_disable_qkv: false
  tp_comm_overlap_rs: true
  tp_comm_overlap_rs_dgrad: false
  tp_comm_split_ag: true
  tp_comm_split_rs: true
  tp_only_amax_red: false
  transformer_impl: transformer_engine
  transformer_layer_spec:
    _call_: false
    _target_: megatron.bridge.models.gpt_provider.default_layer_spec
  use_arbitrary_attention_mask: null
  use_cpu_initialization: false
  use_fused_mhc: false
  use_fused_weighted_squared_relu: false
  use_grouped_gemm_for_dense_mlp: false
  use_grouped_gemm_for_shared_expert: false
  use_inference_optimized_layers: false
  use_kitchen: false
  use_kitchen_attention: false
  use_mamba_mem_eff_path: true
  use_mup: false
  use_ring_exchange_p2p: false
  use_te_activation_func: false
  use_te_rng_tracker: false
  use_transformer_engine_full_layer_spec: false
  use_transformer_engine_op_fuser: false
  variable_seq_lengths: false
  virtual_pipeline_model_parallel_size: null
  vocab_size: 151669
  wgrad_deferral_limit: 0
  window_attn_skip_freq: null
  window_size: null
  yarn_beta_fast: null
  yarn_beta_slow: null
  yarn_correction_range_round_to_int: null
  yarn_mscale: null
  yarn_mscale_all_dim: null
  yarn_original_max_position_embeddings: null
  yarn_rotary_scaling_factor: null
nvrx_straggler: null
optimizer:
  _target_: megatron.bridge.training.config.OptimizerConfig
  adam_beta1: 0.9
  adam_beta2: 0.95
  adam_eps: 1.0e-08
  adaptive_muon_beta2: 0.95
  adaptive_muon_eps: 1.0e-08
  adaptive_muon_moment2_method: adamuon
  apply_wd_to_qk_layernorm: false
  barrier_with_L1_time: false
  bf16: true
  clip_grad: 1.0
  config_logger_dir: ''
  decoupled_lr: null
  decoupled_min_lr: null
  decoupled_weight_decay: true
  exp_avg_dtype:
    _call_: false
    _target_: torch.float32
  exp_avg_sq_dtype:
    _call_: false
    _target_: torch.float32
  fp16: false
  fp8_recipe: tensorwise
  grad_norm_skip_threshold: .inf
  hysteresis: 2
  initial_loss_scale: 4294967296
  lion_beta1: 0.95
  lion_beta2: 0.98
  log_num_zeros_in_grad: false
  loss_scale: null
  loss_scale_window: 1000
  lr: 1.0e-06
  main_grads_dtype:
    _call_: false
    _target_: torch.float32
  main_params_dtype:
    _call_: false
    _target_: torch.float32
  min_loss_scale: 1.0
  min_lr: 1.0e-06
  muon_coefficient_type: quintic
  muon_extra_scale_factor: 1.0
  muon_fp32_matmul_prec: medium
  muon_momentum: 0.95
  muon_nesterov: false
  muon_num_ns_steps: 5
  muon_scalar_optimizer: adam
  muon_scale_mode: spectral
  muon_split_qkv: true
  muon_tp_mode: duplicated
  muon_use_syrk: false
  optimizer: adam
  optimizer_cpu_offload: false
  optimizer_cuda_graph: false
  optimizer_offload_fraction: 0.0
  overlap_cpu_optimizer_d2h_h2d: false
  overlap_param_gather: true
  overlap_param_gather_with_optimizer_step: false
  params_dtype:
    _call_: false
    _target_: torch.bfloat16
  pin_cpu_grads: true
  pin_cpu_params: true
  reuse_grad_buf_for_mxfp8_param_ag: false
  sgd_momentum: 0.9
  soap_shampoo_beta: 0.95
  soap_use_kl_shampoo: true
  store_param_remainders: true
  timers:
    _call_: true
    _target_: megatron.core.timers.Timers
  use_distributed_optimizer: true
  use_layer_wise_distributed_optimizer: false
  use_precision_aware_optimizer: false
  use_torch_optimizer_for_cpu_offload: false
  weight_decay: 0.1
optimizer_config_override_provider:
  _target_: megatron.bridge.training.config.OptimizerConfigOverrideProvider
peft: null
profiling:
  _target_: megatron.bridge.training.config.ProfilingConfig
  memory_snapshot_path: /nemo_run/snapshot.pickle
  nvtx_ranges: false
  profile_ranks:
  - 0
  profile_step_end: 12
  profile_step_start: 10
  pytorch_profiler_collect_callstack: false
  pytorch_profiler_collect_chakra: false
  pytorch_profiler_collect_shapes: false
  record_memory_history: false
  record_shapes: false
  use_nsys_profiler: false
  use_pytorch_profiler: false
rerun_state_machine:
  _target_: megatron.bridge.training.config.RerunStateMachineConfig
  check_for_nan_in_loss: true
  check_for_spiky_loss: false
  error_injection_rate: 0
  error_injection_type: transient_error
  rerun_mode: disabled
  spiky_loss_factor: 10.0
rng:
  _target_: megatron.training.config.common_config.RNGConfig
  data_parallel_random_init: false
  inference_rng_tracker: false
  seed: 1234
  te_rng_tracker: false
scheduler:
  _target_: megatron.bridge.training.config.SchedulerConfig
  end_weight_decay: 0.033
  lr_decay_iters: 25
  lr_decay_samples: null
  lr_decay_steps: 1600
  lr_decay_style: constant
  lr_warmup_fraction: null
  lr_warmup_init: 0.0
  lr_warmup_iters: 0
  lr_warmup_samples: 0
  lr_warmup_steps: 0
  lr_wsd_decay_iters: null
  lr_wsd_decay_samples: null
  lr_wsd_decay_style: minus_sqrt
  max_steps: null
  no_weight_decay_cond_type: null
  override_opt_param_scheduler: true
  start_weight_decay: 0.033
  use_checkpoint_opt_param_scheduler: false
  wd_incr_steps: 1600
  weight_decay_incr_style: constant
  wsd_decay_steps: null
straggler: null
tensor_inspect: null
tokenizer:
  _target_: megatron.bridge.training.tokenizers.config.TokenizerConfig
  chat_template: null
  chat_template_path: null
  force_system_message: false
  hf_tokenizer_kwargs:
    include_special_tokens: true
    trust_remote_code: false
    use_fast: true
  image_tag_type: null
  make_vocab_size_divisible_by: 1
  merge_file: null
  metadata_path: null
  null_tokenizer_eod_id: null
  null_tokenizer_pad_id: -1
  pad_vocab_size: false
  padded_vocab_size: null
  rank: 0
  special_tokens: null
  tensor_model_parallel_size: 1
  tiktoken_num_special_tokens: 1000
  tiktoken_pattern: null
  tokenizer_hf_no_include_special_tokens: false
  tokenizer_hf_no_use_fast: false
  tokenizer_model: /iopsstor/scratch/cscs/anunay/swissai/post-training/ampa/checkpoint/hf/qwen3-30b-a3b-base/1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9
  tokenizer_prompt_format: null
  tokenizer_sentencepiece_ignore_extra_whitespaces: true
  tokenizer_sentencepiece_legacy: false
  tokenizer_type: HuggingFaceTokenizer
  trust_remote_code: false
  use_gigatoken: false
  use_tokenizer_vocab_size: true
  vocab_extra_ids: 0
  vocab_file: null
  vocab_size: null
train:
  _target_: megatron.bridge.training.config.TrainingConfig
  check_optimizer_step_success: true
  check_weight_hash_across_dp_replicas_interval: null
  decrease_batch_size_if_needed: false
  empty_unused_memory_level: 0
  eval_interval: null
  eval_iters: null
  exit_duration_in_mins: null
  exit_interval: 10
  exit_signal:
    _args_:
    - 15
    _call_: true
    _name_: SIGTERM
    _target_: signal.Signals
  exit_signal_handler: false
  exit_signal_handler_for_dataloader: false
  exit_signal_handler_for_training: false
  global_batch_size: 64
  gpu_sniff_test_interval: null
  iterations_to_skip: []
  manual_gc: true
  manual_gc_eval: 100
  manual_gc_interval: 100
  micro_batch_size: 1
  num_epochs: null
  rampup_batch_size: null
  skip_sync_grad_norm_across_mp: false
  skip_train: null
  step_batch_size_schedule: null
  train_iters: 25
  train_samples: null
  train_sync_interval: null
validation:
  _target_: megatron.training.config.training_config.ValidationConfig
  eval_global_batch_size: 64
  eval_interval: 0
  eval_iters: 0
  eval_micro_batch_size: 1
  full_validation: false
  multiple_validation_sets: false
  skip_train: false
  start_eval_at_iter: null
  test_mode: false
  validation_set_names: null
```

</details>

[acquisitions]: ../runs/uccl-debug5x5-reordered-20261104T2221Z-reordered-coordinator/acquisitions.tsv
[acquisition-notes]: ../ai/qwen-debug-reordered-resume/iteration-01/ACQUISITIONS.md
[campaign]: ../runs/qwen-debug-results-20261105T0925Z-cpu-validation/campaign_acceptance.json
[case-proofs]: ../runs/qwen-debug-results-20261105T0925Z-cpu-validation/cases/
[report]: ../runs/qwen-debug-results-20261105T0936Z-report/report.json
[numeric-audit]: ../runs/qwen-debug-results-20261105T0936Z-report/independent-audit.json
[first-proof]: ../runs/uccl-debug5x5-20261104T1415Z-grouped-profile/reviewed_inputs.json
[source-manifest]: ../runs/uccl-debug5x5-20261104T1415Z-source/expected-source.sha256
[dataset-manifest]: ../dataset/manifests/ep16-cache-20261011T1757Z-cp1-ep16.json
[uccl-source]: ../runs/uccl-ep-owned-build-20261101T1713Z/source_identity.json
[ownership-patch]: ../runs/uccl-ep-owned-build-20261101T1713Z/source/uccl-borrowed-scratch.patch
[decisions]: ../progress/DECISIONS.md
[timeline]: ../progress/sessions/2026-11-01-uccl-ep-comparison/timeline.md
[warmup-analysis]: ../ai/qwen-debug-reordered-resume/iteration-01/TIMING.md
[five-v-twenty]: ../ai/qwen-five-step-statistics/iteration-01/STATUS.md
[config-evidence]: ../ai/qwen-experiment-audit/iteration-01/CONFIG_DATA_EVIDENCE.md
[checks-evidence]: ../ai/qwen-experiment-audit/iteration-01/CHECKS_EVIDENCE.md
[history-evidence]: ../ai/qwen-experiment-audit/iteration-01/HISTORY_RELIABILITY_EVIDENCE.md
[audit-snapshot]: ../ai/qwen-experiment-audit/iteration-01/audit_snapshot.json
[audit-extractor]: ../ai/qwen-experiment-audit/iteration-01/build_audit_snapshot.py
[deck-verification]: ../runs/qwen-debug-results-20261105T0936Z-slides-v2/FINAL_VERIFICATION.json
[deck-review]: ../ai/qwen-debug-final-review/iteration-01/REVIEW.md
[sm-evidence]: ../ai/qwen-sm-communication/iteration-01/STATUS.md
[library-map]: ../runs/qwen-debug-results-20261105T0905Z-local-work/container-libraries/library-map.json
[posthoc-review]: ../ai/qwen-debug-final-review/iteration-01/STATUS.md:35
