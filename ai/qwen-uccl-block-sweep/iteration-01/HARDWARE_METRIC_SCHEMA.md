# Hardware metric identity, units, and coverage

Final result: COMPLETE. Metric stream identity is recoverable from the documented `typeId` bit fields. The tiny measured region has complete 1 kHz sample bundles on all four GPUs. Direct percentage fields support sampled device-level summaries. Clock labels conflict with raw scales, and raw warp averages lack explicit scaling metadata. Do not convert those fields into SM counts or warps per SM.

State before execution: bounded read-only CPU inspection pending. Inspect the matched tiny capability database at `runs/uccl-nsys132-retry-20261109T152622Z-cap/captures/gpu_metrics.sqlite`. Determine whether exported metadata binds metric streams to physical GPU identities and gives explicit value scales. Inspect sample windows and gaps before designing measured-step summaries.

The existing authorized `.runtime/venvs/qwen-posthoc-cpu/bin/python` will open SQLite with `mode=ro`. No GPU, training, profiler export, jobs, frozen-source edits, or acquisition edits will run. The root owns task status and training monitoring. This file records the bounded schema findings and limits only.

## Provenance

The CPU programs ran on `clariden-ln004`, using Python 3.12.12 from the authorized virtual environment. They opened the 48,689,152-byte database read-only. The SQLite SHA-256 stayed `3928bfe9c0a3a5366f293ce673e5167772d44994c4e3d88f585e5b7a0ed188eb` before and after all queries. No source or acquisition file changed.

```bash
PYTHONDONTWRITEBYTECODE=1 \
  .runtime/venvs/qwen-posthoc-cpu/bin/python - <<'PY'
# sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)
# Schema, metadata, sample-key, device mapping, and bounded ROI/gap queries.
PY
```

The [capability proof](../../../runs/uccl-nsys132-retry-20261109T152622Z-cap/capabilities.json) records job 3901239 on `nid007051` and no training. Export metadata identifies Nsight Systems 2025.6.3.343 and SQLite schema 3.24.8. The bound native metric report has SHA-256 `397020774c21c43fe5739a7c3defb6529bb95e03a886f5f51470034984ab7182`.

## Metric stream to physical GPU

The actual database has `GPU_METRICS(rawTimestamp,timestamp,typeId,metricId,value)`. Names require the compound key `(typeId,metricId)` in `TARGET_INFO_GPU_METRICS`. Metric IDs repeat across devices. The same installed Nsight recipe performs that compound-key merge in [table_config.py:979](../../../runs/uccl-nsys132-retry-20261109T152622Z/tool/nsight_systems-linux-sbsa-2025.6.3.343-archive/target-linux-sbsa-armv8/python/packages/nsys_recipe/lib/table_config.py:979).

All four types share `sourceId=281479271677952` and an empty type name. `GENERIC_EVENT_SOURCES` resolves its name to `GpuMetrics` and its time source to `Session Nanoseconds`. These source/type names cannot distinguish GPUs. The required mapping uses the packed type identity instead.

The same installed [data_utils.py:470](../../../runs/uccl-nsys132-retry-20261109T152622Z/tool/nsight_systems-linux-sbsa-2025.6.3.343-archive/target-linux-sbsa-armv8/python/packages/nsys_recipe/lib/data_utils.py:470) documents this layout:

```text
<Hardware ID:8><VM ID:8><Source ID:16><Event tag:24><GPU ID:8>
gpuId = typeId & 0xFF
globalVm = typeId & 0xFFFF000000000000
```

The GPU-ID extraction is NVIDIA's implementation. The VM mask follows the documented field layout and matches the actual target records. Join `(globalVm,gpuId)` to `(TARGET_INFO_GPU.vmId,TARGET_INFO_GPU.id)`. Require one physical UUID. NVIDIA's [Analysis Guide](https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html#sqlite-schema) also documents the composite type ID and lower-eight-bit GPU identity. Versioned 2025.6 web pages were unavailable through the browser, so the matching installed source is the version-specific evidence.

| `typeId` | GPU ID | Physical UUID |
| --- | ---: | --- |
| 281479271677952 | 0 | `91808c98-e25f-3a24-3cf7-035f7fe28140` |
| 281479271677953 | 1 | `37b40a63-3710-e0f8-b50a-6e943de50d27` |
| 281479271677954 | 2 | `879f4779-7c73-cdad-c7a0-21c4500ead84` |
| 281479271677955 | 3 | `85179c67-b5bf-b05c-d02d-447e50f52b34` |

All four target records use `vmId=281474976710656`. Each reports 132 SMs and a capacity of 64 warps per SM. The tiny application's kernel owner decodes to OS PID 205740. Its process-specific CUDA-device row maps CUDA device 0 to physical GPU 0. The other GPUs were sampled but did not execute this tiny application's kernels. Their zero activity is consistent with that evidence. Do not assume CUDA activity `deviceId` directly equals physical GPU ID in a general training process.

There are eight generic type records, including four extra types with event tag offset `0x300`. Only the four types above have rows in `GPU_METRICS` and the 32-field metric definitions. The extra types have no exported sample or generic-event records here. Their purpose is not established by this review. Do not interpret them as additional GPUs or fabricate values from their presence.

## Value units and scale limits

`GENERIC_EVENT_TYPE_FIELDS` resolves all 32 field names. It records storage offsets, sizes, and unsigned integer type. `hyperFormat` and `hyperComment` are NULL. No multiplier, divisor, scale, averaging population, or physical-unit conversion field is present. `TARGET_INFO_GPU_METRICS` adds names only. Name suffixes therefore need semantic review rather than automatic conversion.

| Field | Actual storage/example on GPU 0 | Safe interpretation |
| --- | --- | --- |
| `SMs Active [Throughput %]` | Unsigned byte, raw 0–74 in tiny ROI | Sampled percentage of SM-cycle activity, direct 0–100 scale |
| `SM Issue [Throughput %]` | Unsigned byte, raw 0–12 | Sampled instruction-issue throughput percentage, not resident warp count |
| `Compute Warps in Flight [Throughput %]` | Unsigned byte, raw 0–15 | Sampled normalized compute-warp activity percentage |
| `Compute Warps in Flight [Avg]` | Unsigned 4-byte integer, raw up to 35,435,516 | Raw field only. Conversion to resident warp count unavailable |
| `Compute Warps in Flight [Avg Warps per Cycle]` | Unsigned byte, raw 0–19 | Preserve exact field name/raw values. No verified per-SM denominator |
| `GPC Clock Frequency [MHz]` | Unsigned 4-byte integer, raw 1,843,708,001–2,055,066,887 | Label/scale conflict. Raw values must not be reported as MHz |
| `SYS Clock Frequency [MHz]` | Unsigned 4-byte integer, roughly 1.64–1.66 billion | Same scale conflict |

The first three percentage fields have a one-percentage-point sample encoding. NVIDIA's [User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#gpu-metrics) describes SM activity, instruction issue, and metric sampling. Its [earlier detailed metric definitions](https://docs.nvidia.com/nsight-systems/2023.2/pdf/UserGuide.pdf) define compute-warp percentage relative to available resident warp capacity. These remain device-level sampled ratios. A 50% SM-active value does not uniquely identify half the SMs: it can also arise from all SMs active for half an interval.

NVIDIA's [CUPTI metric reference](https://docs.nvidia.com/cupti/13.2.1/main/main.html) describes the GPC/SYS frequency metrics in hertz. The raw values and `TARGET_INFO_GPU.clockRate=1980000000` are consistent with hertz. A conversion to MHz would divide hertz by one million. This export does not serialize that conversion despite its MHz label. Record the conflict and preserve raw values. If a later analysis uses that conversion, bind the metric definition and state its unit provenance explicitly. Automatic suffix-based scaling is unsafe.

Do not divide `[Avg]` by 132 or 64, or reinterpret `[Avg Warps per Cycle]` as warps per SM. The unexported reduction and averaging semantics matter. Do not derive communication-only occupancy, a count of simultaneously active SMs, eligible warps, or issue-stall reasons from the percentage fields. Those quantities are not directly recorded here.

## Actual measured-region coverage

The only marker is `AMPA_NSYS_CAPABILITY_TINY_GPU`, from 55,578,400 to 3,055,756,256 ns. It is a tiny GEMM/ReLU workload, not a Qwen update. The metric source explicitly uses session nanoseconds, and all raw metric timestamps equal their converted session timestamps in this export. No cross-host clock alignment is needed for this one-report query.

The database contains 463,616 metric-value rows, representing 14,488 complete 32-metric bundles. Every `(typeId,timestamp)` bundle has all 32 distinct metrics exactly once. There are no duplicate `(typeId,metricId,timestamp)` keys or NULL core metric values. Each GPU has exactly 3,000 bundles whose timestamps fall inside the tiny marker.

| GPU | All bundles | Full stream first/last timestamp, ns | ROI first/end boundary distance, ns | ROI spacing min/median/max, ns |
| --- | ---: | --- | --- | --- |
| 0 | 3489 | −403569456 / 3084430512 | 852144 / 325712 | 999967 / 1000000 / 1000033 |
| 1 | 3556 | −262187184 / 3292812832 | 234416 / 943440 | 999999 / 1000000 / 1000001 |
| 2 | 3669 | −142047861 / 3525952144 | 373740 / 804128 | 999999 / 1000000 / 1000001 |
| 3 | 3774 | 897424 / 3773897372 | 319024 / 858831 | 999967 / 1000000 / 1000033 |

There are zero internal ROI gaps above 2 ms. The requested sampling rate is 1,000 Hz. The observed spacing matches that request. Sample boundaries differ by device, and whole-collection windows include startup and post-work intervals. A whole-table mean therefore does not describe only the marker.

The diagnostic counter says 14,492 GPU metric events collected. That exceeds the 14,488 exported sample bundles by four. Extra generic type definitions exist, but no exported rows assign those four events a meaning. Do not treat the difference as four lost samples, four missing GPU events, or proof of loss-free export. The measured timestamp/bundle checks are the available bounded evidence.

The export has generic possible-loss advisories for the application's CUDA/NVTX trace. It has no explicit GPU-metric overflow or missing-data diagnostic. No generic-event table exports explicit gap ranges here. Absence of that table is not proof that every possible hardware counter event is present. NVIDIA documents sampling-buffer and inconsistent/missing-data limits in its [sampling guidance](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#gpu-metrics).

## Proposed gates for later training analysis

Use already accepted per-node timelines and source-bound journals first. This schema review alone does not accept a model run. For each rank and measured step, require the physical UUID chain from metric type to target GPU to actual runtime placement. Require the existing same-process CUDA-device mapping separately for trace activity. Match a unique `AMPA/STEP/5..9` range on that rank and use its report-local session timestamps.

Join names by `(typeId,metricId)`, then select the three reviewed percentage metrics by exact name. Require finite 0–100 values and exactly one value per sample key. Preserve their integer source precision. Leave raw warp averages and frequency fields out of quantitative summaries until their scales are explicitly approved. Never replace missing values with zero or silently renormalize a shortened window.

Report the requested rate, actual sample count, first/last sample offsets, minimum/median/p99/maximum spacing, duplicate/null counts, and incomplete bundles for each device and step. A proposed conservative gate is at least two samples, first/last boundary distance at most two nominal periods, and no internal spacing above two periods. At 1 kHz those limits are 2 ms. This is a declared analysis rule, not a NVIDIA guarantee or an established physical loss test. If a step fails, report its metric summary unavailable and retain the coverage diagnostics.

Treat timestamps as sample points inside the closed step range, as the matching Nsight [range-mapping helper](../../../runs/uccl-nsys132-retry-20261109T152622Z/tool/nsight_systems-linux-sbsa-2025.6.3.343-archive/target-linux-sbsa-armv8/python/packages/nsys_recipe/lib/gpu_metrics.py:165) does. Start with count/min/median/p95/max and an explicitly named arithmetic sample mean. Irregular spacing can bias that mean. A time-weighted estimate needs a declared interval convention and boundary/gap policy. Do not invent integration intervals from the raw timestamp alone.

Label results as device activity sampled during the optimizer-step window. A narrower window selected by a UCCL payload kernel still samples all concurrent device work. It cannot attribute active SMs or resident warps exclusively to UCCL. Sub-millisecond individual kernels can have zero samples at this rate. Report those cases as unresolved rather than interpolating per-kernel occupancy or exposed latency.

## Matching source hashes and handoff

The three inspected vendor helper files belong to the same extracted Nsight 2025.6.3.343 package. Their hashes bind the source-based interpretation:

| Helper under `target-linux-sbsa-armv8/python/packages/nsys_recipe/lib/` | SHA-256 |
| --- | --- |
| `data_utils.py` | `c88abb9bdbbaebfcca17f67205c8b020d4e7b93c708c39765d0441534ad4cd13` |
| `table_config.py` | `efc9468f327b417d597e9fb6eda3f907a0ecec81aa3771cc8bca4a0f38f8aa2b` |
| `gpu_metrics.py` | `33b6fbd8b50b6733a85041cebd7180b3ddecf125a430af8432c3bc66d812cf4e` |

Physical metric-stream mapping is available. The three normalized percentage fields are suitable for a separately reviewed per-device/per-step extraction with coverage gates. Raw average-warp scaling is unavailable. Clock export labels require an explicit scale correction and provenance before use. No training metric result or communication-only occupancy follows from this tiny capability inspection.
