# Unprofiled UCCL block-budget sweep

Goal: Compare real optimizer-step time with profiling disabled at payload budgets12,20,32,40. Extend beyond a winning boundary until performance becomes repeatably worse or a supported hardware/runtime limit is reached.

State at 2026-11-11 04:20 UTC: COMPLETE and independently accepted. Twenty full paired allocations,40independent C0 starts,400optimizer timers and200measured updates cover4,8,12,20,32,40,48,56blocks. Every pair preserves the original640sample prefix and actual16GPU placement within the pair. Forty has the lowest observed median ratio0.97849905, with2.00–2.30%shorter steps in two reversed-order pairs. Four is consistently slower,2.43–6.16%. Fifty-six varies from7.71%shorter to2.87%longer; the upper decline is unresolved at its resource limit. One failed full attempt3912016 remains preserved and excluded; the fresh replacement passed. Final QA3913091 passed7tests/closure/syntax/imports in18s. Synthesis3913094 completed0:0 in112s, rechecked all raw evidence and decisions, and generated the metrics/tables/chart. Independent review PASS. [Final audit](../../../docs/QWEN_UCCL_UNPROFILED_BUDGET_AUDIT.md), [metrics](../../../runs/ep-overlap-budget-summary-final-20261111T041250Z/metrics.json), [acceptance](../../../runs/ep-overlap-budget-summary-final-20261111T041250Z/FINAL_ACCEPTANCE.json), [review](../../ep-overlap-budget-summary-review/iteration-01/FINAL_REVIEW.md). No compute remains queued or running.

Decisions: Keep main EP overlap, model and actual DDP delayed weight gradients, early attention-memory release, CUDA connections32 and all previous workload/input/precision choices. Use ordinary top8 UCCL, BF16, EP16/CP1/TP1/PP1/ETP1,4096/MBS1/GBS64/GA4, original C0 and deterministic640-sample prefix,5warmup+5measured updates for every accepted full case. Disable Kineto, Nsight and trace wrappers. Preserve common health, routing, receive and fingerprint observers and state their cost.

Reviewed comparison design: One candidate and a20-block reference start independently from C0 on the same four nodes and16actual GPUs within one21minute debug allocation. Initial six full pairs are12/reference-first,32/candidate-first,40/reference-first,12/candidate-first,32/reference-first,40/candidate-first. Each case completes5+5. Compare allocation-level candidate/reference step-mean ratios. Add a third pair for order disagreement, more than2% ratio spread, greater than2% within-case CV, or a near-baseline result. A2% tolerance is operational, not a significance claim. An outer point is repeatably worse only when min(outer ratios)>1.02*max(best-tested ratios). Extend competitive lower edges12→8→4 and upper40→48→qualified maximum. Overlapping bands remain inconclusive/plateau. A supported limit without degradation is boundary-limited, not a global optimum. [Independent protocol review](../../ep-overlap-unprofiled-protocol-review/iteration-01/PROTOCOL_REVIEW.md).

Execution: Root alone submits. Use debug for this user-authorized campaign, four nodes21minutes, one node at most90minutes, two nodes at most45minutes. Fresh full-user queue check immediately before every sbatch/srun, never more than four total requested/allocated nodes. All Python, tests, training and analysis execute on allocated compute in setup/gh200-cuda130-runtime.toml. Separate one-node posthoc follows each pair. Pair allocation elapsed, per-case acquisition wall time, optimizer timers and posthoc elapsed remain separate.

Changed files: profiling/run_pretrain_ep_overlap_budget.py; tools/ep_overlap_budget_preflight.py; tools/ep_overlap_budget_validate.py; analysis/ep_overlap_budget_compare.py; tests/test_ep_overlap_budget_entrypoint.py; tests/test_ep_overlap_budget_validate.py; jobs/smoke/ep_overlap_budget_preflight.sbatch; jobs/training/ep_overlap_budget_pair.sbatch; jobs/training/ep_overlap_budget_worker.sh; jobs/analysis/ep_overlap_budget_posthoc.sbatch; separately hashed jobs/coordination/ep_overlap_budget_campaign.sh; tools/ep_overlap_budget_summary.py; tests/test_ep_overlap_budget_summary.py; jobs/analysis/ep_overlap_budget_summary.sbatch; docs/QWEN_UCCL_UNPROFILED_BUDGET_AUDIT.md and progress records. Original frozen144/149 sources, scripts/setup and native checkouts stay unchanged. Actual budgets remain truthful in preparation/configuration/all16startup/Buffer/runtime/finish records. Final source159/252 and162/2b170f are archived and verified.

Checks and results: The original twelve-case matrix passed all per-case gates, summary QA8tests, source/raw accounting rechecks, independent arithmetic audit and final review. New unprofiled budgets require new source-qualified QA and a bounded real pair qualification before full acquisition. Current source guards intentionally reject non20 OFF budgets, so a new explicit protocol is necessary.

Delegation: campaign_summary_finish owns new budget adapter and preflight. native_overlap_timing_review owns new paired jobs/worker and scheduling design. campaign_summary_review independently reviews measurement/stopping rules and source gates. No subagent submits compute or delegates further.

Native bounds: Static candidates are even4..64 from the thread limit. Threads=max(128,16B), dynamic sharedbytes=8192*max(4,B/2). QA3911604 reports232448bytes opt-in shared memory (227KiB), so56 fits229376bytes and58 fails237568bytes. The assumed227328byte/54cap was corrected before training. Preflight checks Config/hints for27 admissible budgets4..56; every worker checks its actual selected-device limits before Buffer creation. Resource admissibility does not prove successful kernel execution or performance at every point.

Risk: Sequential cases warm caches and can drift in clocks/network state. Reversed order and repeated same-placement pairs reduce this confound. Extreme budgets can make a pair exceed21minutes. New launchers record declared scheduling reserves and abort incomplete pairs. Do not accept partial5+5 cases or call timeout a precise accepted timing. Five sequential updates are not independent allocation replicates. Adaptive decisions bind immutable snapshots of the accepted ledger, rather than a mutable path that becomes stale after append.

Next action: No required work remains. Preserve the immutable outputs and [approved artifact hashes](../../../runs/ep-overlap-budget-summary-final-20261111T041250Z/FINAL_REVIEWED_ARTIFACTS.sha256). Reproduce with the recorded launch/protocol history and [summary commands](../../ep-overlap-budget-summary/iteration-01/COMMANDS.md). No new files were committed or pushed as part of this measurement request.

Compute milestone: 2026-11-10 21:33:26 UTC job 3911548: BEFORE allocated unprofiled-budget preflight node=nid007497 container=setup/gh200-cuda130-runtime.toml source=ee136afa677a0ab750dbd9fbfbe0b5f3875b292935da826b5c76e49b33eb0b05 CUDA_DEVICE_MAX_CONNECTIONS=32; native launch bounds and OFF-only source/configuration QA, no training

Compute milestone: 2026-11-10 21:34:35 UTC job 3911548: AFTER unprofiled-budget preflight status=0 node=nid007497 container=setup/gh200-cuda130-runtime.toml; no training executed; next inspect source/config/native bounds before paired pilot

Compute milestone: 2026-11-10 21:44:22 UTC job 3911604: BEFORE allocated unprofiled-budget preflight node=nid007273 container=setup/gh200-cuda130-runtime.toml source=9c34673060531047558c385d9b463114d114dc93f501009c5dfe6e8d8a47e159 CUDA_DEVICE_MAX_CONNECTIONS=32; native launch bounds and OFF-only source/configuration QA, no training

Compute milestone: 2026-11-10 21:45:32 UTC job 3911604: AFTER unprofiled-budget preflight status=0 node=nid007273 container=setup/gh200-cuda130-runtime.toml; no training executed; next inspect source/config/native bounds before paired pilot

Compute milestone: 2026-11-10 21:50:21 UTC pair_job=3911619 pair=ep-overlap-budget-off-pilot12-20-20261110T215008Z: BEFORE independent paired C0 acquisition candidate=12 order=candidate-first steps=2 nodes=nid[006173,007327,007340,007348] container=setup/gh200-cuda130-runtime.toml source=9c34673060531047558c385d9b463114d114dc93f501009c5dfe6e8d8a47e159

Compute milestone: 2026-11-10 21:50:25 UTC pair_job=3911619 pair=ep-overlap-budget-off-pilot12-20-20261110T215008Z: BEFORE leg=0 role=candidate budget=12 case=ep-overlap-budget-off-pilot12-20-20261110T215008Z-candidate-b12 fresh C0; declared reserve=360 seconds

Compute milestone: 2026-11-10 21:54:27 UTC pair_job=3911619 pair=ep-overlap-budget-off-pilot12-20-20261110T215008Z: AFTER all16 workers returned0 for leg=candidate; BEFORE small health/all-step timing summaries

Compute milestone: 2026-11-10 21:54:34 UTC pair_job=3911619 pair=ep-overlap-budget-off-pilot12-20-20261110T215008Z: AFTER leg=0 role=candidate budget=12 completed; case host wall recorded; strict paired acceptance pending posthoc

Compute milestone: 2026-11-10 21:54:34 UTC pair_job=3911619 pair=ep-overlap-budget-off-pilot12-20-20261110T215008Z: AFTER both independent cases acquired; release4nodes before1node posthoc; shared allocation elapsed is not a per-case time

Compute milestone: 2026-11-10 21:54:34 UTC pair_job=3911619 pair=ep-overlap-budget-off-pilot12-20-20261110T215008Z: AFTER pair phase=acquisition_complete exit=0 complete=1; preserve every case; partial pairs are not accepted

Compute milestone: 2026-11-10 22:04:03 UTC job 3911658: BEFORE allocated unprofiled-budget preflight node=nid006789 container=setup/gh200-cuda130-runtime.toml source=3d38dec32209504efa82f12f036183fabae8cf86fbae3cca96ce46c97041ade9 CUDA_DEVICE_MAX_CONNECTIONS=32; native launch bounds and OFF-only source/configuration QA, no training

Compute milestone: 2026-11-10 22:05:13 UTC job 3911658: AFTER unprofiled-budget preflight status=0 node=nid006789 container=setup/gh200-cuda130-runtime.toml; no training executed; next inspect source/config/native bounds before paired pilot

Compute milestone: 2026-11-10 22:12:13 UTC job 3911680: BEFORE allocated unprofiled-budget preflight node=nid006907 container=setup/gh200-cuda130-runtime.toml source=252ef8a09ddb96efef52ceebf339a52287e2dc48422f0a9fa472f5d41fc76927 CUDA_DEVICE_MAX_CONNECTIONS=32; native launch bounds and OFF-only source/configuration QA, no training

Compute milestone: 2026-11-10 22:13:24 UTC job 3911680: AFTER unprofiled-budget preflight status=0 node=nid006907 container=setup/gh200-cuda130-runtime.toml; no training executed; next inspect source/config/native bounds before paired pilot

Current execution: qualification3911710 PENDING debug4/21minutes under source252 and QA3911680. Final adaptive summarizer is delegated outside frozen159 sources. No accepted full unprofiled pair yet.

Compute milestone: 2026-11-10 22:23:31 UTC pair_job=3911710 pair=ep-overlap-budget-off-pilot12-20-20261110T221816Z: BEFORE independent paired C0 acquisition candidate=12 order=candidate-first steps=2 nodes=nid[006114,006665,007384,007624] container=setup/gh200-cuda130-runtime.toml source=252ef8a09ddb96efef52ceebf339a52287e2dc48422f0a9fa472f5d41fc76927

Compute milestone: 2026-11-10 22:23:34 UTC pair_job=3911710 pair=ep-overlap-budget-off-pilot12-20-20261110T221816Z: BEFORE leg=0 role=candidate budget=12 case=ep-overlap-budget-off-pilot12-20-20261110T221816Z-candidate-b12 fresh C0; declared reserve=360 seconds

Compute milestone: 2026-11-10 22:27:50 UTC pair_job=3911710 pair=ep-overlap-budget-off-pilot12-20-20261110T221816Z: AFTER all16 workers returned0 for leg=candidate; BEFORE small health/all-step timing summaries

Compute milestone: 2026-11-10 22:28:00 UTC pair_job=3911710 pair=ep-overlap-budget-off-pilot12-20-20261110T221816Z: AFTER leg=0 role=candidate budget=12 completed; case host wall recorded; strict paired acceptance pending posthoc

Compute milestone: 2026-11-10 22:28:00 UTC pair_job=3911710 pair=ep-overlap-budget-off-pilot12-20-20261110T221816Z: BEFORE leg=1 role=reference budget=20 case=ep-overlap-budget-off-pilot12-20-20261110T221816Z-reference-b20 fresh C0; declared reserve=360 seconds

Compute milestone: 2026-11-10 22:31:48 UTC pair_job=3911710 pair=ep-overlap-budget-off-pilot12-20-20261110T221816Z: AFTER all16 workers returned0 for leg=reference; BEFORE small health/all-step timing summaries

Compute milestone: 2026-11-10 22:31:54 UTC pair_job=3911710 pair=ep-overlap-budget-off-pilot12-20-20261110T221816Z: AFTER leg=1 role=reference budget=20 completed; case host wall recorded; strict paired acceptance pending posthoc

Compute milestone: 2026-11-10 22:31:54 UTC pair_job=3911710 pair=ep-overlap-budget-off-pilot12-20-20261110T221816Z: AFTER both independent cases acquired; release4nodes before1node posthoc; shared allocation elapsed is not a per-case time

Compute milestone: 2026-11-10 22:31:54 UTC pair_job=3911710 pair=ep-overlap-budget-off-pilot12-20-20261110T221816Z: AFTER pair phase=acquisition_complete exit=0 complete=1; preserve every case; partial pairs are not accepted

Compute milestone: 2026-11-10 22:32:51 UTC BEFORE qualification posthoc: repaired pilot3911710 COMPLETED0:0 in513s on4debugnodes, both2-update cases and receipts present; allocation released. Separate one debugnode via jobs/analysis/ep_overlap_budget_posthoc.sbatch, source252, configured container. Next strict raw health/routing/budget/source/sameGPU/input checks.

Compute milestone: 2026-11-10 22:32:54 UTC AFTER posthoc submission: job3911748 for qualification3911710; debug1node/up to90minutes. Root waits for strict validation before launching full pairs.

Compute milestone: 2026-11-10 22:33:06 UTC budget_posthoc=3911748 pair_training=3911710: BEFORE separate one-node debug analysis; pair allocation released; strict per-leg and paired placement/source/input gates; container=setup/gh200-cuda130-runtime.toml

Compute milestone: 2026-11-10 22:33:21 UTC budget_posthoc=3911748 pair_training=3911710: AFTER both leg validations and paired comparison returned0; actual16GPU equality, inputs and timing distinction checked

Compute milestone: 2026-11-10 22:33:21 UTC budget_posthoc=3911748 pair_training=3911710: AFTER paired posthoc exit=0 complete=1; preserve original cases and partial analysis

Budget coordinator milestone: 2026-11-10 22:34:34 UTC BEFORE full adaptive sweep: qualification training3911710 and posthoc3911748 COMPLETED0:0; same actual16GPUs/input, strict routing/health/actual budget gates passed. Source159/252 and driverD3 unchanged; allocated driver syntax3911615 passed and independent re-review found no blocker. Start reviewed detached coordinator ep-overlap-budget-off-adaptive-20261110T223433Z with six initial reversed-order pairs and adaptive boundaries. Exclude known-bad nid006657. All training debug4/21minutes and separate posthoc debug1/up to90minutes.

Budget coordinator milestone: 2026-11-10 22:34:37 UTC: Qualified source=252ef8a09ddb96efef52ceebf339a52287e2dc48422f0a9fa472f5d41fc76927 preflight=3911680 two-update pair=ep-overlap-budget-off-pilot12-20-20261110T221816Z; qualification excluded from accepted.tsv

Budget coordinator milestone: 2026-11-10 22:34:37 UTC: BEFORE training candidate=12 order=reference-first replicate=1 attempt=1; four debug nodes/21minutes; independent C0 per leg

Budget coordinator milestone: 2026-11-10 22:34:42 UTC: AFTER training submit job=3911753 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b12-reference-first-r1-attempt1; strict acceptance pending

Budget coordinator milestone: 2026-11-10 22:34:42 UTC: WATCH job=3911753 3911753 debug PENDING 4 (None)

Compute milestone: 2026-11-10 22:35:11 UTC pair_job=3911753 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b12-reference-first-r1-attempt1: BEFORE independent paired C0 acquisition candidate=12 order=reference-first steps=10 nodes=nid[007282,007296,007458,007648] container=setup/gh200-cuda130-runtime.toml source=252ef8a09ddb96efef52ceebf339a52287e2dc48422f0a9fa472f5d41fc76927

Compute milestone: 2026-11-10 22:35:12 UTC pair_job=3911753 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b12-reference-first-r1-attempt1: BEFORE leg=0 role=reference budget=20 case=ep-overlap-budget-off-adaptive-20261110T223433Z-b12-reference-first-r1-attempt1-reference-b20 fresh C0; declared reserve=480 seconds

Budget coordinator milestone: 2026-11-10 22:35:12 UTC: WATCH job=3911753 3911753 debug RUNNING 4 nid[007282,007296,007458,007648]

Scheduler note: 2026-11-10 22:38:55 UTC Full pair3911753: actual Slurm request confirms debug4/21minutes on nid[007282,007296,007458,007648]. SBATCH_EXCLUDE supplied in the environment did not populate ExcNodeList; no claim of an active exclusion. Earlier nid006657 boot issue is historical; the node is now MIXED and absent from this allocation. No change to the reviewed running controller.

Compute milestone: 2026-11-10 22:41:26 UTC pair_job=3911753 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b12-reference-first-r1-attempt1: AFTER all16 workers returned0 for leg=reference; BEFORE small health/all-step timing summaries

Compute milestone: 2026-11-10 22:41:34 UTC pair_job=3911753 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b12-reference-first-r1-attempt1: AFTER leg=0 role=reference budget=20 completed; case host wall recorded; strict paired acceptance pending posthoc

Compute milestone: 2026-11-10 22:41:35 UTC pair_job=3911753 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b12-reference-first-r1-attempt1: BEFORE leg=1 role=candidate budget=12 case=ep-overlap-budget-off-adaptive-20261110T223433Z-b12-reference-first-r1-attempt1-candidate-b12 fresh C0; declared reserve=480 seconds

Compute milestone: 2026-11-10 22:47:32 UTC pair_job=3911753 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b12-reference-first-r1-attempt1: AFTER all16 workers returned0 for leg=candidate; BEFORE small health/all-step timing summaries

Compute milestone: 2026-11-10 22:47:41 UTC pair_job=3911753 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b12-reference-first-r1-attempt1: AFTER leg=1 role=candidate budget=12 completed; case host wall recorded; strict paired acceptance pending posthoc

Compute milestone: 2026-11-10 22:47:41 UTC pair_job=3911753 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b12-reference-first-r1-attempt1: AFTER both independent cases acquired; release4nodes before1node posthoc; shared allocation elapsed is not a per-case time

Compute milestone: 2026-11-10 22:47:41 UTC pair_job=3911753 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b12-reference-first-r1-attempt1: AFTER pair phase=acquisition_complete exit=0 complete=1; preserve every case; partial pairs are not accepted

Budget coordinator milestone: 2026-11-10 22:47:57 UTC: WATCH job=3911753 3911753 debug COMPLETING 1 nid007458

Budget coordinator milestone: 2026-11-10 22:48:27 UTC: BEFORE posthoc pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b12-reference-first-r1-attempt1 training=3911753 released; one debug node/up to90minutes

Budget coordinator milestone: 2026-11-10 22:48:29 UTC: AFTER posthoc submit job=3911822 training=3911753 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b12-reference-first-r1-attempt1; paired and decision gates pending

Budget coordinator milestone: 2026-11-10 22:48:29 UTC: WATCH job=3911822 3911822 debug PENDING 1 (None)

Budget coordinator milestone: 2026-11-10 22:48:59 UTC: WATCH job=3911822 3911822 debug RUNNING 1 nid006665

Compute milestone: 2026-11-10 22:49:04 UTC budget_posthoc=3911822 pair_training=3911753: BEFORE separate one-node debug analysis; pair allocation released; strict per-leg and paired placement/source/input gates; container=setup/gh200-cuda130-runtime.toml

Compute milestone: 2026-11-10 22:49:19 UTC budget_posthoc=3911822 pair_training=3911753: AFTER both leg validations and paired comparison returned0; actual16GPU equality, inputs and timing distinction checked

Compute milestone: 2026-11-10 22:49:19 UTC budget_posthoc=3911822 pair_training=3911753: AFTER paired posthoc exit=0 complete=1; preserve original cases and partial analysis

Budget coordinator milestone: 2026-11-10 22:49:30 UTC: WATCH job=3911822 3911822 debug COMPLETING 1 nid006665

Budget summary milestone: 2026-11-10 22:49:42 UTC Final adaptive summary source162 frozen and archived: runs/ep-overlap-budget-off-final-source-20261110T224818Z/source.sha256 SHA2b170f107aabc07c1da917f37ff6ed8eb5c97fb2da03850e64f62037a4b2c5f6. Independent review passed after recomputing route totals from raw bound ownership/all16audits. Seven allocated tests pending; no runtime on login. Acquisition159/source252 and controllerD3 unchanged. QA will wait until the four-node coordinator finishes to avoid a capacity race.

Budget coordinator milestone: 2026-11-10 22:50:04 UTC: ACCEPTED pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b12-reference-first-r1-attempt1 candidate=12 order=reference-first replicate=1 training=3911753 posthoc=3911822; total=1

Budget coordinator milestone: 2026-11-10 22:50:04 UTC: BEFORE training candidate=32 order=candidate-first replicate=1 attempt=1; four debug nodes/21minutes; independent C0 per leg

Budget coordinator milestone: 2026-11-10 22:50:06 UTC: AFTER training submit job=3911828 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b32-candidate-first-r1-attempt1; strict acceptance pending

Budget coordinator milestone: 2026-11-10 22:50:08 UTC: WATCH job=3911828 3911828 debug PENDING 4 (None)

Compute milestone: 2026-11-10 22:50:37 UTC pair_job=3911828 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b32-candidate-first-r1-attempt1: BEFORE independent paired C0 acquisition candidate=32 order=candidate-first steps=10 nodes=nid[006771,006971,007295,007506] container=setup/gh200-cuda130-runtime.toml source=252ef8a09ddb96efef52ceebf339a52287e2dc48422f0a9fa472f5d41fc76927

Compute milestone: 2026-11-10 22:50:38 UTC pair_job=3911828 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b32-candidate-first-r1-attempt1: BEFORE leg=0 role=candidate budget=32 case=ep-overlap-budget-off-adaptive-20261110T223433Z-b32-candidate-first-r1-attempt1-candidate-b32 fresh C0; declared reserve=480 seconds

Budget coordinator milestone: 2026-11-10 22:50:38 UTC: WATCH job=3911828 3911828 debug RUNNING 4 nid[006771,006971,007295,007506]

Compute milestone: 2026-11-10 22:56:45 UTC pair_job=3911828 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b32-candidate-first-r1-attempt1: AFTER all16 workers returned0 for leg=candidate; BEFORE small health/all-step timing summaries

Compute milestone: 2026-11-10 22:56:52 UTC pair_job=3911828 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b32-candidate-first-r1-attempt1: AFTER leg=0 role=candidate budget=32 completed; case host wall recorded; strict paired acceptance pending posthoc

Compute milestone: 2026-11-10 22:56:53 UTC pair_job=3911828 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b32-candidate-first-r1-attempt1: BEFORE leg=1 role=reference budget=20 case=ep-overlap-budget-off-adaptive-20261110T223433Z-b32-candidate-first-r1-attempt1-reference-b20 fresh C0; declared reserve=480 seconds

Compute milestone: 2026-11-10 23:02:48 UTC pair_job=3911828 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b32-candidate-first-r1-attempt1: AFTER all16 workers returned0 for leg=reference; BEFORE small health/all-step timing summaries

Compute milestone: 2026-11-10 23:02:57 UTC pair_job=3911828 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b32-candidate-first-r1-attempt1: AFTER leg=1 role=reference budget=20 completed; case host wall recorded; strict paired acceptance pending posthoc

Compute milestone: 2026-11-10 23:02:57 UTC pair_job=3911828 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b32-candidate-first-r1-attempt1: AFTER both independent cases acquired; release4nodes before1node posthoc; shared allocation elapsed is not a per-case time

Compute milestone: 2026-11-10 23:02:57 UTC pair_job=3911828 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b32-candidate-first-r1-attempt1: AFTER pair phase=acquisition_complete exit=0 complete=1; preserve every case; partial pairs are not accepted

Budget coordinator milestone: 2026-11-10 23:03:26 UTC: BEFORE posthoc pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b32-candidate-first-r1-attempt1 training=3911828 released; one debug node/up to90minutes

Budget coordinator milestone: 2026-11-10 23:03:28 UTC: AFTER posthoc submit job=3911891 training=3911828 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b32-candidate-first-r1-attempt1; paired and decision gates pending

Budget coordinator milestone: 2026-11-10 23:03:28 UTC: WATCH job=3911891 3911891 debug PENDING 1 (Resources)

Budget coordinator milestone: 2026-11-10 23:03:58 UTC: WATCH job=3911891 3911891 debug RUNNING 1 nid007318

Compute milestone: 2026-11-10 23:04:06 UTC budget_posthoc=3911891 pair_training=3911828: BEFORE separate one-node debug analysis; pair allocation released; strict per-leg and paired placement/source/input gates; container=setup/gh200-cuda130-runtime.toml

Compute milestone: 2026-11-10 23:04:20 UTC budget_posthoc=3911891 pair_training=3911828: AFTER both leg validations and paired comparison returned0; actual16GPU equality, inputs and timing distinction checked

Compute milestone: 2026-11-10 23:04:20 UTC budget_posthoc=3911891 pair_training=3911828: AFTER paired posthoc exit=0 complete=1; preserve original cases and partial analysis

Budget coordinator milestone: 2026-11-10 23:04:35 UTC: ACCEPTED pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b32-candidate-first-r1-attempt1 candidate=32 order=candidate-first replicate=1 training=3911828 posthoc=3911891; total=2

Budget coordinator milestone: 2026-11-10 23:04:35 UTC: BEFORE training candidate=40 order=reference-first replicate=1 attempt=1; four debug nodes/21minutes; independent C0 per leg

Budget coordinator milestone: 2026-11-10 23:04:39 UTC: AFTER training submit job=3911907 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b40-reference-first-r1-attempt1; strict acceptance pending

Budget coordinator milestone: 2026-11-10 23:04:39 UTC: WATCH job=3911907 3911907 debug PENDING 4 (None)

Budget result: 2026-11-10 23:05:00 UTC Accepted 2/6 initial pairs. 32/candidate-first r1 jobs3911828/3911891: 15.37486s vs15.30088s at20, ratio1.00483502 (+0.4835%); shared training allocation748s. Both initial ratios fall within2%; no budget ranking yet. Continue reversed orders and the predeclared extension rule.

Compute milestone: 2026-11-10 23:05:25 UTC pair_job=3911907 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b40-reference-first-r1-attempt1: BEFORE independent paired C0 acquisition candidate=40 order=reference-first steps=10 nodes=nid[006801,007122,007427,007502] container=setup/gh200-cuda130-runtime.toml source=252ef8a09ddb96efef52ceebf339a52287e2dc48422f0a9fa472f5d41fc76927

Compute milestone: 2026-11-10 23:05:26 UTC pair_job=3911907 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b40-reference-first-r1-attempt1: BEFORE leg=0 role=reference budget=20 case=ep-overlap-budget-off-adaptive-20261110T223433Z-b40-reference-first-r1-attempt1-reference-b20 fresh C0; declared reserve=480 seconds

Budget coordinator milestone: 2026-11-10 23:05:40 UTC: WATCH job=3911907 3911907 debug RUNNING 4 nid[006801,007122,007427,007502]

Compute milestone: 2026-11-10 23:11:14 UTC pair_job=3911907 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b40-reference-first-r1-attempt1: AFTER all16 workers returned0 for leg=reference; BEFORE small health/all-step timing summaries

Compute milestone: 2026-11-10 23:11:21 UTC pair_job=3911907 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b40-reference-first-r1-attempt1: AFTER leg=0 role=reference budget=20 completed; case host wall recorded; strict paired acceptance pending posthoc

Compute milestone: 2026-11-10 23:11:22 UTC pair_job=3911907 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b40-reference-first-r1-attempt1: BEFORE leg=1 role=candidate budget=40 case=ep-overlap-budget-off-adaptive-20261110T223433Z-b40-reference-first-r1-attempt1-candidate-b40 fresh C0; declared reserve=480 seconds

Compute milestone: 2026-11-10 23:16:51 UTC pair_job=3911907 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b40-reference-first-r1-attempt1: AFTER all16 workers returned0 for leg=candidate; BEFORE small health/all-step timing summaries

Compute milestone: 2026-11-10 23:17:00 UTC pair_job=3911907 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b40-reference-first-r1-attempt1: AFTER leg=1 role=candidate budget=40 completed; case host wall recorded; strict paired acceptance pending posthoc

Compute milestone: 2026-11-10 23:17:00 UTC pair_job=3911907 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b40-reference-first-r1-attempt1: AFTER both independent cases acquired; release4nodes before1node posthoc; shared allocation elapsed is not a per-case time

Compute milestone: 2026-11-10 23:17:00 UTC pair_job=3911907 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b40-reference-first-r1-attempt1: AFTER pair phase=acquisition_complete exit=0 complete=1; preserve every case; partial pairs are not accepted

Budget coordinator milestone: 2026-11-10 23:17:21 UTC: BEFORE posthoc pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b40-reference-first-r1-attempt1 training=3911907 released; one debug node/up to90minutes

Budget coordinator milestone: 2026-11-10 23:17:24 UTC: AFTER posthoc submit job=3911947 training=3911907 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b40-reference-first-r1-attempt1; paired and decision gates pending

Budget coordinator milestone: 2026-11-10 23:17:24 UTC: WATCH job=3911947 3911947 debug PENDING 1 (None)

Compute milestone: 2026-11-10 23:17:34 UTC budget_posthoc=3911947 pair_training=3911907: BEFORE separate one-node debug analysis; pair allocation released; strict per-leg and paired placement/source/input gates; container=setup/gh200-cuda130-runtime.toml

Compute milestone: 2026-11-10 23:17:54 UTC budget_posthoc=3911947 pair_training=3911907: AFTER both leg validations and paired comparison returned0; actual16GPU equality, inputs and timing distinction checked

Compute milestone: 2026-11-10 23:17:54 UTC budget_posthoc=3911947 pair_training=3911907: AFTER paired posthoc exit=0 complete=1; preserve original cases and partial analysis

Budget coordinator milestone: 2026-11-10 23:17:54 UTC: WATCH job=3911947 3911947 debug RUNNING 1 nid007267

Budget coordinator milestone: 2026-11-10 23:18:32 UTC: ACCEPTED pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b40-reference-first-r1-attempt1 candidate=40 order=reference-first replicate=1 training=3911907 posthoc=3911947; total=3

Budget coordinator milestone: 2026-11-10 23:18:32 UTC: BEFORE training candidate=12 order=candidate-first replicate=2 attempt=1; four debug nodes/21minutes; independent C0 per leg

Budget coordinator milestone: 2026-11-10 23:18:33 UTC: AFTER training submit job=3911951 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b12-candidate-first-r2-attempt1; strict acceptance pending

Budget coordinator milestone: 2026-11-10 23:18:33 UTC: WATCH job=3911951 3911951 debug PENDING 4 (None)

Compute milestone: 2026-11-10 23:18:42 UTC pair_job=3911951 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b12-candidate-first-r2-attempt1: BEFORE independent paired C0 acquisition candidate=12 order=candidate-first steps=10 nodes=nid[006955,006965,006982,007267] container=setup/gh200-cuda130-runtime.toml source=252ef8a09ddb96efef52ceebf339a52287e2dc48422f0a9fa472f5d41fc76927

Compute milestone: 2026-11-10 23:18:42 UTC pair_job=3911951 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b12-candidate-first-r2-attempt1: BEFORE leg=0 role=candidate budget=12 case=ep-overlap-budget-off-adaptive-20261110T223433Z-b12-candidate-first-r2-attempt1-candidate-b12 fresh C0; declared reserve=480 seconds

Budget result: 2026-11-10 23:18:58 UTC Accepted 3/6 initial full pairs, no full attempts failed. 40/reference-first r1 jobs3911907/3911947: 12.71506s vs13.01380s at20, ratio0.97704437 (-2.2956%); shared training allocation704s. Single allocation result, no ranking or adaptive decision yet. Repeat12/candidate-first r2 running3911951. Initial ratios12=0.99691366,32=1.00483502,40=0.97704437. All exact matched640-sample prefix.

Budget coordinator milestone: 2026-11-10 23:19:05 UTC: WATCH job=3911951 3911951 debug RUNNING 4 nid[006955,006965,006982,007267]

Compute milestone: 2026-11-10 23:24:56 UTC pair_job=3911951 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b12-candidate-first-r2-attempt1: AFTER all16 workers returned0 for leg=candidate; BEFORE small health/all-step timing summaries

Compute milestone: 2026-11-10 23:25:03 UTC pair_job=3911951 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b12-candidate-first-r2-attempt1: AFTER leg=0 role=candidate budget=12 completed; case host wall recorded; strict paired acceptance pending posthoc

Compute milestone: 2026-11-10 23:25:04 UTC pair_job=3911951 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b12-candidate-first-r2-attempt1: BEFORE leg=1 role=reference budget=20 case=ep-overlap-budget-off-adaptive-20261110T223433Z-b12-candidate-first-r2-attempt1-reference-b20 fresh C0; declared reserve=480 seconds

Compute milestone: 2026-11-10 23:30:49 UTC pair_job=3911951 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b12-candidate-first-r2-attempt1: AFTER all16 workers returned0 for leg=reference; BEFORE small health/all-step timing summaries

Compute milestone: 2026-11-10 23:30:56 UTC pair_job=3911951 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b12-candidate-first-r2-attempt1: AFTER leg=1 role=reference budget=20 completed; case host wall recorded; strict paired acceptance pending posthoc

Compute milestone: 2026-11-10 23:30:56 UTC pair_job=3911951 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b12-candidate-first-r2-attempt1: AFTER both independent cases acquired; release4nodes before1node posthoc; shared allocation elapsed is not a per-case time

Compute milestone: 2026-11-10 23:30:56 UTC pair_job=3911951 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b12-candidate-first-r2-attempt1: AFTER pair phase=acquisition_complete exit=0 complete=1; preserve every case; partial pairs are not accepted

Budget coordinator milestone: 2026-11-10 23:31:17 UTC: BEFORE posthoc pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b12-candidate-first-r2-attempt1 training=3911951 released; one debug node/up to90minutes

Budget coordinator milestone: 2026-11-10 23:31:19 UTC: AFTER posthoc submit job=3912015 training=3911951 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b12-candidate-first-r2-attempt1; paired and decision gates pending

Budget coordinator milestone: 2026-11-10 23:31:20 UTC: WATCH job=3912015 3912015 debug PENDING 1 (None)

Compute milestone: 2026-11-10 23:31:27 UTC budget_posthoc=3912015 pair_training=3911951: BEFORE separate one-node debug analysis; pair allocation released; strict per-leg and paired placement/source/input gates; container=setup/gh200-cuda130-runtime.toml

Compute milestone: 2026-11-10 23:31:49 UTC budget_posthoc=3912015 pair_training=3911951: AFTER both leg validations and paired comparison returned0; actual16GPU equality, inputs and timing distinction checked

Compute milestone: 2026-11-10 23:31:49 UTC budget_posthoc=3912015 pair_training=3911951: AFTER paired posthoc exit=0 complete=1; preserve original cases and partial analysis

Budget coordinator milestone: 2026-11-10 23:31:50 UTC: WATCH job=3912015 3912015 debug COMPLETING 1 nid006882

Budget coordinator milestone: 2026-11-10 23:32:32 UTC: ACCEPTED pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b12-candidate-first-r2-attempt1 candidate=12 order=candidate-first replicate=2 training=3911951 posthoc=3912015; total=4

Budget coordinator milestone: 2026-11-10 23:32:33 UTC: BEFORE training candidate=32 order=reference-first replicate=2 attempt=1; four debug nodes/21minutes; independent C0 per leg

Budget coordinator milestone: 2026-11-10 23:32:33 UTC: AFTER training submit job=3912016 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b32-reference-first-r2-attempt1; strict acceptance pending

Budget coordinator milestone: 2026-11-10 23:32:33 UTC: WATCH job=3912016 3912016 debug PENDING 4 (None)

Compute milestone: 2026-11-10 23:32:41 UTC pair_job=3912016 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b32-reference-first-r2-attempt1: BEFORE independent paired C0 acquisition candidate=32 order=reference-first steps=10 nodes=nid[007402,007427,007439,007446] container=setup/gh200-cuda130-runtime.toml source=252ef8a09ddb96efef52ceebf339a52287e2dc48422f0a9fa472f5d41fc76927

Compute milestone: 2026-11-10 23:32:42 UTC pair_job=3912016 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b32-reference-first-r2-attempt1: BEFORE leg=0 role=reference budget=20 case=ep-overlap-budget-off-adaptive-20261110T223433Z-b32-reference-first-r2-attempt1-reference-b20 fresh C0; declared reserve=480 seconds

Budget result: 2026-11-10 23:32:47 UTC Accepted 4/6 initial full pairs. 12/candidate-first r2 jobs3911951/3912015 COMPLETED0:0, shared allocation743s, posthoc29s. Measured15.00924s vs14.64670s at20, ratio1.02475233 (+2.4752%). R1 ratio0.99691366 and R2 straddle1 with >2% relative spread, so 12 triggers third pair after the six predeclared pairs. 32/reference-first r2 is job3912016. No failed full attempt yet.

Budget coordinator milestone: 2026-11-10 23:33:05 UTC: WATCH job=3912016 3912016 debug RUNNING 4 nid[007402,007427,007439,007446]

Budget failure triage: 2026-11-10 23:41:55 UTC Full pair3912016 (32/reference-first r2) anomalous shutdown: reference20 rank0 reached update10 at23:37:56UTC and all16 backend cleanup records exist. Outer worker srun.0 remains RUNNING; no case receipt, no native post-run proof or candidate32 start. Fifteen ranks returned SystemExit0; rank15 alone lacks exit and reports repeated W&B upload finalization through23:38:47UTC. Bridge wandb_writer.finish is the strongest hypothesis, no process stack. Treat as an incomplete pair, not a budget slowdown. At 12m30s elapsed, the remaining walltime falls below the mandatory 510s second-leg reserve; if still in the first worker then, stop this attempt and let coordinator retry fresh.

Budget failure triage: 2026-11-10 23:45:10 UTC Stopping incomplete owned attempt 3912016: W&B finish for rank15 retries HTTP500 after all10 reference updates and all16 UCCL cleanups. Candidate32 never started. Live Slurm remaining=504s below mandatory510s second-leg reserve. Save all raw evidence, no paired timing accepted. Reviewed coordinator retries whole pair with new ID after confirmed cancellation/release.

Compute milestone: 2026-11-10 23:45:10 UTC pair_job=3912016 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b32-reference-first-r2-attempt1: AFTER pair phase=leg-0-reference exit=1 complete=0; preserve every case; partial pairs are not accepted

Budget coordinator milestone: 2026-11-10 23:45:13 UTC: WATCH job=3912016 3912016 debug COMPLETING 4 nid[007402,007427,007439,007446]

Budget coordinator milestone: 2026-11-10 23:45:45 UTC: VERIFIED FAILURE job=3912016 state=CANCELLED by 65629 exit=0:0 accounting=3912016|CANCELLED by 65629|0:0|756|nid[007402,007427,007439,007446]

Budget coordinator milestone: 2026-11-10 23:45:45 UTC: FAILED pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b32-reference-first-r2-attempt1 reason=training_scheduler; preserved original legs and analysis; retry only with a fresh whole-pair ID

Budget coordinator milestone: 2026-11-10 23:45:45 UTC: BEFORE training candidate=32 order=reference-first replicate=2 attempt=2; four debug nodes/21minutes; independent C0 per leg

Budget coordinator milestone: 2026-11-10 23:45:50 UTC: AFTER training submit job=3912077 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b32-reference-first-r2-attempt2; strict acceptance pending

Budget coordinator milestone: 2026-11-10 23:45:50 UTC: WATCH job=3912077 3912077 debug PENDING 4 (None)

Compute milestone: 2026-11-10 23:46:34 UTC pair_job=3912077 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b32-reference-first-r2-attempt2: BEFORE independent paired C0 acquisition candidate=32 order=reference-first steps=10 nodes=nid[006160,006305,006781,007624] container=setup/gh200-cuda130-runtime.toml source=252ef8a09ddb96efef52ceebf339a52287e2dc48422f0a9fa472f5d41fc76927

Compute milestone: 2026-11-10 23:46:38 UTC pair_job=3912077 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b32-reference-first-r2-attempt2: BEFORE leg=0 role=reference budget=20 case=ep-overlap-budget-off-adaptive-20261110T223433Z-b32-reference-first-r2-attempt2-reference-b20 fresh C0; declared reserve=480 seconds

Budget coordinator milestone: 2026-11-10 23:46:51 UTC: WATCH job=3912077 3912077 debug RUNNING 4 nid[006160,006305,006781,007624]

Compute milestone: 2026-11-10 23:52:26 UTC pair_job=3912077 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b32-reference-first-r2-attempt2: AFTER all16 workers returned0 for leg=reference; BEFORE small health/all-step timing summaries

Compute milestone: 2026-11-10 23:52:34 UTC pair_job=3912077 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b32-reference-first-r2-attempt2: AFTER leg=0 role=reference budget=20 completed; case host wall recorded; strict paired acceptance pending posthoc

Compute milestone: 2026-11-10 23:52:35 UTC pair_job=3912077 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b32-reference-first-r2-attempt2: BEFORE leg=1 role=candidate budget=32 case=ep-overlap-budget-off-adaptive-20261110T223433Z-b32-reference-first-r2-attempt2-candidate-b32 fresh C0; declared reserve=480 seconds

Compute milestone: 2026-11-10 23:58:07 UTC pair_job=3912077 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b32-reference-first-r2-attempt2: AFTER all16 workers returned0 for leg=candidate; BEFORE small health/all-step timing summaries

Compute milestone: 2026-11-10 23:58:16 UTC pair_job=3912077 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b32-reference-first-r2-attempt2: AFTER leg=1 role=candidate budget=32 completed; case host wall recorded; strict paired acceptance pending posthoc

Compute milestone: 2026-11-10 23:58:16 UTC pair_job=3912077 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b32-reference-first-r2-attempt2: AFTER both independent cases acquired; release4nodes before1node posthoc; shared allocation elapsed is not a per-case time

Compute milestone: 2026-11-10 23:58:16 UTC pair_job=3912077 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b32-reference-first-r2-attempt2: AFTER pair phase=acquisition_complete exit=0 complete=1; preserve every case; partial pairs are not accepted

Budget coordinator milestone: 2026-11-10 23:58:34 UTC: BEFORE posthoc pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b32-reference-first-r2-attempt2 training=3912077 released; one debug node/up to90minutes

Budget coordinator milestone: 2026-11-10 23:58:37 UTC: AFTER posthoc submit job=3912091 training=3912077 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b32-reference-first-r2-attempt2; paired and decision gates pending

Budget coordinator milestone: 2026-11-10 23:58:37 UTC: WATCH job=3912091 3912091 debug PENDING 1 (None)

Compute milestone: 2026-11-10 23:58:52 UTC budget_posthoc=3912091 pair_training=3912077: BEFORE separate one-node debug analysis; pair allocation released; strict per-leg and paired placement/source/input gates; container=setup/gh200-cuda130-runtime.toml

Budget coordinator milestone: 2026-11-10 23:59:07 UTC: WATCH job=3912091 3912091 debug RUNNING 1 nid007125

Compute milestone: 2026-11-10 23:59:16 UTC budget_posthoc=3912091 pair_training=3912077: AFTER both leg validations and paired comparison returned0; actual16GPU equality, inputs and timing distinction checked

Compute milestone: 2026-11-10 23:59:16 UTC budget_posthoc=3912091 pair_training=3912077: AFTER paired posthoc exit=0 complete=1; preserve original cases and partial analysis

Budget coordinator milestone: 2026-11-10 23:59:52 UTC: ACCEPTED pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b32-reference-first-r2-attempt2 candidate=32 order=reference-first replicate=2 training=3912077 posthoc=3912091; total=5

Budget coordinator milestone: 2026-11-10 23:59:52 UTC: BEFORE training candidate=40 order=candidate-first replicate=2 attempt=1; four debug nodes/21minutes; independent C0 per leg

Budget coordinator milestone: 2026-11-10 23:59:53 UTC: AFTER training submit job=3912096 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b40-candidate-first-r2-attempt1; strict acceptance pending

Budget coordinator milestone: 2026-11-10 23:59:53 UTC: WATCH job=3912096 3912096 debug PENDING 4 (None)

Budget result: 2026-11-11 00:00:32 UTC Accepted 5/6 initial full pairs. 32/reference-first r2 fresh attempt2 training3912077/posthoc3912091 passed after the preserved logging-stall attempt3912016. Shared allocation712s. Measured13.33678s at32 vs13.33532s at20, ratio1.00010948 (+0.01095%). R1 ratio1.00483502, both within0.5%; the predeclared near-baseline rule calls for a third pair. Final initial40/candidate-first r2 queued3912096, four debug nodes21minutes. Frozen source159/252 and D3 unchanged.

Budget coordinator milestone: 2026-11-11 00:01:56 UTC: WATCH job=3912096 3912096 debug RUNNING 4 nid[006031,006108,006665,007652]

Compute milestone: 2026-11-11 00:02:03 UTC pair_job=3912096 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b40-candidate-first-r2-attempt1: BEFORE independent paired C0 acquisition candidate=40 order=candidate-first steps=10 nodes=nid[006031,006108,006665,007652] container=setup/gh200-cuda130-runtime.toml source=252ef8a09ddb96efef52ceebf339a52287e2dc48422f0a9fa472f5d41fc76927

Compute milestone: 2026-11-11 00:02:06 UTC pair_job=3912096 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b40-candidate-first-r2-attempt1: BEFORE leg=0 role=candidate budget=40 case=ep-overlap-budget-off-adaptive-20261110T223433Z-b40-candidate-first-r2-attempt1-candidate-b40 fresh C0; declared reserve=480 seconds

Compute milestone: 2026-11-11 00:08:07 UTC pair_job=3912096 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b40-candidate-first-r2-attempt1: AFTER all16 workers returned0 for leg=candidate; BEFORE small health/all-step timing summaries

Compute milestone: 2026-11-11 00:08:15 UTC pair_job=3912096 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b40-candidate-first-r2-attempt1: AFTER leg=0 role=candidate budget=40 completed; case host wall recorded; strict paired acceptance pending posthoc

Compute milestone: 2026-11-11 00:08:16 UTC pair_job=3912096 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b40-candidate-first-r2-attempt1: BEFORE leg=1 role=reference budget=20 case=ep-overlap-budget-off-adaptive-20261110T223433Z-b40-candidate-first-r2-attempt1-reference-b20 fresh C0; declared reserve=480 seconds

Compute milestone: 2026-11-11 00:14:02 UTC pair_job=3912096 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b40-candidate-first-r2-attempt1: AFTER all16 workers returned0 for leg=reference; BEFORE small health/all-step timing summaries

Compute milestone: 2026-11-11 00:14:11 UTC pair_job=3912096 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b40-candidate-first-r2-attempt1: AFTER leg=1 role=reference budget=20 completed; case host wall recorded; strict paired acceptance pending posthoc

Compute milestone: 2026-11-11 00:14:11 UTC pair_job=3912096 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b40-candidate-first-r2-attempt1: AFTER both independent cases acquired; release4nodes before1node posthoc; shared allocation elapsed is not a per-case time

Compute milestone: 2026-11-11 00:14:11 UTC pair_job=3912096 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b40-candidate-first-r2-attempt1: AFTER pair phase=acquisition_complete exit=0 complete=1; preserve every case; partial pairs are not accepted

Budget coordinator milestone: 2026-11-11 00:14:37 UTC: BEFORE posthoc pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b40-candidate-first-r2-attempt1 training=3912096 released; one debug node/up to90minutes

Budget coordinator milestone: 2026-11-11 00:14:38 UTC: AFTER posthoc submit job=3912187 training=3912096 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b40-candidate-first-r2-attempt1; paired and decision gates pending

Budget coordinator milestone: 2026-11-11 00:14:38 UTC: WATCH job=3912187 3912187 debug PENDING 1 (None)

Compute milestone: 2026-11-11 00:14:46 UTC budget_posthoc=3912187 pair_training=3912096: BEFORE separate one-node debug analysis; pair allocation released; strict per-leg and paired placement/source/input gates; container=setup/gh200-cuda130-runtime.toml

Budget coordinator milestone: 2026-11-11 00:15:08 UTC: WATCH job=3912187 3912187 debug RUNNING 1 nid006665

Compute milestone: 2026-11-11 00:15:09 UTC budget_posthoc=3912187 pair_training=3912096: AFTER both leg validations and paired comparison returned0; actual16GPU equality, inputs and timing distinction checked

Compute milestone: 2026-11-11 00:15:09 UTC budget_posthoc=3912187 pair_training=3912096: AFTER paired posthoc exit=0 complete=1; preserve original cases and partial analysis

Budget coordinator milestone: 2026-11-11 00:15:58 UTC: ACCEPTED pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b40-candidate-first-r2-attempt1 candidate=40 order=candidate-first replicate=2 training=3912096 posthoc=3912187; total=6

Budget coordinator milestone: 2026-11-11 00:15:58 UTC: BEFORE training candidate=12 order=reference-first replicate=3 attempt=1; four debug nodes/21minutes; independent C0 per leg

Budget coordinator milestone: 2026-11-11 00:15:58 UTC: AFTER training submit job=3912190 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b12-reference-first-r3-attempt1; strict acceptance pending

Budget coordinator milestone: 2026-11-11 00:15:58 UTC: WATCH job=3912190 3912190 debug PENDING 4 (None)

Compute milestone: 2026-11-11 00:16:08 UTC pair_job=3912190 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b12-reference-first-r3-attempt1: BEFORE independent paired C0 acquisition candidate=12 order=reference-first steps=10 nodes=nid[006031,006108,007438,007453] container=setup/gh200-cuda130-runtime.toml source=252ef8a09ddb96efef52ceebf339a52287e2dc48422f0a9fa472f5d41fc76927

Compute milestone: 2026-11-11 00:16:09 UTC pair_job=3912190 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b12-reference-first-r3-attempt1: BEFORE leg=0 role=reference budget=20 case=ep-overlap-budget-off-adaptive-20261110T223433Z-b12-reference-first-r3-attempt1-reference-b20 fresh C0; declared reserve=480 seconds

Budget coordinator milestone: 2026-11-11 00:16:28 UTC: WATCH job=3912190 3912190 debug RUNNING 4 nid[006031,006108,007438,007453]

Budget result: 2026-11-11 00:16:29 UTC All 6 predeclared initial pairs accepted. 40/candidate-first r2 jobs3912096/3912187 passed; 14.12570s vs14.41466s at20, ratio0.97995374 (-2.0046%), shared allocation738s. 40 r1/r2 both faster, median ratio0.97849905, observed0.97704437..0.97995374, max within-leg CV0.01974761. Initial12 order disagreement and32 near baseline require third pairs; 40 does not trigger a third under predeclared rule. Controller accepted snapshot and queued12/reference-first r3 as3912190. After r3 for12/32, test lower8 and higher48 if competitive under actual adaptive rule. No global optimum conclusion.

Compute milestone: 2026-11-11 00:22:14 UTC pair_job=3912190 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b12-reference-first-r3-attempt1: AFTER all16 workers returned0 for leg=reference; BEFORE small health/all-step timing summaries

Compute milestone: 2026-11-11 00:22:22 UTC pair_job=3912190 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b12-reference-first-r3-attempt1: AFTER leg=0 role=reference budget=20 completed; case host wall recorded; strict paired acceptance pending posthoc

Compute milestone: 2026-11-11 00:22:23 UTC pair_job=3912190 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b12-reference-first-r3-attempt1: BEFORE leg=1 role=candidate budget=12 case=ep-overlap-budget-off-adaptive-20261110T223433Z-b12-reference-first-r3-attempt1-candidate-b12 fresh C0; declared reserve=480 seconds

Compute milestone: 2026-11-11 00:28:11 UTC pair_job=3912190 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b12-reference-first-r3-attempt1: AFTER all16 workers returned0 for leg=candidate; BEFORE small health/all-step timing summaries

Compute milestone: 2026-11-11 00:28:17 UTC pair_job=3912190 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b12-reference-first-r3-attempt1: AFTER leg=1 role=candidate budget=12 completed; case host wall recorded; strict paired acceptance pending posthoc

Compute milestone: 2026-11-11 00:28:17 UTC pair_job=3912190 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b12-reference-first-r3-attempt1: AFTER both independent cases acquired; release4nodes before1node posthoc; shared allocation elapsed is not a per-case time

Compute milestone: 2026-11-11 00:28:17 UTC pair_job=3912190 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b12-reference-first-r3-attempt1: AFTER pair phase=acquisition_complete exit=0 complete=1; preserve every case; partial pairs are not accepted

Budget coordinator milestone: 2026-11-11 00:28:44 UTC: BEFORE posthoc pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b12-reference-first-r3-attempt1 training=3912190 released; one debug node/up to90minutes

Budget coordinator milestone: 2026-11-11 00:28:49 UTC: AFTER posthoc submit job=3912210 training=3912190 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b12-reference-first-r3-attempt1; paired and decision gates pending

Budget coordinator milestone: 2026-11-11 00:28:49 UTC: WATCH job=3912210 3912210 debug PENDING 1 (None)

Compute milestone: 2026-11-11 00:29:00 UTC budget_posthoc=3912210 pair_training=3912190: BEFORE separate one-node debug analysis; pair allocation released; strict per-leg and paired placement/source/input gates; container=setup/gh200-cuda130-runtime.toml

Budget coordinator milestone: 2026-11-11 00:29:19 UTC: WATCH job=3912210 3912210 debug RUNNING 1 nid006903

Compute milestone: 2026-11-11 00:29:27 UTC budget_posthoc=3912210 pair_training=3912190: AFTER both leg validations and paired comparison returned0; actual16GPU equality, inputs and timing distinction checked

Compute milestone: 2026-11-11 00:29:27 UTC budget_posthoc=3912210 pair_training=3912190: AFTER paired posthoc exit=0 complete=1; preserve original cases and partial analysis

Budget coordinator milestone: 2026-11-11 00:30:08 UTC: ACCEPTED pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b12-reference-first-r3-attempt1 candidate=12 order=reference-first replicate=3 training=3912190 posthoc=3912210; total=7

Budget coordinator milestone: 2026-11-11 00:30:09 UTC: BEFORE training candidate=32 order=reference-first replicate=3 attempt=1; four debug nodes/21minutes; independent C0 per leg

Budget coordinator milestone: 2026-11-11 00:30:10 UTC: AFTER training submit job=3912211 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b32-reference-first-r3-attempt1; strict acceptance pending

Budget coordinator milestone: 2026-11-11 00:30:11 UTC: WATCH job=3912211 3912211 debug PENDING 4 (None)

Compute milestone: 2026-11-11 00:30:19 UTC pair_job=3912211 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b32-reference-first-r3-attempt1: BEFORE independent paired C0 acquisition candidate=32 order=reference-first steps=10 nodes=nid[006201,007438,007445,007453] container=setup/gh200-cuda130-runtime.toml source=252ef8a09ddb96efef52ceebf339a52287e2dc48422f0a9fa472f5d41fc76927

Compute milestone: 2026-11-11 00:30:20 UTC pair_job=3912211 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b32-reference-first-r3-attempt1: BEFORE leg=0 role=reference budget=20 case=ep-overlap-budget-off-adaptive-20261110T223433Z-b32-reference-first-r3-attempt1-reference-b20 fresh C0; declared reserve=480 seconds

Budget result: 2026-11-11 00:30:23 UTC Seven full pairs accepted. 12/reference-first r3 training3912190/posthoc3912210 COMPLETED0:0, allocation738s and posthoc36s. Mean14.77010s vs14.77634s at20; ratio0.99957770 (-0.04223%). Across3 ratios0.99691366,1.02475233,0.99957770 there is no consistent12 advantage. Controller selected32/reference-first r3 job3912211, then extends competitive boundaries per frozen decision. No new failures.

Budget coordinator milestone: 2026-11-11 00:30:43 UTC: WATCH job=3912211 3912211 debug RUNNING 4 nid[006201,007438,007445,007453]

Compute milestone: 2026-11-11 00:36:30 UTC pair_job=3912211 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b32-reference-first-r3-attempt1: AFTER all16 workers returned0 for leg=reference; BEFORE small health/all-step timing summaries

Compute milestone: 2026-11-11 00:36:39 UTC pair_job=3912211 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b32-reference-first-r3-attempt1: AFTER leg=0 role=reference budget=20 completed; case host wall recorded; strict paired acceptance pending posthoc

Compute milestone: 2026-11-11 00:36:40 UTC pair_job=3912211 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b32-reference-first-r3-attempt1: BEFORE leg=1 role=candidate budget=32 case=ep-overlap-budget-off-adaptive-20261110T223433Z-b32-reference-first-r3-attempt1-candidate-b32 fresh C0; declared reserve=480 seconds

Compute milestone: 2026-11-11 00:42:27 UTC pair_job=3912211 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b32-reference-first-r3-attempt1: AFTER all16 workers returned0 for leg=candidate; BEFORE small health/all-step timing summaries

Compute milestone: 2026-11-11 00:42:36 UTC pair_job=3912211 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b32-reference-first-r3-attempt1: AFTER leg=1 role=candidate budget=32 completed; case host wall recorded; strict paired acceptance pending posthoc

Compute milestone: 2026-11-11 00:42:36 UTC pair_job=3912211 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b32-reference-first-r3-attempt1: AFTER both independent cases acquired; release4nodes before1node posthoc; shared allocation elapsed is not a per-case time

Compute milestone: 2026-11-11 00:42:36 UTC pair_job=3912211 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b32-reference-first-r3-attempt1: AFTER pair phase=acquisition_complete exit=0 complete=1; preserve every case; partial pairs are not accepted

Budget coordinator milestone: 2026-11-11 00:42:52 UTC: BEFORE posthoc pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b32-reference-first-r3-attempt1 training=3912211 released; one debug node/up to90minutes

Budget coordinator milestone: 2026-11-11 00:42:53 UTC: AFTER posthoc submit job=3912229 training=3912211 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b32-reference-first-r3-attempt1; paired and decision gates pending

Budget coordinator milestone: 2026-11-11 00:42:53 UTC: WATCH job=3912229 3912229 debug PENDING 1 (None)

Compute milestone: 2026-11-11 00:43:02 UTC budget_posthoc=3912229 pair_training=3912211: BEFORE separate one-node debug analysis; pair allocation released; strict per-leg and paired placement/source/input gates; container=setup/gh200-cuda130-runtime.toml

Budget coordinator milestone: 2026-11-11 00:43:23 UTC: WATCH job=3912229 3912229 debug RUNNING 1 nid006201

Compute milestone: 2026-11-11 00:43:29 UTC budget_posthoc=3912229 pair_training=3912211: AFTER both leg validations and paired comparison returned0; actual16GPU equality, inputs and timing distinction checked

Compute milestone: 2026-11-11 00:43:29 UTC budget_posthoc=3912229 pair_training=3912211: AFTER paired posthoc exit=0 complete=1; preserve original cases and partial analysis

Budget coordinator milestone: 2026-11-11 00:45:15 UTC: ACCEPTED pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b32-reference-first-r3-attempt1 candidate=32 order=reference-first replicate=3 training=3912211 posthoc=3912229; total=8

Budget coordinator milestone: 2026-11-11 00:45:16 UTC: BEFORE training candidate=8 order=reference-first replicate=1 attempt=1; four debug nodes/21minutes; independent C0 per leg

Budget coordinator milestone: 2026-11-11 00:45:20 UTC: AFTER training submit job=3912231 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b8-reference-first-r1-attempt1; strict acceptance pending

Budget coordinator milestone: 2026-11-11 00:45:20 UTC: WATCH job=3912231 3912231 debug PENDING 4 (None)

Budget result: 2026-11-11 00:46:17 UTC Eight full pairs accepted. 32/reference-first r3 training3912211/posthoc3912229 COMPLETED0:0, 746s/37s. Means14.58262s vs14.65396s at20, ratio0.99513169 (-0.48683%). Across3 32 ratios1.00483502,1.00010948,0.99513169 no consistent gain. 40 remains best repeated (median0.97849905), 12 observed low0.99691366 is not >1.02*40 high0.97995374. Validated decision extends lower to8/reference-first r1 training3912231 pending4debug/21. Controller verified6287 bound inputs before append. Continue upper48 if competitive after lower evidence. One preserved full-attempt logging failure, replacement accepted.

Compute milestone: 2026-11-11 00:53:45 UTC pair_job=3912231 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b8-reference-first-r1-attempt1: BEFORE independent paired C0 acquisition candidate=8 order=reference-first steps=10 nodes=nid[007385,007391,007438,007445] container=setup/gh200-cuda130-runtime.toml source=252ef8a09ddb96efef52ceebf339a52287e2dc48422f0a9fa472f5d41fc76927

Compute milestone: 2026-11-11 00:53:48 UTC pair_job=3912231 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b8-reference-first-r1-attempt1: BEFORE leg=0 role=reference budget=20 case=ep-overlap-budget-off-adaptive-20261110T223433Z-b8-reference-first-r1-attempt1-reference-b20 fresh C0; declared reserve=480 seconds

Budget coordinator milestone: 2026-11-11 00:53:56 UTC: WATCH job=3912231 3912231 debug RUNNING 4 nid[007385,007391,007438,007445]

Compute milestone: 2026-11-11 00:59:45 UTC pair_job=3912231 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b8-reference-first-r1-attempt1: AFTER all16 workers returned0 for leg=reference; BEFORE small health/all-step timing summaries

Compute milestone: 2026-11-11 00:59:52 UTC pair_job=3912231 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b8-reference-first-r1-attempt1: AFTER leg=0 role=reference budget=20 completed; case host wall recorded; strict paired acceptance pending posthoc

Compute milestone: 2026-11-11 00:59:53 UTC pair_job=3912231 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b8-reference-first-r1-attempt1: BEFORE leg=1 role=candidate budget=8 case=ep-overlap-budget-off-adaptive-20261110T223433Z-b8-reference-first-r1-attempt1-candidate-b8 fresh C0; declared reserve=480 seconds

Compute milestone: 2026-11-11 01:05:38 UTC pair_job=3912231 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b8-reference-first-r1-attempt1: AFTER all16 workers returned0 for leg=candidate; BEFORE small health/all-step timing summaries

Compute milestone: 2026-11-11 01:05:47 UTC pair_job=3912231 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b8-reference-first-r1-attempt1: AFTER leg=1 role=candidate budget=8 completed; case host wall recorded; strict paired acceptance pending posthoc

Compute milestone: 2026-11-11 01:05:47 UTC pair_job=3912231 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b8-reference-first-r1-attempt1: AFTER both independent cases acquired; release4nodes before1node posthoc; shared allocation elapsed is not a per-case time

Compute milestone: 2026-11-11 01:05:47 UTC pair_job=3912231 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b8-reference-first-r1-attempt1: AFTER pair phase=acquisition_complete exit=0 complete=1; preserve every case; partial pairs are not accepted

Budget coordinator milestone: 2026-11-11 01:06:11 UTC: BEFORE posthoc pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b8-reference-first-r1-attempt1 training=3912231 released; one debug node/up to90minutes

Budget coordinator milestone: 2026-11-11 01:06:14 UTC: AFTER posthoc submit job=3912414 training=3912231 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b8-reference-first-r1-attempt1; paired and decision gates pending

Budget coordinator milestone: 2026-11-11 01:06:16 UTC: WATCH job=3912414 3912414 debug PENDING 1 (Priority)

Compute milestone: 2026-11-11 01:06:26 UTC budget_posthoc=3912414 pair_training=3912231: BEFORE separate one-node debug analysis; pair allocation released; strict per-leg and paired placement/source/input gates; container=setup/gh200-cuda130-runtime.toml

Budget coordinator milestone: 2026-11-11 01:06:46 UTC: WATCH job=3912414 3912414 debug RUNNING 1 nid007385

Compute milestone: 2026-11-11 01:06:55 UTC budget_posthoc=3912414 pair_training=3912231: AFTER both leg validations and paired comparison returned0; actual16GPU equality, inputs and timing distinction checked

Compute milestone: 2026-11-11 01:06:55 UTC budget_posthoc=3912414 pair_training=3912231: AFTER paired posthoc exit=0 complete=1; preserve original cases and partial analysis

Budget coordinator milestone: 2026-11-11 01:08:11 UTC: ACCEPTED pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b8-reference-first-r1-attempt1 candidate=8 order=reference-first replicate=1 training=3912231 posthoc=3912414; total=9

Budget coordinator milestone: 2026-11-11 01:08:11 UTC: BEFORE training candidate=8 order=candidate-first replicate=2 attempt=1; four debug nodes/21minutes; independent C0 per leg

Budget coordinator milestone: 2026-11-11 01:08:12 UTC: AFTER training submit job=3912419 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b8-candidate-first-r2-attempt1; strict acceptance pending

Budget coordinator milestone: 2026-11-11 01:08:12 UTC: WATCH job=3912419 3912419 debug PENDING 4 (Priority)

Budget result: 2026-11-11 01:09:15 UTC Nine full pairs accepted. First newlower8/reference-first r1 training3912231/posthoc3912414 passed (actual8 on all16 ranks); shared allocation731s. Measured14.31128s vs14.02338s at20, ratio1.02053000 (+2.0530%). Next8/candidate-first r2 job3912419 pending4debug/21 to establish repeatability before closinglower or extendingto4. Valid decision same-source, immutable ledger retained. Upper48 pendinglower evidence, not cancelled.

Compute milestone: 2026-11-11 01:09:58 UTC pair_job=3912419 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b8-candidate-first-r2-attempt1: BEFORE independent paired C0 acquisition candidate=8 order=candidate-first steps=10 nodes=nid[007125,007267,007445,007449] container=setup/gh200-cuda130-runtime.toml source=252ef8a09ddb96efef52ceebf339a52287e2dc48422f0a9fa472f5d41fc76927

Compute milestone: 2026-11-11 01:10:01 UTC pair_job=3912419 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b8-candidate-first-r2-attempt1: BEFORE leg=0 role=candidate budget=8 case=ep-overlap-budget-off-adaptive-20261110T223433Z-b8-candidate-first-r2-attempt1-candidate-b8 fresh C0; declared reserve=480 seconds

Budget coordinator milestone: 2026-11-11 01:10:16 UTC: WATCH job=3912419 3912419 debug RUNNING 4 nid[007125,007267,007445,007449]

Compute milestone: 2026-11-11 01:16:03 UTC pair_job=3912419 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b8-candidate-first-r2-attempt1: AFTER all16 workers returned0 for leg=candidate; BEFORE small health/all-step timing summaries

Compute milestone: 2026-11-11 01:16:12 UTC pair_job=3912419 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b8-candidate-first-r2-attempt1: AFTER leg=0 role=candidate budget=8 completed; case host wall recorded; strict paired acceptance pending posthoc

Compute milestone: 2026-11-11 01:16:12 UTC pair_job=3912419 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b8-candidate-first-r2-attempt1: BEFORE leg=1 role=reference budget=20 case=ep-overlap-budget-off-adaptive-20261110T223433Z-b8-candidate-first-r2-attempt1-reference-b20 fresh C0; declared reserve=480 seconds

Compute milestone: 2026-11-11 01:22:01 UTC pair_job=3912419 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b8-candidate-first-r2-attempt1: AFTER all16 workers returned0 for leg=reference; BEFORE small health/all-step timing summaries

Compute milestone: 2026-11-11 01:22:10 UTC pair_job=3912419 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b8-candidate-first-r2-attempt1: AFTER leg=1 role=reference budget=20 completed; case host wall recorded; strict paired acceptance pending posthoc

Compute milestone: 2026-11-11 01:22:10 UTC pair_job=3912419 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b8-candidate-first-r2-attempt1: AFTER both independent cases acquired; release4nodes before1node posthoc; shared allocation elapsed is not a per-case time

Compute milestone: 2026-11-11 01:22:10 UTC pair_job=3912419 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b8-candidate-first-r2-attempt1: AFTER pair phase=acquisition_complete exit=0 complete=1; preserve every case; partial pairs are not accepted

Budget coordinator milestone: 2026-11-11 01:22:26 UTC: WATCH job=3912419 3912419 debug COMPLETING 1 nid007449

Budget coordinator milestone: 2026-11-11 01:22:56 UTC: BEFORE posthoc pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b8-candidate-first-r2-attempt1 training=3912419 released; one debug node/up to90minutes

Budget coordinator milestone: 2026-11-11 01:22:57 UTC: AFTER posthoc submit job=3912444 training=3912419 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b8-candidate-first-r2-attempt1; paired and decision gates pending

Budget coordinator milestone: 2026-11-11 01:22:57 UTC: WATCH job=3912444 3912444 debug PENDING 1 (None)

Compute milestone: 2026-11-11 01:23:05 UTC budget_posthoc=3912444 pair_training=3912419: BEFORE separate one-node debug analysis; pair allocation released; strict per-leg and paired placement/source/input gates; container=setup/gh200-cuda130-runtime.toml

Budget coordinator milestone: 2026-11-11 01:23:27 UTC: WATCH job=3912444 3912444 debug RUNNING 1 nid007267

Compute milestone: 2026-11-11 01:23:36 UTC budget_posthoc=3912444 pair_training=3912419: AFTER both leg validations and paired comparison returned0; actual16GPU equality, inputs and timing distinction checked

Compute milestone: 2026-11-11 01:23:37 UTC budget_posthoc=3912444 pair_training=3912419: AFTER paired posthoc exit=0 complete=1; preserve original cases and partial analysis

Budget coordinator milestone: 2026-11-11 01:24:27 UTC: ACCEPTED pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b8-candidate-first-r2-attempt1 candidate=8 order=candidate-first replicate=2 training=3912419 posthoc=3912444; total=10

Budget coordinator milestone: 2026-11-11 01:24:27 UTC: BEFORE training candidate=8 order=reference-first replicate=3 attempt=1; four debug nodes/21minutes; independent C0 per leg

Budget coordinator milestone: 2026-11-11 01:24:28 UTC: AFTER training submit job=3912445 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b8-reference-first-r3-attempt1; strict acceptance pending

Budget coordinator milestone: 2026-11-11 01:24:28 UTC: WATCH job=3912445 3912445 debug PENDING 4 (None)

Compute milestone: 2026-11-11 01:24:39 UTC pair_job=3912445 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b8-reference-first-r3-attempt1: BEFORE independent paired C0 acquisition candidate=8 order=reference-first steps=10 nodes=nid[006035,006044,006073,006114] container=setup/gh200-cuda130-runtime.toml source=252ef8a09ddb96efef52ceebf339a52287e2dc48422f0a9fa472f5d41fc76927

Compute milestone: 2026-11-11 01:24:42 UTC pair_job=3912445 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b8-reference-first-r3-attempt1: BEFORE leg=0 role=reference budget=20 case=ep-overlap-budget-off-adaptive-20261110T223433Z-b8-reference-first-r3-attempt1-reference-b20 fresh C0; declared reserve=480 seconds

Budget coordinator milestone: 2026-11-11 01:24:59 UTC: WATCH job=3912445 3912445 debug RUNNING 4 nid[006035,006044,006073,006114]

Budget result: 2026-11-11 01:25:48 UTC Ten full pairs accepted. 8/candidate-first r2 training3912419/posthoc3912444 passed. Means14.49444s vs14.51232s at20, ratio0.99876794 (-0.12321%), shared allocation744s. R1 was1.02053000, so opposite signs and >2%relative spread trigger predeclared third8/reference-first r3 as3912445, RUNNING4debug/21. Do not marklower slowdownconfirmed. 40 remains current best median.

Compute milestone: 2026-11-11 01:30:50 UTC pair_job=3912445 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b8-reference-first-r3-attempt1: AFTER all16 workers returned0 for leg=reference; BEFORE small health/all-step timing summaries

Compute milestone: 2026-11-11 01:30:57 UTC pair_job=3912445 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b8-reference-first-r3-attempt1: AFTER leg=0 role=reference budget=20 completed; case host wall recorded; strict paired acceptance pending posthoc

Compute milestone: 2026-11-11 01:30:58 UTC pair_job=3912445 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b8-reference-first-r3-attempt1: BEFORE leg=1 role=candidate budget=8 case=ep-overlap-budget-off-adaptive-20261110T223433Z-b8-reference-first-r3-attempt1-candidate-b8 fresh C0; declared reserve=480 seconds

Compute milestone: 2026-11-11 01:36:58 UTC pair_job=3912445 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b8-reference-first-r3-attempt1: AFTER all16 workers returned0 for leg=candidate; BEFORE small health/all-step timing summaries

Compute milestone: 2026-11-11 01:37:05 UTC pair_job=3912445 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b8-reference-first-r3-attempt1: AFTER leg=1 role=candidate budget=8 completed; case host wall recorded; strict paired acceptance pending posthoc

Compute milestone: 2026-11-11 01:37:05 UTC pair_job=3912445 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b8-reference-first-r3-attempt1: AFTER both independent cases acquired; release4nodes before1node posthoc; shared allocation elapsed is not a per-case time

Compute milestone: 2026-11-11 01:37:05 UTC pair_job=3912445 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b8-reference-first-r3-attempt1: AFTER pair phase=acquisition_complete exit=0 complete=1; preserve every case; partial pairs are not accepted

Budget coordinator milestone: 2026-11-11 01:37:11 UTC: WATCH job=3912445 3912445 debug COMPLETING 4 nid[006035,006044,006073,006114]

Budget coordinator milestone: 2026-11-11 01:37:41 UTC: BEFORE posthoc pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b8-reference-first-r3-attempt1 training=3912445 released; one debug node/up to90minutes

Budget coordinator milestone: 2026-11-11 01:37:45 UTC: AFTER posthoc submit job=3912467 training=3912445 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b8-reference-first-r3-attempt1; paired and decision gates pending

Budget coordinator milestone: 2026-11-11 01:37:45 UTC: WATCH job=3912467 3912467 debug PENDING 1 (None)

Compute milestone: 2026-11-11 01:37:55 UTC budget_posthoc=3912467 pair_training=3912445: BEFORE separate one-node debug analysis; pair allocation released; strict per-leg and paired placement/source/input gates; container=setup/gh200-cuda130-runtime.toml

Budget coordinator milestone: 2026-11-11 01:38:16 UTC: WATCH job=3912467 3912467 debug RUNNING 1 nid005422

Compute milestone: 2026-11-11 01:38:28 UTC budget_posthoc=3912467 pair_training=3912445: AFTER both leg validations and paired comparison returned0; actual16GPU equality, inputs and timing distinction checked

Compute milestone: 2026-11-11 01:38:28 UTC budget_posthoc=3912467 pair_training=3912445: AFTER paired posthoc exit=0 complete=1; preserve original cases and partial analysis

Budget coordinator milestone: 2026-11-11 01:39:21 UTC: ACCEPTED pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b8-reference-first-r3-attempt1 candidate=8 order=reference-first replicate=3 training=3912445 posthoc=3912467; total=11

Budget coordinator milestone: 2026-11-11 01:39:21 UTC: BEFORE training candidate=4 order=reference-first replicate=1 attempt=1; four debug nodes/21minutes; independent C0 per leg

Budget coordinator milestone: 2026-11-11 01:39:23 UTC: AFTER training submit job=3912476 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b4-reference-first-r1-attempt1; strict acceptance pending

Budget coordinator milestone: 2026-11-11 01:39:25 UTC: WATCH job=3912476 3912476 debug PENDING 4 (None)

Budget result: 2026-11-11 01:40:19 UTC Eleven full pairs accepted. 8/reference-first r3 training3912445/posthoc3912467 passed (756s trainingallocation). Means15.68190s vs15.08938s at20, ratio1.03926735 (+3.9267%). Three8ratios1.02053000,0.99876794,1.03926735, median1.02053000. Two slower but min0.99876794 is <=1.02*40max0.97995374, so conservative range rule still extends to4. Valid nextdecision4/reference-first r1 queued3912476 debug4/21. Four is native lowest supported budget (internode.cu assert channels*2>3), not a claim of best. Upper48 follows lower boundary.

Budget coordinator milestone: 2026-11-11 01:40:56 UTC: WATCH job=3912476 3912476 debug RUNNING 4 nid[006664,006907,006980,007561]

Compute milestone: 2026-11-11 01:40:59 UTC pair_job=3912476 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b4-reference-first-r1-attempt1: BEFORE independent paired C0 acquisition candidate=4 order=reference-first steps=10 nodes=nid[006664,006907,006980,007561] container=setup/gh200-cuda130-runtime.toml source=252ef8a09ddb96efef52ceebf339a52287e2dc48422f0a9fa472f5d41fc76927

Compute milestone: 2026-11-11 01:41:00 UTC pair_job=3912476 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b4-reference-first-r1-attempt1: BEFORE leg=0 role=reference budget=20 case=ep-overlap-budget-off-adaptive-20261110T223433Z-b4-reference-first-r1-attempt1-reference-b20 fresh C0; declared reserve=480 seconds

Compute milestone: 2026-11-11 01:46:40 UTC pair_job=3912476 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b4-reference-first-r1-attempt1: AFTER all16 workers returned0 for leg=reference; BEFORE small health/all-step timing summaries

Compute milestone: 2026-11-11 01:46:49 UTC pair_job=3912476 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b4-reference-first-r1-attempt1: AFTER leg=0 role=reference budget=20 completed; case host wall recorded; strict paired acceptance pending posthoc

Compute milestone: 2026-11-11 01:46:50 UTC pair_job=3912476 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b4-reference-first-r1-attempt1: BEFORE leg=1 role=candidate budget=4 case=ep-overlap-budget-off-adaptive-20261110T223433Z-b4-reference-first-r1-attempt1-candidate-b4 fresh C0; declared reserve=480 seconds

Compute milestone: 2026-11-11 01:52:24 UTC pair_job=3912476 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b4-reference-first-r1-attempt1: AFTER all16 workers returned0 for leg=candidate; BEFORE small health/all-step timing summaries

Compute milestone: 2026-11-11 01:52:33 UTC pair_job=3912476 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b4-reference-first-r1-attempt1: AFTER leg=1 role=candidate budget=4 completed; case host wall recorded; strict paired acceptance pending posthoc

Compute milestone: 2026-11-11 01:52:34 UTC pair_job=3912476 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b4-reference-first-r1-attempt1: AFTER both independent cases acquired; release4nodes before1node posthoc; shared allocation elapsed is not a per-case time

Compute milestone: 2026-11-11 01:52:34 UTC pair_job=3912476 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b4-reference-first-r1-attempt1: AFTER pair phase=acquisition_complete exit=0 complete=1; preserve every case; partial pairs are not accepted

Budget coordinator milestone: 2026-11-11 01:52:38 UTC: WATCH job=3912476 3912476 debug COMPLETING 4 nid[006664,006907,006980,007561]

Budget coordinator milestone: 2026-11-11 01:53:09 UTC: BEFORE posthoc pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b4-reference-first-r1-attempt1 training=3912476 released; one debug node/up to90minutes

Budget coordinator milestone: 2026-11-11 01:53:12 UTC: AFTER posthoc submit job=3912494 training=3912476 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b4-reference-first-r1-attempt1; paired and decision gates pending

Budget coordinator milestone: 2026-11-11 01:53:14 UTC: WATCH job=3912494 3912494 debug PENDING 1 (None)

Compute milestone: 2026-11-11 01:53:24 UTC budget_posthoc=3912494 pair_training=3912476: BEFORE separate one-node debug analysis; pair allocation released; strict per-leg and paired placement/source/input gates; container=setup/gh200-cuda130-runtime.toml

Budget coordinator milestone: 2026-11-11 01:53:44 UTC: WATCH job=3912494 3912494 debug RUNNING 1 nid007145

Compute milestone: 2026-11-11 01:53:58 UTC budget_posthoc=3912494 pair_training=3912476: AFTER both leg validations and paired comparison returned0; actual16GPU equality, inputs and timing distinction checked

Compute milestone: 2026-11-11 01:53:58 UTC budget_posthoc=3912494 pair_training=3912476: AFTER paired posthoc exit=0 complete=1; preserve original cases and partial analysis

Budget coordinator milestone: 2026-11-11 01:54:55 UTC: ACCEPTED pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b4-reference-first-r1-attempt1 candidate=4 order=reference-first replicate=1 training=3912476 posthoc=3912494; total=12

Budget coordinator milestone: 2026-11-11 01:54:55 UTC: BEFORE training candidate=4 order=candidate-first replicate=2 attempt=1; four debug nodes/21minutes; independent C0 per leg

Budget coordinator milestone: 2026-11-11 01:54:58 UTC: AFTER training submit job=3912495 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b4-candidate-first-r2-attempt1; strict acceptance pending

Budget coordinator milestone: 2026-11-11 01:54:58 UTC: WATCH job=3912495 3912495 debug PENDING 4 (Resources)

Budget result: 2026-11-11 01:55:32 UTC Twelve full pairs accepted. First4/reference-first r1 training3912476/posthoc3912494 passed; actual4 recorded onall16 ranks. Means14.07996s vs13.26308s at20, ratio1.06159052 (+6.1591%), allocation703s. Strong lower slowdown inonepair; reverse-order4/candidate-first r2 job3912495 pending4debug/21 beforeconclusion. Upper48 next afterlowerbound.

Compute milestone: 2026-11-11 01:57:12 UTC pair_job=3912495 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b4-candidate-first-r2-attempt1: BEFORE independent paired C0 acquisition candidate=4 order=candidate-first steps=10 nodes=nid[007140,007222,007404,007496] container=setup/gh200-cuda130-runtime.toml source=252ef8a09ddb96efef52ceebf339a52287e2dc48422f0a9fa472f5d41fc76927

Compute milestone: 2026-11-11 01:57:16 UTC pair_job=3912495 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b4-candidate-first-r2-attempt1: BEFORE leg=0 role=candidate budget=4 case=ep-overlap-budget-off-adaptive-20261110T223433Z-b4-candidate-first-r2-attempt1-candidate-b4 fresh C0; declared reserve=480 seconds

Budget coordinator milestone: 2026-11-11 01:57:31 UTC: WATCH job=3912495 3912495 debug RUNNING 4 nid[007140,007222,007404,007496]

Compute milestone: 2026-11-11 02:03:31 UTC pair_job=3912495 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b4-candidate-first-r2-attempt1: AFTER all16 workers returned0 for leg=candidate; BEFORE small health/all-step timing summaries

Compute milestone: 2026-11-11 02:03:39 UTC pair_job=3912495 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b4-candidate-first-r2-attempt1: AFTER leg=0 role=candidate budget=4 completed; case host wall recorded; strict paired acceptance pending posthoc

Compute milestone: 2026-11-11 02:03:41 UTC pair_job=3912495 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b4-candidate-first-r2-attempt1: BEFORE leg=1 role=reference budget=20 case=ep-overlap-budget-off-adaptive-20261110T223433Z-b4-candidate-first-r2-attempt1-reference-b20 fresh C0; declared reserve=480 seconds

Compute milestone: 2026-11-11 02:09:26 UTC pair_job=3912495 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b4-candidate-first-r2-attempt1: AFTER all16 workers returned0 for leg=reference; BEFORE small health/all-step timing summaries

Compute milestone: 2026-11-11 02:09:36 UTC pair_job=3912495 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b4-candidate-first-r2-attempt1: AFTER leg=1 role=reference budget=20 completed; case host wall recorded; strict paired acceptance pending posthoc

Compute milestone: 2026-11-11 02:09:37 UTC pair_job=3912495 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b4-candidate-first-r2-attempt1: AFTER both independent cases acquired; release4nodes before1node posthoc; shared allocation elapsed is not a per-case time

Compute milestone: 2026-11-11 02:09:37 UTC pair_job=3912495 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b4-candidate-first-r2-attempt1: AFTER pair phase=acquisition_complete exit=0 complete=1; preserve every case; partial pairs are not accepted

Budget coordinator milestone: 2026-11-11 02:09:43 UTC: WATCH job=3912495 3912495 debug COMPLETING 4 nid[007140,007222,007404,007496]

Budget coordinator milestone: 2026-11-11 02:10:13 UTC: BEFORE posthoc pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b4-candidate-first-r2-attempt1 training=3912495 released; one debug node/up to90minutes

Budget coordinator milestone: 2026-11-11 02:10:14 UTC: AFTER posthoc submit job=3912528 training=3912495 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b4-candidate-first-r2-attempt1; paired and decision gates pending

Budget coordinator milestone: 2026-11-11 02:10:14 UTC: WATCH job=3912528 3912528 debug PENDING 1 (None)

Compute milestone: 2026-11-11 02:10:26 UTC budget_posthoc=3912528 pair_training=3912495: BEFORE separate one-node debug analysis; pair allocation released; strict per-leg and paired placement/source/input gates; container=setup/gh200-cuda130-runtime.toml

Budget coordinator milestone: 2026-11-11 02:10:44 UTC: WATCH job=3912528 3912528 debug RUNNING 1 nid007404

Compute milestone: 2026-11-11 02:11:02 UTC budget_posthoc=3912528 pair_training=3912495: AFTER both leg validations and paired comparison returned0; actual16GPU equality, inputs and timing distinction checked

Compute milestone: 2026-11-11 02:11:02 UTC budget_posthoc=3912528 pair_training=3912495: AFTER paired posthoc exit=0 complete=1; preserve original cases and partial analysis

Budget coordinator milestone: 2026-11-11 02:11:14 UTC: WATCH job=3912528 3912528 debug COMPLETING 1 nid007404

Budget coordinator milestone: 2026-11-11 02:12:21 UTC: ACCEPTED pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b4-candidate-first-r2-attempt1 candidate=4 order=candidate-first replicate=2 training=3912495 posthoc=3912528; total=13

Budget coordinator milestone: 2026-11-11 02:12:22 UTC: BEFORE training candidate=4 order=reference-first replicate=3 attempt=1; four debug nodes/21minutes; independent C0 per leg

Budget coordinator milestone: 2026-11-11 02:12:22 UTC: AFTER training submit job=3912544 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b4-reference-first-r3-attempt1; strict acceptance pending

Budget coordinator milestone: 2026-11-11 02:12:22 UTC: WATCH job=3912544 3912544 debug PENDING 4 (None)

Budget result: 2026-11-11 02:12:41 UTC Thirteen full pairs accepted. 4/candidate-first r2 training3912495/posthoc3912528 passed; means14.94344s vs14.58956s at20, ratio1.02425570 (+2.42557%), allocation755s. Both4runs slower (r1+6.1591%), but >2%relative ratio spread triggers third4/reference-first r3 job3912544 pending4debug/21. Four is lowerimplementationlimit, so no furtherlowerbudget; afterthird gohigher48 unless source/validation failure. Current best40 median0.97849905.

Budget coordinator milestone: 2026-11-11 02:13:55 UTC: WATCH job=3912544 3912544 debug RUNNING 4 nid[006266,006626,006666,006668]

Compute milestone: 2026-11-11 02:13:57 UTC pair_job=3912544 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b4-reference-first-r3-attempt1: BEFORE independent paired C0 acquisition candidate=4 order=reference-first steps=10 nodes=nid[006266,006626,006666,006668] container=setup/gh200-cuda130-runtime.toml source=252ef8a09ddb96efef52ceebf339a52287e2dc48422f0a9fa472f5d41fc76927

Compute milestone: 2026-11-11 02:14:00 UTC pair_job=3912544 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b4-reference-first-r3-attempt1: BEFORE leg=0 role=reference budget=20 case=ep-overlap-budget-off-adaptive-20261110T223433Z-b4-reference-first-r3-attempt1-reference-b20 fresh C0; declared reserve=480 seconds

Compute milestone: 2026-11-11 02:20:01 UTC pair_job=3912544 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b4-reference-first-r3-attempt1: AFTER all16 workers returned0 for leg=reference; BEFORE small health/all-step timing summaries

Compute milestone: 2026-11-11 02:20:08 UTC pair_job=3912544 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b4-reference-first-r3-attempt1: AFTER leg=0 role=reference budget=20 completed; case host wall recorded; strict paired acceptance pending posthoc

Compute milestone: 2026-11-11 02:20:09 UTC pair_job=3912544 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b4-reference-first-r3-attempt1: BEFORE leg=1 role=candidate budget=4 case=ep-overlap-budget-off-adaptive-20261110T223433Z-b4-reference-first-r3-attempt1-candidate-b4 fresh C0; declared reserve=480 seconds

Compute milestone: 2026-11-11 02:26:04 UTC pair_job=3912544 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b4-reference-first-r3-attempt1: AFTER all16 workers returned0 for leg=candidate; BEFORE small health/all-step timing summaries

Compute milestone: 2026-11-11 02:26:12 UTC pair_job=3912544 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b4-reference-first-r3-attempt1: AFTER leg=1 role=candidate budget=4 completed; case host wall recorded; strict paired acceptance pending posthoc

Compute milestone: 2026-11-11 02:26:12 UTC pair_job=3912544 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b4-reference-first-r3-attempt1: AFTER both independent cases acquired; release4nodes before1node posthoc; shared allocation elapsed is not a per-case time

Compute milestone: 2026-11-11 02:26:12 UTC pair_job=3912544 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b4-reference-first-r3-attempt1: AFTER pair phase=acquisition_complete exit=0 complete=1; preserve every case; partial pairs are not accepted

Budget coordinator milestone: 2026-11-11 02:26:32 UTC: BEFORE posthoc pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b4-reference-first-r3-attempt1 training=3912544 released; one debug node/up to90minutes

Budget coordinator milestone: 2026-11-11 02:26:34 UTC: AFTER posthoc submit job=3912624 training=3912544 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b4-reference-first-r3-attempt1; paired and decision gates pending

Budget coordinator milestone: 2026-11-11 02:26:34 UTC: WATCH job=3912624 3912624 debug PENDING 1 (Priority)

Compute milestone: 2026-11-11 02:26:46 UTC budget_posthoc=3912624 pair_training=3912544: BEFORE separate one-node debug analysis; pair allocation released; strict per-leg and paired placement/source/input gates; container=setup/gh200-cuda130-runtime.toml

Budget coordinator milestone: 2026-11-11 02:27:04 UTC: WATCH job=3912624 3912624 debug RUNNING 1 nid006937

Compute milestone: 2026-11-11 02:27:23 UTC budget_posthoc=3912624 pair_training=3912544: AFTER both leg validations and paired comparison returned0; actual16GPU equality, inputs and timing distinction checked

Compute milestone: 2026-11-11 02:27:23 UTC budget_posthoc=3912624 pair_training=3912544: AFTER paired posthoc exit=0 complete=1; preserve original cases and partial analysis

Budget coordinator milestone: 2026-11-11 02:27:34 UTC: WATCH job=3912624 3912624 debug COMPLETING 1 nid006937

Budget coordinator milestone: 2026-11-11 02:28:48 UTC: ACCEPTED pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b4-reference-first-r3-attempt1 candidate=4 order=reference-first replicate=3 training=3912544 posthoc=3912624; total=14

Budget coordinator milestone: 2026-11-11 02:28:48 UTC: BEFORE training candidate=48 order=reference-first replicate=1 attempt=1; four debug nodes/21minutes; independent C0 per leg

Budget coordinator milestone: 2026-11-11 02:28:49 UTC: AFTER training submit job=3912805 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b48-reference-first-r1-attempt1; strict acceptance pending

Budget coordinator milestone: 2026-11-11 02:28:49 UTC: WATCH job=3912805 3912805 debug PENDING 4 (Priority)

Compute milestone: 2026-11-11 02:28:59 UTC pair_job=3912805 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b48-reference-first-r1-attempt1: BEFORE independent paired C0 acquisition candidate=48 order=reference-first steps=10 nodes=nid[006929,006937,007368,007456] container=setup/gh200-cuda130-runtime.toml source=252ef8a09ddb96efef52ceebf339a52287e2dc48422f0a9fa472f5d41fc76927

Compute milestone: 2026-11-11 02:29:03 UTC pair_job=3912805 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b48-reference-first-r1-attempt1: BEFORE leg=0 role=reference budget=20 case=ep-overlap-budget-off-adaptive-20261110T223433Z-b48-reference-first-r1-attempt1-reference-b20 fresh C0; declared reserve=480 seconds

Budget coordinator milestone: 2026-11-11 02:29:19 UTC: WATCH job=3912805 3912805 debug RUNNING 4 nid[006929,006937,007368,007456]

Budget result: 2026-11-11 02:30:14 UTC Fourteen full pairs accepted. Final4/reference-first r3 training3912544/posthoc3912624 passed,745s/46s. Means15.10988s vs14.52724s at20, ratio1.04010672 (+4.01067%). Allthree4ratios1.06159052,1.02425570,1.04010672: lower boundary repeatably worse relative to best40 under declared2%range rule. Controller selected48/reference-first r1,3912805 RUNNING4debug/21, source252/D3 unchanged. Upperboundary work remains; finalsummaryQA7/synthesis and independent review pending.

Compute milestone: 2026-11-11 02:34:54 UTC pair_job=3912805 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b48-reference-first-r1-attempt1: AFTER all16 workers returned0 for leg=reference; BEFORE small health/all-step timing summaries

Compute milestone: 2026-11-11 02:35:01 UTC pair_job=3912805 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b48-reference-first-r1-attempt1: AFTER leg=0 role=reference budget=20 completed; case host wall recorded; strict paired acceptance pending posthoc

Compute milestone: 2026-11-11 02:35:02 UTC pair_job=3912805 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b48-reference-first-r1-attempt1: BEFORE leg=1 role=candidate budget=48 case=ep-overlap-budget-off-adaptive-20261110T223433Z-b48-reference-first-r1-attempt1-candidate-b48 fresh C0; declared reserve=480 seconds

Compute milestone: 2026-11-11 02:40:37 UTC pair_job=3912805 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b48-reference-first-r1-attempt1: AFTER all16 workers returned0 for leg=candidate; BEFORE small health/all-step timing summaries

Compute milestone: 2026-11-11 02:40:43 UTC pair_job=3912805 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b48-reference-first-r1-attempt1: AFTER leg=1 role=candidate budget=48 completed; case host wall recorded; strict paired acceptance pending posthoc

Compute milestone: 2026-11-11 02:40:43 UTC pair_job=3912805 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b48-reference-first-r1-attempt1: AFTER both independent cases acquired; release4nodes before1node posthoc; shared allocation elapsed is not a per-case time

Compute milestone: 2026-11-11 02:40:43 UTC pair_job=3912805 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b48-reference-first-r1-attempt1: AFTER pair phase=acquisition_complete exit=0 complete=1; preserve every case; partial pairs are not accepted

Budget coordinator milestone: 2026-11-11 02:40:59 UTC: WATCH job=3912805 3912805 debug COMPLETING 1 nid006937

Budget coordinator milestone: 2026-11-11 02:41:29 UTC: BEFORE posthoc pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b48-reference-first-r1-attempt1 training=3912805 released; one debug node/up to90minutes

Budget coordinator milestone: 2026-11-11 02:41:31 UTC: AFTER posthoc submit job=3912930 training=3912805 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b48-reference-first-r1-attempt1; paired and decision gates pending

Budget coordinator milestone: 2026-11-11 02:41:31 UTC: WATCH job=3912930 3912930 debug PENDING 1 (Priority)

Compute milestone: 2026-11-11 02:41:40 UTC budget_posthoc=3912930 pair_training=3912805: BEFORE separate one-node debug analysis; pair allocation released; strict per-leg and paired placement/source/input gates; container=setup/gh200-cuda130-runtime.toml

Budget coordinator milestone: 2026-11-11 02:42:01 UTC: WATCH job=3912930 3912930 debug RUNNING 1 nid006929

Compute milestone: 2026-11-11 02:42:20 UTC budget_posthoc=3912930 pair_training=3912805: AFTER both leg validations and paired comparison returned0; actual16GPU equality, inputs and timing distinction checked

Compute milestone: 2026-11-11 02:42:20 UTC budget_posthoc=3912930 pair_training=3912805: AFTER paired posthoc exit=0 complete=1; preserve original cases and partial analysis

Budget coordinator milestone: 2026-11-11 02:43:14 UTC: ACCEPTED pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b48-reference-first-r1-attempt1 candidate=48 order=reference-first replicate=1 training=3912805 posthoc=3912930; total=15

Budget coordinator milestone: 2026-11-11 02:43:14 UTC: BEFORE training candidate=48 order=candidate-first replicate=2 attempt=1; four debug nodes/21minutes; independent C0 per leg

Budget coordinator milestone: 2026-11-11 02:43:15 UTC: AFTER training submit job=3912931 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b48-candidate-first-r2-attempt1; strict acceptance pending

Budget coordinator milestone: 2026-11-11 02:43:15 UTC: WATCH job=3912931 3912931 debug PENDING 4 (Priority)

Compute milestone: 2026-11-11 02:43:25 UTC pair_job=3912931 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b48-candidate-first-r2-attempt1: BEFORE independent paired C0 acquisition candidate=48 order=candidate-first steps=10 nodes=nid[007509,007512,007514,007516] container=setup/gh200-cuda130-runtime.toml source=252ef8a09ddb96efef52ceebf339a52287e2dc48422f0a9fa472f5d41fc76927

Compute milestone: 2026-11-11 02:43:28 UTC pair_job=3912931 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b48-candidate-first-r2-attempt1: BEFORE leg=0 role=candidate budget=48 case=ep-overlap-budget-off-adaptive-20261110T223433Z-b48-candidate-first-r2-attempt1-candidate-b48 fresh C0; declared reserve=480 seconds

Budget coordinator milestone: 2026-11-11 02:43:45 UTC: WATCH job=3912931 3912931 debug RUNNING 4 nid[007509,007512,007514,007516]

Budget result: 2026-11-11 02:44:31 UTC Fifteen full pairs accepted. Firstupper48/reference-first r1 training3912805/posthoc3912930 passed,actual48onall16ranks. Means13.73450s vs13.92952s at20, ratio0.98599952 (-1.40005%), allocation714s. Stillcompetitive relative40bestmedian0.97849905. Reverse48/candidate-first r2 job3912931 RUNNING4debug/21. Needreplication/possiblethird before56extension or upperworse decision. Lower4boundaryalreadyrepeatablyworse.

Compute milestone: 2026-11-11 02:49:31 UTC pair_job=3912931 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b48-candidate-first-r2-attempt1: AFTER all16 workers returned0 for leg=candidate; BEFORE small health/all-step timing summaries

Compute milestone: 2026-11-11 02:49:39 UTC pair_job=3912931 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b48-candidate-first-r2-attempt1: AFTER leg=0 role=candidate budget=48 completed; case host wall recorded; strict paired acceptance pending posthoc

Compute milestone: 2026-11-11 02:49:40 UTC pair_job=3912931 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b48-candidate-first-r2-attempt1: BEFORE leg=1 role=reference budget=20 case=ep-overlap-budget-off-adaptive-20261110T223433Z-b48-candidate-first-r2-attempt1-reference-b20 fresh C0; declared reserve=480 seconds

Compute milestone: 2026-11-11 02:55:38 UTC pair_job=3912931 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b48-candidate-first-r2-attempt1: AFTER all16 workers returned0 for leg=reference; BEFORE small health/all-step timing summaries

Compute milestone: 2026-11-11 02:55:46 UTC pair_job=3912931 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b48-candidate-first-r2-attempt1: AFTER leg=1 role=reference budget=20 completed; case host wall recorded; strict paired acceptance pending posthoc

Compute milestone: 2026-11-11 02:55:46 UTC pair_job=3912931 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b48-candidate-first-r2-attempt1: AFTER both independent cases acquired; release4nodes before1node posthoc; shared allocation elapsed is not a per-case time

Compute milestone: 2026-11-11 02:55:46 UTC pair_job=3912931 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b48-candidate-first-r2-attempt1: AFTER pair phase=acquisition_complete exit=0 complete=1; preserve every case; partial pairs are not accepted

Budget coordinator milestone: 2026-11-11 02:55:56 UTC: WATCH job=3912931 3912931 debug COMPLETING 4 nid[007509,007512,007514,007516]

Budget coordinator milestone: 2026-11-11 02:56:26 UTC: BEFORE posthoc pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b48-candidate-first-r2-attempt1 training=3912931 released; one debug node/up to90minutes

Budget coordinator milestone: 2026-11-11 02:56:27 UTC: AFTER posthoc submit job=3912968 training=3912931 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b48-candidate-first-r2-attempt1; paired and decision gates pending

Budget coordinator milestone: 2026-11-11 02:56:27 UTC: WATCH job=3912968 3912968 debug PENDING 1 (Priority)

Compute milestone: 2026-11-11 02:56:37 UTC budget_posthoc=3912968 pair_training=3912931: BEFORE separate one-node debug analysis; pair allocation released; strict per-leg and paired placement/source/input gates; container=setup/gh200-cuda130-runtime.toml

Budget coordinator milestone: 2026-11-11 02:56:57 UTC: WATCH job=3912968 3912968 debug RUNNING 1 nid007517

Compute milestone: 2026-11-11 02:57:19 UTC budget_posthoc=3912968 pair_training=3912931: AFTER both leg validations and paired comparison returned0; actual16GPU equality, inputs and timing distinction checked

Compute milestone: 2026-11-11 02:57:19 UTC budget_posthoc=3912968 pair_training=3912931: AFTER paired posthoc exit=0 complete=1; preserve original cases and partial analysis

Budget coordinator milestone: 2026-11-11 02:57:27 UTC: WATCH job=3912968 3912968 debug COMPLETING 1 nid007517

Budget coordinator milestone: 2026-11-11 02:58:42 UTC: ACCEPTED pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b48-candidate-first-r2-attempt1 candidate=48 order=candidate-first replicate=2 training=3912931 posthoc=3912968; total=16

Budget coordinator milestone: 2026-11-11 02:58:42 UTC: BEFORE training candidate=48 order=reference-first replicate=3 attempt=1; four debug nodes/21minutes; independent C0 per leg

Budget coordinator milestone: 2026-11-11 02:58:43 UTC: AFTER training submit job=3912973 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b48-reference-first-r3-attempt1; strict acceptance pending

Budget coordinator milestone: 2026-11-11 02:58:45 UTC: WATCH job=3912973 3912973 debug PENDING 4 (Priority)

Budget result: 2026-11-11 02:59:39 UTC Sixteen full pairs accepted. 48/candidate-first r2 training3912931/posthoc3912968 passed. Means14.59022s vs14.73496s at20, ratio0.99017710 (-0.98229%), allocation750s. Two48ratios0.98599952,0.99017710, median0.98808831, nearbaseline<2% triggers declared third48/reference-first r3 job3912973 pending4debug/21. 40 stillbestmedian0.97849905;48notrepeatablyworse. Nextqualifiedupper56 afterthirdifcompetitive. Lowerboundaryclosedat4.

Compute milestone: 2026-11-11 03:01:33 UTC pair_job=3912973 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b48-reference-first-r3-attempt1: BEFORE independent paired C0 acquisition candidate=48 order=reference-first steps=10 nodes=nid[006681,006751,006754,006941] container=setup/gh200-cuda130-runtime.toml source=252ef8a09ddb96efef52ceebf339a52287e2dc48422f0a9fa472f5d41fc76927

Compute milestone: 2026-11-11 03:01:36 UTC pair_job=3912973 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b48-reference-first-r3-attempt1: BEFORE leg=0 role=reference budget=20 case=ep-overlap-budget-off-adaptive-20261110T223433Z-b48-reference-first-r3-attempt1-reference-b20 fresh C0; declared reserve=480 seconds

Budget coordinator milestone: 2026-11-11 03:01:45 UTC: WATCH job=3912973 3912973 debug RUNNING 4 nid[006681,006751,006754,006941]

Compute milestone: 2026-11-11 03:07:26 UTC pair_job=3912973 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b48-reference-first-r3-attempt1: AFTER all16 workers returned0 for leg=reference; BEFORE small health/all-step timing summaries

Compute milestone: 2026-11-11 03:07:33 UTC pair_job=3912973 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b48-reference-first-r3-attempt1: AFTER leg=0 role=reference budget=20 completed; case host wall recorded; strict paired acceptance pending posthoc

Compute milestone: 2026-11-11 03:07:34 UTC pair_job=3912973 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b48-reference-first-r3-attempt1: BEFORE leg=1 role=candidate budget=48 case=ep-overlap-budget-off-adaptive-20261110T223433Z-b48-reference-first-r3-attempt1-candidate-b48 fresh C0; declared reserve=480 seconds

Compute milestone: 2026-11-11 03:13:08 UTC pair_job=3912973 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b48-reference-first-r3-attempt1: AFTER all16 workers returned0 for leg=candidate; BEFORE small health/all-step timing summaries

Compute milestone: 2026-11-11 03:13:16 UTC pair_job=3912973 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b48-reference-first-r3-attempt1: AFTER leg=1 role=candidate budget=48 completed; case host wall recorded; strict paired acceptance pending posthoc

Compute milestone: 2026-11-11 03:13:16 UTC pair_job=3912973 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b48-reference-first-r3-attempt1: AFTER both independent cases acquired; release4nodes before1node posthoc; shared allocation elapsed is not a per-case time

Compute milestone: 2026-11-11 03:13:16 UTC pair_job=3912973 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b48-reference-first-r3-attempt1: AFTER pair phase=acquisition_complete exit=0 complete=1; preserve every case; partial pairs are not accepted

Budget coordinator milestone: 2026-11-11 03:13:26 UTC: WATCH job=3912973 3912973 debug COMPLETING 1 nid006681

Budget coordinator milestone: 2026-11-11 03:13:56 UTC: BEFORE posthoc pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b48-reference-first-r3-attempt1 training=3912973 released; one debug node/up to90minutes

Budget coordinator milestone: 2026-11-11 03:13:57 UTC: AFTER posthoc submit job=3913001 training=3912973 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b48-reference-first-r3-attempt1; paired and decision gates pending

Budget coordinator milestone: 2026-11-11 03:13:57 UTC: WATCH job=3913001 3913001 debug PENDING 1 (Priority)

Compute milestone: 2026-11-11 03:15:18 UTC budget_posthoc=3913001 pair_training=3912973: BEFORE separate one-node debug analysis; pair allocation released; strict per-leg and paired placement/source/input gates; container=setup/gh200-cuda130-runtime.toml

Budget coordinator milestone: 2026-11-11 03:15:28 UTC: WATCH job=3913001 3913001 debug RUNNING 1 nid007463

Compute milestone: 2026-11-11 03:16:01 UTC budget_posthoc=3913001 pair_training=3912973: AFTER both leg validations and paired comparison returned0; actual16GPU equality, inputs and timing distinction checked

Compute milestone: 2026-11-11 03:16:01 UTC budget_posthoc=3913001 pair_training=3912973: AFTER paired posthoc exit=0 complete=1; preserve original cases and partial analysis

Budget coordinator milestone: 2026-11-11 03:17:27 UTC: ACCEPTED pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b48-reference-first-r3-attempt1 candidate=48 order=reference-first replicate=3 training=3912973 posthoc=3913001; total=17

Budget coordinator milestone: 2026-11-11 03:17:28 UTC: BEFORE training candidate=56 order=reference-first replicate=1 attempt=1; four debug nodes/21minutes; independent C0 per leg

Budget coordinator milestone: 2026-11-11 03:17:33 UTC: AFTER training submit job=3913007 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b56-reference-first-r1-attempt1; strict acceptance pending

Budget coordinator milestone: 2026-11-11 03:17:33 UTC: WATCH job=3913007 3913007 debug PENDING 4 (Priority)

Budget result: 2026-11-11 03:17:52 UTC Seventeen full pairs accepted. Final48/reference-first r3 training3912973/posthoc3913001 passed; means13.17042s vs13.17684s at20, ratio0.99951278 (-0.04872%), allocation712s. Three48ratios0.98599952,0.99017710,0.99951278; median0.99017710. Notrepeatablyworse than40, so validdecisionextends56/reference-first r1 job3913007 pending4debug/21. This isactualqualifiedupperresourcebound;58needs237568bytes>232448available. Need56reversedpairs/possibler3, then boundedcompletion+summaryQA7/synthesis/independentreview.

Budget coordinator milestone: 2026-11-11 03:24:39 UTC: WATCH job=3913007 3913007 debug RUNNING 4 nid[006941,007327,007340,007503]

Compute milestone: 2026-11-11 03:24:44 UTC pair_job=3913007 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b56-reference-first-r1-attempt1: BEFORE independent paired C0 acquisition candidate=56 order=reference-first steps=10 nodes=nid[006941,007327,007340,007503] container=setup/gh200-cuda130-runtime.toml source=252ef8a09ddb96efef52ceebf339a52287e2dc48422f0a9fa472f5d41fc76927

Compute milestone: 2026-11-11 03:24:45 UTC pair_job=3913007 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b56-reference-first-r1-attempt1: BEFORE leg=0 role=reference budget=20 case=ep-overlap-budget-off-adaptive-20261110T223433Z-b56-reference-first-r1-attempt1-reference-b20 fresh C0; declared reserve=480 seconds

Compute milestone: 2026-11-11 03:30:50 UTC pair_job=3913007 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b56-reference-first-r1-attempt1: AFTER all16 workers returned0 for leg=reference; BEFORE small health/all-step timing summaries

Compute milestone: 2026-11-11 03:30:58 UTC pair_job=3913007 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b56-reference-first-r1-attempt1: AFTER leg=0 role=reference budget=20 completed; case host wall recorded; strict paired acceptance pending posthoc

Compute milestone: 2026-11-11 03:30:59 UTC pair_job=3913007 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b56-reference-first-r1-attempt1: BEFORE leg=1 role=candidate budget=56 case=ep-overlap-budget-off-adaptive-20261110T223433Z-b56-reference-first-r1-attempt1-candidate-b56 fresh C0; declared reserve=480 seconds

Compute milestone: 2026-11-11 03:36:36 UTC pair_job=3913007 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b56-reference-first-r1-attempt1: AFTER all16 workers returned0 for leg=candidate; BEFORE small health/all-step timing summaries

Compute milestone: 2026-11-11 03:36:45 UTC pair_job=3913007 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b56-reference-first-r1-attempt1: AFTER leg=1 role=candidate budget=56 completed; case host wall recorded; strict paired acceptance pending posthoc

Compute milestone: 2026-11-11 03:36:45 UTC pair_job=3913007 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b56-reference-first-r1-attempt1: AFTER both independent cases acquired; release4nodes before1node posthoc; shared allocation elapsed is not a per-case time

Compute milestone: 2026-11-11 03:36:45 UTC pair_job=3913007 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b56-reference-first-r1-attempt1: AFTER pair phase=acquisition_complete exit=0 complete=1; preserve every case; partial pairs are not accepted

Budget coordinator milestone: 2026-11-11 03:37:16 UTC: BEFORE posthoc pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b56-reference-first-r1-attempt1 training=3913007 released; one debug node/up to90minutes

Budget coordinator milestone: 2026-11-11 03:37:21 UTC: AFTER posthoc submit job=3913038 training=3913007 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b56-reference-first-r1-attempt1; paired and decision gates pending

Budget coordinator milestone: 2026-11-11 03:37:21 UTC: WATCH job=3913038 3913038 debug PENDING 1 (Priority)

Compute milestone: 2026-11-11 03:37:32 UTC budget_posthoc=3913038 pair_training=3913007: BEFORE separate one-node debug analysis; pair allocation released; strict per-leg and paired placement/source/input gates; container=setup/gh200-cuda130-runtime.toml

Budget coordinator milestone: 2026-11-11 03:37:51 UTC: WATCH job=3913038 3913038 debug RUNNING 1 nid006941

Compute milestone: 2026-11-11 03:38:21 UTC budget_posthoc=3913038 pair_training=3913007: AFTER both leg validations and paired comparison returned0; actual16GPU equality, inputs and timing distinction checked

Compute milestone: 2026-11-11 03:38:21 UTC budget_posthoc=3913038 pair_training=3913007: AFTER paired posthoc exit=0 complete=1; preserve original cases and partial analysis

Budget coordinator milestone: 2026-11-11 03:38:21 UTC: WATCH job=3913038 3913038 debug COMPLETING 1 nid006941

Budget coordinator milestone: 2026-11-11 03:39:46 UTC: ACCEPTED pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b56-reference-first-r1-attempt1 candidate=56 order=reference-first replicate=1 training=3913007 posthoc=3913038; total=18

Budget coordinator milestone: 2026-11-11 03:39:46 UTC: BEFORE training candidate=56 order=candidate-first replicate=2 attempt=1; four debug nodes/21minutes; independent C0 per leg

Budget coordinator milestone: 2026-11-11 03:39:47 UTC: AFTER training submit job=3913039 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b56-candidate-first-r2-attempt1; strict acceptance pending

Budget coordinator milestone: 2026-11-11 03:39:47 UTC: WATCH job=3913039 3913039 debug PENDING 4 (Priority)

Compute milestone: 2026-11-11 03:40:01 UTC pair_job=3913039 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b56-candidate-first-r2-attempt1: BEFORE independent paired C0 acquisition candidate=56 order=candidate-first steps=10 nodes=nid[006862,006941,006960,007643] container=setup/gh200-cuda130-runtime.toml source=252ef8a09ddb96efef52ceebf339a52287e2dc48422f0a9fa472f5d41fc76927

Compute milestone: 2026-11-11 03:40:04 UTC pair_job=3913039 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b56-candidate-first-r2-attempt1: BEFORE leg=0 role=candidate budget=56 case=ep-overlap-budget-off-adaptive-20261110T223433Z-b56-candidate-first-r2-attempt1-candidate-b56 fresh C0; declared reserve=480 seconds

Budget coordinator milestone: 2026-11-11 03:40:17 UTC: WATCH job=3913039 3913039 debug RUNNING 4 nid[006862,006941,006960,007643]

Budget result: 2026-11-11 03:40:35 UTC Eighteen full pairs accepted. First56/reference-first r1 training3913007/posthoc3913038 passed; actual56onall16ranks. Means13.96642s vs15.13364s at20, ratio0.92287249 (-7.71275%), allocation729s. Largest singlepairedgain, notyetrepeatability. Reverse56/candidate-first r2 job3913039 RUNNING4debug/21. Mustrepeat; thirdifdeclarednoise/order/nearbaseline trigger. Cannotextendpast56nativequalifiedresourcecap; report upperboundarylimitifnoverifieddecline.

Compute milestone: 2026-11-11 03:46:08 UTC pair_job=3913039 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b56-candidate-first-r2-attempt1: AFTER all16 workers returned0 for leg=candidate; BEFORE small health/all-step timing summaries

Compute milestone: 2026-11-11 03:46:16 UTC pair_job=3913039 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b56-candidate-first-r2-attempt1: AFTER leg=0 role=candidate budget=56 completed; case host wall recorded; strict paired acceptance pending posthoc

Compute milestone: 2026-11-11 03:46:17 UTC pair_job=3913039 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b56-candidate-first-r2-attempt1: BEFORE leg=1 role=reference budget=20 case=ep-overlap-budget-off-adaptive-20261110T223433Z-b56-candidate-first-r2-attempt1-reference-b20 fresh C0; declared reserve=480 seconds

Compute milestone: 2026-11-11 03:52:14 UTC pair_job=3913039 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b56-candidate-first-r2-attempt1: AFTER all16 workers returned0 for leg=reference; BEFORE small health/all-step timing summaries

Compute milestone: 2026-11-11 03:52:22 UTC pair_job=3913039 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b56-candidate-first-r2-attempt1: AFTER leg=1 role=reference budget=20 completed; case host wall recorded; strict paired acceptance pending posthoc

Compute milestone: 2026-11-11 03:52:22 UTC pair_job=3913039 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b56-candidate-first-r2-attempt1: AFTER both independent cases acquired; release4nodes before1node posthoc; shared allocation elapsed is not a per-case time

Compute milestone: 2026-11-11 03:52:22 UTC pair_job=3913039 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b56-candidate-first-r2-attempt1: AFTER pair phase=acquisition_complete exit=0 complete=1; preserve every case; partial pairs are not accepted

Budget coordinator milestone: 2026-11-11 03:52:23 UTC: WATCH job=3913039 3913039 debug COMPLETING 4 nid[006862,006941,006960,007643]

Budget coordinator milestone: 2026-11-11 03:52:53 UTC: BEFORE posthoc pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b56-candidate-first-r2-attempt1 training=3913039 released; one debug node/up to90minutes

Budget coordinator milestone: 2026-11-11 03:52:55 UTC: AFTER posthoc submit job=3913057 training=3913039 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b56-candidate-first-r2-attempt1; paired and decision gates pending

Budget coordinator milestone: 2026-11-11 03:52:55 UTC: WATCH job=3913057 3913057 debug PENDING 1 (Priority)

Compute milestone: 2026-11-11 03:53:03 UTC budget_posthoc=3913057 pair_training=3913039: BEFORE separate one-node debug analysis; pair allocation released; strict per-leg and paired placement/source/input gates; container=setup/gh200-cuda130-runtime.toml

Budget coordinator milestone: 2026-11-11 03:53:25 UTC: WATCH job=3913057 3913057 debug RUNNING 1 nid006305

Compute milestone: 2026-11-11 03:53:53 UTC budget_posthoc=3913057 pair_training=3913039: AFTER both leg validations and paired comparison returned0; actual16GPU equality, inputs and timing distinction checked

Compute milestone: 2026-11-11 03:53:53 UTC budget_posthoc=3913057 pair_training=3913039: AFTER paired posthoc exit=0 complete=1; preserve original cases and partial analysis

Budget coordinator milestone: 2026-11-11 03:53:57 UTC: WATCH job=3913057 3913057 debug COMPLETING 1 nid006305

Budget coordinator milestone: 2026-11-11 03:55:24 UTC: ACCEPTED pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b56-candidate-first-r2-attempt1 candidate=56 order=candidate-first replicate=2 training=3913039 posthoc=3913057; total=19

Budget coordinator milestone: 2026-11-11 03:55:24 UTC: BEFORE training candidate=56 order=reference-first replicate=3 attempt=1; four debug nodes/21minutes; independent C0 per leg

Budget coordinator milestone: 2026-11-11 03:55:25 UTC: AFTER training submit job=3913062 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b56-reference-first-r3-attempt1; strict acceptance pending

Budget coordinator milestone: 2026-11-11 03:55:25 UTC: WATCH job=3913062 3913062 debug PENDING 4 (Priority)

Compute milestone: 2026-11-11 03:55:34 UTC pair_job=3913062 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b56-reference-first-r3-attempt1: BEFORE independent paired C0 acquisition candidate=56 order=reference-first steps=10 nodes=nid[006054-006055,006061,006068] container=setup/gh200-cuda130-runtime.toml source=252ef8a09ddb96efef52ceebf339a52287e2dc48422f0a9fa472f5d41fc76927

Compute milestone: 2026-11-11 03:55:35 UTC pair_job=3913062 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b56-reference-first-r3-attempt1: BEFORE leg=0 role=reference budget=20 case=ep-overlap-budget-off-adaptive-20261110T223433Z-b56-reference-first-r3-attempt1-reference-b20 fresh C0; declared reserve=480 seconds

Budget result: 2026-11-11 03:55:52 UTC Nineteen full pairs accepted. 56/candidate-first r2 training3913039/posthoc3913057 passed. Means14.43548s vs14.11516s at20, ratio1.02269333 (+2.26933%), allocation751s. Opposite tofirst56r1ratio0.92287249 (-7.71%); observedrange~10percentagepoints. Third56/reference-first r3 job3913062 RUNNING4debug/21 isrequired. No reliable56winnerclaim. Thisfinalbudgetisnativequalifiedupperlimit; afterr3 the boundedtrainingstage canend withupperboundary-limited/no-decline conclusionifrange staysoverlapping. Source252/D3unchanged; finalsummary162source2b170f QA7/synthesis/independentreviewremain.

Budget coordinator milestone: 2026-11-11 03:55:56 UTC: WATCH job=3913062 3913062 debug RUNNING 4 nid[006054-006055,006061,006068]

Compute milestone: 2026-11-11 04:01:31 UTC pair_job=3913062 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b56-reference-first-r3-attempt1: AFTER all16 workers returned0 for leg=reference; BEFORE small health/all-step timing summaries

Compute milestone: 2026-11-11 04:01:39 UTC pair_job=3913062 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b56-reference-first-r3-attempt1: AFTER leg=0 role=reference budget=20 completed; case host wall recorded; strict paired acceptance pending posthoc

Compute milestone: 2026-11-11 04:01:40 UTC pair_job=3913062 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b56-reference-first-r3-attempt1: BEFORE leg=1 role=candidate budget=56 case=ep-overlap-budget-off-adaptive-20261110T223433Z-b56-reference-first-r3-attempt1-candidate-b56 fresh C0; declared reserve=480 seconds

Compute milestone: 2026-11-11 04:07:26 UTC pair_job=3913062 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b56-reference-first-r3-attempt1: AFTER all16 workers returned0 for leg=candidate; BEFORE small health/all-step timing summaries

Compute milestone: 2026-11-11 04:07:32 UTC pair_job=3913062 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b56-reference-first-r3-attempt1: AFTER leg=1 role=candidate budget=56 completed; case host wall recorded; strict paired acceptance pending posthoc

Compute milestone: 2026-11-11 04:07:32 UTC pair_job=3913062 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b56-reference-first-r3-attempt1: AFTER both independent cases acquired; release4nodes before1node posthoc; shared allocation elapsed is not a per-case time

Compute milestone: 2026-11-11 04:07:32 UTC pair_job=3913062 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b56-reference-first-r3-attempt1: AFTER pair phase=acquisition_complete exit=0 complete=1; preserve every case; partial pairs are not accepted

Budget coordinator milestone: 2026-11-11 04:07:35 UTC: WATCH job=3913062 3913062 debug COMPLETING 4 nid[006054-006055,006061,006068]

Budget coordinator milestone: 2026-11-11 04:08:05 UTC: BEFORE posthoc pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b56-reference-first-r3-attempt1 training=3913062 released; one debug node/up to90minutes

Budget coordinator milestone: 2026-11-11 04:08:06 UTC: AFTER posthoc submit job=3913087 training=3913062 pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b56-reference-first-r3-attempt1; paired and decision gates pending

Budget coordinator milestone: 2026-11-11 04:08:06 UTC: WATCH job=3913087 3913087 debug PENDING 1 (Priority)

Compute milestone: 2026-11-11 04:08:15 UTC budget_posthoc=3913087 pair_training=3913062: BEFORE separate one-node debug analysis; pair allocation released; strict per-leg and paired placement/source/input gates; container=setup/gh200-cuda130-runtime.toml

Budget coordinator milestone: 2026-11-11 04:08:36 UTC: WATCH job=3913087 3913087 debug RUNNING 1 nid006711

Compute milestone: 2026-11-11 04:09:04 UTC budget_posthoc=3913087 pair_training=3913062: AFTER both leg validations and paired comparison returned0; actual16GPU equality, inputs and timing distinction checked

Compute milestone: 2026-11-11 04:09:04 UTC budget_posthoc=3913087 pair_training=3913062: AFTER paired posthoc exit=0 complete=1; preserve original cases and partial analysis

Budget coordinator milestone: 2026-11-11 04:09:08 UTC: WATCH job=3913087 3913087 debug COMPLETING 1 nid006711

Budget coordinator milestone: 2026-11-11 04:10:35 UTC: ACCEPTED pair=ep-overlap-budget-off-adaptive-20261110T223433Z-b56-reference-first-r3-attempt1 candidate=56 order=reference-first replicate=3 training=3913062 posthoc=3913087; total=20

Budget coordinator milestone: 2026-11-11 04:10:35 UTC: COMPLETE bounded paired sweep: 20 accepted allocations; analyzer stopping/boundary result retained; final report review remains separate

Budget coordinator milestone: 2026-11-11 04:10:35 UTC: EXIT status=0 complete=1 phase=bounded_acquisitions_complete active_job=none; preserve every attempt

Budget summary milestone: 2026-11-11 04:11:20 UTC BEFORE final-summary QA: adaptive coordinator COMPLETE0,20pairs/40C0cases/400updates,finalreportpending. Run seven focused summary tests/import closure/plot imports/shell syntax on fresh1debugnode10min in setup/gh200-cuda130-runtime.toml. Source162/2b170f frozen and rechecked, no training or raw trace parsing.

Budget summary milestone: 2026-11-11 04:11:21 UTC AFTER final-summary QA submission: job3913091 IDep-overlap-budget-summary-qa-20261111T041120Z one debug node10minutes,source162/2b170f; next require7tests/closure/PASSdigest and terminal0 before synthesis.

Budget summary milestone: 2026-11-11 04:12:50 UTC BEFORE final adaptive synthesis: QA3913091 passed7tests/closureempty/syntax/plotimports,18s,source162/2b170f. Read completed20pair ledger, rederive400rawtimers/health/routing/actualbudget/inputs/placement and everydecision snapshot. FreshUTC training/posthoc accounting. One debugnode90min in configuredcontainer, no training or trace reparsing. Generate metrics/CSVs/plot/report in new unique run; independentreviewstillpending.

Budget summary milestone: 2026-11-11 04:12:54 UTC AFTER final synthesis submission: job3913094 IDep-overlap-budget-summary-final-20261111T041250Z one debugnode90min,source162/2b170f. Alltrainingcomplete; nextterminal0/metricsdigest/chartinspection/independentreview.

Final milestone: 2026-11-11 04:20:40 UTC FINAL ACCEPTANCE: complete20pairs/40cases/400timers, source159/252 and162/2b170f rechecked. QA3913091 COMPLETED0:0/18s seven tests; summary3913094 COMPLETED0:0/112s. IndependentFINAL_REVIEW PASS a50f41d9…6ef0; reportf610ca9e…b858, metricsc8aa699b…59e7a. Approved15artifactbindings plusreview allhashcheckOK; previousauditunchanged. FINAL_ACCEPTANCE.json SHAada4b8c95e35f90ed4e3e6b51f093c799c00c7bf377a735d06e5c361af4fdc9b. Results40bestobservedmedian2.15%shorter,4consistentlyslower,56variable/hardware-limitwithoutclearupperdecline. Userqueueempty, allrequiredworkcomplete; no newcommit/push.
