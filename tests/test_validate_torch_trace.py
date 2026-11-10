"""A Chrome trace is usable only when labels and CUDA launches reconcile."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
import tempfile
import unittest

from analysis.validate_torch_trace import inspect_rank
from analysis.aggregate_deep_trace import _device_scope_links, aggregate_rank


class TorchTraceValidationTest(unittest.TestCase):
    def test_untagged_cp_peer_kernel_requires_exact_selected_step_owner(self):
        events = [
            {"name": "AMPA/STEP/5", "cat": "user_annotation", "ph": "X", "pid": 1,
             "tid": 2, "ts": 0, "dur": 100},
            {"name": "AMPA/TRAIN/microbatch_forward", "cat": "user_annotation", "ph": "X",
             "pid": 1, "tid": 2, "ts": 10, "dur": 60},
            {"name": "AttnFuncWithCPAndKVP2P", "cat": "cpu_op", "ph": "X", "pid": 1,
             "tid": 2, "ts": 20, "dur": 10, "args": {"External id": 11}},
            {"name": "ncclDevKernel_SendRecv", "cat": "kernel", "ph": "X", "pid": 0,
             "tid": 0, "ts": 32, "dur": 3, "args": {"External id": 11}},
        ]
        _, stats = _device_scope_links(events)
        self.assertEqual(stats["untagged_cp_kv_p2p_by_owner"],
                         {"AttnFuncWithCPAndKVP2P": 1})
        events[2]["name"] = "_AllToAll"
        with self.assertRaisesRegex(ValueError, "untagged NCCL kernel lacks"):
            _device_scope_links(events)

    def test_exact_step_labels_and_cuda_correlation(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            (run / "traces" / "torch_profile").mkdir(parents=True)
            (run / "raw").mkdir()
            (run / "logs").mkdir()
            label = "AMPA/FWD/layer_00/residual_bda"
            a2a = "AMPA/FWD/layer_00/dispatch_activations_a2a"
            train_forward = "AMPA/TRAIN/microbatch_forward"
            train_backward = "AMPA/TRAIN/microbatch_backward"
            (run / "raw" / "deep_trace_rank-0.json").write_text(json.dumps({
                "rank": 0, "steps": [{"step": 5, "layers": list(range(48)),
                                      "counts": {label: 1, a2a: 1, train_forward: 1, train_backward: 1}}]
            }), encoding="utf-8")
            events = [
                {"name": "AMPA/STEP/5", "cat": "user_annotation", "ph": "X", "pid": 1, "tid": 2, "ts": 0, "dur": 100},
                {"name": train_forward, "cat": "user_annotation", "ph": "X", "pid": 1, "tid": 2, "ts": 5, "dur": 40},
                {"name": label, "cat": "user_annotation", "ph": "X", "pid": 1, "tid": 2, "ts": 10, "dur": 10},
                {"name": "cudaLaunchKernel", "cat": "cuda_runtime", "ph": "X", "pid": 1, "tid": 2, "ts": 11, "dur": 1, "args": {"correlation": 7}},
                {"name": "kernel", "cat": "kernel", "ph": "X", "pid": 0, "tid": 0, "ts": 13, "dur": 2, "args": {"correlation": 7}},
                {"name": a2a, "cat": "user_annotation", "ph": "X", "pid": 1, "tid": 2, "ts": 30, "dur": 10},
                {"name": "cudaLaunchKernel", "cat": "cuda_runtime", "ph": "X", "pid": 1, "tid": 2, "ts": 31, "dur": 1, "args": {"correlation": 8}},
                {"name": "ncclDevKernel", "cat": "kernel", "ph": "X", "pid": 0, "tid": 0, "ts": 33, "dur": 3,
                 "args": {"correlation": 8, "Process Group Description": "EXPERT_MODEL_PARALLEL_GROUP",
                          "Process Group Ranks": str(list(range(8))),
                          "Collective name": "all_to_allv"}},
                {"name": train_backward, "cat": "user_annotation", "ph": "X", "pid": 1, "tid": 2, "ts": 50, "dur": 20},
                {"name": "_AllToAll", "cat": "cpu_op", "ph": "X", "pid": 1, "tid": 2,
                 "ts": 30.5, "dur": 9, "args": {"External id": 12, "Sequence number": 7}},
                {"name": "autograd::engine::evaluate_function: _AllToAllBackward", "cat": "cpu_op", "ph": "X", "pid": 1, "tid": 2,
                 "ts": 52, "dur": 12, "args": {"External id": 11, "Sequence number": 7}},
                {"name": "fwdbwd", "cat": "fwdbwd", "ph": "s", "pid": 1, "tid": 2, "ts": 30.5, "id": 7},
                {"name": "fwdbwd", "cat": "fwdbwd", "ph": "f", "pid": 1, "tid": 2, "ts": 52, "id": 7},
                {"name": "record_param_comms", "cat": "cpu_op", "ph": "X", "pid": 1, "tid": 2,
                 "ts": 55, "dur": 1, "args": {"External id": 9}},
                {"name": "ncclDevKernel", "cat": "kernel", "ph": "X", "pid": 0, "tid": 0,
                 "ts": 57, "dur": 4, "args": {"correlation": 99, "External id": 9,
                      "Process Group Description": "EXPERT_MODEL_PARALLEL_GROUP",
                      "Process Group Ranks": str(list(range(8))),
                      "Collective name": "all_to_allv"}},
                {"name": "record_param_comms", "cat": "cpu_op", "ph": "X", "pid": 1, "tid": 2,
                 "ts": 80, "dur": 1, "args": {"External id": 10}},
                {"name": "oneRankReduce", "cat": "kernel", "ph": "X", "pid": 0, "tid": 0,
                 "ts": 82, "dur": 6, "args": {"External id": 10,
                                               "Process Group Description": "DENSE_DATA_PARALLEL_GROUP",
                                               "Collective name": "reduce_scatter_tensor_coalesced"}},
                {"name": "AMPA/STEP/5", "cat": "gpu_user_annotation", "ph": "X", "pid": 0, "tid": 7, "ts": 1, "dur": 98},
                {"name": a2a, "cat": "gpu_user_annotation", "ph": "X", "pid": 0, "tid": 7, "ts": 32, "dur": 3},
            ]
            # Make a small synthetic export cross the file-size plausibility gate.
            events.extend({"name": f"metadata-{i:06d}-" + "x" * 100, "ph": "M"} for i in range(2500))
            path = run / "traces" / "torch_profile" / "rank-0.json.gz"
            distributed = {"rank": 0, "world_size": 8, "pg_config": [
                {"pg_desc": "EXPERT_MODEL_PARALLEL_GROUP", "pg_size": 8, "ranks": list(range(8))},
                {"pg_desc": "CONTEXT_PARALLEL_GROUP", "pg_size": 2, "ranks": [0, 1]},
            ]}
            with gzip.open(path, "wt", encoding="utf-8", compresslevel=0) as stream:
                json.dump({"distributedInfo": distributed, "traceEvents": events}, stream)
            self.assertGreater(path.stat().st_size, 100_000)
            result = inspect_rank(run, 0, 5)
            self.assertEqual(result["device_events_linked_to_runtime"], 2)
            self.assertEqual(result["device_link_stats"]["linked_by_either_route"], 4)
            self.assertEqual(result["device_link_stats"]["selected_step_unassigned_activity_sum_us"], 6)
            self.assertEqual(result["device_link_stats"]["selected_step_unassigned_activity_union_us"], 6)
            self.assertEqual(result["device_link_stats"]["selected_step_unassigned_nccl_by_group_us"],
                             {"DENSE_DATA_PARALLEL_GROUP": 6})
            self.assertEqual(result["device_link_stats"]["selected_step_nccl_by_group_us"],
                             {"DENSE_DATA_PARALLEL_GROUP": 6, "EXPERT_MODEL_PARALLEL_GROUP": 7})
            self.assertEqual(result["linked_device_activity_fraction"], 1.0)
            self.assertEqual(result["selected_step_device_activity_fraction"], 1.0)
            self.assertEqual(result["ep_a2a_calls_with_correlated_nccl"], 1)
            timing = aggregate_rank(run, 0)
            phases = {row["phase"]: row for row in timing["phase_rows"]}
            self.assertEqual(phases["residual_bda"]["gpu_launched_activity_sum_ms"], 0.002)
            self.assertEqual(phases["dispatch_activations_a2a"]["gpu_exclusive_nccl_sum_ms"], 0.003)
            self.assertEqual(phases["dispatch_activations_a2a"]["gpu_exclusive_nccl_median_call_ms"], 0.003)
            self.assertEqual(phases["train_microbatch_forward"]["gpu_launched_activity_sum_ms"], 0.005)
            self.assertEqual(phases["train_microbatch_backward"]["gpu_exclusive_nccl_sum_ms"], 0.004)
            events[0]["name"] = "AMPA/STEP/4"
            with gzip.open(path, "wt", encoding="utf-8", compresslevel=0) as stream:
                json.dump({"distributedInfo": distributed, "traceEvents": events}, stream)
            with self.assertRaisesRegex(ValueError, "captured step scopes"):
                inspect_rank(run, 0, 5)


if __name__ == "__main__":
    unittest.main()
