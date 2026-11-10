"""Allocated-compute checks for EP16 evidence gates and legacy compatibility."""

import gzip
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))

from analyze_all_step_trace import _workload, main as analyze
from compare_deep_trace_overhead import _compare_consumed_batches
from compare_profile_overhead import _environment
from validate_torch_trace import inspect_rank


def _contract(world=16, cp=1, steps=25):
    return {"parallelism": {"world_size": world, "tp": 1, "pp": 1, "cp": cp,
                            "ep": world, "etp": 1, "dense_dp": world // cp},
            "training": {"sequence_length": 4096, "micro_batch_size": 1,
                         "global_batch_size": 64, "gradient_accumulation": 64 * cp // world,
                         "warmup_steps": 5, "measured_steps": 20,
                         "log_interval": 1, "completed_steps": steps}}


def _environment_record(world=16, cp=1):
    topology = {"world_size": world, "tp": 1, "pp": 1, "cp": cp,
                "ep": world, "etp": 1, "dense_dp": world // cp, "expert_dp": 1}
    if cp == 2:
        topology["dense_dp_cp"] = world
    return {"runtime_topology": topology, "rank_placement": [
        {"rank": rank, "hostname": f"node-{rank // 4}",
         "physical_gpu_uuid": f"GPU-00000000-0000-0000-0000-{rank:012d}"}
        for rank in range(world)]}


def _trace(root, rank=15, cp=1, step=0, cp_events=False, ep_ranks=None):
    forward = "AMPA/TRAIN/microbatch_forward"
    backward = "AMPA/TRAIN/microbatch_backward"
    residual = "AMPA/FWD/layer_00/residual_bda"
    alltoall = "AMPA/FWD/layer_00/dispatch_activations_a2a"
    events = [
        {"name": f"AMPA/STEP/{step}", "cat": "user_annotation", "ph": "X",
         "pid": 1, "tid": 2, "ts": 0, "dur": 100},
        {"name": forward, "cat": "user_annotation", "ph": "X",
         "pid": 1, "tid": 2, "ts": 5, "dur": 40},
        {"name": residual, "cat": "user_annotation", "ph": "X",
         "pid": 1, "tid": 2, "ts": 10, "dur": 10},
        {"name": "cudaLaunchKernel", "cat": "cuda_runtime", "ph": "X",
         "pid": 1, "tid": 2, "ts": 11, "dur": 1, "args": {"correlation": 7}},
        {"name": "kernel", "cat": "kernel", "ph": "X",
         "pid": 0, "tid": 0, "ts": 13, "dur": 2, "args": {"correlation": 7}},
        {"name": alltoall, "cat": "user_annotation", "ph": "X",
         "pid": 1, "tid": 2, "ts": 30, "dur": 10},
        {"name": "cudaLaunchKernel", "cat": "cuda_runtime", "ph": "X",
         "pid": 1, "tid": 2, "ts": 31, "dur": 1, "args": {"correlation": 8}},
        {"name": "ncclDevKernel", "cat": "kernel", "ph": "X",
         "pid": 0, "tid": 0, "ts": 33, "dur": 3,
         "args": {"correlation": 8, "Process Group Description": "EXPERT_MODEL_PARALLEL_GROUP",
                  "Process Group Ranks": str(list(range(16)) if ep_ranks is None else ep_ranks),
                  "Collective name": "all_to_allv"}},
        {"name": backward, "cat": "user_annotation", "ph": "X",
         "pid": 1, "tid": 2, "ts": 50, "dur": 30},
        {"name": "_AllToAll", "cat": "cpu_op", "ph": "X", "pid": 1, "tid": 2,
         "ts": 30.2, "dur": 9, "args": {"External id": 8999, "Sequence number": 1}},
        {"name": "autograd::engine::evaluate_function: _AllToAllBackward", "cat": "cpu_op", "ph": "X",
         "pid": 1, "tid": 3, "ts": 51, "dur": 20, "args": {"External id": 9000, "Sequence number": 1}},
        {"name": "fwdbwd", "cat": "fwdbwd", "ph": "s", "pid": 1, "tid": 2, "ts": 30.2, "id": 1},
        {"name": "fwdbwd", "cat": "fwdbwd", "ph": "f", "pid": 1, "tid": 3, "ts": 51, "id": 1},
        {"name": "record_param_comms", "cat": "cpu_op", "ph": "X",
         "pid": 1, "tid": 3, "ts": 52, "dur": 15, "args": {"External id": 9001}},
        {"name": "cudaLaunchKernel", "cat": "cuda_runtime", "ph": "X",
         "pid": 1, "tid": 3, "ts": 53, "dur": 1, "args": {"correlation": 9}},
        {"name": "ncclDevKernel", "cat": "kernel", "ph": "X",
         "pid": 0, "tid": 0, "ts": 54, "dur": 3,
         "args": {"correlation": 9, "External id": 9001,
                  "Process Group Description": "EXPERT_MODEL_PARALLEL_GROUP",
                  "Process Group Ranks": str(list(range(16))), "Collective name": "all_to_allv"}},
    ]
    if cp_events:
        for owner, count, start, offset in (("AttnFuncWithCPAndKVP2P", 48 * 8, 21, 100),
                                            ("AttnFuncWithCPAndKVP2PBackward", 48 * 8 * 2, 51, 1000)):
            for index in range(count):
                external = offset + index
                timestamp = start + index / 10000
                events.extend([
                    {"name": owner, "cat": "cpu_op", "ph": "X", "pid": 1,
                     "tid": 2, "ts": timestamp, "dur": 0.00001,
                     "args": {"External id": external}},
                    {"name": "ncclDevKernel_SendRecv", "cat": "kernel", "ph": "X",
                     "pid": 0, "tid": 0, "ts": timestamp + 0.00002, "dur": 0.00001,
                     "args": {"External id": external}},
                ])
    events.extend({"name": f"metadata-{index:06d}-" + "x" * 100, "ph": "M"}
                  for index in range(2500))
    trace = root / "traces" / "torch_profile" / f"step-{step:05d}" / f"rank-{rank}.json.gz"
    counts = root / "raw" / "deep_trace" / f"step-{step:05d}" / f"rank-{rank}.json"
    trace.parent.mkdir(parents=True, exist_ok=True)
    counts.parent.mkdir(parents=True, exist_ok=True)
    groups = [{"pg_desc": "EXPERT_MODEL_PARALLEL_GROUP", "pg_size": 16, "ranks": list(range(16))},
              {"pg_desc": "CONTEXT_PARALLEL_GROUP", "pg_size": cp,
               "ranks": [14, 15] if cp == 2 else [rank]}]
    with gzip.open(trace, "wt", encoding="utf-8", compresslevel=0) as stream:
        json.dump({"distributedInfo": {"rank": rank, "world_size": 16, "pg_config": groups},
                   "traceEvents": events}, stream)
    counts.write_text(json.dumps({"rank": rank, "steps": [
        {"step": step, "layers": list(range(48)),
         "counts": {forward: 1, backward: 1, residual: 1, alltoall: 1}}]}))
    return trace, counts


class EP16AnalysisTest(unittest.TestCase):
    def test_workload_keeps_global_tokens_and_rejects_wrong_accumulation(self):
        self.assertEqual(_workload(_contract()), (16, 1, 4))
        self.assertEqual(_workload(_contract(cp=2)), (16, 2, 8))
        legacy = _contract(world=8, cp=2)
        del legacy["parallelism"]["world_size"]
        del legacy["training"]["completed_steps"]
        self.assertEqual(_workload(legacy), (8, 2, 16))
        wrong = _contract()
        wrong["training"]["gradient_accumulation"] = 8
        with self.assertRaisesRegex(ValueError, "fixed-token workload"):
            _workload(wrong)

    def test_actual_placement_requires_four_nodes_and_unique_gpus(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            for cp in (1, 2):
                record = _environment_record(cp=cp)
                (root / "environment.json").write_text(json.dumps(record))
                self.assertEqual(_environment(root, cp=cp, world_size=16, ep=16), record)
            record["rank_placement"][-1]["hostname"] = "node-0"
            (root / "environment.json").write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, "four-rank placement"):
                _environment(root, cp=2, world_size=16, ep=16)
            record = _environment_record()
            record["rank_placement"][-1]["physical_gpu_uuid"] = record["rank_placement"][0]["physical_gpu_uuid"]
            (root / "environment.json").write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, "duplicate physical GPUs"):
                _environment(root, cp=1, world_size=16, ep=16)
            legacy = _environment_record(world=8, cp=2)
            (root / "environment.json").write_text(json.dumps(legacy))
            self.assertEqual(_environment(root, cp=2), legacy)

    def test_actual_batches_cover_rank15_and_preserve_order(self):
        with TemporaryDirectory() as temporary:
            controls = [Path(temporary) / name for name in ("control", "trace")]
            for accumulation in (4, 8):
                for run in controls:
                    (run / "raw").mkdir(parents=True, exist_ok=True)
                    for rank in range(16):
                        rows = [{"step": step, "batches": [{"tokens": {
                            "sha256": hashlib.sha256(f"rank-{rank}:step-{step}:micro-{micro}".encode()).hexdigest(),
                            "shape": [1, 4096], "dtype": "torch.int64",
                        }} for micro in range(accumulation)]} for step in range(25)]
                        (run / "raw" / f"sample_fingerprint_rank-{rank}.json").write_text(
                            json.dumps({"schema_version": 1, "kind": "consumed_cpu_token_batch_sha256",
                                        "rank": rank, "steps": rows}))
                result = _compare_consumed_batches(*controls, world_size=16, accumulation=accumulation)
                self.assertEqual(result["compared_rank_microbatches"], 16 * 25 * accumulation)
            path = controls[1] / "raw" / "sample_fingerprint_rank-15.json"
            record = json.loads(path.read_text())
            record["steps"][-1]["batches"].reverse()
            path.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, "rank 15: control and trace consumed different"):
                _compare_consumed_batches(*controls, world_size=16, accumulation=8)

    def test_ep16_trace_rejects_ep8_payload_membership_and_cp_in_cp1(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            trace, counts = _trace(root)
            result = inspect_rank(root, 15, 0, trace_path=trace, count_path=counts, world_size=16, cp=1)
            self.assertEqual(result["ep_a2a_calls_with_correlated_nccl"], 1)
            trace, counts = _trace(root, ep_ranks=list(range(8)))
            with self.assertRaisesRegex(ValueError, "EP AllToAll NCCL rank membership"):
                inspect_rank(root, 15, 0, trace_path=trace, count_path=counts, world_size=16, cp=1)
            trace, counts = _trace(root, cp_events=True)
            with self.assertRaisesRegex(ValueError, "CP1 trace contains CP KV"):
                inspect_rank(root, 15, 0, trace_path=trace, count_path=counts, world_size=16, cp=1)

    def test_two_step_analysis_checks_cp1_zero_or_cp2_exact_peer_counts(self):
        for cp in (1, 2):
            with self.subTest(cp=cp), TemporaryDirectory() as temporary:
                root = Path(temporary)
                (root / "config").mkdir()
                (root / "raw").mkdir()
                (root / "config" / "comparison_contract.json").write_text(json.dumps(_contract(cp=cp, steps=2)))
                (root / "config" / "all_step_capture.json").write_text(json.dumps(
                    {"enabled": True, "ranks": list(range(16)), "steps": [0, 1]}))
                (root / "config" / "resolved.yaml").write_text(
                    "profiling:\n  use_pytorch_profiler: false\n  use_nsys_profiler: false\n")
                (root / "raw" / "routing_audit_rank-15.json").write_text(json.dumps(
                    {"valid": True, "steps": [{"step": 0}, {"step": 1}]}))
                for step in (0, 1):
                    _trace(root, cp=cp, step=step, cp_events=cp == 2)
                with patch.object(sys, "argv", ["analyze_all_step_trace.py", "--run-dir", str(root),
                                                "--rank", "15", "--steps", "2"]):
                    analyze()
                result = json.loads((root / "analysis" / "all_steps" / "rank-15" / "summary.json").read_text())
                self.assertEqual(result["steps"], [0, 1])
                self.assertEqual(result["world_size"], 16)
                self.assertEqual(result["cp"], cp)
                self.assertEqual(result["gradient_accumulation"], 4 * cp)


if __name__ == "__main__":
    unittest.main()
