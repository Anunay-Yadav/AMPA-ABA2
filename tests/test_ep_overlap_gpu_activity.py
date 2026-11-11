"""Adversarial exact-correlation and same-device overlap checks.

Synthetic events test attribution rules only. Execute on allocated compute.
"""

import copy
import unittest

from analysis.ep_overlap_gpu_activity import inspect_events, intersection_us
from profiling.ampa_profile.overlap_trace import OverlapMethodLabels, node_label


def event(name, category, start, duration, *, tid=1, args=None, pid=1):
    return {"ph": "X", "name": name, "cat": category, "ts": start, "dur": duration,
            "pid": pid, "tid": tid, "args": {} if args is None else args}


def fixture(backend="uccl_ep"):
    labels = OverlapMethodLabels(None, [], communication_backend=backend)
    contract = labels.contract_record(4)
    counts = labels.expected_counts(4)
    annotations = [event("AMPA/STEP/0", "user_annotation", 0, 3000000)]
    by_name = {}
    serial = 0
    for name, count in counts.items():
        by_name[name] = []
        for _ in range(count):
            row = event(name, "user_annotation", 2000000 + serial * 2, 0.5)
            serial += 1
            annotations.append(row)
            by_name[name].append(row)
    trace = {"distributedInfo": {"rank": 0, "world_size": 16}, "traceEvents": annotations}
    record = {"schema_version": 1, "kind": "ampa_ep_overlap_calls", "rank": 0, "step": 0,
              "capture_mode": "torch", "skipped_iter": False, "layers": list(range(48)),
              "counts": dict(counts), "observation_contract": contract,
              "uccl_autograd_pairs": []}
    payloads, runtimes = [], []

    def place(name, occurrence, start, *, tid=1):
        row = by_name[name][occurrence]
        row.update(ts=start, dur=1, tid=tid)
        return row

    def launch(name, host_time, gpu_time, correlation, *, external=None, tid=1):
        runtime = event("cudaLaunchKernel", "cuda_runtime", host_time + 0.2, 0.1,
                        tid=tid, args={"correlation": correlation})
        args = {"correlation": correlation, "device": 0, "stream": 7}
        if external is not None:
            args["External id"] = external
        kernel = event(name, "kernel", gpu_time, 10, pid=0, tid=7, args=args)
        trace["traceEvents"].extend((runtime, kernel))
        return runtime, kernel

    if backend == "uccl_ep":
        phases = {("dispatch", "FWD"): "uccl_dispatch", ("dispatch", "BWD"): "uccl_combine_dinput_dprob",
                  ("combine", "FWD"): "uccl_combine", ("combine", "BWD"): "uccl_dispatch_doutput"}
        for microbatch in range(4):
            for layer in range(48):
                for family in ("dispatch", "combine"):
                    call_id = len(record["uccl_autograd_pairs"])
                    record["uccl_autograd_pairs"].append({
                        "call_id": call_id, "layer": layer, "family": family, "epoch": 1,
                        "context_id": call_id + 1, "forward_returned": True, "backward_returned": True})
                    for direction in ("FWD", "BWD"):
                        ordinal = len(payloads)
                        host_time = 100 + ordinal * 10
                        annotations.append(event(f"AMPA/UCCL_CALL/{call_id}/{direction}",
                                                 "user_annotation", host_time, 1))
                        place(f"AMPA/{direction}/layer_{layer:02d}/{phases[family, direction]}",
                              microbatch, host_time)
                        slot = "moe_dispatch" if family == "dispatch" else "moe_combine"
                        place(node_label(microbatch, direction, slot, layer), 0, host_time)
                        operation = "dispatch" if (family == "dispatch") == (direction == "FWD") else "combine"
                        runtime, kernel = launch(f"uccl::internode::{operation}<BF16>", host_time,
                                                 500 + ordinal * 20, ordinal + 1)
                        runtimes.append(runtime)
                        payloads.append(kernel)
    else:
        place("AMPA/FWD/layer_00/dispatch_activations_a2a", 0, 100)
        place(node_label(0, "FWD", "moe_dispatch", 0), 0, 100)
        runtime, kernel = launch("ncclDevKernel_AllToAll", 100, 500, 1)
        runtimes.append(runtime)
        payloads.append(kernel)

    place(node_label(0, "FWD", "mlp", 1), 0, 102)
    runtime, compute = launch("cutlass_gemm", 102, 505, 9001)
    return trace, record, {"by_name": by_name, "payloads": payloads, "runtimes": runtimes,
                           "compute": compute, "compute_runtime": runtime}


class EpOverlapGpuActivityTest(unittest.TestCase):
    def test_interval_unions_do_not_double_count_nested_or_overlapping_kernels(self):
        self.assertEqual(intersection_us([(0, 10), (2, 5), (8, 12)], [(3, 7), (6, 15)]), 9)

    def test_all_384_uccl_pairs_and_768_directions_have_exact_payload_witnesses(self):
        trace, record, _ = fixture()
        result = inspect_events(trace, record)
        self.assertTrue(result["valid_known_payload_witness"])
        self.assertTrue(result["observed_positive_gpu_concurrency"])
        self.assertEqual(result["known_payload_kernels"], 768)
        self.assertEqual(result["correlated_compute_kernels"], 1)
        self.assertAlmostEqual(result["intersection_ms"], 0.005)
        self.assertIsNone(result["whole_session_event_completeness"])

    def test_native_proof_is_limited_to_exact_forward_payload_scopes(self):
        trace, record, _ = fixture("alltoall")
        result = inspect_events(trace, record)
        self.assertFalse(result["valid_known_payload_witness"])
        self.assertTrue(result["native_forward_only"])
        self.assertEqual(result["known_payload_kernels"], 1)
        self.assertAlmostEqual(result["intersection_ms"], 0.005)

    def test_enabled_schedule_does_not_imply_positive_gpu_overlap(self):
        trace, record, rows = fixture()
        rows["compute"]["ts"] = 900000
        result = inspect_events(trace, record)
        self.assertFalse(result["observed_positive_gpu_concurrency"])
        self.assertEqual(result["intersection_ms"], 0)

    def test_nearby_timestamps_cannot_replace_exact_payload_launch_links(self):
        trace, record, rows = fixture()
        rows["payloads"][0]["args"].pop("correlation")
        with self.assertRaises(ValueError):
            inspect_events(trace, record)

    def test_duplicate_exact_correlation_is_rejected(self):
        trace, record, rows = fixture()
        trace["traceEvents"].append(copy.deepcopy(rows["runtimes"][0]))
        with self.assertRaises(ValueError):
            inspect_events(trace, record)

    def test_missing_one_adjoint_payload_invalidates_the_known_call_grid(self):
        trace, record, rows = fixture()
        trace["traceEvents"].remove(rows["payloads"][-1])
        with self.assertRaises(ValueError):
            inspect_events(trace, record)

    def test_pair_grid_and_completed_autograd_returns_are_required(self):
        for field, value in (("layer", 47), ("forward_returned", False),
                             ("backward_returned", False), ("epoch", 0), ("context_id", 0)):
            with self.subTest(field=field):
                trace, record, _ = fixture()
                record["uccl_autograd_pairs"][0][field] = value
                with self.assertRaises(ValueError):
                    inspect_events(trace, record)

    def test_pair_identity_must_be_under_its_own_layer_and_phase(self):
        trace, record, rows = fixture()
        correct = rows["by_name"]["AMPA/FWD/layer_00/uccl_dispatch"][0]
        wrong = rows["by_name"]["AMPA/FWD/layer_01/uccl_dispatch"][0]
        correct["ts"], wrong["ts"] = wrong["ts"], correct["ts"]
        with self.assertRaises(ValueError):
            inspect_events(trace, record)

    def test_conflicting_uccl_runtime_and_external_call_owners_are_rejected(self):
        trace, record, rows = fixture()
        second_time = rows["runtimes"][2]["ts"]
        trace["traceEvents"].append(event("host_launch", "cpu_op", second_time, 0.1,
                                           args={"External id": 9876}))
        rows["payloads"][0]["args"]["External id"] = 9876
        with self.assertRaises(ValueError):
            inspect_events(trace, record)

    def test_conflicting_native_forward_payload_owners_are_rejected(self):
        trace, record, rows = fixture("alltoall")
        wrong = rows["by_name"]["AMPA/FWD/layer_01/combine_expert_output_a2a"][0]
        wrong.update(ts=100, dur=1, tid=2)
        trace["traceEvents"].append(event("host_launch", "cpu_op", 100.2, 0.1, tid=2,
                                           args={"External id": 9876}))
        rows["payloads"][0]["args"]["External id"] = 9876
        with self.assertRaises(ValueError):
            inspect_events(trace, record)

    def test_mixed_or_unknown_gpu_device_cannot_prove_same_gpu_concurrency(self):
        for device in (1, None, "0"):
            with self.subTest(device=device):
                trace, record, rows = fixture()
                if device is None:
                    rows["compute"]["args"].pop("device")
                else:
                    rows["compute"]["args"]["device"] = device
                with self.assertRaises(ValueError):
                    inspect_events(trace, record)

    def test_self_declared_incomplete_count_contract_is_rejected(self):
        trace, record, rows = fixture()
        missing = node_label(3, "WGRAD", "mlp", 47)
        record["counts"].pop(missing)
        record["observation_contract"]["expected_counts"].pop(missing)
        trace["traceEvents"].remove(rows["by_name"][missing][0])
        with self.assertRaises(ValueError):
            inspect_events(trace, record)

    def test_extra_obsolete_ordinary_scope_is_rejected(self):
        trace, record, _ = fixture()
        trace["traceEvents"].append(event("AMPA/TRAIN/microbatch_backward", "user_annotation", 30, 1))
        with self.assertRaises(ValueError):
            inspect_events(trace, record)

    def test_communication_helper_name_cannot_be_counted_as_compute(self):
        trace, record, rows = fixture()
        rows["compute"]["name"] = "elementwise_copy"
        # The substring "mma" appears in "command". This helper is under
        # a communication node and is not a named GEMM/attention computation.
        trace["traceEvents"].extend((
            event("cudaLaunchKernel", "cuda_runtime", 100.4, 0.1, args={"correlation": 9002}),
            event("command_buffer_wait", "kernel", 505, 10, pid=0, tid=7,
                  args={"correlation": 9002, "device": 0, "stream": 7})))
        with self.assertRaises(ValueError):
            inspect_events(trace, record)


if __name__ == "__main__":
    unittest.main()
