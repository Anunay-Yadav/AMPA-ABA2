"""Independent integer fixtures for all-source EP16 receive reconciliation."""
import copy
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from analysis import validate_ep16_received_routes as routes


def fixture(backend="uccl_ep", grouped=True, steps=2):
    # Permute ownership within and across nodes, and order the actual EP node
    # blocks differently from global-rank order. Receive ID order is reversed.
    owner_order = [10, 8, 11, 9, 2, 0, 3, 1, 14, 12, 15, 13, 6, 4, 7, 5]
    owners = [owner_order[expert // 8] for expert in range(128)]
    hosts = {str(rank): f"node-{rank // 4}" for rank in range(16)}
    ep = list(range(8, 12)) + list(range(0, 4)) + list(range(12, 16)) + list(range(4, 8))
    node_order = ["node-2", "node-0", "node-3", "node-1"]
    local = {rank: list(reversed([expert for expert, owner in enumerate(owners) if owner == rank])) for rank in range(16)}
    group_owners = [sorted(set(owners[group * 32:(group + 1) * 32])) for group in range(4)]
    group_hosts = [[hosts[str(peers[0])]] for peers in group_owners]
    contract = {"communication_backend": backend, "launch_policy": "torchrun_node4", "routing_audit": True,
        "audit_received_experts": True, "model": {"precision": "bf16"},
        "parallelism": {"tp": 1, "pp": 1, "cp": 1, "ep": 16, "etp": 1, "dense_dp": 16},
        "training": {"sequence_length": 4096, "micro_batch_size": 1, "global_batch_size": 64,
            "gradient_accumulation": 4, "warmup_steps": 5, "measured_steps": 20, "log_interval": 1, "completed_steps": steps},
        "routing": {"topk": 8, "num_groups": 4 if grouped else None, "group_topk": 2 if grouped else None}}
    ownership = {"communication_backend": backend, "record_receives": True, "rank_hosts": hosts,
        "num_groups": 4 if grouped else None, "group_topk": 2 if grouped else None, "diagnostic_num_groups": 4,
        "expert_owners": {str(layer): list(owners) for layer in range(48)},
        "group_owners": {str(layer): copy.deepcopy(group_owners) for layer in range(48)},
        "group_hosts": {str(layer): copy.deepcopy(group_hosts) for layer in range(48)}}
    if backend == "uccl_ep":
        ownership["uccl_node_hosts_in_pg_order"] = {str(layer): list(node_order) for layer in range(48)}
    environment = {"launch_policy": "torchrun_node4", "rank_placement": [
        {"rank": rank, "hostname": hosts[str(rank)], "local_expert_indices": list(local[rank]),
         "process_groups": {"ep": list(ep)}} for rank in range(16)]}
    audits = [{"rank": rank, "valid": True, "communication_backend": backend, "record_receives": True,
        "num_groups": 4 if grouped else None, "group_topk": 2 if grouped else None, "steps": []} for rank in range(16)]
    for step in range(steps):
        for audit in audits:
            audit["steps"].append({"step": step, "layers": []})
        for layer in range(48):
            received = [0] * 128
            for source in range(16):
                first = (source + step + layer) % 4
                selected = []
                for group in ((first, (first + 1) % 4) if grouped else range(4)):
                    for offset in ((0, 1) if grouped else (0,)):
                        start = group * 32 + ((source + layer + offset) % 4) * 8
                        selected += [start + step % 8, start + (step + 1) % 8]
                expert_counts, traffic = [0] * 128, [0] * 16
                token_ranks, token_nodes = set(), set()
                for expert in selected:
                    expert_counts[expert] += 16384
                    received[expert] += 16384
                    traffic[owners[expert]] += 16384
                    token_ranks.add(owners[expert])
                    token_nodes.add(hosts[str(owners[expert])])
                local_assignments = traffic[source]
                intra = sum(value for peer, value in enumerate(traffic) if peer != source and hosts[str(peer)] == hosts[str(source)])
                row = {"layer": layer, "expert_counts": expert_counts, "rank_traffic": traffic,
                    "routed_tokens": 16384, "assignments": 131072, "local": local_assignments,
                    "intra_node": intra, "inter_node": 131072 - local_assignments - intra,
                    "received_expert_ids": list(local[source]), "receive_calls": 4}
                if backend == "uccl_ep":
                    row.update(unique_token_rank_counts=[16384 if peer in token_ranks else 0 for peer in range(16)],
                        unique_token_node_counts=[16384 if host in token_nodes else 0 for host in node_order],
                        unique_token_node_hosts=list(node_order), layout_calls=4, metadata_calls=4)
                audits[source]["steps"][step]["layers"].append(row)
            # Each actual destination owns eight expert IDs, independently of
            # source or expert enumeration order.
            for rank in range(16):
                row = audits[rank]["steps"][step]["layers"][layer]
                row["received_expert_counts"] = [received[expert] for expert in local[rank]]
                row["received_assignments"] = sum(row["received_expert_counts"])
    return contract, ownership, audits, environment


class ReceivedRoutesTests(unittest.TestCase):
    def validate(self, documents, steps=2):
        return routes.validate_records(*documents, steps)

    def test_all_four_cases_accept_permuted_runtime_ownership_and_real_id_order(self):
        for backend, grouped in (("alltoall", False), ("alltoall", True), ("uccl_ep", False), ("uccl_ep", True)):
            with self.subTest(backend=backend, grouped=grouped):
                result = self.validate(fixture(backend, grouped))
                total = 2 * 16 * 48 * 131072
                self.assertTrue(result["valid_send_receive_audit"])
                self.assertEqual(result["total_source_expert_assignments"], total)
                self.assertEqual(result["total_received_expert_assignments"], total)
                self.assertEqual(result["checked_expert_equalities"], 2 * 48 * 128)
                if backend == "uccl_ep":
                    self.assertEqual(result["unique_token_rank_deliveries"], total // 2)
                    self.assertEqual(result["unique_token_node_deliveries"], total // (4 if grouped else 2))
                    self.assertEqual(result["node_hosts_in_actual_ep_order"], ["node-2", "node-0", "node-3", "node-1"])

    def test_full_25_update_grid_and_explicit_fresh_schema_are_required(self):
        result = self.validate(fixture("alltoall", False, 25), 25)
        self.assertEqual(result["steps"], list(range(25)))
        for mutate in (lambda d: d[0].update(audit_received_experts=False),
                       lambda d: d[1].pop("record_receives"), lambda d: d[2][0].pop("communication_backend"),
                       lambda d: d[2][0].update(valid=1), lambda d: d[0]["training"].update(log_interval=True)):
            documents = fixture()
            mutate(documents)
            with self.assertRaises(ValueError):
                self.validate(documents)

    def test_ordinary_uccl_can_reach_four_nodes_but_grouped_remains_limited_to_two(self):
        documents = fixture("uccl_ep", False)
        result = self.validate(documents)
        self.assertEqual(result["unique_token_node_deliveries"], 2 * 16 * 48 * 4 * 16384)
        contract, ownership, audits, _ = documents
        contract["routing"].update(num_groups=4, group_topk=2)
        ownership.update(num_groups=4, group_topk=2)
        for audit in audits:
            audit.update(num_groups=4, group_topk=2)
        with self.assertRaisesRegex(ValueError, "routing bounds"):
            self.validate(documents)

    def test_changed_expert_receives_fail_even_when_destination_and_global_totals_match(self):
        documents = fixture()
        row = documents[2][0]["steps"][0]["layers"][0]
        values = row["received_expert_counts"]
        donor = next(index for index, value in enumerate(values) if value)
        receiver = (donor + 1) % 8
        before = sum(values)
        values[donor] -= 1
        values[receiver] += 1
        self.assertEqual(sum(values), before)
        with self.assertRaisesRegex(ValueError, "all-source expert sends"):
            self.validate(documents)

    def test_source_histogram_and_locality_errors_cannot_cancel_in_pooled_totals(self):
        for kind in ("rank_traffic", "locality", "expert_counts", "receive_total", "receive_calls"):
            documents = fixture()
            row = documents[2][0]["steps"][0]["layers"][0]
            if kind in ("rank_traffic", "expert_counts"):
                values = row[kind]
                donor = next(index for index, value in enumerate(values) if value)
                target = (donor + (8 if kind == "expert_counts" else 1)) % len(values)
                values[donor] -= 1
                values[target] += 1
            elif kind == "locality":
                row["local"] += 1
                row["inter_node"] -= 1
            elif kind == "receive_total":
                row["received_assignments"] += 1
            else:
                row["receive_calls"] -= 1
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                self.validate(documents)

    def test_wrong_owner_host_and_ordered_pg_node_mapping_are_rejected(self):
        for kind in ("owner", "receive_ids", "hostname", "ep_order", "node_order", "owner_bool"):
            documents = fixture()
            ownership, audits, environment = documents[1:]
            if kind == "owner":
                vector = ownership["expert_owners"]["0"]
                vector[0], vector[8] = vector[8], vector[0]
            elif kind == "receive_ids":
                audits[0]["steps"][0]["layers"][0]["received_expert_ids"][0] = 0
            elif kind == "hostname":
                environment["rank_placement"][0]["hostname"] = "node-other"
            elif kind == "ep_order":
                for row in environment["rank_placement"]:
                    peers = row["process_groups"]["ep"]
                    peers[0], peers[4] = peers[4], peers[0]
            elif kind == "node_order":
                ownership["uccl_node_hosts_in_pg_order"]["0"].reverse()
            else:
                ownership["expert_owners"]["0"][40] = False
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                self.validate(documents)

    def test_missing_duplicate_or_misindexed_coverage_is_rejected(self):
        for kind in ("rank", "duplicate_rank", "step", "step_bool", "layer", "layer_index"):
            documents = fixture()
            audits = documents[2]
            if kind == "rank":
                audits.pop()
            elif kind == "duplicate_rank":
                audits[1]["rank"] = 0
            elif kind == "step":
                audits[0]["steps"].pop()
            elif kind == "step_bool":
                audits[0]["steps"][0]["step"] = False
            elif kind == "layer":
                audits[0]["steps"][0]["layers"].pop()
            else:
                audits[0]["steps"][0]["layers"][0]["layer"] = 1
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                self.validate(documents)

    def test_unique_deliveries_are_not_assignment_counts_or_summed_rank_copies(self):
        for kind in ("rank_assignments", "node_rank_sum", "node_host_order", "zero_unique", "layout_calls"):
            documents = fixture()
            row = documents[2][0]["steps"][0]["layers"][0]
            if kind == "rank_assignments":
                row["unique_token_rank_counts"] = list(row["rank_traffic"])
            elif kind == "node_rank_sum":
                row["unique_token_node_counts"] = [value * 2 for value in row["unique_token_node_counts"]]
            elif kind == "node_host_order":
                row["unique_token_node_hosts"].reverse()
            elif kind == "zero_unique":
                row["unique_token_rank_counts"] = [0] * 16
            else:
                row["layout_calls"] = 3
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                self.validate(documents)

    def write_fixture(self, root):
        contract, ownership, audits, environment = fixture()
        (root / "config").mkdir()
        (root / "raw").mkdir()
        for path, value in [(root / "config/comparison_contract.json", contract),
                            (root / "config/routing_ownership.json", ownership), (root / "environment.json", environment)] + [
                            (root / "raw" / f"routing_audit_rank-{rank}.json", audit) for rank, audit in enumerate(audits)]:
            path.write_text(json.dumps(value))

    def test_hash_binding_detects_changed_inputs_and_cli_requires_separate_unique_output(self):
        with TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            root.mkdir()
            self.write_fixture(root)
            result = routes.validate_run(root, 2)
            self.assertEqual(len(result["input_files"]), 19)
            self.assertEqual(len({row["path"] for row in result["input_files"]}), 19)
            actual_validate = routes.validate_records
            def changed(*args):
                result = actual_validate(*args)
                path = root / "raw/routing_audit_rank-0.json"
                path.write_text(path.read_text() + "\n")
                return result
            with patch.object(routes, "validate_records", side_effect=changed), self.assertRaisesRegex(ValueError, "input changed"):
                routes.validate_run(root, 2)
            output = Path(directory) / "received-proof.json"
            argv = ["validate_ep16_received_routes.py", "--run-dir", str(root), "--steps", "2", "--output", str(output)]
            with patch.object(sys, "argv", argv), patch.dict("os.environ", {"SLURM_JOB_ID": "test", "SLURM_JOB_PARTITION": "preemptable"}):
                routes.main()
                self.assertTrue(json.loads(output.read_text())["valid_send_receive_audit"])
                with self.assertRaisesRegex(ValueError, "new unique"):
                    routes.main()
            argv[-1] = str(root / "analysis.json")
            with patch.object(sys, "argv", argv), patch.dict("os.environ", {"SLURM_JOB_ID": "test", "SLURM_JOB_PARTITION": "preemptable"}), \
                 self.assertRaisesRegex(ValueError, "separate"):
                routes.main()


if __name__ == "__main__":
    unittest.main()
