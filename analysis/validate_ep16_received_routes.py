#!/usr/bin/env python3
"""Reconcile fresh EP16 source assignments with actual owner receive counts.

This validates aggregate routing records. It does not measure physical bytes,
prove token values/gradients, or replace training-health and trace checks.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path


WORLD, LAYERS, EXPERTS, TOPK, ACCUMULATION, TOKENS = 16, 48, 128, 8, 4, 16384
ASSIGNMENTS = TOKENS * TOPK


def require(condition, message):
    if not condition:
        raise ValueError(message)


def expect(actual, expected, label):
    require(type(actual) is type(expected) and actual == expected, f"{label} differs")


def counts(value, length, maximum, label):
    require(isinstance(value, list) and len(value) == length
            and all(type(item) is int and 0 <= item <= maximum for item in value),
            f"invalid {label}")
    return value


def validate_records(contract, ownership, audits, environment, steps):
    """Validate complete records with runtime ownership and ordered EP peers."""
    require(type(steps) is int and steps in (2, 25), "only two or25 updates are supported")
    backend = contract.get("communication_backend")
    require(backend in ("alltoall", "uccl_ep"), "fresh explicit communication backend is required")
    for key, value in {"tp": 1, "pp": 1, "cp": 1, "ep": 16, "etp": 1, "dense_dp": 16}.items():
        expect(contract.get("parallelism", {}).get(key), value, f"parallelism.{key}")
    for key, value in {"sequence_length": 4096, "micro_batch_size": 1, "global_batch_size": 64,
                       "gradient_accumulation": 4, "warmup_steps": 5, "measured_steps": 20,
                       "log_interval": 1, "completed_steps": steps}.items():
        expect(contract.get("training", {}).get(key), value, f"training.{key}")
    for key in ("routing_audit", "audit_received_experts"):
        expect(contract.get(key), True, key)
    expect(contract.get("launch_policy"), "torchrun_node4", "launch policy")
    expect(contract.get("model", {}).get("precision"), "bf16", "precision")
    routing = contract.get("routing", {})
    expect(routing.get("topk"), TOPK, "top-k")
    grouped = routing.get("num_groups") is not None
    for key, value in (("num_groups", 4 if grouped else None), ("group_topk", 2 if grouped else None)):
        expect(routing.get(key), value, f"routing.{key}")
        expect(ownership.get(key), value, f"ownership.{key}")
    expect(ownership.get("communication_backend"), backend, "ownership backend")
    expect(ownership.get("record_receives"), True, "ownership receive recording")
    expect(ownership.get("diagnostic_num_groups"), 4, "diagnostic groups")

    hosts = ownership.get("rank_hosts", {})
    require(isinstance(hosts, dict) and set(hosts) == {str(rank) for rank in range(WORLD)}
            and all(type(host) is str and host for host in hosts.values())
            and sorted(Counter(hosts.values()).values()) == [4] * 4, "ownership must describe16 ranks on four physical hosts")
    owners = ownership.get("expert_owners", {})
    require(isinstance(owners, dict) and set(owners) == {str(layer) for layer in range(LAYERS)},
            "ownership must cover48 layers")
    owned = {}
    for layer in range(LAYERS):
        vector = counts(owners[str(layer)], EXPERTS, WORLD - 1, "expert owners")
        require(Counter(vector) == Counter({rank: 8 for rank in range(WORLD)}), "expert ownership must be balanced and complete")
        owned[layer] = [[expert for expert, owner in enumerate(vector) if owner == rank] for rank in range(WORLD)]
        group_owners, group_hosts = [], []
        for group in range(4):
            peers = sorted(set(vector[group * 32:(group + 1) * 32]))
            node_names = sorted({hosts[str(peer)] for peer in peers})
            require(len(peers) == 4 and len(node_names) == 1, "each expert group must cover one four-rank physical host")
            group_owners.append(peers)
            group_hosts.append(node_names)
        require(len({names[0] for names in group_hosts}) == 4, "expert groups must span four distinct physical hosts")
        reported_groups = ownership.get("group_owners", {}).get(str(layer))
        require(isinstance(reported_groups, list) and len(reported_groups) == 4, "diagnostic owner groups must cover four nodes")
        for peers in reported_groups:
            counts(peers, 4, WORLD - 1, "diagnostic group owner ranks")
        expect(reported_groups, group_owners, "diagnostic group owners")
        expect(ownership.get("group_hosts", {}).get(str(layer)), group_hosts, "diagnostic group hosts")

    # The actual ordered process group defines UCCL's four-peer node blocks.
    # Do not infer those blocks from expert IDs or global rank arithmetic.
    expect(environment.get("launch_policy"), "torchrun_node4", "environment launch policy")
    placement = environment.get("rank_placement")
    require(isinstance(placement, list) and len(placement) == WORLD, "environment placement must cover16 ranks")
    by_rank, ep_peers = {}, None
    for row in placement:
        rank = row.get("rank")
        require(type(rank) is int and rank in range(WORLD) and rank not in by_rank, "invalid or duplicate environment rank")
        expect(row.get("hostname"), hosts[str(rank)], "runtime owner host")
        local = counts(row.get("local_expert_indices"), 8, EXPERTS - 1, "runtime local expert IDs")
        require(len(set(local)) == 8 and all(set(local) == set(owned[layer][rank]) for layer in range(LAYERS)),
                "runtime local expert IDs disagree with ownership")
        peers = counts(row.get("process_groups", {}).get("ep"), WORLD, WORLD - 1, "runtime ordered EP ranks")
        require(len(set(peers)) == WORLD, "runtime EP ranks must be distinct")
        if ep_peers is None:
            ep_peers = peers
        expect(peers, ep_peers, "runtime EP rank order")
        by_rank[rank] = row
    node_hosts = []
    if backend == "uccl_ep":
        for first in range(0, WORLD, 4):
            block = {hosts[str(rank)] for rank in ep_peers[first:first + 4]}
            require(len(block) == 1, "actual EP four-peer block spans physical hosts")
            node_hosts.append(next(iter(block)))
        require(len(set(node_hosts)) == 4, "actual EP node blocks must cover four distinct hosts")
        mapping = ownership.get("uccl_node_hosts_in_pg_order", {})
        require(isinstance(mapping, dict) and set(mapping) == {str(layer) for layer in range(LAYERS)}, "missing UCCL node-order proof")
        for layer in range(LAYERS):
            expect(mapping[str(layer)], node_hosts, "UCCL layout physical node order")

    require(isinstance(audits, list) and len(audits) == WORLD, "all16 source audits are required")
    sent = [[[0] * EXPERTS for _ in range(LAYERS)] for _ in range(steps)]
    received = [[[None] * EXPERTS for _ in range(LAYERS)] for _ in range(steps)]
    total_sent = total_received = unique_ranks = unique_nodes = 0
    seen_ranks = set()
    for audit in audits:
        rank = audit.get("rank")
        require(type(rank) is int and rank in range(WORLD) and rank not in seen_ranks, "invalid or duplicate audit rank")
        seen_ranks.add(rank)
        expect(audit.get("valid"), True, "audit validity")
        expect(audit.get("communication_backend"), backend, "audit backend")
        expect(audit.get("record_receives"), True, "audit receive recording")
        for key in ("num_groups", "group_topk"):
            expect(audit.get(key), routing.get(key), f"audit {key}")
        records = audit.get("steps")
        require(isinstance(records, list) and len(records) == steps, "incomplete audit step grid")
        for step, record in enumerate(records):
            expect(record.get("step"), step, "audit step index")
            layers = record.get("layers")
            require(isinstance(layers, list) and len(layers) == LAYERS, "incomplete audit layer grid")
            for layer, row in enumerate(layers):
                expect(row.get("layer"), layer, "audit layer index")
                expert_counts = counts(row.get("expert_counts"), EXPERTS, TOKENS, "source expert counts")
                traffic = counts(row.get("rank_traffic"), WORLD, ASSIGNMENTS, "source rank assignment counts")
                expect(row.get("routed_tokens"), TOKENS, "routed tokens")
                expect(row.get("assignments"), ASSIGNMENTS, "source assignments")
                require(sum(expert_counts) == ASSIGNMENTS, "source expert assignments do not equal tokens times8")
                rebuilt = [sum(expert_counts[expert] for expert in owned[layer][peer]) for peer in range(WORLD)]
                expect(traffic, rebuilt, "source assignment rank traffic")
                local = traffic[rank]
                intra = sum(value for peer, value in enumerate(traffic) if peer != rank and hosts[str(peer)] == hosts[str(rank)])
                inter = sum(value for peer, value in enumerate(traffic) if hosts[str(peer)] != hosts[str(rank)])
                for key, value in (("local", local), ("intra_node", intra), ("inter_node", inter)):
                    expect(row.get(key), value, f"source {key} assignments")
                require(local + intra + inter == ASSIGNMENTS, "source locality does not reconcile")
                ids = counts(row.get("received_expert_ids"), 8, EXPERTS - 1, "received expert IDs")
                require(len(set(ids)) == 8 and set(ids) == set(owned[layer][rank]), "received IDs differ from actual owner")
                values = counts(row.get("received_expert_counts"), 8, WORLD * TOKENS, "received expert counts")
                expect(row.get("received_assignments"), sum(values), "received row total")
                expect(row.get("receive_calls"), ACCUMULATION, "received call coverage")
                for expert, value in enumerate(expert_counts):
                    sent[step][layer][expert] += value
                for expert, value in zip(ids, values):
                    require(received[step][layer][expert] is None, "expert has multiple receive owners")
                    received[step][layer][expert] = value
                total_sent += ASSIGNMENTS
                total_received += sum(values)
                if backend == "uccl_ep":
                    expect(row.get("layout_calls"), ACCUMULATION, "UCCL layout coverage")
                    expect(row.get("metadata_calls"), ACCUMULATION, "UCCL selected metadata coverage")
                    copies = counts(row.get("unique_token_rank_counts"), WORLD, TOKENS, "unique token/rank deliveries")
                    nodes = counts(row.get("unique_token_node_counts"), 4, TOKENS, "unique token/node deliveries")
                    expect(row.get("unique_token_node_hosts"), node_hosts, "UCCL per-layer physical node order")
                    for peer in range(WORLD):
                        require(max(expert_counts[expert] for expert in owned[layer][peer]) <= copies[peer] <= traffic[peer],
                                "unique rank deliveries contradict expert assignments")
                    for node, host in enumerate(node_hosts):
                        peers = [peer for peer in range(WORLD) if hosts[str(peer)] == host]
                        assignments = sum(traffic[peer] for peer in peers)
                        require(max((assignments + TOPK - 1) // TOPK, max(copies[peer] for peer in peers)) <= nodes[node]
                                <= min(assignments, sum(copies[peer] for peer in peers)),
                                "unique node deliveries contradict assignments or rank membership")
                    node_limit = 2 if grouped else min(TOPK, len(node_hosts))
                    require(TOKENS <= sum(nodes) <= node_limit * TOKENS and sum(nodes) <= sum(copies) <= ASSIGNMENTS,
                            "unique delivery totals violate top8/routing bounds")
                    unique_ranks += sum(copies)
                    unique_nodes += sum(nodes)
    for step in range(steps):
        for layer in range(LAYERS):
            require(sent[step][layer] == received[step][layer],
                    f"all-source expert sends differ from actual owner receives at step{step}, layer{layer}")
    require(total_sent == total_received == steps * WORLD * LAYERS * ASSIGNMENTS, "global assignment/receive totals differ")
    return {"schema_version": 1, "kind": "ampa_ep16_received_route_validation", "valid_send_receive_audit": True,
        "communication_backend": backend, "steps": list(range(steps)), "ranks": WORLD, "layers": LAYERS,
        "gradient_accumulation": ACCUMULATION, "source_tokens_per_step_layer": TOKENS,
        "source_assignments_per_step_layer": ASSIGNMENTS, "total_source_expert_assignments": total_sent,
        "total_received_expert_assignments": total_received, "checked_expert_equalities": steps * LAYERS * EXPERTS,
        **({"unique_token_rank_deliveries": unique_ranks, "unique_token_node_deliveries": unique_nodes,
            "node_hosts_in_actual_ep_order": node_hosts} if backend == "uccl_ep" else {}),
        "claim_limit": "Exact aggregate expert send/receive equality and logical delivery bounds; not per-token value/gradient proof, physical traffic, training health or trace completeness. UCCL membership and selected-weight equality require the independently bound online CUDA audit."}


def validate_run(run_dir, steps):
    run_dir = Path(run_dir)
    paths = [run_dir / "config/comparison_contract.json", run_dir / "config/routing_ownership.json",
             run_dir / "environment.json"] + [run_dir / "raw" / f"routing_audit_rank-{rank}.json" for rank in range(WORLD)]
    documents, bindings = [], []
    for path in paths:
        require(path.is_file() and not path.is_symlink(), f"input must be a regular nonsymlink file: {path}")
        payload = path.read_bytes()
        bindings.append({"path": str(path.resolve()), "sha256": hashlib.sha256(payload).hexdigest()})
        documents.append(json.loads(payload))
    result = validate_records(documents[0], documents[1], documents[3:], documents[2], steps)
    for path, binding in zip(paths, bindings):
        require(path.is_file() and not path.is_symlink()
                and hashlib.sha256(path.read_bytes()).hexdigest() == binding["sha256"], f"input changed during validation: {path}")
    result.update(run_dir=str(run_dir.resolve()), input_files=bindings)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, choices=(2, 25), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(os.environ.get("SLURM_JOB_ID") and os.environ.get("SLURM_JOB_PARTITION") == "preemptable", "allocated preemptable compute is required")
    require(not args.output.exists() and not args.output.is_symlink(), "output must be a new unique file")
    require(args.output.resolve() != args.run_dir.resolve() and args.run_dir.resolve() not in args.output.resolve().parents,
            "output must be separate from the source run")
    result = validate_run(args.run_dir, args.steps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"valid_send_receive_audit": True, "communication_backend": result["communication_backend"],
                      "total_received_expert_assignments": result["total_received_expert_assignments"]}), flush=True)


if __name__ == "__main__":
    main()
