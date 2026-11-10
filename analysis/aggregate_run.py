#!/usr/bin/env python3
"""Validate AMPA rank-local routing summaries and consolidate Parquet metrics."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


class ValidationError(RuntimeError):
    """Raised only for malformed invocation; data failures are written to validation.json."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args()


def read_raw_records(run_dir: Path) -> list[dict[str, Any]]:
    paths = sorted((run_dir / "raw").glob("routing-rank-*.json"))
    if not paths:
        raise ValidationError(f"no rank-local routing records found in {run_dir / 'raw'}")
    records = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if any(record.get("schema_version") != 2 for record in records):
        raise ValidationError("unsupported or missing raw routing schema version")
    return records


def _run_coverage_contract(run_dir: Path, records: list[dict[str, Any]]) -> tuple[int, int, int]:
    """Derive expected coverage from run provenance, never from observed counters."""
    contract = json.loads((run_dir / "config" / "comparison_contract.json").read_text(encoding="utf-8"))
    environment = json.loads((run_dir / "environment.json").read_text(encoding="utf-8"))
    parallelism = contract.get("parallelism", {})
    training = contract.get("training", {})
    cp = parallelism.get("cp")
    expected_topology = {
        "world_size": 8, "tp": 1, "pp": 1, "cp": cp, "ep": 8, "etp": 1,
        "dense_dp": 4 if cp == 2 else 8, "expert_dp": 1,
    }
    if cp == 2:
        expected_topology["dense_dp_cp"] = 8
    if cp not in {1, 2} or environment.get("runtime_topology") != expected_topology:
        raise ValidationError("runtime topology does not match the recorded CP1/CP2 EP8 contract")
    if cp == 2 and (
        contract.get("run_kind") != "cp2_ep8_feasibility"
        or contract.get("claim_scope", {}).get("canonical_profile_data") is not False
    ):
        raise ValidationError("CP2 records must retain their separate feasibility-only claim scope")
    if cp == 1 and contract.get("run_kind") != "canonical_indexed_profile":
        raise ValidationError("CP1 records lack the canonical indexed profile contract")
    calls = training.get("gradient_accumulation")
    sequence_length = training.get("sequence_length")
    micro_batch_size = training.get("micro_batch_size")
    if (
        not isinstance(calls, int) or calls <= 0
        or not isinstance(sequence_length, int) or sequence_length <= 0
        or not isinstance(micro_batch_size, int) or micro_batch_size <= 0
        or sequence_length % cp
    ):
        raise ValidationError("invalid recorded gradient-accumulation or CP token-shard contract")
    if (cp, calls, sequence_length, micro_batch_size) not in {(1, 8, 4096, 1), (2, 16, 4096, 1)}:
        raise ValidationError("unexpected CP1/CP2 measurement workload")
    expected_window = (training.get("warmup_steps"), training.get("warmup_steps", 0) + training.get("measured_steps", 0))
    placement = environment.get("rank_placement")
    if not isinstance(placement, list) or len(placement) != 8:
        raise ValidationError("runtime rank placement is missing or incomplete")
    by_rank = {row.get("rank"): row for row in placement}
    if set(by_rank) != set(range(8)):
        raise ValidationError("runtime rank placement has missing or duplicate ranks")
    for record in records:
        rank = record.get("rank")
        if rank not in by_rank or record.get("run_id") != run_dir.name:
            raise ValidationError("raw rank record belongs to a different run")
        if record.get("slurm_job_id") != environment.get("slurm_job_id"):
            raise ValidationError("raw rank record belongs to a different Slurm job")
        if (record.get("profile_start_step"), record.get("profile_stop_step")) != expected_window:
            raise ValidationError("raw profile window differs from the training contract")
        if record.get("expected_calls_per_layer") != calls or record.get("expected_tokens_per_call") != micro_batch_size * sequence_length // cp:
            raise ValidationError("raw route coverage declaration differs from the training contract")
        inventory = record.get("inventory", {})
        metadata = inventory.get("rank_metadata", [])
        if not isinstance(metadata, list) or len(metadata) != 8:
            raise ValidationError("raw all-rank inventory is incomplete")
        for row in metadata:
            runtime = by_rank.get(row.get("rank"))
            if runtime is None or any(
                row.get(raw_key) != runtime.get(env_key)
                for raw_key, env_key in (
                    ("hostname", "hostname"), ("physical_gpu_uuid", "physical_gpu_uuid"),
                    ("slurm_nodeid", "slurm_node_id"), ("local_rank", "slurm_local_id"),
                )
            ):
                raise ValidationError("raw rank placement disagrees with runtime provenance")
            sites = row.get("sites", [])
            if len(sites) != 48 or any(
                site.get("ep_global_ranks") != runtime.get("process_groups", {}).get("ep")
                or site.get("experts") != runtime.get("local_expert_indices")
                for site in sites
            ):
                raise ValidationError("raw MoE ownership disagrees with runtime provenance")
    return calls, calls * micro_batch_size * sequence_length // cp, 48


def validate_and_consolidate(
    records: list[dict[str, Any]], *, expected_calls_per_layer: int | None = None,
    expected_positions_per_rank: int | None = None, expected_layer_count: int | None = None,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    """Check accounting equations while generating normalized metric rows."""
    errors: list[str] = []
    world_sizes = {int(record["world_size"]) for record in records}
    ranks = {int(record["rank"]) for record in records}
    if len(world_sizes) != 1:
        errors.append(f"inconsistent world sizes: {sorted(world_sizes)}")
    world_size = next(iter(world_sizes)) if world_sizes else 0
    if len(records) != world_size or ranks != set(range(world_size)):
        errors.append(f"rank records are {sorted(ranks)}, expected {list(range(world_size))}")
    for record in records:
        errors.extend(f"rank {record['rank']}: {reason}" for reason in record.get("invalid_reasons", []))

    profile_windows = {
        (record.get("profile_start_step"), record.get("profile_stop_step")) for record in records
    }
    if len(profile_windows) != 1 or None in next(iter(profile_windows), (None, None)):
        errors.append(f"inconsistent or missing profile window metadata: {list(profile_windows)}")
        expected_steps: list[int] = []
    else:
        profile_start_step, profile_stop_step = next(iter(profile_windows))
        expected_steps = list(range(int(profile_start_step), int(profile_stop_step)))
        for record in records:
            actual_steps = [int(step["step"]) for step in record.get("steps", [])]
            if actual_steps != expected_steps:
                errors.append(
                    f"rank {record['rank']}: profile steps are {actual_steps}, expected {expected_steps}"
                )

    placement = _validate_inventories(records, world_size, errors)

    metrics: dict[str, list[dict[str, Any]]] = {
        "steps": [],
        "layers": [],
        "experts": [],
        "rank_traffic": [],
    }
    # Without a trusted all-rank inventory, even a syntactically complete
    # traffic matrix cannot be classified as local or inter-node.
    if errors:
        return _validation(errors, world_size, records, expected_steps, metrics), metrics

    by_step_layer: dict[tuple[int, int], list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for record in records:
        for step in record.get("steps", []):
            for layer in step.get("layers", []):
                by_step_layer[(int(step["step"]), int(layer["layer"]))].append((record, {**step, "layer": layer}))
            expected_layers = {int(site["layer"]) for site in record["inventory"]["local_sites"]}
            observed_layers = [int(layer["layer"]) for layer in step.get("layers", [])]
            if not expected_layers or len(observed_layers) != len(expected_layers) or set(observed_layers) != expected_layers:
                errors.append(
                    f"step {step['step']}, rank {record['rank']}: observed MoE layers "
                    f"{observed_layers}, expected {sorted(expected_layers)}"
                )
            if expected_layer_count is not None and expected_layers != set(range(expected_layer_count)):
                errors.append(f"rank {record['rank']}: expected {expected_layer_count} MoE layers")

    step_totals: dict[int, dict[str, int | bool]] = defaultdict(
        lambda: {"assignments": 0, "local": 0, "intra": 0, "inter": 0, "skipped": False}
    )
    for (step, layer), entries in sorted(by_step_layer.items()):
        if len(entries) != world_size:
            errors.append(f"step {step}, layer {layer}: found {len(entries)} rank rows, expected {world_size}")
            continue
        expected_ranks = {int(record["rank"]) for record, _ in entries}
        if expected_ranks != set(range(world_size)):
            errors.append(f"step {step}, layer {layer}: rank rows are incomplete")
            continue

        expert_totals: list[int] | None = None
        global_assignments = 0
        probability_sum = 0.0
        probability_square_sum = 0.0
        routed_token_positions = 0
        all_preprocess_calls: list[int] = []
        local_count = intra_count = inter_count = 0
        topk_values: set[int] = set()
        incoming_by_rank: dict[int, int] = defaultdict(int)
        owned_receiver_by_rank: dict[int, int] = {}
        owned_experts: set[int] = set()
        owner_rank_by_expert: dict[int, int] = {}
        ep_rank_orders: set[tuple[int, ...]] = set()
        for record, merged in entries:
            rank = int(record["rank"])
            row = merged["layer"]
            if merged.get("skipped_iter"):
                errors.append(f"step {step}, layer {layer}, rank {rank}: optimizer step was skipped")
            if merged.get("loss_is_finite") is not True:
                errors.append(f"step {step}, layer {layer}, rank {rank}: loss was absent or non-finite")
            grad_norm = merged.get("grad_norm")
            if grad_norm is None or not math.isfinite(float(grad_norm)):
                errors.append(f"step {step}, layer {layer}, rank {rank}: gradient norm was absent or non-finite")
            try:
                site = _site_for_layer(record["inventory"], layer)
            except ValidationError as error:
                errors.append(f"step {step}, layer {layer}, rank {rank}: {error}")
                continue
            topk_values.add(int(site["router_topk"]))
            destinations = [int(value) for value in row["destination_counts"]]
            destination_ranks = [int(value) for value in site["ep_global_ranks"]]
            ep_rank_orders.add(tuple(destination_ranks))
            assignment_count = int(row["assignment_count"])
            token_positions = int(row["routed_token_positions"])
            preprocess_calls = int(row["preprocess_calls"])
            route_calls = row.get("route_calls")
            route_positions = row.get("route_token_positions")
            if expected_calls_per_layer is not None and (
                route_calls != expected_calls_per_layer or preprocess_calls != expected_calls_per_layer
            ):
                errors.append(
                    f"step {step}, layer {layer}, rank {rank}: route/dispatch calls "
                    f"{route_calls}/{preprocess_calls} != expected {expected_calls_per_layer}"
                )
            if expected_positions_per_rank is not None and (
                route_positions != expected_positions_per_rank or token_positions != expected_positions_per_rank
            ):
                errors.append(
                    f"step {step}, layer {layer}, rank {rank}: route/dispatch token positions "
                    f"{route_positions}/{token_positions} != expected {expected_positions_per_rank}"
                )
            if assignment_count < 0 or token_positions < 0 or preprocess_calls < 0:
                errors.append(f"step {step}, layer {layer}, rank {rank}: negative count")
            if any(value < 0 for value in destinations):
                errors.append(f"step {step}, layer {layer}, rank {rank}: negative destination count")
            if assignment_count != token_positions * int(site["router_topk"]):
                errors.append(
                    f"step {step}, layer {layer}, rank {rank}: assignments {assignment_count} != "
                    f"routed token positions {token_positions} × top-k {site['router_topk']}"
                )
            if sum(destinations) != assignment_count:
                errors.append(
                    f"step {step}, layer {layer}, rank {rank}: destination sum != assignment count"
                )
            if len(destinations) != len(destination_ranks) or set(destination_ranks) != set(range(world_size)):
                errors.append(f"step {step}, layer {layer}, rank {rank}: malformed destination vector")
                continue
            receiver_counts = [int(value) for value in row["receiver_expert_counts"]]
            if any(value < 0 for value in receiver_counts):
                errors.append(f"step {step}, layer {layer}, rank {rank}: negative receiver expert count")
            if expert_totals is None:
                expert_totals = [0] * len(receiver_counts)
            if len(receiver_counts) != len(expert_totals):
                errors.append(f"step {step}, layer {layer}, rank {rank}: expert-count shape mismatch")
                continue
            local_experts = [int(value) for value in site["experts"]]
            if (
                len(local_experts) != len(set(local_experts))
                or any(expert < 0 or expert >= len(receiver_counts) for expert in local_experts)
                or owned_experts.intersection(local_experts)
            ):
                errors.append(f"step {step}, layer {layer}, rank {rank}: invalid or duplicate expert ownership")
                continue
            owned_experts.update(local_experts)
            owner_rank_by_expert.update({expert: rank for expert in local_experts})
            owned_receiver_by_rank[rank] = sum(receiver_counts[expert] for expert in local_experts)
            if any(count for expert, count in enumerate(receiver_counts) if expert not in local_experts):
                errors.append(f"step {step}, layer {layer}, rank {rank}: non-owned receiver expert count is nonzero")
            for expert, value in enumerate(receiver_counts):
                expert_totals[expert] += value
            global_assignments += assignment_count
            probability_sum += float(row["probability_sum"])
            probability_square_sum += float(row["probability_square_sum"])
            if (
                not math.isfinite(float(row["probability_sum"]))
                or not math.isfinite(float(row["probability_square_sum"]))
                or float(row["probability_sum"]) < 0
                or float(row["probability_square_sum"]) < 0
            ):
                errors.append(f"step {step}, layer {layer}, rank {rank}: invalid router probability moments")
            routed_token_positions += token_positions
            all_preprocess_calls.append(preprocess_calls)
            source_host = placement[rank]["hostname"]
            for destination_rank, count in zip(destination_ranks, destinations):
                incoming_by_rank[destination_rank] += count
                destination_host = placement[destination_rank]["hostname"]
                traffic_class = _traffic_class(rank, destination_rank, source_host, destination_host)
                if traffic_class == "local":
                    local_count += count
                elif traffic_class == "intra_node":
                    intra_count += count
                else:
                    inter_count += count
                metrics["rank_traffic"].append(
                    {
                        "step": step,
                        "layer": layer,
                        "src_rank": rank,
                        "dst_rank": destination_rank,
                        "assignments": count,
                        "measurement": "forward_dispatch_routed_assignments",
                        "unit": "token_expert_assignments",
                        "traffic_class": traffic_class,
                        "src_hostname": source_host,
                        "dst_hostname": destination_host,
                    }
                )

        if len(topk_values) != 1:
            errors.append(f"step {step}, layer {layer}: inconsistent router top-k values {sorted(topk_values)}")
            continue
        if len(ep_rank_orders) != 1:
            errors.append(f"step {step}, layer {layer}: inconsistent EP rank ordering")
            continue
        topk = next(iter(topk_values))
        if topk <= 0 or global_assignments != routed_token_positions * topk:
            errors.append(
                f"step {step}, layer {layer}: assignments {global_assignments} != routed token positions "
                f"{routed_token_positions} × top-k {topk}"
            )
        if expected_positions_per_rank is not None and routed_token_positions != world_size * expected_positions_per_rank:
            errors.append(
                f"step {step}, layer {layer}: global routed positions {routed_token_positions} != "
                f"expected {world_size * expected_positions_per_rank}"
            )
        if expert_totals is None:
            errors.append(f"step {step}, layer {layer}: missing expert counts")
            continue
        if not expert_totals or owned_experts != set(range(len(expert_totals))):
            errors.append(f"step {step}, layer {layer}: expert ownership is incomplete")
            continue
        if not all_preprocess_calls:
            errors.append(f"step {step}, layer {layer}: no valid rank-local dispatcher records")
            continue
        owner_by_layer = records[0]["inventory"]["owner_by_layer"]
        reported_owners = owner_by_layer.get(str(layer)) if isinstance(owner_by_layer, dict) else None
        if reported_owners != [owner_rank_by_expert[expert] for expert in range(len(expert_totals))]:
            errors.append(f"step {step}, layer {layer}: expert ownership inventory disagrees with local sites")
        for destination_rank in range(world_size):
            incoming = incoming_by_rank[destination_rank]
            received = owned_receiver_by_rank.get(destination_rank)
            if received != incoming:
                errors.append(
                    f"step {step}, layer {layer}, destination rank {destination_rank}: "
                    f"incoming assignments {incoming} != owned receiver expert count {received}"
                )
        if sum(expert_totals) != global_assignments:
            errors.append(
                f"step {step}, layer {layer}: receiver expert total {sum(expert_totals)} != traffic total {global_assignments}"
            )
        if local_count + intra_count + inter_count != global_assignments:
            errors.append(f"step {step}, layer {layer}: traffic classes do not sum to total assignments")
        mean_expert_load = global_assignments / len(expert_totals)
        max_expert_load = max(expert_totals)
        variance = sum((count - mean_expert_load) ** 2 for count in expert_totals) / len(expert_totals)
        metrics["layers"].append(
            {
                "step": step,
                "layer": layer,
                "top_k": topk,
                "routed_token_positions": routed_token_positions,
                "assignments": global_assignments,
                "local_assignments": local_count,
                "intra_node_assignments": intra_count,
                "inter_node_assignments": inter_count,
                "local_fraction": _fraction(local_count, global_assignments),
                "intra_node_fraction": _fraction(intra_count, global_assignments),
                "inter_node_fraction": _fraction(inter_count, global_assignments),
                "expert_load_mean": mean_expert_load,
                "expert_load_max": max_expert_load,
                "expert_load_cv": math.sqrt(variance) / mean_expert_load if mean_expert_load else None,
                "probability_sum": probability_sum,
                "probability_square_sum": probability_square_sum,
                "preprocess_calls_min": min(all_preprocess_calls),
                "preprocess_calls_max": max(all_preprocess_calls),
            }
        )
        for expert, count in enumerate(expert_totals):
            metrics["experts"].append({"step": step, "layer": layer, "expert": expert, "assignments": count})
        totals = step_totals[step]
        totals["assignments"] = int(totals["assignments"]) + global_assignments
        totals["local"] = int(totals["local"]) + local_count
        totals["intra"] = int(totals["intra"]) + intra_count
        totals["inter"] = int(totals["inter"]) + inter_count
        totals["skipped"] = bool(totals["skipped"]) or any(
            bool(merged.get("skipped_iter")) for _, merged in entries
        )

    for step, totals in sorted(step_totals.items()):
        assignments = int(totals["assignments"])
        metrics["steps"].append(
            {
                "step": step,
                "valid_profile": not errors and not bool(totals["skipped"]),
                "skipped_iter": bool(totals["skipped"]),
                "assignments": assignments,
                "local_assignments": int(totals["local"]),
                "intra_node_assignments": int(totals["intra"]),
                "inter_node_assignments": int(totals["inter"]),
                "local_fraction": _fraction(int(totals["local"]), assignments),
                "intra_node_fraction": _fraction(int(totals["intra"]), assignments),
                "inter_node_fraction": _fraction(int(totals["inter"]), assignments),
            }
        )
    validation = _validation(errors, world_size, records, expected_steps, metrics)
    return validation, metrics


def _validate_inventories(
    records: list[dict[str, Any]], world_size: int, errors: list[str]
) -> dict[int, dict[str, Any]]:
    """Require all ranks to report the same complete, physical placement."""
    first = records[0].get("inventory", {}) if records else {}
    rank_metadata = first.get("rank_metadata") if isinstance(first, dict) else None
    if not isinstance(rank_metadata, list) or len(rank_metadata) != world_size:
        errors.append("rank placement metadata is incomplete")
        return {}
    placement: dict[int, dict[str, Any]] = {}
    for row in rank_metadata:
        if not isinstance(row, dict) or not isinstance(row.get("rank"), int):
            errors.append("rank placement metadata contains an invalid rank")
            continue
        rank = row["rank"]
        if rank in placement:
            errors.append(f"rank placement metadata duplicates rank {rank}")
        placement[rank] = row
        if not isinstance(row.get("hostname"), str) or not row["hostname"]:
            errors.append(f"rank {rank}: missing physical hostname")
        if not isinstance(row.get("physical_gpu_uuid"), str) or not row["physical_gpu_uuid"]:
            errors.append(f"rank {rank}: missing physical GPU UUID")
    if set(placement) != set(range(world_size)):
        errors.append("rank placement metadata is incomplete")
    gpu_uuids = [row.get("physical_gpu_uuid") for row in placement.values() if isinstance(row.get("physical_gpu_uuid"), str)]
    if len(set(gpu_uuids)) != len(gpu_uuids):
        errors.append("rank placement repeats a physical GPU UUID")
    if world_size == 8:
        host_counts = Counter(
            row["hostname"] for row in placement.values() if isinstance(row.get("hostname"), str)
        )
        if sorted(host_counts.values()) != [4, 4]:
            errors.append(f"EP8 rank placement is not four GPUs on each of two nodes: {dict(host_counts)}")
    first_owners = first.get("owner_by_layer") if isinstance(first, dict) else None
    if not isinstance(first_owners, dict):
        errors.append("missing expert ownership inventory")
    for record in records:
        rank = int(record["rank"])
        inventory = record.get("inventory")
        if not isinstance(inventory, dict):
            errors.append(f"rank {rank}: missing inventory")
            continue
        if inventory.get("rank") != rank or inventory.get("world_size") != world_size:
            errors.append(f"rank {rank}: inventory rank/world size disagrees with record")
        if inventory.get("rank_metadata") != rank_metadata:
            errors.append(f"rank {rank}: inconsistent all-rank placement inventory")
        local_sites = inventory.get("local_sites")
        if not isinstance(local_sites, list) or local_sites != placement.get(rank, {}).get("sites"):
            errors.append(f"rank {rank}: local MoE sites disagree with placement inventory")
        if inventory.get("owner_by_layer") != first_owners:
            errors.append(f"rank {rank}: inconsistent expert ownership inventory")
    return placement


def _validation(
    errors: list[str],
    world_size: int,
    records: list[dict[str, Any]],
    expected_steps: list[int],
    metrics: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    return {
        "valid_profile": not errors,
        "errors": errors,
        "world_size": world_size,
        "rank_records": len(records),
        "measured_steps": sorted({int(row["step"]) for row in metrics["steps"]}),
        "expected_steps": expected_steps,
        "invariant_definitions": {
            "traffic": "each destination rank's incoming assignments equal its owned receiver expert counts",
            "topk": "each source rank's assignments equal its routed token positions times runtime router top-k",
            "topology": "local + intra_node + inter_node equals total assignments using agreed physical placement",
            "probabilities": "probability summaries are descriptive; routing_map-derived assignments are exact",
        },
    }


def _site_for_layer(inventory: dict[str, Any], layer: int) -> dict[str, Any]:
    for site in inventory.get("local_sites", []):
        if int(site["layer"]) == layer:
            return site
    raise ValidationError(f"inventory has no local metadata for layer {layer}")


def _traffic_class(source_rank: int, destination_rank: int, source_host: str, destination_host: str) -> str:
    if source_rank == destination_rank:
        return "local"
    if source_host == destination_host:
        return "intra_node"
    return "inter_node"


def _fraction(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def write_parquet(metrics_dir: Path, metrics: dict[str, list[dict[str, Any]]]) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ModuleNotFoundError as error:
        raise ValidationError("pyarrow is required to create the requested Parquet output") from error
    metrics_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in metrics.items():
        table = pa.Table.from_pylist(rows)
        pq.write_table(table, metrics_dir / f"{name}.parquet")


def write_summaries(run_dir: Path, validation: dict[str, Any], metrics: dict[str, list[dict[str, Any]]]) -> None:
    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    layers = metrics["layers"]
    topology = {
        "valid_profile": validation["valid_profile"],
        "measurement": "forward_dispatch_routed_assignments",
        "unit": "token_expert_assignments",
        "limitation": "These assignment counts are not bytes or total EP communication volume.",
        "mean_inter_node_fraction": _mean(row["inter_node_fraction"] for row in layers),
        "mean_intra_node_fraction": _mean(row["intra_node_fraction"] for row in layers),
        "mean_local_fraction": _mean(row["local_fraction"] for row in layers),
    }
    routing = {
        "valid_profile": validation["valid_profile"],
        "mean_expert_load_cv": _mean(row["expert_load_cv"] for row in layers),
        "max_expert_load": max((row["expert_load_max"] for row in layers), default=None),
    }
    (analysis_dir / "topology_summary.json").write_text(json.dumps(topology, indent=2), encoding="utf-8")
    (analysis_dir / "routing_summary.json").write_text(json.dumps(routing, indent=2), encoding="utf-8")
    (analysis_dir / "stability_summary.json").write_text(
        json.dumps({"valid_profile": validation["valid_profile"], "note": "computed from per-step Parquet rows"}, indent=2),
        encoding="utf-8",
    )


def _mean(values: Any) -> float | None:
    filtered = [float(value) for value in values if value is not None]
    return sum(filtered) / len(filtered) if filtered else None


def main() -> None:
    args = parse_args()
    try:
        records = read_raw_records(args.run_dir)
        calls, positions, layers = _run_coverage_contract(args.run_dir, records)
        validation, metrics = validate_and_consolidate(
            records, expected_calls_per_layer=calls,
            expected_positions_per_rank=positions, expected_layer_count=layers,
        )
    except (ValidationError, ValueError, KeyError, TypeError, json.JSONDecodeError, FileNotFoundError) as error:
        validation = {"valid_profile": False, "errors": [str(error)]}
        (args.run_dir / "validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
        raise SystemExit("AMPA profile input/provenance failed; see validation.json") from error
    (args.run_dir / "validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    write_parquet(args.run_dir / "metrics", metrics)
    write_summaries(args.run_dir, validation, metrics)
    if not validation["valid_profile"]:
        raise SystemExit("AMPA profile accounting failed; see validation.json")


if __name__ == "__main__":
    main()
