#!/usr/bin/env python3
"""Measure labeled MoE host spans and correlated CUDA work in a valid trace.

The Chrome trace is large. Run only on an allocated compute node. Reported
CUDA durations are activity launched by a stage, not exposed step latency.
Nested CPU scopes and overlapping CUDA streams must never be summed as a
partition of complete optimizer-step time.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gc
import gzip
import hashlib
import json
from pathlib import Path
import re
import statistics


LABEL = re.compile(r"^AMPA/FWD/layer_(\d{2})/(.+)$")
TRAIN_LABEL = re.compile(r"^AMPA/TRAIN/(.+)$")
EP_FORWARD_PAYLOADS = {
    "dispatch_activations_a2a", "dispatch_probabilities_a2a", "combine_expert_output_a2a",
}


def _is_labeled_scope(name: str) -> bool:
    return bool(LABEL.fullmatch(name) or TRAIN_LABEL.fullmatch(name))


def _phase(name: str) -> str:
    match = LABEL.fullmatch(name)
    if match:
        return match.group(2)
    train = TRAIN_LABEL.fullmatch(name)
    return f"train_{train.group(1)}" if train else name


def _collective_kernel(event: dict) -> bool:
    """Recognize NCCL and collective GPU work without guessing its process group.

    Some collective-tagged one-rank reducers omit ``nccl`` in the symbol.
    Context-parallel peer SendRecv kernels have an NCCL symbol but no
    collective/process-group metadata; their CPU owner is checked separately.
    """
    if event.get("cat") != "kernel":
        return False
    args = event.get("args", {})
    named = "nccl" in event.get("name", "").lower()
    collective = args.get("Collective name")
    group = args.get("Process Group Description")
    if bool(collective) != bool(group):
        raise ValueError("collective GPU kernel has incomplete process-group metadata")
    return bool(named or (collective and group))


def _resolve_forward_driver_metadata(events: list[dict], distributed: dict,
                                     selected_steps: tuple[int, ...]) -> tuple[list[dict], dict]:
    """Recover absent forward EP metadata through one exact driver/CPU chain.

    Only GPU event copies receive derived metadata. CPU identities, raw GPU
    External IDs, and the raw runtime route remain unchanged. No timestamp
    proximity, neighboring kernel, or backward inference is a recovery route.
    """
    metadata_keys = {
        "Collective name", "Process Group Description", "Process Group Name",
        "Process Group Ranks", "Group size", "Rank", "dtype", "In msg nelems",
        "Out msg nelems", "In split size", "Out split size",
    }
    runtime_correlations = {event.get("args", {}).get("correlation") for event in events
                            if event.get("cat") == "cuda_runtime"}
    candidates = []
    for index, event in enumerate(events):
        args = event.get("args", {})
        if (event.get("ph") != "X" or event.get("cat") != "kernel"
                or event.get("dur", 0) <= 0 or "nccl" not in event.get("name", "").lower()
                or "External id" in args
                or args.get("correlation") in runtime_correlations
                or (args.get("Collective name") and args.get("Process Group Description"))):
            continue
        if metadata_keys.intersection(args):
            raise ValueError("driver-forward recovery rejects partial or conflicting GPU collective metadata")
        candidates.append((index, event))
    if not candidates:
        return events, {}

    def contains(parent, child):
        return (parent.get("pid") == child.get("pid") and parent["ts"] <= child["ts"]
                and child["ts"] + child["dur"] <= parent["ts"] + parent["dur"])

    def overlaps(left, right):
        return left["ts"] < right["ts"] + right["dur"] and right["ts"] < left["ts"] + left["dur"]

    def describe(event):
        return {"name": event["name"], "category": event.get("cat"),
                "pid": event.get("pid"), "tid": event.get("tid"),
                "ts_us": event["ts"], "duration_us": event["dur"],
                "args": dict(event.get("args", {}))}

    def valid_group_name(value):
        return ((type(value) is int and value >= 0)
                or (isinstance(value, str) and bool(value.strip())))

    drivers = defaultdict(list)
    comms = defaultdict(list)
    labels = defaultdict(list)
    cpu_ids = Counter()
    steps = []
    selected_names = {f"AMPA/STEP/{number}" for number in selected_steps}
    for event in events:
        if event.get("ph") != "X":
            continue
        thread = (event.get("pid"), event.get("tid"))
        if event.get("cat") == "cuda_driver":
            drivers[event.get("args", {}).get("correlation")].append(event)
        elif event.get("cat") == "cpu_op":
            external = event.get("args", {}).get("External id")
            if external is not None:
                cpu_ids[external] += 1
            if event.get("name") == "record_param_comms":
                comms[thread].append(event)
        elif event.get("cat") == "user_annotation":
            if event.get("name") in selected_names:
                steps.append(event)
            if _is_labeled_scope(event.get("name", "")):
                labels[thread].append(event)
    resolved = list(events)
    recoveries = {}
    for index, kernel in candidates:
        correlation = kernel.get("args", {}).get("correlation")
        matching_drivers = drivers.get(correlation, [])
        if (type(correlation) is not int or correlation <= 0 or len(matching_drivers) != 1
                or matching_drivers[0].get("name") not in {"cuLaunchKernel", "cuLaunchKernelEx"}
                or matching_drivers[0].get("dur", 0) <= 0):
            raise ValueError("driver-forward recovery requires one exact positive CUDA driver kernel launch")
        driver = matching_drivers[0]
        thread = (driver.get("pid"), driver.get("tid"))
        matching_comms = [comm for comm in comms.get(thread, []) if overlaps(comm, driver)]
        if (len(matching_comms) != 1 or matching_comms[0].get("dur", 0) <= 0
                or not contains(matching_comms[0], driver)):
            raise ValueError("driver-forward recovery requires one fully containing same-thread record_param_comms")
        comm = matching_comms[0]
        metadata = comm.get("args", {})
        external = metadata.get("External id")
        if type(external) is not int or external <= 0 or cpu_ids[external] != 1:
            raise ValueError("driver-forward recovery requires a globally unique positive CPU External id")
        matching_payloads = [scope for scope in labels.get(thread, [])
                             if (match := LABEL.fullmatch(scope["name"]))
                             and match.group(2) in EP_FORWARD_PAYLOADS and contains(scope, comm)]
        if (len(matching_payloads) != 1 or any(
                overlaps(scope, comm) and not contains(scope, comm) for scope in labels.get(thread, []))):
            raise ValueError("driver-forward recovery requires one fully containing exact AMPA forward EP payload")
        payload = matching_payloads[0]
        matching_steps = [step for step in steps if contains(step, comm) and contains(step, driver)
                          and contains(step, payload)]
        if len(matching_steps) != 1:
            raise ValueError("driver-forward recovery chain must be fully inside one selected optimizer step")
        selected_step = matching_steps[0]

        ranks = metadata.get("Process Group Ranks")
        if isinstance(ranks, str):
            try:
                ranks = json.loads(ranks)
            except json.JSONDecodeError:
                raise ValueError("driver-forward recovery has invalid CPU EP rank metadata") from None
        rank, world = distributed.get("rank"), distributed.get("world_size")
        group_name = metadata.get("Process Group Name")
        if (type(world) is not int or world not in (8, 16) or type(rank) is not int
                or rank not in range(world) or type(metadata.get("Rank")) is not int
                or metadata["Rank"] != rank or type(metadata.get("Group size")) is not int
                or metadata["Group size"] != world or ranks != list(range(world))
                or not isinstance(ranks, list) or any(type(value) is not int for value in ranks)
                or metadata.get("Collective name") != "all_to_allv"
                or metadata.get("Process Group Description") != "EXPERT_MODEL_PARALLEL_GROUP"
                or not valid_group_name(group_name)):
            raise ValueError("driver-forward recovery CPU collective/EP topology metadata disagrees")
        matching_groups = [group for group in distributed.get("pg_config", [])
                           if valid_group_name(group.get("pg_name"))
                           and str(group["pg_name"]) == str(group_name)]
        if (len(matching_groups) != 1
                or matching_groups[0].get("pg_desc") != metadata["Process Group Description"]
                or type(matching_groups[0].get("pg_size")) is not int
                or matching_groups[0]["pg_size"] != world
                or not isinstance(matching_groups[0].get("ranks"), list)
                or any(type(value) is not int for value in matching_groups[0]["ranks"])
                or matching_groups[0]["ranks"] != ranks):
            raise ValueError("driver-forward recovery CPU EP process group differs from distributedInfo")
        derived_metadata = {key: value for key, value in metadata.items() if key in metadata_keys}
        recovered = dict(kernel)
        recovered["args"] = {**kernel.get("args", {}), **derived_metadata}
        resolved[index] = recovered
        recoveries[id(recovered)] = {
            "owner": comm, "driver": driver, "payload": payload, "step": selected_step,
            "provenance": {
                "evidence_route": "exact_driver_correlation_to_unique_cpu_comms_to_forward_ep_payload",
                "gpu": describe(kernel), "driver": describe(driver),
                "cpu_record_param_comms": describe(comm), "payload_scope": describe(payload),
                "selected_step": describe(selected_step),
                "validated_process_group": {**matching_groups[0], "rank": rank, "world_size": world},
                "derived_gpu_metadata": derived_metadata,
                "original_gpu_runtime_present": False, "original_gpu_external_id_present": False,
                "original_gpu_collective_metadata_present": False,
            },
        }
    return resolved, recoveries


def _union_us(intervals: list[tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
    intervals.sort()
    start, end = intervals[0]
    total = 0.0
    for left, right in intervals[1:]:
        if left > end:
            total += end - start
            start, end = left, right
        else:
            end = max(end, right)
    return total + end - start


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = fraction * (len(ordered) - 1)
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def _runtime_scope_events(events: list[dict]) -> dict[int, list[dict]]:
    """Map each unique CUDA launch correlation to containing AMPA scopes."""
    by_thread: dict[tuple[int, int], list[dict]] = defaultdict(list)
    launches: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for event in events:
        if event.get("ph") != "X":
            continue
        thread = (event.get("pid"), event.get("tid"))
        if event.get("cat") == "user_annotation" and _is_labeled_scope(event.get("name", "")):
            by_thread[thread].append(event)
        elif event.get("cat") == "cuda_runtime":
            launches[thread].append(event)
    result: dict[int, list[dict]] = {}
    seen_correlations: set[int] = set()
    for thread, runtime_events in launches.items():
        scopes = sorted(by_thread.get(thread, []), key=lambda event: (event["ts"], -event["dur"]))
        active: list[dict] = []
        next_scope = 0
        for event in sorted(runtime_events, key=lambda item: item["ts"]):
            timestamp = event["ts"]
            while next_scope < len(scopes) and scopes[next_scope]["ts"] <= timestamp:
                active.append(scopes[next_scope])
                next_scope += 1
            active = [scope for scope in active if timestamp < scope["ts"] + scope["dur"]]
            correlation = event.get("args", {}).get("correlation")
            if correlation is not None:
                if correlation in seen_correlations:
                    raise ValueError(f"ambiguous duplicate CUDA runtime correlation {correlation}")
                seen_correlations.add(correlation)
            if correlation is not None and active:
                result[correlation] = list(active)
    return result


def _runtime_scopes(events: list[dict]) -> dict[int, list[str]]:
    return {correlation: [scope["name"] for scope in scopes]
            for correlation, scopes in _runtime_scope_events(events).items()}


def _external_cpu_scope_events(events: list[dict]) -> tuple[dict[int, list[dict]], dict[int, dict]]:
    """Find the unique CPU op linked to a GPU External id, then its AMPA scopes.

    CUDA driver events share an External id with the CPU op; they are not the
    semantic launch owner and are deliberately excluded here.
    """
    by_thread: dict[tuple[int, int], list[dict]] = defaultdict(list)
    cpu_ops: list[dict] = []
    cpu_ids: Counter[int] = Counter()
    for event in events:
        if event.get("ph") != "X":
            continue
        if event.get("cat") == "user_annotation" and _is_labeled_scope(event.get("name", "")):
            by_thread[(event.get("pid"), event.get("tid"))].append(event)
        elif event.get("cat") == "cpu_op":
            external = event.get("args", {}).get("External id")
            if external is not None:
                cpu_ops.append(event)
                cpu_ids[external] += 1
    operations: dict[tuple[int, int], list[dict]] = defaultdict(list)
    unique_cpu_ops: dict[int, dict] = {}
    for event in cpu_ops:
        if cpu_ids[event["args"]["External id"]] == 1:
            operations[(event.get("pid"), event.get("tid"))].append(event)
            unique_cpu_ops[event["args"]["External id"]] = event
    result: dict[int, list[dict]] = {}
    for thread, host_events in operations.items():
        scopes = sorted(by_thread.get(thread, []), key=lambda item: (item["ts"], -item["dur"]))
        active: list[dict] = []
        next_scope = 0
        for event in sorted(host_events, key=lambda item: item["ts"]):
            left, right = event["ts"], event["ts"] + event["dur"]
            while next_scope < len(scopes) and scopes[next_scope]["ts"] <= left:
                active.append(scopes[next_scope])
                next_scope += 1
            active = [scope for scope in active if left < scope["ts"] + scope["dur"]]
            # Full interval containment is necessary: a CPU op crossing a
            # stage boundary must not assign its kernels to either stage.
            # If the operation crosses even one nested stage boundary, the
            # remaining parent must not silently inherit its GPU kernels.
            result[event["args"]["External id"]] = (
                list(active) if all(right <= scope["ts"] + scope["dur"] for scope in active)
                else []
            )
    return result, unique_cpu_ops


def _backward_operation_contexts(events: list[dict], step_scopes: list[dict]) -> tuple[dict, dict]:
    """Resolve host ownership through explicit autograd engine ancestry.

    Time containment alone never establishes backward work. An owner must be
    fully inside a same-thread autograd engine node, and that entire node must
    be inside one same-process backward call and selected optimizer step.
    """
    calls = [event for event in events if event.get("ph") == "X"
             and event.get("cat") == "user_annotation"
             and event.get("name") == "AMPA/TRAIN/microbatch_backward"]
    def contains(parent, child):
        return (parent.get("pid") == child.get("pid") and parent["ts"] <= child["ts"]
                and child["ts"] + child["dur"] <= parent["ts"] + parent["dur"])
    nodes = {}
    by_thread = defaultdict(list)
    owners = defaultdict(list)
    for event in events:
        if event.get("ph") != "X" or event.get("dur", 0) <= 0:
            continue
        thread = (event.get("pid"), event.get("tid"))
        if event.get("cat") in {"cpu_op", "cuda_runtime", "cuda_driver"}:
            owners[thread].append(event)
        if (event.get("cat") == "cpu_op"
                and event.get("name", "").startswith("autograd::engine::evaluate_function:")):
            matching_calls = [call for call in calls if contains(call, event)]
            matching_steps = [step for step in step_scopes if contains(step, event)]
            if (len(matching_calls) == len(matching_steps) == 1
                    and contains(matching_steps[0], matching_calls[0])):
                nodes[id(event)] = {"node": event, "scope": matching_calls[0], "step": matching_steps[0]}
                by_thread[thread].append(event)
    contexts = {}
    for thread, operations in owners.items():
        engines = sorted(by_thread.get(thread, []), key=lambda row: (row["ts"], -row["dur"]))
        active, position = [], 0
        for owner in sorted(operations, key=lambda row: (row["ts"], -row["dur"])):
            while position < len(engines) and engines[position]["ts"] <= owner["ts"]:
                active.append(engines[position])
                position += 1
            active = [node for node in active if owner["ts"] < node["ts"] + node["dur"]]
            candidates = [node for node in active if contains(node, owner)]
            if candidates:
                # Nested engine nodes must agree on their enclosing call.
                if len({id(nodes[id(node)]["scope"]) for node in candidates}) != 1:
                    raise ValueError("ambiguous autograd backward-call ownership")
                ordered = sorted(candidates, key=lambda node: (node["ts"], -node["dur"]))
                if any(not contains(parent, child) for parent, child in zip(ordered, ordered[1:])):
                    raise ValueError("autograd engine scopes overlap without nesting")
                deepest = min(candidates, key=lambda node: node["dur"])
                if sum(node["dur"] == deepest["dur"] for node in candidates) != 1:
                    raise ValueError("ambiguous duplicate autograd engine ownership")
                contexts[id(owner)] = nodes[id(deepest)]
    return contexts, nodes


def _ep_forward_backward_flow_links(events, backward_contexts, backward_nodes):
    """Resolve exact exported CPU flow endpoints; never equate logical/OS tids."""
    payloads = [event for event in events if event.get("ph") == "X"
                and event.get("cat") == "user_annotation"
                and event.get("name", "").startswith("AMPA/FWD/")
                and event["name"].endswith("_a2a")]
    targets = {key: context for key, context in backward_nodes.items()
               if context["node"]["name"] == "autograd::engine::evaluate_function: _AllToAllBackward"}
    if not targets:
        # Generic aggregation can inspect forward-only records. The training
        # validator separately requires the expected backward node count.
        return {}, {}
    starts = defaultdict(list)
    flows = defaultdict(list)
    forward_nodes = []
    for event in events:
        if event.get("ph") == "X" and event.get("cat") == "cpu_op":
            starts[(event.get("pid"), event.get("tid"), event.get("ts"))].append(event)
            if event.get("name") == "_AllToAll":
                forward_nodes.append(event)
        if event.get("cat") == "fwdbwd" and event.get("ph") in {"s", "f"}:
            flows[(event.get("pid"), event.get("id"))].append(event)
    def contains(parent, child):
        return (parent.get("pid") == child.get("pid") and parent.get("tid") == child.get("tid")
                and parent["ts"] <= child["ts"] and child["ts"] + child["dur"] <= parent["ts"] + parent["dur"])
    matched, origin_counts = {}, Counter()
    for pair in flows.values():
        endings = [row for row in pair if row["ph"] == "f"]
        for endpoint in endings:
            candidates = starts[(endpoint.get("pid"), endpoint.get("tid"), endpoint.get("ts"))]
            ep_owners = [backward_contexts[id(candidate)] for candidate in candidates
                         if id(candidate) in backward_contexts and id(backward_contexts[id(candidate)]["node"]) in targets]
            if not ep_owners:
                continue
            if len(candidates) != 1 or len(ep_owners) != 1:
                raise ValueError("ambiguous backward EP flow endpoint")
            begin = [row for row in pair if row["ph"] == "s"]
            if len(pair) != 2 or len(begin) != 1 or len(endings) != 1 or begin[0]["ts"] >= endpoint["ts"]:
                raise ValueError("backward EP flow is not one ordered start/end pair")
            sources = starts[(begin[0].get("pid"), begin[0].get("tid"), begin[0].get("ts"))]
            if len(sources) != 1:
                raise ValueError("ambiguous forward EP flow endpoint")
            source = sources[0]
            scopes = [scope for scope in payloads if contains(scope, source)]
            ancestors = [node for node in forward_nodes if contains(node, source)]
            if len(scopes) != 1 or len(ancestors) != 1 or not contains(scopes[0], ancestors[0]):
                raise ValueError("backward EP flow lacks one forward AllToAll payload origin")
            selected_step = ep_owners[0]["step"]
            if any(selected_step.get("pid") != row.get("pid")
                   or row["ts"] < selected_step["ts"]
                   or row["ts"] + row["dur"] > selected_step["ts"] + selected_step["dur"]
                   for row in (source, ancestors[0], scopes[0])):
                raise ValueError("forward/backward EP flow crosses the selected optimizer step")
            source_sequence = source.get("args", {}).get("Sequence number")
            target_sequence = candidates[0].get("args", {}).get("Sequence number")
            if source_sequence is not None and target_sequence is not None and source_sequence != target_sequence:
                raise ValueError("forward/backward EP flow sequence metadata disagrees")
            node_id = id(ep_owners[0]["node"])
            if node_id in matched:
                raise ValueError("backward EP node has multiple forward flow origins")
            matched[node_id] = scopes[0]
            origin_counts[scopes[0]["name"].rsplit("/", 1)[-1]] += 1
    if (set(matched) != set(targets) or len(matched) != len(payloads)
            or len({id(scope) for scope in matched.values()}) != len(payloads)):
        raise ValueError("forward/backward EP payload flow is not a complete bijection")
    return matched, dict(origin_counts)


def _device_scope_links(events: list[dict], selected_steps: tuple[int, ...] = (5,),
                        *, driver_forward_recoveries: dict | None = None) -> tuple[dict[int, list[dict]], dict]:
    """Attribute activity through raw routes or validated driver-forward recovery."""
    driver_forward_recoveries = driver_forward_recoveries or {}
    runtime_owners = {event.get("args", {}).get("correlation"): event for event in events
                      if event.get("ph") == "X" and event.get("cat") == "cuda_runtime"
                      and event.get("args", {}).get("correlation") is not None}
    runtime_scopes = _runtime_scope_events(events)
    external_scopes, external_owners = _external_cpu_scope_events(events)
    step_scopes = [event for event in events if event.get("ph") == "X"
                   and event.get("cat") == "user_annotation"
                   and event.get("name") in {f"AMPA/STEP/{number}" for number in selected_steps}]
    if len(step_scopes) != len(selected_steps) or len(set(selected_steps)) != len(selected_steps):
        raise ValueError(f"expected one host scope for each selected step {selected_steps}")
    backward_contexts, backward_nodes = _backward_operation_contexts(events, step_scopes)
    backward_origins, backward_origin_counts = _ep_forward_backward_flow_links(events, backward_contexts, backward_nodes)
    driver_owners = {}
    for event in events:
        if (event.get("ph") == "X" and event.get("cat") == "cuda_driver"
                and "LaunchKernel" in event.get("name", "")):
            correlation = event.get("args", {}).get("correlation")
            if correlation is not None:
                if correlation in driver_owners:
                    raise ValueError("ambiguous duplicate CUDA driver correlation")
                driver_owners[correlation] = event

    def in_step(owner: dict) -> bool:
        return any(step["ts"] <= owner["ts"] and owner["ts"] + owner["dur"] <= step["ts"] + step["dur"]
                   for step in step_scopes)

    links: dict[int, list[dict]] = {}
    stats = Counter()
    selected_intervals = []
    unassigned_intervals = []
    unlinked_intervals = []
    selected_nccl_intervals = []
    selected_nccl_by_group_us = defaultdict(float)
    unassigned_nccl_intervals = []
    unassigned_nccl_by_group_us = defaultdict(float)
    unlinked_nccl_by_group_us = defaultdict(float)
    untagged_cp_peer_by_owner = Counter()
    untagged_cp_peer_activity_us = Counter()
    backward_ep_nodes_with_kernel = set()
    recovery_provenance = []
    stats["backward_ep_alltoall_cpu_nodes"] = sum(
        context["node"]["name"] == "autograd::engine::evaluate_function: _AllToAllBackward"
        for context in backward_nodes.values())
    for event in events:
        if event.get("ph") != "X" or event.get("cat") not in {"kernel", "gpu_memcpy", "gpu_memset"}:
            continue
        if event.get("dur", 0) <= 0:
            continue
        stats["positive_device_events"] += 1
        stats["positive_device_activity_sum_us"] += event["dur"]
        interval = (event["ts"], event["ts"] + event["dur"])
        nccl = _collective_kernel(event)
        group = str(event.get("args", {}).get("Process Group Description") or "unidentified_group")
        correlation = event.get("args", {}).get("correlation")
        external = event.get("args", {}).get("External id")
        has_runtime = correlation in runtime_owners
        has_external = external in external_scopes
        recovery = driver_forward_recoveries.get(id(event))
        if has_runtime:
            stats["runtime_linked"] += 1
        if has_external:
            stats["external_cpu_op_linked"] += 1
        if has_runtime or has_external:
            stats["linked_by_either_route"] += 1
        if has_runtime or has_external or recovery is not None:
            stats["linked_by_any_route"] += 1
        direct = runtime_scopes.get(correlation, []) if has_runtime else []
        fallback = external_scopes.get(external, []) if has_external else []
        contextual = []
        for host in (runtime_owners.get(correlation), external_owners.get(external), driver_owners.get(correlation)):
            context = backward_contexts.get(id(host)) if host is not None else None
            if context is not None:
                contextual.append(context)
        if contextual and (len({id(item["scope"]) for item in contextual}) != 1
                           or len({id(item["node"]) for item in contextual}) != 1):
            raise ValueError("CUDA and External-id routes disagree on autograd engine ownership")
        runtime_backward = backward_contexts.get(id(runtime_owners.get(correlation))) if has_runtime else None
        external_backward = backward_contexts.get(id(external_owners.get(external))) if has_external else None
        if not direct and runtime_backward is not None:
            direct = [runtime_backward["scope"]]
        if not fallback and external_backward is not None:
            fallback = [external_backward["scope"]]
        if has_runtime and has_external:
            stats["dual_route"] += 1
            if len(direct) != len(fallback) or any(
                left is not right for left, right in zip(direct, fallback)
            ):
                raise ValueError(
                    f"CUDA runtime and exact External-id host op disagree on scope instance for correlation "
                    f"{correlation}, External id {external}"
                )
            if in_step(runtime_owners[correlation]) != in_step(external_owners[external]):
                raise ValueError(
                    f"CUDA runtime and exact External-id CPU op disagree on selected-step membership "
                    f"for correlation {correlation}, External id {external}"
                )
        scopes = direct if has_runtime else fallback
        owner = runtime_owners[correlation] if has_runtime else external_owners.get(external)
        if recovery is not None:
            recovered_owner = recovery["owner"]
            recovered_external = recovered_owner["args"]["External id"]
            scopes = external_scopes.get(recovered_external, [])
            if (has_runtime or has_external or contextual
                    or external_owners.get(recovered_external) is not recovered_owner
                    or driver_owners.get(correlation) is not recovery["driver"]
                    or not any(scope is recovery["payload"] for scope in scopes)
                    or not any(step is recovery["step"] for step in step_scopes)):
                raise ValueError("driver-forward recovery disagrees with exact CPU scope ownership")
            owner = recovered_owner
            stats["driver_forward_recovered_device_events"] += 1
            stats["driver_forward_recovered_activity_sum_us"] += event["dur"]
            recovery_provenance.append(recovery["provenance"])
        if (event.get("cat") == "kernel" and "nccl" in event.get("name", "").lower()
                and not event.get("args", {}).get("Collective name")):
            cpu_owner = external_owners.get(external)
            expected_cp_owners = {"AttnFuncWithCPAndKVP2P", "AttnFuncWithCPAndKVP2PBackward"}
            if (cpu_owner is None or cpu_owner.get("name") not in expected_cp_owners
                    or not in_step(cpu_owner)
                    or any(scope["name"].endswith("_a2a") for scope in scopes)):
                raise ValueError("untagged NCCL kernel lacks selected-step CP KV peer ownership or overlaps EP A2A")
            untagged_cp_peer_by_owner[cpu_owner["name"]] += 1
            untagged_cp_peer_activity_us[cpu_owner["name"]] += event["dur"]
        if owner is None:
            stats["unlinked_device_events"] += 1
            stats["unlinked_device_activity_sum_us"] += event["dur"]
            unlinked_intervals.append(interval)
            if nccl:
                unlinked_nccl_by_group_us[group] += event["dur"]
        elif in_step(owner):
            stats["device_events_launched_inside_selected_step"] += 1
            stats["selected_step_device_activity_sum_us"] += event["dur"]
            selected_intervals.append(interval)
            if nccl:
                stats["selected_step_nccl_activity_sum_us"] += event["dur"]
                selected_nccl_intervals.append(interval)
                selected_nccl_by_group_us[group] += event["dur"]
                if (contextual and group == "EXPERT_MODEL_PARALLEL_GROUP"
                        and contextual[0]["node"]["name"] == "autograd::engine::evaluate_function: _AllToAllBackward"):
                    collective = re.sub(r"[^a-z]", "", str(event.get("args", {}).get("Collective name", "")).lower())
                    if collective not in {"alltoall", "alltoallv"}:
                        raise ValueError("backward EP AllToAll node owns a different collective")
                    backward_ep_nodes_with_kernel.add(id(contextual[0]["node"]))
                    stats["backward_ep_alltoall_kernel_count"] += 1
            if scopes:
                stats["selected_step_events_inside_labeled_stage"] += 1
            else:
                stats["selected_step_events_unassigned_to_stage"] += 1
                stats["selected_step_unassigned_activity_sum_us"] += event["dur"]
                unassigned_intervals.append(interval)
                if nccl:
                    stats["selected_step_unassigned_nccl_sum_us"] += event["dur"]
                    unassigned_nccl_intervals.append(interval)
                    unassigned_nccl_by_group_us[group] += event["dur"]
        else:
            stats["device_events_launched_outside_selected_step"] += 1
            stats["outside_selected_step_activity_sum_us"] += event["dur"]
        if scopes:
            links[id(event)] = scopes
            stats["inside_labeled_scopes"] += 1
    stats["selected_step_device_activity_union_us"] = _union_us(selected_intervals)
    stats["selected_step_nccl_activity_union_us"] = _union_us(selected_nccl_intervals)
    stats["selected_step_unassigned_activity_union_us"] = _union_us(unassigned_intervals)
    stats["selected_step_unassigned_nccl_union_us"] = _union_us(unassigned_nccl_intervals)
    stats["unlinked_device_activity_union_us"] = _union_us(unlinked_intervals)
    result = dict(stats)
    result["driver_forward_recovered_device_events"] = stats["driver_forward_recovered_device_events"]
    result["driver_forward_recovered_activity_sum_us"] = stats["driver_forward_recovered_activity_sum_us"]
    result["driver_forward_recovery_provenance"] = recovery_provenance
    result["backward_ep_alltoall_nodes_with_exact_forward_flow"] = len(backward_origins)
    result["backward_ep_forward_payload_origin_counts"] = backward_origin_counts
    result["backward_ep_alltoall_nodes_with_correlated_nccl"] = len(backward_ep_nodes_with_kernel)
    result["selected_step_unassigned_nccl_by_group_us"] = dict(sorted(unassigned_nccl_by_group_us.items()))
    result["selected_step_nccl_by_group_us"] = dict(sorted(selected_nccl_by_group_us.items()))
    result["unlinked_nccl_by_group_us"] = dict(sorted(unlinked_nccl_by_group_us.items()))
    result["untagged_cp_kv_p2p_by_owner"] = dict(sorted(untagged_cp_peer_by_owner.items()))
    result["untagged_cp_kv_p2p_activity_us"] = dict(sorted(untagged_cp_peer_activity_us.items()))
    return links, result


def aggregate_rank(run_dir: Path, rank: int, selected_steps: tuple[int, ...] = (5,),
                   *, trace_path: Path | None = None) -> dict:
    trace_path = trace_path or run_dir / "traces" / "torch_profile" / f"rank-{rank}.json.gz"
    with gzip.open(trace_path, "rt", encoding="utf-8") as stream:
        trace = json.load(stream)
    events, driver_forward_recoveries = _resolve_forward_driver_metadata(
        trace["traceEvents"], trace.get("distributedInfo", {}), selected_steps)
    scopes = [event for event in events if event.get("ph") == "X"
              and event.get("cat") == "user_annotation"
              and _is_labeled_scope(event.get("name", ""))]
    if not any(LABEL.fullmatch(event["name"]) for event in scopes):
        raise ValueError(f"rank {rank}: no labeled MoE forward scopes")
    stages: dict[str, dict] = defaultdict(lambda: {
        "cpu_scope_us": [], "gpu_kernel_us": 0.0, "gpu_kernel_count": 0,
        "gpu_nccl_us": 0.0, "gpu_nccl_count": 0, "gpu_intervals": [], "gpu_nccl_intervals": [],
        "gpu_exclusive_us": 0.0, "gpu_exclusive_count": 0,
        "gpu_exclusive_nccl_us": 0.0, "gpu_exclusive_intervals": [],
        "gpu_launched_nccl_by_group_us": defaultdict(float),
        "gpu_exclusive_nccl_by_group_us": defaultdict(float),
        "gpu_exclusive_nccl_by_group_collective_us": defaultdict(float),
    })
    layers: dict[str, dict] = defaultdict(lambda: {
        "cpu_scope_us": [], "cpu_exclusive_us": [],
        "gpu_kernel_us": 0.0, "gpu_nccl_us": 0.0,
        "gpu_exclusive_us": 0.0, "gpu_exclusive_nccl_us": 0.0,
    })
    direct_children: dict[int, list[tuple[float, float]]] = defaultdict(list)
    by_thread: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for event in scopes:
        by_thread[(event.get("pid"), event.get("tid"))].append(event)
    for thread_scopes in by_thread.values():
        active: list[dict] = []
        for scope in sorted(thread_scopes, key=lambda item: (item["ts"], -item["dur"])):
            while active and scope["ts"] >= active[-1]["ts"] + active[-1]["dur"]:
                active.pop()
            if active:
                if scope["ts"] + scope["dur"] > active[-1]["ts"] + active[-1]["dur"]:
                    raise ValueError(f"rank {rank}: labeled CPU scopes overlap without nesting")
                direct_children[id(active[-1])].append((scope["ts"], scope["ts"] + scope["dur"]))
            active.append(scope)
    cpu_exclusive: dict[str, list[float]] = defaultdict(list)
    for event in scopes:
        phase = _phase(event["name"])
        stages[phase]["cpu_scope_us"].append(event["dur"])
        if LABEL.fullmatch(event["name"]):
            layers[event["name"]]["cpu_scope_us"].append(event["dur"])
        exclusive = event["dur"] - _union_us(direct_children[id(event)])
        if exclusive < -1e-3:
            raise ValueError(f"rank {rank}: negative exclusive CPU scope for {event['name']}")
        cpu_exclusive[phase].append(max(0.0, exclusive))
        if LABEL.fullmatch(event["name"]):
            layers[event["name"]]["cpu_exclusive_us"].append(max(0.0, exclusive))
    device_links, link_stats = _device_scope_links(
        events, selected_steps, driver_forward_recoveries=driver_forward_recoveries)
    per_call_gpu: dict[int, float] = defaultdict(float)
    per_call_exclusive_gpu: dict[int, float] = defaultdict(float)
    per_call_exclusive_nccl: dict[int, float] = defaultdict(float)
    attributed_device_events = 0
    device_events = 0
    for event in events:
        if event.get("ph") != "X" or event.get("cat") not in {"kernel", "gpu_memcpy", "gpu_memset"}:
            continue
        if event.get("dur", 0) <= 0:
            continue
        device_events += 1
        scope_events = device_links.get(id(event), [])
        if not scope_events:
            continue
        interval = (event["ts"], event["ts"] + event["dur"])
        nccl = _collective_kernel(event)
        group = str(event.get("args", {}).get("Process Group Description") or "unidentified_group")
        collective = str(event.get("args", {}).get("Collective name") or "unidentified_collective")
        for scope in scope_events:
            name = scope["name"]
            phase = _phase(name)
            entry = stages[phase]
            entry["gpu_kernel_us"] += event["dur"]
            entry["gpu_kernel_count"] += 1
            entry["gpu_intervals"].append(interval)
            if LABEL.fullmatch(name):
                layers[name]["gpu_kernel_us"] += event["dur"]
            per_call_gpu[id(scope)] += event["dur"]
            if nccl:
                entry["gpu_nccl_us"] += event["dur"]
                entry["gpu_nccl_count"] += 1
                entry["gpu_nccl_intervals"].append(interval)
                entry["gpu_launched_nccl_by_group_us"][group] += event["dur"]
                if LABEL.fullmatch(name):
                    layers[name]["gpu_nccl_us"] += event["dur"]
        deepest_scope = scope_events[-1]
        deepest = _phase(deepest_scope["name"])
        exclusive_entry = stages[deepest]
        exclusive_entry["gpu_exclusive_us"] += event["dur"]
        exclusive_entry["gpu_exclusive_count"] += 1
        exclusive_entry["gpu_exclusive_intervals"].append(interval)
        if LABEL.fullmatch(deepest_scope["name"]):
            layers[deepest_scope["name"]]["gpu_exclusive_us"] += event["dur"]
        per_call_exclusive_gpu[id(deepest_scope)] += event["dur"]
        if nccl:
            exclusive_entry["gpu_exclusive_nccl_us"] += event["dur"]
            per_call_exclusive_nccl[id(deepest_scope)] += event["dur"]
            exclusive_entry["gpu_exclusive_nccl_by_group_us"][group] += event["dur"]
            exclusive_entry["gpu_exclusive_nccl_by_group_collective_us"][f"{group}/{collective}"] += event["dur"]
            if LABEL.fullmatch(deepest_scope["name"]):
                layers[deepest_scope["name"]]["gpu_exclusive_nccl_us"] += event["dur"]
        attributed_device_events += 1
    phase_rows = []
    calls_by_phase: dict[str, list[dict]] = defaultdict(list)
    calls_by_layer: dict[str, list[dict]] = defaultdict(list)
    for event in scopes:
        calls_by_phase[_phase(event["name"])].append(event)
        if LABEL.fullmatch(event["name"]):
            calls_by_layer[event["name"]].append(event)
    for phase, data in sorted(stages.items()):
        calls = data["cpu_scope_us"]
        phase_events = calls_by_phase[phase]
        gpu_calls = [per_call_gpu[id(event)] for event in phase_events]
        exclusive_gpu_calls = [per_call_exclusive_gpu[id(event)] for event in phase_events]
        exclusive_nccl_calls = [per_call_exclusive_nccl[id(event)] for event in phase_events]
        phase_rows.append({
            "phase": phase, "calls": len(calls),
            "cpu_scope_sum_ms": sum(calls) / 1000,
            "cpu_exclusive_sum_ms": sum(cpu_exclusive[phase]) / 1000,
            "cpu_mean_call_ms": statistics.fmean(calls) / 1000,
            "cpu_median_call_ms": statistics.median(calls) / 1000,
            "cpu_p95_call_ms": _percentile(calls, 0.95) / 1000,
            "cpu_max_call_ms": max(calls) / 1000,
            "gpu_launched_activity_median_call_ms": statistics.median(gpu_calls) / 1000,
            "gpu_launched_activity_p95_call_ms": _percentile(gpu_calls, 0.95) / 1000,
            "gpu_launched_activity_sum_ms": data["gpu_kernel_us"] / 1000,
            "gpu_launched_activity_union_ms": _union_us(data["gpu_intervals"]) / 1000,
            "gpu_launched_activity_count": data["gpu_kernel_count"],
            "gpu_exclusive_activity_sum_ms": data["gpu_exclusive_us"] / 1000,
            "gpu_exclusive_activity_union_ms": _union_us(data["gpu_exclusive_intervals"]) / 1000,
            "gpu_exclusive_activity_count": data["gpu_exclusive_count"],
            "gpu_exclusive_activity_median_call_ms": statistics.median(exclusive_gpu_calls) / 1000,
            "gpu_exclusive_activity_p95_call_ms": _percentile(exclusive_gpu_calls, 0.95) / 1000,
            "gpu_exclusive_nccl_sum_ms": data["gpu_exclusive_nccl_us"] / 1000,
            "gpu_exclusive_nccl_median_call_ms": statistics.median(exclusive_nccl_calls) / 1000,
            "gpu_exclusive_nccl_p95_call_ms": _percentile(exclusive_nccl_calls, 0.95) / 1000,
            "gpu_launched_nccl_sum_ms": data["gpu_nccl_us"] / 1000,
            "gpu_launched_nccl_union_ms": _union_us(data["gpu_nccl_intervals"]) / 1000,
            "gpu_launched_nccl_count": data["gpu_nccl_count"],
            "gpu_launched_nccl_by_group_ms": {
                group: duration / 1000 for group, duration in sorted(data["gpu_launched_nccl_by_group_us"].items())
            },
            "gpu_exclusive_nccl_by_group_ms": {
                group: duration / 1000 for group, duration in sorted(data["gpu_exclusive_nccl_by_group_us"].items())
            },
            "gpu_exclusive_nccl_by_group_collective_ms": {
                name: duration / 1000
                for name, duration in sorted(data["gpu_exclusive_nccl_by_group_collective_us"].items())
            },
        })
    layer_rows = [{
        "name": name, "layer": int(LABEL.fullmatch(name).group(1)), "phase": _phase(name),
        "calls": len(data["cpu_scope_us"]),
        "cpu_scope_sum_ms": sum(data["cpu_scope_us"]) / 1000,
        "cpu_exclusive_sum_ms": sum(data["cpu_exclusive_us"]) / 1000,
        "cpu_median_call_ms": statistics.median(data["cpu_scope_us"]) / 1000,
        "cpu_p95_call_ms": _percentile(data["cpu_scope_us"], 0.95) / 1000,
        "gpu_launched_activity_sum_ms": data["gpu_kernel_us"] / 1000,
        "gpu_launched_nccl_sum_ms": data["gpu_nccl_us"] / 1000,
        "gpu_exclusive_activity_sum_ms": data["gpu_exclusive_us"] / 1000,
        "gpu_exclusive_activity_median_call_ms": statistics.median(
            per_call_exclusive_gpu[id(event)] for event in calls_by_layer[name]
        ) / 1000,
        "gpu_exclusive_activity_p95_call_ms": _percentile(
            [per_call_exclusive_gpu[id(event)] for event in calls_by_layer[name]], 0.95
        ) / 1000,
        "gpu_exclusive_nccl_sum_ms": data["gpu_exclusive_nccl_us"] / 1000,
        "gpu_exclusive_nccl_median_call_ms": statistics.median(
            per_call_exclusive_nccl[id(event)] for event in calls_by_layer[name]
        ) / 1000,
    } for name, data in sorted(layers.items())]
    step_events = [event for event in events if event.get("name", "").startswith("AMPA/STEP/")
                   and event.get("ph") == "X" and event.get("cat") == "user_annotation"]
    result = {
        "rank": rank,
        "step_scopes_ms": {event["name"]: event["dur"] / 1000 for event in step_events},
        "phase_rows": phase_rows,
        "layer_rows": layer_rows,
        "device_events": device_events,
        "device_events_launched_inside_labeled_scopes": attributed_device_events,
        "device_link_stats": link_stats,
        "claim_limit": "CPU scope durations include launch/wait overhead. GPU values are correlated CUDA activity and can overlap across streams. Exclusive launch buckets avoid nested phase duplication but their sums still do not measure exposed critical-path time.",
    }
    del events, trace
    gc.collect()
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--ranks", default="0,1,2,3,4,5,6,7")
    args = parser.parse_args()
    validation = json.loads((args.run_dir / "analysis" / "torch_trace_validation.json").read_text(encoding="utf-8"))
    ranks = [int(value) for value in args.ranks.split(",")]
    if validation.get("valid_capture") is not True or validation.get("requested_ranks") != ranks:
        raise SystemExit("Cannot aggregate a missing, failed or different-rank trace validation")
    source_hashes = validation.get("analysis_source_sha256", {})
    for name in ("validate_torch_trace.py", "aggregate_deep_trace.py"):
        path = Path(__file__).with_name(name)
        if source_hashes.get(name) != _sha256(path):
            raise SystemExit(f"Trace validator/aggregator source changed after validation: {name}")
    records = {record["rank"]: record for record in validation.get("ranks", [])}
    if set(records) != set(ranks):
        raise SystemExit("Trace validation lacks one or more requested ranks")
    output_dir = args.run_dir / "analysis"
    summaries = []
    for rank in ranks:
        record = records[rank]
        for path_key, digest_key in (("trace_path", "trace_sha256"), ("call_record_path", "call_record_sha256")):
            path = Path(record[path_key])
            if not path.is_file() or _sha256(path) != record[digest_key]:
                raise SystemExit(f"Rank {rank} {path_key} changed after trace validation")
        summary = aggregate_rank(args.run_dir, rank, tuple(range(validation["step"], validation["step"] + validation["capture_steps"])))
        summary_path = output_dir / f"moe_deep_timing_rank-{rank}.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )
        summaries.append({
            "rank": rank, "step_scopes_ms": summary["step_scopes_ms"],
            "device_events": summary["device_events"],
            "device_events_launched_inside_labeled_scopes": summary["device_events_launched_inside_labeled_scopes"],
            "summary_path": str(summary_path), "summary_sha256": _sha256(summary_path),
        })
    (output_dir / "moe_deep_timing_index.json").write_text(json.dumps({
        "schema_version": 1, "ranks": summaries,
        "claim_scope": validation["claim_scope"],
        "analysis_source_sha256": source_hashes,
        "claim_limit": "Per-rank phase tables are launch-attributed diagnostics; inspect profiling overhead before treating them as representative unprofiled timings.",
    }, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"ranks": len(summaries), "outputs": str(output_dir)}))


if __name__ == "__main__":
    main()
