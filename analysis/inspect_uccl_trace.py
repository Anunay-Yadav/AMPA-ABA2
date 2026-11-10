"""Validate UCCL call pairs and exact host-to-device trace correlations.

Run on allocated compute. Inclusive phase activities overlap and are not a
partition of optimizer-step latency. CPU proxy work and wire bytes remain
unavailable. A visible payload kernel is required for each recorded adjoint.
"""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import re

from analysis.aggregate_deep_trace import _backward_operation_contexts


PAIR = re.compile(r"^AMPA/UCCL_CALL/(\d+)/(FWD|BWD)$")
PHASE = re.compile(r"^AMPA/(?:FWD|BWD)/layer_\d{2}/.+$|^AMPA/TRAIN/.+$")
DEVICE = {"kernel", "gpu_memcpy", "gpu_memset"}
PAYLOAD = re.compile(r"uccl::internode::(dispatch|combine)<")
CALL_PHASES = {("dispatch", "FWD"): "uccl_dispatch",
               ("dispatch", "BWD"): "uccl_combine_dinput_dprob",
               ("combine", "FWD"): "uccl_combine",
               ("combine", "BWD"): "uccl_dispatch_doutput"}
REQUIRED_MOE_PHASES = {"FWD": (
    "pre_mlp_norm", "moe_total", "router_total", "router_gate", "router_routing",
    "router_main_topk", "router_aux_scores", "router_aux_loss", "dispatch_preprocess",
    "dispatch_total", "uccl_layout", "uccl_dispatch", "local_expert_sort", "local_expert_sort_core",
    "expert_total", "expert_fc1", "expert_fc2", "inverse_local_sort", "inverse_local_sort_core",
    "combine_total", "uccl_combine", "final_unpermute", "residual_bda"),
    "BWD": ("uccl_combine_dinput_dprob", "uccl_dispatch_doutput")}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def union_us(intervals):
    total, right = 0.0, None
    for start, end in sorted(intervals):
        if right is None or start > right:
            total += end - start
        elif end > right:
            total += end - right
        right = end if right is None else max(right, end)
    return total


def contains(parent, child):
    return (parent.get("pid") == child.get("pid") and parent.get("tid") == child.get("tid")
            and parent["ts"] <= child["ts"]
            and child["ts"] + child["dur"] <= parent["ts"] + parent["dur"])


def scope_routes(events, annotations):
    """Use same-thread full containment of the exact correlated host event."""
    scopes, hosts = defaultdict(list), defaultdict(list)
    for event in annotations:
        scopes[(event.get("pid"), event.get("tid"))].append(event)
    for event in events:
        if event.get("cat") in {"cpu_op", "user_annotation", "cuda_runtime", "cuda_driver"}:
            hosts[(event.get("pid"), event.get("tid"))].append(event)
    routes = {}
    for thread, operations in hosts.items():
        ordered = sorted(scopes[thread], key=lambda row: (row["ts"], -row["dur"]))
        active, index = [], 0
        for event in sorted(operations, key=lambda row: row["ts"]):
            while index < len(ordered) and ordered[index]["ts"] <= event["ts"]:
                active.append(ordered[index])
                index += 1
            active = [scope for scope in active if scope["ts"] + scope["dur"] >= event["ts"]]
            routes[id(event)] = [scope for scope in active if contains(scope, event)]
    return routes


def _unique(index, key):
    rows = index.get(key, [])
    if len(rows) > 1:
        raise ValueError(f"ambiguous exact trace identity: {key}")
    return rows[0] if rows else None


def _pair_tokens(scopes):
    return {(int(match[1]), match[2]) for scope in scopes
            if (match := PAIR.fullmatch(scope.get("name", "")))}


def inspect_events(trace, record, *, synthetic=False):
    events = [event for event in trace["traceEvents"] if event.get("ph") == "X"]
    for event in events:
        if (type(event.get("ts")) not in (int, float) or not math.isfinite(event["ts"])
                or type(event.get("dur")) not in (int, float) or not math.isfinite(event["dur"])
                or event["dur"] < 0):
            raise ValueError("invalid complete-event timestamp/duration")
    step_scopes = []
    if synthetic:
        if record.get("enabled") is not True or record.get("synthetic_computation_completed") is not True:
            raise ValueError("synthetic observation did not finish")
        pairs, counts = record["call_pairs"], record["phase_counts"]
        layers = list(range(record["additional_public_layout_checks"]))
        if type(record["additional_public_layout_checks"]) is not int or len(layers) not in range(2, 5):
            raise ValueError("unreviewed synthetic geometry")
        accumulation = 1
    else:
        if (record.get("schema_version") != 2 or record.get("kind") != "ampa_pytorch_uccl_deep_trace_calls"
                or len(record.get("steps", [])) != 1):
            raise ValueError("not one UCCL optimizer-step record")
        distributed = trace.get("distributedInfo", {})
        if (type(record.get("rank")) is not int or record["rank"] not in range(16)
                or distributed.get("rank") != record["rank"] or distributed.get("world_size") != 16):
            raise ValueError("trace distributed identity differs from its call record")
        step = record["steps"][0]
        if step.get("skipped_iter") is not False or step.get("layers") != list(range(48)):
            raise ValueError("incomplete or skipped optimizer step")
        pairs, counts = step["uccl_autograd_pairs"], step["counts"]
        layers, accumulation = list(range(48)), 4
        for direction, phases in REQUIRED_MOE_PHASES.items():
            if any(counts.get(f"AMPA/{direction}/layer_{layer:02d}/{phase}") != accumulation
                   for layer in layers for phase in phases):
                raise ValueError("incomplete 48-layer/GA4 MoE phase grid")
        step_scopes = [event for event in events if event.get("cat") == "user_annotation"
                       and event.get("name") == f"AMPA/STEP/{step['step']}"]
        if len(step_scopes) != 1:
            raise ValueError("expected one complete optimizer-step annotation")
        for phase, expected in (("microbatch_forward", 4), ("microbatch_backward", 4),
                                ("gradient_finalize", 1), ("optimizer_step", 1)):
            if counts.get(f"AMPA/TRAIN/{phase}") != expected:
                raise ValueError("incomplete complete-step lifecycle counts")
    if any(type(pair.get(key)) is not int for pair in pairs for key in ("call_id", "layer", "epoch", "context_id")):
        raise ValueError("malformed integer UCCL call identity")
    if any(pair["epoch"] <= 0 or pair["context_id"] <= 0 for pair in pairs):
        raise ValueError("nonpositive UCCL context or epoch identity")
    expected_grid = Counter({(layer, family): accumulation for layer in layers for family in ("dispatch", "combine")})
    if (Counter((pair.get("layer"), pair.get("family")) for pair in pairs) != expected_grid
            or [pair.get("call_id") for pair in pairs] != list(range(len(pairs)))
            or not all(pair.get("forward_returned") is True and pair.get("backward_returned") is True for pair in pairs)
            or len({pair.get("epoch") for pair in pairs}) != 1):
        raise ValueError("incomplete UCCL forward/backward call grid")
    annotations = [event for event in events if event.get("cat") == "user_annotation"
                   and (PAIR.fullmatch(event.get("name", "")) or PHASE.fullmatch(event.get("name", "")))]
    annotations_by_name = defaultdict(list)
    for event in annotations:
        annotations_by_name[event["name"]].append(event)
    phase_counts = Counter(event["name"] for event in annotations if PHASE.fullmatch(event["name"])
                           and event["name"] != "AMPA/TRAIN/routing_audit")
    # Routing-audit scopes come from the separate observer, not this journal.
    if (not counts or any(type(count) is not int or count <= 0 for count in counts.values())
            or phase_counts != Counter(counts)):
        raise ValueError("trace annotation counts disagree with observer")
    pair_scopes = defaultdict(list)
    for event in annotations:
        if match := PAIR.fullmatch(event["name"]):
            pair_scopes[(int(match[1]), match[2])].append(event)
    expected_tokens = {(pair["call_id"], direction) for pair in pairs for direction in ("FWD", "BWD")}
    if set(pair_scopes) != expected_tokens or any(len(rows) != 1 for rows in pair_scopes.values()):
        raise ValueError("pair journal and trace scopes are not bijective")
    if step_scopes:
        step_scope = step_scopes[0]
        for rows in pair_scopes.values():
            scope = rows[0]
            if (scope.get("pid") != step_scope.get("pid") or scope["ts"] < step_scope["ts"]
                    or scope["ts"] + scope["dur"] > step_scope["ts"] + step_scope["dur"]):
                raise ValueError("UCCL call lies outside its declared optimizer step")
    kernel_correlations = {event.get("args", {}).get("correlation") for event in events
                           if event.get("cat") == "kernel" and event["dur"] > 0}
    for event in events:
        if event.get("cat") in {"cuda_runtime", "cuda_driver"} and "LaunchKernel" in event.get("name", ""):
            correlation = event.get("args", {}).get("correlation")
            if correlation is None or correlation not in kernel_correlations:
                raise ValueError("eligible kernel launch lacks positive recorded device activity")
    routes = scope_routes(events, annotations)
    backward_contexts, _ = _backward_operation_contexts(events, step_scopes) if step_scopes else ({}, {})
    # Reuse the native analyzer's explicit autograd-engine ownership proof.
    # Same-process time containment alone cannot assign another CPU thread's work.
    for owner_id, context in backward_contexts.items():
        owner_scopes = routes.setdefault(owner_id, [])
        if not any(scope is context["scope"] for scope in owner_scopes):
            owner_scopes.append(context["scope"])
    for pair in pairs:
        for direction in ("FWD", "BWD"):
            scope = pair_scopes[(pair["call_id"], direction)][0]
            phase = f"AMPA/{direction}/layer_{pair['layer']:02d}/{CALL_PHASES[(pair['family'], direction)]}"
            if phase not in {row["name"] for row in routes[id(scope)]}:
                raise ValueError("UCCL call identity is under the wrong phase/layer")
    runtime, driver, external = defaultdict(list), defaultdict(list), defaultdict(list)
    for event in events:
        args = event.get("args", {})
        if event.get("cat") == "cuda_runtime" and "correlation" in args:
            runtime[args["correlation"]].append(event)
        if event.get("cat") == "cuda_driver" and "LaunchKernel" in event.get("name", "") and "correlation" in args:
            driver[args["correlation"]].append(event)
        if event.get("cat") in {"cpu_op", "user_annotation"} and "External id" in args:
            external[args["External id"]].append(event)
    phase_activity, phase_intervals = Counter(), defaultdict(list)
    pair_kernels, pair_payloads = Counter(), Counter()
    stats, unlinked, kernel_names = Counter(), [], Counter()
    for event in events:
        if event.get("cat") not in DEVICE or event["dur"] <= 0:
            continue
        stats["positive_device_events"] += 1
        args = event.get("args", {})
        candidates = []
        for name, index, key in (("runtime", runtime, args.get("correlation")),
                                 ("driver", driver, args.get("correlation")),
                                 ("external", external, args.get("External id"))):
            owner = _unique(index, key) if key is not None else None
            if owner is not None:
                stats[f"{name}_linked"] += 1
                candidates.append((owner, routes.get(id(owner), [])))
        if not candidates:
            unlinked.append({"name": event.get("name"), "category": event.get("cat"), "args": args})
            continue
        # Broad External-id owners may contain a more precise runtime launch.
        # Conflicting concrete call identities are never resolved by proximity.
        tokens = [_pair_tokens(scopes) for _, scopes in candidates]
        concrete = [value for value in tokens if value]
        if any(len(value) != 1 for value in concrete) or (concrete and any(value != concrete[0] for value in concrete)):
            raise ValueError("exact correlation routes disagree on UCCL call identity")
        scope_sets = [{id(scope) for scope in scopes} for _, scopes in candidates]
        if any(not (left <= right or right <= left) for left in scope_sets for right in scope_sets):
            raise ValueError("exact correlation routes disagree on semantic scope ancestry")
        concrete_candidates = [item for item in candidates if _pair_tokens(item[1])]
        owner, scopes = max(concrete_candidates or candidates, key=lambda item: len(item[1]))
        if not scopes:
            stats["linked_device_events_without_phase_scope"] += 1
        interval = (event["ts"], event["ts"] + event["dur"])
        for phase in {row["name"] for row in scopes if PHASE.fullmatch(row["name"])}:
            phase_activity[phase] += event["dur"]
            phase_intervals[phase].append(interval)
        if event.get("cat") == "kernel":
            kernel_names[event.get("name", "")] += 1
            if PAYLOAD.search(event.get("name", "")) and len(_pair_tokens(scopes)) != 1:
                raise ValueError("native UCCL payload kernel lacks one exact recorded call owner")
            for token in _pair_tokens(scopes):
                pair_kernels[token] += 1
                family = pairs[token[0]]["family"]
                expected = "dispatch" if (family == "dispatch") == (token[1] == "FWD") else "combine"
                if match := PAYLOAD.search(event.get("name", "")):
                    if match[1] != expected:
                        raise ValueError("native payload operation differs from the recorded forward/adjoint")
                    pair_payloads[token] += 1
    if unlinked:
        raise ValueError(f"{len(unlinked)} positive device events lack an exact host route; examples={unlinked[:3]}")
    if any(pair_payloads[token] < 1 for token in expected_tokens):
        raise ValueError("one or more UCCL forward/adjoint calls lack a correlated native payload kernel")
    phases = []
    for name, scopes in sorted(annotations_by_name.items()):
        if not PHASE.fullmatch(name):
            continue
        phases.append({"phase": name, "calls": len(scopes), "cpu_scope_sum_ms": sum(row["dur"] for row in scopes) / 1000,
                       "gpu_inclusive_activity_sum_ms": phase_activity[name] / 1000,
                       "gpu_inclusive_activity_union_ms": union_us(phase_intervals[name]) / 1000})
    return {"valid_uccl_trace_capture": True, "synthetic": synthetic, "pair_count": len(pairs),
            "paired_direction_count": len(expected_tokens), "device_link_stats": dict(stats),
            "phases": phases, "kernel_name_counts": dict(kernel_names),
            "physical_bytes": None, "cpu_proxy_time": None, "exposed_communication_latency": None,
            "claim_limits": ["Inclusive nested scopes and concurrent GPU activity are not additive step-time partitions.",
                             "Backward comparisons must use common inclusive ancestors or complete disjoint descendant rollups.",
                             "Exact linkage validates recorded activity; unrecorded persistent/proxy activity is not measured."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()
    inputs = {str(path.resolve()): sha256(path) for path in (args.trace, args.record)}
    opener = gzip.open if args.trace.suffix == ".gz" else open
    with opener(args.trace, "rt") as handle:
        trace = json.load(handle)
    result = inspect_events(trace, json.loads(args.record.read_text()), synthetic=args.synthetic)
    if inputs != {str(path.resolve()): sha256(path) for path in (args.trace, args.record)}:
        raise ValueError("trace inputs changed during inspection")
    result.update(inputs=inputs, analysis_source_sha256=sha256(__file__))
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(json.dumps({"valid_uccl_trace_capture": True, "pair_count": result["pair_count"]}), flush=True)


if __name__ == "__main__":
    main()
