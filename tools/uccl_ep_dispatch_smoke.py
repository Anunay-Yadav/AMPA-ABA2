#!/usr/bin/env python3
"""Exercise native Flex/UCCL forward and backward on allocated compute.

This is a dispatcher correctness smoke, not a replacement dispatcher, trainer,
throughput benchmark, or physical-traffic measurement.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import datetime
import gc
import gzip
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import socket
import shutil
import subprocess
import sys
import traceback

from tools.uccl_ep_source_identity import verify_uccl_source


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pci_bdf(value):
    domain, bus, device = value.strip().lower().split(":")
    return f"{int(domain, 16):04x}:{int(bus, 16):02x}:{device}"


def device_topology(ep, local_rank, gpu_uuid):
    """Scope the GPU query by selected UUID; retain real CXI PCI/NUMA facts."""
    result = subprocess.run(["nvidia-smi", "--id", gpu_uuid,
        "--query-gpu=index,pci.bus_id", "--format=csv,noheader,nounits"],
        check=True, text=True, capture_output=True, timeout=20)
    rows = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    require(len(rows) == 1, "selected UUID did not resolve to exactly one NVML GPU")
    nvml_index, bdf = rows[0].split(",")
    physical_rank = int(ep.get_physical_gpu_rank(local_rank))
    require(physical_rank == local_rank, "CUDA-visible order differs from UCCL physical PCI order")
    devices = {}
    for entry in sorted(Path("/sys/class/cxi").glob("cxi*")):
        device = (entry / "device").resolve()
        numa = device / "numa_node"
        cpus = device / "local_cpulist"
        devices[entry.name] = {"pci_device": str(device), "pci_bus_id": pci_bdf(device.name),
            "numa_node": int(numa.read_text().strip()) if numa.is_file() else None,
            "local_cpulist": cpus.read_text().strip() if cpus.is_file() else None}
    require(f"cxi{local_rank}" in devices, "expected CXI domain is absent; do not accept a fallback NIC")
    gpu_device = Path("/sys/bus/pci/devices") / pci_bdf(bdf)
    gpu_numa = gpu_device / "numa_node"
    return {"selected_uuid": gpu_uuid, "uuid_scoped_nvml_index": int(nvml_index),
        "selected_gpu_pci_bus_id": pci_bdf(bdf), "uccl_physical_gpu_rank": physical_rank,
        "gpu_numa_node": int(gpu_numa.read_text().strip()) if gpu_numa.is_file() else None,
        "cxi_devices": devices, "expected_cxi_domain": f"cxi{local_rank}",
        "actual_cxi_endpoint_requires_native_log_check": True}


def write_new(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def selected_routes(rank, tokens, cycle, nodes, placement, local_experts):
    """Choose eight experts on four ranks/two actual nodes, with duplicates."""
    ranks_by_host = {host: sorted(row["rank"] for row in placement if row["hostname"] == host) for host in nodes}
    own_node = nodes.index(placement[rank]["hostname"])
    routes = []
    for token in range(tokens):
        # The second case deliberately leaves two nodes with zero receives in
        # EP16. EP8 still exercises uneven local expert counts and new handles.
        first = (token + own_node + cycle) % len(nodes) if cycle % 2 == 0 else 0
        destinations = (first, (first + 1) % len(nodes))
        row = []
        for ordinal, node in enumerate(destinations):
            peers = ranks_by_host[nodes[node]]
            for offset in (0, 1):
                peer = peers[(token + cycle + 2 * ordinal + offset) % 4]
                owned = local_experts[peer]
                start = (token + rank + cycle) % len(owned)
                row.extend((owned[start], owned[(start + 1) % len(owned)]))
        require(len(row) == len(set(row)) == 8, "fixture expert IDs must be distinct")
        routes.append(row)
    return routes


def verify_equal(torch, actual, expected, label):
    require(actual is not None, f"{label}: missing gradient or output")
    actual = actual.detach().cpu()
    expected = expected.to(dtype=actual.dtype, device="cpu")
    require(actual.shape == expected.shape, f"{label}: shape differs")
    require(bool(torch.isfinite(actual).all()), f"{label}: nonfinite result")
    delta = (actual.double() - expected.double()).abs()
    maximum = float(delta.max()) if delta.numel() else 0.0
    require(torch.equal(actual, expected), f"{label}: exact dyadic reference differs; max_abs_error={maximum}")
    return {"exact": True, "max_abs_error": maximum, "dtype": str(actual.dtype), "shape": list(actual.shape)}


@contextmanager
def trace_observation(torch, args, rank):
    """Own opt-in synthetic labels and trace export, including failed scopes."""
    info = {"enabled": args.trace_observer, "kind": "synthetic_dispatcher_only",
            "full_qwen_or_optimizer_step": False}
    if not args.trace_observer:
        yield None, info
        return
    from profiling.ampa_profile.uccl_trace import UcclMethodLabels
    from megatron.core.transformer.moe import token_dispatcher

    observer = UcclMethodLabels(torch, [])
    profiler = None
    annotation = None
    completed = False
    try:
        observer._attach_communication(token_dispatcher, None)
        profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA], record_shapes=False, with_stack=False)
        profiler.start()
        annotation = torch.profiler.record_function("AMPA/SYNTHETIC/uccl_dispatch_smoke")
        annotation.__enter__()
        yield observer, info
        completed = True
    finally:
        active_error = sys.exc_info()
        # Restore all global/instance attributes before any export operation can
        # fail. Cleanup must not replace the original body exception or enter
        # CUDA cleanup after a failed detach/annotation. No 48-layer API runs.
        cleanup_errors = []
        failure = active_error
        info.update(observer_detached=False, wrappers_restored=False, annotation_closed=annotation is None)
        try:
            observer.detach()
            info.update(observer_detached=True, wrappers_restored=True)
        except BaseException:
            error = sys.exc_info()
            cleanup_errors.append(("observer.detach", error))
            if failure[0] is None:
                failure = error
        if annotation is not None:
            try:
                annotation.__exit__(*failure)
                info["annotation_closed"] = True
            except BaseException:
                error = sys.exc_info()
                cleanup_errors.append(("annotation.__exit__", error))
                if failure[0] is None:
                    failure = error
        if failure[0] is not None:
            # A failed transport may leave kernels waiting for peers. Preserve
            # host-side evidence without another CUDA wait; outer failure
            # handling exits immediately and lets torchrun terminate peers.
            info.update(valid=False, trace_aborted=True,
                failure_cleanup_cuda_synchronize_or_export_attempted=False,
                synthetic_computation_completed=completed,
                error=f"{failure[0].__name__}: {failure[1]}")
            if cleanup_errors:
                info["cleanup_errors"] = [{"phase": phase, "error": f"{error[0].__name__}: {error[1]}"}
                                          for phase, error in cleanup_errors]
                info["cleanup_error"] = info["cleanup_errors"][0]["error"]
            try:
                info.update(phase_counts=dict(sorted(observer.counts.items())), call_pairs=observer.pair_records())
            except BaseException as error:
                info["journal_error"] = f"{type(error).__name__}: {error}"
            try:
                write_new(args.output_dir / f"rank-{rank}-trace-aborted.json", info)
            except BaseException:
                traceback.print_exc(file=sys.stderr)
            if active_error[0] is None:
                raise failure[1].with_traceback(failure[2])
        elif profiler is not None:
            torch.cuda.synchronize()
            profiler.stop()
            temporary = args.output_dir / f"rank-{rank}-trace.json"
            destination = args.output_dir / f"rank-{rank}-trace.json.gz"
            require(not temporary.exists() and not destination.exists(), "trace evidence already exists")
            profiler.export_chrome_trace(str(temporary))
            with temporary.open("rb") as source, destination.open("xb") as output:
                with gzip.GzipFile(filename="", mode="wb", fileobj=output, mtime=0) as compressed:
                    shutil.copyfileobj(source, compressed)
            temporary.unlink()
            info.update(trace_path=str(destination.resolve()), trace_sha256=sha256(destination),
                synthetic_computation_completed=completed,
                phase_counts=dict(sorted(observer.counts.items())), call_pairs=observer.pair_records(),
                additional_public_layout_checks=args.cycles,
                extra_layout_getter_is_outside_native_call_pairs=True,
                observer_sources={str(Path(inspect.getfile(owner)).resolve()): sha256(inspect.getfile(owner))
                                  for owner in (UcclMethodLabels, UcclMethodLabels.__bases__[0])})
            write_new(args.output_dir / f"rank-{rank}-trace-observer.json", info)
        if completed:
            pairs = info.get("call_pairs", [])
            require(len(pairs) == 2 * args.cycles, "synthetic observer call-pair count differs")
            require({(pair["layer"], pair["family"]) for pair in pairs}
                    == {(cycle, family) for cycle in range(args.cycles) for family in ("dispatch", "combine")},
                    "synthetic observer layer/family coverage differs")
            require(all(pair["forward_returned"] and pair["backward_returned"] for pair in pairs),
                    "synthetic observer has an incomplete forward/backward pair")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--uccl-source", type=Path,
                        help="Approved build's uccl source directory; defaults to the loaded isolated binary's build.")
    parser.add_argument("--expected-extension-sha256")
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--trace-observer", action="store_true",
                        help="Trace synthetic native Flex calls only; not a full Qwen/optimizer-step profile.")
    args = parser.parse_args()
    require(os.environ.get("SLURM_JOB_ID") and os.environ.get("SLURM_JOB_PARTITION") == "preemptable", "fresh preemptable compute required")
    require(args.output_dir.is_dir(), "output directory must be prepared once by the launcher")
    require(args.tokens in (32, 64) and args.hidden == 2048 and 2 <= args.cycles <= 4, "bounded smoke geometry required")
    require(os.environ.get("UCCL_EP_TRANSPORT") == "cxi", "explicit CXI transport required")
    require(os.environ.get("CUDA_DEVICE_ORDER") == "PCI_BUS_ID", "explicit PCI_BUS_ID CUDA ordering required")
    import torch
    import torch.distributed as dist

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    require(world in (8, 16) and int(os.environ["LOCAL_WORLD_SIZE"]) == 4, "EP8/EP16 with four workers per node required")
    require(0 <= local_rank < 4 and torch.cuda.device_count() == 4, "four node-visible GPUs and distinct torchrun local ranks required")
    torch.cuda.set_device(local_rank)
    require(torch.cuda.current_device() == local_rank, "selected CUDA device differs from worker local rank")
    write_new(args.output_dir / f"rank-{rank}-started.json", {"rank": rank, "local_rank": local_rank, "hostname": socket.gethostname(), "pid": os.getpid(), "job": os.environ["SLURM_JOB_ID"], "trace_observer": args.trace_observer})

    import uccl.ep as ep
    import deep_ep
    from deep_ep.utils import destroy_uccl
    from tools.inspect_gpu_identity import _cuda_driver_identity

    require(ep.__name__ == "uccl.ep" and ep.num_max_nvl_peers() == 4, "actual UCCL extension must be built for four local peers")
    binary = Path(ep.__file__).resolve()
    require(binary.is_file() and binary.name.endswith(".so"), "uccl.ep must resolve to a native extension")
    binary_hash = sha256(binary)
    if args.expected_extension_sha256:
        require(binary_hash == args.expected_extension_sha256, "native extension differs from the reviewed build")
    build_dir = binary.parents[2]
    require(binary.parent == build_dir / "site/uccl", "UCCL extension is outside an isolated build site")
    source = args.uccl_source.resolve() if args.uccl_source is not None else build_dir / "uccl"
    require(source == build_dir / "uccl", "source directory differs from the loaded binary's build")
    source_identity = verify_uccl_source(build_dir)
    revision = source_identity["base_revision"]
    wrapper = {}
    for name in ("deep_ep", "deep_ep.buffer", "deep_ep.utils"):
        path = Path(inspect.getfile(importlib.import_module(name))).resolve()
        relative = "__init__.py" if name == "deep_ep" else f"{name.rsplit('.', 1)[1]}.py"
        expected = source / "ep/deep_ep_wrapper/deep_ep" / relative
        require(sha256(path) == sha256(expected), f"{name}: official pinned wrapper bytes differ")
        wrapper[name] = {"path": str(path), "sha256": sha256(path)}
    require(deep_ep.Buffer.__module__ == "deep_ep.buffer", "native manager must use the official compatibility Buffer")

    dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=3), device_id=torch.device("cuda", local_rank))
    require(dist.get_rank() == rank and dist.get_world_size() == world, "torchrun/distributed identity differs")
    count = 128 // world
    owned = list(range(rank * count, (rank + 1) * count))
    identity = _cuda_driver_identity()
    require(identity.get("device_id") == local_rank and str(identity.get("uuid", "")).startswith("GPU-"), "selected physical UUID unavailable")
    topology = device_topology(ep, local_rank, identity["uuid"])
    local = {"rank": rank, "hostname": socket.gethostname(), "local_rank": local_rank,
             "local_world_size": 4, "cuda_visible_device_count": torch.cuda.device_count(),
             "cuda_current_device": torch.cuda.current_device(), "physical_gpu_uuid": identity["uuid"],
             "slurm_local_id": os.environ.get("SLURM_LOCALID"), "slurm_node_id": os.environ.get("SLURM_NODEID"),
             "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "device_topology": topology,
             "experts": owned, "extension_sha256": binary_hash, "source_revision": revision,
             "source_identity": source_identity,
             "cpu_affinity": sorted(os.sched_getaffinity(0))}
    placement = [None] * world
    dist.all_gather_object(placement, local)
    placement.sort(key=lambda row: row["rank"])
    require([row["rank"] for row in placement] == list(range(world)), "actual rank coverage")
    require(len({row["physical_gpu_uuid"] for row in placement}) == world, "workers share a selected physical GPU")
    require({row["extension_sha256"] for row in placement} == {binary_hash} and {row["source_revision"] for row in placement} == {revision}, "worker backend identities differ")
    require(all(row["source_identity"] == source_identity for row in placement), "worker approved source identities differ")
    nodes = sorted({row["hostname"] for row in placement}, key=lambda host: min(row["rank"] for row in placement if row["hostname"] == host))
    require(len(nodes) == world // 4, "actual node count differs")
    for host in nodes:
        peers = [row for row in placement if row["hostname"] == host]
        require(len(peers) == 4 and {row["local_rank"] for row in peers} == set(range(4)), "actual local worker slots differ")
        require({row["cuda_current_device"] for row in peers} == set(range(4)), "actual local CUDA devices differ")
        require(len({row["device_topology"]["selected_gpu_pci_bus_id"] for row in peers}) == 4,
                "local workers do not select distinct PCI GPUs")
        require(len({row["cuda_visible_devices"] for row in peers}) == 1,
                "local workers have inconsistent CUDA visibility ordering")
    for first in range(0, world, 4):
        require(len({row["hostname"] for row in placement[first:first + 4]}) == 1, "UCCL normal rank blocks do not match physical nodes")
    owners = [None] * 128
    local_experts = {}
    for row in placement:
        local_experts[row["rank"]] = row["experts"]
        for expert in row["experts"]:
            require(owners[expert] is None, "duplicate expert owner")
            owners[expert] = row["rank"]
    require(all(owner is not None for owner in owners), "expert ownership is incomplete")

    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.transformer.transformer_config import TransformerConfig
    from megatron.core.transformer.moe.token_dispatcher import MoEFlexTokenDispatcher
    from megatron.core.transformer.moe import fused_a2a

    require(fused_a2a.HAVE_DEEP_EP and fused_a2a.Buffer is deep_ep.Buffer and fused_a2a._buffer is None, "native adapter did not bind the verified UCCL wrapper cleanly")
    singleton = None
    for peer in range(world):
        group = dist.new_group([peer], backend="nccl")
        if peer == rank:
            singleton = group
    pgs = ProcessGroupCollection()
    pgs.tp = pgs.expt_tp = singleton
    pgs.ep = pgs.tp_ep = dist.group.WORLD
    config = TransformerConfig(num_layers=1, hidden_size=args.hidden, num_attention_heads=16,
        tensor_model_parallel_size=1, expert_model_parallel_size=world, expert_tensor_parallel_size=1,
        bf16=True, params_dtype=torch.bfloat16, use_cpu_initialization=True,
        num_moe_experts=128, moe_router_topk=8, moe_router_dtype=None,
        moe_token_dispatcher_type="flex", moe_flex_dispatcher_backend="deepep",
        moe_grouped_gemm=True, moe_permute_fusion=True, moe_expert_capacity_factor=None,
        moe_router_padding_for_quantization=False)
    def exercise_dispatcher():
        expert_ids_cpu = torch.arange(128, dtype=torch.float64)
        gains_cpu = ((expert_ids_cpu + torch.floor(expert_ids_cpu / 8)) % 8) + 1
        cases = []
        buffer_identity = None
        with trace_observation(torch, args, rank) as (observer, trace_info):
            for cycle in range(args.cycles):
                indices_cpu = torch.tensor(selected_routes(rank, args.tokens, cycle, nodes, placement, local_experts), dtype=torch.int64)
                weights_cpu = torch.tensor([1, 2, 3, 4, 5, 6, 7, 4], dtype=torch.float64).div(32).expand(args.tokens, -1)
                probabilities_cpu = torch.zeros((args.tokens, 128), dtype=torch.bfloat16)
                probabilities_cpu.scatter_(1, indices_cpu, weights_cpu.to(torch.bfloat16))
                routing_cpu = torch.zeros_like(probabilities_cpu, dtype=torch.bool).scatter_(1, indices_cpu, True)
                row = torch.arange(args.tokens).unsqueeze(1)
                feature = torch.arange(args.hidden).unsqueeze(0)
                x_cpu = torch.pow(2.0, -rank - ((row + feature + cycle) % 4)).to(torch.bfloat16)
                q_cpu = torch.pow(2.0, rank - 8 - ((row + feature + cycle + 1) % 4)).to(torch.bfloat16)
                # Encode every global source token with ten repeated bit signs. Apply
                # the same sign to q: the analytical positive dot product is unchanged.
                signs = 1 - 2 * (((rank * args.tokens + row) >> (feature % 10)) & 1)
                x_cpu = x_cpu * signs.to(torch.bfloat16)
                q_cpu = q_cpu * signs.to(torch.bfloat16)
                require(torch.equal(probabilities_cpu.float().sum(1), torch.ones(args.tokens)), "fixture weights are not normalized")
                x = x_cpu.cuda().requires_grad_()
                probabilities = probabilities_cpu.cuda().requires_grad_()
                routing = routing_cpu.cuda()
                q = q_cpu.cuda()
                scales = gains_cpu[owned].float().cuda().requires_grad_()
                dispatcher = MoEFlexTokenDispatcher(count, owned, config, pgs)
                if observer is not None:
                    observer._attach_dispatcher(dispatcher, cycle)
                flat_x, token_probs = dispatcher.dispatch_preprocess(x, routing, probabilities)
                actual_indices = dispatcher._comm_manager.token_indices.detach().cpu()
                require(torch.equal(actual_indices.sort(1).values, indices_cpu.sort(1).values), "native Flex reconstructed a different expert set")
                gathered_inputs = [None] * world
                dist.all_gather_object(gathered_inputs, {"x": x_cpu, "q": q_cpu, "probabilities": probabilities_cpu, "indices": indices_cpu})
                global_probabilities = torch.cat([item["probabilities"].double() for item in gathered_inputs])
                global_dot = torch.cat([(item["x"].double() * item["q"].double()).sum(1) for item in gathered_inputs])
                expected_counts = torch.cat([item["probabilities"].ne(0).sum(0).unsqueeze(0) for item in gathered_inputs]).sum(0)
                require(int(expected_counts.sum()) == world * args.tokens * 8, "global assignment accounting")
                unique_ranks = sum(len({owners[expert] for expert in choices}) for choices in indices_cpu.tolist())
                unique_nodes = sum(len({placement[owners[expert]]["hostname"] for expert in choices}) for choices in indices_cpu.tolist())
                require(unique_ranks == 4 * args.tokens and unique_nodes == 2 * args.tokens, "fixture does not expose both deduplication levels")
                print(json.dumps({"rank": rank, "cycle": cycle, "stage": "before_dispatch", "assignments": args.tokens * 8, "unique_token_rank": unique_ranks, "unique_token_node": unique_nodes}), flush=True)
                received_x, received_probs = dispatcher.token_dispatch(flat_x, token_probs)
                require(received_x.dtype == torch.bfloat16 and received_probs.dtype == torch.float32,
                        "native UCCL payload/probability precision differs from the intended adapter")
                require(isinstance(fused_a2a._buffer.runtime, ep.Buffer), "native buffer runtime is not UCCL")
                # Independently inspect the public layout API. This diagnostic does not
                # replace native FusedDispatch's own async layout/dispatch invocation.
                layout = fused_a2a._buffer.get_dispatch_layout(dispatcher._comm_manager.token_indices, 128)
                rank_counts, node_counts, expert_counts, membership, _ = layout
                expected_membership = torch.zeros((args.tokens, world), dtype=torch.bool)
                expected_node_membership = torch.zeros((args.tokens, len(nodes)), dtype=torch.bool)
                for token, choices in enumerate(indices_cpu.tolist()):
                    for expert in choices:
                        destination = owners[expert]
                        expected_membership[token, destination] = True
                        expected_node_membership[token, nodes.index(placement[destination]["hostname"])] = True
                require(torch.equal(membership.cpu(), expected_membership), "UCCL token/rank membership differs")
                require(torch.equal(rank_counts.cpu().to(torch.int64), expected_membership.sum(0)), "UCCL deduplicated rank counts differ")
                require(node_counts is not None and torch.equal(node_counts.cpu().to(torch.int64), expected_node_membership.sum(0)), "UCCL deduplicated node counts differ")
                require(torch.equal(expert_counts.cpu().to(torch.int64), routing_cpu.sum(0)), "UCCL per-expert assignment layout differs")
                current_buffer = id(fused_a2a._buffer)
                if buffer_identity is None:
                    buffer_identity = current_buffer
                require(current_buffer == buffer_identity, "native shared buffer was unexpectedly replaced")
                grouped_x, counts, grouped_probs = dispatcher.dispatch_postprocess(received_x, received_probs)
                counts_cpu = counts.detach().cpu().to(torch.int64)
                require(torch.equal(counts_cpu, expected_counts[owned]), "received local expert counts differ from all-source assignments")
                require(grouped_x.shape == (int(counts_cpu.sum()), args.hidden), "expert permutation shape differs")
                local_ids = torch.repeat_interleave(torch.arange(count, device=x.device), counts_cpu.cuda())
                weighted_output = (grouped_x.float() * scales[local_ids].unsqueeze(1) * grouped_probs.float().unsqueeze(1)).to(torch.bfloat16)
                restored = dispatcher.combine_preprocess(weighted_output)
                combined = dispatcher.token_combine(restored)
                output = dispatcher.combine_postprocess(combined)
                require(output.dtype == torch.bfloat16, "combine returned a different activation dtype")
                factor = (probabilities_cpu.double() * gains_cpu).sum(1).unsqueeze(1)
                expected_output = x_cpu.double() * factor
                expected_dx = q_cpu.double() * factor
                expected_dp = (x_cpu.double() * q_cpu.double()).sum(1).unsqueeze(1) * gains_cpu.unsqueeze(0) * routing_cpu
                expected_da = (global_probabilities[:, owned] * global_dot.unsqueeze(1)).sum(0)
                output_check = verify_equal(torch, output, expected_output, f"cycle{cycle}/output")
                loss = (output.float() * q.float()).sum()
                require(bool(torch.isfinite(loss)), "nonfinite synthetic loss")
                cases.append({"cycle": cycle, "dispatcher": dispatcher, "x": x, "probabilities": probabilities,
                    "scales": scales, "output": output, "loss": loss, "expected_dx": expected_dx,
                    "expected_dp": expected_dp, "expected_da": expected_da,
                    "evidence": {"inputs": x_cpu, "upstream_gradient": q_cpu, "routing_map": routing_cpu,
                        "probabilities": probabilities_cpu, "indices": indices_cpu, "expected_output": expected_output,
                        "expected_input_gradient": expected_dx, "expected_probability_gradient": expected_dp,
                        "expected_expert_scale_gradient": expected_da, "output": output.detach().cpu()},
                    "record": {"cycle": cycle, "assignments": args.tokens * 8, "unique_token_rank": unique_ranks,
                        "unique_token_node": unique_nodes, "received_expert_assignments": int(counts_cpu.sum()),
                        "actual_layout_rank_counts": rank_counts.cpu().tolist(),
                        "actual_layout_node_counts": node_counts.cpu().tolist(),
                        "actual_layout_expert_counts": expert_counts.cpu().tolist(),
                        "local_expert_counts": counts_cpu.tolist(), "output": output_check,
                        "input_dtype": str(x.dtype), "router_probability_dtype": str(probabilities.dtype),
                        "transport_probability_dtype": str(received_probs.dtype),
                        "weighted_expert_payload_dtype": str(weighted_output.dtype)}})
            require(len(cases) >= 2, "outstanding handle coverage requires two forwards")
            for case in reversed(cases):
                case["loss"].backward()
                for label, actual, expected in (("input_gradient", case["x"].grad, case["expected_dx"]),
                                                 ("probability_gradient", case["probabilities"].grad, case["expected_dp"]),
                                                 ("expert_scale_gradient", case["scales"].grad, case["expected_da"])):
                    case["record"][label] = verify_equal(torch, actual, expected, f"cycle{case['cycle']}/{label}")
                    case["evidence"][label] = actual.detach().cpu()
                print(json.dumps({"rank": rank, "cycle": case["cycle"], "stage": "backward_exact"}), flush=True)
            torch.cuda.synchronize()
        evidence_path = args.output_dir / f"rank-{rank}-tensors.pt"
        require(not evidence_path.exists(), "tensor evidence already exists")
        torch.save([case["evidence"] for case in cases], evidence_path)
        records = [case["record"] for case in cases]
        return records, trace_info, evidence_path

    records, trace_info, evidence_path = exercise_dispatcher()
    # Release all workload tensors and their recorded-stream allocator events
    # while the native communication stream is still alive.
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    dist.barrier()
    # Native get_buffer constructs the default destructor-managed Buffer. Stop
    # proxies through the official helper, then release the shared runtime on
    # every rank together; do not call the explicit-only Python destroy API.
    buffer = fused_a2a._buffer
    backend_state = {"num_nvl_bytes": buffer.num_nvl_bytes, "num_rdma_bytes": buffer.num_rdma_bytes,
                     "num_max_nvl_peers": ep.num_max_nvl_peers(), "runtime_class": f"{type(buffer.runtime).__module__}.{type(buffer.runtime).__name__}",
                     "scratch_device": str(buffer.scratch.device), "normal_mode": not buffer.low_latency_mode}
    destroy_uccl(buffer.proxies, buffer.workers)
    # Release the destructor-managed C++ runtime while its scratch allocation
    # and IPC backing tensors are still alive. Calling runtime.destroy() here
    # would double-destroy when its default native destructor runs.
    runtime = buffer.runtime
    buffer.runtime = None
    del runtime
    gc.collect()
    dist.barrier()
    fused_a2a._buffer = None
    del buffer
    gc.collect()
    dist.barrier()
    result = {"schema_version": 1, "valid": True, "rank": rank, "world_size": world,
        "job": os.environ["SLURM_JOB_ID"], "placement": local, "hidden": args.hidden, "tokens_per_source": args.tokens,
        "expert_count": 128, "topk": 8, "cycles": records, "outstanding_forward_handles": len(records),
        "backward_order": list(reversed(range(args.cycles))), "native_shared_buffer_reused": True,
        "backend": {"module": ep.__name__, "binary": str(binary), "binary_sha256": binary_hash,
                    "source": str(source), "source_revision": revision, "source_identity": source_identity,
                    "official_wrapper": wrapper, "transport": os.environ["UCCL_EP_TRANSPORT"], **backend_state},
        "native_adapter_sources": {str(Path(inspect.getfile(owner)).resolve()): sha256(inspect.getfile(owner))
                                   for owner in (MoEFlexTokenDispatcher, fused_a2a)},
        "tensor_evidence": {"path": str(evidence_path.resolve()), "sha256": sha256(evidence_path)},
        "smoke_source_sha256": sha256(__file__), "physical_traffic_measured": False,
        "trace_observer": trace_info,
        "precision": "BF16 inputs/weights/payload; FP32 expert scale parameters; strict exact dyadic references; leaf gradients cast once to their leaf dtype"}
    write_new(args.output_dir / f"rank-{rank}.json", result)
    summaries = [None] * world
    dist.all_gather_object(summaries, {"rank": rank, "valid": True, "result_sha256": sha256(args.output_dir / f"rank-{rank}.json")})
    if rank == 0:
        write_new(args.output_dir / "model_evidence.json", {"valid": True, "kind": "native_flex_uccl_dispatcher_smoke",
            "world_size": world, "placement": placement, "ranks": summaries, "cycles": args.cycles,
            "all_forward_before_reverse_backward": True, "strict_absolute_tolerance": 0, "strict_relative_tolerance": 0,
            "trace_observer_enabled": args.trace_observer,
            "limitations": ["Synthetic linear experts, not a complete Qwen training or optimizer-step result.",
                            "Logical assignment/rank/node counts are not physical traffic measurements."]})
    dist.barrier()
    dist.destroy_process_group()
    print(json.dumps({"rank": rank, "valid": True, "stage": "complete"}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # C++ Buffer destruction includes a local GPU collective barrier. A
        # failed worker cannot safely enter it while its peers are elsewhere.
        # Preserve failure evidence, then let torchrun terminate the peer group.
        rank = os.environ.get("RANK", "unknown")
        detail = {"valid": False, "rank": rank, "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()}
        try:
            if "--output-dir" in sys.argv:
                destination = Path(sys.argv[sys.argv.index("--output-dir") + 1])
                write_new(destination / f"rank-{rank}-failure.json", detail)
        except BaseException:
            pass
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
