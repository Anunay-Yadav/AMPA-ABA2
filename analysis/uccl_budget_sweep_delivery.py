#!/usr/bin/env python3
"""Build a bound, portable index of four accepted UCCL Nsight diagnostics."""
from __future__ import annotations

import argparse
import copy
import csv
import datetime
import hashlib
import json
from pathlib import Path
import statistics
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.accept_uccl_pilot import Inputs, equal, equal_resolved_configuration, require


BUDGETS = (20, 12, 32, 40)


def write_new(path, record):
    with path.open("x") as stream:
        json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def normalized_config(configuration, run, budget):
    """Remove only the requested budget and three checked output locations."""
    value = copy.deepcopy(configuration)
    equal(value["model"].pop("moe_flex_dispatcher_num_sms"), budget, "resolved native budget")
    for key, expected in {"tensorboard_dir": str(run / "logs/tensorboard"),
                          "wandb_exp_name": run.name, "wandb_save_dir": str(run / "logs/wandb")}.items():
        equal(value["logger"].pop(key), expected, f"resolved output {key}")
    return value


def placement_key(rows):
    return [{key: row[key] for key in ("rank", "hostname", "local_rank", "physical_gpu_uuid")}
            for row in sorted(rows, key=lambda value: value["rank"])]


def native_runtime_identity(run, orchestration, inputs):
    environment = inputs.read(run / "environment.json")
    result = {}
    for label in ("megatron_lm", "megatron_bridge"):
        identity = environment[label]
        require(isinstance(identity, dict) and all(isinstance(identity.get(key), str)
                for key in ("revision", "status", "diff_sha256")), "incomplete native Git identity")
        for kind in ("revision", "status", "diff"):
            path = orchestration / "source" / f"{label}_{kind}.txt"
            inputs.bind(path)
            value = path.read_text().strip()
            if kind == "diff":
                value = hashlib.sha256(value.encode()).hexdigest()
                kind = "diff_sha256"
            equal(value, identity[kind], f"archived and runtime native {label} {kind}")
        result[label] = identity
    for key in ("pytorch_version", "cuda_version", "nccl_version", "container_sqsh_sha256",
                "hf_model_id", "hf_model_revision", "gpu_name", "cuda_allocator_conf"):
        require(key in environment, f"runtime identity missing {key}")
        result[key] = environment[key]
    result["model_manifest_sha256"] = inputs.bind(run / "config/model_manifest.json")
    return result


def checked_case(budget, job, orchestration, run, output, inputs):
    path = output / "cases" / f"sms{budget}" / "acceptance.json"
    accepted = inputs.read(path)
    for key in ("valid_budget_run", "valid_training", "valid_native_identity", "valid_callback_capture", "valid_nsight_timeline"):
        equal(accepted.get(key), True, f"budget{budget} {key}")
    equal(accepted.get("whole_session_event_completeness"), "unresolved", "explicit session-completeness limit")
    require(isinstance(accepted.get("quantitative_event_scope"), str) and accepted["quantitative_event_scope"],
            "accepted timeline must state its checked event set")
    for key, expected in {"communication_sms": budget, "training_slurm_job_id": job,
                          "run_dir": str(run), "orchestration_dir": str(orchestration), "consumed_cpu_batches": 640}.items():
        equal(accepted.get(key), expected, f"accepted identity {key}")
    for record in accepted["input_files"]:
        inputs.bind(record["path"], record["sha256"])
    config_path = run / "config/resolved.yaml"
    inputs.bind(config_path)
    config = normalized_config(yaml.safe_load(config_path.read_text()), run, budget)
    accepted["native_runtime_identity"] = native_runtime_identity(run, orchestration, inputs)
    inventory = inputs.read(path.parent / "nsight_inventory.json")
    nodes = inventory["nodes"]
    equal(sorted(row["node_rank"] for row in nodes), [0, 1, 2, 3], "four report inventory nodes")
    reports, payloads = [], []
    for node in sorted(nodes, key=lambda item: item["node_rank"]):
        equal(node.get("valid_nsight_timeline"), True, "report timeline acceptance")
        report = run / "nsight" / f"node-{node['node_rank']}-native.nsys-rep"
        require(report.stat().st_size > 0, "empty native report")
        digest = inputs.bind(report)
        equal(digest, node["export"]["report_sha256"], "native report export binding")
        reports.append({"communication_sms": budget, "node_rank": node["node_rank"],
                        "path": str(report), "sha256": digest, "bytes": report.stat().st_size,
                        "hardware_metrics": node.get("hardware_metrics", {})})
        payloads.extend(dict(communication_sms=budget, **row) for row in node["payload_by_rank_step"])
    equal(len(payloads), 160, "16 ranks x five updates x two payload families")
    return accepted, config, reports, payloads


def build(control, output, capability):
    inputs = Inputs()
    inputs.bind(Path(__file__))
    table = control / "acquisitions.tsv"
    inputs.bind(table)
    rows = list(csv.reader(table.read_text().splitlines(), delimiter="\t"))
    require(len(rows) == 4 and all(len(row) == 4 for row in rows), "four acquisition rows required")
    equal([int(row[0]) for row in rows], list(BUDGETS), "ordered budget sweep")
    cases, reports, activity = [], [], []
    first = first_config = None
    for row in rows:
        budget, job, orchestration, run = int(row[0]), row[1], Path(row[2]), Path(row[3])
        require(job.isdigit() and orchestration.parent == ROOT / "runs" and run.parent == ROOT / "runs",
                "invalid acquisition identity")
        accepted, configuration, native, payload = checked_case(budget, job, orchestration, run, output, inputs)
        if first is None:
            first, first_config = accepted, configuration
        for key in ("c0_identity_manifest_sha256", "dataset_manifest_sha256", "runtime_source_sha256",
                    "binary_sha256", "adapter_source_sha256", "sample_prefix_sha256", "native_runtime_identity"):
            equal(accepted[key], first[key], f"cross-budget fixed {key}")
        equal_resolved_configuration(configuration, first_config, "only budget and output locations may differ")
        timing = accepted["timing"]
        values = [record["mean_optimizer_step_ms"] for record in timing["by_update"]]
        equal(timing["count"], 5, "five measured complete updates")
        equal(len(values), 5, "five timing records")
        equal(timing["mean"], statistics.fmean(values), "recorded timer mean")
        equal(timing["sample_sd"], statistics.stdev(values), "recorded sample deviation")
        cases.append({"communication_sms": budget, "training_job": job, "run_dir": str(run),
                      "acceptance_path": str(output / "cases" / f"sms{budget}" / "acceptance.json"),
                      "acceptance_sha256": inputs.bind(output / "cases" / f"sms{budget}" / "acceptance.json"),
                      "mean_step_seconds": timing["mean"] / 1000, "sample_sd_seconds": timing["sample_sd"] / 1000,
                      "coefficient_of_variation_percent": 100 * timing["sample_sd"] / timing["mean"],
                      "measured_step_seconds": [value / 1000 for value in values],
                      "same_rank_gpu_placement_as_budget20": placement_key(accepted["rank_placement"]) == placement_key(first["rank_placement"]),
                      "rank_placement": placement_key(accepted["rank_placement"]),
                      "native_runtime_identity": accepted["native_runtime_identity"],
                      "quantitative_event_scope": accepted["quantitative_event_scope"],
                      "whole_session_event_completeness": "unresolved",
                      "hardware_metrics": accepted["hardware_metrics"], "effective_configs": accepted["effective_configs"]})
        reports.extend(native)
        activity.extend(payload)
    equal(len(reports), 16, "sixteen native reports")
    equal(len({row["path"] for row in reports}), 16, "unique report paths")
    cap = inputs.read(capability / "capabilities.json")
    equal(cap.get("native_report_generated"), True, "native capability proof")
    version = capability / "logs/nsys_version.out"
    version_text = "See the capability report for the collector version."
    if version.is_file():
        inputs.bind(version)
        version_text = version.read_text().strip()
    with (output / "diagnostic_timing.csv").open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["communication_blocks", "measured_update_index", "complete_step_seconds", "timing_kind"])
        for case in cases:
            for step, seconds in enumerate(case["measured_step_seconds"], start=5):
                writer.writerow([case["communication_sms"], step, seconds, "Nsight diagnostic"])
    with (output / "payload_activity.csv").open("x", newline="") as stream:
        fields = ["communication_sms", "rank", "step", "family", "count", "gpu_activity_sum_ns"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in activity)
    write_new(output / "native_reports.json", {"collector_version": version_text, "reports": reports})
    lines = ["# UCCL 12 / 20 / 32 / 40: Nsight timelines", "",
             "All four runs passed training, input, routing, native-budget and scoped timeline checks. "
             "Each starts independently from the same Qwen3-30B-A3B-Base checkpoint and consumes the same input prefix. "
             "Routing uses the model's ordinary top8. Each run has five warmup and five measured complete optimizer updates.", "",
             "Timeline acceptance covers the exact source-bound AMPA call ranges and UCCL payload launches. "
             "Their required counts, owners, forward/backward identities, launch correlations and final activity are checked. "
             "Whole-session CUDA/NVTX event completeness remains unresolved. Generic possible-loss advisories stay in the report inventories. "
             "Missing required events, explicit drops, errors and unknown warnings reject acceptance.", "",
             "## Open on your PC", "",
             "Copy the native `.nsys-rep` files below to your PC. Open each file in the Nsight Systems desktop GUI. "
             "Use a GUI version compatible with this collector; the same version or a newer version is the first choice.", "",
             f"Collector: `{version_text.replace(chr(10), ' ')}`", "",
             "There are four reports per budget. Each report covers one node and its four training GPU processes. "
             "The reports cover the five measured updates, with zero-based NVTX labels `AMPA/STEP/5` through `AMPA/STEP/9` "
             "(human updates 6–10). Warmup is outside the capture window. The four files are separate node timelines; "
             "timestamps from different files must not be treated as a synchronized cross-node clock without calibration.", "",
             "Expand each process's CUDA and NVTX tracks. Locate an `AMPA/STEP` range, then inspect the dispatch, expert, "
             "combine, backward and optimizer scopes. CUDA runtime-to-kernel correlation links launches to GPU work. "
             "Compare the checked payload kernel grid and duration at the same rank and update. "
             "Other displayed GPU activity is exploratory; its complete capture is not independently established. "
             "A long communication kernel may include polling; its duration alone does not show that every resident warp does useful work.", "",
             "## Diagnostic timings", "",
             "These times include Nsight and observer overhead. They are not profiler-free throughput. "
             "Five updates describe these runs; one allocation per budget does not establish repeatability or statistical equivalence. "
             "CV is the sample standard deviation as a percentage of the mean: the spread across these five consecutive updates.", "",
             "| Communication blocks | Mean step (s) | Sample SD (s) | CV (%) | Same rank/GPU placement as 20 | GPU metric rows, all nodes |",
             "| --- | ---: | ---: | ---: | --- | --- |"]
    for case in cases:
        lines.append(f"| {case['communication_sms']} | {case['mean_step_seconds']:.3f} | {case['sample_sd_seconds']:.3f} | "
                     f"{case['coefficient_of_variation_percent']:.2f} | {case['same_rank_gpu_placement_as_budget20']} | "
                     f"{case['hardware_metrics']['sample_rows_present_all_nodes']} |")
    lines += ["", "The budget is a launch parameter, not a count of permanently reserved physical SMs. "
              "Counter rows and actual metric names are listed per report in `native_reports.json` and the case inventories. "
              "Counter absence means active SMs and achieved warp occupancy are unavailable. Even when counter rows exist, "
              "only the metrics actually named in the report are available. Device-wide samples may include concurrent compute; "
              "do not label them communication-only occupancy. Raw metric summaries do not establish qualified SM/warp values: "
              "device identity, the measured window, units and sample gaps need separate checks. See the [NVIDIA GPU metrics guide]"
              "(https://docs.nvidia.com/nsight-systems/UserGuide/index.html#gpu-metrics).", "",
              "`diagnostic_timing.csv` contains all 20 measured step times. `payload_activity.csv` contains separate dispatch "
              "and combine kernel-activity sums for each rank/update. Those sums can overlap and are not additive parts "
              "of the full step, physical traffic, or exposed communication delay.", "", "## Native reports", "",
              "| Blocks | Node | Native report | Bytes | SHA256 |", "| --- | --- | --- | ---: | --- |"]
    for report in reports:
        path = Path(report["path"])
        lines.append(f"| {report['communication_sms']} | {report['node_rank']} | [{path.name}]({path}) | "
                     f"{report['bytes']} | `{report['sha256']}` |")
    lines += ["", "## Evidence and limits", "",
              "Each `cases/sms{12,20,32,40}/acceptance.json` binds the actual records and exported timelines. "
              "The case folders contain health, routing/receive checks, timing and Nsight inventories. "
              "All 16 ranks and all five measured updates have source-bound call journals and correlated payload launches. "
              "The launch parameter is also checked against actual kernel grids and preallocation buffer hints.", "",
              "Quantitative timeline claims cover the checked UCCL payload event set. Unenumerated expert/optimizer GPU activity, "
              "exposed communication delay and a warning-free whole process are not established. Complete optimizer-step wall time "
              "comes from the independent training timer and health records.", "",
              "Selected source hashes, native Git revision/status/diff identities, recorded software versions and model manifests "
              "match across the four budgets. These records do not hash every installed container file. "
              "Process/device identity joins follow the [NVIDIA analysis schema]"
              "(https://docs.nvidia.com/nsight-systems/AnalysisGuide/).", "",
              "Historical immutable checkpoint/data payload verification was reused by explicit agreement. "
              "This is not a fresh payload rehash or a full activation/gradient equivalence test. "
              "Finite training-health values and successful updates are required. A faster diagnostic is a candidate "
              "for a later matched profiler-free comparison, not a proven optimization.", ""]
    with (output / "README.md").open("x") as stream:
        stream.write("\n".join(lines))
    for name in ("README.md", "diagnostic_timing.csv", "payload_activity.csv", "native_reports.json"):
        inputs.bind(output / name)
    inputs.recheck()
    write_new(output / "DELIVERY.json", {"valid_delivery": True, "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "control_dir": str(control), "collector_version": version_text, "cases": cases, "native_reports": 16,
              "same_workload_and_input_prefix": True,
              "whole_session_event_completeness": "unresolved",
              "quantitative_scope": ["complete optimizer-step wall time from training records",
                                     "source-bound AMPA call ranges and exact UCCL payload launches"],
              "unenumerated_gpu_activity_quantitatively_accepted": False,
              "raw_metric_values_quantitatively_qualified": False,
              "same_rank_gpu_placement_all_cases": all(case["same_rank_gpu_placement_as_budget20"] for case in cases),
              "timing_kind": "Nsight-instrumented diagnostic; no profiler-free speedup or occupancy conclusion", 
              "input_files": [{"path": name, "sha256": digest} for name, digest in sorted(inputs.hashes.items())]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-dir", required=True, type=Path)
    parser.add_argument("--posthoc-dir", required=True, type=Path)
    parser.add_argument("--capability-root", required=True, type=Path)
    args = parser.parse_args()
    control, output, capability = args.control_dir.resolve(), args.posthoc_dir.resolve(), args.capability_root.resolve()
    require(all(path.is_dir() and path.parent == ROOT / "runs" for path in (control, output, capability)), "existing runs roots required")
    build(control, output, capability)
    print(json.dumps({"valid_delivery": True, "output": str(output)}), flush=True)


if __name__ == "__main__":
    main()
