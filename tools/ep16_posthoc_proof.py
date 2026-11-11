#!/usr/bin/env python3
"""Bind separate EP16 reanalysis to immutable training evidence and new sources."""

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket


ROUTES = ("normal", "group4-top2")
ALLOWED_ANALYSIS_CHANGES = (
    "analysis/aggregate_deep_trace.py", "analysis/validate_torch_trace.py",
    "analysis/analyze_all_step_trace.py", "analysis/compare_ep16_recovery.py",
)
POSTHOC_ENTRYPOINTS = ("tools/ep16_posthoc_proof.py", "jobs/smoke/ep16_posthoc_reanalysis.sbatch")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f"missing or symlinked evidence: {path}")
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def inventory(files):
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return {"files": files, "sha256": hashlib.sha256(encoded).hexdigest()}


def source_changes(original, current):
    require(set(original["files"]) <= set(current["files"]), "posthoc inventory omits training sources")
    changed = sorted(name for name, digest in original["files"].items()
                     if current["files"][name] != digest)
    require(set(changed) <= set(ALLOWED_ANALYSIS_CHANGES), "non-analysis training source changed")
    return changed


def import_closure(root, seeds):
    """Bind static repository-local imports; native checkouts are bound separately."""
    root = Path(root)
    search_roots = (root, root / "analysis", root / "profiling")
    pending, found = list(seeds), set()

    def resolve(name):
        suffix = Path(*name.split("."))
        for base in search_roots:
            for candidate in (base / suffix.with_suffix(".py"), base / suffix / "__init__.py"):
                if candidate.is_file():
                    return candidate.relative_to(root).as_posix()
        return None

    while pending:
        relative = pending.pop()
        if relative in found:
            continue
        path = root / relative
        require(path.is_file() and not path.is_symlink(), f"missing import-closure source: {relative}")
        found.add(relative)
        for parent in path.parents:
            if parent == root:
                break
            initializer = parent / "__init__.py"
            if initializer.is_file():
                pending.append(initializer.relative_to(root).as_posix())
        if path.suffix != ".py":
            continue
        for node in ast.walk(ast.parse(path.read_text(), filename=relative)):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    package = list(Path(relative).parent.parts)
                    package = package[:len(package) - node.level + 1]
                    prefix = ".".join(package + ([node.module] if node.module else []))
                else:
                    prefix = node.module or ""
                names = [prefix] + [f"{prefix}.{alias.name}" for alias in node.names if alias.name != "*"]
            for name in names:
                resolved = resolve(name) if name else None
                if resolved:
                    pending.append(resolved)
    return sorted(found)


def current_inventory(root, original):
    from ampa_profile.provenance import _ampa_runtime_sources

    current = _ampa_runtime_sources(root)
    require(set(current["files"]) == set(original["files"]), "original runtime inventory membership changed")
    closure = import_closure(root, ALLOWED_ANALYSIS_CHANGES + POSTHOC_ENTRYPOINTS)
    names = set(current["files"]) | set(closure)
    result = inventory({name: sha256(root / name) for name in sorted(names)})
    source_changes(original, result)
    return result, closure


def binding(path):
    return {"path": str(Path(path).resolve()), "sha256": sha256(path)}


def check_binding(record):
    require(isinstance(record, dict) and set(record) == {"path", "sha256"}, "invalid evidence binding")
    path = Path(record["path"])
    require(path.is_absolute() and sha256(path) == record["sha256"], "bound evidence bytes changed")
    return path


def snapshot_identity(path):
    entries = {}
    for line in Path(path).read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        source = Path(name)
        require(source.is_absolute() and str(source) not in entries, "invalid original orchestration snapshot")
        require(re.fullmatch(r"[0-9a-f]{64}", digest), "invalid original orchestration digest")
        entries[str(source)] = digest
    return entries


def original_state(root, pair_id):
    from ampa_profile.provenance import _git_identity

    orchestration = root / "runs" / f"{pair_id}-reviewed-orchestration"
    input_path = orchestration / "reviewed_inputs.json"
    require(not input_path.is_symlink(), "original training proof is a symlink")
    inputs = read(input_path)
    require(inputs.get("valid_reviewed_inputs") is True
            and inputs.get("checkpoint_payload", {}).get("verified") is True, "original input proof is not passed")
    original = inputs["ampa_runtime_sources"]
    require(inventory(original["files"]) == original, "original source inventory digest differs")
    evidence = {str(input_path.resolve()): sha256(input_path)}
    original_archive = orchestration / "source/runtime"
    for name, digest in original["files"].items():
        require(sha256(original_archive / name) == digest, "original runtime source archive changed")
    runs, run_evidence = {}, {}
    job = inputs.get("slurm_job_id")
    require(isinstance(job, str) and job, "original training allocation is missing")
    for route in ROUTES:
        runs[route], run_evidence[route] = {}, {}
        for leg, suffix in (("control", "control"), ("trace", "torchprof-all")):
            run = root / "runs" / f"{pair_id}-cp1-{route}-{suffix}"
            runs[route][leg] = str(run.resolve())
            env_path = run / "environment.json"
            env = read(env_path)
            require(env.get("slurm_job_id") == job and env.get("ampa_runtime_sources") == original,
                    "training allocation/source identity differs from original proof")
            require(env.get("reviewed_inputs_proof") == binding(input_path), "training input-proof binding differs")
            for checkout, key in (("Megatron-LM", "megatron_lm"), ("Megatron-Bridge", "megatron_bridge")):
                require(_git_identity(root / checkout) == env[key] == inputs["native_checkouts"][key],
                        "native checkout changed after training")
            files = [env_path] + [run / "config" / name for name in (
                "comparison_contract.json", "c0_manifest.json", "dataset_manifest.json", "model_manifest.json",
                "resolved.yaml", "observation_mode.json", "all_step_capture.json")]
            contract = read(run / "config/comparison_contract.json")
            require(contract["c0_identity_manifest_sha256"] == inputs["c0_identity_manifest_sha256"]
                    and contract["dataset"]["manifest_sha256"] == inputs["dataset_manifest_sha256"],
                    "training C0/data identity differs from original input proof")
            require(sha256(run / "config/c0_manifest.json") == inputs["c0_identity_manifest_sha256"]
                    and sha256(run / "config/dataset_manifest.json") == inputs["dataset_manifest_sha256"],
                    "training C0/data manifest bytes changed")
            for rank in range(16):
                files.extend((run / "raw" / f"routing_audit_rank-{rank}.json",
                              run / "raw" / f"sample_fingerprint_rank-{rank}.json",
                              run / "logs" / f"rank-{rank}.out"))
                if leg == "trace":
                    for step in range(25):
                        files.extend((run / "traces/torch_profile" / f"step-{step:05d}" / f"rank-{rank}.json.gz",
                                      run / "raw/deep_trace" / f"step-{step:05d}" / f"rank-{rank}.json"))
            digests = {str(path.resolve()): sha256(path) for path in files}
            evidence.update(digests)
            run_evidence[route][leg] = digests
    for name in ("c0_manifest.json", "dataset_manifest.json", "model_manifest.json"):
        values = {evidence[str((Path(runs[route][leg]) / "config" / name).resolve())]
                  for route in ROUTES for leg in ("control", "trace")}
        require(len(values) == 1, "four-run input manifests differ")
    snapshot = orchestration / "source_sha256.txt"
    identities = snapshot_identity(snapshot)
    expected = {str(root / "analysis/compare_ep16_recovery.py"),
                str(root / "jobs/training/ep16_all_steps_reviewed.sbatch")}
    require(set(identities) == expected, "original orchestration snapshot has unexpected members")
    for source, digest in identities.items():
        relative = Path(source).relative_to(root).as_posix()
        require(original["files"][relative] == digest, "original snapshot differs from training inventory")
        require(sha256(orchestration / "source" / Path(source).name) == digest,
                "original orchestration source archive changed")
    evidence[str(snapshot.resolve())] = sha256(snapshot)
    return inputs, runs, run_evidence, evidence, identities


def posthoc_environment(root):
    import tomllib
    import torch

    configuration = root / "setup/gh200-cuda130-runtime.toml"
    image = Path(tomllib.loads(configuration.read_text())["image"])
    require(image.is_absolute(), "posthoc container image path is not absolute")
    return {"slurm_job_id": os.environ.get("SLURM_JOB_ID"), "hostname": socket.gethostname(),
            "container_environment": binding(configuration), "container_image": binding(image),
            "pytorch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
            "nccl_version": list(torch.cuda.nccl.version())}


def check_posthoc_environment(runtime, inputs, training_environment):
    require(all(runtime.get(key) == value for key, value in inputs["runtime_versions"].items()),
            "posthoc PyTorch/CUDA/NCCL runtime differs from training")
    require(runtime["container_image"]["sha256"] == training_environment["container_sqsh_sha256"],
            "posthoc container image bytes differ from the training image identity")


def failure_evidence(root, pair_id, job):
    suite = root / "runs" / f"{pair_id}-suite"
    orchestration = root / "runs" / f"{pair_id}-reviewed-orchestration"
    current, timeline = suite / "CURRENT.md", suite / "timeline.md"
    text, history = current.read_text(), timeline.read_text()
    require(f"job {job}: EXIT code=143" in text, "original suite failure is not the retained status 143")
    require(f"BEFORE compute all-step validation {root}/runs/{pair_id}-cp1-group4-top2-torchprof-all" in history,
            "original failure lacks grouped validation stage")
    require("BEFORE compute matched comparison" not in history and "COMPLETE matched comparison" not in history,
            "original strict comparator ran")
    require(not (root / "runs" / f"{pair_id}-comparison").exists()
            and not (root / "runs" / f"{pair_id}-reviewed-comparison").exists(),
            "original comparison output exists")
    rank_log = root / "runs" / f"{pair_id}-cp1-group4-top2-torchprof-all/logs/all-step-analysis-rank-11.out"
    rank_history = rank_log.read_text()
    require("untagged NCCL kernel lacks selected-step CP KV peer ownership or overlaps EP A2A" in rank_history,
            "original grouped rank-11 validation failure is missing")
    completed = [json.loads(line) for line in rank_history.splitlines() if line.startswith('{"rank":')]
    require(completed == [{"rank": 11, "step": step, "valid": True} for step in range(14)],
            "original failing rank/step is not grounded by the retained validation log")
    additional = (timeline, orchestration / "CURRENT.md", orchestration / "timeline.md", rank_log,
                  root / ".runtime/slurms" / f"slurm-ampa-ep16-reviewed-{job}.err")
    return {**binding(current), "exit_status": 143, "stage": "grouped_trace_validation",
            "strict_comparator_ran": False, "failed_rank": 11, "failed_step": 14,
            "evidence": [binding(path) for path in additional]}


def validate_output_root(output, root, pair_id, runs):
    """Keep every new artifact outside the original training and suite roots."""
    output = Path(output)
    original_roots = {Path(run) for route in runs.values() for run in route.values()}
    original_roots.update(root / "runs" / f"{pair_id}-{suffix}" for suffix in
                          ("reviewed-orchestration", "suite", "comparison", "reviewed-comparison"))
    require(output.is_absolute() and output == output.resolve() and not output.is_symlink()
            and output.parent == root / "runs" and output not in original_roots,
            "posthoc output must be a separate directory directly under runs")


def validate_posthoc_proof(path, root, *, runs=None, analysis_dirs=None):
    """Check both source generations and all bound bytes without rewriting either."""
    path, root = Path(path), Path(root).resolve()
    proof_hash = sha256(path)
    proof = read(path)
    require(type(proof.get("schema_version")) is int and proof["schema_version"] == 1
            and proof.get("valid_posthoc_analysis") is True,
            "invalid posthoc proof schema")
    training_job, posthoc_job = proof.get("training_slurm_job_id"), proof.get("posthoc_slurm_job_id")
    require(isinstance(training_job, str) and isinstance(posthoc_job, str) and training_job
            and posthoc_job and training_job != posthoc_job, "posthoc allocation must differ from training")
    require(os.environ.get("SLURM_JOB_ID") == posthoc_job, "posthoc proof belongs to a different allocation")
    require(proof.get("allowed_analysis_changes") == list(ALLOWED_ANALYSIS_CHANGES), "posthoc analysis allowlist changed")
    require(re.fullmatch(r"[a-zA-Z0-9_-]+", proof.get("pair_id", "")), "invalid original pair ID")
    failure_record = proof.get("original_suite_failure")
    require(isinstance(failure_record, dict)
            and all(type(failure_record.get(key)) is int for key in ("exit_status", "failed_rank", "failed_step"))
            and failure_record.get("strict_comparator_ran") is False,
            "original failure counts must be exact integers and strict comparator flag must be false")
    actual_inputs, actual_runs, actual_run_evidence, actual_evidence, actual_identities = original_state(root, proof["pair_id"])
    validate_output_root(path.absolute().parent, root, proof["pair_id"], actual_runs)
    expected_input_path = root / "runs" / f"{proof['pair_id']}-reviewed-orchestration/reviewed_inputs.json"
    require(proof["original_reviewed_inputs"] == binding(expected_input_path), "original proof path or bytes substituted")
    require(proof["runs"] == actual_runs and proof["run_evidence"] == actual_run_evidence
            and proof["original_evidence_sha256"] == actual_evidence, "original evidence inventory omitted or substituted files")
    require(proof["original_orchestration_source_sha256"] == actual_identities,
            "original orchestration source identity differs")
    original_input_path = check_binding(proof["original_reviewed_inputs"])
    inputs = read(original_input_path)
    require(inputs == actual_inputs and inputs["ampa_runtime_sources"] == proof["original_runtime_sources"]
            and inputs["slurm_job_id"] == training_job, "original training proof identity differs")
    original = proof["original_runtime_sources"]
    require(inventory(original["files"]) == original, "original source inventory digest differs")
    for relative, digest in original["files"].items():
        require(sha256(original_input_path.parent / "source/runtime" / relative) == digest,
                "original source archive changed")
    current, closure = current_inventory(root, original)
    require(current == proof["posthoc_runtime_sources"] and closure == proof["import_closure"],
            "posthoc current sources or import closure changed")
    require(proof["changed_original_analysis_sources"] == source_changes(original, current),
            "posthoc source-change list differs")
    archive = Path(proof["source_archive_dir"])
    require(archive.is_absolute() and archive == path.resolve().parent / "source/runtime", "posthoc source archive path differs")
    for relative, digest in current["files"].items():
        require(sha256(archive / relative) == digest, "posthoc archived source bytes changed")
    if runs is not None:
        require(proof["runs"] == runs, "posthoc run paths differ")
    if analysis_dirs is not None:
        require(proof["analysis_dirs"] == analysis_dirs, "posthoc analysis paths differ")
    expected_dirs = {route: str(path.resolve().parent / "analysis" / route) for route in ROUTES}
    require(proof["analysis_dirs"] == expected_dirs, "posthoc analysis must use the proof's new output roots")
    for route in ROUTES:
        require(not Path(expected_dirs[route]).is_symlink(), "posthoc analysis root is a symlink")
        for leg in ("control", "trace"):
            env = read(Path(proof["runs"][route][leg]) / "environment.json")
            require(env["slurm_job_id"] == training_job and env["ampa_runtime_sources"] == original,
                    "posthoc evidence substitutes a training environment")
    failure = failure_evidence(root, proof["pair_id"], training_job)
    require(failure == proof["original_suite_failure"], "original failure evidence changed")
    snapshot = check_binding(proof["original_orchestration_sources"])
    require(snapshot_identity(snapshot) == proof["original_orchestration_source_sha256"],
            "original orchestration snapshot changed")
    runtime = posthoc_environment(root)
    require(proof["posthoc_environment"] == runtime, "posthoc container/runtime/node identity changed")
    check_posthoc_environment(runtime, inputs, read(Path(actual_runs["normal"]["control"]) / "environment.json"))
    require(sha256(path) == proof_hash, "posthoc proof changed during validation")
    proof["binding"] = {"proof_path": str(path.resolve()), "proof_sha256": proof_hash,
                        "training_slurm_job_id": training_job, "posthoc_slurm_job_id": posthoc_job}
    return proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--pair-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    require(re.fullmatch(r"[a-zA-Z0-9_-]+", args.pair_id), "invalid original pair ID")
    require(os.environ.get("SLURM_JOB_PARTITION") == "preemptable" and os.environ.get("SLURM_JOB_ID"),
            "posthoc proof requires a preemptable Slurm allocation")
    root, output = args.root.resolve(), args.output_dir
    require(output.is_absolute() and output == output.resolve() and not output.is_symlink()
            and output.parent == root / "runs", "posthoc output must be directly under runs")
    proof_path = output / "posthoc_proof.json"
    require(not proof_path.exists() and not proof_path.is_symlink(), "posthoc proof already exists")
    inputs, runs, run_evidence, evidence, identities = original_state(root, args.pair_id)
    validate_output_root(output, root, args.pair_id, runs)
    output.mkdir(parents=True, exist_ok=True)
    job = os.environ["SLURM_JOB_ID"]
    require(job != inputs["slurm_job_id"], "posthoc allocation cannot be the training allocation")
    runtime = posthoc_environment(root)
    check_posthoc_environment(runtime, inputs, read(Path(runs["normal"]["control"]) / "environment.json"))
    current, closure = current_inventory(root, inputs["ampa_runtime_sources"])
    source_archive = output / "source/runtime"
    source_archive.mkdir(parents=True, exist_ok=False)
    for relative, digest in current["files"].items():
        destination = source_archive / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / relative, destination)
        require(sha256(destination) == digest == sha256(root / relative), "source changed during posthoc snapshot")
    proof = {"schema_version": 1, "valid_posthoc_analysis": True, "pair_id": args.pair_id,
             "training_slurm_job_id": inputs["slurm_job_id"], "posthoc_slurm_job_id": job,
             "original_reviewed_inputs": binding(root / "runs" / f"{args.pair_id}-reviewed-orchestration/reviewed_inputs.json"),
             "original_runtime_sources": inputs["ampa_runtime_sources"], "posthoc_runtime_sources": current,
             "allowed_analysis_changes": list(ALLOWED_ANALYSIS_CHANGES),
             "changed_original_analysis_sources": source_changes(inputs["ampa_runtime_sources"], current),
             "import_closure": closure, "import_closure_policy": "static repository-local imports; native checkout identities bound separately",
             "source_archive_dir": str(source_archive.resolve()), "runs": runs, "run_evidence": run_evidence,
             "posthoc_environment": runtime,
             "analysis_dirs": {route: str(output / "analysis" / route) for route in ROUTES},
             "original_evidence_sha256": evidence,
             "original_suite_failure": failure_evidence(root, args.pair_id, inputs["slurm_job_id"]),
             "original_orchestration_sources": binding(root / "runs" / f"{args.pair_id}-reviewed-orchestration/source_sha256.txt"),
             "original_orchestration_source_sha256": identities}
    with proof_path.open("x") as stream:
        json.dump(proof, stream, indent=2, sort_keys=True)
        stream.write("\n")
    validate_posthoc_proof(proof_path, root)
    print(json.dumps({"posthoc_proof": str(proof_path), "posthoc_slurm_job_id": job,
                      "training_slurm_job_id": inputs["slurm_job_id"], "sources": current["sha256"]}))


if __name__ == "__main__":
    main()
