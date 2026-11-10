#!/usr/bin/env python3
"""Qualify a saved known-workload capture without changing its failed QA record."""
from __future__ import annotations

import argparse
import ast
import datetime
import hashlib
import importlib.util
import json
import re
import socket
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / 'tools/check_uccl_budget_runtime.py'
OLD_HELPER_SHA = 'b8bfa864760bc505f38a7e740a4dc194afa8cc966c48ec98a45584c2448dca70'
NATIVE_SHA = '110d27746db1c326c0e6216f413c5c45cf7d26eca83037b3874241c73abb8d36'
ALLOWED = {HELPER, ROOT / 'tools/uccl_budget_sweep_posthoc.py', ROOT / 'analysis/uccl_budget_sweep_delivery.py'}


def require(value, message):
    if not value:
        raise ValueError(message)


class Inputs:
    def __init__(self):
        self.hashes = {}

    def bind(self, path, expected=None):
        path = Path(path)
        require(path.is_absolute() and path.is_file() and not path.is_symlink(), f'expected regular absolute input: {path}')
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        key = str(path)
        require(key not in self.hashes or self.hashes[key] == digest, f'input changed: {path}')
        require(expected is None or digest == expected, f'input digest differs: {path}')
        self.hashes[key] = digest
        return digest

    def read(self, path):
        digest = self.bind(path)
        data = Path(path).read_bytes()
        require(hashlib.sha256(data).hexdigest() == digest, f'JSON changed while reading: {path}')
        return json.loads(data)

    def recheck(self):
        for path, digest in list(self.hashes.items()):
            self.bind(Path(path), digest)


def manifest(path, inputs, expected=None):
    inputs.bind(path, expected)
    result = {}
    for line in path.read_text().splitlines():
        match = re.fullmatch(r'([0-9a-f]{64})  (.+)', line)
        require(match is not None, f'invalid source inventory row: {path}')
        digest, name = match.groups()
        item = Path(name)
        require(item.is_absolute() and item.is_relative_to(ROOT) and item not in result, 'source paths must be unique project files')
        result[item] = digest
    return result


def definitions(path):
    tree = ast.parse(path.read_text(), filename=str(path))
    return {node.name: ast.dump(node, include_attributes=False) for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}, {
        target.id: ast.dump(node.value, include_attributes=False) for node in tree.body if isinstance(node, ast.Assign)
        for target in node.targets if isinstance(target, ast.Name)}


def qualify(args, inputs):
    qa, archive = args.captured_qa_root, args.captured_helper_source
    checks = inputs.read(qa / 'checks.json')
    require(checks.get('valid_runtime_checks') is False and checks.get('training_executed') is False
            and checks.get('focused_tests') == 11, 'saved QA must remain failed after eleven successful tests without training')
    job = checks.get('job')
    inputs.bind(qa / 'job-id')
    require(type(job) is str and job.isdigit() and (qa / 'job-id').read_text().strip() == job, 'captured QA job identities differ')
    inputs.bind(qa / 'exit.txt')
    exit_record = dict(line.split('=', 1) for line in (qa / 'exit.txt').read_text().splitlines() if '=' in line)
    require(exit_record.get('exit_code') == '1' and exit_record.get('job') == job and not (qa / 'PASS.txt').exists(),
            'original QA must retain its failed exit and absent PASS marker')
    prefix = 'RuntimeError: Nsight capture has event-quality concerns: '
    error = checks.get('error', '')
    require(error.startswith(prefix), 'original QA failed for a reason other than capture diagnostics')
    old_concerns = ast.literal_eval(error[len(prefix):])
    require(isinstance(old_concerns, list) and old_concerns, 'original diagnostic failure is empty or malformed')

    old_manifest = qa / 'source.sha256'
    require(checks.get('source_manifest') == str(old_manifest), 'captured source inventory path differs')
    old = manifest(old_manifest, inputs, checks.get('expected_manifest_sha256'))
    require(len(old) == 73 and old.get(HELPER) == OLD_HELPER_SHA, 'unsupported acquisition source inventory/helper')
    require(qa.name.startswith('uccl-budget4-qa-'), 'unsupported captured QA directory naming')
    root_manifest = ROOT / 'runs' / qa.name.replace('uccl-budget4-qa-', 'uccl-budget4-source-', 1) / 'expected-source.sha256'
    require(manifest(root_manifest, inputs, checks['expected_manifest_sha256']) == old, 'root and captured source inventories differ')
    inputs.bind(archive, OLD_HELPER_SHA)
    archived_inventory = archive.parents[2] / 'acquired-source.sha256'
    require(manifest(archived_inventory, inputs).get(archive) == OLD_HELPER_SHA, 'archived helper inventory differs')
    current = manifest(args.source_manifest, inputs, args.source_sha256)
    require(len(current) == 74 and set(current) == set(old) | {Path(__file__).resolve()}, 'new freeze must retain all 73 prior paths and add this qualifier')
    changed = {path for path in old if old[path] != current[path]}
    require(changed <= ALLOWED, f'a runtime/training source changed: {sorted(map(str, changed - ALLOWED))}')
    for path, digest in current.items():
        inputs.bind(path, digest)
    old_defs, old_constants = definitions(archive)
    new_defs, new_constants = definitions(HELPER)
    unchanged = ('require', 'write', 'sha', 'barrier', 'child', 'children_parent', 'checks',
                 'inspect_math_footprint', 'expected_parent_warning')
    require(all(old_defs.get(name) == new_defs.get(name) is not None for name in unchanged), 'captured GPU program or strict ROI proof changed')
    require(old_constants['EXPECTED_PARENT_WARNINGS'] == new_constants['EXPECTED_PARENT_WARNINGS'], 'parent warning policy changed')

    commands = inputs.read(qa / 'commands.json')
    names = ['source_before', 'launcher_syntax', 'qa_launcher_syntax', 'unit_tests', 'child_capture', 'child_export']
    require([row.get('name') for row in commands] == names and all(row.get('exit_code') == 0 for row in commands),
            'original source/test/capture/export command sequence did not succeed')
    for name in names + ['checks']:
        for suffix in ('out', 'err'):
            inputs.bind(qa / 'logs' / f'{name}.{suffix}')
    require(commands[0]['argv'] == ['sha256sum', '--check', str(old_manifest)], 'source-before command differs')
    require(re.search(r'Ran 11 tests\b', (qa / 'logs/unit_tests.err').read_text()) is not None, 'eleven-test result is absent')
    configs = inputs.read(qa / 'native_configs.json')
    require(configs == checks.get('native_configs') and set(configs) == {'20', '12', '32', '40'}, 'native configuration records differ')
    for budget, directions in configs.items():
        require(set(directions) == {'dispatch', 'combine'}, 'native configuration direction missing')
        for direction, row in directions.items():
            expected = {'num_sms': int(budget), 'num_max_nvl_chunked_recv_tokens': 288,
                'num_max_nvl_chunked_send_tokens': 36 if direction == 'dispatch' else 4,
                'num_max_rdma_chunked_send_tokens': 20 if direction == 'dispatch' else 12,
                'num_max_rdma_chunked_recv_tokens': 140 if direction == 'dispatch' else 132}
            require(row.get('config') == expected and row.get('hidden_bytes') == 4096 and row.get('ep_ranks') == 16
                    and all(type(row.get(k)) is int and row[k] > 0 for k in ('nvl_buffer_size_hint', 'rdma_buffer_size_hint')),
                    'native budget/nonbudget fields or positive hints differ')
    identity = checks['native_identity']
    native = ROOT / 'runs/uccl-ep-owned-build-20261101T1713Z/site/uccl/ep.abi3.so'
    require(identity.get('extension') == str(native) and identity.get('sha256') == NATIVE_SHA
            and identity.get('native_buffer_allocated') is False and identity.get('num_max_nvl_peers') == 4
            and identity.get('restored_budget') == 20
            and identity.get('fused_module') == str(ROOT / 'Megatron-LM/megatron/core/transformer/moe/fused_a2a.py')
            and identity.get('buffer_module') == str(ROOT / 'runs/uccl-ep-owned-build-20261101T1713Z/uccl/ep/deep_ep_wrapper/deep_ep/buffer.py'),
            'captured native identity differs')
    inputs.bind(native, NATIVE_SHA)

    cap = Path(checks['capability_root'])
    capability, provenance = inputs.read(cap / 'capabilities.json'), inputs.read(cap / 'provenance.json')
    require(capability.get('native_report_generated') is True and capability.get('native_capture_exit_code') == 0
            and capability.get('training_executed') is False, 'native capability proof is invalid')
    tool = Path(capability['nsys_path'])
    require(provenance['nsys']['path'] == str(tool), 'capability tool paths differ')
    inputs.bind(tool, provenance['nsys']['sha256'])
    cap_commands = inputs.read(cap / 'commands.json')
    inputs.bind(cap / 'exit.txt')
    inputs.bind(cap / 'job-id')
    inputs.bind(cap / 'logs/nsys_version.out')
    require('exit_code=0' in (cap / 'exit.txt').read_text().splitlines()
            and (cap / 'job-id').read_text().strip() == capability.get('job_id'), 'capability job/exit records differ')
    require('2025.6.3.343' in (cap / 'logs/nsys_version.out').read_text(), 'saved tool version is not the matched version')
    for name in ('native_report', 'gpu_metrics_report'):
        if name in capability:
            inputs.bind(Path(capability[name]['path']), capability[name]['sha256'])
    require(commands[-2]['argv'][0] == commands[-1]['argv'][0] == str(tool), 'QA and capability tools differ')
    original_capture = next(row for row in cap_commands if row['name'] == 'capture_native')['argv']
    output_index = next(index for index, value in enumerate(original_capture) if value.startswith('--output='))
    require(commands[-2]['argv'] == [*original_capture[:output_index], f'--output={qa / "two_children"}',
            checks['python'], str(HELPER), '--children-parent', '--output-dir', str(qa / 'children')],
            'saved capture command does not name the bound two-child GPU program')
    require(commands[-1]['argv'][1:] == ['export', '--type=sqlite', f'--output={qa / "two_children.sqlite"}', str(qa / 'two_children.nsys-rep')],
            'saved SQLite export is not bound to the captured native report')
    inputs.bind(qa / 'two_children.nsys-rep')
    inputs.bind(qa / 'two_children.sqlite')
    for name in ('parent.json', 'child-0.json', 'child-1.json', 'ready-0', 'ready-1', 'started-0', 'started-1', 'done-0', 'done-1'):
        inputs.bind(qa / 'children' / name)
    require('torch' not in sys.modules, 'qualification must remain CPU/std-library-only')
    spec = importlib.util.spec_from_file_location('ampa_scoped_qa_policy', HELPER)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    require('torch' not in sys.modules, 'QA policy import loaded torch')
    proof = helper.inspect_capture(qa / 'two_children.sqlite', qa / 'children', expected_source_sha256=OLD_HELPER_SHA)
    require(proof.get('valid_child_tree_capture') is True and len(proof.get('children', [])) == 2
            and proof.get('whole_session_event_completeness') == 'unresolved', 'known ROI capture proof did not pass')
    require(old_concerns == proof.get('generic_possible_loss_advisories'), 'original failure includes diagnostics outside the scoped advisory policy')
    return {'valid_scoped_runtime_checks': True, 'valid_known_roi_coverage': True,
        'original_strict_status': 'failed', 'original_capture_job': job, 'original_strict_error': error,
        'whole_session_event_completeness': 'unresolved', 'known_roi_proof': proof,
        'advisories_retained': proof['generic_possible_loss_advisories'], 'native_configs': configs,
        'native_identity': identity, 'unchanged_acquired_functions': list(unchanged),
        'changed_validation_sources': sorted(map(str, changed)), 'archived_helper_sha256': OLD_HELPER_SHA}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--captured-qa-root', type=Path, required=True)
    parser.add_argument('--captured-helper-source', type=Path, required=True)
    parser.add_argument('--source-manifest', type=Path, required=True)
    parser.add_argument('--source-sha256', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    require(all(path.is_absolute() for path in (args.captured_qa_root, args.captured_helper_source, args.source_manifest, args.output_dir)), 'all CLI paths must be absolute')
    require(args.output_dir.parent == ROOT / 'runs' and not args.output_dir.exists() and not args.output_dir.is_symlink(), 'output must be a new direct project runs child')
    require(re.fullmatch(r'[0-9a-f]{64}', args.source_sha256), 'invalid new source manifest SHA')
    require(Path(sys.prefix).resolve() == (ROOT / '.runtime/venvs/qwen-posthoc-cpu').resolve(), 'use the authorized isolated CPU-only venv')
    args.output_dir.mkdir()
    inputs = Inputs()
    record = {'valid_scoped_runtime_checks': False, 'valid_known_roi_coverage': False,
        'execution_location': 'authorized_login_CPU_saved_capture_revalidation', 'hostname': socket.gethostname(),
        'python': sys.executable, 'python_version': sys.version, 'new_source_manifest': str(args.source_manifest),
        'new_manifest_sha256': args.source_sha256, 'training_executed': False,
        'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    try:
        record.update(qualify(args, inputs))
    except Exception as error:
        record.update(valid_scoped_runtime_checks=False, valid_known_roi_coverage=False, error=f'{type(error).__name__}: {error}')
    try:
        inputs.recheck()
    except Exception as error:
        record.update(valid_scoped_runtime_checks=False, valid_known_roi_coverage=False, recheck_error=f'{type(error).__name__}: {error}')
    record['input_files'] = [{'path': path, 'sha256': digest} for path, digest in sorted(inputs.hashes.items())]
    (args.output_dir / 'qualification.json').write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + '\n')
    (args.output_dir / 'README.md').write_text('# Saved capture qualification\n\n'
        f'Known-workload qualification passed: {record["valid_scoped_runtime_checks"]}. '
        'The original QA job and its failed strict result remain unchanged.\n\n'
        'This separate proof covers the fixed two-child GEMM workload and its exact launch/kernel correlations. '
        'Whole-session CUDA/NVTX completeness remains unresolved. Generic advisories remain in qualification.json. '
        'It does not accept any Qwen training run or assert observer-free timing.\n')
    print(json.dumps({'valid_scoped_runtime_checks': record['valid_scoped_runtime_checks'], 'output_dir': str(args.output_dir)}))
    raise SystemExit(0 if record['valid_scoped_runtime_checks'] else 1)


if __name__ == '__main__':
    main()
