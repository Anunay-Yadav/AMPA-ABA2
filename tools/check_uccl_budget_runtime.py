#!/usr/bin/env python3
"""Allocated, bounded checks of budgets and one-collector/two-child capture."""
from __future__ import annotations

import argparse
import ast
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import sqlite3
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / 'runs/uccl-ep-owned-build-20261101T1713Z'


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def barrier(out, phase, index, deadline):
    (out / f'{phase}-{index}').write_text(str(os.getpid()) + '\n')
    while not all((out / f'{phase}-{other}').is_file() for other in (0, 1)):
        require(time.monotonic() < deadline, f'child barrier timeout: {phase}')
        time.sleep(0.02)


def child(out, index):
    import torch

    deadline = time.monotonic() + 60
    torch.cuda.set_device(0)
    a = torch.ones((1024, 1024), device='cuda', dtype=torch.bfloat16)
    b, c = torch.full_like(a, 0.015625), torch.empty_like(a)
    for _ in range(4):
        torch.mm(a, b, out=c)
    torch.cuda.synchronize()
    barrier(out, 'ready', index, deadline)
    torch.cuda.check_error(torch.cuda.cudart().cudaProfilerStart())
    barrier(out, 'started', index, deadline)
    torch.cuda.nvtx.range_push(f'QA_CHILD_{index}')
    started = time.monotonic()
    iterations = 0
    while time.monotonic() - started < 1.0:
        for _ in range(8):
            torch.mm(a, b, out=c)
        torch.cuda.synchronize()
        iterations += 8
    torch.cuda.nvtx.range_push(f'QA_CHILD_{index}_TAIL')
    torch.mm(a, b, out=c)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    barrier(out, 'done', index, deadline)
    torch.cuda.check_error(torch.cuda.cudart().cudaProfilerStop())
    write(out / f'child-{index}.json', {'child': index, 'pid': os.getpid(), 'cuda_device': 0,
        'iterations': iterations, 'gpu_work_complete_before_done_barrier': True,
        'stopped_after_both_children_done': True, 'torch': torch.__version__})


def children_parent(out):
    require('torch' not in sys.modules, 'the recorded launcher parent must remain standard-library-only')
    require(not (out / 'parent.json').exists(), 'parent role proof already exists')
    write(out / 'parent.json', {'pid': os.getpid(), 'role': 'non_cuda_launcher', 'torch_imported': False,
                              'source_sha256': sha(__file__)})
    children = [subprocess.Popen([sys.executable, __file__, '--child', str(i), '--output-dir', str(out)]) for i in (0, 1)]
    deadline = time.monotonic() + 75
    try:
        for proc in children:
            require(proc.wait(timeout=max(1, deadline - time.monotonic())) == 0, 'a CUDA child failed')
    finally:
        for proc in children:
            if proc.poll() is None:
                proc.kill()
                proc.wait()


EXPECTED_PARENT_WARNINGS = {
    'No NVTX events collected. Does the process use NVTX?',
    'No CUDA events collected. Does the process use CUDA?',
    'CUDA profiling might have not been started correctly.',
    'Not all NVTX events might have been collected.',
}


def expected_parent_warning(diagnostic, parent_pid, cuda_owners):
    """Only exact empty/control warnings from the proved non-CUDA parent."""
    pid = diagnostic.get('global_pid')
    vm_ids = {owner >> 48 for owner in cuda_owners}
    return (diagnostic.get('severity') == 2 and type(pid) is int and pid > 0
            and len(vm_ids) == 1 and pid == (next(iter(vm_ids)) << 48) + (parent_pid << 24)
            and pid not in cuda_owners
            and diagnostic.get('text') in EXPECTED_PARENT_WARNINGS)


def generic_possible_loss_advisory(diagnostic, checked_cuda_owners):
    """Classify exact advisories only after all claimed ROI events are proved."""
    return (type(diagnostic.get('severity')) is int and diagnostic['severity'] == 2
            and type(diagnostic.get('global_pid')) is int
            and diagnostic['global_pid'] in checked_cuda_owners
            and diagnostic.get('text') in {
                'Not all NVTX events might have been collected.',
                'Not all CUDA events might have been collected.',
            })


def inspect_math_footprint(db, strings, owner, main, tail, iterations):
    """Fail closed outside this fixed-shape, observed one-kernel GEMM footprint."""
    require(type(iterations) is int and iterations > 0 and iterations % 8 == 0, 'bad recorded main matrix-operation count')
    kernels = db.execute('SELECT start,end,correlationId,demangledName FROM CUPTI_ACTIVITY_KIND_KERNEL '
                         'WHERE globalPid=?', (owner,)).fetchall()
    inside = lambda row, scope: scope['start'] <= row[0] < row[1] <= scope['end']
    tails = [row for row in kernels if inside(row, tail)]
    require(len(tails) == 1, 'the fixed QA tail must have exactly one GEMM kernel')
    name = strings.get(tails[0][3], '')
    require(re.search(r'gemm|nvjet', name, re.I), 'unreviewed QA matrix kernel name')
    require(len(kernels) == iterations + 1 and all(inside(row, main) and strings.get(row[3]) == name for row in kernels),
            'QA main/tail matrix-operation kernel footprint differs')
    kernel_ids = [row[2] for row in kernels]
    require(len(set(kernel_ids)) == len(kernel_ids), 'duplicate QA kernel correlation IDs')
    launches = []
    tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    for table in ('CUPTI_ACTIVITY_KIND_RUNTIME', 'CUPTI_ACTIVITY_KIND_DRIVER'):
        if table not in tables:
            continue
        fields = {row[1] for row in db.execute(f'PRAGMA table_info({table})')}
        require({'start', 'end', 'globalTid', 'correlationId', 'nameId', 'returnValue'} <= fields,
                'launch API schema lacks exact correlation/success fields')
        for row in db.execute(f'SELECT start,end,globalTid,correlationId,nameId,returnValue FROM {table} '
                              'WHERE globalTid>=? AND globalTid<?', (owner, owner + (1 << 24))):
            if re.search(r'LaunchKernel|LaunchCooperativeKernel', strings.get(row[4], '')):
                launches.append(row)
    require(len(launches) == iterations + 1, 'matrix-operation launch count differs')
    require(all(row[2] == main['global_tid'] and row[5] == 0 and inside(row, main) for row in launches),
            'QA matrix launch has wrong thread, failed return, or lies outside its scope')
    launch_ids = [row[3] for row in launches]
    require(len(set(launch_ids)) == len(launch_ids) and set(launch_ids) == set(kernel_ids),
            'QA launch/kernel correlation is not one-to-one')
    require(sum(inside(row, tail) for row in launches) == 1, 'QA tail must contain exactly one successful launch')
    return {'main_matrix_operations': iterations, 'tail_matrix_operations': 1,
            'recorded_kernels': len(kernels), 'recorded_launches': len(launches),
            'unique_launch_kernel_correlations': len(kernel_ids), 'kernel_name': name,
            'one_kernel_per_operation_footprint': True,
            'claim_limit': 'Fixed 1024x1024 BF16 QA footprint only; multi-kernel implementations fail closed.'}


def inspect_capture(database, out, *, expected_source_sha256=None):
    """Check known GEMM events; callers must verify any archived source override."""
    expected_source_sha256 = sha(__file__) if expected_source_sha256 is None else expected_source_sha256
    require(type(expected_source_sha256) is str and re.fullmatch(r'[0-9a-f]{64}', expected_source_sha256),
            'invalid expected acquisition source hash')
    with sqlite3.connect(f'file:{database}?mode=ro', uri=True) as db:
        tables = {x[0] for x in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        require({'NVTX_EVENTS', 'CUPTI_ACTIVITY_KIND_KERNEL'} <= tables, 'Nsight export lacks NVTX or kernel events')
        nvcols = {x[1] for x in db.execute('PRAGMA table_info(NVTX_EVENTS)')}
        strings = dict(db.execute('SELECT id,value FROM StringIds')) if 'StringIds' in tables else {}
        text_fields = [x for x in ('text', 'textId') if x in nvcols]
        require(text_fields and 'globalTid' in nvcols, 'unsupported NVTX label/owner schema')
        columns = ','.join(['start', 'end', 'globalTid', *text_fields])
        labels = {}
        for row in db.execute(f'SELECT {columns} FROM NVTX_EVENTS'):
            values = dict(zip(['start', 'end', 'globalTid', *text_fields], row))
            label = values.get('text') or strings.get(values.get('textId'))
            if label in ('QA_CHILD_0', 'QA_CHILD_1', 'QA_CHILD_0_TAIL', 'QA_CHILD_1_TAIL'):
                labels.setdefault(label, []).append(values)
        kernel_columns = {x[1] for x in db.execute('PRAGMA table_info(CUPTI_ACTIVITY_KIND_KERNEL)')}
        require('globalPid' in kernel_columns, 'kernel process ownership unavailable')
        context_field = 'contextId' if 'contextId' in kernel_columns else None
        inspected, owners = [], set()
        parent = json.loads((out / 'parent.json').read_text())
        require(type(parent.get('pid')) is int and parent['pid'] > 0
                and parent.get('role') == 'non_cuda_launcher' and parent.get('torch_imported') is False
                and parent.get('source_sha256') == expected_source_sha256, 'unproved non-CUDA parent role')
        for index in (0, 1):
            child_record = json.loads((out / f'child-{index}.json').read_text())
            require(child_record['child'] == index and child_record['stopped_after_both_children_done'], 'child completion proof missing')
            per_child = {'child': index, 'os_pid': child_record['pid'], 'ranges': []}
            for label in (f'QA_CHILD_{index}', f'QA_CHILD_{index}_TAIL'):
                rows = labels.get(label, [])
                require(len(rows) == 1, f'expected one NVTX range {label}; found {len(rows)}')
                row = rows[0]
                require(isinstance(row['start'], int) and isinstance(row['end'], int) and row['end'] > row['start'],
                        f'unclosed/empty NVTX range {label}')
                group = 'globalPid' + (',contextId' if context_field else '')
                matches = db.execute(f'SELECT {group},COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL '
                    f'WHERE start>=? AND end<=? GROUP BY {group}', (row['start'], row['end'])).fetchall()
                matching = [x for x in matches if 0 <= row['globalTid'] - x[0] < (1 << 24) and x[-1] > 0]
                require(len(matching) == 1, f'no unique child GPU kernel owner inside {label}: {matching}')
                owner = matching[0]
                per_child['ranges'].append({'label': label, 'start': row['start'], 'end': row['end'],
                    'global_tid': row['globalTid'], 'global_pid': owner[0],
                    'context_id': owner[1] if context_field else None, 'kernel_count': owner[-1]})
            require(len({x['global_pid'] for x in per_child['ranges']}) == 1, 'child main/tail owners differ')
            require(((per_child['ranges'][0]['global_pid'] >> 24) & ((1 << 24) - 1)) == child_record['pid']
                    and child_record['pid'] != parent['pid'], 'CUDA owner differs from the recorded child PID')
            per_child['math_footprint'] = inspect_math_footprint(db, strings, per_child['ranges'][0]['global_pid'],
                per_child['ranges'][0], per_child['ranges'][1], child_record['iterations'])
            owners.add(per_child['ranges'][0]['global_pid'])
            inspected.append(per_child)
        require(len(owners) == 2, 'one collector did not capture two distinct CUDA child processes')
        observed_owners = {row[0] for row in db.execute('SELECT DISTINCT globalPid FROM CUPTI_ACTIVITY_KIND_KERNEL')}
        require(observed_owners == owners and all(((pid >> 24) & ((1 << 24) - 1)) != parent['pid'] for pid in owners),
                'unexpected CUDA process or non-CUDA parent executed kernels')
        diagnostics = []
        if 'DIAGNOSTIC_EVENT' in tables:
            diagnostics = [dict(zip(('severity', 'global_pid', 'text'), row)) for row in db.execute(
                'SELECT severity,globalPid,text FROM DIAGNOSTIC_EVENT WHERE severity IN (2,3)')]
        ignored = [d for d in diagnostics if expected_parent_warning(d, parent['pid'], owners)]
        advisories = [d for d in diagnostics if generic_possible_loss_advisory(d, owners)]
        concerns = [d for d in diagnostics if not expected_parent_warning(d, parent['pid'], owners)
                    and not generic_possible_loss_advisory(d, owners)]
        require(not concerns, f'Nsight capture has event-quality concerns: {concerns}')
        return {'valid_child_tree_capture': True, 'children': inspected, 'kernel_global_pids': sorted(owners),
                'context_id_column_available': bool(context_field), 'diagnostics': diagnostics,
                'known_non_cuda_parent': parent, 'ignored_exact_parent_control_warnings': ignored,
                'acquisition_source_sha256': expected_source_sha256,
                'generic_possible_loss_advisories': advisories,
                'whole_session_event_completeness': 'unresolved',
                'quantitative_event_scope': 'Known fixed-shape QA GEMM main operations and one tail per child, with exact successful same-thread launch/kernel correlations.',
                'claim_limit': 'No completeness claim for unenumerated CUDA/NVTX events or exposed latency.'}


def checks(args):
    out = args.output_dir
    require(os.environ.get('SLURM_JOB_PARTITION') == 'debug' and os.environ.get('SLURM_JOB_NUM_NODES') == '1'
            and os.environ.get('SLURM_JOB_ID', '').isdigit(), 'checks require one allocated debug node')
    deadline, commands = time.monotonic() + 240, []
    result = {'schema_version': 1, 'valid_runtime_checks': False, 'job': os.environ['SLURM_JOB_ID'],
        'hostname': socket.gethostname(), 'execution_location': 'allocated_configured_container',
        'python': sys.executable, 'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'source_manifest': str(args.source_manifest), 'expected_manifest_sha256': args.source_sha256,
        'capability_root': str(args.capability_root), 'training_executed': False}

    def run(name, argv, timeout, env=None):
        available = min(timeout, max(0, deadline - time.monotonic() - 3))
        require(available > 0, 'allocated QA deadline exhausted')
        record = {'name': name, 'argv': [str(x) for x in argv], 'timeout_seconds': available}
        with (out / 'logs' / f'{name}.out').open('w') as stdout, (out / 'logs' / f'{name}.err').open('w') as stderr:
            proc = subprocess.Popen(record['argv'], cwd=ROOT, stdout=stdout, stderr=stderr,
                                    env=env, start_new_session=True)
            try:
                status = proc.wait(timeout=available)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                status = 124
            record['exit_code'] = status
        commands.append(record)
        write(out / 'commands.json', commands)
        require(status == 0, f'{name} failed with exit {status}; inspect saved stdout/stderr')

    try:
        require(sha(args.source_manifest) == args.source_sha256, 'frozen manifest digest differs')
        manifest = args.source_manifest.read_text()
        for relative in ('tools/check_uccl_budget_runtime.py', 'jobs/smoke/uccl_budget_runtime_checks.sbatch',
                         'profiling/run_pretrain_uccl_budget.py', 'tests/test_uccl_budget_entrypoint.py',
                         'jobs/training/uccl_budget_sweep.sbatch', 'jobs/training/uccl_budget_sweep_worker.sh'):
            require(str(ROOT / relative) in manifest, f'frozen manifest omits {relative}')
        run('source_before', ['sha256sum', '--check', str(args.source_manifest)], 15)
        ast.parse(Path(__file__).read_text(), filename=__file__)
        run('launcher_syntax', ['bash', 'jobs/training/uccl_budget_sweep.sbatch', '--selfcheck'], 15)
        run('qa_launcher_syntax', ['bash', '-n', 'jobs/smoke/uccl_budget_runtime_checks.sbatch'], 5)
        masked = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'PYTHONDONTWRITEBYTECODE': '1'}
        run('unit_tests', [sys.executable, '-m', 'unittest', 'discover', '-s', 'tests',
                          '-p', 'test_uccl_budget_entrypoint.py', '-v'], 45, masked)
        logs = (out / 'logs/unit_tests.out').read_text() + (out / 'logs/unit_tests.err').read_text()
        require(re.search(r'Ran 11 tests\b', logs) is not None, 'expected exactly eleven focused tests')
        result['focused_tests'] = 11

        from profiling import run_pretrain_uccl_budget as adapter
        import uccl.ep as native
        fused = adapter.runtime_adapter()
        require(Path(native.__file__).resolve() == BUILD / 'site/uccl/ep.abi3.so', 'native extension is not the pinned build')
        require(native.num_max_nvl_peers() == 4 and fused._buffer is None, 'native topology differs or Buffer already allocated')
        configs = {}
        try:
            for budget in (20, 12, 32, 40):
                configs[str(budget)] = adapter.configure_before_model(fused, budget)
                require(fused._buffer is None, 'configuration check allocated a native Buffer')
                for kind in ('dispatch', 'combine'):
                    cfg = getattr(fused.Buffer, f'get_{kind}_config')(16)
                    require(isinstance(cfg, native.Config), 'effective configuration is not the native Config')
                    row = configs[str(budget)][kind]
                    require(row['config']['num_sms'] == budget and row['nvl_buffer_size_hint'] > 0
                            and row['rdma_buffer_size_hint'] > 0, 'bad effective budget or buffer hints')
                    baseline = configs['20'][kind]['config']
                    require(all(value == baseline[key] for key, value in row['config'].items() if key != 'num_sms'),
                            'a nonbudget configuration field changed')
        finally:
            fused.Buffer.set_num_sms(20)
        require(fused.Buffer.num_sms == 20 and fused._buffer is None, 'configuration state was not restored')
        result['native_configs'] = configs
        result['native_identity'] = {'extension': native.__file__, 'sha256': sha(native.__file__),
            'fused_module': fused.__file__, 'buffer_module': sys.modules[fused.Buffer.__module__].__file__,
            'num_max_nvl_peers': 4, 'native_buffer_allocated': False, 'restored_budget': 20}
        write(out / 'native_configs.json', result['native_configs'])

        proof = json.loads((args.capability_root / 'capabilities.json').read_text())
        require(proof.get('native_report_generated') is True and proof.get('cuda_profiler_api_capture_requested') is True,
                'native CUDAProfilerAPI capability proof is required')
        native_command = next(x for x in json.loads((args.capability_root / 'commands.json').read_text())
                              if x['name'] == 'capture_native')['argv']
        index = next(i for i, x in enumerate(native_command) if x.startswith('--output='))
        prefix = native_command[:index]
        require('--capture-range=cudaProfilerApi' in prefix and '--capture-range-end=stop' in prefix,
                'capability command does not prove the requested capture mode')
        children = out / 'children'
        children.mkdir()
        report, database = out / 'two_children.nsys-rep', out / 'two_children.sqlite'
        run('child_capture', [*prefix, f'--output={out / "two_children"}', sys.executable,
                             __file__, '--children-parent', '--output-dir', str(children)], 90)
        require(report.is_file() and report.stat().st_size > 0, 'native child-tree report missing')
        run('child_export', [proof['nsys_path'], 'export', '--type=sqlite', f'--output={database}', str(report)], 35)
        result['child_capture'] = inspect_capture(database, children)
        result['report'] = {'path': str(report), 'sha256': sha(report), 'bytes': report.stat().st_size}
        require(sha(args.source_manifest) == args.source_sha256, 'source manifest changed during QA')
        run('source_after', ['sha256sum', '--check', str(args.source_manifest)], 15)
        result['valid_runtime_checks'] = True
        result['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        write(out / 'checks.json', result)
        (out / 'PASS.txt').write_text(f"runtime_checks_passed=true\njob={os.environ['SLURM_JOB_ID']}\n"
            f"source_sha256={args.source_sha256}\nchecks_sha256={sha(out / 'checks.json')}\n")
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
        write(out / 'checks.json', result)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--capability-root', type=Path)
    parser.add_argument('--source-manifest', type=Path)
    parser.add_argument('--source-sha256')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--child', type=int, choices=(0, 1))
    mode.add_argument('--children-parent', action='store_true')
    args = parser.parse_args()
    if args.child is not None:
        child(args.output_dir, args.child)
    elif args.children_parent:
        children_parent(args.output_dir)
    else:
        checks(args)


if __name__ == '__main__':
    main()
