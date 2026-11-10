"""Read four accepted representative traces; do not measure new GPU counters."""
import csv
from collections import Counter, defaultdict
import gc
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import socket
import sys

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
REPORT = ROOT / 'runs/qwen-debug-results-20261105T0936Z-report/report.json'
CAMPAIGN = ROOT / 'runs/qwen-debug-results-20261105T0925Z-cpu-validation/campaign_acceptance.json'
CASES = ('ordinary', 'grouped', 'uccl_ordinary', 'uccl')
RANK = 0
STEP = 5


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(path.read_text())


def relative(path):
    return str(path.relative_to(ROOT))


def binding(proof, path):
    return next(row['sha256'] for row in proof['input_files'] if row['path'] == str(path))


def family(event):
    name = event['name']
    args = event.get('args', {})
    match = re.search(r'uccl::internode::(dispatch|combine|notify_dispatch|cached_notify)<', name)
    if match:
        return 'uccl_' + match[1]
    if 'uccl::layout::get_dispatch_layout<' in name:
        return 'uccl_layout'
    if 'nccl' in name.lower():
        collective = re.sub('[^a-z]', '', str(args.get('Collective name', '')).lower())
        if args.get('Process Group Description') == 'EXPERT_MODEL_PARALLEL_GROUP' and collective in ('alltoall', 'alltoallv'):
            return 'native_ep_alltoall'
        return 'other_nccl'
    return None


def main():
    report, campaign = read(REPORT), read(CAMPAIGN)
    assert campaign['valid_debug_campaign'] is True
    results, rows = [], []
    for case in CASES:
        leg = report['cases'][case]['legs']['profile']
        proof_record = campaign['case_acceptance'][case]['profile']
        proof_path = Path(proof_record['path'])
        assert sha(proof_path) == proof_record['sha256']
        proof = read(proof_path)
        assert proof['valid_debug_case'] is True
        run = Path(leg['run_dir'])
        assert campaign['case_runs'][case]['profile'] == str(run)
        path = run / f'traces/torch_profile/step-{STEP:05d}/rank-{RANK}.json.gz'
        expected = binding(proof, path)
        assert sha(path) == expected
        with gzip.open(path, 'rt') as stream:
            trace = json.load(stream)
        props = trace['deviceProperties']
        kernels = [event for event in trace['traceEvents'] if event.get('cat') == 'kernel' and event.get('dur', 0) > 0]
        assert kernels
        all_keys, occupancy, groups = Counter(), Counter(), defaultdict(list)
        for event in kernels:
            args = event.get('args', {})
            all_keys.update(args.keys())
            occupancy[str(args.get('est. achieved occupancy %', 'missing'))] += 1
            kind = family(event)
            if not kind:
                continue
            device = next(p for p in props if p['id'] == args['device'])
            sms = device['numSms']
            warp = device['warpSize']
            max_warps = device['maxThreadsPerMultiprocessor'] / warp
            grid, block = math.prod(args['grid']), math.prod(args['block'])
            assert math.isclose(args['blocks per SM'], grid / sms, abs_tol=1e-5)
            assert math.isclose(args['warps per SM'], grid / sms * math.ceil(block / warp), abs_tol=1e-5)
            key = (kind, event['name'], tuple(args['grid']), tuple(args['block']),
                   args['registers per thread'], args['shared memory'],
                   args['blocks per SM'], args['warps per SM'],
                   args['est. achieved occupancy %'], sms, warp, max_warps,
                   args.get('Process Group Description', ''), args.get('Collective name', ''))
            groups[key].append(event['dur'])
        for key, durations in groups.items():
            kind, name, grid_shape, block_shape, regs, shared, blocks_sm, warps_sm, estimate, sms, warp, max_warps, pg, collective = key
            grid, block = math.prod(grid_shape), math.prod(block_shape)
            rows.append({'case': case, 'rank': RANK, 'human_update': STEP + 1, 'family': kind,
                         'kernel': name, 'process_group': pg, 'collective': collective,
                         'count': len(durations), 'gpu_duration_sum_ms': sum(durations) / 1000,
                         'gpu_duration_mean_us': sum(durations) / len(durations),
                         'grid': str(grid_shape), 'block': str(block_shape),
                         'grid_blocks': grid, 'threads_per_block': block,
                         'launched_warps_per_block': math.ceil(block / warp),
                         'registers_per_thread': regs, 'shared_memory_bytes': shared,
                         'trace_blocks_per_sm_launch_average': blocks_sm,
                         'trace_warps_per_sm_launch_average': warps_sm,
                         'trace_static_estimated_occupancy_percent': estimate,
                         'gpu_sms': sms, 'max_warps_per_sm': max_warps,
                         'individual_kernel_concurrent_sm_upper_bound': min(grid, sms),
                         'sm_upper_bound_percent': 100 * min(grid, sms) / sms,
                         'grid_launched_warps_over_gpu_slot_capacity_percent': 100 * grid * math.ceil(block / warp) / (sms * max_warps)})
        results.append({'case': case, 'rank': RANK, 'human_update': STEP + 1,
                        'trace': relative(path), 'trace_sha256': expected,
                        'acceptance': relative(proof_path), 'acceptance_sha256': proof_record['sha256'],
                        'device_properties': props, 'positive_kernel_events': len(kernels),
                        'kernel_arg_key_counts': dict(all_keys), 'estimated_occupancy_histogram': dict(occupancy),
                        'counter_event_count': sum(event.get('ph') == 'C' for event in trace['traceEvents']),
                        'selected_kernel_launches': sum(len(values) for values in groups.values())})
        print(json.dumps({'case': case, 'kernels': len(kernels), 'selected_shapes': len(groups),
                          'zero_estimates': occupancy.get('0', 0),
                          'nonzero_estimates': len(kernels) - occupancy.get('0', 0)}), flush=True)
        del trace, kernels, groups
        gc.collect()
    runtime_records = []
    for case in ('uccl_ordinary', 'uccl'):
        for kind in ('control', 'profile'):
            run = Path(report['cases'][case]['legs'][kind]['run_dir'])
            for rank in range(16):
                path = run / f'raw/uccl_backend_rank-{rank}_runtime.json'
                record = read(path)
                assert record['rank'] == rank and record['num_sms'] == 20
                assert record['valid_identity'] is True and record['low_latency_mode'] is False
                runtime_records.append({'case': case, 'leg': kind, 'rank': rank,
                                        'path': relative(path), 'sha256': sha(path), 'num_sms': record['num_sms']})
    payload = {'kind': 'accepted_trace_launch_metadata_NOT_hardware_occupancy',
               'hostname': socket.gethostname(), 'python': sys.executable, 'gpu_execution': False,
               'scope': 'Four representative profiler-ON traces: rank0, human update6. Not all320 measured rank-update captures.',
               'extraction_source': {'path': relative(Path(__file__).resolve()), 'sha256': sha(Path(__file__).resolve())},
               'report': {'path': relative(REPORT), 'sha256': sha(REPORT)},
               'campaign': {'path': relative(CAMPAIGN), 'sha256': sha(CAMPAIGN)},
               'traces': results, 'communication_kernel_geometry': rows,
               'uccl_runtime_records': runtime_records,
               'limits': ['Grid/block/resource fields describe launches, not achieved active SMs or useful work.',
                          'Main UCCL payload and NCCL kernels have zero estimated occupancy; other kernels can have nonzero static estimates. Neither is measured achieved occupancy.',
                          'The SM bound applies to one kernel, not simultaneous kernels or all communication.',
                          'Warp-slot ratios assume all launched warps resident together, ignore exits and are not achieved occupancy.',
                          'Kernel durations include polling/waiting and may overlap. Do not sum them into a wall-time partition.']}
    (OUT / 'launch_metrics.json').write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    with (OUT / 'communication_kernel_geometry.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({'complete': True, 'traces': len(results), 'geometry_rows': len(rows), 'uccl_runtime_records': len(runtime_records)}), flush=True)


if __name__ == '__main__':
    main()
