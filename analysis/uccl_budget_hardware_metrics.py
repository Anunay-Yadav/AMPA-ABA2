#!/usr/bin/env python3
"""Bound percentage samples to accepted UCCL intervals without exclusive attribution."""
from __future__ import annotations

import argparse
from array import array
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
import csv
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import sqlite3
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
NAMES = ('SMs Active [Throughput %]', 'SM Issue [Throughput %]', 'Compute Warps in Flight [Throughput %]')
VIEWS = ('optimizer_step_window', 'during_any_payload', 'during_dispatch', 'during_combine')
STEPS = tuple(range(5, 10))
VM_MASK = 0xFFFF000000000000
PID_MASK = (1 << 24) - 1


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class Inputs:
    def __init__(self):
        self.hashes = {}

    def bind(self, path, expected=None):
        path = Path(path)
        require(path.is_absolute() and path.is_file() and not path.is_symlink(), f'bad regular input: {path}')
        key = str(path)
        if key not in self.hashes:
            self.hashes[key] = digest(path)
        require(expected is None or self.hashes[key] == expected, f'input SHA differs: {path}')
        return self.hashes[key]

    def read(self, path, expected=None):
        sha = self.bind(path, expected)
        data = Path(path).read_bytes()
        require(hashlib.sha256(data).hexdigest() == sha, f'JSON changed: {path}')
        return json.loads(data)

    def recheck(self):
        for path, sha in self.hashes.items():
            require(digest(path) == sha, f'consumed input changed: {path}')


def uuid_key(value):
    require(type(value) is str, 'GPU UUID unavailable')
    result = value.lower().removeprefix('gpu-').replace('-', '')
    require(re.fullmatch('[0-9a-f]{32}', result), 'malformed GPU UUID')
    return result


def union(intervals):
    result = []
    for start, end in sorted(intervals):
        require(type(start) is int and type(end) is int and end > start, 'bad interval')
        if result and start <= result[-1][1]:
            result[-1] = result[-1][0], max(end, result[-1][1])
        else:
            result.append((start, end))
    return result


def percentile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    low, high = math.floor(pos), math.ceil(pos)
    return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)


def histogram_summary(hist):
    count = sum(hist.values())
    if not count:
        return None
    def quantile(q):
        pos = (count - 1) * q
        found, total = {}, 0
        for value, frequency in sorted(hist.items()):
            for index in (math.floor(pos), math.ceil(pos)):
                if total <= index < total + frequency:
                    found[index] = value
            total += frequency
        lo, hi = math.floor(pos), math.ceil(pos)
        return found[lo] + (found[hi] - found[lo]) * (pos - lo)
    return {'count': count, 'arithmetic_sample_mean_percent': sum(k * n for k, n in hist.items()) / count,
            'min_percent': min(hist), 'median_percent': quantile(.5), 'p95_percent': quantile(.95), 'max_percent': max(hist)}


def coverage(timestamps, valid, start, end, period):
    lo, hi = bisect_left(timestamps, start), bisect_left(timestamps, end)
    points = timestamps[lo:hi]
    gaps = [b - a for a, b in zip(points, points[1:])]
    first, last = (points[0] - start, end - points[-1]) if points else (None, None)
    bad = sum(not x for x in valid[lo:hi])
    available = (len(points) >= 2 and not bad and first <= 2 * period and last <= 2 * period
                 and max(gaps, default=0) <= 2 * period)
    return {'valid_step_sample_coverage': bool(available), 'sample_bundles': len(points),
            'verified_window_start_ns': start, 'verified_window_end_ns': end,
            'invalid_selected_metric_bundles': bad, 'first_boundary_distance_ns': first,
            'last_boundary_distance_ns': last, 'spacing_min_ns': min(gaps) if gaps else None,
            'spacing_median_ns': percentile(gaps, .5), 'spacing_p99_ns': percentile(gaps, .99),
            'spacing_max_ns': max(gaps) if gaps else None, 'internal_gaps_over_two_periods': sum(x > 2 * period for x in gaps),
            'nominal_period_ns': period, 'gate': '>=2 samples; complete finite percentage bundles; boundaries/gaps<=2 periods'}, lo, hi


def summarize_view(stream, intervals, step_coverage):
    times, values, valid = stream['timestamps'], stream['values'], stream['valid']
    hist = [Counter() for _ in NAMES]
    zeros = singles = invalid = count = zero_duration = 0
    merged = union(intervals)
    bounded = all(step_coverage['verified_window_start_ns'] <= start < end <= step_coverage['verified_window_end_ns']
                  for start, end in merged)
    for start, end in merged:
        lo, hi = bisect_left(times, start), bisect_left(times, end)
        n = hi - lo
        zeros += n == 0
        singles += n == 1
        zero_duration += (end - start) if n == 0 else 0
        count += n
        for index in range(lo, hi):
            if not valid[index]:
                invalid += 1
            else:
                for metric in range(len(NAMES)):
                    hist[metric][values[metric][index]] += 1
    available = step_coverage['valid_step_sample_coverage'] and bounded and count >= 2 and invalid == 0
    return {'metrics_available': available, 'interval_count': len(merged),
            'intervals_within_verified_sample_window': bounded,
            'interval_union_duration_ns': sum(b - a for a, b in merged), 'selected_sample_bundles': count,
            'invalid_selected_metric_bundles': invalid, 'intervals_without_sample': zeros,
            'intervals_with_one_sample': singles, 'duration_in_zero_sample_intervals_ns': zero_duration,
            'metrics': {name: histogram_summary(h) if available else None for name, h in zip(NAMES, hist)},
            'unavailable_reason': None if available else 'intervals exceed covered step, step coverage failed, invalid bundle, or fewer than two selected samples',
            'selection': 'sample timestamp inside half-open interval union; no interpolation or invented sample-support duration'}, hist


def metric_target(type_id, gpu, expected_uuid=None):
    target = gpu.get((type_id & VM_MASK, type_id & 0xff))
    require(target is not None, 'metric stream has no unique VM/physical GPU target')
    require(expected_uuid is None or target['uuid'] == uuid_key(expected_uuid), 'metric/rank physical UUID differs')
    return target


def mappings(db, strings, proof, placement, node):
    gpu_rows = db.execute('SELECT vmId,id,uuid,name,smCount,maxWarpsPerSm FROM TARGET_INFO_GPU').fetchall()
    gpu = {}
    for vm, device, uuid, name, sms, warps in gpu_rows:
        require((vm, device) not in gpu, 'ambiguous VM/GPU target identity')
        gpu[vm, device] = {'global_vm': vm, 'physical_gpu_id': device, 'uuid': uuid_key(uuid),
                           'gpu_name': name, 'hardware_sm_count': sms, 'hardware_max_warps_per_sm': warps}
    metadata = list(db.execute('SELECT typeId,sourceId,metricId,metricName FROM TARGET_INFO_GPU_METRICS'))
    sources = {row[0]: row[1:] for row in db.execute('SELECT sourceId,nameId,timeSourceId FROM GENERIC_EVENT_SOURCES')}
    clock_types = dict(db.execute('SELECT id,name FROM ENUM_NSYS_GENERIC_EVENT_SOURCE'))
    fields = {(row[0], row[1]): row[2:] for row in db.execute(
        'SELECT typeId,fieldIdx,fieldNameId,size,type,hyperFormat,hyperComment FROM GENERIC_EVENT_TYPE_FIELDS')}
    types, keys = {}, {}
    for type_id, source, metric_id, name in metadata:
        if name not in NAMES:
            continue
        target = metric_target(type_id, gpu)
        require(strings.get(sources[source][0]) == 'GpuMetrics'
                and clock_types.get(sources[source][1]) == 'SessionNs',
                'metric timestamp/source definition is unverified')
        field = fields.get((type_id, metric_id))
        require(field is not None and strings.get(field[0]) == name and field[1:3] == (1, 1)
                and field[3:] == (None, None), 'percentage encoding differs from the reviewed unsigned-byte schema')
        require((type_id, metric_id) not in keys, 'duplicate compound metric definition')
        keys[type_id, metric_id] = NAMES.index(name)
        types.setdefault(type_id, {**target, 'metric_names': []})['metric_names'].append(name)
    require(len(types) == 4 and all(sorted(t['metric_names']) == sorted(NAMES) for t in types.values()), 'expected three percentage fields on four GPU streams')
    uuid_to_type = {row['uuid']: type_id for type_id, row in types.items()}
    require(len(uuid_to_type) == 4, 'metric streams do not have four distinct physical UUIDs')
    owners = {int(pid): rank for pid, rank in proof['rank_by_global_pid'].items()}
    require(len(owners) == 4, 'accepted node lacks four CUDA owners')
    cuda = db.execute('SELECT gpuId,cudaId,pid,uuid FROM TARGET_INFO_CUDA_DEVICE').fetchall()
    slots = {slot['rank']: slot for slot in placement if str(slot['slurm_node_id']) == str(node)}
    result = {}
    for owner, rank in owners.items():
        require(rank in slots, 'accepted owner/rank is outside its runtime node')
        expected_uuid = uuid_key(slots[rank]['physical_gpu_uuid'])
        type_id = uuid_to_type.get(expected_uuid)
        require(type_id is not None and type_id & VM_MASK == owner & VM_MASK, 'metric/rank VM or physical UUID differs')
        metric_target(type_id, gpu, expected_uuid)
        result[owner] = {'rank': rank, 'node_rank': node, 'type_id': type_id, **types[type_id],
                         'os_pid': (owner >> 24) & PID_MASK, 'global_pid': owner,
                         'cuda_mappings': [row for row in cuda if row[2] == (owner >> 24) & PID_MASK]}
    return result, keys


def payload_intervals(db, strings, proof, identities, budget):
    owners = set(identities)
    text_ids = {key for key, value in strings.items() if value in {f'AMPA/STEP/{step}' for step in STEPS}}
    nvcols = {row[1] for row in db.execute('PRAGMA table_info(NVTX_EVENTS)')}
    step_ranges = {}
    expression = 'COALESCE(n.text,s.value)' if 'textId' in nvcols else 'n.text'
    join = 'LEFT JOIN StringIds s ON n.textId=s.id' if 'textId' in nvcols else ''
    for start, end, tid, label in db.execute(f'SELECT n.start,n.end,n.globalTid,{expression} FROM NVTX_EVENTS n {join} WHERE {expression} LIKE \'AMPA/STEP/%\''):
        match = re.fullmatch(r'AMPA/STEP/(\d+)', label)
        owner = tid & ~PID_MASK
        require(match is not None and owner in owners and int(match[1]) in STEPS and end > start, 'bad measured update owner/range')
        key = owner, int(match[1])
        require(key not in step_ranges, 'duplicate measured update range')
        step_ranges[key] = (start, end)
    require(set(step_ranges) == {(pid, step) for pid in owners for step in STEPS}, 'incomplete four-rank/five-update mapping')
    raw = db.execute('SELECT k.start,k.end,k.globalPid,k.deviceId,k.contextId,k.correlationId,k.demangledName,k.gridX,k.gridY,k.gridZ '
        'FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.demangledName=s.id '
        "WHERE s.value LIKE '%uccl::internode::dispatch<%' OR s.value LIKE '%uccl::internode::combine<%'").fetchall()
    kernels = {}
    for start, end, owner, device, context, correlation, name_id, x, y, z in raw:
        require(owner in owners and end > start and x * y * z == budget, 'unexpected accepted payload launch')
        require(context in proof['cuda_contexts'][str(owner)] and len(proof['cuda_contexts'][str(owner)]) == 1, 'payload context changed')
        key = owner, correlation
        require(key not in kernels, 'duplicate payload correlation identity')
        identity = identities[owner]
        mapped = [(gpu_id, uuid) for gpu_id, cuda_id, pid, uuid in identity['cuda_mappings'] if cuda_id == device]
        require(len(mapped) == 1 and mapped[0][0] == identity['physical_gpu_id'], 'process-specific CUDA device does not map to metric GPU')
        if mapped[0][1]:
            require(uuid_key(mapped[0][1]) == identity['uuid'], 'CUDA and metric UUIDs differ')
        family = re.search(r'uccl::internode::(dispatch|combine)<', strings[name_id])[1]
        kernels[key] = {'start': start, 'end': end, 'global_pid': owner, 'cuda_device_id': device,
                        'context_id': context, 'correlation_id': correlation, 'family': family, 'rank': identity['rank']}
    routes = defaultdict(set)
    tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    for table in ('CUPTI_ACTIVITY_KIND_RUNTIME', 'CUPTI_ACTIVITY_KIND_DRIVER'):
        if table not in tables:
            continue
        for timestamp, tid, correlation in db.execute(f'SELECT start,globalTid,correlationId FROM {table}'):
            owner = tid & ~PID_MASK
            if (owner, correlation) not in kernels:
                continue
            for step in STEPS:
                start, end = step_ranges[owner, step]
                if start <= timestamp < end:
                    routes[owner, correlation].add(step)
    counts = Counter()
    by_step = defaultdict(list)
    for key, kernel in kernels.items():
        require(len(routes[key]) == 1, 'payload lacks one exact in-update API launch correlation')
        kernel['step'] = next(iter(routes[key]))
        counts[kernel['rank'], kernel['step'], kernel['family']] += 1
        by_step[key[0], kernel['step']].append(kernel)
    accepted_counts = {(row['rank'], row['step'], row['family']): row['count'] for row in proof['payload_by_rank_step']}
    require(dict(counts) == accepted_counts and len(kernels) == proof['payload_kernel_count'], 'payload inventory differs from accepted exact call coverage')
    return step_ranges, by_step


def metric_streams(db, keys):
    streams = {type_id: {'timestamps': array('q'), 'values': [array('B') for _ in NAMES], 'valid': array('b'),
                         'duplicate_rows': 0, 'incomplete_bundles': 0, 'bad_values': 0}
               for type_id, _ in keys}
    ids = sorted({metric_id for _, metric_id in keys})
    query = 'SELECT timestamp,rawTimestamp,typeId,metricId,value FROM GPU_METRICS WHERE metricId IN (' + ','.join('?' for _ in ids) + ') ORDER BY typeId,timestamp,metricId'
    previous, bundle, duplicate, bad = None, {}, 0, 0
    def emit():
        if previous is None:
            return
        type_id, timestamp = previous
        stream = streams[type_id]
        complete = set(bundle) == set(range(len(NAMES)))
        stream['timestamps'].append(timestamp)
        stream['valid'].append(complete and duplicate == 0 and bad == 0)
        stream['duplicate_rows'] += duplicate
        stream['incomplete_bundles'] += not complete
        stream['bad_values'] += bad
        for index in range(len(NAMES)):
            stream['values'][index].append(bundle.get(index, 0))
    for timestamp, raw, type_id, metric_id, value in db.execute(query, ids):
        if (type_id, metric_id) not in keys:
            continue
        if previous != (type_id, timestamp):
            emit()
            previous, bundle, duplicate, bad = (type_id, timestamp), {}, 0, 0
        index = keys[type_id, metric_id]
        duplicate += index in bundle
        good = raw == timestamp and type(value) is int and 0 <= value <= 100
        bad += not good
        bundle[index] = value if good else 0
    emit()
    return streams


def analyze(args):
    out = args.output_dir
    require(out.is_absolute() and out.parent == ROOT / 'runs' and not out.exists(), 'output must be a unique direct runs child')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'mask CUDA_VISIBLE_DEVICES for CPU-only analysis')
    require(Path(sys.prefix).resolve() == (ROOT / '.runtime/venvs/qwen-posthoc-cpu').resolve(), 'use the authorized CPU venv')
    out.mkdir()
    inputs = Inputs()
    inputs.bind(Path(__file__).resolve())
    inputs.bind(ROOT / 'ai/qwen-uccl-block-sweep/iteration-01/HARDWARE_METRIC_SCHEMA.md')
    acceptance = inputs.read(args.acceptance_dir / 'acceptance.json')
    require(acceptance.get('valid_budget_run') is True and acceptance.get('valid_nsight_timeline') is True,
            'quantitative metrics require accepted training and exact per-node payload evidence')
    accepted = {row['path']: row['sha256'] for row in acceptance['input_files']}
    inventory_path = args.acceptance_dir / 'nsight_inventory.json'
    inventory = inputs.read(inventory_path, accepted[str(inventory_path)])
    require(sorted(row['node_rank'] for row in inventory['nodes']) == list(range(4)), 'four distinct accepted node proofs required')
    windows, nodes, pooled = [], [], {view: [Counter() for _ in NAMES] for view in VIEWS}
    deadline = time.monotonic() + args.max_seconds
    intervals_path = out / 'payload_intervals.jsonl'
    with intervals_path.open('x') as interval_log:
        for proof in sorted(inventory['nodes'], key=lambda row: row['node_rank']):
            node = proof['node_rank']
            require(proof.get('valid_timeline_contents') is True and proof.get('valid_nsight_timeline') is True
                    and proof.get('report_export_binding_valid') is True, 'node event inventory is not accepted')
            path = Path(proof['path'])
            inputs.bind(path, accepted[str(path)])
            sidecar = Path(proof['export'].get('source_sidecar', path.with_suffix('.source.json')))
            exported = inputs.read(sidecar, accepted[str(sidecar)])
            require(exported['sqlite_sha256'] == accepted[str(path)], 'SQLite/report export binding differs')
            report = Path(acceptance['run_dir']) / 'nsight' / f'node-{node}-native.nsys-rep'
            require(exported['report_path'] == str(report) and exported['report_sha256'] == accepted[str(report)],
                    'sidecar does not name the exact accepted native report')
            inputs.bind(report, accepted[str(report)])
            tool = Path(exported['nsys_path'])
            inputs.bind(tool, exported['nsys_sha256'])
            recipe = tool.parent / 'python/packages/nsys_recipe/lib'
            inputs.bind(recipe / 'data_utils.py', 'c88abb9bdbbaebfcca17f67205c8b020d4e7b93c708c39765d0441534ad4cd13')
            inputs.bind(recipe / 'gpu_metrics.py', '33b6fbd8b50b6733a85041cebd7180b3ddecf125a430af8432c3bc66d812cf4e')
            run = Path(acceptance['run_dir'])
            launch_path = run / 'nsight' / f'node-{node}-launch.json'
            launch = inputs.read(launch_path, accepted[str(launch_path)])
            frequencies = [x.split('=', 1)[1] for x in launch['collector_argv_prefix'] if x.startswith('--gpu-metrics-frequency=')]
            require(len(frequencies) == 1 and frequencies[0].isdigit() and int(frequencies[0]) > 0, 'requested metric rate unavailable')
            period = 1_000_000_000 / int(frequencies[0])
            with sqlite3.connect(path.as_uri() + '?mode=ro', uri=True) as db:
                db.execute('PRAGMA query_only=ON')
                db.execute('PRAGMA cache_size=-32768')
                db.set_progress_handler(lambda: int(time.monotonic() > deadline), 10000)
                version = dict(db.execute('SELECT name,value FROM META_DATA_EXPORT'))
                require(version.get('EXPORT_PRODUCT_VERSION') == '2025.6.3.343', 'export version is outside the reviewed schema')
                strings = dict(db.execute('SELECT id,value FROM StringIds'))
                identities, metric_keys = mappings(db, strings, proof, acceptance['rank_placement'], node)
                step_ranges, payloads = payload_intervals(db, strings, proof, identities, acceptance['communication_sms'])
                streams = metric_streams(db, metric_keys)  # Exactly one selected-field metrics scan per node.
                nodes.append({'node_rank': node, 'sqlite_path': str(path), 'metric_streams': [
                    {key: value for key, value in identity.items() if key != 'cuda_mappings'} for identity in identities.values()],
                    'stream_quality': {str(t): {k: v for k, v in stream.items() if k not in ('timestamps', 'values', 'valid')}
                                       for t, stream in streams.items()}})
                for (owner, step), step_range in sorted(step_ranges.items()):
                    identity = identities[owner]
                    stream = streams[identity['type_id']]
                    cov, _, _ = coverage(stream['timestamps'], stream['valid'], *step_range, period)
                    kernels = payloads[owner, step]
                    for kernel in kernels:
                        interval_log.write(json.dumps({**kernel, 'node_rank': node, 'physical_uuid': identity['uuid']}, sort_keys=True) + '\n')
                    selections = {'optimizer_step_window': [step_range],
                        'during_any_payload': [(k['start'], k['end']) for k in kernels],
                        'during_dispatch': [(k['start'], k['end']) for k in kernels if k['family'] == 'dispatch'],
                        'during_combine': [(k['start'], k['end']) for k in kernels if k['family'] == 'combine']}
                    views = {}
                    for view, selected in selections.items():
                        views[view], hist = summarize_view(stream, selected, cov)
                        if views[view]['metrics_available']:
                            for metric in range(len(NAMES)):
                                pooled[view][metric].update(hist[metric])
                    windows.append({'rank': identity['rank'], 'node_rank': node, 'step_index': step,
                        'human_update': step + 1, 'physical_uuid': identity['uuid'], 'type_id': identity['type_id'],
                        'step_start_ns': step_range[0], 'step_end_ns': step_range[1], 'sample_coverage': cov,
                        'payload_kernels': len(kernels), 'payloads_outside_host_step_bounds': sum(
                            k['start'] < step_range[0] or k['end'] > step_range[1] for k in kernels), 'views': views})
            print(f'node={node} complete; rank_update_windows={len(windows)}', flush=True)
    require(len(windows) == 80 and {(row['rank'], row['step_index']) for row in windows}
            == {(rank, step) for rank in range(16) for step in STEPS}, 'expected sixteen distinct ranks by five measured updates')
    result = {'schema_version': 1, 'valid_bound_metric_analysis': True,
        'training_slurm_job_id': acceptance['training_slurm_job_id'], 'communication_sms': acceptance['communication_sms'],
        'acceptance_path': str(args.acceptance_dir / 'acceptance.json'), 'metric_unit': 'percent, source resolution one percentage point',
        'statistics': 'arithmetic means/linear quantiles of selected sample points; no time integration or interpolation',
        'execution': {'hostname': socket.gethostname(), 'python': sys.executable, 'python_version': sys.version,
                      'location': 'authorized_login_CPU_posthoc', 'cuda_visible_devices': '', 'gpu_execution': False},
        'nodes': nodes, 'rank_update_windows': windows, 'pooled_sample_summaries': {},
        'unavailable': {'communication_only_attribution': 'Device metrics include all concurrent GPU work during selected payload intervals.',
            'achieved_warp_occupancy': 'No conversion from unscaled raw average warp fields.',
            'simultaneous_active_sm_count': 'SM-cycle activity percentage does not identify an instantaneous physical SM count.',
            'per_kernel_metrics': 'Individual sub-millisecond kernels can have no sample; no per-kernel interpolation.'},
        'claim_limits': ['Independent node clocks are not aligned or compared.', 'Nsight-instrumented diagnostics only.',
            'Whole-session event completeness remains unresolved; accepted event scope is the known UCCL payload/call set.',
            'Conditional interval selections overlap and are not additive partitions.',
            'A sample inside a kernel interval can summarize an acquisition period that straddles the kernel boundary.']}
    for view in VIEWS:
        available = all(row['views'][view]['metrics_available'] for row in windows)
        result['pooled_sample_summaries'][view] = {'all_80_windows_available': available,
            'metrics': {name: histogram_summary(hist) if available else None for name, hist in zip(NAMES, pooled[view])}}
    inputs.recheck()
    result['input_files'] = [{'path': p, 'sha256': s} for p, s in sorted(inputs.hashes.items())]
    result['payload_interval_artifact'] = {'path': str(intervals_path), 'sha256': digest(intervals_path)}
    result['created_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    (out / 'metrics.json').write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n')
    with (out / 'metrics.csv').open('x', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['rank', 'node', 'update', 'view', 'metric', 'available', 'samples', 'mean_percent', 'median_percent', 'p95_percent'])
        for row in windows:
            for view, selected in row['views'].items():
                for name, summary in selected['metrics'].items():
                    writer.writerow([row['rank'], row['node_rank'], row['human_update'], view, name, selected['metrics_available'],
                        selected['selected_sample_bundles'], summary['arithmetic_sample_mean_percent'] if summary else '',
                        summary['median_percent'] if summary else '', summary['p95_percent'] if summary else ''])
    lines = ['# Bound hardware metric samples', '', f'Budget: {acceptance["communication_sms"]}. Training job: {acceptance["training_slurm_job_id"]}.', '',
        'Results describe all-device sampled activity during accepted UCCL payload intervals. They do not isolate communication-only activity or achieved warp occupancy.', '',
        '| Selection | Metric | Pooled arithmetic sample mean (%) | Samples |', '| --- | --- | ---: | ---: |']
    for view, item in result['pooled_sample_summaries'].items():
        for name, summary in item['metrics'].items():
            lines.append(f'| {view} | {name} | {summary["arithmetic_sample_mean_percent"]:.4f} | {summary["count"]} |' if summary
                         else f'| {view} | {name} | Unavailable: coverage gate | |')
    lines += ['', 'See metrics.json for all 80 per-rank/update coverage records, zero-sample payload intervals, device identities, limits and exact input hashes.',
        'Only the three reviewed unsigned-byte percentage fields are used. Raw warp averages and clock fields are excluded.']
    (out / 'README.md').write_text('\n'.join(lines) + '\n')


def selfcheck():
    require(uuid_key('GPU-00112233-4455-6677-8899-aabbccddeeff') == '00112233445566778899aabbccddeeff', 'UUID normalization')
    require(union([(1, 3), (2, 5), (5, 6)]) == [(1, 6)], 'overlap/adjacency must not duplicate samples')
    for bad in (None, 'GPU-123'):
        try:
            uuid_key(bad)
        except ValueError:
            pass
        else:
            raise AssertionError('missing/bad GPU UUID was accepted')
    good, _, _ = coverage([0, 1000, 2000], [1, 1, 1], 0, 3000, 1000)
    require(good['valid_step_sample_coverage'], 'valid sample window rejected')
    for times, valid in (([0, 1000, 5000], [1, 1, 1]), ([0, 1000, 2000], [1, 0, 1]), ([0], [1])):
        bad, _, _ = coverage(times, valid, 0, 6000 if len(times) == 3 and times[-1] == 5000 else 3000, 1000)
        require(not bad['valid_step_sample_coverage'], 'gap/incomplete/single sample window accepted')
    stream = {'timestamps': [0, 1000, 2000], 'values': [[10, 20, 30]] * 3, 'valid': [1, 1, 1]}
    row, _ = summarize_view(stream, [(500, 1500), (1000, 2000)], good)
    require(row['selected_sample_bundles'] == 1 and not row['metrics_available'], 'overlap duplicated a sample or sparse view accepted')
    tail, _ = summarize_view(stream, [(0, 2000), (4000, 9000)], good)
    require(not tail['metrics_available'] and not tail['intervals_within_verified_sample_window'],
            'an unsupported tail outside the verified sample window was accepted')
    fixture_uuid = '00112233445566778899aabbccddeeff'
    fixture_gpu = {(1 << 48, 3): {'uuid': fixture_uuid}}
    require(metric_target((1 << 48) + 3, fixture_gpu, fixture_uuid)['uuid'] == fixture_uuid, 'valid metric target rejected')
    for wrong_type, wrong_uuid in (((2 << 48) + 3, fixture_uuid), ((1 << 48) + 3, 'ffeeddccbbaa99887766554433221100')):
        try:
            metric_target(wrong_type, fixture_gpu, wrong_uuid)
        except ValueError:
            pass
        else:
            raise AssertionError('wrong VM/UUID mapping was accepted')
    require(histogram_summary(Counter({10: 1, 30: 1}))['median_percent'] == 20, 'linear median convention')
    with sqlite3.connect(':memory:') as db:
        db.execute('CREATE TABLE GPU_METRICS(rawTimestamp INTEGER,timestamp INTEGER,typeId INTEGER,metricId INTEGER,value INTEGER)')
        keys = {(1, 3): 0, (1, 4): 1, (1, 12): 2}
        base = [(1000, 1000, 1, metric, 10) for metric in (3, 4, 12)]
        for rows, field in ((base + [base[0]], 'duplicate_rows'),
                            (base[:-1], 'incomplete_bundles'),
                            (base[:-1] + [(1000, 1000, 1, 12, 101)], 'bad_values'),
                            (base[:-1] + [(999, 1000, 1, 12, 10)], 'bad_values')):
            db.execute('DELETE FROM GPU_METRICS')
            db.executemany('INSERT INTO GPU_METRICS VALUES (?,?,?,?,?)', rows)
            observed = metric_streams(db, keys)[1]
            require(observed[field] == 1 and list(observed['valid']) == [0], 'duplicate/missing/out-of-range/time-scale error accepted')
        db.execute('DELETE FROM GPU_METRICS')
        db.executemany('INSERT INTO GPU_METRICS VALUES (?,?,?,?,?)', base + [(1000, 1000, 1, 14, 35435516)])
        require(list(metric_streams(db, keys)[1]['valid']) == [1], 'unscaled raw warp field contaminated percentage selection')
    print('selfcheck_passed=true: UUID/VM mapping rejection, interval union, unsupported tails, gaps, sparse views, duplicate/missing/out-of-range/time-scale errors, raw-warp exclusion and quantiles')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--acceptance-dir', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--max-seconds', type=int, default=600)
    parser.add_argument('--selfcheck', action='store_true')
    args = parser.parse_args()
    if args.selfcheck:
        selfcheck()
    else:
        require(args.acceptance_dir is not None and args.acceptance_dir.is_absolute(), 'absolute accepted evidence directory required')
        require(args.output_dir is not None and 1 <= args.max_seconds <= 1200, 'output and bounded CPU deadline required')
        analyze(args)


if __name__ == '__main__':
    main()
