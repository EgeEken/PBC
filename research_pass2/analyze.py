from collections import defaultdict
import json
import math
import statistics
import sys

from common import OUT, PASS1, PRESETS, save

METRICS = ('mse', 'bpp', 'final_bytes', 'raw_bytes', 'encode_seconds', 'decode_seconds', 'initialization_patches', 'iterative_patches')


def read(name):
    path = OUT / (name + '.json')
    return json.loads(path.read_text()) if path.exists() else []


def collapse(rows, keys=('dataset', 'image', 'preset', 'label')):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    result = []
    for key, group in groups.items():
        row = dict(group[0])
        for metric in METRICS:
            if metric in row:
                values = [r[metric] for r in group]
                row[metric] = statistics.median(values)
                if metric.endswith('seconds'):
                    row[metric + '_mean'] = statistics.mean(values)
                    row[metric + '_cv'] = statistics.pstdev(values) / statistics.mean(values)
                    row[metric + '_min'] = min(values)
                    row[metric + '_max'] = max(values)
        row['repeats'] = len(group)
        row['deterministic'] = len({r['stream_sha256'] for r in group}) == 1
        result.append(row)
    return result


def pair(rows, labels, name):
    collapsed = collapse(rows)
    comparisons = []
    for candidate, baseline in labels:
        for row in [r for r in collapsed if r['label'] == candidate]:
            other = next((r for r in collapsed if (r['label'], r['dataset'], r['image'], r['preset']) == (baseline, row['dataset'], row['image'], row['preset'])), None)
            if other is None:
                continue
            result = {k: row[k] for k in ('dataset', 'image', 'preset')}
            result.update(candidate=candidate, baseline=baseline, bytes_equal=row['stream_sha256'] == other['stream_sha256'],
                pixels_equal=row['pixels_sha256'] == other['pixels_sha256'], patches_equal=row['patch_signatures'] == other['patch_signatures'])
            for metric in METRICS:
                if metric in row and metric in other:
                    result[metric + '_baseline'] = other[metric]
                    result[metric + '_candidate'] = row[metric]
                    result[metric + '_pct'] = 100 * (row[metric] / other[metric] - 1) if other[metric] else None
            comparisons.append(result)
    save(name, comparisons)
    return comparisons


def summaries(rows):
    rows = collapse(rows)
    groups = defaultdict(list)
    for row in rows:
        groups[(row['dataset'], row['preset'], row['label'])].append(row)
        groups[(row['dataset'], 'all', row['label'])].append(row)
    results = []
    for (dataset, preset, label), group in groups.items():
        row = dict(dataset=dataset, preset=preset, label=label, cases=len(group), deterministic=all(r['deterministic'] for r in group))
        for metric in METRICS:
            if metric in group[0]:
                row[metric] = statistics.mean(r[metric] for r in group)
                if metric.endswith('seconds'):
                    row[metric + '_mean_cv'] = statistics.mean(r[metric + '_cv'] for r in group)
                    row[metric + '_mean_of_calls'] = statistics.mean(r[metric + '_mean'] for r in group)
        results.append(row)
    return results


def frontier(points):
    result = []
    best = math.inf
    for row in sorted(points, key=lambda r: (r['bpp'], r['mse'])):
        if row['mse'] < best:
            result.append(row)
            best = row['mse']
    return result


def interpolate(points, value, axis='bpp'):
    other = 'mse' if axis == 'bpp' else 'bpp'
    ordered = sorted(points, key=lambda r: r[axis])
    if not ordered or value < ordered[0][axis] or value > ordered[-1][axis]:
        return None
    for a, b in zip(ordered, ordered[1:]):
        if a[axis] <= value <= b[axis] and a[axis] != b[axis]:
            weight = (value - a[axis]) / (b[axis] - a[axis])
            return {'value': a[other] * (1 - weight) + b[other] * weight,
                'lower': {k: a[k] for k in ('bpp', 'mse', 'label', 'preset') if k in a},
                'upper': {k: b[k] for k in ('bpp', 'mse', 'label', 'preset') if k in b}, 'weight': weight}
    exact = next((r for r in ordered if r[axis] == value), None)
    return {'value': exact[other], 'lower': exact, 'upper': exact, 'weight': 0} if exact else None


def rd():
    baseline_rows = [json.loads(line) for line in (PASS1 / 'baseline.jsonl').read_text().splitlines()]
    controls = []
    for row in baseline_rows:
        if row['trial'] == 0 and row['source'] in ('release', 'current'):
            controls.append({**row, 'label': row['source'] + '_' + row['preset'], 'kind': 'default_control'})
    candidates = []
    for name in ('resolution_screen', 'allocation_screen', 'rate_screen', 'quality_confirmation'):
        for row in collapse(read(name)):
            row['experiment'] = name
            row['kind'] = 'candidate'
            candidates.append(row)
    comparison = []
    frontiers = []
    for row in candidates:
        if row.get('config', {}).get('auto_downsample_max_pixels') == 250000 and row['label'] in ('release_250000', 'current_250000'):
            continue
        for source in ('release', 'current'):
            points = [r for r in controls if (r['dataset'], r['image'], r['source']) == (row['dataset'], row['image'], source)]
            boundary = frontier(points)
            matched_rate = interpolate(boundary, row['bpp'])
            matched_mse = interpolate(boundary, row['mse'], 'mse')
            nearest = min(points, key=lambda p: abs(p['bpp'] / row['bpp'] - 1))
            quality_region = bool(matched_rate and all(matched_rate[edge]['label'] in (source + '_quality', source + '_high_quality') for edge in ('lower', 'upper')))
            comparison.append({k: row[k] for k in ('dataset', 'image', 'preset', 'label', 'experiment', 'mse', 'bpp')} | {
                'baseline_source': source, 'matched_rate': matched_rate, 'matched_mse': matched_mse,
                'quality_to_high_quality_bracket': quality_region,
                'matched_rate_mse_pct': 100 * (row['mse'] / matched_rate['value'] - 1) if matched_rate else None,
                'matched_mse_bpp_pct': 100 * (row['bpp'] / matched_mse['value'] - 1) if matched_mse else None,
                'nearest_measured_control': {k: nearest[k] for k in ('label', 'mse', 'bpp')},
                'nearest_rate_distance_pct': 100 * abs(nearest['bpp'] / row['bpp'] - 1),
                'dominated_by_observed_control': any(p['bpp'] <= row['bpp'] and p['mse'] <= row['mse'] and (p['bpp'] < row['bpp'] or p['mse'] < row['mse']) for p in points)})
    for dataset, image in sorted({(r['dataset'], r['image']) for r in candidates}):
        points = [r for r in candidates + controls if (r['dataset'], r['image']) == (dataset, image)]
        for row in frontier(points):
            frontiers.append({k: row[k] for k in ('dataset', 'image', 'label', 'preset', 'mse', 'bpp', 'kind')} | {'source': row['source']})
    save('rd_matched_comparisons', comparison)
    save('observed_rd_frontiers', frontiers)
    save('rd_raw_points', [{k: r.get(k) for k in ('dataset', 'image', 'label', 'source', 'preset', 'mse', 'bpp', 'final_bytes', 'raw_bytes', 'encode_seconds', 'decode_seconds', 'kind', 'experiment', 'config')} for r in candidates + controls])
    groups = defaultdict(list)
    for row in comparison:
        groups[(row['experiment'], row['dataset'], row['label'], row['preset'], row['baseline_source'])].append(row)
    result = []
    for (experiment, dataset, label, preset, source), rows in groups.items():
        valid = [r['matched_rate_mse_pct'] for r in rows if r['matched_rate_mse_pct'] is not None]
        quality_region = [r['matched_rate_mse_pct'] for r in rows if r['quality_to_high_quality_bracket']]
        rate = [r['matched_mse_bpp_pct'] for r in rows if r['matched_mse_bpp_pct'] is not None]
        result.append(dict(experiment=experiment, dataset=dataset, label=label, preset=preset, baseline_source=source,
            cases=len(rows), rate_overlap_cases=len(valid), mean_matched_rate_mse_pct=statistics.mean(valid) if valid else None,
            matched_rate_wins=sum(v < 0 for v in valid), mse_overlap_cases=len(rate),
            q_hq_overlap_cases=len(quality_region), q_hq_mean_matched_rate_mse_pct=statistics.mean(quality_region) if quality_region else None,
            q_hq_matched_rate_wins=sum(v < 0 for v in quality_region),
            mean_matched_mse_bpp_pct=statistics.mean(rate) if rate else None,
            observed_control_dominated=sum(r['dominated_by_observed_control'] for r in rows),
            closely_matched_rate_cases=sum(r['nearest_rate_distance_pct'] <= 5 for r in rows)))
    save('rd_matched_summary', result)
    curve_comparisons = []
    for dataset, image in sorted({(r['dataset'], r['image']) for r in candidates}):
        for limit in (500000, 1000000):
            curve = frontier([r for r in candidates if (r['dataset'], r['image'], r['source']) == (dataset, image, 'current') and r['config']['auto_downsample_max_pixels'] == limit])
            for control in [r for r in controls if (r['dataset'], r['image']) == (dataset, image) and r['preset'] in ('quality', 'high_quality')]:
                match = interpolate(curve, control['bpp'])
                rate = interpolate(curve, control['mse'], 'mse')
                curve_comparisons.append(dict(dataset=dataset, image=image, limit=limit,
                    measured_nondominated_points=len(curve),
                    control_source=control['source'], control_preset=control['preset'],
                    control_bpp=control['bpp'], control_mse=control['mse'], match=match, matched_mse_rate=rate,
                    mse_at_control_rate_pct=100 * (match['value'] / control['mse'] - 1) if match else None,
                    rate_at_control_mse_pct=100 * (rate['value'] / control['bpp'] - 1) if rate else None))
    save('curves_at_control_rates', curve_comparisons)


def entropy():
    rows = sum([read(name) for name in ('entropy_screen', 'entropy_hpt_data', 'entropy_hpt_data_val')], [])
    groups = defaultdict(list)
    for row in rows:
        # Each screening/confirmation table is summarized separately to avoid duplicate cases.
        groups[(row['dataset'], row['source'], row['preset'], row['kind'], row['method'], 'screen' if len(rows) and row.get('screen', False) else row.get('phase', 'unspecified'))].append(row)
    results = []
    for key, values in groups.items():
        dataset, source, preset, kind, method, phase = key
        cases = defaultdict(list)
        for row in values:
            cases[row['image']].append(row)
        result = dict(dataset=dataset, source=source, preset=preset, kind=kind, method=method, phase=phase, cases=len(cases))
        for metric in ('final_bytes', 'baseline_bytes', 'bpp', 'representation_pack_seconds', 'decode_seconds', 'entropy_pack_seconds', 'entropy_unpack_seconds'):
            if metric in values[0]:
                result[metric] = statistics.mean(statistics.median(v[metric] for v in group) for group in cases.values())
        result['byte_change_pct'] = 100 * (result['final_bytes'] / result['baseline_bytes'] - 1)
        result['byte_wins'] = sum(group[0]['final_bytes'] < group[0]['baseline_bytes'] for group in cases.values())
        result['production_decoder_compatible'] = all(v.get('production_decoder_compatible', kind != 'representation' or method == 'control') for v in values)
        results.append(result)
    save('entropy_summary', results)


def all_results():
    names = ('rng_control', 'resolution_screen', 'allocation_screen', 'rate_screen', 'quality_confirmation', 'performance_screen', 'performance_confirmation', 'performance_components')
    for name in names:
        rows = read(name)
        if rows:
            save(name + '_summary', summaries(rows))
    if read('rng_control'):
        pair(read('rng_control'), [('numpy_root', 'release')], 'rng_paired')
    for name in ('performance_screen', 'performance_confirmation'):
        rows = read(name)
        if rows:
            pair(rows, [(label, 'none') for label in sorted({r['label'] for r in rows}) if label != 'none'], name + '_paired')
    rd()
    entropy()
    quality_pairs()
    profile_components()


def quality_pairs():
    rows = collapse(read('quality_confirmation'))
    paired = []
    for candidate in [r for r in rows if r['label'].startswith('q500')]:
        for control in [r for r in rows if r['image'] == candidate['image'] and r['dataset'] == candidate['dataset'] and not r['label'].startswith('q500')]:
            row = {k: candidate[k] for k in ('dataset', 'image', 'label')}
            row.update(control_label=control['label'], control_preset=control['preset'], control_source=control['source'])
            for metric in METRICS:
                row[metric + '_baseline'] = control[metric]
                row[metric + '_candidate'] = candidate[metric]
                row[metric + '_pct'] = 100 * (candidate[metric] / control[metric] - 1) if control[metric] else None
            row['strict_rd_win'] = candidate['mse'] <= control['mse'] and candidate['bpp'] <= control['bpp'] and (candidate['mse'] < control['mse'] or candidate['bpp'] < control['bpp'])
            row['observed_rd_regression'] = candidate['mse'] >= control['mse'] and candidate['bpp'] >= control['bpp'] and (candidate['mse'] > control['mse'] or candidate['bpp'] > control['bpp'])
            row['rd_classification'] = ('RD improvement at measured points' if row['strict_rd_win'] else
                'Regression at measured points' if row['observed_rd_regression'] else
                'Quality-for-rate tradeoff' if candidate['mse'] < control['mse'] else
                'Rate-for-quality tradeoff' if candidate['bpp'] < control['bpp'] else 'Unchanged')
            paired.append(row)
    save('quality_confirmation_paired', paired)
    trajectories = []
    for a in [r for r in rows if r['label'] == 'q500_p20']:
        b = next(r for r in rows if (r['label'], r['dataset'], r['image']) == ('q500_p35', a['dataset'], a['image']))
        trajectories.append(dict(dataset=a['dataset'], image=a['image'],
            initialization_equal=a['patch_signatures'][:3] == b['patch_signatures'][:3],
            first_differing_stream_patch=next((i for i, (x, y) in enumerate(zip(a['patch_signatures'], b['patch_signatures'])) if x != y), None),
            p20_count=a['iterative_patches'], p35_count=b['iterative_patches']))
    save('budget_trajectory_comparisons', trajectories)


def profile_components():
    result = []
    targets = ('integral', 'search_boxes', 'sample_box', 'extract_features', 'final_mse', 'read_grid', 'write_grid', 'read_patch', 'write_patch', 'channel_sum_error', 'channel_sum')
    for name in ('rng_profile', 'performance_components'):
        rows = read(name)
        for label in sorted({r['label'] for r in rows}):
            selected = [r for r in rows if r['label'] == label]
            for target in targets:
                stats = [p for r in selected for p in r['profile'] if p['function'].lstrip('_') == target]
                if stats:
                    result.append(dict(experiment=name, variant=label, function=target, cases=len(selected),
                        self_seconds_per_case=sum(p['self_seconds'] for p in stats) / len(selected),
                        cumulative_seconds_per_case=sum(p['cumulative_seconds'] for p in stats) / len(selected),
                        calls_per_case=sum(p['calls'] for p in stats) / len(selected)))
    save('profile_component_summary', result)
    decoder = read('decoder_components')
    summary = []
    for method in sorted({r['method'] for r in decoder}):
        for phase in sorted({r['phase'] for r in decoder}):
            selected = [r for r in decoder if r['method'] == method and r['phase'] == phase]
            summary.append(dict(method=method, phase=phase, calls=len(selected),
                inclusive_seconds=statistics.mean(r['inclusive_seconds'] for r in selected),
                exclusive_seconds=statistics.mean(r['exclusive_seconds'] for r in selected)))
    save('decoder_component_summary', summary)


if __name__ == '__main__':
    globals()[sys.argv[1] if len(sys.argv) > 1 else 'all_results']()
