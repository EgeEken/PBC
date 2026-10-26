from collections import defaultdict
import json
import statistics
import sys

from support import OUT, PASS1, PASS2, save, read
from analyze import collapse, summaries, pair, frontier, interpolate


def allocation():
    controls = [json.loads(line) for line in (PASS1 / 'baseline.jsonl').read_text().splitlines()]
    controls = [{**r, 'label': r['source'] + '_' + r['preset']} for r in controls if r['trial'] == 0 and r['source'] in ('release', 'current')]
    previous = json.loads((PASS2 / 'quality_confirmation.json').read_text())
    controls += [{**r, 'label': 'pass2_500p35'} for r in previous if r['trial'] == 0 and r['label'] == 'q500_p35']
    prior_points = json.loads((PASS2 / 'rd_raw_points.json').read_text())
    prior_points = [{**r, 'label': 'pass2:' + r['label'] + ':' + r['preset'] + ':' + str(r.get('experiment'))} for r in prior_points]
    comparisons = []
    all_points = list(controls)
    for name in ('fixed_screen', 'integrated_screen', 'allocation_full'):
        raw = read(name)
        if not raw:
            continue
        save(name + '_summary', summaries(raw))
        collapsed = collapse(raw)
        all_points += [{**r, 'experiment': name} for r in collapsed]
        for row in collapsed:
            if row['optimization'] == 'none':
                continue
            for source in ('release', 'current', 'best_existing_current', 'best_existing_release', 'best_existing_all'):
                if source.startswith('best_existing_'):
                    actual_source = source.removeprefix('best_existing_')
                    points = frontier([r for r in prior_points if (r['dataset'], r['image']) == (row['dataset'], row['image']) and (actual_source == 'all' or r['source'] == actual_source)])
                else:
                    points = frontier([r for r in controls if (r['dataset'], r['image'], r['source']) == (row['dataset'], row['image'], source) and r['label'] != 'pass2_500p35'])
                match = interpolate(points, row['bpp'])
                mse_match = interpolate(points, row['mse'], 'mse')
                nearest = min(points, key=lambda p: abs(p['bpp'] / row['bpp'] - 1))
                qhq = bool(match and all(match[k]['label'] in (source + '_quality', source + '_high_quality') for k in ('lower', 'upper')))
                ordinary = next((r for r in collapsed if r['dataset'] == row['dataset'] and r['image'] == row['image'] and r['label'] == row['label'].split('_')[0] + '_none'), None)
                comparisons.append(dict(experiment=name, dataset=row['dataset'], image=row['image'], label=row['label'], source=row['source'],
                    preset=row['preset'], mse=row['mse'], bpp=row['bpp'], baseline_source=source,
                    matched_rate=match, matched_mse=mse_match, q_hq_bracket=qhq,
                    nearest_control={k: nearest[k] for k in ('label', 'bpp', 'mse')},
                    nearest_rate_distance_pct=100 * abs(nearest['bpp'] / row['bpp'] - 1),
                    nearest_mse_pct=100 * (row['mse'] / nearest['mse'] - 1),
                    matched_rate_mse_pct=100 * (row['mse'] / match['value'] - 1) if match else None,
                    matched_mse_bpp_pct=100 * (row['bpp'] / mse_match['value'] - 1) if mse_match else None,
                    observed_control_dominated=any(p['mse'] <= row['mse'] and p['bpp'] <= row['bpp'] and (p['mse'] < row['mse'] or p['bpp'] < row['bpp']) for p in points),
                    ordinary_mse=ordinary['mse'] if ordinary else None, ordinary_bpp=ordinary['bpp'] if ordinary else None,
                    ordinary_encode=ordinary['encode_seconds'] if ordinary else None,
                    strict_rd_vs_ordinary=bool(ordinary and row['mse'] <= ordinary['mse'] and row['bpp'] <= ordinary['bpp'] and (row['mse'] < ordinary['mse'] or row['bpp'] < ordinary['bpp']))))
    save('allocation_matched', comparisons)
    groups = defaultdict(list)
    for row in comparisons:
        groups[(row['experiment'], row['dataset'], row['label'], row['baseline_source'])].append(row)
    result = []
    for (experiment, dataset, label, source), group in groups.items():
        valid = [r['matched_rate_mse_pct'] for r in group if r['matched_rate_mse_pct'] is not None]
        quality = [r['matched_rate_mse_pct'] for r in group if r['q_hq_bracket']]
        distortions = [r['matched_mse_bpp_pct'] for r in group if r['matched_mse_bpp_pct'] is not None]
        result.append(dict(experiment=experiment, dataset=dataset, label=label, baseline_source=source, cases=len(group),
            overlap=len(valid), mean_matched_rate_mse_pct=statistics.mean(valid) if valid else None,
            median_matched_rate_mse_pct=statistics.median(valid) if valid else None, matched_rate_wins=sum(v < 0 for v in valid),
            q_hq_overlap=len(quality), q_hq_mse_pct=statistics.mean(quality) if quality else None, q_hq_wins=sum(v < 0 for v in quality),
            mse_overlap=len(distortions), matched_mse_bpp_pct=statistics.mean(distortions) if distortions else None,
            observed_dominated=sum(r['observed_control_dominated'] for r in group), strict_rd_vs_ordinary=sum(r['strict_rd_vs_ordinary'] for r in group)))
        result[-1]['near_5pct_rate_cases'] = sum(r['nearest_rate_distance_pct'] <= 5 for r in group)
        result[-1]['near_5pct_mse_wins'] = sum(r['nearest_rate_distance_pct'] <= 5 and r['nearest_mse_pct'] < 0 for r in group)
    save('allocation_matched_summary', result)
    save('allocation_points', [{k: r.get(k) for k in ('dataset', 'image', 'label', 'preset', 'source', 'mse', 'bpp', 'encode_seconds', 'decode_seconds', 'final_bytes', 'raw_bytes', 'config', 'optimization', 'experiment', 'stream_path')} for r in all_points])
    if read('allocation_full'):
        outcomes()
        geometry_ablation()


def outcomes():
    rows = collapse(read('allocation_full'))
    pairs = []
    for candidate in [r for r in rows if r['optimization'] != 'none']:
        for baseline in [r for r in rows if (r['dataset'], r['image']) == (candidate['dataset'], candidate['image']) and r['optimization'] == 'none']:
            record = {k: candidate[k] for k in ('dataset', 'image', 'label', 'optimization')}
            record['baseline'] = baseline['label']
            for metric in ('mse', 'bpp', 'raw_bytes', 'final_bytes', 'encode_seconds', 'decode_seconds', 'iterative_patches'):
                record[metric + '_baseline'] = baseline[metric]
                record[metric + '_candidate'] = candidate[metric]
                record[metric + '_pct'] = 100 * (candidate[metric] / baseline[metric] - 1) if baseline[metric] else None
            rd_win = candidate['mse'] <= baseline['mse'] and candidate['bpp'] <= baseline['bpp'] and (candidate['mse'] < baseline['mse'] or candidate['bpp'] < baseline['bpp'])
            regression = candidate['mse'] >= baseline['mse'] and candidate['bpp'] >= baseline['bpp'] and (candidate['mse'] > baseline['mse'] or candidate['bpp'] > baseline['bpp'])
            record['strict_rd_win'] = rd_win
            record['classification'] = 'RD improvement at observed points' if rd_win else 'Observed RD regression' if regression else 'Quality-for-rate tradeoff' if candidate['mse'] < baseline['mse'] else 'Rate-for-quality tradeoff'
            pairs.append(record)
    save('allocation_paired', pairs)


def geometry_ablation():
    rows = collapse(read('allocation_full'))
    pairs = []
    for candidate in [r for r in rows if r['optimization'].startswith('integrated_')]:
        fixed = frontier([r for r in rows if (r['dataset'], r['image']) == (candidate['dataset'], candidate['image']) and
            (r['optimization'].startswith('fixed_') or r['label'] == 'q500p35_none')])
        match = interpolate(fixed, candidate['bpp'])
        pairs.append(dict(dataset=candidate['dataset'], image=candidate['image'], label=candidate['label'],
            candidate_mse=candidate['mse'], candidate_bpp=candidate['bpp'], matched_fixed_geometry=match,
            mse_pct=100 * (candidate['mse'] / match['value'] - 1) if match else None))
    save('geometry_ablation_matched', pairs)
    summary = []
    for dataset in ('hpt_data', 'hpt_data_val'):
        for label in sorted({r['label'] for r in pairs}):
            valid = [r['mse_pct'] for r in pairs if r['dataset'] == dataset and r['label'] == label and r['mse_pct'] is not None]
            summary.append(dict(dataset=dataset, label=label, overlap=len(valid),
                mean_mse_pct=statistics.mean(valid) if valid else None, wins=sum(v < 0 for v in valid)))
    save('geometry_ablation_summary', summary)


def traces():
    groups = defaultdict(list)
    for row in read('search_profile'):
        for phase, stats in row['codec_trace']['summary'].items():
            groups[(row['label'], phase)].append(stats)
    save('search_trace_summary', [dict(variant=v, phase=p, cases=len(rows),
        inclusive_seconds=statistics.mean(r['total_ns'] for r in rows) / 1e9,
        exclusive_seconds=statistics.mean(r['exclusive_ns'] for r in rows) / 1e9,
        count=statistics.mean(r['count'] for r in rows)) for (v, p), rows in groups.items()])


def trajectory_comparison():
    import sys
    from support import ROOT
    sys.path.insert(0, str(ROOT))
    from stream_tools import parse
    full = collapse(read('allocation_full'))
    old = json.loads((PASS2 / 'quality_confirmation.json').read_text())
    compared = []
    for row in [r for r in full if r['optimization'] != 'none']:
        baseline = next(r for r in old if r['dataset'] == row['dataset'] and r['image'] == row['image'] and r['label'] == 'q500_p35' and r['trial'] == 0)
        _, a, _ = parse((PASS2 / baseline['stream_path']).read_bytes())
        _, b, _ = parse((OUT / row['stream_path']).read_bytes())
        geometry = lambda p: tuple(p[k] for k in ('channel', 'x', 'y', 'w', 'h'))
        equal = len(a) == len(b) and all(geometry(x) == geometry(y) for x, y in zip(a, b))
        if row['optimization'].startswith('fixed_'):
            assert equal
        compared.append(dict(dataset=row['dataset'], image=row['image'], label=row['label'],
            same_geometry_order_count=equal, initialization_equal=row['patch_signatures'][:3] == baseline['patch_signatures'][:3],
            first_geometry_change=next((i for i, (x, y) in enumerate(zip(a, b)) if geometry(x) != geometry(y)), None),
            first_payload_change=next((i for i, (x, y) in enumerate(zip(row['patch_signatures'], baseline['patch_signatures'])) if x != y), None),
            baseline_patches=len(a), candidate_patches=len(b)))
    save('allocation_trajectories', compared)


def performance():
    for name in ('performance_screen', 'performance_full', 'search_screen', 'search_setup_screen', 'search_full'):
        rows = read(name)
        if not rows:
            continue
        save(name + '_summary', summaries(rows))
        baseline = 'baseline'
        save(name + '_paired', pair(rows, [(v, baseline) for v in sorted({r['label'] for r in rows}) if v != baseline], name + '_paired'))
    summaries_combined = []
    for name in ('performance_screen', 'performance_full'):
        rows = read(name + '_summary')
        for dataset in ('hpt_data', 'hpt_data_val'):
            selected = {r['label']: r for r in rows if r['dataset'] == dataset and r['preset'] == 'all'}
            if len(selected) != 4:
                continue
            b, g, s, c = [selected[k] for k in ('baseline', 'grid', 'skip', 'both')]
            summaries_combined.append(dict(experiment=name, dataset=dataset, baseline_encode=b['encode_seconds'],
                grid_encode=g['encode_seconds'], skip_encode=s['encode_seconds'], combined_encode=c['encode_seconds'],
                baseline_decode=b['decode_seconds'], grid_decode=g['decode_seconds'], combined_decode=c['decode_seconds'],
                sum_of_separate_time_savings=(b['encode_seconds'] - g['encode_seconds']) + (b['encode_seconds'] - s['encode_seconds']),
                measured_combined_saving=b['encode_seconds'] - c['encode_seconds'],
                combined_offset_seconds=c['encode_seconds'] - (g['encode_seconds'] + s['encode_seconds'] - b['encode_seconds'])))
    save('combined_effects', summaries_combined)


def static_study():
    rows = read('representation_study')
    groups = defaultdict(list)
    for row in rows:
        if row['representation'] != 'original':
            groups[(row['label'], row['representation'])].append(row)
    result = []
    for (label, method), group in groups.items():
        valid = [r for r in group if r['original_sse_gain'] > 0]
        result.append(dict(label=label, method=method, patches=len(group),
            near_lossless_cheap=sum(r['retained_gain_fraction'] >= .98 and r['raw_bit_fraction'] <= .75 for r in valid),
            positive_original_cases=len(valid), mean_retained_gain=statistics.mean(r['retained_gain_fraction'] for r in valid) if valid else None,
            mean_bit_fraction=statistics.mean(r['raw_bit_fraction'] for r in group),
            mean_evaluation_seconds=statistics.mean(r['evaluation_seconds'] for r in group)))
    save('representation_study_summary', result)


if __name__ == '__main__':
    globals()[sys.argv[1]]()
