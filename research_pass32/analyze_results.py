from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
from PIL import Image, ImageOps

import pbc32_codec as codec

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
OUT = HERE / 'outputs'
VIS = HERE / 'visuals'
NEW = ('cathedral.png', 'park.jpg', 'seagull.jpg')
COLORS = {'B': '#606c80', 'D0': '#3288bd', 'D1': '#e58e26', 'D': '#c23b57'}
MODES = ('skip', 'dc', 'vertical_no_offset', 'horizontal_no_offset', 'local_detail', 'vertical_with_offset', 'horizontal_with_offset')
MODE_LABELS = ('SKIP', 'DC', 'Vertical', 'Horizontal', 'Detail', 'Vertical + DC', 'Horizontal + DC')
MODE_COLORS = ('#30343b', '#f6bd60', '#2a9d8f', '#457b9d', '#e76f51', '#8e6cbb', '#d1495b')


def frontier(rows: list[dict]) -> list[dict]:
    kept = []
    for row in sorted(rows, key=lambda x: (x['total_bytes'], x['metrics']['mse'])):
        if any(other['total_bytes'] <= row['total_bytes'] and other['metrics']['mse'] <= row['metrics']['mse'] and
               (other['total_bytes'] < row['total_bytes'] or other['metrics']['mse'] < row['metrics']['mse']) for other in rows):
            continue
        kept.append(row)
    return kept


def pair_stats(b: dict, c: dict) -> dict:
    rate_gap = 100 * (c['total_bytes'] / b['total_bytes'] - 1)
    mse_gap = 100 * (c['metrics']['mse'] / b['metrics']['mse'] - 1)
    return {'baseline': b['key'], 'candidate': c['key'], 'baseline_bytes': b['total_bytes'],
            'candidate_bytes': c['total_bytes'], 'signed_rate_gap_pct': rate_gap,
            'baseline_mse': b['metrics']['mse'], 'candidate_mse': c['metrics']['mse'], 'signed_mse_gap_pct': mse_gap}


def compare_configs(rows: list[dict], baseline_config: str, candidate_config: str, estimator: str) -> dict:
    base = [r for r in rows if r['rate_estimator'] == estimator and r['config'] == baseline_config]
    cand = [r for r in rows if r['rate_estimator'] == estimator and r['config'] == candidate_config]
    pairs = [(b, c) for b in base for c in cand]
    if not pairs:
        return {'estimator': estimator, 'baseline': baseline_config, 'candidate': candidate_config, 'pairs': []}
    near = min(pairs, key=lambda p: abs(p[1]['total_bytes'] / p[0]['total_bytes'] - 1))
    near_rate = [p for p in pairs if abs(p[1]['total_bytes'] / p[0]['total_bytes'] - 1) <= .05]
    near_mse = [p for p in pairs if abs(p[1]['metrics']['mse'] / p[0]['metrics']['mse'] - 1) <= .05]
    best_rate = max(near_mse, key=lambda p: (p[0]['total_bytes'] - p[1]['total_bytes']) / p[0]['total_bytes'], default=None)
    best_mse = max(near_rate, key=lambda p: (p[0]['metrics']['mse'] - p[1]['metrics']['mse']) / p[0]['metrics']['mse'], default=None)
    out = {'estimator': estimator, 'baseline': baseline_config, 'candidate': candidate_config,
           'closest_measured_pair': pair_stats(*near),
           'closest_pair_within_5pct_rate': pair_stats(*min(near_rate, key=lambda p: abs(p[1]['total_bytes'] / p[0]['total_bytes'] - 1))) if near_rate else None,
           'best_byte_saving_at_within_5pct_mse': pair_stats(*best_rate) if best_rate else None,
           'best_mse_gain_at_within_5pct_rate': pair_stats(*best_mse) if best_mse else None,
           'screen_threshold_met': bool((best_rate and (best_rate[0]['total_bytes'] - best_rate[1]['total_bytes']) / best_rate[0]['total_bytes'] >= .10) or
                                        (best_mse and (best_mse[0]['metrics']['mse'] - best_mse[1]['metrics']['mse']) / best_mse[0]['metrics']['mse'] >= .10))}
    return out


def compare_estimators(rows: list[dict], config: str) -> dict:
    r0 = [r for r in rows if r['rate_estimator'] == 'R0' and r['config'] == config]
    r1 = [r for r in rows if r['rate_estimator'] == 'R1' and r['config'] == config]
    pairs = [(a, b) for a in r0 for b in r1]
    near = min(pairs, key=lambda p: abs(p[1]['total_bytes'] / p[0]['total_bytes'] - 1))
    by_rate = [p for p in pairs if abs(p[1]['total_bytes'] / p[0]['total_bytes'] - 1) <= .05]
    by_mse = [p for p in pairs if abs(p[1]['metrics']['mse'] / p[0]['metrics']['mse'] - 1) <= .05]
    best_rate = max(by_mse, key=lambda p: (p[0]['total_bytes'] - p[1]['total_bytes']) / p[0]['total_bytes'], default=None)
    best_mse = max(by_rate, key=lambda p: (p[0]['metrics']['mse'] - p[1]['metrics']['mse']) / p[0]['metrics']['mse'], default=None)
    return {'config': config, 'closest_measured_pair': pair_stats(*near),
            'closest_pair_within_5pct_rate': pair_stats(*min(by_rate, key=lambda p: abs(p[1]['total_bytes'] / p[0]['total_bytes'] - 1))) if by_rate else None,
            'best_R1_byte_saving_at_within_5pct_mse': pair_stats(*best_rate) if best_rate else None,
            'best_R1_mse_gain_at_within_5pct_rate': pair_stats(*best_mse) if best_mse else None,
            'R1_point_count': len(r1), 'R0_point_count': len(r0)}


def estimate_accuracy(rows: list[dict]) -> dict:
    out = {}
    for estimator in ('R0', 'R1'):
        group = [r for r in rows if r['rate_estimator'] == estimator]
        actual = np.array([r['enhancement_bytes'] for r in group], np.float64)
        predicted = np.array([r['estimated_enhancement_bytes'] for r in group], np.float64)
        rel = (predicted - actual) / actual
        out[estimator] = {'candidate_count': len(group), 'mean_absolute_percentage_error': float(np.mean(np.abs(rel)) * 100),
                          'signed_mean_error_pct': float(np.mean(rel) * 100),
                          'pearson_correlation_estimated_vs_actual_bytes': float(np.corrcoef(predicted, actual)[0, 1])}
        per_mode = {}
        for mode in MODES:
            profiles = [r['parameter_profile'][str(i)] for r in group for i, name in enumerate(MODES) if name == mode and str(i) in r['parameter_profile']]
            if profiles:
                per_mode[mode] = {'serialized_parameter_symbols': int(sum(x['symbols'] for x in profiles)),
                                  'mean_empirical_symbol_entropy_bits': float(np.average([x['shannon_bits_per_symbol'] for x in profiles], weights=[x['symbols'] for x in profiles])),
                                  'mean_zero_fraction': float(np.average([x['zero_fraction'] for x in profiles], weights=[x['symbols'] for x in profiles]))}
        out[estimator]['observed_parameter_diagnostics_by_mode'] = per_mode
    return out


def make_rd_plot(images: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), constrained_layout=True)
    for ax, name in zip(axes, NEW):
        rows = images[name]['results']
        for config in ('B', 'D0', 'D1', 'D'):
            for estimator, style in (('R0', '--'), ('R1', '-')):
                points = sorted((r for r in rows if r['config'] == config and r['rate_estimator'] == estimator), key=lambda x: x['total_bytes'])
                ax.plot([x['total_bytes'] / 1000 for x in points], [x['metrics']['mse'] for x in points],
                        color=COLORS[config], linestyle=style, linewidth=1.5, marker='o', markersize=3)
        ax.set_title(name)
        ax.set_xlabel('Total bytes (base + enhancement), thousands')
        ax.set_ylabel('Original RGB MSE')
        ax.grid(True, alpha=.25)
    handles = [Line2D([0], [0], color=COLORS[c], lw=2, label=c) for c in ('B', 'D0', 'D1', 'D')]
    handles += [Line2D([0], [0], color='#222222', lw=2, linestyle='--', label='R0 fixed'),
                Line2D([0], [0], color='#222222', lw=2, linestyle='-', label='R1 histogram')]
    fig.legend(handles=handles, loc='outside lower center', ncol=6, frameon=False)
    fig.savefig(VIS / 'rate_distortion.png', dpi=170)
    plt.close(fig)


def make_mode_map(images: dict) -> dict:
    name, estimator, config, lam = 'cat.jpg', 'R1', 'D', 20.0
    row = next(r for r in images[name]['results'] if r['rate_estimator'] == estimator and r['config'] == config and r['lambda'] == lam)
    base_row = next(r for r in json.loads((OUT / 'manifest.json').read_text())['images'] if r['image'] == name)
    base = np.load(base_row['base_npy'], allow_pickle=False)
    path = OUT / 'streams' / name / f'{estimator}_{config}_{lam:g}.p31r'
    _, modes, _ = codec.decode(path.read_bytes(), base)
    counts = np.bincount(modes, minlength=7).astype(int).tolist()
    expected = [row['mode_counts'][mode] for mode in MODES]
    if counts != expected:
        raise AssertionError(f'mode map counts disagree with result row: {counts} != {expected}')
    cols, rows = math.ceil(base.shape[1] / 16), math.ceil(base.shape[0] / 16)
    cmap = plt.matplotlib.colors.ListedColormap(MODE_COLORS)
    fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)
    ax.imshow(modes.reshape(rows, cols), cmap=cmap, vmin=-.5, vmax=6.5, interpolation='nearest')
    ax.set_title('Cat — D / R1 / lambda 20; per-cell action')
    ax.set_xlabel('16-pixel cell column')
    ax.set_ylabel('16-pixel cell row')
    legend = [Patch(facecolor=MODE_COLORS[i], label=f'{i}: {MODE_LABELS[i]} ({counts[i]:,})') for i in range(7)]
    ax.legend(handles=legend, loc='upper left', bbox_to_anchor=(1.01, 1), frameon=False)
    fig.savefig(VIS / 'mode_selection_cat_D_R1_lambda20.png', dpi=170, bbox_inches='tight')
    plt.close(fig)
    return {'image': name, 'rate_estimator': estimator, 'config': config, 'lambda': lam,
            'decoded_mode_counts': counts, 'summary_mode_counts': expected, 'counts_match': True,
            'display_labels': list(MODE_LABELS)}


def compact(rows: dict, analysis: dict) -> dict:
    top = {'provenance': rows['provenance'], 'config': rows['config'], 'analysis': analysis, 'images': {}}
    for name, im in rows['images'].items():
        top['images'][name] = {k: im[k] for k in ('class', 'width', 'height', 'source_sha256', 'base_stream_sha256',
            'decoded_base_rgb_sha256', 'base_bytes', 'base_mse', 'encode_process_peak_working_set_bytes',
            'fresh_decode_process_peak_working_set_bytes')}
        kept = []
        for r in im['results']:
            kept.append({k: v for k, v in r.items() if k not in ('stream_path', 'origin', 'fresh_process_verified')})
        top['images'][name]['results'] = kept
    return top


def main() -> None:
    VIS.mkdir(parents=True, exist_ok=True)
    data = json.loads((OUT / 'summary.json').read_text())
    images = data['images']
    report = {'new_detailed_photograph_generalization': {}, 'per_image': {},
              'rate_estimator_accuracy': {}, 'R1_vs_R0': {}, 'mode_map_verification': None}
    for name, im in images.items():
        rows = im['results']
        report['per_image'][name] = {'nondominated_by_config_and_estimator': {},
            'B_vs_prediction_modes': {}, 'D_R1_vs_R0': {}}
        for estimator in ('R0', 'R1'):
            for config in ('B', 'D0', 'D1', 'D'):
                subset = [r for r in rows if r['rate_estimator'] == estimator and r['config'] == config]
                report['per_image'][name]['nondominated_by_config_and_estimator'][f'{estimator}/{config}'] = [
                    {'key':r['key'],'total_bytes':r['total_bytes'],'mse':r['metrics']['mse']} for r in frontier(subset)]
            for config in ('D0', 'D1', 'D'):
                report['per_image'][name]['B_vs_prediction_modes'][f'{estimator}/{config}'] = compare_configs(rows, 'B', config, estimator)
        for config in ('B', 'D0', 'D1', 'D'):
            report['per_image'][name]['D_R1_vs_R0'][config] = compare_estimators(rows, config)
        if name in NEW:
            report['new_detailed_photograph_generalization'][name] = {
                estimator: {config: compare_configs(rows, 'B', config, estimator) for config in ('D0', 'D1', 'D')}
                for estimator in ('R0', 'R1')}
    all_rows = [r for im in images.values() for r in im['results']]
    report['rate_estimator_accuracy'] = estimate_accuracy(all_rows)
    for config in ('B', 'D0', 'D1', 'D'):
        r0 = [r for r in all_rows if r['rate_estimator'] == 'R0' and r['config'] == config]
        r1 = [r for r in all_rows if r['rate_estimator'] == 'R1' and r['config'] == config]
        combined = r0 + r1
        f0, f1 = frontier(r0), frontier(r1)
        report['R1_vs_R0'][config] = {'R0_nondominated_points': len(f0), 'R1_nondominated_points': len(f1),
            'R1_points_strictly_dominated_by_R0': sum(any(a['total_bytes'] <= b['total_bytes'] and a['metrics']['mse'] <= b['metrics']['mse'] and
                (a['total_bytes'] < b['total_bytes'] or a['metrics']['mse'] < b['metrics']['mse']) for a in f0) for b in f1),
            'combined_frontier_keys': [r['key'] for r in frontier(combined)]}
    report['mode_map_verification'] = make_mode_map(images)
    make_rd_plot(images)
    analysis_path = OUT / 'analysis.json'
    analysis_path.write_text(json.dumps(report, indent=2))
    summary = compact(data, report)
    (HERE / 'PBC_V4_RATE_AWARE_PREDICTION_SUMMARY.json').write_text(json.dumps(summary, indent=2) + '\n')
    print('images', len(images), 'candidate rows', len(all_rows), 'heldout', len(NEW),
          'new summary bytes', (HERE / 'PBC_V4_RATE_AWARE_PREDICTION_SUMMARY.json').stat().st_size)
    for name in NEW:
        print('\n', name)
        for estimator in ('R0', 'R1'):
            for config in ('D0', 'D1', 'D'):
                c = report['new_detailed_photograph_generalization'][name][estimator][config]
                p = c['closest_pair_within_5pct_rate'] or c['closest_measured_pair']
                print(estimator, config, 'nearest/near-rate', p['signed_rate_gap_pct'], p['signed_mse_gap_pct'],
                      'threshold', c['screen_threshold_met'])


if __name__ == '__main__':
    main()