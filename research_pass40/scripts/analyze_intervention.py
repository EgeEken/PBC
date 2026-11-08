import csv
import json
import math
import pathlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
from scipy.ndimage import sobel, uniform_filter

ROOT = pathlib.Path(__file__).resolve().parents[2]
PASS = ROOT / 'research_pass40'
OUT = PASS / 'outputs' / 'intervention'
LEDGER = json.loads((OUT / 'results.json').read_text())
QMAPS = {(r['image'], r['qcolor'], r['config']): r for r in json.loads((PASS / 'outputs' / 'intervention_qmaps.json').read_text())['rows']}
PAIRS = [('googlewiki.png', 5, 3), ('googlewiki.png', 20, 20), ('mario64.png', 50, 50),
         ('flower.jpg', 80, 83), ('cat.jpg', 35, 38), ('cat.jpg', 50, 53), ('coast.jpg', 80, 84)]


def read_rgb(path):
    with Image.open(path) as im:
        return np.asarray(ImageOps.exif_transpose(im).convert('RGB'), dtype=np.uint8)


def masks(source):
    v = source.astype(np.float64) / 255.0
    linear = np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)
    luma = linear @ np.array([0.2126, 0.7152, 0.0722])
    edge_score = np.hypot(sobel(luma, axis=1), sobel(luma, axis=0))
    local_var = np.maximum(uniform_filter(luma * luma, size=8) - uniform_filter(luma, size=8) ** 2, 0)
    edge = edge_score >= np.quantile(edge_score, 0.8)
    remain = ~edge
    texture = remain & (local_var >= np.quantile(local_var[remain], 0.8))
    return {'edge': edge, 'texture': texture, 'smooth': remain & ~texture}


def qindex(image, q, config, shape):
    roots = np.zeros((math.ceil(shape[0] / 64), math.ceil(shape[1] / 64)), dtype=np.uint8)
    path = OUT / QMAPS[(image, q, config)]['csv']
    with path.open(newline='') as f:
        for row in csv.DictReader(f):
            roots[int(row['mi_row']) // 16, int(row['mi_col']) // 16] = int(row['effective_qindex'])
    return np.repeat(np.repeat(roots, 64, axis=0), 64, axis=1)[:shape[0], :shape[1]], roots


def chosen_row(image, q, config):
    choices = [r for r in LEDGER['rows'] if r['image'] == image and r['qcolor'] == q and r['config'] == config and r['phase'] in ('timed', 'warm', 'parity', 'calibration')]
    rank = {'parity': 0, 'warm': 1, 'calibration': 2, 'timed': 3}
    return max(choices, key=lambda r: (rank[r['phase']], r['repeat']))


def summarize_region(mask, error, qmap):
    count = int(mask.sum())
    return {'pixels': count, 'area_fraction': float(mask.mean()),
            'rgb_mse': float(error[mask].sum() / (3 * count)),
            'qindex_mean_area_weighted': float(qmap[mask].mean())}


results = []
for image, ref_q, exp_q in PAIRS:
    source = read_rgb(ROOT / 'hpt_data' / image)
    region_masks = masks(source)
    rows = {}
    maps = {}
    regions = {}
    for config, q in (('REF', ref_q), ('EXP', exp_q)):
        row = chosen_row(image, q, config)
        png = OUT / f"s6_{pathlib.Path(image).stem}_q{q}_{config}_{row['phase']}{row['repeat']}.png"
        recon = read_rgb(png)
        error = np.sum((recon.astype(np.float64) - source.astype(np.float64)) ** 2, axis=2)
        qmap, roots = qindex(image, q, config, source.shape[:2])
        rows[config], maps[config] = row, qmap
        regions[config] = {name: summarize_region(mask, error, qmap) for name, mask in region_masks.items()}
    ref, exp = rows['REF'], rows['EXP']
    gap = abs(exp['bytes'] - ref['bytes']) / ((exp['bytes'] + ref['bytes']) / 2)
    result = {'image': image, 'ref_qcolor': ref_q, 'exp_qcolor': exp_q,
              'ref_phase': ref['phase'], 'exp_phase': exp['phase'],
              'ref_bytes': ref['bytes'], 'exp_bytes': exp['bytes'],
              'rate_gap_percent': gap * 100, 'matched_within_5_percent': gap <= 0.05,
              'ref_rgb_mse': ref['rgb_mse'], 'exp_rgb_mse': exp['rgb_mse'],
              'mse_change_percent': (exp['rgb_mse'] / ref['rgb_mse'] - 1) * 100,
              'ref_ssimulacra2': ref['ssimulacra2'], 'exp_ssimulacra2': exp['ssimulacra2'],
              'ssimulacra2_change': exp['ssimulacra2'] - ref['ssimulacra2'],
              'ref_encode_median_s': float(np.median([r['encode_seconds'] for r in LEDGER['rows'] if r['image'] == image and r['qcolor'] == ref_q and r['config'] == 'REF' and r['phase'] == 'timed'])),
              'exp_encode_median_s': float(np.median([r['encode_seconds'] for r in LEDGER['rows'] if r['image'] == image and r['qcolor'] == exp_q and r['config'] == 'EXP' and r['phase'] in ('timed', 'calibration')])),
              'regions': {name: {'ref': regions['REF'][name], 'exp': regions['EXP'][name],
                                 'mse_change_percent': (regions['EXP'][name]['rgb_mse'] / regions['REF'][name]['rgb_mse'] - 1) * 100,
                                 'qindex_change': regions['EXP'][name]['qindex_mean_area_weighted'] - regions['REF'][name]['qindex_mean_area_weighted']}
                          for name in region_masks}}
    results.append(result)
    print(f"{image:14} q{ref_q}->{exp_q} rate gap {gap*100:.2f}% matched={gap<=.05} MSE {result['mse_change_percent']:+.2f}% S2 {result['ssimulacra2_change']:+.3f}", flush=True)

analysis = {'method': 'fixed P39 masks: linear-light luma; top 20% Sobel magnitude=edge; among remaining pixels top 20% local 8x8 variance=texture; rest=smooth. Each region is one pixel mask used for REF and EXP.',
            'qindex_method': 'byte-parity-checked diagnostic root qindex, expanded to 64x64 pixels; it describes selected qindex, not regional bitrate allocation.',
            'rate_match_rule': 'nearest predeclared measured EXP rate probe to the REF fixed-q rate; no distortion used to select the probe; matched if relative gap <=5%.',
            'pairs': results}
(PASS / 'outputs' / 'intervention_analysis.json').write_text(json.dumps(analysis, indent=2) + '\n')

matched = [r for r in results if r['matched_within_5_percent']]
fig, ax = plt.subplots(figsize=(8.5, 5.5), constrained_layout=True)
for r in results:
    ref_bytes, exp_bytes = r['ref_bytes'], r['exp_bytes']
    ref_mse, exp_mse = r['ref_rgb_mse'], r['exp_rgb_mse']
    ax.plot([ref_bytes, exp_bytes], [ref_mse, exp_mse], color='#888888', alpha=0.55, linewidth=1)
    ax.scatter(ref_bytes, ref_mse, color='#2878b5', s=32, zorder=3)
    ax.scatter(exp_bytes, exp_mse, color='#f28e2b', s=32, zorder=3)
    label = f"{pathlib.Path(r['image']).stem} q{r['ref_qcolor']}"
    ax.annotate(label, (math.sqrt(ref_bytes * exp_bytes), math.sqrt(ref_mse * exp_mse)),
                fontsize=7, xytext=(4, 5), textcoords='offset points')
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([0], [0], marker='o', linestyle='', color='#2878b5', label='REF'),
                   Line2D([0], [0], marker='o', linestyle='', color='#f28e2b', label='EXP')])
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('AVIF bytes (log scale)'); ax.set_ylabel('RGB MSE (log scale)')
ax.set_title('Pass 40 spatial redistribution: development points')
ax.grid(True, which='both', alpha=0.25)
(PASS / 'visuals').mkdir(exist_ok=True)
fig.savefig(PASS / 'visuals' / 'intervention_rd.png', dpi=180)
plt.close(fig)

if matched:
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    labels = [f"{r['image'].split('.')[0]} q{r['ref_qcolor']}" for r in matched]
    vals = [r['mse_change_percent'] for r in matched]
    ax.bar(labels, vals, color=['#b85c5c' if v > 0 else '#4d8a68' for v in vals])
    ax.axhline(0, color='black', linewidth=0.8); ax.set_ylabel('RGB MSE change at matched rate (%)')
    ax.set_title('Rate-matched REF to EXP comparisons'); ax.tick_params(axis='x', rotation=25); ax.grid(axis='y', alpha=0.25)
    fig.savefig(PASS / 'visuals' / 'intervention_matched_mse.png', dpi=180)
    plt.close(fig)
print(f'matched {len(matched)}/{len(results)}')
