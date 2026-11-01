from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps

import pbc33_codec as codec

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
OUT = HERE / 'outputs'
VIS = HERE / 'visuals'
ORDER = ('googlewiki.png', 'mario64.png', 'flower.jpg', 'cat.jpg')
COLORS = {'A': '#5b6573', 'B': '#2878a5', 'S16': '#d44d5c', 'S32': '#e69f00', 'S64': '#168a72'}


def frontier(rows: list[dict]) -> list[dict]:
    kept = []
    for row in sorted(rows, key=lambda r: (r['total_bytes'], r['metrics']['mse'])):
        if not any(other['total_bytes'] <= row['total_bytes'] and other['metrics']['mse'] <= row['metrics']['mse'] and
                   (other['total_bytes'] < row['total_bytes'] or other['metrics']['mse'] < row['metrics']['mse']) for other in rows):
            kept.append(row)
    return kept


def pair(b: dict, s: dict) -> dict:
    return {'baseline': b['key'], 'sparse': s['key'], 'baseline_total_bytes': b['total_bytes'],
        'sparse_total_bytes': s['total_bytes'], 'signed_rate_gap_pct': 100 * (s['total_bytes'] / b['total_bytes'] - 1),
        'baseline_mse': b['metrics']['mse'], 'sparse_mse': s['metrics']['mse'],
        'signed_mse_gap_pct': 100 * (s['metrics']['mse'] / b['metrics']['mse'] - 1),
        'baseline_edge_mse': b['metrics']['edge_top10_mse'], 'sparse_edge_mse': s['metrics']['edge_top10_mse'],
        'baseline_smooth_mse': b['metrics']['smooth_bottom50_mse'], 'sparse_smooth_mse': s['metrics']['smooth_bottom50_mse']}


def compare(rows: list[dict], baseline_config: str) -> dict:
    b = [r for r in rows if r['config'] == baseline_config]
    s = [r for r in rows if r['config'] == 'S']
    pairs = [pair(x, y) for x in b for y in s]
    nearest = min(pairs, key=lambda p: abs(p['signed_rate_gap_pct']))
    near_rate = [p for p in pairs if abs(p['signed_rate_gap_pct']) <= 5]
    near_mse = [p for p in pairs if abs(p['signed_mse_gap_pct']) <= 5]
    best_mse = min(near_rate, key=lambda p: p['signed_mse_gap_pct'], default=None)
    best_rate = min(near_mse, key=lambda p: p['signed_rate_gap_pct'], default=None)
    return {'baseline': baseline_config, 'closest_measured_pair': nearest,
        'best_S_MSE_at_within_5pct_rate': best_mse,
        'best_S_rate_at_within_5pct_MSE': best_rate,
        'mse_gate_met': bool(best_mse and best_mse['signed_mse_gap_pct'] <= -10),
        'byte_gate_met': bool(best_rate and best_rate['signed_rate_gap_pct'] <= -10)}


def percentile_masks(source: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gray = source.astype(np.float32).mean(2)
    from scipy.ndimage import sobel
    mag = np.hypot(sobel(gray, axis=0), sobel(gray, axis=1))
    return mag >= np.quantile(mag, .9), mag <= np.quantile(mag, .5)


def distortion(source: np.ndarray, rec: np.ndarray, edge: np.ndarray, smooth: np.ndarray) -> dict:
    err = np.square(source.astype(np.int32) - rec.astype(np.int32), dtype=np.int64).sum(2)
    return {'mse': float(err.mean() / 3), 'edge_top10_mse': float(err[edge].mean() / 3),
            'smooth_bottom50_mse': float(err[smooth].mean() / 3)}


def active_map(modes: np.ndarray, params: bytes, width: int, height: int) -> np.ndarray:
    blocks = np.zeros((math.ceil(height / 8), math.ceil(width / 8)), np.uint8)
    cols = math.ceil(width / 16)
    cursor = 0
    for index, mode_v in enumerate(modes):
        mode = int(mode_v)
        if mode == 1:
            cursor += 3
        elif mode == 4:
            cursor += 48
        elif mode == 2:
            mask = params[cursor]
            cursor += 1
            cy, cx = divmod(index, cols)
            for block in range(4):
                if mask & (1 << block):
                    count = params[cursor]
                    cursor += 1
                    by, bx = divmod(block, 2)
                    blocks[cy * 2 + by, cx * 2 + bx] = 1
                    for _ in range(count):
                        cursor += 1
                        _, cursor = codec._read_varint(params, cursor)
    if cursor != len(params):
        raise ValueError('active-block map parameter parse mismatch')
    return blocks


def make_rate_distortion(images: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    for ax, name in zip(axes.flat, ORDER):
        rows = images[name]['rows']
        for config in ('A', 'B'):
            points = [r for r in rows if r['config'] == config]
            ax.scatter([r['total_bytes'] / 1000 for r in points], [r['metrics']['mse'] for r in points],
                       color=COLORS[config], marker='o', label=f'{config} actual')
        for step in (16, 32, 64):
            points = [r for r in rows if r['config'] == 'S' and r['qstep'] == step]
            ax.scatter([r['total_bytes'] / 1000 for r in points], [r['metrics']['mse'] for r in points],
                       color=COLORS[f'S{step}'], marker={16: 's', 32: '^', 64: 'D'}[step], label=f'S q={step} actual')
        ax.set_title(name)
        ax.set_xlabel('Total bytes, thousands')
        ax.set_ylabel('Original RGB MSE')
        ax.grid(True, alpha=.25)
        ax.legend(fontsize=7)
    fig.suptitle('PBC V4 Pass 33 — measured complete-file operating points')
    fig.savefig(VIS / 'rate_distortion.png', dpi=170)
    plt.close(fig)


def make_reconstruction_view(images: dict) -> dict:
    name = 'googlewiki.png'
    row_s = next(r for r in images[name]['rows'] if r['key'] == 'S|q64|lambda1600')
    row_b = next(r for r in images[name]['rows'] if r['key'] == 'B|q0|lambda20')
    manifest = json.loads((OUT / 'input_manifest.json').read_text())
    info = next(r for r in manifest['images'] if r['image'] == name)
    source = np.asarray(ImageOps.exif_transpose(Image.open(info['source_path'])).convert('RGB'), np.uint8)
    base = np.load(info['base_npy_path'], allow_pickle=False)
    base_copy = base.copy()
    root = OUT / 'streams' / name
    bdata = (root / 'B_q0_l20.p33s').read_bytes()
    sdata = (root / 'S_q64_l1600.p33s').read_bytes()
    b_rec, _, _ = codec.decode(bdata, base)
    s_rec, s_modes, s_params = codec.decode(sdata, base)
    amap = active_map(s_modes, s_params, source.shape[1], source.shape[0])
    x0, y0, x1, y1 = 16, 192, 528, 416
    fig, axes = plt.subplots(1, 5, figsize=(20, 5), constrained_layout=True)
    views = [(source, 'Original'), (base_copy, 'Frozen PBC base'), (b_rec, f'B: {row_b["total_bytes"]:,} B, MSE {row_b["metrics"]["mse"]:.1f}'),
             (s_rec, f'S: {row_s["total_bytes"]:,} B, MSE {row_s["metrics"]["mse"]:.1f}')]
    for ax, (image, title) in zip(axes[:4], views):
        ax.imshow(image[y0:y1, x0:x1], interpolation='nearest')
        ax.set_title(title)
        ax.axis('off')
    mx0, my0, mx1, my1 = x0 // 8, y0 // 8, x1 // 8, y1 // 8
    axes[4].imshow(amap[my0:my1, mx0:mx1], cmap='Greys', vmin=0, vmax=1,
                   extent=(x0, x1, y1, y0), interpolation='nearest')
    axes[4].set_title('Active 8×8 blocks')
    axes[4].set_xlim(x0, x1); axes[4].set_ylim(y1, y0)
    axes[4].set_xticks([]); axes[4].set_yticks([])
    fig.suptitle('Googlewiki text crop, identical coordinates and pixel scale')
    fig.savefig(VIS / 'sparse_reconstruction_googlewiki.png', dpi=180)
    plt.close(fig)
    return {'image': name, 'crop_xyxy': [x0, y0, x1, y1], 'S_key': row_s['key'], 'B_key': row_b['key'],
        'active_blocks_in_crop': int(amap[my0:my1, mx0:mx1].sum()), 'total_blocks_in_crop': int(amap[my0:my1, mx0:mx1].size)}


def main() -> None:
    VIS.mkdir(parents=True, exist_ok=True)
    data = json.loads((OUT / 'summary.json').read_text())
    image_analysis = {}
    all_rows = []
    for name, image in data['images'].items():
        rows = image['rows']
        all_rows.extend(rows)
        comps = {cfg: compare(rows, cfg) for cfg in ('A', 'B')}
        p = next(r for r in json.loads((OUT / 'input_manifest.json').read_text())['images'] if r['image'] == name)
        source = np.asarray(ImageOps.exif_transpose(Image.open(p['source_path'])).convert('RGB'), np.uint8)
        base = np.load(p['base_npy_path'], allow_pickle=False)
        edge, smooth = percentile_masks(source)
        base_metrics = distortion(source, base, edge, smooth)
        possible_blocks = math.ceil(image['height'] / 8) * math.ceil(image['width'] / 8)
        s_rows = []
        for r in rows:
            if r['config'] == 'S':
                r['possible_8x8_blocks'] = possible_blocks
                r['active_block_fraction'] = r['active_blocks'] / possible_blocks
                r['active_cell_fraction'] = r['active_cells'] / (math.ceil(image['height'] / 16) * math.ceil(image['width'] / 16))
                r['coefficients_per_active_block_of_max8'] = r['coefficients'] / max(1, 8 * r['active_blocks'])
                r['xz_reduction_pct'] = 100 * (1 - r['xz_payload_bytes'] / r['raw_payload_bytes'])
                r['estimated_raw_bits_error'] = r['estimated_rate_bits'] - r['raw_payload_bytes'] * 8
                r['high_frequency_coefficient_fraction_u_plus_v_ge4'] = sum(r['coefficient_frequency_diagonal_counts'][4:]) / max(1, r['coefficients'])
                s_rows.append(r)
        image_analysis[name] = {'content_class': {'googlewiki.png': 'text/web graphics', 'mario64.png': 'synthetic graphics',
            'flower.jpg': 'smooth photography', 'cat.jpg': 'detailed photography', 'cathedral.png': 'architectural facade',
            'park.jpg': 'foliage and bridge photography'}[name],
            'base_metrics': base_metrics, 'comparisons': comps,
            'frontier_all': [{'key': r['key'], 'total_bytes': r['total_bytes'], 'mse': r['metrics']['mse']} for r in frontier(rows)],
            'frontier_S': [{'key': r['key'], 'total_bytes': r['total_bytes'], 'mse': r['metrics']['mse']} for r in frontier(s_rows)],
            'sparse_statistics': s_rows,
            'encoder_peak_working_set_bytes': image['encoder_process_peak_working_set_bytes'],
            'fresh_decoder_peak_working_set_bytes': image['fresh_decoder_process_peak_working_set_bytes'],
            'estimated_family_image_buffers_bytes': image['estimated_family_image_buffers_bytes']}
    primary_names = ('googlewiki.png', 'mario64.png', 'flower.jpg', 'cat.jpg')
    extension_names = tuple(name for name in image_analysis if name not in primary_names)
    gates = {}
    for name in primary_names:
        result = image_analysis[name]
        gates[name] = any(result['comparisons'][b]['mse_gate_met'] or result['comparisons'][b]['byte_gate_met'] for b in ('A', 'B'))
    analysis = {'per_image': image_analysis,
        'progression_gate': {'passed': sum(gates.values()) >= 2, 'images_passing': [name for name, passed in gates.items() if passed],
          'image_gate_results': gates, 'criterion': 'at least two initial content cases with >=10% lower RGB MSE within 5% rate or >=10% fewer bytes within 5% MSE, at measured points'},
        'conditional_extension': {'images': list(extension_names), 'selected_before_pass33_measurements': True,
            'image_gate_results': {name: any(image_analysis[name]['comparisons'][b]['mse_gate_met'] or image_analysis[name]['comparisons'][b]['byte_gate_met'] for b in ('A', 'B')) for name in extension_names}},
        'timing_summary': {'encoder_complete_amortized_seconds': {'median': float(np.median([r['encode_seconds_complete_amortized'] for r in all_rows])),
          'range': [float(min(r['encode_seconds_complete_amortized'] for r in all_rows)), float(max(r['encode_seconds_complete_amortized'] for r in all_rows))]},
          'decoder_seconds_median': {'median': float(np.median([r['decode_seconds']['median'] for r in all_rows])),
          'range': [float(min(r['decode_seconds']['median'] for r in all_rows)), float(max(r['decode_seconds']['median'] for r in all_rows))]},
          'encoder_timing_definition': 'DCT/block analysis family time amortized equally over three lambda outputs, plus median repeated complete stream serialization; not isolated single-point encode timing'},
        'candidate_count': len(all_rows), 'fresh_process_verified_count': sum(bool(r.get('fresh_process_verified')) for r in all_rows),
        'pass31_control_verified_count': sum(bool(r.get('pass31_control_verified')) for r in all_rows),
        'mode_and_parameter_accounting': 'all modes and payloads included; sparse position/value/count/mask bytes separately recorded'}
    plot_mode = make_reconstruction_view(data['images'])
    analysis['reconstruction_visualization'] = plot_mode
    make_rate_distortion(data['images'])
    analysis_path = OUT / 'analysis.json'
    analysis_path.write_text(json.dumps(analysis, indent=2))
    summary = {'provenance': data['provenance'], 'config': data['config'], 'analysis': analysis,
        'images': {name: {k: v for k, v in image.items() if k != 'rows'} | {'rows': image['rows']} for name, image in data['images'].items()}}
    (HERE / 'PBC_V4_SPARSE_DETAIL_SUMMARY.json').write_text(json.dumps(summary, indent=2) + '\n')
    print('candidates', len(all_rows), 'fresh', analysis['fresh_process_verified_count'],
          'controls', analysis['pass31_control_verified_count'], 'gate', analysis['progression_gate']['passed'])
    for name, result in image_analysis.items():
        print('\n', name, result['content_class'])
        for baseline in ('A', 'B'):
            comp = result['comparisons'][baseline]
            print(baseline, comp['closest_measured_pair']['baseline'], 'vs', comp['closest_measured_pair']['sparse'],
                  f"rate {comp['closest_measured_pair']['signed_rate_gap_pct']:+.2f}%",
                  f"MSE {comp['closest_measured_pair']['signed_mse_gap_pct']:+.2f}%",
                  'gate', comp['mse_gate_met'] or comp['byte_gate_met'])


if __name__ == '__main__':
    main()
