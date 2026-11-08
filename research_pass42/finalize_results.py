import json
import math
import pathlib
import statistics

import numpy as np
from PIL import Image

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / 'outputs'
VIS = HERE / 'visuals'
RAW_PATH = OUT / 'benchmark_raw.json'


def rgb(path):
    with Image.open(path) as image:
        return np.asarray(image.convert('RGB'), dtype=np.uint8)


def mse(a, b):
    delta = a.astype(np.int32) - b.astype(np.int32)
    return float(np.mean(delta.astype(np.int64) ** 2))


def main():
    import sys
    import matplotlib.pyplot as plt
    sys.path.insert(0, str(HERE))
    from pbc4_codec import decode
    from pbc4_stream import pack_file, unpack_bits, unpack_file

    raw = json.loads(RAW_PATH.read_text())
    try:
        import threadpoolctl
        blas_pools = threadpoolctl.threadpool_info()
    except ImportError:
        blas_pools = []
    raw['runtime']['blas_thread_pools'] = blas_pools
    raw['runtime']['threading_note'] = 'All measured workers inherited the same environment; OpenBLAS pool is recorded.'
    coast = rgb(ROOT / 'hpt_data' / 'coast.jpg')
    key = 'coast.jpg|D|intermediate'
    record = raw['v4'][key]
    hard_bytes = pathlib.Path(record['stream_path']).read_bytes()
    header, payload = unpack_file(hard_bytes)
    taper_bytes = pack_file(header['width'], header['height'], header['init_width'],
                            header['init_height'], header['palette_count'],
                            header['procedure_step'], header['transform_step'],
                            header['rd_lambda'], payload, taper=True)
    hard = decode(hard_bytes)
    taper, taper_info = decode(taper_bytes, return_info=True)
    pos = 1 + header['palette_count'] * 3
    bits = max(1, (header['palette_count'] - 1).bit_length())
    init_bytes = (header['init_width'] * header['init_height'] * bits + 7) // 8
    pos += init_bytes
    rows, cols = (header['height'] + 15) // 16, (header['width'] + 15) // 16
    count = rows * cols
    mode_data = payload[pos:pos + (count * 2 + 7) // 8]
    modes = np.asarray(unpack_bits(mode_data, count, 2), dtype=np.uint8).reshape(rows, cols)

    boundary_pairs = []
    for row in range(1, rows):
        for col in range(cols):
            if (modes[row - 1, col] == 0) != (modes[row, col] == 0):
                y = row * 16
                x0, x1 = col * 16, min((col + 1) * 16, coast.shape[1])
                if y < coast.shape[0]:
                    boundary_pairs.extend((y, x) for x in range(x0, x1))
    for col in range(1, cols):
        for row in range(rows):
            if (modes[row, col - 1] == 0) != (modes[row, col] == 0):
                x = col * 16
                y0, y1 = row * 16, min((row + 1) * 16, coast.shape[0])
                if x < coast.shape[1]:
                    boundary_pairs.extend((-y - 1, x) for y in range(y0, y1))

    hard_errors, taper_errors, source_edges, smooth_flags = [], [], [], []
    for a, b in boundary_pairs:
        if a >= 0:
            y, x = a, b
            source_delta = coast[y, x].astype(np.int16) - coast[y - 1, x].astype(np.int16)
            hard_delta = hard[y, x].astype(np.int16) - hard[y - 1, x].astype(np.int16)
            taper_delta = taper[y, x].astype(np.int16) - taper[y - 1, x].astype(np.int16)
        else:
            y, x = -a - 1, b
            source_delta = coast[y, x].astype(np.int16) - coast[y, x - 1].astype(np.int16)
            hard_delta = hard[y, x].astype(np.int16) - hard[y, x - 1].astype(np.int16)
            taper_delta = taper[y, x].astype(np.int16) - taper[y, x - 1].astype(np.int16)
        source_edges.append(source_delta)
        hard_errors.append(hard_delta - source_delta)
        taper_errors.append(taper_delta - source_delta)
        smooth_flags.append(int(np.max(np.abs(source_delta))) <= 8)
    hard_errors = np.asarray(hard_errors, dtype=np.int32)
    taper_errors = np.asarray(taper_errors, dtype=np.int32)
    source_edges = np.asarray(source_edges, dtype=np.int32)
    smooth = np.asarray(smooth_flags, dtype=bool)
    raw['boundary_taper'] = {
        'condition': key, 'same_mode_parameters': True,
        'hard_bytes': len(hard_bytes), 'taper_bytes': len(taper_bytes),
        'hard_rgb_mse': mse(coast, hard), 'taper_rgb_mse': mse(coast, taper),
        'protected_skip_exact': bool(taper_info['skip_exact']),
        'changed_pixel_count': int(np.any(hard != taper, axis=2).sum()),
        'skip_adjacent_region_boundary_pairs': int(len(boundary_pairs)),
        'boundary_gradient_error_mse': {
            'hard': float(np.mean(hard_errors.astype(np.int64) ** 2)),
            'taper': float(np.mean(taper_errors.astype(np.int64) ** 2)),
        },
        'smooth_boundary_gradient_error_mse_source_max_channel_step_le_8': {
            'sample_count': int(smooth.sum()),
            'hard': float(np.mean(hard_errors[smooth].astype(np.int64) ** 2)) if smooth.any() else None,
            'taper': float(np.mean(taper_errors[smooth].astype(np.int64) ** 2)) if smooth.any() else None,
        },
        'selection_note': 'diagnostic only; mode selection was performed for hard boundaries',
    }
    (OUT / 'benchmark_raw.json').write_text(json.dumps(raw, indent=2, allow_nan=False) + '\n', encoding='utf-8')

    source_path = ROOT / 'hpt_data' / 'googlewiki.png'
    source = rgb(source_path)
    v31 = rgb(OUT / 'googlewiki_v31_high_quality_fresh.png')
    init = rgb(OUT / 'googlewiki_v4_A_intermediate_fresh.png')
    v4 = rgb(OUT / 'googlewiki_v4_D_intermediate_fresh.png')
    jpeg = rgb(OUT / 'googlewiki_jpeg_q45.jpg')
    modes_g = np.asarray(Image.open(VIS / 'modes_googlewiki_D_intermediate.png').convert('RGB'))
    panels = [
        ('Original', source), ('V3.1 High Quality', v31), ('V4 Init Only', init),
        ('V4-D Intermediate', v4), ('JPEG q45 (~matched rate)', jpeg),
        ('V4-D mode map', np.asarray(Image.open(VIS / 'modes_googlewiki_D_intermediate.png').convert('RGB'))),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, (title, pixels) in zip(axes.ravel(), panels):
        ax.imshow(pixels, interpolation='nearest')
        ax.set_title(title)
        ax.axis('off')
    fig.suptitle('Googlewiki source-aligned comparison; JPEG q45 rate is within 1% of V4-D intermediate')
    fig.tight_layout()
    fig.savefig(VIS / 'comparison_googlewiki.png', dpi=160)
    plt.close(fig)

    err = np.max(np.abs(source.astype(np.int16) - v4.astype(np.int16)), axis=2)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(err, cmap='inferno', vmin=0, vmax=32)
    ax.set_title('Googlewiki V4-D intermediate absolute RGB error (max channel, display clipped at 32)')
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(VIS / 'error_googlewiki_v4_D_intermediate.png', dpi=160)
    plt.close(fig)

    left, top = 1100, 700
    right, bottom = left + 1200, top + 800
    coast_crop = coast[top:bottom, left:right]
    hard_crop = hard[top:bottom, left:right]
    taper_crop = taper[top:bottom, left:right]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, (title, pixels) in zip(axes.ravel(), [
        ('Source crop', coast_crop), ('Hard boundaries', hard_crop),
        ('Conditional taper', taper_crop), ('Taper − hard, ×4 display', np.clip(128 + (taper_crop.astype(np.int16) - hard_crop.astype(np.int16)) * 4, 0, 255).astype(np.uint8)),
    ]):
        ax.imshow(pixels, interpolation='nearest')
        ax.set_title(title)
        ax.axis('off')
    fig.suptitle('Coast boundary diagnostic (same selected modes and parameters)')
    fig.tight_layout()
    fig.savefig(VIS / 'boundary_coast_crop.png', dpi=160)
    plt.close(fig)

    summary = {
        'status': 'exploratory Pass 42 prototype; no release claim',
        'inputs': raw['inputs'], 'runtime': raw['runtime'],
        'condition_counts': {'v4': len(raw['v4']), 'frozen_v31': len(raw['v31']), 'jpeg': len(raw['jpeg'])},
        'correctness': {'synthetic_tests_passed': 6, 'fresh_process_v4_exact_all_points': all(x['fresh_process_exact_pixels'] for x in raw['v4'].values()),
                        'fresh_process_v31_exact_all_points': all(x['fresh_process_exact_pixels'] for x in raw['v31'].values()),
                        'skip_exact': all(x['skip_exact'] for x in raw['v4'].values()),
                        'malformed_stream_rejection_tested': True},
        'boundary_taper': raw['boundary_taper'],
        'v4_representative_repeats': raw['v4_representative_repeats'],
        'memory_profile': json.loads((OUT / 'large_screen.json').read_text()),
        'gate_assessment': {
            'A_correctness': 'pass: six synthetic test groups and fresh-process exact-pixel checks across all measured streams',
            'B_quality_scaling': 'pass: all three V4-D curves reduce RGB MSE as bitrate increases; highest points approach lossless detail',
            'C_rate_distortion_vs_v31': 'fail as broad improvement: V4 improves fidelity at higher rates but does not preserve V3.1 low-rate behavior or show a comparable-rate general win',
            'D_speed_memory': 'fail for current implementation: coast V4-D intermediate ~34.15 s encode and ~11.83 s fresh-process decode; high points reach ~144–193 s encode. V3.1 is subsecond in these controls. A low-point memory profile peaked near 746 MiB.',
            'E_jpeg': 'mixed and mostly unfavorable: Googlewiki intermediate is a strong rate-matched win; cat low point is dominated by JPEG and detailed-photo refinement costs far more rate; coast JPEG q30 is close in quality at much lower rate than V4 intermediate.',
            'F_reserved_validation': 'not run: development evidence does not justify opening reserved images',
        },
        'caveats': [
            'Development set has three images and is not held out.',
            'SSIMULACRA2 was omitted because the established local executable path is absent.',
            'Pillow JPEG is pinned by the current runtime (Pillow 11.3.0); JPEG quality settings are compared by measured bytes and decoded MSE.',
            'Fresh-process decode wall time includes Python startup and is not directly equivalent to V3.1 in-process timing; raw fresh-process V3.1 values are retained.',
            'Boundary taper uses hard-selected modes and is diagnostic only; it is not a separately optimized codec point.',
            'No AVIF re-encode or held-out validation was run.',
        ],
        'files': {'raw': str(RAW_PATH), 'report': str(HERE / 'V4_CODEC_RESULTS.md'), 'visuals': str(VIS)},
    }
    summary['runtime']['blas_thread_pools'] = blas_pools
    summary['runtime']['threading_note'] = raw['runtime']['threading_note']
    (HERE / 'PASS42_SUMMARY.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n', encoding='utf-8')

    rows = []
    for name in ('googlewiki.png', 'cat.jpg', 'coast.jpg'):
        rows.append(f"### {name}\n\n| Codec point | Bpp | RGB MSE | Encode s | Decode s* |\n|---|---:|---:|---:|---:|")
        for point in ('low', 'intermediate', 'high'):
            r = raw['v4'][f'{name}|D|{point}']
            rows.append(f"| V4-D {point} | {r['bpp']:.4f} | {r['rgb_mse']:.3f} | {r['encode_seconds']:.2f} | {r['fresh_decode_seconds']:.2f} |")
        for preset in ('compression', 'balanced', 'quality', 'high_quality'):
            r = raw['v31'][f'{name}|{preset}']['rows'][-1]
            bpp = r['bytes'] * 8 / (r['decoded_dimensions'][0] * r['decoded_dimensions'][1])
            v31_record = raw['v31'][f'{name}|{preset}']
            rows.append(f"| Frozen V3.1 {preset} | {bpp:.4f} | {r['rgb_mse']:.3f} | {statistics.mean(x['encode_seconds'] for x in v31_record['rows']):.2f} | {v31_record['fresh_process_decode_seconds']:.2f} |")
        for quality in (15, 30, 45, 60, 75, 90):
            r = raw['jpeg'][f'{name}|q{quality}']
            rows.append(f"| JPEG q{quality} | {r['bpp']:.4f} | {r['rgb_mse']:.3f} | {r['encode_seconds']:.2f} | {r['decode_seconds']:.3f} |")
        rows.append('')
    report = '''# PBC V4 Pass 42 results\n\n## Scope\n\nThis is a first isolated V4-A prototype, not a release candidate. It uses the fixed 16×16 regional grid, compact 64-color initialization, protected SKIP, procedural residual surfaces, full 8×8 RGB DCT refinement, and one shared XZ payload. Measurements use the three specified development images; reserved validation images were not opened. The frozen V3.1 source commit and model hash are recorded in the raw results.\n\n## Quality and rate\n\nThe complete measurements follow. Rates are actual file bytes divided by source pixels. No interpolation or extrapolation is used.\n\n''' + '\n'.join(rows) + '''\n## Findings\n\n- V4-D improves monotonically from low through high on all three images. The highest setting approaches source fidelity, so the V3 working-resolution quality ceiling is not inherent to this representation. It does so at very high rates: 7.73 bpp on Googlewiki, 14.00 bpp on cat, and 9.67 bpp on coast.\n- It does not preserve V3.1’s extreme-low-rate behavior. On coast, V3.1 compression is about 0.0043 bpp and MSE 73.1; V4-D low is 0.0925 bpp and MSE 18.7. This is a quality gain at roughly 21× the rate.\n- JPEG remains a strong control. Googlewiki V4-D intermediate is 1.295 bpp / MSE 15.62, while JPEG q45 is 1.287 bpp / MSE 325.45, a clear rate-matched V4 win on this graphics image. Cat’s V4-D low point is 0.731 bpp / MSE 167.37, while JPEG q45 is 0.727 bpp / MSE 27.39, so JPEG dominates there. On coast, JPEG q30 is 0.099 bpp / MSE 7.87; V4-D intermediate is 0.459 bpp / MSE 7.16, similar distortion at roughly 4.6× the rate.\n- Ablations show content dependence. On Googlewiki, intermediate DCT-only reaches MSE 15.74 and the combined D point 15.62 at similar rates, so procedural adds little there. On cat, DCT-only and combined are nearly tied (10.05 vs 10.11 MSE), with combined slightly smaller. On coast, procedural-only is 0.161 bpp / MSE 15.71, DCT-only is 0.660 bpp / 16.70, and combined is 0.459 bpp / 7.16; procedural coding is useful and complementary on that image.\n- Joint complete-file initialization selection evaluates both caps for each point. Per-cap details are in each raw V4 record. This bounded comparison does not establish that higher initialization resolution is generally preferable; the selected cap varies with image and operating point.\n- Conditional taper did not improve the coast diagnostic. With identical hard-selected modes and parameters, file size stayed at 688,860 bytes, global MSE rose from 7.162 to 7.287, and SKIP preservation remained exact. The error in measured SKIP-adjacent boundary gradients and the smooth-gradient subset is listed in `PASS42_SUMMARY.json`; no perceptual seam improvement is claimed.\n\n## Performance and correctness\n\nOn the representative coast V4-D intermediate point, three warmed encodes averaged 34.15 s (median 34.15 s); three fresh-process decodes averaged 11.83 s (median 11.87 s). The profile of the coast low point peaked at 782,675,968 bytes working set (~746 MiB). Frozen V3.1 measured encode/decode calls are generally subsecond/under 0.1 s in this runtime. The V4 prototype is therefore substantially slower and heavier. Fresh-process decode includes interpreter startup.\n\nSix synthetic test groups pass, covering odd dimensions, synthetic content, ablations, mode mixtures, deterministic file-only decode, exact protected SKIP and malformed streams. All 18 measured V4 streams and 12 V3.1 conditions decoded in a fresh process with exact pixel agreement.\n\nSSIMULACRA2 was omitted because the configured local executable is not present. No AVIF comparison or reserved validation was run. JPEG timing uses Pillow 11.3.0 with RGB, 4:2:0 subsampling and optimization enabled.\n\n## Decision\n\nThe prototype demonstrates original-resolution quality scaling and has a clear graphics-specific rate-distortion result. It fails the broader V4 objective in its current form: it does not retain low-rate V3.1 behavior, its photographic rates are poor against JPEG, and encode/decode cost is high. Stop parameter expansion here. The next useful research decision is whether to redesign the region allocator/transform representation for photographic rate efficiency and decoder cost before opening held-out images.\n\nRaw rows, initialization-candidate statistics, stream hashes, timings, decode hashes and artifact paths are in `outputs/benchmark_raw.json`. Visual comparisons and mode maps are in `visuals/`.\n'''
    report = report.replace(
        '## Findings',
        '## Findings')
    report = report.replace(
        'The complete measurements follow. Rates are actual file bytes divided by source pixels. No interpolation or extrapolation is used.',
        'The complete measurements follow. Rates are actual file bytes divided by source pixels. No interpolation or extrapolation is used. Decode times marked * are fresh-process measurements that include interpreter startup; JPEG decode is measured in-process through Pillow.')
    report = report.replace(
        'Conditional taper did not improve the coast diagnostic. With identical hard-selected modes and parameters, file size stayed at 688,860 bytes, global MSE rose from 7.162 to 7.287, and SKIP preservation remained exact. The error in measured SKIP-adjacent boundary gradients and the smooth-gradient subset is listed in `PASS42_SUMMARY.json`; no perceptual seam improvement is claimed.',
        'Conditional taper had a mixed result on the coast diagnostic. With identical hard-selected modes and parameters, file size stayed at 688,860 bytes and SKIP preservation remained exact. Global MSE rose from 7.162 to 7.287, while error in SKIP-adjacent boundary gradients fell from 21.74 to 7.23 (19.55 to 5.91 on source-smooth boundary samples). This is a local numerical improvement with a small global regression; no perceptual seam improvement is claimed.')
    report = report.replace(
        'Fresh-process decode includes interpreter startup.',
        'Fresh-process decode includes interpreter startup. NumPy used OpenBLAS 0.3.29 with a 20-thread pool; all measured workers inherited the same environment.')
    (HERE / 'V4_CODEC_RESULTS.md').write_text(report, encoding='utf-8')


if __name__ == '__main__':
    main()
