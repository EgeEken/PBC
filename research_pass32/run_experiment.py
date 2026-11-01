from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import psutil
from PIL import Image, ImageOps
from threadpoolctl import threadpool_limits, threadpool_info

import pbc32_codec as codec
from rate_model import HistogramRateModel

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
OUT = HERE / 'outputs'
R0_LAMBDAS = (0.0, 20.0, 100.0)
CONFIGS = ('B', 'D0', 'D1', 'D')
NAMES = ('cat.jpg', 'flower.jpg', 'mario64.png', 'cathedral.png', 'park.jpg', 'seagull.jpg')
MODE_NAMES = ('skip', 'dc', 'vertical_no_offset', 'horizontal_no_offset', 'local_detail', 'vertical_with_offset', 'horizontal_with_offset')
PARAM_LENGTH = (0, 3, 0, 0, 48, 3, 3)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def edge_masks(source: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lum = source.astype(np.float32).mean(2)
    gy, gx = np.gradient(lum)
    mag = np.hypot(gx, gy)
    return mag >= np.quantile(mag, .9), mag <= np.quantile(mag, .5)


def rgb_metrics(source: np.ndarray, recon: np.ndarray, edge: np.ndarray, smooth: np.ndarray) -> dict:
    diff = source.astype(np.int32) - recon.astype(np.int32)
    per_pixel = np.square(diff, dtype=np.int64).sum(2)
    return {'mse': float(per_pixel.mean() / 3), 'edge_top10_mse': float(per_pixel[edge].mean() / 3),
            'smooth_bottom50_mse': float(per_pixel[smooth].mean() / 3)}


def cell_sse(source: np.ndarray, base: np.ndarray, recon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = source.shape[:2]
    rows, cols = math.ceil(h / 16), math.ceil(w / 16)
    hp, wp = rows * 16, cols * 16
    skip_diff = np.zeros((hp, wp, 3), np.int32)
    selected_diff = np.zeros((hp, wp, 3), np.int32)
    skip_diff[:h, :w] = source.astype(np.int32) - base.astype(np.int32)
    selected_diff[:h, :w] = source.astype(np.int32) - recon.astype(np.int32)
    shape = (rows, 16, cols, 16, 3)
    skip = np.square(skip_diff.reshape(shape), dtype=np.int64).sum(axis=(1, 3, 4)).reshape(-1)
    selected = np.square(selected_diff.reshape(shape), dtype=np.int64).sum(axis=(1, 3, 4)).reshape(-1)
    return skip, selected


def parameter_profile(modes: np.ndarray, params: bytes) -> tuple[dict, dict]:
    chunks = [bytearray() for _ in range(7)]
    zero_offsets = {'vertical': 0, 'horizontal': 0}
    cursor = 0
    for mode_v in modes:
        mode = int(mode_v)
        n = PARAM_LENGTH[mode]
        if n:
            chunk = params[cursor:cursor + n]
            if len(chunk) != n:
                raise ValueError('truncated decoded parameter sequence')
            chunks[mode].extend(chunk)
            if mode in (5, 6):
                zero_offsets['vertical' if mode == 5 else 'horizontal'] += int(not any(chunk))
            cursor += n
    if cursor != len(params):
        raise ValueError('unexpected decoded parameters')
    profile = {}
    for mode, raw in enumerate(chunks):
        if not raw:
            continue
        values = np.frombuffer(raw, np.uint8).view(np.int8).astype(np.int16)
        hist = np.bincount(values.astype(np.int16) + 128, minlength=256)
        probs = hist[hist > 0] / values.size
        profile[str(mode)] = {'serialized_bytes': len(raw), 'symbols': int(values.size),
            'distinct_signed_values': int(np.count_nonzero(hist)), 'zero_fraction': float(np.mean(values == 0)),
            'mean_absolute_value': float(np.mean(np.abs(values))), 'min': int(values.min()), 'max': int(values.max()),
            'shannon_bits_per_symbol': float(-(probs * np.log2(probs)).sum())}
    return profile, zero_offsets


def mode_sse(modes: np.ndarray, skip: np.ndarray, selected: np.ndarray) -> dict:
    out = {}
    for mode, name in enumerate(MODE_NAMES):
        mask = modes == mode
        if not np.any(mask):
            out[name] = {'cells': 0, 'skip_sse': 0, 'selected_sse': 0, 'gain_sse': 0}
            continue
        before, after = int(skip[mask].sum()), int(selected[mask].sum())
        out[name] = {'cells': int(mask.sum()), 'skip_sse': before, 'selected_sse': after, 'gain_sse': before - after}
    return out


def get_cached_cat(estimator: str, config: str, lam: float, calibration: dict) -> tuple[Path, dict] | None:
    if estimator == 'R0':
        rows = calibration['R0_calibration_rows']
        key = next((r for r in rows if r['config'] == config and r['lambda'] == lam), None)
        if key is None:
            return None
        return OUT / 'calibration/r0' / f'{config}_{lam:g}.p31r', key
    rows = calibration['R1_calibration_rows']
    key = next((r for r in rows if r['config'] == config and r['lambda'] == lam), None)
    if key is None:
        return None
    return OUT / 'calibration/r1' / f'{config}_{lam:g}.p31r', key


def encode_or_load(name: str, config: str, estimator: str, lam: float, source: np.ndarray, base: np.ndarray,
                   model: HistogramRateModel, calibration: dict, output_dir: Path) -> tuple[bytes, dict]:
    if name == 'cat.jpg' and estimator == 'R1':
        cached = get_cached_cat(estimator, config, lam, calibration)
        if cached is not None:
            path, meta = cached
            data = path.read_bytes()
            if sha(data) != meta['stream_sha256']:
                raise ValueError(f'calibration stream hash mismatch: {name}/{estimator}/{config}/{lam:g}')
            return data, {'encode_seconds': meta.get('encode_seconds'), 'estimated_bits': meta['rate_estimated_bits'] if estimator == 'R0' else meta['estimated_bits'],
                          'estimated_mode_bits': meta.get('estimated_mode_bits'), 'estimated_parameter_bits': meta.get('estimated_parameter_bits'),
                          'estimated_framing_bits': meta.get('estimated_framing_bits', 0), 'origin': 'Cat calibration'}
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f'{estimator}_{config}_{lam:g}.p31r'
    if path.exists():
        path.unlink()
    encoded = codec.encode(source, base, config, lam, estimator, model if estimator == 'R1' else None)
    data = encoded['data']
    path.write_bytes(data)
    return data, {'encode_seconds': encoded['encode_seconds'], 'estimated_bits': encoded['estimated_bits'],
                  'estimated_mode_bits': encoded['estimated_mode_bits'], 'estimated_parameter_bits': encoded['estimated_parameter_bits'],
                  'estimated_framing_bits': encoded['estimated_framing_bits'], 'origin': 'new encoding'}


def fresh_process_check(name: str, base_path: Path, rows: list[dict]) -> tuple[bool, int]:
    request = OUT / 'fresh' / f'{name}.json'
    request.parent.mkdir(parents=True, exist_ok=True)
    streams = [{'key': r['key'], 'stream': r['stream_path']} for r in rows]
    request.write_text(json.dumps({'base_npy': str(base_path.resolve()), 'streams': streams}))
    result = subprocess.run([sys.executable, '-B', str(HERE / 'worker_decode.py'), str(request)], cwd=ROOT,
                            check=True, capture_output=True, text=True)
    output = json.loads(result.stdout)
    by_key = {r['key']: r for r in output['rows']}
    for row in rows:
        got = by_key[row['key']]
        for field in ('pixels_sha256', 'modes_sha256', 'params_sha256'):
            if got[field] != row[field]:
                raise AssertionError(f'fresh process {field} mismatch: {name}/{row["key"]}')
        if got['mode_counts'] != [row['mode_counts'][n] for n in MODE_NAMES]:
            raise AssertionError(f'fresh process mode counts mismatch: {name}/{row["key"]}')
    return True, int(output['process_peak_working_set_bytes'])


def main() -> None:
    limits = threadpool_limits(limits=1)
    points = json.loads((HERE / 'calibration/operating_points.json').read_text())
    model_path = HERE / 'calibration/rate_model_cat.json'
    model, model_hash = HistogramRateModel.load(model_path)
    if model_hash != points['model_sha256']:
        raise ValueError('frozen R1 model hash does not match operating-point manifest')
    r0_lambdas = tuple(map(float, points['rate_models']['R0']['lambdas']))
    r1_lambdas = tuple(map(float, points['rate_models']['R1']['lambdas']))
    calibration = json.loads((OUT / 'calibration/calibration.json').read_text())
    bases = json.loads((OUT / 'manifest.json').read_text())
    base_rows = {r['image']: r for r in bases['images']}
    results = {'provenance': {'starting_head': 'ec52d479399eefed33ca93740b0dd89e44b275f8',
        'python': sys.version.split()[0], 'platform': platform.platform(), 'cpu': platform.processor(),
        'numpy': np.__version__, 'pillow': Image.__version__, 'thread_limit': 1, 'threadpools': threadpool_info(),
        'base_encoder_revision': bases['baseline_revision'], 'model_sha256': model_hash},
        'config': {'cell': 16, 'dc_step': 8, 'detail_step': 16, 'xz_preset': 6,
            'mode_names': dict(enumerate(MODE_NAMES)), 'mode_masks': codec.MASKS,
            'configs': list(CONFIGS), 'rate_models': {'R0': list(r0_lambdas), 'R1': list(r1_lambdas)},
            'R1_cost': 'smoothed conditional mode histogram (previous selected mode × raster boundary class), plus per-mode serialized parameter-slot byte histograms; alpha=0.5; encoder-only; fixed header+empty-XZ framing added to predicted file bits'},
        'images': {}}

    parser = argparse.ArgumentParser()
    parser.add_argument('--images', nargs='*', choices=NAMES, default=list(NAMES))
    selected_names = parser.parse_args().images
    warm_base = np.zeros((16, 16, 3), np.uint8)
    warm_data = codec.encode(warm_base, warm_base, 'B', 0.0, 'R0')['data']
    codec.decode(warm_data, warm_base)
    codec.decode(warm_data, warm_base)

    class_names = {'cat.jpg': 'detailed photography (development)', 'flower.jpg': 'smooth photography',
                   'mario64.png': 'synthetic graphics', 'cathedral.png': 'architecture detail (held-out)',
                   'park.jpg': 'foliage detail (held-out)', 'seagull.jpg': 'water texture (held-out)'}
    for name in selected_names:
        print('image', name, flush=True)
        base_row = base_rows[name]
        source_path = ROOT / 'hpt_data' / name
        source_bytes = source_path.read_bytes()
        source_hash = sha(source_bytes)
        if source_hash != base_row['source_sha256']:
            raise ValueError(f'benchmark source hash changed: {name}')
        source = np.asarray(ImageOps.exif_transpose(Image.open(source_path)).convert('RGB'), np.uint8)
        base_path = Path(base_row['base_npy'])
        base = np.load(base_path, allow_pickle=False)
        if list(source.shape) != base_row['source_shape'] or base.shape != source.shape:
            raise ValueError(f'base dimensions changed: {name}')
        if sha(base.tobytes()) != base_row['decoded_base_sha256']:
            raise ValueError(f'decoded base pixels changed: {name}')
        edge, smooth = edge_masks(source)
        rows = []
        image_stream_dir = OUT / 'streams' / name
        for estimator, lambdas in (('R0', r0_lambdas), ('R1', r1_lambdas)):
            for config in CONFIGS:
                for lam in lambdas:
                    data, meta = encode_or_load(name, config, estimator, lam, source, base, model, calibration, image_stream_dir)
                    stream_path = image_stream_dir / f'{estimator}_{config}_{lam:g}.p31r'
                    if not stream_path.exists():
                        stream_path.write_bytes(data)
                    profile = {}
                    recon, modes, params = codec.decode(data, base, profile)
                    stream_hash = sha(data)
                    if estimator == 'R0' and name == 'cat.jpg':
                        ref = next(r for r in calibration['R0_calibration_rows'] if r['config'] == config and r['lambda'] == lam)
                        if stream_hash != ref['stream_sha256']:
                            raise AssertionError('final R0 differs from Cat calibration stream')
                        if config in ('B', 'D'):
                            p31 = json.loads((ROOT / 'research_pass31/PBC_V4_PREDICTIVE_REFINEMENT_SUMMARY.json').read_text())
                            old = next(r for r in p31['images']['cat.jpg']['results'] if r['key'] == f'{config}|{lam:g}')
                            if stream_hash != old['stream_sha256']:
                                raise AssertionError('R0 B/D stream is not byte-identical to Pass 31')
                    if estimator == 'R1' and name == 'cat.jpg':
                        ref = next(r for r in calibration['R1_calibration_rows'] if r['config'] == config and r['lambda'] == lam)
                        if stream_hash != ref['stream_sha256']:
                            raise AssertionError('final R1 differs from frozen Cat calibration stream')
                    metric = rgb_metrics(source, recon, edge, smooth)
                    skip_cell_sse, selected_cell_sse = cell_sse(source, base, recon)
                    if modes.size != skip_cell_sse.size:
                        raise AssertionError('mode/cell count mismatch')
                    counts = np.bincount(modes, minlength=7).astype(int)
                    param_summary, zero_offsets = parameter_profile(modes, params)
                    mode_distortion = mode_sse(modes, skip_cell_sse, selected_cell_sse)
                    mode_bytes = math.ceil(modes.size * codec.MODE_BITS / 8)
                    raw_len = mode_bytes + len(params)
                    xz_len = len(data) - codec.HEADER.size
                    framing_bits = 8 * (codec.HEADER.size + len(__import__('lzma').compress(b'', format=__import__('lzma').FORMAT_XZ, preset=6)))
                    payload_est_bits = float(meta['estimated_bits']) if estimator == 'R0' else float(meta['estimated_bits'] - meta['estimated_framing_bits'])
                    if estimator == 'R0':
                        estimated_mode_bits = float(modes.size * codec.MODE_BITS)
                        estimated_parameter_bits = float(len(params) * 8)
                        if int(round(estimated_mode_bits + estimated_parameter_bits)) != int(meta['estimated_bits']):
                            raise AssertionError('R0 estimate differs from Pass 31 fixed symbol costs')
                    else:
                        estimated_mode_bits = meta['estimated_mode_bits']
                        estimated_parameter_bits = meta['estimated_parameter_bits']
                    estimated_file_bits = payload_est_bits + framing_bits
                    enhanced_bytes = len(data)
                    base_bytes = int(base_row['base_bytes'])
                    rows.append({'key': f'{estimator}|{config}|{lam:g}', 'rate_estimator': estimator, 'config': config, 'lambda': lam,
                        'enhancement_bytes': enhanced_bytes, 'base_bytes': base_bytes, 'total_bytes': base_bytes + enhanced_bytes,
                        'enhancement_bpp': 8 * enhanced_bytes / (source.shape[0] * source.shape[1]),
                        'total_bpp': 8 * (base_bytes + enhanced_bytes) / (source.shape[0] * source.shape[1]),
                        'metrics': metric, 'mode_counts': {MODE_NAMES[i]: int(counts[i]) for i in range(7)},
                        'detail_cells': int(counts[4]), 'directional_no_offset_cells': int(counts[2] + counts[3]),
                        'directional_offset_cells': int(counts[5] + counts[6]), 'directional_cells': int(counts[2] + counts[3] + counts[5] + counts[6]),
                        'zero_offset_cells': zero_offsets, 'mode_sse': mode_distortion, 'parameter_profile': param_summary,
                        'estimated_bits': float(meta['estimated_bits']), 'estimated_mode_bits': estimated_mode_bits,
                        'estimated_parameter_bits': estimated_parameter_bits, 'estimated_framing_bits': int(meta.get('estimated_framing_bits', 0)),
                        'estimated_file_bits_with_fixed_framing': estimated_file_bits,
                        'estimated_enhancement_bytes': estimated_file_bits / 8,
                        'estimator_error_pct': 100 * (estimated_file_bits / 8 - enhanced_bytes) / enhanced_bytes,
                        'mode_signal_raw_bytes': mode_bytes, 'parameter_raw_bytes': len(params), 'raw_payload_bytes': raw_len,
                        'header_bytes': codec.HEADER.size, 'xz_payload_bytes': xz_len,
                        'xz_reduction_pct': 100 * (1 - xz_len / max(1, raw_len)),
                        'encode_seconds': meta['encode_seconds'], 'decode': profile,
                        'stream_path': str(stream_path.resolve()), 'stream_sha256': stream_hash, 'pixels_sha256': sha(recon.tobytes()),
                        'modes_sha256': sha(modes.tobytes()), 'params_sha256': sha(params),
                        'base_stream_sha256': base_row['stream_sha256'], 'fresh_process_verified': False,
                        'origin': meta['origin'], 'rate_model_sha256': model_hash if estimator == 'R1' else None})
                    print(' ', estimator, config, int(lam), 'bytes', enhanced_bytes, 'MSE', round(metric['mse'], 3), flush=True)

        verified, decode_peak = fresh_process_check(name, base_path, rows)
        for row in rows:
            row['fresh_process_verified'] = verified
            row['fresh_process_decode_peak_working_set_bytes'] = decode_peak
        results['images'][name] = {'class': class_names[name], 'width': source.shape[1], 'height': source.shape[0],
            'source_sha256': source_hash, 'base_stream_sha256': base_row['stream_sha256'],
            'decoded_base_rgb_sha256': base_row['decoded_base_sha256'], 'base_bytes': base_row['base_bytes'],
            'base_mse': base_row['base_mse'], 'encode_process_peak_working_set_bytes': psutil.Process().memory_info().peak_wset,
            'fresh_decode_process_peak_working_set_bytes': decode_peak, 'results': rows}
        partial = OUT / 'summary.partial.json'
        partial.write_text(json.dumps(results, indent=2))
        print('fresh-process verified', name, len(rows), 'rows', flush=True)
    limits.restore_original_limits()
    (OUT / 'summary.json').write_text(json.dumps(results, indent=2))
    print('complete', sum(len(x['results']) for x in results['images'].values()), 'candidate files', flush=True)


if __name__ == '__main__':
    main()