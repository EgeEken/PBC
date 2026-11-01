from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from threadpoolctl import threadpool_limits

import pbc32_codec as codec
from rate_model import HistogramRateModel, samples_from_modes

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
OUT = HERE / 'outputs'
R0_LAMBDAS = (0.0, 20.0, 100.0)
R1_LAMBDAS = (0.0, 20.0, 100.0, 400.0, 1600.0)
CONFIGS = ('B', 'D0', 'D1', 'D')


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def metrics(source: np.ndarray, recon: np.ndarray) -> dict:
    diff = source.astype(np.int32) - recon.astype(np.int32)
    per_pixel = np.square(diff, dtype=np.int64).sum(2)
    lum = source.astype(np.float32).mean(2)
    gy, gx = np.gradient(lum)
    mag = np.hypot(gx, gy)
    return {'mse': float(per_pixel.mean() / 3), 'edge_top10_mse': float(per_pixel[mag >= np.quantile(mag, .9)].mean() / 3),
            'smooth_bottom50_mse': float(per_pixel[mag <= np.quantile(mag, .5)].mean() / 3)}


def offset_stats(modes: np.ndarray, params: bytes) -> dict:
    cursor = zero = 0
    counts = {str(m): 0 for m in (1, 4, 5, 6)}
    zero_by = {'5': 0, '6': 0}
    for mode_v in modes:
        mode = int(mode_v)
        n = (3 if mode in (1, 5, 6) else 48 if mode == 4 else 0)
        if mode in counts:
            counts[str(mode)] += 1
        if mode in (5, 6):
            zero_by[str(mode)] += int(not any(params[cursor:cursor + 3]))
        cursor += n
    if cursor != len(params):
        raise ValueError('serialized calibration parameter length mismatch')
    return {'parameter_cells': counts, 'zero_offset_cells': zero_by}


def main() -> None:
    limits = threadpool_limits(limits=1)
    source = np.asarray(ImageOps.exif_transpose(Image.open(ROOT / 'hpt_data/cat.jpg')).convert('RGB'), np.uint8)
    base = np.load(OUT / 'bases/cat.jpg.npy', allow_pickle=False)
    p31 = json.loads((ROOT / 'research_pass31/PBC_V4_PREDICTIVE_REFINEMENT_SUMMARY.json').read_text())
    p31_rows = {r['key']: r for r in p31['images']['cat.jpg']['results']}
    samples, r0_rows = [], []
    (OUT / 'calibration/r0').mkdir(parents=True, exist_ok=True)
    (OUT / 'calibration/r1').mkdir(parents=True, exist_ok=True)

    for config in CONFIGS:
        for lam in R0_LAMBDAS:
            encoded = codec.encode(source, base, config, lam, 'R0')
            digest = sha(encoded['data'])
            key = f'{config}|{lam:g}'
            if config in ('B', 'D'):
                expected = p31_rows[key]['stream_sha256']
                if digest != expected:
                    raise ValueError(f'R0 compatibility check failed against Pass 31: {key}')
            stream = OUT / 'calibration/r0' / f'{config}_{lam:g}.p31r'
            stream.write_bytes(encoded['data'])
            rec, modes, params = codec.decode(stream.read_bytes(), base)
            if not np.array_equal(rec, encoded['reconstruction']) or not np.array_equal(modes, encoded['modes']) or params != encoded['params']:
                raise AssertionError('R0 calibration stream failed exact round-trip')
            samples.append(samples_from_modes(modes, params, source.shape[1], source.shape[0], digest))
            r0_rows.append({'config': config, 'lambda': lam, 'stream_sha256': digest, 'enhancement_bytes': len(encoded['data']),
                            'total_bytes': len(encoded['data']) + 63297, 'metrics': metrics(source, rec),
                            'mode_counts': np.bincount(modes, minlength=7).tolist(), 'rate_estimated_bits': encoded['estimated_bits']})

    model = HistogramRateModel.fit(samples, alpha=.5)
    model_path = OUT / 'calibration/rate_model.json'
    model_hash = model.save(model_path)
    r1_rows = []
    for config in CONFIGS:
        for lam in R1_LAMBDAS:
            start = time.perf_counter()
            encoded = codec.encode(source, base, config, lam, 'R1', model)
            elapsed = time.perf_counter() - start
            path = OUT / 'calibration/r1' / f'{config}_{lam:g}.p31r'
            path.write_bytes(encoded['data'])
            rec, modes, params = codec.decode(path.read_bytes(), base)
            if not np.array_equal(rec, encoded['reconstruction']) or not np.array_equal(modes, encoded['modes']) or params != encoded['params']:
                raise AssertionError('R1 calibration stream failed exact round-trip')
            r1_rows.append({'config': config, 'lambda': lam, 'stream_sha256': sha(encoded['data']),
                            'enhancement_bytes': len(encoded['data']), 'total_bytes': len(encoded['data']) + 63297,
                            'metrics': metrics(source, rec), 'mode_counts': np.bincount(modes, minlength=7).tolist(),
                            'estimated_bits': encoded['estimated_bits'], 'estimated_mode_bits': encoded['estimated_mode_bits'],
                            'estimated_parameter_bits': encoded['estimated_parameter_bits'],
                            'estimated_framing_bits': encoded['estimated_framing_bits'],
                            'encode_seconds': elapsed, **offset_stats(modes, params)})
            print('R1', config, int(lam), 'bytes', len(encoded['data']), 'MSE', round(r1_rows[-1]['metrics']['mse'], 4),
                  'modes', r1_rows[-1]['mode_counts'], flush=True)
    summary = {'purpose': 'Cat-only diagnostic calibration; held-out images were not read by this script.',
               'calibration_image': 'cat.jpg', 'source_sha256': p31['images']['cat.jpg']['source_sha256'],
               'base_stream_sha256': p31['images']['cat.jpg']['base_sha256'], 'base_bytes': 63297,
               'R0_lambdas': list(R0_LAMBDAS), 'R1_lambdas': list(R1_LAMBDAS), 'configs': list(CONFIGS),
               'R0_calibration_rows': r0_rows, 'R1_calibration_rows': r1_rows,
               'rate_model_sha256': model_hash, 'rate_model_samples': len(samples), 'rate_model_alpha': .5,
               'rate_model_path': 'outputs/calibration/rate_model.json',
               'R1_lambda_selection_rule': 'Frozen Cat calibration grid [0,20,100,400,1600]; five values span the rates achieved by the four mode sets. No evaluation-image results were used to choose these points.',
               'fixed_frame_estimate': {'header_bytes': codec.HEADER.size,
                   'empty_xz_bytes': len(__import__('lzma').compress(b'', format=__import__('lzma').FORMAT_XZ, preset=6))}}
    (OUT / 'calibration/calibration.json').write_text(json.dumps(summary, indent=2))
    limits.restore_original_limits()
    print('model', model_hash, 'calibration_rows', len(r0_rows), len(r1_rows), 'seconds', round(time.perf_counter() - start, 3))


if __name__ == '__main__':
    main()