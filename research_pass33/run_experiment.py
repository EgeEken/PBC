from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import psutil
from PIL import Image, ImageOps
from scipy.ndimage import sobel

import pbc33_codec as codec

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
OUT = HERE / 'outputs'
LAMBDAS = (0.0, 20.0, 100.0)
SPARSE_LAMBDAS = (100.0, 400.0, 1600.0)
NAMES = ('googlewiki.png', 'mario64.png', 'flower.jpg', 'cat.jpg', 'cathedral.png', 'park.jpg')
TIMING_REPEATS = 3


def masks(source: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gray = source.astype(np.float32).mean(2)
    magnitude = np.hypot(sobel(gray, axis=0), sobel(gray, axis=1))
    flat = magnitude.reshape(-1)
    edge = magnitude >= np.quantile(flat, .90)
    smooth = magnitude <= np.quantile(flat, .50)
    return edge, smooth


def metrics(source: np.ndarray, rec: np.ndarray, edge: np.ndarray, smooth: np.ndarray) -> dict:
    error = np.square(source.astype(np.int32) - rec.astype(np.int32), dtype=np.int64).sum(2)
    return {'mse': float(error.mean() / 3), 'edge_top10_mse': float(error[edge].mean() / 3),
            'smooth_bottom50_mse': float(error[smooth].mean() / 3)}


def sparse_overhead(config: str, modes: np.ndarray, params: bytes) -> dict:
    cursor = 0
    active_blocks = coeffs = position_bytes = value_bytes = count_bytes = mask_bytes = 0
    by_frequency = [0] * 15
    for mode_v in modes:
        mode = int(mode_v)
        if mode == 1:
            cursor += 3
        elif mode == 2 and config == 'S':
            mask = params[cursor]
            cursor += 1
            mask_bytes += 1
            for block in range(4):
                if mask & (1 << block):
                    count = params[cursor]
                    cursor += 1
                    active_blocks += 1
                    count_bytes += 1
                    for _ in range(count):
                        pos = params[cursor]
                        cursor += 1
                        position_bytes += 1
                        start = cursor
                        _, cursor = codec._read_varint(params, cursor)
                        value_bytes += cursor - start
                        rem = pos % 64
                        u, v = divmod(rem, 8)
                        by_frequency[u + v] += 1
                        coeffs += 1
        elif mode == 3:
            cursor += 48
        elif mode == 4 and config == 'B':
            cursor += 48
    if cursor != len(params):
        raise ValueError('parameter accounting mismatch')
    return {'active_blocks': active_blocks, 'coefficients': coeffs, 'position_bytes': position_bytes,
            'value_bytes': value_bytes, 'count_bytes': count_bytes, 'mask_bytes': mask_bytes,
            'frequency_diagonal_counts': by_frequency}


def modes_for_display(config: str, modes: np.ndarray) -> np.ndarray:
    return modes.copy()


def candidate_rows(source: np.ndarray, base: np.ndarray, name: str, base_info: dict) -> list[dict]:
    edge, smooth = masks(source)
    p31_images = json.loads((ROOT / 'research_pass31' / 'outputs' / 'summary.json').read_text())['images']
    p31 = p31_images[name]['results'] if name in p31_images else None
    rows = []
    families = [('A', 0), ('B', 0)] + [('S', q) for q in codec.QUANT_STEPS]
    for config, qstep in families:
        family = codec.encode_family(source, base, config, SPARSE_LAMBDAS if config == 'S' else LAMBDAS, qstep)
        for result in family:
            data = result['data']
            verification = {}
            rec, modes, params = codec.decode(data, base, verification)
            if (not np.array_equal(rec, result['reconstruction']) or not np.array_equal(modes, result['modes']) or
                params != result['params']):
                raise AssertionError(f'in-process decode mismatch: {name} {config} {qstep} {result["lambda"]}')
            reference = None
            if config in ('A', 'B'):
                if p31 is not None:
                    reference = next(r for r in p31 if r['config'] == config and r['lambda'] == result['lambda'])
                    expected = (reference['enhancement_bytes'], reference['modes_sha256'], reference['params_sha256'], reference['pixels_sha256'])
                else:
                    if str(ROOT / 'research_pass31') not in sys.path:
                        sys.path.insert(0, str(ROOT / 'research_pass31'))
                    import pbc31_codec
                    reference = pbc31_codec.encode(source, base, config, result['lambda'])
                    expected = (len(reference['data']), codec.sha(reference['modes'].tobytes()),
                                codec.sha(reference['params']), codec.sha(reference['reconstruction'].tobytes()))
                actual = (len(data), codec.sha(modes.tobytes()), codec.sha(params), codec.sha(rec.tobytes()))
                if actual != expected:
                    raise AssertionError(f'Pass 31 control mismatch: {name} {config} lambda {result["lambda"]}')
            timed = []
            stage_rows = []
            for _ in range(TIMING_REPEATS):
                profile = {}
                start = time.perf_counter()
                decoded, decoded_modes, decoded_params = codec.decode(data, base, profile)
                timed.append(time.perf_counter() - start)
                stage_rows.append(profile)
                if not np.array_equal(decoded, rec) or not np.array_equal(decoded_modes, modes) or decoded_params != params:
                    raise AssertionError('timed decode changed output')
            overhead = sparse_overhead(config, modes, params)
            pixels = source.shape[0] * source.shape[1]
            total_bytes = base_info['base_bytes'] + len(data)
            row = {'key': f'{config}|q{qstep}|lambda{result["lambda"]:g}', 'image': name,
                'config': config, 'lambda': result['lambda'], 'qstep': qstep,
                'enhancement_bytes': len(data), 'base_bytes': base_info['base_bytes'], 'total_bytes': total_bytes,
                'enhancement_bpp': 8 * len(data) / pixels, 'total_bpp': 8 * total_bytes / pixels,
                'metrics': metrics(source, rec, edge, smooth), 'mode_counts': result['mode_counts'],
                'active_cells': result['active_cells'], 'active_blocks': overhead['active_blocks'],
                'coefficients': overhead['coefficients'],
                'mean_coefficients_per_active_block': overhead['coefficients'] / max(1, overhead['active_blocks']),
                'coefficient_position_bytes': overhead['position_bytes'], 'coefficient_value_bytes': overhead['value_bytes'],
                'block_count_bytes': overhead['count_bytes'], 'cell_mask_bytes': overhead['mask_bytes'],
                'coefficient_frequency_diagonal_counts': overhead['frequency_diagonal_counts'],
                'mode_bytes': result['mode_bytes'], 'parameter_bytes': result['parameter_bytes'],
                'raw_payload_bytes': result['raw_bytes'], 'xz_payload_bytes': result['xz_bytes'],
                'header_bytes': result['header_bytes'], 'estimated_rate_bits': result['rate_bits'],
                'encode_seconds_amortized': result['encode_seconds'], 'family_analysis_seconds': result['family_analysis_seconds'],
                'decode_seconds': {'samples': timed, 'mean': statistics.mean(timed), 'median': statistics.median(timed),
                    'cv_percent': 100 * statistics.pstdev(timed) / statistics.mean(timed) if statistics.mean(timed) else 0,
                    'stages_mean': {key: statistics.mean(p[key] for p in stage_rows) for key in stage_rows[0]}},
                'stream_sha256': codec.sha(data), 'pixels_sha256': codec.sha(rec.tobytes()),
                'modes_sha256': codec.sha(modes.tobytes()), 'params_sha256': codec.sha(params),
                'in_process_verified': True, 'pass31_control_verified': bool(reference)}
            if config == 'S':
                row['active_subblocks'] = overhead['active_blocks']
                row['transmitted_coefficients'] = overhead['coefficients']
            rows.append(row)
            path = OUT / 'streams' / name / f'{config}_q{qstep}_l{result["lambda"]:g}.p33s'
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        del family
    return rows


def verify_fresh(name: str, base_info: dict, rows: list[dict]) -> tuple[list[dict], int]:
    manifest = {'streams': []}
    paths = {}
    for row in rows:
        key = row['key']
        config = row['config']; qstep = row['qstep']; lam = row['lambda']
        stream = OUT / 'streams' / name / f'{config}_q{qstep}_l{lam:g}.p33s'
        manifest['streams'].append({'key': key, 'stream': str(stream), 'base_npy': base_info['base_npy_path']})
        paths[key] = row
    manifest_path = OUT / 'fresh' / f'{name}.json'
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    worker = HERE / 'worker_decode.py'
    env = os.environ.copy()
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
        env[key] = '1'
    proc = subprocess.run([sys.executable, '-B', str(worker), str(manifest_path)], check=True,
                          capture_output=True, text=True, env=env)
    result = json.loads(proc.stdout)
    for check in result['rows']:
        row = paths[check['key']]
        if (row['pixels_sha256'] != check['pixels_sha256'] or row['modes_sha256'] != check['modes_sha256'] or
            row['params_sha256'] != check['params_sha256']):
            raise AssertionError(f'fresh-process mismatch for {name} {check["key"]}')
        row['fresh_process_verified'] = True
        row['fresh_process_decode_profile'] = check['profile']
    return rows, int(result.get('process_peak_working_set_bytes', 0))


def warmup() -> None:
    base = np.full((17, 19, 3), 96, np.uint8)
    source = base.copy()
    source[3:13, 4:15] = 138
    result = codec.encode_family(source, base, 'S', (20.0,), 32)[0]
    codec.decode(result['data'], base)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', nargs='*', choices=NAMES)
    parser.add_argument('--append', action='store_true')
    args = parser.parse_args()
    manifest = json.loads((OUT / 'input_manifest.json').read_text())
    inputs = {row['image']: row for row in manifest['images']}
    names = tuple(args.images) if args.images else NAMES
    warmup()
    proc = psutil.Process()
    initial_summary = OUT / 'summary.json'
    if args.append:
        if not initial_summary.exists():
            raise FileNotFoundError('cannot append before the primary four-image run')
        summary = json.loads(initial_summary.read_text())
        if any(name in summary['images'] for name in names):
            raise ValueError('append image already exists in final summary')
    else:
        summary = {'provenance': {'starting_head': '134869d949d7563691beaf89ce828ef5367306f4', 'python': sys.version,
        'platform': sys.platform, 'thread_limit': 1, 'base_provenance': 'Pass 29 frozen PBC V3.1 Quality streams; decoded arrays from Pass 31',
        'dct': 'SciPy type-II orthonormal DCT/IDCT, float64, axes spatial (0,1), workers=1'},
        'config': {'cell': 16, 'dc_step': 8, 'detail_step': 16, 'sparse_quant_steps': list(codec.QUANT_STEPS),
        'control_lambdas': list(LAMBDAS), 'sparse_lambdas': list(SPARSE_LAMBDAS),
        'mode_ids': {'0': 'SKIP', '1': 'DC', '2': 'SPARSE_DCT', '4': 'B_DETAIL'},
        'sparse': '8x8 RGB DCT, 16 energy/estimated-byte shortlist, exact single-coefficient clipped SSE ranking, at most 8 coefficients per block',
        'xz_format': 'single shared XZ stream, preset 6'}, 'images': {}}
    process_peak_before = proc.memory_info().peak_wset
    for name in names:
        info = inputs[name]
        source = np.asarray(ImageOps.exif_transpose(Image.open(info['source_path'])).convert('RGB'), np.uint8)
        base = np.load(info['base_npy_path'], allow_pickle=False)
        if source.shape != base.shape or codec.sha(Path(info['source_path']).read_bytes()) != info['source_sha256']:
            raise ValueError(f'input changed after manifest creation: {name}')
        rows = candidate_rows(source, base, name, info)
        rows, fresh_peak = verify_fresh(name, info, rows)
        image_peak = int(proc.memory_info().peak_wset)
        image_peak = max(image_peak, process_peak_before)
        image_manifest = {'image': name, 'source_sha256': info['source_sha256'], 'base_stream_sha256': info['base_stream_sha256'],
            'decoded_base_rgb_sha256': info['decoded_base_rgb_sha256'], 'width': info['width'], 'height': info['height'],
            'base_bytes': info['base_bytes'], 'base_mse': info['base_mse'], 'rows': rows,
            'encoder_process_peak_working_set_bytes': image_peak, 'fresh_decoder_process_peak_working_set_bytes': fresh_peak,
            'estimated_family_image_buffers_bytes': int(source.nbytes * 5)}
        summary['images'][name] = image_manifest
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / 'summary.partial.json').write_text(json.dumps(summary, indent=2))
        print('verified', name, len(rows), 'fresh peak MiB', round(fresh_peak / 1048576, 1), flush=True)
        del source, base
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2))
    print('complete candidates', sum(len(x['rows']) for x in summary['images'].values()))


if __name__ == '__main__':
    main()
