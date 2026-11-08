import hashlib
import json
import math
import os
import pathlib
import statistics
import subprocess
import sys
import time
from io import BytesIO

import numpy as np
from PIL import Image, ImageOps

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / 'outputs'
VIS = HERE / 'visuals'
RAW = OUT / 'benchmark_raw.json'
IMAGES = ('googlewiki.png', 'cat.jpg', 'coast.jpg')
V4_CASES = [(name, 'D', point) for name in IMAGES for point in ('low', 'intermediate', 'high')]
V4_CASES += [(name, variant, 'intermediate') for name in IMAGES for variant in ('A', 'B', 'C')]
V31_PRESETS = ('compression', 'balanced', 'quality', 'high_quality')
JPEG_QUALITIES = (15, 30, 45, 60, 75, 90)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def mse(a, b):
    delta = a.astype(np.int32) - b.astype(np.int32)
    return float(np.mean(delta.astype(np.int64) ** 2))


def image_rgb(path):
    with Image.open(path) as image:
        return np.asarray(ImageOps.exif_transpose(image).convert('RGB'), dtype=np.uint8).copy()


def load_raw():
    return json.loads(RAW.read_text()) if RAW.exists() else {
        'schema': 1, 'source_manifest': 'research_pass35/SOURCE_MANIFEST.json',
        'runtime': {}, 'inputs': {}, 'v4': {}, 'v31': {}, 'jpeg': {},
        'boundary_taper': None,
    }


def save(raw):
    RAW.write_text(json.dumps(raw, indent=2, allow_nan=False) + '\n')


def verify_inputs(raw):
    manifest = json.loads((ROOT / 'research_pass35' / 'SOURCE_MANIFEST.json').read_text())
    assets = {item['name']: item for item in manifest['assets']}
    for name in IMAGES:
        path = ROOT / 'hpt_data' / name
        content = path.read_bytes()
        spec = assets[name]
        if sha(content) != spec['sha256'] or len(content) != spec['bytes']:
            raise RuntimeError(f'input manifest mismatch: {name}')
        source = image_rgb(path)
        if [source.shape[1], source.shape[0]] != [spec['width'], spec['height']]:
            raise RuntimeError(f'input dimensions mismatch: {name}')
        raw['inputs'][name] = {
            'path': str(path), 'sha256': spec['sha256'], 'bytes': len(content),
            'width': int(source.shape[1]), 'height': int(source.shape[0]),
        }
    raw['runtime'] = {
        'python': sys.version, 'platform': sys.platform,
        'pillow': Image.__version__ if hasattr(Image, '__version__') else __import__('PIL').__version__,
        'numpy': np.__version__, 'cpu': __import__('platform').processor(),
        'logical_cpu_count': os.cpu_count(), 'ssimulacra2': None,
        'ssimulacra2_status': 'local Pass 39 configured executable absent; metric omitted',
        'jpeg': 'Pillow JPEG encoder; RGB, subsampling=2, optimize=True',
        'v31_commit': '8006f30debfc4a57289df06df21636aeb52baffd',
        'v31_model_sha256': 'ffaf24aa5447b3602c53a5c4c8df14a68ee16e6bded39331d973f38d0f1ea62f',
    }


def run_v4(raw):
    sys.path.insert(0, str(HERE))
    from pbc4_codec import decode, encode
    for name, variant, point in V4_CASES:
        key = f'{name}|{variant}|{point}'
        if key in raw['v4']:
            continue
        src_path = ROOT / 'hpt_data' / name
        source = image_rgb(src_path)
        start = time.perf_counter()
        result = encode(src_path, variant=variant, point=point)
        encode_seconds = time.perf_counter() - start
        stem = pathlib.Path(name).stem + f'_v4_{variant}_{point}'
        stream_path = OUT / f'{stem}.pbc'
        pixels_path = OUT / f'{stem}.png'
        stream_path.write_bytes(result['data'])
        Image.fromarray(result['reconstruction']).save(pixels_path)
        decoded = decode(result['data'])
        if not np.array_equal(decoded, result['reconstruction']):
            raise RuntimeError(f'in-process V4 reconstruction mismatch: {key}')
        fresh_path = OUT / f'{stem}_fresh.png'
        start = time.perf_counter()
        proc = subprocess.run([sys.executable, str(HERE / 'worker_decode.py'),
                               str(stream_path), str(fresh_path)], cwd=ROOT,
                              capture_output=True, text=True)
        fresh_decode_seconds = time.perf_counter() - start
        if proc.returncode:
            raise RuntimeError(f'fresh V4 decode failed {key}: {proc.stderr}')
        fresh_pixels = image_rgb(fresh_path)
        if not np.array_equal(fresh_pixels, result['reconstruction']):
            raise RuntimeError(f'fresh V4 pixels differ: {key}')
        diff_mse = mse(source, fresh_pixels)
        raw['v4'][key] = {
            'image': name, 'variant': variant, 'point': point,
            'input_sha256': raw['inputs'][name]['sha256'],
            'width': int(source.shape[1]), 'height': int(source.shape[0]),
            'bytes': len(result['data']), 'bpp': len(result['data']) * 8 / (source.shape[0] * source.shape[1]),
            'rgb_mse': diff_mse, 'rgb_psnr': float('inf') if diff_mse == 0 else 10 * math.log10(255**2 / diff_mse),
            'encode_seconds': encode_seconds, 'fresh_decode_seconds': fresh_decode_seconds,
            'file_sha256': sha(result['data']), 'pixel_sha256': sha(fresh_pixels.tobytes()),
            'fresh_process_exact_pixels': True, 'stream_path': str(stream_path),
            'pixels_path': str(pixels_path), 'fresh_pixels_path': str(fresh_path),
            'init_candidates': result['init_candidate_results'],
            'selected_init_cap': result['init_cap'], 'mode_counts': result['mode_counts'],
            'byte_accounting': result['byte_accounting'], 'skip_exact': result['skip_exact'],
        }
        save(raw)
        print(f'V4 {key}: {len(result["data"]):,} bytes, MSE {diff_mse:.3f}, encode {encode_seconds:.2f}s', flush=True)


def run_v31(raw):
    for name in IMAGES:
        for preset in V31_PRESETS:
            key = f'{name}|{preset}'
            if key in raw['v31']:
                continue
            prefix = OUT / f'{pathlib.Path(name).stem}_v31_{preset}'
            proc = subprocess.run([sys.executable, str(HERE / 'v31_worker.py'),
                                   str(ROOT / 'hpt_data' / name), preset, str(prefix)],
                                  cwd=ROOT, capture_output=True, text=True)
            if proc.returncode:
                raise RuntimeError(f'V3.1 worker failed {key}:\n{proc.stderr}\n{proc.stdout}')
            row = json.loads(proc.stdout)
            if row['source_sha256'] != raw['inputs'][name]['sha256']:
                raise RuntimeError(f'V3.1 input hash mismatch: {key}')
            third = prefix.with_name(prefix.name + '_r3.pbc')
            fresh_png = prefix.with_name(prefix.name + '_fresh.png')
            start = time.perf_counter()
            fresh = subprocess.run([sys.executable, str(HERE / 'v31_decode_worker.py'),
                                    str(third), str(fresh_png)], cwd=ROOT,
                                   capture_output=True, text=True)
            fresh_seconds = time.perf_counter() - start
            if fresh.returncode:
                raise RuntimeError(f'fresh V3.1 decode failed {key}: {fresh.stderr}')
            fresh_pixels = image_rgb(fresh_png)
            if sha(fresh_pixels.tobytes()) != row['rows'][-1]['pixel_sha256']:
                raise RuntimeError(f'fresh V3.1 pixels differ: {key}')
            row['fresh_process_decode_seconds'] = fresh_seconds
            row['fresh_process_pixel_sha256'] = sha(fresh_pixels.tobytes())
            row['fresh_process_exact_pixels'] = True
            row['streams'] = [str(prefix.with_name(prefix.name + f'_r{n}.pbc')) for n in (1, 2, 3)]
            raw['v31'][key] = row
            save(raw)
            print(f'V3.1 {key}: {row["rows"][-1]["bytes"]:,} bytes, MSE {row["rows"][-1]["rgb_mse"]:.3f}', flush=True)


def run_jpeg(raw):
    for name in IMAGES:
        source = image_rgb(ROOT / 'hpt_data' / name)
        for quality in JPEG_QUALITIES:
            key = f'{name}|q{quality}'
            if key in raw['jpeg']:
                continue
            start = time.perf_counter()
            buffer = BytesIO()
            Image.fromarray(source).save(buffer, format='JPEG', quality=quality,
                                         subsampling=2, optimize=True)
            encode_seconds = time.perf_counter() - start
            data = buffer.getvalue()
            path = OUT / f'{pathlib.Path(name).stem}_jpeg_q{quality}.jpg'
            path.write_bytes(data)
            start = time.perf_counter()
            with Image.open(BytesIO(data)) as compressed:
                decoded = np.asarray(compressed.convert('RGB'), dtype=np.uint8).copy()
            decode_seconds = time.perf_counter() - start
            value = mse(source, decoded)
            raw['jpeg'][key] = {
                'image': name, 'quality': quality, 'bytes': len(data),
                'bpp': len(data) * 8 / (source.shape[0] * source.shape[1]),
                'rgb_mse': value,
                'rgb_psnr': float('inf') if value == 0 else 10 * math.log10(255**2 / value),
                'encode_seconds': encode_seconds, 'decode_seconds': decode_seconds,
                'file_sha256': sha(data), 'pixel_sha256': sha(decoded.tobytes()),
                'file_path': str(path), 'pillow_version': __import__('PIL').__version__,
            }
            save(raw)
            print(f'JPEG {key}: {len(data):,} bytes, MSE {value:.3f}', flush=True)


def run_repeats_and_taper(raw):
    key = 'coast.jpg|D|intermediate'
    source_path = ROOT / 'hpt_data' / 'coast.jpg'
    source = image_rgb(source_path)
    if 'v4_representative_repeats' not in raw:
        sys.path.insert(0, str(HERE))
        from pbc4_codec import decode, encode
        encode(source_path, variant='D', point='intermediate')
        enc, dec = [], []
        for rep in range(3):
            start = time.perf_counter()
            result = encode(source_path, variant='D', point='intermediate')
            enc.append(time.perf_counter() - start)
            stream_path = OUT / f'coast_v4_D_intermediate_repeat{rep + 1}.pbc'
            stream_path.write_bytes(result['data'])
            start = time.perf_counter()
            subprocess.run([sys.executable, str(HERE / 'worker_decode.py'), str(stream_path),
                            str(OUT / f'coast_v4_D_intermediate_repeat{rep + 1}.png')],
                           cwd=ROOT, check=True, capture_output=True, text=True)
            dec.append(time.perf_counter() - start)
        raw['v4_representative_repeats'] = {
            'condition': key, 'warmups': 1, 'encode_seconds': enc,
            'encode_mean': statistics.mean(enc), 'encode_median': statistics.median(enc),
            'fresh_process_decode_seconds': dec,
            'decode_mean': statistics.mean(dec), 'decode_median': statistics.median(dec),
        }
        save(raw)
    if raw.get('boundary_taper') is None and key in raw['v4']:
        from pbc4_stream import pack_file, unpack_file
        record = raw['v4'][key]
        data = pathlib.Path(record['stream_path']).read_bytes()
        header, payload = unpack_file(data)
        tapered_data = pack_file(header['width'], header['height'], header['init_width'],
                                 header['init_height'], header['palette_count'],
                                 header['procedure_step'], header['transform_step'],
                                 header['rd_lambda'], payload, taper=True)
        hard = np.asarray(Image.open(record['pixels_path']).convert('RGB'), dtype=np.uint8)
        tapered = decode(tapered_data)
        grid = np.indices(hard.shape[:2])
        edge = np.zeros(hard.shape[:2], dtype=bool)
        for y in range(16, hard.shape[0], 16):
            edge[max(0, y - 2):min(hard.shape[0], y + 2), :] = True
        for x in range(16, hard.shape[1], 16):
            edge[:, max(0, x - 2):min(hard.shape[1], x + 2)] = True
        # The local diagnostic covers only pixels that actually changed under taper.
        changed = np.any(hard != tapered, axis=2)
        band = edge & changed
        raw['boundary_taper'] = {
            'condition': key, 'same_mode_parameters': True,
            'hard_bytes': len(data), 'taper_bytes': len(tapered_data),
            'hard_rgb_mse': mse(source, hard), 'taper_rgb_mse': mse(source, tapered),
            'changed_pixel_count': int(changed.sum()),
            'grid_edge_changed_pixel_count': int(band.sum()),
            'hard_grid_edge_mse': mse(source[band], hard[band]) if band.any() else None,
            'taper_grid_edge_mse': mse(source[band], tapered[band]) if band.any() else None,
            'taper_pixel_sha256': sha(tapered.tobytes()),
            'selection_note': 'diagnostic only; mode selection was performed for hard boundaries',
        }
        save(raw)


def make_visuals(raw):
    import matplotlib.pyplot as plt
    VIS.mkdir(exist_ok=True)
    for name in IMAGES:
        fig, ax = plt.subplots(figsize=(8, 5))
        for family, rows, marker in (
            ('V4', [r for r in raw['v4'].values() if r['image'] == name], 'o'),
            ('V3.1', [r for r in raw['v31'].values() if r['source'] == name], 's'),
            ('JPEG', [r for r in raw['jpeg'].values() if r['image'] == name], '^'),
        ):
            if family == 'V3.1':
                xx = [row['rows'][-1]['bytes'] * 8 / (row['rows'][-1]['decoded_dimensions'][0] * row['rows'][-1]['decoded_dimensions'][1]) for row in rows]
                yy = [row['rows'][-1]['rgb_mse'] for row in rows]
                labels = [row['preset'] for row in rows]
            else:
                xx = [row['bpp'] for row in rows]
                yy = [row['rgb_mse'] for row in rows]
                labels = [row.get('variant', f"q{row.get('quality')}") + (f' {row.get("point", "")}' if family == 'V4' else '') for row in rows]
            ax.scatter(xx, yy, marker=marker, label=family)
            for x, y, label in zip(xx, yy, labels):
                ax.annotate(label, (x, y), fontsize=7, xytext=(3, 3), textcoords='offset points')
        ax.set(xlabel='bits per pixel', ylabel='RGB MSE', title=f'PBC Pass 42 rate-distortion screen: {name}')
        ax.set_yscale('log')
        ax.grid(True, alpha=.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(VIS / f'rd_{pathlib.Path(name).stem}.png', dpi=150)
        plt.close(fig)
    for name, variant, point in V4_CASES:
        key = f'{name}|{variant}|{point}'
        if key not in raw['v4']:
            continue
        record = raw['v4'][key]
        stream = pathlib.Path(record['stream_path']).read_bytes()
        from pbc4_stream import unpack_file, unpack_bits
        header, payload = unpack_file(stream)
        pos = 1 + header['palette_count'] * 3
        index_bits = max(1, (header['palette_count'] - 1).bit_length())
        index_bytes = (header['init_width'] * header['init_height'] * index_bits + 7) // 8
        pos += index_bytes
        count = ((header['height'] + 15) // 16) * ((header['width'] + 15) // 16)
        modes = np.asarray(unpack_bits(payload[pos:pos + (count * 2 + 7) // 8], count, 2), dtype=np.uint8)
        mode_grid = modes.reshape((header['height'] + 15) // 16, (header['width'] + 15) // 16)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.imshow(mode_grid, interpolation='nearest', vmin=0, vmax=2, cmap='viridis')
        ax.set(title=f"V4-{variant} {point}: 0 skip / 1 procedural / 2 transform", xlabel='16 px regions', ylabel='16 px regions')
        fig.tight_layout()
        fig.savefig(VIS / f"modes_{pathlib.Path(name).stem}_{variant}_{point}.png", dpi=150)
        plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    VIS.mkdir(exist_ok=True)
    raw = load_raw()
    verify_inputs(raw)
    save(raw)
    run_v4(raw)
    run_v31(raw)
    run_jpeg(raw)
    run_repeats_and_taper(raw)
    make_visuals(raw)
    print(f'Raw benchmark: {RAW}', flush=True)


if __name__ == '__main__':
    main()
