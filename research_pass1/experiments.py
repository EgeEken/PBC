import argparse
import csv
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from benchmark import OUT, ROOT, THREAD_ENV, PRESETS

os.environ.update(THREAD_ENV)
parser = argparse.ArgumentParser()
parser.add_argument('mode', choices=('floor', 'profile', 'screen', 'compare'))
args = parser.parse_args()
sys.path.insert(0, str(ROOT if args.mode in ('profile', 'compare') else OUT / 'release'))

import numpy as np
import psutil
from PIL import Image, ImageOps
from PBC3 import PBC3
from pbc3_types import BitReader, BitWriter, PBC3Config
import pbc3_ops as ops

process = psutil.Process()
process.cpu_affinity([process.cpu_affinity()[0]])
SCREEN = ('flower.jpg', 'monet.jpg', 'googlewiki.png')


def save_rows(name, rows):
    (OUT / (name + '.json')).write_text(json.dumps(rows, indent=2))
    if rows:
        keys = list(dict.fromkeys(k for row in rows for k in row if not isinstance(row[k], (list, dict))))
        with open(OUT / (name + '.csv'), 'w', newline='') as output:
            writer = csv.DictWriter(output, fieldnames=keys, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(rows)


def load_image(path):
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert('RGB')


def mse(source, reconstruction):
    assert source.shape == reconstruction.shape
    error = np.subtract(source, reconstruction, dtype=np.float32)
    np.square(error, out=error)
    return float(np.mean(error, dtype=np.float64))


def parse(data):
    _, body = PBC3._open_body(data)
    br = BitReader(body)
    header = PBC3._read_header(br)
    assert not header[12], 'This diagnostic handles default single-stage streams only'
    patches = []
    for _ in range(header[10]):
        c = br.read(header[7])
        x, y, w, h = (br.read(16) for _ in range(4))
        assert br.read(1) == 0
        mask = [br.read(1) for _ in range(br.read(10))]
        neg, pos, max_bits, cell = br.read(8), br.read(8), br.read(4), br.read(16)
        bits = ops.resolve_palette_bitcount(mask, max_bits, neg, pos, header[8])
        indices = PBC3._read_grid(br, ops.ceil_div(w, cell) * ops.ceil_div(h, cell), bits).reshape(ops.ceil_div(h, cell), ops.ceil_div(w, cell))
        patches.append(dict(channel=c, x=x, y=y, w=w, h=h, mask=mask, neg=neg, pos=pos,
                            max_bitcount=max_bits, cell_size=cell, indices=indices, bitcount=bits))
    return header, patches


def serialize(header, patches):
    bw = BitWriter()
    PBC3._write_header(bw, header[3], header[4], header[1], header[2], header[0],
                       PBC3.COLOR_SPACES[header[5]], header[6], header[7], header[8], header[9], len(patches), header[11])
    for patch in patches:
        PBC3._write_patch(bw, patch, header[7])
    raw = bw.finish()
    method, packed = PBC3._entropy_pack(raw)
    return PBC3.MAGIC + bytes((PBC3.VERSION, method)) + packed, len(raw) + 6


def evaluate(data, reference):
    started = time.perf_counter()
    decoded = PBC3.decompress(data)
    seconds = time.perf_counter() - started
    return {'mse': mse(reference, np.asarray(decoded.image.convert('RGB'))),
            'bpp': 8 * len(data) / (reference.shape[0] * reference.shape[1]), 'final_bytes': len(data),
            'decode_seconds': seconds, 'stream_sha256': hashlib.sha256(data).hexdigest()}


def floor():
    baseline = [json.loads(line) for line in (OUT / 'baseline.jsonl').read_text().splitlines()]
    rows = []
    for path in [p for folder in ('hpt_data', 'hpt_data_val') for p in sorted((ROOT / folder).iterdir()) if p.suffix.lower() in ('.jpg', '.jpeg', '.png')]:
        image = load_image(path)
        reference = np.asarray(image)
        for limit in (250000, 500000, 1000000, 2000000):
            cfg = PBC3Config(auto_downsample_max_pixels=limit)
            prep = PBC3.prepare(image, cfg)
            reconstructed = Image.fromarray(prep['arr'], cfg.color_space).convert('RGB')
            reconstructed = reconstructed.resize(image.size, PBC3.RESAMPLE_FILTER)
            value = mse(reference, np.asarray(reconstructed))
            for preset in PRESETS:
                base = next(r for r in baseline if r['source'] == 'release' and r['image'] == path.name and r['preset'] == preset and r['trial'] == 0)
                rows.append({'dataset': path.parent.name, 'image': path.name, 'preset': preset,
                             'width': image.width, 'height': image.height, 'limit': limit,
                             'working_width': prep['w'], 'working_height': prep['h'], 'roundtrip_mse': value,
                             'baseline_mse': base['mse'], 'roundtrip_to_baseline_ratio': value / base['mse']})
        print('floor', path.name, flush=True)
    save_rows('resolution_floor', rows)


def profile():
    rows = []
    for name in SCREEN:
        image = load_image(ROOT / 'hpt_data' / name)
        cfg = PBC3Config.balanced(learned_filler_model_path=str(ROOT / 'patch_policy.npz'))
        PBC3.decompress(PBC3.compress(image, config=cfg).data)
        for repeat in range(2):
            encoded = PBC3.compress(image, config=cfg, trace=True)
            decoded = PBC3.decompress(encoded.data, trace=True)
            for operation, result in (('encode', encoded), ('decode', decoded)):
                trace = result.timings
                (OUT / f'profile_{name}_{repeat}_{operation}.json').write_text(json.dumps(trace))
                for phase, stats in trace['summary'].items():
                    rows.append({'image': name, 'repeat': repeat, 'operation': operation, 'phase': phase,
                                 'total_seconds': trace['total_ns'] / 1e9, 'inclusive_seconds': stats['total_ns'] / 1e9,
                                 'exclusive_seconds': stats['exclusive_ns'] / 1e9, 'count': stats['count']})
        print('profile', name, flush=True)
    save_rows('profile_summary', rows)


def compare():
    rows = []
    for path in sorted((OUT / 'streams' / 'release').glob('*/*.pbc3')):
        relative = path.relative_to(OUT / 'streams' / 'release')
        release_data = path.read_bytes()
        header, release_patches = parse(release_data)
        release_pixels = np.asarray(PBC3.decompress(release_data).image)
        rebuilt, _ = serialize(header, release_patches)
        assert rebuilt == release_data
        for source in ('old', 'current'):
            data = (OUT / 'streams' / source / relative).read_bytes()
            other_header, other = parse(data)
            first = next((i for i, (a, b) in enumerate(zip(release_patches, other))
                          if any(not np.array_equal(a[k], b[k]) for k in a)), None)
            rows.append({'file': str(relative), 'source': source, 'byte_equal': release_data == data,
                         'header_equal': header == other_header, 'first_differing_patch_index': first,
                         'init_patches_equal': all(all(np.array_equal(a[k], b[k]) for k in a) for a, b in zip(release_patches[:3], other[:3])),
                         'decoded_equal': bool(np.array_equal(release_pixels, np.asarray(PBC3.decompress(data).image)))})
    save_rows('stream_comparison', rows)


def screen():
    baseline = [json.loads(line) for line in (OUT / 'baseline.jsonl').read_text().splitlines()]
    rows, singles = [], []
    for name in SCREEN:
        image = load_image(ROOT / 'hpt_data' / name)
        reference = np.asarray(image)
        for preset in PRESETS:
            data = (OUT / 'streams' / 'release' / 'hpt_data' / (name + '.' + preset + '.pbc3')).read_bytes()
            header, patches = parse(data)
            assert serialize(header, patches)[0] == data
            base = evaluate(data, reference)
            baseline_times = [r['encode_seconds'] for r in baseline if r['source'] == 'release' and r['image'] == name and r['preset'] == preset]
            base_encode = float(np.median(baseline_times))
            common = {'image': name, 'preset': preset, 'baseline_mse': base['mse'], 'baseline_bpp': base['bpp'],
                      'baseline_final_bytes': len(data), 'baseline_encode_seconds': base_encode,
                      'baseline_decode_seconds': base['decode_seconds'], 'baseline_raw_bytes': serialize(header, patches)[1]}
            n = len(patches) - header[6]
            for budget in sorted(set((0, 5, 10, 15, 20, n // 2, max(0, n - 5), 30, 40, n))):
                if budget > n:
                    continue
                started = time.perf_counter()
                prefix_data, raw_bytes = serialize(header, patches[:header[6] + budget])
                measured = evaluate(prefix_data, reference)
                cost = time.perf_counter() - started
                rows.append({**common, **measured, 'experiment': 'E_prefix', 'budget': budget,
                             'raw_bytes': raw_bytes, 'extra_seconds': cost, 'encode_seconds': base_encode + cost,
                             'patch_count': header[6] + budget})
                if budget in (max(0, n - 5), n // 2, 10, 20):
                    cfg = getattr(PBC3Config, preset)(patch_count=budget, learned_filler_model_path=str(OUT / 'release' / 'patch_policy.npz'))
                    started = time.perf_counter()
                    result = PBC3.compress(image, config=cfg)
                    encode = time.perf_counter() - started
                    rows.append({**common, **evaluate(result.data, reference), 'experiment': 'E_normal_budget',
                                 'budget': budget, 'raw_bytes': len(PBC3._open_body(result.data)[1]) + 6,
                                 'extra_seconds': 0, 'encode_seconds': encode, 'patch_count': parse(result.data)[0][10]})
            if preset != 'balanced':
                continue
            started = time.perf_counter()
            candidates = []
            for index in range(header[6], len(patches)):
                removed = patches[:index] + patches[index + 1:]
                candidate, raw_bytes = serialize(header, removed)
                measured = evaluate(candidate, reference)
                entry = {**common, **measured, 'experiment': 'C_single_test', 'patch_index': index, 'raw_bytes': raw_bytes}
                singles.append(entry)
                candidates.append((measured, candidate, raw_bytes, index))
            single_cost = time.perf_counter() - started
            for tolerance in (0.0, 0.001):
                eligible = [c for c in candidates if c[0]['mse'] <= base['mse'] * (1 + tolerance) and c[0]['final_bytes'] < len(data)]
                chosen = min(eligible, key=lambda c: (c[0]['final_bytes'], c[0]['mse'])) if eligible else (base, data, serialize(header, patches)[1], -1)
                rows.append({**common, **chosen[0], 'experiment': 'C_single_strict' if tolerance == 0 else 'C_single_tolerance_0.1pct',
                             'raw_bytes': chosen[2], 'removed_patch': chosen[3], 'extra_seconds': single_cost,
                             'encode_seconds': base_encode + single_cost, 'patch_count': len(patches) - int(chosen[3] >= 0)})
            started = time.perf_counter()
            active = list(patches)
            current = dict(base)
            accepted = 0
            for index in range(len(active) - 1, header[6] - 1, -1):
                candidate_patches = active[:index] + active[index + 1:]
                candidate, _ = serialize(header, candidate_patches)
                measured = evaluate(candidate, reference)
                if measured['mse'] <= current['mse'] and measured['final_bytes'] < current['final_bytes']:
                    active, current = candidate_patches, measured
                    accepted += 1
            extra = time.perf_counter() - started
            rows.append({**common, **current, 'experiment': 'C_greedy', 'accepted': accepted,
                         'raw_bytes': serialize(header, active)[1], 'extra_seconds': extra, 'encode_seconds': base_encode + extra,
                         'patch_count': len(active)})
            for variant, count in (('5', 5), ('15', 15), ('all', n)):
                started = time.perf_counter()
                active = list(patches)
                active_data = data
                current = dict(base)
                accepted = 0
                cfg = getattr(PBC3Config, preset)()
                target = PBC3.prepare(image, cfg)['target']
                for index in range(len(active) - 1, max(header[6] - 1, len(active) - count - 1), -1):
                    patch = active[index]
                    canvas = PBC3._decode_to_canvas(active_data)[0]
                    palette = ops.palette_generator(patch['mask'], patch['max_bitcount'], patch['neg'], patch['pos'], header[8])
                    values = palette[patch['indices']]
                    delta = ops.signed_resample(values, patch['h'], patch['w']).astype(np.int32)
                    c, x, y, w, h = (patch[k] for k in ('channel', 'x', 'y', 'w', 'h'))
                    residual = target[y:y + h, x:x + w, c] - (canvas[y:y + h, x:x + w, c] - delta)
                    local_cfg = dataclasses.replace(cfg, mask_size=len(patch['mask']), positive_bias=header[8])
                    refitted, _ = ops.make_patch(c, x, y, w, h, patch['cell_size'], residual, local_cfg, patch['max_bitcount'])
                    proposed = active[:index] + [refitted] + active[index + 1:]
                    candidate, _ = serialize(header, proposed)
                    measured = evaluate(candidate, reference)
                    if measured['mse'] < current['mse'] and measured['final_bytes'] <= current['final_bytes']:
                        active, active_data, current = proposed, candidate, measured
                        accepted += 1
                extra = time.perf_counter() - started
                rows.append({**common, **current, 'experiment': 'D_refit_' + variant, 'accepted': accepted,
                             'patches_revisited': min(count, n),
                             'raw_bytes': serialize(header, active)[1], 'extra_seconds': extra, 'encode_seconds': base_encode + extra,
                             'patch_count': len(active)})
            print('screen', name, preset, flush=True)
            save_rows('screen_results', rows)
            save_rows('pruning_individual_tests', singles)
    save_rows('screen_results', rows)
    save_rows('pruning_individual_tests', singles)


if __name__ == '__main__':
    {'floor': floor, 'profile': profile, 'screen': screen, 'compare': compare}[args.mode]()
