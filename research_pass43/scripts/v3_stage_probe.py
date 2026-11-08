import hashlib
import json
import math
import os
import pathlib
import sys

import numpy as np
from PIL import Image, ImageOps

HERE = pathlib.Path(__file__).resolve().parents[1]
ROOT = HERE.parent
OUT = HERE / 'outputs'
FROZEN = ROOT / 'research_pass42' / 'outputs' / 'v31_frozen'
IMAGES = ('googlewiki.png', 'cat.jpg', 'coast.jpg')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def rgb_mse(a, b):
    delta = a.astype(np.int32) - b.astype(np.int32)
    return float(np.mean(delta.astype(np.int64) ** 2))


def source_rgb(path):
    with Image.open(path) as im:
        return np.asarray(ImageOps.exif_transpose(im).convert('RGB'), dtype=np.uint8).copy()


def build_prefix(PBC3, BitWriter, config, prep, patches, count, warmup_split):
    h, w, channels = prep['h'], prep['w'], prep['channels']
    channel_bits = max(1, math.ceil(math.log2(channels)))
    bases = [int(round(float(np.mean(prep['arr'][:, :, c])))) for c in range(channels)]
    warmup = None
    if warmup_split is not None and count > warmup_split:
        warmup = (prep['warm_w'], prep['warm_h'], warmup_split)
    bw = BitWriter()
    PBC3._write_header(
        bw, w, h, prep['original_w'], prep['original_h'], prep['downsampled'],
        prep['color_id'], channels, channel_bits, config.positive_bias,
        prep['has_alpha'], count, bases, warmup=warmup,
    )
    for patch in patches[:count]:
        PBC3._write_patch(bw, patch, channel_bits)
    raw_body = bw.finish()
    method, packed = PBC3._entropy_pack(raw_body, config.use_lzma)
    return PBC3.MAGIC + bytes([PBC3.VERSION, method]) + packed, len(raw_body)


def main():
    OUT.mkdir(exist_ok=True)
    pass42 = json.loads((ROOT / 'research_pass42' / 'outputs' / 'benchmark_raw.json').read_text())
    manifest = json.loads((ROOT / 'research_pass35' / 'SOURCE_MANIFEST.json').read_text())
    manifest_assets = {x['name']: x for x in manifest['assets']}
    os.chdir(FROZEN)
    sys.path.insert(0, str(FROZEN))
    import pbc3_heads
    import pbc3_ops
    from PBC3 import PBC3
    from pbc3_types import BitReader, BitWriter, PBC3Config

    rows = {}
    for name in IMAGES:
        path = ROOT / 'hpt_data' / name
        source = source_rgb(path)
        pil_source = Image.fromarray(source)
        spec = manifest_assets[name]
        if sha(path.read_bytes()) != spec['sha256'] or pass42['inputs'][name]['sha256'] != spec['sha256']:
            raise RuntimeError(f'input hash mismatch: {name}')
        cfg = PBC3Config.compression()
        init_patches, refine_patches = [], []
        old_init = pbc3_heads.DownsampleInitHead.select
        old_fill = pbc3_heads.FillerHead.select

        def capture_init(self, *args, **kwargs):
            result = old_init(self, *args, **kwargs)
            init_patches.append(result[0])
            return result

        def capture_fill(self, *args, **kwargs):
            result = old_fill(self, *args, **kwargs)
            if result[0] is not None:
                refine_patches.append(result[0])
            return result

        pbc3_heads.DownsampleInitHead.select = capture_init
        pbc3_heads.FillerHead.select = capture_fill
        try:
            final = PBC3.compress(pil_source, config=cfg, rng_mode='release')
        finally:
            pbc3_heads.DownsampleInitHead.select = old_init
            pbc3_heads.FillerHead.select = old_fill
        patches = init_patches + refine_patches
        _, body = PBC3._open_body(final.data)
        decoded_header = PBC3._read_header(BitReader(body))
        (downsampled, original_w, original_h, w, h, color_space, channels,
         channel_bits, positive_bias, has_alpha, total_patches, base_values,
         warmup_on, warm_w, warm_h, warmup_split) = decoded_header
        if len(patches) != total_patches or len(init_patches) != channels:
            raise RuntimeError(f'captured patch count mismatch for {name}')
        prep = PBC3.prepare(pil_source, cfg)
        if prep['warmup_patches'] is None or not warmup_on:
            warmup_split = None
        full_hash = sha(final.data)
        old_v31_path = ROOT / 'research_pass42' / 'outputs' / f'{pathlib.Path(name).stem}_v31_compression_r3.pbc'
        if old_v31_path.exists() and sha(old_v31_path.read_bytes()) != full_hash:
            raise RuntimeError(f'captured V3.1 run differs from Pass 42 frozen baseline: {name}')

        desired = [0, len(init_patches), len(init_patches) + 1,
                   len(init_patches) + 3, len(init_patches) + 10, total_patches]
        counts = sorted({max(0, min(total_patches, n)) for n in desired})
        channel_bits = max(1, math.ceil(math.log2(channels)))
        patch_metadata = []
        for index, patch in enumerate(patches):
            patch_metadata.append({
                'stage': 'init' if index < len(init_patches) else 'refinement',
                'patch_index': index,
                'channel': int(patch['channel']), 'x': int(patch['x']), 'y': int(patch['y']),
                'width': int(patch['w']), 'height': int(patch['h']),
                'cell_size': int(patch['cell_size']), 'bitcount': int(patch['bitcount']),
                'grid_dimensions': [int(patch['indices'].shape[1]), int(patch['indices'].shape[0])],
                'raw_patch_bits': int(pbc3_ops.patch_bits_for(patch, channel_bits)),
            })
        stages = []
        for count in counts:
            stream, raw_bytes = build_prefix(PBC3, BitWriter, cfg, prep, patches, count, warmup_split)
            result = PBC3.decompress(stream)
            pixels = np.asarray(result.image.convert('RGB'), dtype=np.uint8)
            value = rgb_mse(source, pixels)
            rel = pathlib.Path('outputs') / f'{pathlib.Path(name).stem}_v31_prefix_{count:03d}.pbc'
            stream_path = HERE / rel
            stream_path.write_bytes(stream)
            image_path = stream_path.with_suffix('.png')
            Image.fromarray(pixels).save(image_path)
            patch_bits = sum(pbc3_ops.patch_bits_for(p, max(1, math.ceil(math.log2(channels)))) for p in patches[:count])
            stages.append({
                'patch_count': count, 'init_patches': min(count, len(init_patches)),
                'refinement_patches': max(0, count - len(init_patches)),
                'bytes': len(stream), 'bpp': len(stream) * 8 / (source.shape[0] * source.shape[1]),
                'raw_body_bytes': raw_bytes, 'raw_patch_bits_estimate': patch_bits,
                'rgb_mse': value, 'rgb_psnr': float('inf') if value == 0 else 10 * math.log10(255**2 / value),
                'file_sha256': sha(stream), 'pixel_sha256': sha(pixels.tobytes()),
                'stream_path': str(stream_path), 'image_path': str(image_path),
                'warmup_applied': warmup_split is not None and count > warmup_split,
            })
        rows[name] = {
            'source_sha256': spec['sha256'], 'dimensions': [int(source.shape[1]), int(source.shape[0])],
            'frozen_source_commit': '8006f30debfc4a57289df06df21636aeb52baffd',
            'model_sha256': sha((FROZEN / 'patch_policy.npz').read_bytes()),
            'config': cfg.__dict__, 'working_dimensions': [w, h],
            'downsampled': downsampled, 'warmup_on': warmup_on,
            'warmup_split_stream_patch_index': warmup_split,
            'init_patch_count': len(init_patches), 'refinement_patch_count': len(refine_patches),
            'patch_metadata': patch_metadata,
            'final_stream_sha256': full_hash,
            'pass42_frozen_v31_compression_stream_sha256': sha(old_v31_path.read_bytes()) if old_v31_path.exists() else None,
            'stages': stages,
            'full_v31_record': pass42['v31'][f'{name}|compression'],
            'attribution_note': 'Prefix files are independently entropy-packed complete PBC3 files. Raw patch bits are a field-size estimate; post-LZMA bytes are measured from each prefix file.'
        }
        (OUT / 'v3_stage_probe.json').write_text(json.dumps(rows, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        print(f"{name}: init-only {stages[1]['bytes']:,} B / MSE {stages[1]['rgb_mse']:.3f}; full {len(final.data):,} B / MSE {final.mse:.3f}; patches {len(init_patches)} init + {len(refine_patches)} refine", flush=True)


if __name__ == '__main__':
    main()
