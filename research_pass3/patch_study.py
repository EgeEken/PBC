import json
import os
import sys
import time

from support import ROOT, OUT, PASS2, SCREEN, THREAD_ENV, save, snapshot, sha

os.environ.update(THREAD_ENV)
sys.path.insert(0, str(ROOT))
import numpy as np
import psutil
from PIL import Image, ImageOps
from PBC3 import PBC3
from pbc3_types import PBC3Config
import pbc3_ops as ops
import learned_filler as filler
from stream_tools import parse, serialize
from representation import choices


def rgb_mse(reference, image):
    error = np.subtract(reference, np.asarray(image.convert('RGB')), dtype=np.float32)
    return float(np.mean(error ** 2, dtype=np.float64))


def study():
    before = snapshot()
    psutil.Process().cpu_affinity([0])
    previous = json.loads((PASS2 / 'quality_confirmation.json').read_text())
    controls = [r for r in previous if r['trial'] == 0 and r['image'] in SCREEN and r['label'] != 'q500_p20']
    assert len(controls) == 30
    rows, alternatives, shortlisted = [], [], []
    grids = OUT / 'quantized_grids'
    grids.mkdir(exist_ok=True)
    for row in controls:
        data = (PASS2 / row['stream_path']).read_bytes()
        header, patches, accounting = parse(data)
        with Image.open(ROOT / row['dataset'] / row['image']) as opened:
            image = ImageOps.exif_transpose(opened).convert('RGB')
        reference = np.asarray(image)
        config = PBC3Config(**row['config'])
        target = PBC3.prepare(image, config)['target']
        canvas = PBC3._decode_to_canvas(data, max_patches=0)[0]
        assert canvas.shape == target.shape
        original_decoded = PBC3.decompress(data)
        assert sha(np.asarray(original_decoded.image).tobytes()) == row['pixels_sha256']
        working_sse = int(np.sum((target - np.clip(canvas, 0, 255)).astype(np.int64) ** 2))
        filler._build_action(target, canvas, (0, 0, 0, min(32, header[3]), min(32, header[4])), config, header[7], 2, 2)
        n = len(patches) - header[6]
        indices = sorted({header[6] + i for i in (0, n // 3, 2 * n // 3, n - 1) if i >= 0 and i < n} |
            ({max(range(header[6], len(patches)), key=lambda i: accounting['patches'][i]['total_bits'])} if n else set()))
        quantized = {}
        prefix_cache = {}
        stream_shortlists = 0
        for index, patch in enumerate(patches):
            c, x, y, w, h = (patch[k] for k in ('channel', 'x', 'y', 'w', 'h'))
            palette = ops.palette_generator(patch['mask'], patch['max_bitcount'], patch['neg'], patch['pos'], header[8])
            values = palette[patch['indices']]
            delta = ops.signed_resample(values, h, w)
            base = canvas[y:y + h, x:x + w, c]
            residual = target[y:y + h, x:x + w, c] - np.clip(base, 0, 255)
            before_sse = int(np.sum(residual.astype(np.int64) ** 2))
            after_error = target[y:y + h, x:x + w, c] - np.clip(base + delta, 0, 255)
            after_sse = int(np.sum(after_error.astype(np.int64) ** 2))
            gain = before_sse - after_sse
            if index >= header[6]:
                quantized[f'p{index:03d}'] = values
                record = dict(source=row['source'], label=row['label'], dataset=row['dataset'], image=row['image'], preset=row['preset'],
                    working_width=header[3], working_height=header[4], working_limit=config.auto_downsample_max_pixels,
                    original_width=image.width, original_height=image.height, stream_patch=index,
                    iterative_position=index - header[6] + 1, channel=c, x=x, y=y, w=w, h=h, area=w * h,
                    cell_size=patch['cell_size'], grid_width=values.shape[1], grid_height=values.shape[0], grid_cells=values.size,
                    max_bitcount=patch['max_bitcount'], actual_bitcount=patch['bitcount'], mask=patch['mask'],
                    raw_bits=ops.patch_bits_for(patch, header[7]), local_sse_gain=gain,
                    global_working_sse_before=working_sse, global_working_sse_after=working_sse - gain,
                    gain_per_raw_bit=gain / ops.patch_bits_for(patch, header[7]),
                    quantized_values_sha256=sha(values.tobytes()), quantized_values_key=f'p{index:03d}',
                    quantized_values_file=f"quantized_grids/{row['label']}.{row['image']}.{row['preset']}.npz",
                    source_stream=str((PASS2 / row['stream_path']).relative_to(ROOT)), source_stream_sha256=sha(data),
                    input_sha256=row['input_sha256'], model_sha256=row['model_sha256'], source_revision=row['source_revision'], config=row['config'])
                assert record['raw_bits'] == accounting['patches'][index]['total_bits']
                if index in indices:
                    for count in (index, index + 1):
                        if count not in prefix_cache:
                            prefix = serialize(header, patches[:count])
                            decoded = PBC3.decompress(prefix)
                            prefix_cache[count] = (len(prefix), rgb_mse(reference, decoded.image))
                    a, b = prefix_cache[index], prefix_cache[index + 1]
                    record.update(marginal_final_bits=8 * (b[0] - a[0]),
                        original_rgb_mse_before=a[1], original_rgb_mse_after=b[1], original_rgb_mse_gain=a[1] - b[1])
                    box = c, x, y, w, h
                    variants = [('original', patch['cell_size'], patch['max_bitcount'])] + choices(patch['cell_size'], patch['max_bitcount'], w, h, config, finer=True)
                    for name, cell, precision in variants:
                        started = time.perf_counter()
                        new, new_values, _, new_gain, bits = filler._build_action(target, canvas, box, config, header[7], cell, precision)
                        seconds = time.perf_counter() - started
                        if name == 'original':
                            assert new_gain == gain and np.array_equal(new_values, values)
                            assert all(np.array_equal(new[k], patch[k]) for k in patch)
                        alternative = dict(label=row['label'], source=row['source'], dataset=row['dataset'], image=row['image'], preset=row['preset'],
                            stream_patch=index, representation=name, cell_size=cell, max_bitcount=precision,
                            actual_bitcount=new['bitcount'], grid_cells=new_values.size,
                            raw_bits=bits, sse_gain=new_gain, original_raw_bits=record['raw_bits'], original_sse_gain=gain,
                            retained_gain_fraction=new_gain / gain if gain > 0 else None,
                            raw_bit_fraction=bits / record['raw_bits'], evaluation_seconds=seconds,
                            values_sha256=sha(new_values.tobytes()), geometry_fixed=True, baseline_state_fixed=True)
                        alternatives.append(alternative)
                        if name not in ('original', 'finer_grid') and gain > 0 and new_gain >= .98 * gain and bits <= .75 * record['raw_bits'] and stream_shortlists < 2:
                            replaced = list(patches)
                            replaced[index] = new
                            started = time.perf_counter()
                            candidate = serialize(header, replaced)
                            repack = time.perf_counter() - started
                            started = time.perf_counter()
                            decoded = PBC3.decompress(candidate)
                            decode = time.perf_counter() - started
                            shortlisted.append({**alternative, 'final_bytes': len(candidate), 'original_final_bytes': len(data),
                                'bpp': 8 * len(candidate) / (image.width * image.height), 'mse': rgb_mse(reference, decoded.image),
                                'original_mse': row['mse'], 'repack_seconds': repack, 'decode_seconds': decode,
                                'interpretation': 'One representation replaced; all later original payloads fixed; not a new sequential encode'})
                            stream_shortlists += 1
                rows.append(record)
            base += delta
            working_sse -= gain
            if index in indices:
                assert working_sse == int(np.sum((target - np.clip(canvas, 0, 255)).astype(np.int64) ** 2))
        assert np.array_equal(np.asarray(PBC3._canvas_to_image(canvas, header[5], header[9]).resize(image.size, PBC3.RESAMPLE_FILTER)), np.asarray(original_decoded.image))
        np.savez_compressed(grids / f"{row['label']}.{row['image']}.{row['preset']}.npz", **quantized)
        print('patch study', row['source'], row['image'], row['preset'], config.auto_downsample_max_pixels, flush=True)
    save('patch_ledger', rows)
    save('representation_study', alternatives)
    save('representation_shortlist', shortlisted)
    save('patch_study_provenance', dict(before=before, after=snapshot(), selected_positions='first, one-third, two-thirds, last, largest raw patch; deduplicated',
        warmup='One discarded action build per trajectory before representation timings; existing disk caches retained',
        shortlist='at most two per stream, >=98% local SSE gain with <=75% raw bits, selected before LZMA/decode; no search rerun'))


if __name__ == '__main__':
    study()
