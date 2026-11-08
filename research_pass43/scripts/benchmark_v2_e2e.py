import hashlib
import json
import math
import os
import pathlib
import statistics
import sys
import time

import numpy as np
import threadpoolctl
from PIL import Image

HERE = pathlib.Path(__file__).resolve().parents[1]
ROOT = HERE.parent
OUT = HERE / 'outputs'
FROZEN = ROOT / 'research_pass42' / 'outputs' / 'v31_frozen'
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / 'research_pass42'))
sys.path.insert(0, str(FROZEN))
import pbc4_codec as v1
import pbc4_codec_v2 as v2
import pbc3_heads
from PBC3 import PBC3
from pbc3_types import BitReader, BitWriter, PBC3Config


def encode_b(source, image):
    init_patches, refinements = [], []
    old_init, old_fill = pbc3_heads.DownsampleInitHead.select, pbc3_heads.FillerHead.select

    def capture_init(self, *args, **kwargs):
        result = old_init(self, *args, **kwargs)
        init_patches.append(result[0])
        return result

    def capture_fill(self, *args, **kwargs):
        result = old_fill(self, *args, **kwargs)
        if result[0] is not None:
            refinements.append(result[0])
        return result

    pbc3_heads.DownsampleInitHead.select = capture_init
    pbc3_heads.FillerHead.select = capture_fill
    try:
        full = PBC3.compress(Image.fromarray(source), config=PBC3Config.compression(), rng_mode='release')
    finally:
        pbc3_heads.DownsampleInitHead.select = old_init
        pbc3_heads.FillerHead.select = old_fill
    _, raw = PBC3._open_body(full.data)
    header = PBC3._read_header(BitReader(raw))
    (downsampled, original_w, original_h, working_w, working_h, color_space,
     channels, channel_bits, positive_bias, has_alpha, patch_count, base_values,
     warmup_on, warm_w, warm_h, warmup_split) = header
    config = PBC3Config.compression()
    prep = PBC3.prepare(Image.fromarray(source), config)
    if prep['warmup_patches'] is None or not warmup_on:
        warmup_split = None
    patches = init_patches + refinements
    count = min(3, len(patches))
    bw = BitWriter()
    PBC3._write_header(bw, working_w, working_h, original_w, original_h, downsampled,
                       prep['color_id'], channels, max(1, math.ceil(math.log2(channels))),
                       config.positive_bias, has_alpha, count,
                       [int(round(float(np.mean(prep['arr'][:, :, c])))) for c in range(channels)])
    for patch in patches[:count]:
        PBC3._write_patch(bw, patch, max(1, math.ceil(math.log2(channels))))
    raw_prefix = bw.finish()
    prefix = PBC3.MAGIC + bytes((PBC3.VERSION, PBC3.ENTROPY_LZMA)) + PBC3._entropy_pack(raw_prefix, True)[1]
    return v2.encode_from_prefix(source, prefix, variant='D', point='low')['data']


def main():
    image = ROOT / 'hpt_data' / 'coast.jpg'
    source = v1.load_rgb(image)
    expected_a = (ROOT / 'research_pass42' / 'outputs' / 'coast_v4_D_low.pbc').read_bytes()
    previous = pathlib.Path.cwd()
    os.chdir(FROZEN)
    try:
        rows = {'warmups_per_encoder': 1, 'repetitions': 3, 'order': [], 'samples_seconds': {'A': [], 'B': []}}
        with threadpoolctl.threadpool_limits(limits=1, user_api='blas'):
            v1.encode(image, variant='D', point='low')
            encode_b(source, image)
            for i in range(3):
                for label in (('A', 'B') if i % 2 == 0 else ('B', 'A')):
                    start = time.perf_counter()
                    data = (v1.encode(image, variant='D', point='low')['data'] if label == 'A'
                            else encode_b(source, image))
                    elapsed = time.perf_counter() - start
                    if label == 'A' and data != expected_a:
                        raise RuntimeError('V4-A timing stream differs from Pass 42')
                    rows['order'].append(label)
                    rows['samples_seconds'][label].append(elapsed)
                    rows.setdefault('files', {})[label] = {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
        rows['median_seconds'] = {k: statistics.median(v) for k, v in rows['samples_seconds'].items()}
        (OUT / 'v2_encode_e2e_benchmark.json').write_text(json.dumps(rows, indent=2) + '\n')
        print(json.dumps(rows, indent=2))
    finally:
        os.chdir(previous)


if __name__ == '__main__':
    main()
