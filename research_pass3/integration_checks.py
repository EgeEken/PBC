import contextlib
import os
import sys

from support import ROOT, OUT, THREAD_ENV, save, snapshot, load, sha

os.environ.update(THREAD_ENV)
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image
from PBC3 import PBC3
from pbc3_types import PBC3Config, BitReader, BitWriter
import pbc3_stream as production


def check():
    before = snapshot()
    proposed = load('pass3_integration_check', OUT / 'integration_candidate' / 'pbc3_stream.py')
    rng = np.random.default_rng(2003)
    cases = []
    for mode in ('scalar', 'numpy'):
        proposed.GRID_IO_MODE = mode
        for width in range(17):
            for offset in range(8):
                values = rng.integers(0, 1 << width, 37, dtype=np.uint32)
                a, b = BitWriter(), BitWriter()
                a.write((1 << offset) - 1, offset)
                b.write((1 << offset) - 1, offset)
                production.write_grid(a, values, width)
                proposed.write_grid(b, values, width)
                assert a.finish() == b.finish()
                a, b = BitReader(a.finish()), BitReader(b.finish())
                a.read(offset)
                b.read(offset)
                assert np.array_equal(production.read_grid(a, len(values), width), proposed.read_grid(b, len(values), width))
                assert (a.i, a.acc, a.nbits) == (b.i, b.acc, b.nbits)
                cases.append(dict(mode=mode, width=width, offset=offset, exact=True))
    images = []
    for channels in (3, 4):
        image = Image.fromarray(rng.integers(0, 256, (97, 131, channels), dtype=np.uint8))
        cfg = PBC3Config.quality(patch_count=5, auto_downsample_max_pixels=4096,
            learned_filler_model_path=str(ROOT / 'patch_policy.npz'))
        original = PBC3.compress(image, config=cfg)
        pixels = np.asarray(PBC3.decompress(original.data).image)
        read_descriptor = PBC3.__dict__['_read_patch']
        write_descriptor = PBC3.__dict__['_write_patch']
        try:
            PBC3._read_patch = staticmethod(proposed.read_patch)
            PBC3._write_patch = staticmethod(proposed.write_patch)
            for mode in ('scalar', 'numpy'):
                proposed.GRID_IO_MODE = mode
                for metric in (False, True):
                    result = PBC3.compress(image, config=cfg, compute_final_mse=metric)
                    assert result.data == original.data
                    assert np.array_equal(np.asarray(PBC3.decompress(result.data).image), pixels)
                    assert (result.mse is None) == (not metric)
                    images.append(dict(channels=channels, mode=mode, metric=metric, exact=True, sha256=sha(result.data)))
        finally:
            PBC3._read_patch = read_descriptor
            PBC3._write_patch = write_descriptor
    save('integration_checks', dict(bit_layout=cases, images=images, unapplied=True))
    save('integration_checks_provenance', dict(before=before, after=snapshot()))


if __name__ == '__main__':
    check()
