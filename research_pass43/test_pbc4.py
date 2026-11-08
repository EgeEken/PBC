import pathlib
import os
import sys
import unittest

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / 'research_pass42'))
sys.path.insert(0, str(HERE))
FROZEN = ROOT / 'research_pass42' / 'outputs' / 'v31_frozen'
sys.path.insert(0, str(FROZEN))
import pbc4_codec
import pbc4_codec_v2
import pbc4_decode_opt
from PBC3 import PBC3
from pbc3_types import PBC3Config
from PIL import Image


class OptimizedDecoderTests(unittest.TestCase):
    def test_batched_surface_is_exact(self):
        rng = np.random.default_rng(43)
        controls = [np.zeros((4, 3), dtype=np.int16),
                    np.full((4, 3), 32767, dtype=np.int16),
                    np.full((4, 3), -32768, dtype=np.int16)]
        controls.extend(rng.integers(-1024, 1025, size=(4, 3), dtype=np.int16) for _ in range(200))
        for item in controls:
            np.testing.assert_array_equal(pbc4_decode_opt._surface_from_controls(item),
                                          pbc4_codec._surface_from_controls(item))

    def test_existing_stream_pixels_and_skip_are_exact(self):
        files = [ROOT / 'research_pass42' / 'outputs' / 'coast_v4_D_intermediate.pbc',
                 ROOT / 'research_pass42' / 'outputs' / 'googlewiki_v4_D_intermediate.pbc']
        for path in files:
            data = path.read_bytes()
            expected, expected_info = pbc4_codec.decode(data, return_info=True)
            actual, actual_info = pbc4_decode_opt.decode(data, return_info=True)
            np.testing.assert_array_equal(actual, expected)
            self.assertEqual(actual_info, expected_info)
            self.assertTrue(actual_info['skip_exact'])

    def test_v2_odd_dimensions_and_protected_skip(self):
        y, x = np.mgrid[:59, :73]
        source = np.stack(((x * 3 + y) % 256, (x + y * 5) % 256,
                           (x * 7 + y * 11) % 256), axis=2).astype(np.uint8)
        previous = pathlib.Path.cwd()
        try:
            os.chdir(FROZEN)
            prefix = PBC3.compress(Image.fromarray(source), config=PBC3Config.compression(),
                                   rng_mode='release').data
        finally:
            os.chdir(previous)
        encoded = pbc4_codec_v2.encode_from_prefix(source, prefix, variant='A', point='low')
        actual, info = pbc4_codec_v2.decode(encoded['data'], return_info=True)
        self.assertEqual(actual.shape, source.shape)
        self.assertTrue(info['skip_exact'])
        np.testing.assert_array_equal(actual, encoded['base'])


if __name__ == '__main__':
    unittest.main()
