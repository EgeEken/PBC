import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from research_pass29.experiment import (
    CELL, decode_enhancement, decode_template, features_of, grouped_palette,
    pack2, pack4, permutation, quantize, read_template, read_virtual,
    reconstruct, template_file, unpack2, unpack4, verify, virtual_file,
)


class ResearchPass29Tests(unittest.TestCase):
    def test_prng_vectors_and_packers(self):
        self.assertEqual(permutation(10, 0, 0).tolist(), [9, 7, 6, 3, 5, 1, 2, 0, 4, 8])
        self.assertEqual(permutation(8, 7, 2).tolist(), [7, 1, 5, 6, 3, 4, 2, 0])
        a = np.arange(39, dtype=np.uint8) % 4
        b = np.arange(39, dtype=np.uint8) % 16
        np.testing.assert_array_equal(unpack2(pack2(a), len(a)), a)
        np.testing.assert_array_equal(unpack4(pack4(b), len(b)), b)

    def test_partial_cells_constant_and_signed_residual(self):
        residual = np.zeros((17, 19, 3), dtype=np.int16)
        residual[0, 0] = [-12, 5, 21]
        dist, structure = features_of(residual)
        self.assertEqual(dist.shape, (4, 14))
        self.assertEqual(structure.shape, (4, 3))
        assign = np.array([0, 1, 2, 3], dtype=np.uint8)
        palette = grouped_palette(residual, assign)
        indices = quantize(residual, palette, assign)
        base = np.full((17, 19, 3), 128, dtype=np.uint8)
        rec = reconstruct(base, indices, palette, assign)
        self.assertEqual(rec.shape, (17, 19, 3))
        self.assertTrue(np.all((rec >= 0) & (rec <= 255)))

    def test_virtual_and_template_round_trip(self):
        rng = np.random.default_rng(22)
        h, w = 17, 19
        residual = rng.integers(-32, 33, (h, w, 3), dtype=np.int16)
        base = np.full((h, w, 3), 128, dtype=np.uint8)
        assign = np.array([0, 1, 2, 3], dtype=np.uint8)
        palette = np.tile(np.array([[-20, 0, 0], [0, 0, 0], [12, 3, -2], [30, -4, 6]], np.int16), (4, 1, 1))
        indices = quantize(residual, palette, assign)
        expected = reconstruct(base, indices, palette, assign)
        desc = np.arange(4 * 17, dtype=np.float64).reshape(4, 17)
        for strategy in (0, 1, 2, 3):
            blob, _ = virtual_file(indices, assign, palette, w, h, strategy, 1, 2 if strategy else 0, desc)
            verify(blob, indices, strategy)
            np.testing.assert_array_equal(decode_enhancement(blob, base), expected)
        blob, _ = template_file(indices, assign, palette, w, h)
        np.testing.assert_array_equal(read_template(blob), indices)
        np.testing.assert_array_equal(decode_template(blob, base), expected)
        for bad in (blob[:5], blob[:-3], blob + b"trailing"):
            with self.assertRaises(Exception):
                read_template(bad)

    def test_four_bit_global_palette(self):
        rng = np.random.default_rng(4)
        h, w = 17, 19
        residual = rng.integers(-100, 100, (h, w, 3), dtype=np.int16)
        base = np.full((h, w, 3), 128, dtype=np.uint8)
        palette = np.arange(48, dtype=np.int16).reshape(16, 3) - 24
        indices = quantize(residual, palette, None)
        blob, parts = virtual_file(indices, np.zeros(4, np.uint8), palette, w, h)
        recovered, metadata = read_virtual(blob)
        self.assertEqual(parts["index_bits"], 4)
        np.testing.assert_array_equal(recovered, indices)
        np.testing.assert_array_equal(decode_enhancement(blob, base), reconstruct(base, indices, palette, None))

    def test_file_only_fresh_process(self):
        h, w = 17, 19
        base = np.full((h, w, 3), 120, dtype=np.uint8)
        assign = np.array([0, 1, 2, 3], dtype=np.uint8)
        palette = np.tile(np.array([[-8, 0, 0], [0, 0, 0], [8, 0, 0], [16, 0, 0]], np.int16), (4, 1, 1))
        indices = np.arange(4 * CELL * CELL, dtype=np.uint8).reshape(4, -1) % 4
        blob, _ = virtual_file(indices, assign, palette, w, h, 1, 2, 2)
        template, _ = template_file(indices, assign, palette, w, h)
        with tempfile.TemporaryDirectory() as td:
            stream_path, template_path, base_path, out_path, out_template = [
                Path(td) / n for n in ("x.p29", "x.p29t", "base.npy", "out.npy", "out_template.npy")]
            stream_path.write_bytes(blob)
            template_path.write_bytes(template)
            np.save(base_path, base)
            code = ("import numpy as np; from pathlib import Path; "
                    "from research_pass29.experiment import decode_enhancement, decode_template; "
                    f"base=np.load(r'{base_path}'); "
                    f"np.save(r'{out_path}', decode_enhancement(Path(r'{stream_path}').read_bytes(), base)); "
                    f"np.save(r'{out_template}', decode_template(Path(r'{template_path}').read_bytes(), base))")
            subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[1], check=True)
            np.testing.assert_array_equal(np.load(out_path), decode_enhancement(blob, base))
            np.testing.assert_array_equal(np.load(out_template), decode_template(template, base))


if __name__ == "__main__":
    unittest.main()
