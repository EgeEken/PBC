from __future__ import annotations

import lzma
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

import pbc33_codec as codec

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'research_pass31'))
import pbc31_codec


def stream_with_params(width: int, height: int, mode: int, params: bytes, qstep: int = 16) -> bytes:
    count = ((height + 15) // 16) * ((width + 15) // 16)
    modes = np.zeros(count, np.uint8)
    modes[0] = mode
    raw = codec.pack_modes(modes) + params
    xz = lzma.compress(raw, format=lzma.FORMAT_XZ, preset=6)
    header = codec.HEADER.pack(codec.MAGIC, 1, width, height, 16, 8, 16, qstep, 3, count, len(raw), len(xz))
    return header + xz


class SparseDetailTests(unittest.TestCase):
    def source_base(self, height=19, width=23):
        rng = np.random.default_rng(8301)
        base = rng.integers(40, 210, (height, width, 3), dtype=np.uint8)
        src = base.copy()
        src[2:min(height, 12), 3:min(width, 17)] = np.clip(src[2:min(height, 12), 3:min(width, 17)].astype(np.int16) + 31, 0, 255).astype(np.uint8)
        src[height // 2:, width // 2:, 0] = 255
        return src, base

    def test_A_and_B_match_pass31_reconstruction_and_parameters(self):
        src, base = self.source_base(31, 35)
        for config in ('A', 'B'):
            reference_config = config
            reference = pbc31_codec.encode(src, base, reference_config, 20.0)
            got = next(x for x in codec.encode_family(src, base, config, (20.0,)) if x['lambda'] == 20.0)
            self.assertTrue(np.array_equal(reference['modes'], got['modes']))
            self.assertEqual(reference['params'], got['params'])
            self.assertTrue(np.array_equal(reference['reconstruction'], got['reconstruction']))
            decoded, modes, params = codec.decode(got['data'], base)
            self.assertTrue(np.array_equal(decoded, got['reconstruction']))
            self.assertTrue(np.array_equal(modes, got['modes']))
            self.assertEqual(params, got['params'])

    def test_signed_sparse_coefficients_and_quantizers(self):
        src, base = self.source_base(17, 25)
        src[:8, :8] = np.clip(base[:8, :8].astype(np.int16) - 34, 0, 255).astype(np.uint8)
        for step in codec.QUANT_STEPS:
            rows = codec.encode_family(src, base, 'S', (0.0, 20.0, 100.0), step)
            for row in rows:
                rec, modes, params = codec.decode(row['data'], base)
                self.assertTrue(np.array_equal(rec, row['reconstruction']))
                self.assertTrue(np.array_equal(modes, row['modes']))
                self.assertEqual(params, row['params'])
                self.assertLessEqual(row['coefficients'], 8 * row['active_blocks'])

    def test_zero_residual_empty_enhancement_and_constant(self):
        base = np.full((16, 16, 3), 127, np.uint8)
        for config, qstep in (('A', 0), ('B', 0), ('S', 16)):
            result = codec.encode_family(base, base, config, (100.0,), qstep)[0]
            self.assertEqual(result['mode_counts'], [1, 0, 0])
            rec, modes, params = codec.decode(result['data'], base)
            self.assertTrue(np.array_equal(rec, base))
            self.assertEqual(bytes(params), b'')

    def test_sharp_text_like_gradient_and_clipping(self):
        h, w = 33, 41
        base = np.zeros((h, w, 3), np.uint8)
        yy, xx = np.indices((h, w))
        original = np.where(((xx // 3 + yy // 4) % 2)[..., None], 255, 16).astype(np.uint8)
        source = np.repeat(original, 3, axis=2)
        source[:, :w // 2] = np.minimum(source[:, :w // 2].astype(np.int16) + 20, 255).astype(np.uint8)
        result = codec.encode_family(source, base, 'S', (0.0, 20.0, 100.0), 16)[1]
        decoded, _, _ = codec.decode(result['data'], base)
        self.assertTrue(np.array_equal(decoded, result['reconstruction']))
        self.assertGreater(result['active_blocks'], 0)

    def test_mode_and_coefficient_packing_boundaries(self):
        for count in range(1, 12):
            modes = np.arange(count, dtype=np.uint8) % 3
            packed = codec.pack_modes(modes)
            self.assertTrue(np.array_equal(codec.unpack_modes(packed, count), modes))
        for value in (-32768, -128, -1, 1, 127, 32767):
            record = codec._coefficient_record(191, value)
            decoded, end = codec._read_varint(record, 1)
            self.assertEqual((decoded >> 1) ^ -(decoded & 1), value)
            self.assertEqual(end, len(record))
        with self.assertRaises(ValueError):
            codec._coefficient_record(192, 1)
        with self.assertRaises(ValueError):
            codec._coefficient_record(0, 0)

    def test_decoder_reconstructs_only_active_blocks(self):
        src, base = self.source_base(32, 32)
        result = codec.encode_family(src, base, 'S', (20.0,), 32)[0]
        calls = 0
        original = codec.idctn
        def counted(*args, **kwargs):
            nonlocal calls
            calls += 1
            return original(*args, **kwargs)
        codec.idctn = counted
        try:
            codec.decode(result['data'], base)
        finally:
            codec.idctn = original
        self.assertEqual(calls, result['active_blocks'])
        self.assertLessEqual(result['active_blocks'], 4 * int(np.count_nonzero(result['modes'] == 2)))

    def test_malformed_duplicate_invalid_and_truncated_streams(self):
        base = np.full((16, 16, 3), 90, np.uint8)
        good = stream_with_params(16, 16, 2, bytes((1, 2, 0, 2, 0, 2)))
        cases = [good[:-1], good + b'x', stream_with_params(16, 16, 2, bytes((1, 2, 192, 2))),
                 stream_with_params(16, 16, 2, bytes((1, 2, 0, 0))),
                 stream_with_params(16, 16, 2, bytes((1, 9))),
                 stream_with_params(16, 16, 2, bytes((0,))),
                 stream_with_params(16, 16, 2, bytes((1, 1, 0, 255, 255, 127)))]
        duplicate = stream_with_params(16, 16, 2, bytes((1, 2, 0, 2, 0, 2)))
        for data in [duplicate] + cases:
            with self.assertRaises(ValueError):
                codec.decode(data, base)
        bad_header = list(codec.HEADER.unpack_from(good)); bad_header[7] = 12
        malformed = codec.HEADER.pack(*bad_header) + good[codec.HEADER.size:]
        with self.assertRaises(ValueError):
            codec.decode(malformed, base)

    def test_fresh_process_file_only_decode(self):
        src, base = self.source_base(19, 21)
        result = codec.encode_family(src, base, 'S', (20.0,), 64)[0]
        worker = Path(__file__).with_name('worker_decode.py')
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            np.save(root / 'base.npy', base, allow_pickle=False)
            (root / 'stream.p33s').write_bytes(result['data'])
            manifest = {'streams': [{'key': 'case', 'stream': str(root / 'stream.p33s'), 'base_npy': str(root / 'base.npy')}]}
            (root / 'manifest.json').write_text(__import__('json').dumps(manifest))
            proc = subprocess.run([sys.executable, '-B', str(worker), str(root / 'manifest.json')],
                                  check=True, capture_output=True, text=True)
            row = __import__('json').loads(proc.stdout)['rows'][0]
            self.assertEqual(row['pixels_sha256'], codec.sha(result['reconstruction'].tobytes()))
            self.assertEqual(row['modes_sha256'], codec.sha(result['modes'].tobytes()))
            self.assertEqual(row['params_sha256'], codec.sha(result['params']))


if __name__ == '__main__':
    unittest.main()
