import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import pbc4_codec as codec
from pbc4_stream import HEADER, pack_bits, pack_file


def gradient(height=35, width=37):
    y, x = np.mgrid[:height, :width]
    return np.stack(((x * 7 + y * 3) % 256, (x * 3) % 256,
                     (y * 9) % 256), axis=2).astype(np.uint8)


def mixed_stream():
    source = gradient(16, 48)
    iw, ih, count, base, init_payload, _ = codec._make_initialization(source, 768)
    modes = pack_bits((1, 2, 0), 2)
    params = bytearray((0,))
    params.extend(struct.pack('<3h', 12, -4, 0))
    for _ in range(12):
        codec._write_coefficients(params, np.zeros(64, dtype=np.int16))
    payload = init_payload + modes + bytes(params)
    return pack_file(48, 16, iw, ih, count, 1, 1, 0, payload), base


class PBC4CodecTests(unittest.TestCase):
    def test_synthetic_images_and_odd_boundaries(self):
        y, x = np.mgrid[:33, :47]
        edge = np.where(x < 23, 20, 235).astype(np.uint8)
        text = (((x // 4 + y // 3) % 2) * 255).astype(np.uint8)
        noise = np.random.default_rng(4).integers(0, 256, (19, 23), dtype=np.uint8)
        images = [
            np.full((17, 19, 3), 42, dtype=np.uint8),
            gradient(),
            np.repeat(edge[:, :, None], 3, axis=2),
            np.repeat(text[:, :, None], 3, axis=2),
            np.repeat(noise[:, :, None], 3, axis=2),
        ]
        for image in images:
            encoded = codec.encode(image, 'D', 'intermediate', init_caps=(256, 4096))
            decoded = codec.decode(encoded['data'])
            self.assertEqual(decoded.shape, image.shape)
            self.assertTrue(np.array_equal(decoded, encoded['reconstruction']))
            self.assertTrue(encoded['skip_exact'])

    def test_each_ablation_and_quality_scaling(self):
        image = gradient()
        base = codec.encode(image, 'A', 'low', init_caps=(256, 4096))
        procedural = codec.encode(image, 'B', 'high', init_caps=(256, 4096))
        coarse = codec.encode(image, 'C', 'low', init_caps=(256, 4096))
        fine = codec.encode(image, 'C', 'high', init_caps=(256, 4096))
        both = codec.encode(image, 'D', 'high', init_caps=(256, 4096))
        self.assertEqual(base['mode_counts']['skip'], 9)
        self.assertGreater(procedural['mode_counts']['procedural'], 0)
        self.assertGreater(coarse['mode_counts']['skip'], 0)
        self.assertGreater(fine['mode_counts']['transform'], 0)
        self.assertLess(fine['mse'], coarse['mse'])
        self.assertTrue(both['skip_exact'])

    def test_mixed_modes_and_exact_skip_preservation(self):
        data, base = mixed_stream()
        decoded, info = codec.decode(data, return_info=True)
        self.assertEqual(info['mode_counts'], {'skip': 1, 'procedural': 1, 'transform': 1})
        self.assertTrue(info['skip_exact'])
        self.assertTrue(np.array_equal(decoded[:, 32:48], base[:, 32:48]))
        self.assertTrue(np.array_equal(decoded[:, :16], np.clip(base[:, :16].astype(np.int16) + (12, -4, 0), 0, 255).astype(np.uint8)))

    def test_taper_never_changes_skip_pixels(self):
        source = gradient(16, 48)
        iw, ih, count, base, init_payload, _ = codec._make_initialization(source, 768)
        mode_map = pack_bits((1, 0, 0), 2)
        params = bytes((0,)) + struct.pack('<3h', 80, -40, 20)
        payload = init_payload + mode_map + params
        stream = pack_file(48, 16, iw, ih, count, 1, 1, 0, payload, taper=True)
        decoded = codec.decode(stream)
        self.assertTrue(np.array_equal(decoded[:, 16:], base[:, 16:]))
        self.assertFalse(np.array_equal(decoded[:, :16], base[:, :16]))

    def test_fresh_process_file_only_decode_and_determinism(self):
        image = gradient()
        first = codec.encode(image, 'D', 'high', init_caps=(256, 4096))
        second = codec.encode(image, 'D', 'high', init_caps=(256, 4096))
        self.assertEqual(first['file_sha256'], second['file_sha256'])
        with tempfile.TemporaryDirectory() as directory:
            stream = Path(directory) / 'image.pbc'
            output = Path(directory) / 'decoded.png'
            stream.write_bytes(first['data'])
            subprocess.run([sys.executable, str(ROOT / 'worker_decode.py'), str(stream), str(output)],
                           check=True, capture_output=True, text=True)
            with Image.open(output) as decoded:
                self.assertTrue(np.array_equal(np.asarray(decoded.convert('RGB')), first['reconstruction']))

    def test_malformed_and_truncated_streams_rejected(self):
        data = codec.encode(gradient(), 'C', 'high', init_caps=(256, 4096))['data']
        for invalid in (data[:10], data[:-1], b'NOPE' + data[4:]):
            with self.assertRaises(ValueError):
                codec.decode(invalid)
        bad_crc = bytearray(data)
        bad_crc[HEADER.size - 1] ^= 1
        with self.assertRaises(ValueError):
            codec.decode(bad_crc)
        fields = list(HEADER.unpack(data[:HEADER.size]))
        fields[3] = 9000
        huge = HEADER.pack(*fields) + data[HEADER.size:]
        with self.assertRaises(ValueError):
            codec.decode(huge)


if __name__ == '__main__':
    unittest.main()
