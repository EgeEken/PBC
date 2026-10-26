import unittest

import numpy as np
from PIL import Image

from PBC3 import PBC3
from pbc3_types import PBC3Config
from pbc31_multistage import (
    MultistagePBC31,
    SINGLE_STAGE,
    T2_COARSE_INIT_ONLY,
    T2_20_80,
    T3_10_20_70,
)


class MultistagePBC31Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        y, x = np.mgrid[:65, :97]
        cls.rgb = np.stack([
            (x * 3 + y) % 256,
            (x + y * 5) % 256,
            (x * 7 + y * 2) % 256,
        ], axis=2).astype(np.uint8)
        alpha = ((x * 2 + y * 3) % 256).astype(np.uint8)
        cls.rgba = np.dstack([cls.rgb, alpha])
        cls.config = PBC3Config.compression(
            patch_count=8,
            search_depth=20,
            proposal_depth=10,
            exact_depth=3,
            learned_filler_enabled=False,
        )

    def test_v0_compatibility_and_single_stage_identity(self):
        image = Image.fromarray(self.rgb)
        v0 = PBC3.compress(image, self.config)
        decoded_v0 = MultistagePBC31.decompress(v0.data)
        v1 = MultistagePBC31.compress(image, self.config, plan=SINGLE_STAGE)
        self.assertTrue(np.array_equal(np.asarray(v0.image), np.asarray(decoded_v0.image)))
        self.assertTrue(np.array_equal(np.asarray(v0.image), np.asarray(v1.image)))

    def test_two_and_three_stage_roundtrip(self):
        image = Image.fromarray(self.rgb)
        for plan in (T2_20_80, T3_10_20_70):
            with self.subTest(plan=plan.name):
                encoded = MultistagePBC31.compress(
                    image, self.config, plan=plan, diagnostics=True
                )
                decoded = MultistagePBC31.decompress(encoded.data)
                self.assertTrue(
                    np.array_equal(np.asarray(encoded.image), np.asarray(decoded.image))
                )
                self.assertEqual(encoded.data[4], MultistagePBC31.VERSION)
                self.assertEqual(len(encoded.timings["stages"]), len(plan.scales))

    def test_coarse_initialization_only_uses_full_final_budget(self):
        encoded = MultistagePBC31.compress(
            Image.fromarray(self.rgb),
            self.config,
            plan=T2_COARSE_INIT_ONLY,
            diagnostics=True,
        )
        self.assertEqual(encoded.timings["stages"][0]["budget"], 0)
        self.assertEqual(
            encoded.timings["stages"][1]["budget"], self.config.patch_count
        )
        decoded = MultistagePBC31.decompress(encoded.data)
        self.assertTrue(np.array_equal(np.asarray(encoded.image), np.asarray(decoded.image)))

    def test_odd_rgba_without_lzma(self):
        config = PBC3Config.quality(
            patch_count=5,
            search_depth=15,
            proposal_depth=8,
            exact_depth=2,
            learned_filler_enabled=False,
            use_lzma=False,
        )
        encoded = MultistagePBC31.compress(
            Image.fromarray(self.rgba), config, plan=T2_20_80
        )
        decoded = MultistagePBC31.decompress(encoded.data)
        self.assertTrue(np.array_equal(np.asarray(encoded.image), np.asarray(decoded.image)))
        self.assertEqual(decoded.image.mode, "RGBA")

    def test_partial_decode_transitions_on_next_patch(self):
        image = Image.fromarray(self.rgb)
        encoded = MultistagePBC31.compress(
            image, self.config, plan=T2_20_80, diagnostics=True
        )
        split = encoded.timings["stages"][1]["stream_start"]
        before = MultistagePBC31.decompress(encoded.data, max_patches=split)
        after = MultistagePBC31.decompress(encoded.data, max_patches=split + 1)
        self.assertNotEqual(before.image.size, image.size)
        self.assertEqual(after.image.size, image.size)

    def test_determinism(self):
        image = Image.fromarray(self.rgb)
        first = MultistagePBC31.compress(image, self.config, plan=T3_10_20_70)
        second = MultistagePBC31.compress(image, self.config, plan=T3_10_20_70)
        self.assertEqual(first.data, second.data)


if __name__ == "__main__":
    unittest.main()
