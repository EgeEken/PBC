import unittest

import numpy as np
from PIL import Image

import learned_filler
import pbc3_ops as ops
from PBC3 import PBC3, _canvas_from_bases
from pbc3_types import PBC3Config
from pbc31_proposal_replay import (
    exact_fallback_proposal_function,
    mixed_propose_boxes,
)


class ProposalReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        y, x = np.mgrid[:97, :129]
        cls.image = Image.fromarray(
            np.stack(
                [
                    (x * 3 + y * 5) % 256,
                    (x * 7 + y) % 256,
                    (x + y * 11) % 256,
                ],
                axis=2,
            ).astype(np.uint8)
        )
        cls.config = PBC3Config.compression(
            patch_count=8,
            search_depth=40,
            proposal_depth=10,
        )

    def test_mixed_proposals_are_deterministic_and_in_bounds(self):
        prep = PBC3.prepare(self.image, self.config)
        height, width, channels = prep["target"].shape
        bases = [
            int(round(float(np.mean(prep["arr"][:, :, channel]))))
            for channel in range(channels)
        ]
        canvas = _canvas_from_bases((height, width, channels), bases)
        first, _ = mixed_propose_boxes(
            prep["target"],
            canvas,
            self.config,
            ops.PBC3Rng(self.config.random_seed),
            0,
            1,
        )
        second, _ = mixed_propose_boxes(
            prep["target"],
            canvas,
            self.config,
            ops.PBC3Rng(self.config.random_seed),
            0,
            1,
        )
        self.assertEqual(
            [(item["box"], item["source"]) for item in first],
            [(item["box"], item["source"]) for item in second],
        )
        self.assertEqual(len(first), self.config.proposal_depth)
        for item in first:
            _, x, y, box_width, box_height = item["box"]
            self.assertGreaterEqual(x, 0)
            self.assertGreaterEqual(y, 0)
            self.assertLessEqual(x + box_width, width)
            self.assertLessEqual(y + box_height, height)

    def test_exact_fallback_is_byte_identical(self):
        baseline = PBC3.compress(self.image, self.config)
        original = learned_filler.propose_boxes
        learned_filler.propose_boxes = exact_fallback_proposal_function
        try:
            fallback = PBC3.compress(self.image, self.config)
        finally:
            learned_filler.propose_boxes = original
        self.assertEqual(baseline.data, fallback.data)
        self.assertEqual(baseline.data[4], PBC3.VERSION)


if __name__ == "__main__":
    unittest.main()
