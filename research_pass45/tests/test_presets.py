import hashlib
import importlib
import json
import os
import pathlib
import sys
import unittest

import numpy as np
from PIL import Image, ImageOps

source = pathlib.Path(os.environ.get("PBC_SOURCE", pathlib.Path(__file__).resolve().parents[2])).resolve()
root = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(source))
sys.path.insert(0, str(source / "pbc3"))
PBC3 = importlib.import_module("PBC3").PBC3
PBC3Config = importlib.import_module("pbc3_types").PBC3Config


class CanonicalPresets(unittest.TestCase):
    expected_q = {"compression": 0.4, "balanced": 0.6, "quality": 0.8, "high_quality": 0.95}
    expected = json.loads((root / "research_pass10" / "baseline_manifest.json").read_text())["presets"]

    def test_resolved_preset_values(self):
        for name, q in self.expected_q.items():
            cfg = getattr(PBC3Config, name)()
            actual = dict(cfg.__dict__)
            actual["learned_filler_model_path"] = "patch_policy.npz"
            self.assertEqual(actual, self.expected[name], name)
            self.assertEqual(cfg.learned_filler_q, q, name)
            self.assertEqual(cfg.patch_count, 20 if name == "high_quality" else 50, name)
            self.assertEqual((cfg.search_depth, cfg.proposal_depth, cfg.exact_depth), (200, 50, 10), name)
            self.assertEqual(cfg.random_seed, 2003, name)
            self.assertEqual(cfg.color_space, "YCbCr", name)
            self.assertEqual(cfg.use_lzma, True, name)
            self.assertEqual(cfg.learned_filler_enabled, True, name)

    def test_model_identity_and_format(self):
        cfg = PBC3Config.compression()
        model = pathlib.Path(cfg.learned_filler_model_path)
        if not model.is_file():
            model = next(path for path in (source / "pbc3" / "patch_policy.npz", source / "patch_policy.npz") if path.is_file())
        self.assertEqual(hashlib.sha256(model.read_bytes()).hexdigest(), "ffaf24aa5447b3602c53a5c4c8df14a68ee16e6bded39331d973f38d0f1ea62f")
        self.assertEqual(PBC3.MAGIC, b"PBC3")
        self.assertEqual(PBC3.VERSION, 0)

    def test_coast_compression_golden(self):
        with Image.open(root / "hpt_data" / "coast.jpg") as image:
            original = ImageOps.exif_transpose(image).convert("RGB")
        config = PBC3Config.compression()
        if not pathlib.Path(config.learned_filler_model_path).is_file():
            config.learned_filler_model_path = str(next(path for path in (source / "pbc3" / "patch_policy.npz", source / "patch_policy.npz") if path.is_file()))
        result = PBC3.compress(original, config=config, rng_mode="release")
        self.assertEqual(hashlib.sha256(result.data).hexdigest(), "792e9602ba3c4a407c143c1e1e2fdb6e15d35ae1ca8ad2446303770a0833a029")
        decoded = PBC3.decompress(result.data).image.convert("RGB")
        actual = np.asarray(decoded, dtype=np.uint8)
        self.assertEqual(hashlib.sha256(actual.tobytes()).hexdigest(), "0c0527f2fb9f95694509acd5fe91b1440126c80b068bd1476a77970e7150f655")
        source_pixels = np.asarray(original, dtype=np.int32)
        diff = source_pixels - actual.astype(np.int32)
        self.assertAlmostEqual(float(np.mean(diff * diff, dtype=np.float64)), 73.13358281255552, places=12)


if __name__ == "__main__":
    unittest.main()
