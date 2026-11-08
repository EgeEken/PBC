import hashlib
import json
import os
import pathlib
import statistics
import sys
import time

os.environ["NUMBA_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
from PIL import Image, ImageOps

ROOT = pathlib.Path(__file__).resolve().parents[2]
SOURCE = ROOT / "research_pass45" / "outputs" / "published_release_python"
sys.path.insert(0, str(SOURCE))
sys.path.insert(0, str(SOURCE / "pbc3"))
from PBC3 import PBC3
from pbc3_types import PBC3Config

EXPECTED = json.loads((ROOT / "research_pass44" / "outputs" / "competitive_baseline.json").read_text())
EXPECTED = {row["preset"]: row for row in EXPECTED["v31_points"] if row["image"] == "coast.jpg"}


def main():
    with Image.open(ROOT / "hpt_data" / "coast.jpg") as image:
        original = ImageOps.exif_transpose(image).convert("RGB")
    source_pixels = np.asarray(original, dtype=np.int32)
    rows = []
    for preset in ("compression", "balanced", "quality", "high_quality"):
        config = getattr(PBC3Config, preset)()
        warm = PBC3.compress(original, config=config, rng_mode="release")
        expected = EXPECTED[preset]
        if hashlib.sha256(warm.data).hexdigest() != expected["file_sha256"]:
            raise RuntimeError(f"stream mismatch: {preset}")
        if hashlib.sha256(np.asarray(warm.image.convert("RGB"), dtype=np.uint8).tobytes()).hexdigest() != expected["pixel_sha256"]:
            raise RuntimeError(f"reconstruction mismatch: {preset}")
        PBC3.decompress(warm.data)
        encode_samples = []
        decode_samples = []
        for _ in range(3):
            start = time.perf_counter()
            result = PBC3.compress(original, config=config, rng_mode="release")
            encode_samples.append(time.perf_counter() - start)
            if hashlib.sha256(result.data).hexdigest() != expected["file_sha256"]:
                raise RuntimeError(f"nondeterministic stream: {preset}")
            start = time.perf_counter()
            decoded_result = PBC3.decompress(result.data)
            decode_samples.append(time.perf_counter() - start)
            decoded = decoded_result.image.convert("RGB")
            pixels = np.asarray(decoded, dtype=np.uint8)
            difference = source_pixels - pixels.astype(np.int32)
            mse = float(np.mean(difference * difference, dtype=np.float64))
            if mse != expected["rgb_mse"]:
                raise RuntimeError(f"MSE mismatch: {preset}")
        rows.append({
            "preset": preset,
            "bytes": len(warm.data),
            "stream_sha256": expected["file_sha256"],
            "decoded_rgb_sha256": expected["pixel_sha256"],
            "rgb_mse": expected["rgb_mse"],
            "encode_samples_seconds": encode_samples,
            "decode_samples_seconds": decode_samples,
            "median_encode_seconds": statistics.median(encode_samples),
            "median_decode_seconds": statistics.median(decode_samples),
        })
    print(json.dumps({"source": "published ZIP embedded Python", "threads": 1, "warmups": 1, "repetitions": 3, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
