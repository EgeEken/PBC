import hashlib
import json
import os
import pathlib
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

PRESETS = ("compression", "balanced", "quality", "high_quality")
IMAGES = ("coast.jpg", "cat.jpg", "googlewiki.png", "mario64.png", "flower.jpg")
P9 = json.loads((ROOT / "research_pass9" / "runtime_manifest.json").read_text())
PASS44 = json.loads((ROOT / "research_pass44" / "outputs" / "competitive_baseline.json").read_text())
EXPECTED_STREAMS = {(row["image"], row["preset"]): row for row in PASS44["v31_points"]}


def run_case(name, image, preset):
    source = np.asarray(image, dtype=np.uint8)
    config = getattr(PBC3Config, preset)()
    start = time.perf_counter()
    first = PBC3.compress(image, config=config, rng_mode="release")
    encode_seconds = time.perf_counter() - start
    second = PBC3.compress(image, config=config, rng_mode="release")
    stream_hash = hashlib.sha256(first.data).hexdigest()
    if hashlib.sha256(second.data).hexdigest() != stream_hash:
        raise RuntimeError(f"nondeterministic encode: {name}/{preset}")
    start = time.perf_counter()
    decoded = PBC3.decompress(first.data).image
    decode_seconds = time.perf_counter() - start
    decoded_pixels = np.asarray(decoded, dtype=np.uint8)
    if not np.array_equal(decoded_pixels, np.asarray(first.image, dtype=np.uint8)):
        raise RuntimeError(f"encoder preview differs from file-only decode: {name}/{preset}")
    channels = source.shape[-1] if source.ndim == 3 else 1
    result = {
        "image": name,
        "preset": preset,
        "dimensions": list(image.size),
        "channels": channels,
        "source_pixel_sha256": hashlib.sha256(source.tobytes()).hexdigest(),
        "stream_sha256": stream_hash,
        "decoded_pixel_sha256": hashlib.sha256(decoded_pixels.tobytes()).hexdigest(),
        "bytes": len(first.data),
        "bpp": len(first.data) * 8 / (image.width * image.height),
        "encode_seconds": encode_seconds,
        "decode_seconds": decode_seconds,
        "deterministic_repeat": True,
    }
    expected_file = P9["assets"].get(f"hpt_data/{name}")
    if expected_file and hashlib.sha256((ROOT / "hpt_data" / name).read_bytes()).hexdigest() != expected_file:
        raise RuntimeError(f"source file hash mismatch: {name}")
    if channels == 4:
        diff = source[..., :3].astype(np.int32) - decoded_pixels[..., :3].astype(np.int32)
        alpha_diff = source[..., 3].astype(np.int32) - decoded_pixels[..., 3].astype(np.int32)
        result["rgb_mse"] = float(np.mean(diff * diff, dtype=np.float64))
        result["alpha_mse"] = float(np.mean(alpha_diff * alpha_diff, dtype=np.float64))
        result["alpha_exact"] = bool(np.array_equal(source[..., 3], decoded_pixels[..., 3]))
    else:
        diff = source.astype(np.int32) - decoded_pixels.astype(np.int32)
        result["rgb_mse"] = float(np.mean(diff * diff, dtype=np.float64))
    expected = EXPECTED_STREAMS.get((name, preset))
    if expected:
        if stream_hash != expected["file_sha256"]:
            raise RuntimeError(f"frozen stream mismatch: {name}/{preset}")
        if result["decoded_pixel_sha256"] != expected["pixel_sha256"]:
            raise RuntimeError(f"frozen reconstruction mismatch: {name}/{preset}")
        if result["rgb_mse"] != expected["rgb_mse"]:
            raise RuntimeError(f"frozen MSE mismatch: {name}/{preset}")
    return result


def main():
    rows = []
    images = {}
    for name in IMAGES:
        with Image.open(ROOT / "hpt_data" / name) as image:
            images[name] = ImageOps.exif_transpose(image).convert("RGB")
    for preset in PRESETS:
        PBC3.compress(images["coast.jpg"], config=getattr(PBC3Config, preset)(), rng_mode="release")
    for name, image in images.items():
        for preset in PRESETS:
            row = run_case(name, image, preset)
            row["source_file_sha256"] = hashlib.sha256((ROOT / "hpt_data" / name).read_bytes()).hexdigest()
            rows.append(row)
    y, x = np.mgrid[:64, :96]
    rgba = np.empty((64, 96, 4), dtype=np.uint8)
    rgba[:, :, 0] = (3 * x + y) % 256
    rgba[:, :, 1] = (x + 4 * y) % 256
    rgba[:, :, 2] = (2 * x + 2 * y) % 256
    rgba[:, :, 3] = ((x + y) % 4) * 85
    rgba_image = Image.fromarray(rgba, "RGBA")
    for preset in PRESETS:
        rows.append(run_case("synthetic_rgba_96x64", rgba_image, preset))
    print(json.dumps({"source": "published release ZIP embedded Python", "threads": 1, "images": list(IMAGES), "presets": list(PRESETS), "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
