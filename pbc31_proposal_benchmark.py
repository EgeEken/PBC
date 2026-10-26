import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numba
import numpy as np
from PIL import Image, ImageOps

import learned_filler
import pbc3_ops as ops
from PBC3 import PBC3, _canvas_from_bases, preload_numba
from pbc3_types import PBC3Config
from pbc31_multistage_benchmark import evaluate_old, mse
from pbc31_proposal_replay import (
    MIX,
    exact_fallback_proposal_function,
    mixed_propose_boxes,
    replay_image,
)


ROOT = Path(__file__).resolve().parent
PRESETS = ("compression", "balanced", "quality", "high_quality")
IMAGES = {
    "flower.jpg": "smooth_bokeh",
    "monet.jpg": "textured_art",
    "googlewiki.png": "text_synthetic",
    "cat.jpg": "object_portrait",
    "panorama.png": "landscape",
}


def load_images():
    output = []
    for name, image_type in IMAGES.items():
        image = ImageOps.exif_transpose(Image.open(ROOT / "hpt_data" / name)).convert(
            "RGB"
        )
        output.append((name, image_type, image))
    return output


def warm_proposals(image):
    config = PBC3Config.compression()
    prep = PBC3.prepare(image, config)
    height, width, channels = prep["target"].shape
    bases = [
        int(round(float(np.mean(prep["arr"][:, :, channel]))))
        for channel in range(channels)
    ]
    canvas = _canvas_from_bases((height, width, channels), bases)
    mixed_propose_boxes(
        prep["target"],
        canvas,
        config,
        ops.PBC3Rng(config.random_seed),
        0,
        1,
    )


def run_replay(output):
    images = load_images()
    preload_numba(str(ROOT / "patch_policy.npz"))
    warm_proposals(images[0][2])
    rows = []
    for preset in PRESETS:
        config = getattr(PBC3Config, preset)()
        capture_steps = {
            1,
            max(1, int(round(config.patch_count * 0.25))),
            max(1, int(round(config.patch_count * 0.50))),
            max(1, int(round(config.patch_count * 0.75))),
            config.patch_count,
        }
        for image_name, image_type, image in images:
            image_rows = replay_image(
                image,
                config,
                image_name,
                image_type,
                capture_steps,
            )
            for row in image_rows:
                row["preset"] = preset
                rows.append(row)
                print(
                    f"{preset:12s} {image_name:18s} step {row['preset_step']:2d} "
                    f"recall {int(row['top1_recall'])}/"
                    f"{int(row['top4_recall'])}/{int(row['top10_recall'])} "
                    f"proposal {row['mixed_proposal_seconds'] * 1000:.2f}ms",
                    flush=True,
                )
    result = {
        "created": time.time(),
        "mode": "proposal_replay",
        "commit": "039174f144981ae09830f98a225299c1299d2067",
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "numpy": np.__version__,
            "numba": numba.__version__,
        },
        "mixture": MIX,
        "images": IMAGES,
        "rows": rows,
    }
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")


def evaluate_codec(image_name, image, reference, config, variant, expected_data=None):
    result = PBC3.compress(image, config=config)
    decode_start = time.perf_counter()
    decoded = PBC3.decompress(result.data)
    decode_seconds = time.perf_counter() - decode_start
    reconstructed = np.asarray(decoded.image.convert("RGB").resize(image.size))
    _, raw_body = PBC3._open_body(result.data)
    return {
        "source": "current",
        "variant": variant,
        "image": image_name,
        "width": image.width,
        "height": image.height,
        "mse": mse(reference, reconstructed),
        "bpp": len(result.data) * 8 / (image.width * image.height),
        "encode_seconds": float(result.encode_seconds),
        "decode_seconds": float(decode_seconds),
        "raw_bytes": len(raw_body),
        "final_bytes": len(result.data),
        "roundtrip_equal": bool(
            np.array_equal(np.asarray(result.image), np.asarray(decoded.image))
        ),
        "baseline_bytes_equal": (
            None if expected_data is None else result.data == expected_data
        ),
        "_data": result.data,
    }


def run_fallback(output):
    images = load_images()
    preload_numba(str(ROOT / "patch_policy.npz"))
    warm_proposals(images[0][2])
    old_images = [
        (name, image, np.asarray(image))
        for name, _, image in images
    ]
    rows = evaluate_old(old_images, PRESETS)
    for preset in PRESETS:
        config = getattr(PBC3Config, preset)()
        for image_name, _, image in images:
            reference = np.asarray(image)
            PBC3.compress(image, config=config)
            baseline = evaluate_codec(
                image_name, image, reference, config, "single_stage"
            )
            baseline_data = baseline.pop("_data")
            baseline["preset"] = preset
            rows.append(baseline)
            original = learned_filler.propose_boxes
            learned_filler.propose_boxes = exact_fallback_proposal_function
            try:
                fallback = evaluate_codec(
                    image_name,
                    image,
                    reference,
                    getattr(PBC3Config, preset)(),
                    "mixed_exact_fallback",
                    expected_data=baseline_data,
                )
            finally:
                learned_filler.propose_boxes = original
            fallback.pop("_data")
            fallback["preset"] = preset
            rows.append(fallback)
            print(
                f"{preset:12s} {image_name:18s} "
                f"bytes_equal={fallback['baseline_bytes_equal']} "
                f"enc {baseline['encode_seconds']:.3f}s -> "
                f"{fallback['encode_seconds']:.3f}s",
                flush=True,
            )
    result = {
        "created": time.time(),
        "mode": "proposal_exact_fallback",
        "mixture": MIX,
        "images": IMAGES,
        "rows": rows,
    }
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("replay", "fallback"), default="replay")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "pbc31_proposal_replay.json"
    )
    args = parser.parse_args()
    if args.mode == "replay":
        run_replay(args.output)
    else:
        run_fallback(args.output)


if __name__ == "__main__":
    main()
