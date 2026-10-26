import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numba
import numpy as np
from PIL import Image, ImageOps

from PBC3 import PBC3, preload_numba
from pbc3_types import PBC3Config
from pbc31_multistage import (
    MultistagePBC31,
    SINGLE_STAGE,
    T2_COARSE_INIT_ONLY,
    T2_05_95,
    T2_10_90,
    T2_20_80,
    T2_40_60,
    T2_075_10_90,
    T2_075_20_80,
    T3_10_20_70,
)


ROOT = Path(__file__).resolve().parent
PRESETS = ("compression", "balanced", "quality", "high_quality")
DIAGNOSTIC_IMAGES = ("flower.jpg", "monet.jpg", "googlewiki.png", "cat.jpg", "panorama.png")
WARMUP_VARIANTS = {
    "single_stage": None,
    "W1": (0.25, 0.10),
    "W2": (0.25, 0.25),
    "W3": (0.25, 0.50),
    "W4": (0.0625, 0.10),
    "W5": (0.0625, 0.25),
    "W6": (0.0625, 0.50),
}
MULTISTAGE_PLANS = (
    SINGLE_STAGE,
    T2_10_90,
    T2_20_80,
    T2_40_60,
    T3_10_20_70,
)
REFINED_PLANS = (T2_05_95, T2_075_10_90, T2_075_20_80)
RD_PATCH_COUNTS = {
    "compression": (30, 35, 40, 45, 50),
    "balanced": (30, 35, 40, 45, 50),
    "quality": (30, 35, 40, 45, 50),
    "high_quality": (10, 14, 16, 18, 20),
}


def load_images(names):
    images = []
    for name in names:
        path = ROOT / "hpt_data" / name
        image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        images.append((name, image, np.asarray(image)))
    return images


def load_all_images(directory="hpt_data"):
    names = sorted(
        path.name
        for path in (ROOT / directory).iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    )
    images = []
    for name in names:
        path = ROOT / directory / name
        image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        images.append((name, image, np.asarray(image)))
    return images


def mse(reference, reconstructed):
    delta = reference.astype(np.float32) - reconstructed.astype(np.float32)
    return float(np.mean(delta * delta))


def evaluate(codec, config, images, source, variant, preset):
    rows = []
    for name, image, reference in images:
        result = codec.compress(image, config=config)
        start = time.perf_counter()
        decoded = codec.decompress(result.data)
        decode_seconds = time.perf_counter() - start
        reconstructed = np.asarray(decoded.image.convert("RGB").resize(image.size))
        _, raw_body = codec._open_body(result.data)
        rows.append({
            "source": source,
            "variant": variant,
            "preset": preset,
            "image": name,
            "width": image.width,
            "height": image.height,
            "mse": mse(reference, reconstructed),
            "bpp": len(result.data) * 8 / (image.width * image.height),
            "encode_seconds": float(result.encode_seconds),
            "decode_seconds": float(decode_seconds),
            "raw_bytes": len(raw_body),
            "final_bytes": len(result.data),
            "roundtrip_equal": bool(np.array_equal(np.asarray(result.image), np.asarray(decoded.image))),
        })
        print(
            f"{source:7s} {variant:12s} {preset:12s} {name:18s} "
            f"MSE {rows[-1]['mse']:.3f} bpp {rows[-1]['bpp']:.5f} "
            f"enc {rows[-1]['encode_seconds']:.3f}s dec {rows[-1]['decode_seconds']:.4f}s",
            flush=True,
        )
    return rows


def evaluate_old(images, presets):
    payload = [{"name": name, "path": str(ROOT / "hpt_data" / name)} for name, _, _ in images]
    code = r"""
import json
import os
import sys
import time

import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, sys.argv[1])
from PBC3 import PBC3, preload_numba
from pbc3_types import PBC3Config

payload = json.loads(sys.stdin.read())
presets = json.loads(sys.argv[2])
preload_numba()
rows = []
for preset in presets:
    config = getattr(PBC3Config, preset)()
    for item in payload:
        image = ImageOps.exif_transpose(Image.open(item["path"])).convert("RGB")
        reference = np.asarray(image)
        result = PBC3.compress(image, config=config)
        start = time.perf_counter()
        decoded = PBC3.decompress(result.data)
        decode_seconds = time.perf_counter() - start
        reconstructed = np.asarray(decoded.image.convert("RGB").resize(image.size))
        delta = reference.astype(np.float32) - reconstructed.astype(np.float32)
        _, raw_body = PBC3._open_body(result.data)
        rows.append({
            "source": "old_pbc",
            "variant": "single_stage",
            "preset": preset,
            "image": item["name"],
            "width": image.width,
            "height": image.height,
            "mse": float(np.mean(delta * delta)),
            "bpp": len(result.data) * 8 / (image.width * image.height),
            "encode_seconds": float(result.encode_seconds),
            "decode_seconds": float(decode_seconds),
            "raw_bytes": len(raw_body),
            "final_bytes": len(result.data),
            "roundtrip_equal": bool(np.array_equal(np.asarray(result.image), np.asarray(decoded.image))),
        })
print(json.dumps(rows))
"""
    env = os.environ.copy()
    old_dir = str(ROOT / "old_pbc")
    env["PYTHONPATH"] = old_dir + os.pathsep + env.get("PYTHONPATH", "")
    process = subprocess.run(
        [sys.executable, "-c", code, old_dir, json.dumps(list(presets))],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        cwd=ROOT,
        env=env,
        check=False,
    )
    if process.returncode:
        raise RuntimeError(process.stderr[-2000:])
    rows = json.loads(process.stdout)
    for row in rows:
        print(
            f"old_pbc single_stage {row['preset']:12s} {row['image']:18s} "
            f"MSE {row['mse']:.3f} bpp {row['bpp']:.5f} "
            f"enc {row['encode_seconds']:.3f}s dec {row['decode_seconds']:.4f}s",
            flush=True,
        )
    return rows


def evaluate_multistage(config, images, plan, preset, diagnostics=True):
    rows = []
    for name, image, reference in images:
        result = MultistagePBC31.compress(
            image, config=config, plan=plan, diagnostics=diagnostics
        )
        start = time.perf_counter()
        decoded = MultistagePBC31.decompress(result.data)
        decode_seconds = time.perf_counter() - start
        reconstructed = np.asarray(decoded.image.convert("RGB").resize(image.size))
        rows.append({
            "source": "multistage",
            "variant": plan.name,
            "preset": preset,
            "image": name,
            "width": image.width,
            "height": image.height,
            "mse": mse(reference, reconstructed),
            "bpp": len(result.data) * 8 / (image.width * image.height),
            "encode_seconds": float(result.encode_seconds),
            "decode_seconds": float(decode_seconds),
            "raw_bytes": result.timings["raw_bytes"],
            "final_bytes": len(result.data),
            "roundtrip_equal": bool(
                np.array_equal(np.asarray(result.image), np.asarray(decoded.image))
            ),
            "stages": result.timings["stages"],
            "stage_header_bits": result.timings["stage_header_bits"],
        })
        print(
            f"v1      {plan.name:12s} {preset:12s} {name:18s} "
            f"MSE {rows[-1]['mse']:.3f} bpp {rows[-1]['bpp']:.5f} "
            f"enc {rows[-1]['encode_seconds']:.3f}s dec {rows[-1]['decode_seconds']:.4f}s",
            flush=True,
        )
    return rows


def phase0(output):
    images = load_images(DIAGNOSTIC_IMAGES)
    preload_numba(str(ROOT / "patch_policy.npz"))
    rows = evaluate_old(images, PRESETS)
    for preset in PRESETS:
        base = getattr(PBC3Config, preset)()
        for variant, settings in WARMUP_VARIANTS.items():
            config = getattr(PBC3Config, preset)()
            if settings is not None:
                pixel_fraction, early_ratio = settings
                config.auto_downsample_max_pixels = max(
                    1, int(round(base.auto_downsample_max_pixels * pixel_fraction))
                )
                config.warm_downsample_max_pixels = base.auto_downsample_max_pixels
                config.warmup_ratio = early_ratio
            rows.extend(evaluate(PBC3, config, images, "current", variant, preset))
    result = {
        "created": time.time(),
        "mode": "phase0",
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "numpy": np.__version__,
            "numba": numba.__version__,
        },
        "images": list(DIAGNOSTIC_IMAGES),
        "variants": WARMUP_VARIANTS,
        "rows": rows,
    }
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")


def phase1(output, plans=MULTISTAGE_PLANS, mode="phase1"):
    images = load_images(DIAGNOSTIC_IMAGES)
    preload_numba(str(ROOT / "patch_policy.npz"))
    rows = []
    for preset in PRESETS:
        for plan in plans:
            config = getattr(PBC3Config, preset)()
            rows.extend(evaluate_multistage(config, images, plan, preset))
    result = {
        "created": time.time(),
        "mode": mode,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "numpy": np.__version__,
            "numba": numba.__version__,
        },
        "images": list(DIAGNOSTIC_IMAGES),
        "plans": [
            {
                "name": plan.name,
                "scales": plan.scales,
                "budget_ratios": plan.budget_ratios,
            }
            for plan in plans
        ],
        "rows": rows,
    }
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")


def full_main(output):
    images = load_all_images()
    preload_numba(str(ROOT / "patch_policy.npz"))
    rows = evaluate_old(images, PRESETS)
    for preset in PRESETS:
        config = getattr(PBC3Config, preset)()
        rows.extend(evaluate(PBC3, config, images, "current", "single_stage", preset))
        rows.extend(
            evaluate_multistage(
                getattr(PBC3Config, preset)(),
                images,
                T2_05_95,
                preset,
                diagnostics=False,
            )
        )
    result = {
        "created": time.time(),
        "mode": "full_main",
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "numpy": np.__version__,
            "numba": numba.__version__,
        },
        "images": [name for name, _, _ in images],
        "plan": {
            "name": T2_05_95.name,
            "scales": T2_05_95.scales,
            "budget_ratios": T2_05_95.budget_ratios,
        },
        "rows": rows,
    }
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")


def heuristic_diagnostic(output):
    images = load_images(DIAGNOSTIC_IMAGES)
    preload_numba(str(ROOT / "patch_policy.npz"))
    rows = []
    for preset in PRESETS:
        current_config = getattr(PBC3Config, preset)(learned_filler_enabled=False)
        rows.extend(
            evaluate(
                PBC3, current_config, images, "current", "heuristic_single", preset
            )
        )
        multistage_config = getattr(PBC3Config, preset)(learned_filler_enabled=False)
        rows.extend(
            evaluate_multistage(
                multistage_config,
                images,
                T2_05_95,
                preset,
                diagnostics=False,
            )
        )
    result = {
        "created": time.time(),
        "mode": "heuristic_diagnostic",
        "images": list(DIAGNOSTIC_IMAGES),
        "plan": T2_05_95.name,
        "rows": rows,
    }
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")


def closure_diagnostic(output):
    images = load_images(DIAGNOSTIC_IMAGES)
    preload_numba(str(ROOT / "patch_policy.npz"))
    rows = []
    for preset in PRESETS:
        for patch_count in RD_PATCH_COUNTS[preset]:
            config = getattr(PBC3Config, preset)(patch_count=patch_count)
            rows.extend(
                evaluate(
                    PBC3,
                    config,
                    images,
                    "current",
                    f"single_p{patch_count}",
                    preset,
                )
            )
        for plan in (T2_05_95, T2_COARSE_INIT_ONLY):
            config = getattr(PBC3Config, preset)()
            rows.extend(
                evaluate_multistage(
                    config, images, plan, preset, diagnostics=True
                )
            )
    result = {
        "created": time.time(),
        "mode": "closure_diagnostic",
        "images": list(DIAGNOSTIC_IMAGES),
        "patch_counts": RD_PATCH_COUNTS,
        "rows": rows,
    }
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")


def closure_compression_q(output):
    images = load_images(DIAGNOSTIC_IMAGES)
    preload_numba(str(ROOT / "patch_policy.npz"))
    rows = []
    for patch_count in (45, 50):
        for learned_q in (0.20, 0.25, 0.30, 0.35):
            config = PBC3Config.compression(
                patch_count=patch_count,
                learned_filler_q=learned_q,
            )
            rows.extend(
                evaluate(
                    PBC3,
                    config,
                    images,
                    "current",
                    f"single_p{patch_count}_q{learned_q:.2f}",
                    "compression",
                )
            )
    result = {
        "created": time.time(),
        "mode": "closure_compression_q",
        "images": list(DIAGNOSTIC_IMAGES),
        "rows": rows,
    }
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=(
            "phase0",
            "phase1",
            "phase1b",
            "full",
            "heuristic",
            "closure",
            "closure_q",
        ),
        default="phase0",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or ROOT / f"pbc31_{args.mode}_diagnostic.json"
    if args.mode == "phase0":
        phase0(output)
    elif args.mode == "phase1":
        phase1(output)
    elif args.mode == "full":
        full_main(output)
    elif args.mode == "heuristic":
        heuristic_diagnostic(output)
    elif args.mode == "closure":
        closure_diagnostic(output)
    elif args.mode == "closure_q":
        closure_compression_q(output)
    else:
        phase1(output, REFINED_PLANS, args.mode)


if __name__ == "__main__":
    main()
