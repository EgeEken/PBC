import json
import pathlib
import random
import sys

import numpy as np
import PIL

ROOT = pathlib.Path(__file__).resolve().parents[2]
PASS = ROOT / "research_pass40"
OUT = PASS / "outputs" / "ablation"
sys.path.insert(0, str(ROOT / "research_pass39" / "scripts"))
import benchmark as p39

OUT.mkdir(parents=True, exist_ok=True)
result_path = OUT / "results.json"
if result_path.exists():
    raise FileExistsError(result_path)
p39.OUT = OUT
configs = {
    "REF": ["--advanced", "aq-mode=0", "--advanced", "tune=psnr", "--advanced", "sb-size=64"],
    "A": ["--advanced", "aq-mode=0", "--advanced", "tune=psnr", "--advanced", "deltaq-mode=0", "--advanced", "sb-size=64"],
    "B": ["--advanced", "aq-mode=0", "--advanced", "tune=psnr", "--advanced", "deltaq-mode=6", "--advanced", "sb-size=64"],
    "C": ["--advanced", "aq-mode=0", "--advanced", "tune=iq", "--advanced", "deltaq-mode=0", "--advanced", "sb-size=64"],
    "D": ["--advanced", "aq-mode=0", "--advanced", "tune=iq", "--advanced", "deltaq-mode=6", "--advanced", "sb-size=64"],
}
settings = {"depth": 8, "yuv": "420", "range": "full", "jobs": 1, "speed": 6,
            "superblock_size": 64, "cicp": "1/13/1", "metadata": "ignored",
            "quality_points": {"googlewiki.png": [5, 20], "cat.jpg": [35, 50], "coast.jpg": [80, 90]},
            "warmups": 1, "timed_repetitions": 3,
            "mse": "RGB arrays promoted to float64 before subtraction and squaring"}
rows = []
p39.verify_assets()
if float(np.mean((np.array([0.0]) - np.array([255.0])) ** 2)) != 65025.0:
    raise RuntimeError("float64 MSE fixture failed")

for name, qualities in settings["quality_points"].items():
    src = ROOT / "hpt_data" / name
    for q in qualities:
        order = list(configs)
        random.Random(f"p40-{name}-{q}").shuffle(order)
        for config in order:
            p39.CONFIGS = configs
            row, _, _ = p39.one(src, name, q, config, 6, 0, "warm")
            row["source_sha256"] = p39.digest(src)
            rows.append(row)
        for repeat in range(1, 4):
            rotated = order[repeat % len(order):] + order[:repeat % len(order)]
            for config in rotated:
                row, _, _ = p39.one(src, name, q, config, 6, repeat, "timed")
                row["source_sha256"] = p39.digest(src)
                rows.append(row)
                print(f"{name:14} q{q:2} {config:3} r{repeat} {row['bytes']:8}B {row['bpp']:.4f}bpp MSE {row['rgb_mse']:.4f} enc {row['encode_seconds']:.3f}s", flush=True)
        result_path.write_text(json.dumps({"sources": {"libaom": "3.12.1", "libaom_commit": "10aece4157eb79315da205f39e19bf6ab3ee30d0", "libavif": "1.3.0", "libavif_commit": "1aadfad932c98c069a1204261b1856f81f3bc199"},
                                           "build": {"avifenc_sha256": p39.digest(p39.BIN / "avifenc.exe"), "avifdec_sha256": p39.digest(p39.BIN / "avifdec.exe"), "ssimulacra2_sha256": p39.digest(p39.SSIMULACRA), "python": sys.version.split()[0], "numpy": np.__version__, "pillow": PIL.__version__},
                                           "settings": settings, "configurations": configs, "rows": rows}, indent=2) + "\n")
    print(f"completed {name} q{q}", flush=True)
