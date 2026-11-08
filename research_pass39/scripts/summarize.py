import hashlib
import json
import math
import pathlib
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchmark import SSIMULACRA

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "research_pass39" / "outputs" / "benchmark"
VIS = ROOT / "research_pass39" / "visuals"
RESULT = json.loads((OUT / "results.json").read_text())
ROWS = RESULT["rows"]
CONFIG_ORDER = ["REF", "A", "B", "C", "D"]
COLORS = {"REF": "#777777", "A": "#1f77b4", "B": "#2ca02c", "C": "#d62728", "D": "#9467bd"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def aggregate(speed=6):
    groups = {}
    for row in ROWS:
        if row["speed"] == speed and row["phase"] in ("timed", "calibration"):
            groups.setdefault((row["image"], row["qcolor"], row["config"]), []).append(row)
    out = []
    for (image, q, config), rows in sorted(groups.items()):
        out.append({
            "image": image, "qcolor": q, "config": config, "speed": speed,
            "bytes": int(statistics.mean(r["bytes"] for r in rows)),
            "bpp": statistics.mean(r["bpp"] for r in rows),
            "rgb_mse": statistics.mean(r["rgb_mse"] for r in rows),
            "rgb_psnr": statistics.mean(r["rgb_psnr"] for r in rows),
            "ssimulacra2": statistics.mean(r["ssimulacra2"] for r in rows),
            "encode_median_seconds": statistics.median(r["encode_seconds"] for r in rows),
            "encode_mean_seconds": statistics.mean(r["encode_seconds"] for r in rows),
            "encode_stdev_seconds": statistics.stdev(r["encode_seconds"] for r in rows),
            "decode_median_seconds": statistics.median(r["decode_seconds"] for r in rows),
            "decode_mean_seconds": statistics.mean(r["decode_seconds"] for r in rows),
            "decode_stdev_seconds": statistics.stdev(r["decode_seconds"] for r in rows),
            "avif_sha256": rows[0]["avif_sha256"],
            "decoded_rgb_sha256": rows[0]["decoded_rgb_sha256"],
            "repeat_hashes_stable": len({r["avif_sha256"] for r in rows}) == 1,
            "decoder_verified": all(r["decoder_ok"] for r in rows),
            "repetitions": len(rows),
        })
    return out


def matched_pairs(points):
    by_image = {}
    for point in points:
        by_image.setdefault(point["image"], {}).setdefault(point["config"], []).append(point)
    rows = []
    for image, configs in by_image.items():
        for a in configs.get("A", []):
            for config in CONFIG_ORDER:
                if config in ("A",) or config not in configs:
                    continue
                candidates = []
                for b in configs[config]:
                    diff = abs(a["bpp"] - b["bpp"]) / ((a["bpp"] + b["bpp"]) / 2)
                    if diff <= 0.05:
                        candidates.append((diff, b))
                if candidates:
                    diff, b = min(candidates, key=lambda x: x[0])
                    rows.append({
                        "image": image, "config_a": "A", "config_b": config,
                        "a_qcolor": a["qcolor"], "b_qcolor": b["qcolor"],
                        "a_bytes": a["bytes"], "b_bytes": b["bytes"],
                        "a_bpp": a["bpp"], "b_bpp": b["bpp"],
                        "rate_difference_fraction": diff,
                        "a_rgb_mse": a["rgb_mse"], "b_rgb_mse": b["rgb_mse"],
                        "mse_change_percent": (b["rgb_mse"] / a["rgb_mse"] - 1) * 100,
                        "a_ssimulacra2": a["ssimulacra2"], "b_ssimulacra2": b["ssimulacra2"],
                        "ssimulacra2_change": b["ssimulacra2"] - a["ssimulacra2"],
                    })
                else:
                    rows.append({"image": image, "a_qcolor": a["qcolor"], "config_a": "A", "config_b": config, "near_rate_pair": False})
    return rows


def plot_rd(points):
    images = list(dict.fromkeys(p["image"] for p in points))
    fig, axes = plt.subplots(2, len(images), figsize=(17, 7), constrained_layout=True)
    for col, image in enumerate(images):
        subset = [p for p in points if p["image"] == image]
        for config in CONFIG_ORDER:
            pts = sorted((p for p in subset if p["config"] == config), key=lambda p: p["bpp"])
            if not pts:
                continue
            axes[0, col].plot([p["bpp"] for p in pts], [p["rgb_mse"] for p in pts], "o-", label=config, color=COLORS[config])
            axes[1, col].plot([p["bpp"] for p in pts], [p["ssimulacra2"] for p in pts], "o-", label=config, color=COLORS[config])
        axes[0, col].set_title(pathlib.Path(image).stem)
        axes[0, col].set_xlabel("Completed AVIF rate (bpp)")
        axes[1, col].set_xlabel("Completed AVIF rate (bpp)")
        for ax in (axes[0, col], axes[1, col]):
            ax.grid(True, alpha=.25)
    axes[0, 0].set_ylabel("RGB MSE (lower is better)")
    axes[1, 0].set_ylabel("SSIMULACRA 2 (higher is better)")
    axes[0, -1].legend(loc="best", fontsize=8)
    fig.suptitle("Actual speed-6 AVIF operating points; markers are encoded files")
    VIS.mkdir(parents=True, exist_ok=True)
    fig.savefig(VIS / "rate_distortion.png", dpi=180)
    plt.close(fig)


def main():
    s6, s8 = aggregate(6), aggregate(8)
    pairs = matched_pairs(s6)
    metadata = RESULT["build"]
    metadata.update({
        "avifenc_sha256": sha(ROOT / "research_pass35" / "outputs" / "binaries" / "reference" / "avifenc.exe"),
        "avifdec_sha256": sha(ROOT / "research_pass35" / "outputs" / "binaries" / "reference" / "avifdec.exe"),
        "ssimulacra2_sha256": sha(SSIMULACRA),
        "ssimulacra2_package": "libjxl 0.11.2 (WinGet)",
        "ssimulacra2_input": "EXIF-transposed source converted to 8-bit RGB PNG with Pillow; avifdec PNG reconstruction; no ICC profiles; source and reconstruction compared at original pixel dimensions",
    })
    result = {"build": metadata, "settings": RESULT["settings"], "speed6_points": s6, "speed8_points": s8, "matched_rate_pairs": pairs}
    (OUT / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    plot_rd(s6)
    for p in s6:
        print(f"{p['image']:14} q{p['qcolor']:2} {p['config']:3} {p['bytes']:8}B {p['bpp']:.4f} MSE {p['rgb_mse']:.4f} S2 {p['ssimulacra2']:.3f} enc {p['encode_median_seconds']:.3f}s dec {p['decode_median_seconds']:.3f}s hashes={p['repeat_hashes_stable']}")
    print("\nNear-rate pairs (<=5%):")
    for p in pairs:
        if p.get("near_rate_pair", True):
            print(f"{p['image']:14} A q{p['a_qcolor']:2} vs {p['config_b']} q{p['b_qcolor']:2}: rate {p['rate_difference_fraction']*100:.2f}% MSE {p['mse_change_percent']:+.2f}% SSIMULACRA2 {p['ssimulacra2_change']:+.3f}")
        else:
            print(f"{p['image']:14} A vs {p['config_b']}: no measured pair within 5%")


if __name__ == "__main__":
    main()
