import csv
import json
import math
import pathlib

import numpy as np
from PIL import Image, ImageOps
from scipy.ndimage import sobel, uniform_filter

ROOT = pathlib.Path(__file__).resolve().parents[2]
PASS = ROOT / "research_pass40"
OUT = PASS / "outputs"
LEDGER = json.loads((OUT / "ablation" / "results.json").read_text())
QMAP_MANIFEST = json.loads((OUT / "qmaps" / "manifest.json").read_text())
QMAPS = {(r["image"], r["qcolor"], r["config"]): r for r in QMAP_MANIFEST["rows"]}
POINTS = sorted(QMAPS)
PHASE_RANK = {"timed": 1, "calibration": 2, "followup": 3}


def read_rgb(path):
    with Image.open(path) as im:
        return np.asarray(ImageOps.exif_transpose(im).convert("RGB"), dtype=np.uint8)


def masks(source):
    v = source.astype(np.float64) / 255.0
    linear = np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)
    luma = linear @ np.array([0.2126, 0.7152, 0.0722])
    edge_score = np.hypot(sobel(luma, axis=1), sobel(luma, axis=0))
    local_var = np.maximum(uniform_filter(luma * luma, size=8) - uniform_filter(luma, size=8) ** 2, 0)
    edge = edge_score >= np.quantile(edge_score, 0.8)
    remain = ~edge
    texture = remain & (local_var >= np.quantile(local_var[remain], 0.8))
    return {"edge": edge, "texture": texture, "smooth": remain & ~texture}


def qmap(image, q, config, shape):
    path = PASS / QMAPS[(image, q, config)]["qmap_csv"]
    rows, cols = math.ceil(shape[0] / 64), math.ceil(shape[1] / 64)
    roots = np.zeros((rows, cols), dtype=np.uint8)
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            roots[int(row["mi_row"]) // 16, int(row["mi_col"]) // 16] = int(row["effective_qindex"])
    expanded = np.repeat(np.repeat(roots, 64, axis=0), 64, axis=1)[:shape[0], :shape[1]]
    return roots, expanded


def select_row(image, q, config):
    choices = [r for r in LEDGER["rows"] if r["image"] == image and r["qcolor"] == q and r["config"] == config and r["phase"] in PHASE_RANK]
    return max(choices, key=lambda r: (PHASE_RANK[r["phase"]], r["repeat"]))


def region_stats(mask, error, qindex):
    count = int(mask.sum())
    sse = float(error[mask].sum())
    qvals, qcounts = np.unique(qindex[mask], return_counts=True)
    return {"pixels": count, "area_fraction": float(mask.mean()), "rgb_mse": sse / (3 * count),
            "sse_fraction": sse / float(error.sum()), "qindex_mean_area_weighted": float(np.average(qvals, weights=qcounts)),
            "qindex_histogram_pixels": {str(int(v)): int(n) for v, n in zip(qvals, qcounts)}}


def main():
    points = {}
    source_cache = {}
    masks_cache = {}
    for image, q, config in POINTS:
        if image not in source_cache:
            source_cache[image] = read_rgb(ROOT / "hpt_data" / image)
            masks_cache[image] = masks(source_cache[image])
        source = source_cache[image]
        row = select_row(image, q, config)
        recon_file = OUT / "ablation" / f"s6_{pathlib.Path(image).stem}_q{q}_{config}_{row['phase']}{row['repeat']}.png"
        recon = read_rgb(recon_file)
        error = np.sum((recon.astype(np.float64) - source.astype(np.float64)) ** 2, axis=2)
        roots, expanded = qmap(image, q, config, source.shape[:2])
        hist, counts = np.unique(roots, return_counts=True)
        key = (image, q, config)
        points[key] = {"image": image, "qcolor": q, "config": config, "phase": row["phase"], "bytes": row["bytes"], "bpp": row["bpp"],
                       "rgb_mse": row["rgb_mse"], "ssimulacra2": row["ssimulacra2"], "encode_samples_seconds": [r["encode_seconds"] for r in LEDGER["rows"] if r["image"] == image and r["qcolor"] == q and r["config"] == config and r["phase"] == row["phase"]],
                       "qindex_min": int(roots.min()), "qindex_max": int(roots.max()), "qindex_mean": float(roots.mean()),
                       "qindex_histogram_superblocks": {str(int(v)): int(n) for v, n in zip(hist, counts)}, "regions": {}}
        for region, mask in masks_cache[image].items():
            points[key]["regions"][region] = region_stats(mask, error, expanded)

    pairs = []
    for image, on_config, off_config in ((im, "B", "A") for im in source_cache):
        pairs.extend(find_matches(image, on_config, off_config, points, masks_cache))
    for image in source_cache:
        pairs.extend(find_matches(image, "D", "C", points, masks_cache))

    result = {"mask_method": "linear-light luma; top 20% Sobel magnitude as edge; among remaining pixels top 20% local 8x8 variance as texture; rest smooth; masks fixed per image",
              "qindex_method": "instrumented coded-superblock-root qindex; all variants use 64x64 superblocks; spatial qindex is not regional bitrate attribution",
              "point_count": len(points), "points": list(points.values()), "matched_rate_pairs": pairs}
    (OUT / "spatial_analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"spatial points={len(points)} near-rate pairs={sum(p['matched'] for p in pairs)}")
    for p in pairs:
        if p["matched"]:
            print(f"{p['image']:14} {p['active_config']}/{p['off_config']} q{p['active_qcolor']} vs q{p['off_qcolor']} rate {p['rate_gap']*100:.2f}% MSE {p['mse_change_percent']:+.2f}% S2 {p['ssimulacra2_change']:+.3f}")


def find_matches(image, on_config, off_config, points, masks_cache):
    on_points = [p for (im, _, config), p in points.items() if im == image and config == on_config and p["phase"] == "calibration"]
    off_points = [p for (im, _, config), p in points.items() if im == image and config == off_config]
    pairs = []
    for on in on_points:
        candidates = [(abs(on["bpp"] - off["bpp"]) / ((on["bpp"] + off["bpp"]) / 2), off) for off in off_points]
        gap, off = min(candidates, key=lambda item: item[0])
        matched = gap <= 0.05
        pair = {"image": image, "active_config": on_config, "off_config": off_config, "active_qcolor": on["qcolor"],
                "off_qcolor": off["qcolor"], "active_bytes": on["bytes"], "off_bytes": off["bytes"],
                "active_bpp": on["bpp"], "off_bpp": off["bpp"], "rate_gap": gap, "matched": matched,
                "active_rgb_mse": on["rgb_mse"], "off_rgb_mse": off["rgb_mse"],
                "mse_change_percent": (on["rgb_mse"] / off["rgb_mse"] - 1) * 100,
                "active_ssimulacra2": on["ssimulacra2"], "off_ssimulacra2": off["ssimulacra2"],
                "ssimulacra2_change": on["ssimulacra2"] - off["ssimulacra2"], "regions": {}}
        for region in ("edge", "texture", "smooth"):
            active, base = on["regions"][region], off["regions"][region]
            pair["regions"][region] = {"active_rgb_mse": active["rgb_mse"], "off_rgb_mse": base["rgb_mse"],
                                        "mse_change_percent": (active["rgb_mse"] / base["rgb_mse"] - 1) * 100,
                                        "active_qindex_mean": active["qindex_mean_area_weighted"],
                                        "off_qindex_mean": base["qindex_mean_area_weighted"]}
        pairs.append(pair)
    return pairs


if __name__ == "__main__":
    main()
