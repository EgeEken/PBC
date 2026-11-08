import csv
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
from scipy.ndimage import sobel, uniform_filter

ROOT = pathlib.Path(__file__).resolve().parents[2]
PASS = ROOT / "research_pass39"
OUT = PASS / "outputs" / "benchmark"
QMAP = PASS / "outputs" / "qmap"
VIS = PASS / "visuals"
SRC = ROOT / "hpt_data" / "cat.jpg"


def read_rgb(path):
    with Image.open(path) as im:
        return np.asarray(ImageOps.exif_transpose(im).convert("RGB"), dtype=np.uint8)


def qindex_map(config, shape):
    step = 16 if config in ("C", "D") else 32
    grid = np.full(((shape[0] + step * 4 - 1) // (step * 4),
                    (shape[1] + step * 4 - 1) // (step * 4)), np.nan)
    with (QMAP / f"cat_q35_{config}.csv").open(newline="") as f:
        for row in csv.DictReader(f):
            mi_row, mi_col = int(row["mi_row"]), int(row["mi_col"])
            if mi_row % step == 0 and mi_col % step == 0:
                grid[mi_row // step, mi_col // step] = int(row["effective_qindex"])
    if np.isnan(grid).any():
        raise ValueError(f"missing superblock roots for {config}: {np.isnan(grid).sum()}")
    root_values, root_counts = np.unique(grid.astype(np.int16), return_counts=True)
    expanded = np.repeat(np.repeat(grid, step * 4, axis=0), step * 4, axis=1)[:shape[0], :shape[1]]
    return expanded, int(grid.size), {str(int(v)): int(c) for v, c in zip(root_values, root_counts)}


def main():
    src = read_rgb(SRC)
    rgb = src.astype(np.float64) / 255.0
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    luma = linear @ np.array([0.2126, 0.7152, 0.0722])
    gx, gy = sobel(luma, axis=1), sobel(luma, axis=0)
    edge_score = np.hypot(gx, gy)
    variance = np.maximum(uniform_filter(luma * luma, size=8) - uniform_filter(luma, size=8) ** 2, 0)
    edge_cut = np.quantile(edge_score, 0.8)
    edge = edge_score >= edge_cut
    remaining = ~edge
    texture_cut = np.quantile(variance[remaining], 0.8)
    texture = remaining & (variance >= texture_cut)
    masks = {"edge": edge, "texture": texture, "smooth": remaining & ~texture}

    recons = {}
    maps = {}
    for config in ("A", "B", "C"):
        recons[config] = read_rgb(OUT / f"s6_cat_q35_{config}_timed3.png")
        maps[config] = qindex_map(config, src.shape[:2])
    err = {k: np.sum((v.astype(np.float64) - src.astype(np.float64)) ** 2, axis=2) for k, v in recons.items()}

    manifest = json.loads((ROOT / "research_pass35" / "SOURCE_MANIFEST.json").read_text())
    source_hash = next(item["sha256"] for item in manifest["assets"] if item["name"] == "cat.jpg")
    results = {"source": "hpt_data/cat.jpg", "source_sha256": source_hash,
               "shape": list(src.shape), "qcolor": 35, "region_masks": "linear-light RGB luma; top 20% Sobel magnitude as edge; top 20% local 8x8 variance among remaining as texture; rest smooth",
               "qindex_map_sampling": "one top-left root sample per coded superblock; 64x64 pixels for IQ (C/D), 128x128 for other modes; nearest superblock expansion for visualization",
               "regions": {}, "qindex": {}}
    total = sum(e.sum() for e in err.values())
    for name, mask in masks.items():
        result = {"pixels": int(mask.sum()), "area_fraction": float(mask.mean()), "configs": {}}
        for config, e in err.items():
            sse = float(e[mask].sum())
            result["configs"][config] = {"rgb_mse": sse / (3 * int(mask.sum())), "sse_fraction": sse / float(e.sum())}
        for config in ("B", "C"):
            result["configs"][config]["mse_change_vs_A_percent"] = (result["configs"][config]["rgb_mse"] / result["configs"]["A"]["rgb_mse"] - 1) * 100
        results["regions"][name] = result
    for config, (qmap, sb_count, histogram) in maps.items():
        vals = np.array([int(v) for v in histogram])
        counts = np.array(list(histogram.values()))
        results["qindex"][config] = {"superblocks": sb_count, "min": int(vals.min()), "max": int(vals.max()),
                                      "mean": float(np.average(vals, weights=counts)), "histogram_superblocks": histogram}

    fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    axes[0, 0].imshow(src); axes[0, 0].set_title("Source")
    for i, config in enumerate(("A", "B", "C"), 1):
        axes[0, i].imshow(recons[config]); axes[0, i].set_title(f"{config} reconstruction")
        im = axes[1, i].imshow(maps[config][0], cmap="viridis", vmin=maps["C"][0].min(), vmax=maps["C"][0].max())
        axes[1, i].set_title(f"{config} root qindex map")
    diff = err["C"] - err["A"]
    vmax = np.quantile(np.abs(diff), 0.99)
    d = axes[1, 0].imshow(diff, cmap="coolwarm", vmin=-vmax, vmax=vmax)
    axes[1, 0].set_title("C − A pixel RGB SSE")
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=axes[1, 1:4], shrink=0.75, label="AV1 qindex")
    fig.colorbar(d, ax=axes[1, 0], shrink=0.75, label="RGB squared error difference")
    VIS.mkdir(parents=True, exist_ok=True)
    fig.savefig(VIS / "spatial_allocation.png", dpi=180)
    plt.close(fig)
    (PASS / "outputs" / "benchmark" / "spatial_analysis.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
