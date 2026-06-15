"""
PBC3 benchmark harness.

Runs JPEG and AVIF at multiple quality settings over a folder of images,
averages MSE (quality) and bpp (compression) per setting, and plots them on a
shared quality/compression graph. PBC3 configs are overlaid as a scatter, with
optional single points for PBC2 (default) and lossless PNG. Everything is also
written to CSV (summary + per-image detail) so settings can be tweaked later.
"""

import io
import os
import csv
import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from PBC3 import PBC3

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")

JPEG_QUALITIES = (1, 5, 10, 20, 40, 60, 80, 95)
AVIF_QUALITIES = (0, 2, 4, 8, 10, 20, 40, 60, 80, 95)


def load_images(folder):
    paths = sorted(p for p in glob.glob(os.path.join(folder, "*")) if p.lower().endswith(IMG_EXTS))
    return [(os.path.basename(p), Image.open(p).convert("RGB")) for p in paths]


def _mse(a, b):
    return float(np.mean((np.asarray(a, np.float32) - np.asarray(b, np.float32)) ** 2))


def _bpp(nbytes, img):
    return nbytes * 8 / (img.width * img.height)


def jpeg_point(img, q):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=int(q))
    dec = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    return _mse(img, dec), _bpp(buf.tell(), img)


def avif_point(img, q):
    buf = io.BytesIO()
    img.save(buf, format="AVIF", quality=int(q))
    dec = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    return _mse(img, dec), _bpp(buf.tell(), img)


def png_point(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return 0.0, _bpp(buf.tell(), img)


def pbc3_point(img, kwargs):
    r = PBC3.compress(img, **kwargs)
    return r.mse, _bpp(len(r.data), img)


def _avif_available():
    try:
        Image.new("RGB", (8, 8)).save(io.BytesIO(), format="AVIF", quality=50)
        return True
    except Exception as e:
        print(f"[AVIF unavailable] {e}\n  -> install `pillow-avif-plugin` or Pillow built with libavif. Skipping AVIF.")
        return False


def run_benchmark(folder, pbc_configs,
                  jpeg_qualities=JPEG_QUALITIES, avif_qualities=AVIF_QUALITIES,
                  include_png=True, pbc2_encode=None, pbc2_label="PBC2 default",
                  out_prefix="bench"):
    """
    folder        : path to a folder of images.
    pbc_configs   : list of (label, kwargs_dict) for PBC3.compress.
    pbc2_encode   : optional callable(img) -> (mse, bpp) for a PBC2 default point.
    """
    images = load_images(folder)
    if not images:
        raise ValueError(f"no images found in {folder}")
    print(f"loaded {len(images)} images")

    detail_rows = []
    summary = []

    def record(codec, setting, fn):
        mses, bpps = [], []
        for name, img in images:
            try:
                m, b = fn(img)
            except Exception as e:
                print(f"  skip {codec}:{setting} on {name}: {e}")
                continue
            mses.append(m)
            bpps.append(b)
            detail_rows.append({"codec": codec, "setting": str(setting), "image": name, "mse": m, "bpp": b})
        if mses:
            summary.append({"codec": codec, "setting": str(setting),
                            "avg_mse": float(np.mean(mses)), "avg_bpp": float(np.mean(bpps)),
                            "n": len(mses)})
            print(f"  {codec:5} {str(setting):24} MSE={np.mean(mses):8.2f}  bpp={np.mean(bpps):7.4f}")

    print("JPEG...")
    for q in jpeg_qualities:
        record("JPEG", q, lambda im, q=q: jpeg_point(im, q))

    if _avif_available():
        print("AVIF...")
        for q in avif_qualities:
            record("AVIF", q, lambda im, q=q: avif_point(im, q))

    print("PBC3...")
    for label, kwargs in pbc_configs:
        record("PBC3", label, lambda im, kw=kwargs: pbc3_point(im, kw))

    if include_png:
        print("PNG (lossless)...")
        record("PNG", "lossless", png_point)

    if pbc2_encode is not None:
        print("PBC2...")
        record("PBC2", pbc2_label, lambda im: pbc2_encode(im))

    with open(f"{out_prefix}_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["codec", "setting", "avg_mse", "avg_bpp", "n"])
        w.writeheader()
        w.writerows(summary)
    with open(f"{out_prefix}_detail.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["codec", "setting", "image", "mse", "bpp"])
        w.writeheader()
        w.writerows(detail_rows)
    print(f"wrote {out_prefix}_summary.csv and {out_prefix}_detail.csv")

    _plot(summary, out_prefix)
    return summary


def _plot(summary, out_prefix, folder=None):
    def curve(codec):
        rows = sorted((r for r in summary if r["codec"] == codec), key=lambda r: r["avg_bpp"])
        return [r["avg_bpp"] for r in rows], [r["avg_mse"] for r in rows], rows

    plt.figure(figsize=(9.5, 6.5), dpi=130)
    for codec, style in (("JPEG", "-o"), ("AVIF", "-s")):
        x, y, _ = curve(codec)
        if x:
            plt.plot(x, y, style, markersize=5, label=codec)

    x, y, rows = curve("PBC3")
    if x:
        plt.scatter(x, y, c="red", marker="*", s=120, zorder=5, label="PBC3")
        for r in rows:
            plt.annotate(r["setting"], (r["avg_bpp"], r["avg_mse"]), fontsize=7,
                         xytext=(4, 4), textcoords="offset points")

    for codec, marker, color in (("PBC2", "D", "purple"), ("PNG", "P", "green")):
        x, y, _ = curve(codec)
        if x:
            plt.scatter(x, y, marker=marker, color=color, s=90, zorder=6, label=codec)

    plt.xlabel("bpp  (compression — lower is better)")
    plt.ylabel("MSE  (quality — lower is better)")
    plt.yscale("symlog")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plot_title = "PBC3 vs "
    for codec in ("JPEG", "AVIF", "PBC2", "PNG"):
        if any(r for r in summary if r["codec"] == codec):
            plot_title += f"{codec} / "
    plot_title = plot_title.rstrip(" / ")
    if folder:
        plot_title += f"\n{folder}"
    plt.title(plot_title)
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_plot.png")
    print(f"wrote {out_prefix}_plot.png")
    plt.show()


if __name__ == "__main__":
    pbc_configs = [
        ("baseline", dict(color_space="YCbCr", patch_count=50)),
        ("baseline-p20", dict(color_space="YCbCr", patch_count=20)),
        ("baseline-p10", dict(color_space="YCbCr", patch_count=10)),
        ("baseline-p20-q99", dict(color_space="YCbCr", patch_count=20, q_start=0.99, q_end=0.99)),
        ("baseline-p20-q95-q50", dict(color_space="YCbCr", patch_count=20, q_start=0.95, q_end=0.50)),
        ("baseline-p20-q5-q1", dict(color_space="YCbCr", patch_count=20, q_start=0.50, q_end=0.1)),
        ("baseline-p50-s100", dict(color_space="YCbCr", patch_count=50, search_depth=100)),
        ("baseline-p50-s200", dict(color_space="YCbCr", patch_count=50, search_depth=200)),
        ("baseline-p50-s200-pro100", dict(color_space="YCbCr", patch_count=50, search_depth=200, proposal_depth=100)),
        
    ]

    # Optional PBC2 point: wire your PBC2 encoder here, returning (mse, bpp).
    from PBC2_4 import PBC
    def pbc2_encode(img):
        res = PBC.compress(img)          # adjust to your PBC2 API
        mse = float(np.mean((np.asarray(img, np.float32) - np.asarray(res.image, np.float32)) ** 2))
        return mse, len(res.data) * 8 / (img.width * img.height)

    run_benchmark("test_images/one_dataset/", pbc_configs, pbc2_encode=pbc2_encode, out_prefix="bench_ducklings")
    