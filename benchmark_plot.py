"""
Re-plot a benchmark summary CSV (from benchmark.py) with custom view options,
without re-running any compression.

Example:
    from benchmark_plot import plot_benchmark
    plot_benchmark("bench_summary.csv",
                   include=("JPEG", "AVIF", "PBC3"),   # or exclude=("PNG",)
                   xlim=(0, 2.0), ylim=(0, 300),
                   yscale="linear")
"""

import csv
import matplotlib.pyplot as plt

# codec -> (matplotlib style, is_line, color)
_STYLE = {
    "JPEG": ("-o", True, None),
    "AVIF": ("-s", True, None),
    "PBC3": ("*", False, "red"),
    "PBC2": ("D", False, "purple"),
    "PNG": ("P", False, "green"),
}


def load_summary(csv_path):
    rows = []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append({"codec": r["codec"], "setting": r["setting"],
                         "mse": float(r["avg_mse"]), "bpp": float(r["avg_bpp"]),
                         "n": int(r.get("n", 0) or 0)})
    return rows


def plot_benchmark(csv_path,
                   include=None,            # iterable of codecs to keep (None = all)
                   exclude=None,            # iterable of codecs to drop
                   xlim=None,               # (min_bpp, max_bpp)
                   ylim=None,               # (min_mse, max_mse)
                   xscale="linear",         # "linear" | "log" | "symlog"
                   yscale="symlog",         # "linear" | "log" | "symlog"
                   annotate_codecs=("PBC3",),  # codecs whose points get text labels
                   line_codecs=("JPEG", "AVIF"),  # codecs drawn as sorted curves
                   markersize=6,
                   figsize=(9.5, 6.5), dpi=130,
                   title="Compression benchmark",
                   out_path=None,           # save path; None = don't save
                   show=True):
    rows = load_summary(csv_path)
    codecs = []
    for r in rows:
        if r["codec"] not in codecs:
            codecs.append(r["codec"])
    if include is not None:
        codecs = [c for c in codecs if c in set(include)]
    if exclude is not None:
        codecs = [c for c in codecs if c not in set(exclude)]

    plt.figure(figsize=figsize, dpi=dpi)
    for codec in codecs:
        pts = sorted((r for r in rows if r["codec"] == codec), key=lambda r: r["bpp"])
        if not pts:
            continue
        style, default_line, color = _STYLE.get(codec, ("o", False, None))
        x = [p["bpp"] for p in pts]
        y = [p["mse"] for p in pts]
        as_line = codec in line_codecs if line_codecs is not None else default_line
        if as_line:
            plt.plot(x, y, style if any(ch in style for ch in "-:") else "-o",
                     markersize=markersize, label=codec, color=color)
        else:
            marker = style.lstrip("-:")
            plt.scatter(x, y, marker=marker or "o", s=markersize ** 2 + 30,
                        color=color, zorder=5, label=codec)
        if annotate_codecs and codec in set(annotate_codecs):
            for p in pts:
                plt.annotate(p["setting"], (p["bpp"], p["mse"]), fontsize=7,
                             xytext=(4, 4), textcoords="offset points")

    plt.xscale(xscale)
    plt.yscale(yscale)
    if xlim:
        plt.xlim(*xlim)
    if ylim:
        plt.ylim(*ylim)
    plt.xlabel("bpp  (compression — lower is better)")
    plt.ylabel("MSE  (quality — lower is better)")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path)
        print(f"wrote {out_path}")
    if show:
        plt.show()


if __name__ == "__main__":
    plot_benchmark("bench_summary.csv", exclude=("PNG",), out_path="bench_plot.png")
