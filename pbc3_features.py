import numpy as np
from numba import njit


FEATURE_NAMES = [
    "w_frac", "h_frac", "area_frac", "log_aspect", "channel", "step_frac", "q",
    "res_mean", "res_std", "res_abs_mean", "res_rms", "res_min", "res_max",
    "res_p05", "res_p25", "res_p50", "res_p75", "res_p95",
    "pos_frac", "neg_frac", "zeroish_frac", "uniq_frac",
    "mean_abs_dx", "mean_abs_dy", "grad_energy", "lap_abs_mean",
    "target_mean", "target_std", "canvas_mean", "canvas_std", "before_mse", "before_mae",
]
FEATURE_DIM = len(FEATURE_NAMES)
SORT_FEATURES = {"res_p05", "res_p25", "res_p50", "res_p75", "res_p95", "uniq_frac"}


def residual_sum_error(target, canvas, channel: int) -> float:
    """## Returns visible absolute error for one channel"""
    return float(np.sum(np.abs(target[:, :, channel] - np.clip(canvas[:, :, channel], 0, 255))))


def extract_global_features(target, canvas, channel: int, step: int = 0, patch_count: int = 1) -> dict:
    """## Returns simple global error stats for one channel"""
    cv = np.clip(canvas[:, :, channel], 0, 255).astype(np.int32)
    err = (target[:, :, channel] - cv).astype(np.float64)
    abs_err = np.abs(err)
    return {
        "channel": int(channel),
        "step": int(step),
        "step_frac": float(step) / max(1, int(patch_count)),
        "sum_error": float(abs_err.sum()),
        "mse": float((err ** 2).mean()),
        "mae": float(abs_err.mean()),
        "rms": float(np.sqrt((err ** 2).mean())),
    }


@njit(cache=True)
def _integer_order_stats(flat, lo, hi):
    counts = np.zeros(hi - lo + 1, dtype=np.int64)
    for value in flat:
        counts[int(value) - lo] += 1
    ranks = np.array([0.05, 0.25, 0.5, 0.75, 0.95]) * (flat.size - 1)
    lower = np.floor(ranks).astype(np.int64)
    upper = np.ceil(ranks).astype(np.int64)
    a = np.empty(5, dtype=np.float64)
    b = np.empty(5, dtype=np.float64)
    cumulative = 0
    unique = 0
    previous = np.inf
    for i in range(counts.size):
        count = counts[i]
        if count == 0:
            continue
        value = float(i + lo)
        quantized = np.rint(value / 2.0)
        if quantized != previous:
            unique += 1
            previous = quantized
        end = cumulative + count
        for j in range(5):
            if cumulative <= lower[j] < end:
                a[j] = value
            if cumulative <= upper[j] < end:
                b[j] = value
        cumulative = end
    for j in range(5):
        gamma = ranks[j] - lower[j]
        diff = b[j] - a[j]
        if gamma >= 0.5:
            a[j] = b[j] - diff * (1.0 - gamma)
        else:
            a[j] = a[j] + diff * gamma
    return a, unique / flat.size


def extract_features(target, canvas, box, step: int, q: float, image_w: int, image_h: int, patch_count: int, n_channels: int) -> np.ndarray:
    """## Returns the full learned-filler feature vector for one candidate box"""
    c, x, y, bw, bh = box
    t = target[y:y + bh, x:x + bw, c].astype(np.float64)
    cv = canvas[y:y + bh, x:x + bw, c].astype(np.float64)
    res = t - cv
    before = t - np.clip(cv, 0, 255)
    flat = res.ravel()
    n = max(1, flat.size)
    absflat = np.abs(flat)
    lo, hi = flat.min(), flat.max()
    if target.dtype.kind in "iu" and canvas.dtype.kind in "iu" and -2**52 <= lo <= hi <= 2**52 and hi - lo <= 65535:
        percentiles, uniq = _integer_order_stats(flat, int(lo), int(hi))
        p05, p25, p50, p75, p95 = percentiles
    else:
        p05, p25, p50, p75, p95 = np.percentile(flat, [5, 25, 50, 75, 95])
        uniq = np.unique(np.rint(flat / 2.0)).size / n
    dx = float(np.abs(np.diff(res, axis=1)).mean()) if bw > 1 else 0.0
    dy = float(np.abs(np.diff(res, axis=0)).mean()) if bh > 1 else 0.0
    lapx = float(np.abs(res[:, 2:] - 2 * res[:, 1:-1] + res[:, :-2]).mean()) if bw > 2 else 0.0
    lapy = float(np.abs(res[2:] - 2 * res[1:-1] + res[:-2]).mean()) if bh > 2 else 0.0
    return np.array([
        bw / image_w,
        bh / image_h,
        (bw * bh) / (image_w * image_h),
        np.log((bw + 1e-6) / (bh + 1e-6)),
        c / max(1, n_channels - 1),
        step / max(1, patch_count),
        q,
        float(flat.mean()),
        float(flat.std()),
        float(absflat.mean()),
        float(np.sqrt((flat ** 2).mean())),
        float(flat.min()),
        float(flat.max()),
        p05,
        p25,
        p50,
        p75,
        p95,
        float((flat > 0).mean()),
        float((flat < 0).mean()),
        float((absflat < 2).mean()),
        uniq,
        dx,
        dy,
        dx + dy,
        lapx + lapy,
        float(t.mean()),
        float(t.std()),
        float(cv.mean()),
        float(cv.std()),
        float((before ** 2).mean()),
        float(np.abs(before).mean()),
    ], dtype=np.float32)


def extract_cheap(names, target, canvas, box, step: int, q: float, image_w: int, image_h: int, patch_count: int, n_channels: int) -> np.ndarray:
    """## Returns only the requested cheap feature subset for one candidate box"""
    c, x, y, bw, bh = box
    t = target[y:y + bh, x:x + bw, c].astype(np.float64)
    cv = canvas[y:y + bh, x:x + bw, c].astype(np.float64)
    res = t - cv
    flat = res.ravel()
    absflat = np.abs(flat)
    d = {
        "w_frac": bw / image_w,
        "h_frac": bh / image_h,
        "area_frac": (bw * bh) / (image_w * image_h),
        "log_aspect": np.log((bw + 1e-6) / (bh + 1e-6)),
        "channel": c / max(1, n_channels - 1),
        "step_frac": step / max(1, patch_count),
        "q": q,
        "res_mean": flat.mean(),
        "res_std": flat.std(),
        "res_abs_mean": absflat.mean(),
        "res_rms": np.sqrt((flat ** 2).mean()),
        "res_min": flat.min(),
        "res_max": flat.max(),
        "pos_frac": (flat > 0).mean(),
        "neg_frac": (flat < 0).mean(),
        "zeroish_frac": (absflat < 2).mean(),
        "target_mean": t.mean(),
        "target_std": t.std(),
        "canvas_mean": cv.mean(),
        "canvas_std": cv.std(),
    }
    nameset = set(names)
    if {"mean_abs_dx", "grad_energy"} & nameset:
        d["mean_abs_dx"] = float(np.abs(np.diff(res, axis=1)).mean()) if bw > 1 else 0.0
    if {"mean_abs_dy", "grad_energy"} & nameset:
        d["mean_abs_dy"] = float(np.abs(np.diff(res, axis=0)).mean()) if bh > 1 else 0.0
    if "grad_energy" in names:
        d["grad_energy"] = d["mean_abs_dx"] + d["mean_abs_dy"]
    if "lap_abs_mean" in names:
        lx = float(np.abs(res[:, 2:] - 2 * res[:, 1:-1] + res[:, :-2]).mean()) if bw > 2 else 0.0
        ly = float(np.abs(res[2:] - 2 * res[1:-1] + res[:-2]).mean()) if bh > 2 else 0.0
        d["lap_abs_mean"] = lx + ly
    before = t - np.clip(cv, 0, 255)
    if "before_mse" in names:
        d["before_mse"] = float((before ** 2).mean())
    if "before_mae" in names:
        d["before_mae"] = float(np.abs(before).mean())
    return np.array([d[nm] for nm in names], dtype=np.float32)
