# Numba-accelerated search kernels for PBC3 (experimental, opt-in).
#
# Enable with:
#     import pbc3_kernels; pbc3_kernels.enable()
# Disable (restore pure-numpy methods) with:
#     pbc3_kernels.disable()
#
# These are NOT bit-compatible with the pure-numpy path: numba sums floats in a
# different order than numpy's pairwise summation, so argsort/argpartition ties
# may occasionally flip and select different patches. Output stays quality-
# equivalent but the stream (and GOLDEN_FILE) will differ. Validate with verify()
# (kernel-level numeric agreement) and re-check corpus quality before relying on it.

import numpy as np

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        def wrap(f):
            return f
        return wrap(args[0]) if args and callable(args[0]) else wrap


@njit(cache=True)
def box_cell_bound(integral, x, y, bw, bh, cell_size):
    nx = (bw + cell_size - 1) // cell_size
    ny = (bh + cell_size - 1) // cell_size
    total = 0.0
    for iy in range(ny):
        y0 = y + iy * cell_size
        y1 = y + bh if iy == ny - 1 else y + (iy + 1) * cell_size
        for ix in range(nx):
            x0 = x + ix * cell_size
            x1 = x + bw if ix == nx - 1 else x + (ix + 1) * cell_size
            s = float(integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0])
            total += s * s / ((y1 - y0) * (x1 - x0))
    return total


@njit(cache=True)
def base_cell_size(res, max_cell):
    h, w = res.shape
    s = 0.0
    for i in range(h):
        for j in range(w):
            v = res[i, j]
            s += v if v >= 0 else -v
    mean_abs = s / (h * w)
    if mean_abs <= 0.0:
        return max_cell
    gx = 0.0
    if w > 1:
        for i in range(h):
            for j in range(w - 1):
                d = res[i, j + 1] - res[i, j]
                gx += d if d >= 0 else -d
        gx /= h * (w - 1)
    gy = 0.0
    if h > 1:
        for i in range(h - 1):
            for j in range(w):
                d = res[i + 1, j] - res[i, j]
                gy += d if d >= 0 else -d
        gy /= (h - 1) * w
    ratio = (gx + gy) / (mean_abs + 1.0)
    if ratio < 0.25:
        return 32
    if ratio < 0.5:
        return 16
    if ratio < 1.0:
        return 8
    return 4


@njit(cache=True)
def anchor_block_scores(err, block_size):
    h, w = err.shape
    ny = (h + block_size - 1) // block_size
    nx = (w + block_size - 1) // block_size
    n = ny * nx
    scores = np.empty(n, dtype=np.float64)
    ys = np.empty(n, dtype=np.int64)
    xs = np.empty(n, dtype=np.int64)
    k = 0
    for by in range(ny):
        y0 = by * block_size
        y1 = min(h, y0 + block_size)
        for bx in range(nx):
            x0 = bx * block_size
            x1 = min(w, x0 + block_size)
            s = 0.0
            for i in range(y0, y1):
                for j in range(x0, x1):
                    s += err[i, j]
            scores[k] = s / ((y1 - y0) * (x1 - x0))
            ys[k] = (y0 + y1 - 1) // 2
            xs[k] = (x0 + x1 - 1) // 2
            k += 1
    return scores, ys, xs


def _patched_box_cell_bound(cls, integral, x, y, bw, bh, cell_size):
    return box_cell_bound(np.ascontiguousarray(integral, dtype=np.int64),
                          int(x), int(y), int(bw), int(bh), int(cell_size))


def _patched_base_cell_size(cls, residual_patch, config):
    res = np.ascontiguousarray(residual_patch, dtype=np.float64)
    if res.size == 0:
        return int(config.max_cell_size)
    return int(base_cell_size(res, int(config.max_cell_size)))


def _patched_top_anchors(cls, visible_error_channel, top_k, block_size, channel):
    h, w = visible_error_channel.shape
    block_size = max(1, int(block_size))
    if block_size == 1:
        flat = visible_error_channel.reshape(-1)
        k = min(int(top_k), flat.size)
        idx = np.argpartition(flat, -k)[-k:]
        idx = idx[np.argsort(flat[idx])[::-1]]
        return [(channel, int(i) // w, int(i) % w) for i in idx]
    scores, ys, xs = anchor_block_scores(np.ascontiguousarray(visible_error_channel, dtype=np.float64), block_size)
    if scores.size == 0:
        return []
    k = min(int(top_k), scores.size)
    idx = np.argpartition(scores, -k)[-k:]
    order = idx[np.argsort(scores[idx])[::-1]]
    return [(channel, int(ys[i]), int(xs[i])) for i in order]


_ORIGINALS = {}


def enable():
    from PBC3 import PBC3
    if not NUMBA_AVAILABLE:
        raise RuntimeError("numba is not importable; run: pdm add numba")
    if not _ORIGINALS:
        _ORIGINALS["_box_cell_bound"] = PBC3._box_cell_bound
        _ORIGINALS["_base_cell_size"] = PBC3._base_cell_size
        _ORIGINALS["_top_anchors"] = PBC3._top_anchors
    PBC3._box_cell_bound = classmethod(_patched_box_cell_bound)
    PBC3._base_cell_size = classmethod(_patched_base_cell_size)
    PBC3._top_anchors = classmethod(_patched_top_anchors)


def disable():
    from PBC3 import PBC3
    for name, fn in _ORIGINALS.items():
        setattr(PBC3, name, fn)


# Reference numpy formulas (copied from PBC3) for tolerance-checking the kernels,
# independent of whether enable() has patched the live class.
def _ref_box_cell_bound(integral, x, y, bw, bh, cell_size):
    cd = lambda a, b: (a + b - 1) // b
    def edges(start, length):
        n = cd(length, cell_size)
        e = start + np.arange(n + 1) * cell_size
        e[n] = start + length
        return e
    xe, ye = edges(x, bw), edges(y, bh)
    corners = integral[np.ix_(ye, xe)].astype(np.float64)
    cell_sum = corners[1:, 1:] - corners[:-1, 1:] - corners[1:, :-1] + corners[:-1, :-1]
    counts = (np.diff(ye)[:, None] * np.diff(xe)[None, :]).astype(np.float64)
    return float(np.sum(cell_sum * cell_sum / counts))


def _ref_base_cell_size(res, max_cell):
    mean_abs = float(np.mean(np.abs(res)))
    if mean_abs <= 0:
        return int(max_cell)
    gx = float(np.mean(np.abs(np.diff(res, axis=1)))) if res.shape[1] > 1 else 0.0
    gy = float(np.mean(np.abs(np.diff(res, axis=0)))) if res.shape[0] > 1 else 0.0
    ratio = (gx + gy) / (mean_abs + 1.0)
    return 32 if ratio < 0.25 else 16 if ratio < 0.5 else 8 if ratio < 1.0 else 4


def _ref_block_scores(err, block_size):
    h, w = err.shape
    out = []
    for y0 in range(0, h, block_size):
        y1 = min(h, y0 + block_size)
        for x0 in range(0, w, block_size):
            x1 = min(w, x0 + block_size)
            out.append(float(err[y0:y1, x0:x1].astype(np.float64).sum() / ((y1 - y0) * (x1 - x0))))
    return np.array(out)


def verify(trials=400, seed=0):
    rng = np.random.default_rng(seed)
    worst_box = 0.0
    bcs_mismatch = 0
    worst_anchor = 0.0
    for _ in range(trials):
        H, W = int(rng.integers(8, 200)), int(rng.integers(8, 200))
        e = rng.integers(-80, 80, size=(H, W)).astype(np.int64)
        integral = np.pad(e.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
        bw, bh = int(rng.integers(4, W + 1)), int(rng.integers(4, H + 1))
        x, y = int(rng.integers(0, W - bw + 1)), int(rng.integers(0, H - bh + 1))
        cell = int(rng.integers(1, max(2, min(bw, bh))))
        ref = _ref_box_cell_bound(integral, x, y, bw, bh, cell)
        got = box_cell_bound(integral, x, y, bw, bh, cell)
        worst_box = max(worst_box, abs(got - ref) / (abs(ref) + 1e-9))

        res = rng.integers(-60, 60, size=(bh, bw)).astype(np.float64)
        if _ref_base_cell_size(res, 64) != int(base_cell_size(res, 64)):
            bcs_mismatch += 1

        bs = int(rng.integers(2, 16))
        err = np.abs(e).astype(np.float64)
        ref_s = _ref_block_scores(err, bs)
        got_s, _, _ = anchor_block_scores(err, bs)
        worst_anchor = max(worst_anchor, float(np.max(np.abs(ref_s - got_s) / (np.abs(ref_s) + 1e-9))))
    print(f"numba_available={NUMBA_AVAILABLE} trials={trials}")
    print(f"box_cell_bound worst_relative_diff={worst_box:.3e}")
    print(f"base_cell_size mismatches={bcs_mismatch}/{trials}")
    print(f"anchor_block_scores worst_relative_diff={worst_anchor:.3e}")
    return worst_box, bcs_mismatch, worst_anchor


if __name__ == "__main__":
    verify()
