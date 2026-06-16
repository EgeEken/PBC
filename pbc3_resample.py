# Numba reimplementation of Pillow's float-mode bicubic resampler.
# Faithful to Pillow's Resample.c: a=-0.5 cubic, support scaled by the reduction
# factor (antialiasing on downscale), truncated+renormalized kernel at borders,
# horizontal pass then vertical pass, float32 intermediate, float64 accumulation.
# Enable in the codec with: PBC3.USE_NUMBA_RESAMPLE = True
#
# Pillow is already optimized C/SIMD; this is unlikely to beat it standalone. Its
# purpose is to be a verified, hackable building block for fusing the whole patch
# evaluation into one compiled kernel (the real future speedup) and for custom
# resampling. Validate it matches PIL with compare_with_pil() before relying on it.

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
def _bicubic(x):
    a = -0.5
    if x < 0.0:
        x = -x
    if x < 1.0:
        return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0
    if x < 2.0:
        return (((x - 5.0) * x + 8.0) * x - 4.0) * a
    return 0.0


@njit(cache=True)
def _coeffs(in_size, out_size):
    support = 2.0
    scale = in_size / out_size
    filterscale = scale if scale >= 1.0 else 1.0
    fsupport = support * filterscale
    ksize = int(np.ceil(fsupport)) * 2 + 1
    bounds = np.empty((out_size, 2), dtype=np.int64)
    kk = np.zeros((out_size, ksize), dtype=np.float64)
    inv = 1.0 / filterscale
    for xx in range(out_size):
        center = (xx + 0.5) * scale
        xmin = int(center - fsupport + 0.5)
        if xmin < 0:
            xmin = 0
        xmax = int(center + fsupport + 0.5)
        if xmax > in_size:
            xmax = in_size
        xmax -= xmin
        ww = 0.0
        for x in range(xmax):
            w = _bicubic((x + xmin - center + 0.5) * inv)
            kk[xx, x] = w
            ww += w
        if ww != 0.0:
            for x in range(xmax):
                kk[xx, x] /= ww
        bounds[xx, 0] = xmin
        bounds[xx, 1] = xmax
    return bounds, kk


@njit(cache=True)
def _resample(src, out_h, out_w):
    in_h, in_w = src.shape
    bx, kx = _coeffs(in_w, out_w)
    tmp = np.empty((in_h, out_w), dtype=np.float32)
    for y in range(in_h):
        for xx in range(out_w):
            xmin = bx[xx, 0]
            acc = 0.0
            for k in range(bx[xx, 1]):
                acc += src[y, xmin + k] * kx[xx, k]
            tmp[y, xx] = np.float32(acc)
    by, ky = _coeffs(in_h, out_h)
    out = np.empty((out_h, out_w), dtype=np.float32)
    for yy in range(out_h):
        ymin = by[yy, 0]
        for xx in range(out_w):
            acc = 0.0
            for k in range(by[yy, 1]):
                acc += tmp[ymin + k, xx] * ky[yy, k]
            out[yy, xx] = np.float32(acc)
    return out


def resample_bicubic(values, out_h, out_w):
    src = np.ascontiguousarray(values, dtype=np.float32)
    return _resample(src, int(out_h), int(out_w))


def compare_with_pil(trials=300, seed=0):
    from PIL import Image
    rng = np.random.default_rng(seed)
    worst = 0.0
    mismatches = 0
    total = 0
    for _ in range(trials):
        ih, iw = int(rng.integers(2, 220)), int(rng.integers(2, 220))
        oh, ow = int(rng.integers(1, 220)), int(rng.integers(1, 220))
        src = (rng.standard_normal((ih, iw)) * 70.0).astype(np.float32)
        pil = np.asarray(
            Image.fromarray(src).resize((ow, oh), Image.Resampling.BICUBIC, reducing_gap=None),
            dtype=np.float32,
        )
        ours = resample_bicubic(src, oh, ow)
        worst = max(worst, float(np.max(np.abs(pil - ours))))
        mismatches += int(np.sum(np.rint(pil).astype(np.int16) != np.rint(ours).astype(np.int16)))
        total += pil.size
    print(f"numba_available={NUMBA_AVAILABLE} trials={trials}")
    print(f"max_abs_float_diff={worst:.6g}")
    print(f"int16_rint_mismatches={mismatches}/{total} ({100.0 * mismatches / total:.5f}%)")
    return worst, mismatches, total


if __name__ == "__main__":
    compare_with_pil()
