import numpy as np
from numba import njit


@njit(cache=True)
def _integral_kernel(a):
    height, width = a.shape
    out = np.zeros((height + 1, width + 1), dtype=np.int64)
    for y in range(height):
        row_sum = np.int64(0)
        for x in range(width):
            row_sum += np.int64(a[y, x])
            out[y + 1, x + 1] = row_sum + out[y, x + 1]
    return out


def integral(a):
    source = np.asarray(a)
    if source.ndim != 2:
        raise ValueError("integral input must be 2D")
    return _integral_kernel(source)
