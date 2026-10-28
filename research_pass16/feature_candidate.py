import numpy as np
from numba import njit


@njit(cache=True)
def integer_order_stats(flat, lo, hi):
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
