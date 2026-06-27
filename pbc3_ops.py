
import math
import numpy as np
from numba import njit
from PIL import Image


RESAMPLE_FILTER = Image.Resampling.BICUBIC
RESAMPLE_REDUCING_GAP = None
PALETTE_GENERATED = 0


# ===== CORE MATH =====

def ceil_div(a: int, b: int) -> int:
    """## Returns ceil(a / b) for integer cell/grid math"""
    return (int(a) + int(b) - 1) // int(b)


def norm(values) -> np.ndarray:
    """## Returns values normalized to 0..1, or all 1s when the range is flat"""
    arr = np.asarray(values, dtype=np.float64)
    rng = arr.max() - arr.min()
    if rng <= 0:
        return np.ones_like(arr)
    return (arr - arr.min()) / rng


def interp(start: float, end: float, step: int, count: int) -> float:
    """## Returns a linear interpolation between start and end for the current patch step"""
    if count <= 1:
        return float(end)
    p = (step - 1) / max(1, count - 1)
    return float(start) * (1 - p) + float(end) * p


def integral(a) -> np.ndarray:
    """## Returns a padded integral image for fast rectangle sums"""
    return np.pad(a.astype(np.int64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))


def cell_edges(start: int, length: int, cell_size: int) -> np.ndarray:
    """## Returns the edge coordinates of patch cells, clamped to the patch end"""
    n = ceil_div(length, cell_size)
    edges = start + np.arange(n + 1) * cell_size
    edges[n] = start + length
    return edges


# ===== NUMBA KERNELS =====

@njit(cache=True)
def _box_cell_bound_kernel(integral_arr, x, y, bw, bh, cell_size):
    nx = (bw + cell_size - 1) // cell_size
    ny = (bh + cell_size - 1) // cell_size
    total = 0.0
    for iy in range(ny):
        y0 = y + iy * cell_size
        y1 = y + bh if iy == ny - 1 else y + (iy + 1) * cell_size
        for ix in range(nx):
            x0 = x + ix * cell_size
            x1 = x + bw if ix == nx - 1 else x + (ix + 1) * cell_size
            s = float(integral_arr[y1, x1] - integral_arr[y0, x1] - integral_arr[y1, x0] + integral_arr[y0, x0])
            total += s * s / ((y1 - y0) * (x1 - x0))
    return total


@njit(cache=True)
def _base_cell_size_kernel(res, max_cell):
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
def _anchor_block_scores_kernel(err, block_size):
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


@njit(cache=True)
def _mse_kernel(a, b):
    total = 0.0
    n = a.size
    flat_a = a.ravel()
    flat_b = b.ravel()
    for i in range(n):
        d = float(flat_a[i]) - float(flat_b[i])
        total += d * d
    return total / n



@njit(cache=True)
def _palette_bounds_kernel(values):
    flat = values.ravel()
    mn = int(flat[0])
    mx = int(flat[0])
    for i in range(1, flat.size):
        v = int(flat[i])
        if v < mn:
            mn = v
        if v > mx:
            mx = v
    neg = -mn if mn < 0 else 0
    pos = mx if mx > 0 else 0
    if neg > 255:
        neg = 255
    if pos > 255:
        pos = 255
    return neg, pos


@njit(cache=True)
def _mask_from_values_kernel(flat, mask_size, negative_max, positive_max, positive_bias):
    mask = np.zeros(mask_size, dtype=np.uint8)
    mask[0] = 1
    side_bits = mask_size - 1
    negative_max = max(0, int(negative_max))
    positive_max = max(0, int(positive_max))
    if side_bits == 0 or (negative_max == 0 and positive_max == 0):
        return mask

    if negative_max == 0:
        pos_count = min(side_bits, positive_max)
        neg_count = 0
    elif positive_max == 0:
        pos_count = 0
        neg_count = min(side_bits, negative_max)
    else:
        raw_pos = side_bits * positive_max / (positive_max + negative_max)
        pos_count = int(math.ceil(raw_pos)) if positive_bias else int(math.floor(raw_pos))
        pos_count = min(side_bits - 1, max(1, pos_count), positive_max)
        neg_count = min(side_bits - pos_count, negative_max)
        if neg_count == 0 and negative_max > 0 and side_bits > pos_count:
            neg_count = 1
            pos_count = max(1, pos_count - 1)

    if pos_count > 0 and positive_max > 0:
        for i in range(flat.size):
            v = int(flat[i])
            if v > 0:
                if v > positive_max:
                    v = positive_max
                b = 1 + min(((v - 1) * pos_count) // positive_max, pos_count - 1)
                if b < mask_size:
                    mask[b] = 1

    if neg_count > 0 and negative_max > 0:
        for i in range(flat.size):
            v = int(flat[i])
            if v < 0:
                mag = -v
                if mag > negative_max:
                    mag = negative_max
                b = 1 + pos_count + min(((mag - 1) * neg_count) // negative_max, neg_count - 1)
                if b < mask_size:
                    mask[b] = 1
    return mask


@njit(cache=True)
def _quantize_signed_kernel(vals, pal):
    out = np.empty(vals.size, dtype=np.uint16)
    flat = vals.ravel()
    for i in range(flat.size):
        v = int(flat[i])
        best_i = 0
        best_d = abs(v - int(pal[0]))
        for j in range(1, pal.size):
            d = abs(v - int(pal[j]))
            if d < best_d:
                best_d = d
                best_i = j
        out[i] = best_i
    return out

# ===== PALETTE OPS =====


def palette_bounds(values) -> tuple[int, int]:
    """## Returns negative max (min or 0) and positive max (true max)
    The reason there is a "negative max" instead of just a "min" is that palette generation is symmetric, and the negative max is used to determine the number of negative bins in the mask."""
    return _palette_bounds_kernel(np.ascontiguousarray(values, dtype=np.int16))



def range_counts(mask_size, negative_max=255, positive_max=255, positive_bias=True) -> tuple[int, int]:
    """## Returns the number of positive and negative bins in the mask, given the mask size and the maximum positive and negative values.
    - The number of bins is determined by the relative sizes of the positive and negative ranges, and positive_bias is a tiebreaker
    - If for example mask size is 4, negative max is 10, positive max is 20, then the positive range is supposed to be twice as large as the negative range, so counting the 1 zero-bin, there should be 2 positive bins and 1 negative bin, representing ranges (-10,-1), (0), (1, 10), (11, 20)
    - If in that example mask size was 3 though, to make sure there is still a negative and a positive range, the positive range would have to be 1 bin"""
    side_bits = max(0, int(mask_size) - 1)
    negative_max = max(0, int(negative_max))
    positive_max = max(0, int(positive_max))
    if side_bits == 0 or (negative_max == 0 and positive_max == 0):
        # The reason it's 0,0 if mask size is 1 is that the only range that can be represented by it is the 0 bin
        # This actually just makes mask size 1 useless for palette generation.
        return 0, 0
    if negative_max == 0:
        # Obviously if there are no negative values, all bins should be positive
        return min(side_bits, positive_max), 0
    if positive_max == 0:
        # Obviously same
        return 0, min(side_bits, negative_max)
    raw_pos = side_bits * positive_max / (positive_max + negative_max)
    pos_count = math.ceil(raw_pos) if positive_bias else math.floor(raw_pos)
    pos_count = min(side_bits - 1, max(1, pos_count), positive_max)
    neg_count = min(side_bits - pos_count, negative_max)
    if neg_count == 0 and negative_max > 0 and side_bits > pos_count:
        neg_count = 1
        pos_count = max(1, pos_count - 1)
    return pos_count, neg_count


def range_for_mask_index(index, mask_size, negative_max=255, positive_max=255, positive_bias=True) -> tuple[int, int] | None:
    """## Returns the range of values for a given index in a mask. 
    ### 0 is always the zero bin
    PS: It might seem like negative max and positive max are useless here but it's needed for a bunch of functions so
    it's best to just compute them once and pass them around instead of recomputing them every time inside the functions"""
    pos_count, neg_count = range_counts(mask_size, negative_max, positive_max, positive_bias)
    if index == 0:
        return 0, 0
    if 1 <= index <= pos_count:
        bin_i = index - 1
        start = 1 + (bin_i * positive_max) // pos_count
        end = ((bin_i + 1) * positive_max) // pos_count
        return (start, end) if start <= end else None
    bin_i = index - 1 - pos_count
    if 0 <= bin_i < neg_count:
        low_mag = 1 + (bin_i * negative_max) // neg_count
        high_mag = ((bin_i + 1) * negative_max) // neg_count
        return (-high_mag, -low_mag) if high_mag >= low_mag else None
    return None


def mask_from_values(values, mask_size, negative_max=255, positive_max=255, positive_bias=True) -> list[int]:
    """## Returns a mask of the given size, with 1s for the bins that have values in them, and 0s for the bins that don't"""
    flat = np.clip(np.rint(np.asarray(values)).astype(np.int32).ravel(), -negative_max, positive_max)
    mask = _mask_from_values_kernel(
        np.ascontiguousarray(flat, dtype=np.int32),
        int(mask_size),
        int(negative_max),
        int(positive_max),
        bool(positive_bias),
    )
    return [int(x) for x in mask]



def active_value_count(mask, negative_max=255, positive_max=255, positive_bias=True) -> int:
    """## Returns the number of values within the mask selected ranges"""
    count = 0
    for i, bit in enumerate(mask):
        if bit:
            r = range_for_mask_index(i, len(mask), negative_max, positive_max, positive_bias)
            if r is not None:
                start, end = r
                count += end - start + 1
    return max(1, count)


def resolve_palette_bitcount(mask, max_bitcount, negative_max=255, positive_max=255, positive_bias=True) -> int:
    """## Returns the number of bits needed to represent the active values in the mask
    This is so for example if the max bitcount is 8 but the mask only has 60 active values then the bitcount
    for this palette can be reduced to 6 instead with no loss"""
    value_count = active_value_count(mask, negative_max, positive_max, positive_bias)
    needed = max(1, math.ceil(math.log2(value_count)))
    return min(int(max_bitcount), needed)


def palette_generator(mask, max_bitcount, negative_max=255, positive_max=255, positive_bias=True) -> np.ndarray:
    """## Returns a 1D array palette which is all the selected values for the patch"""
    bitcount = resolve_palette_bitcount(mask, max_bitcount, negative_max, positive_max, positive_bias)
    size = 1 << bitcount
    active_ranges = []
    for i, bit in enumerate(mask):
        if bit:
            r = range_for_mask_index(i, len(mask), negative_max, positive_max, positive_bias)
            if r is not None:
                active_ranges.append(r)
    palette = []
    if mask and mask[0]:
        palette.append(0)
        active_ranges = [r for r in active_ranges if r != (0, 0)]
    value_count = active_value_count(mask, negative_max, positive_max, positive_bias)
    if size >= value_count:
        for start, end in active_ranges:
            palette.extend(range(start, end + 1))
        if len(palette) < size:
            palette.extend([palette[-1] if palette else 0] * (size - len(palette)))
        return np.array(palette[:size], dtype=np.int16)
    if not active_ranges:
        return np.zeros(size, dtype=np.int16)
    remaining = size - len(palette)
    counts = [0] * len(active_ranges)
    for i in range(remaining):
        counts[i % len(active_ranges)] += 1
    for (start, end), count in zip(active_ranges, counts):
        if count == 1:
            palette.append(int(round((start + end) / 2)))
        elif count > 1:
            for j in range(count):
                t = (j + 1) / (count + 1)
                palette.append(int(round(start + (end - start) * t)))
    if len(palette) < size:
        palette.extend([palette[-1] if palette else 0] * (size - len(palette)))
    return np.array(palette[:size], dtype=np.int16)


def quantize_signed(values, palette):
    """## Returns the indices of the closest palette values for each value in the input array
    `quantize_signed([8,13,20], [5,30]) == [0,0,1]`"""
    vals = np.asarray(values, dtype=np.int16)
    pal = np.asarray(palette, dtype=np.int16)
    out = _quantize_signed_kernel(np.ascontiguousarray(vals), np.ascontiguousarray(pal))
    return out.reshape(vals.shape)



# ===== IMAGE / GRID OPS =====

def signed_resample(values, out_h: int, out_w: int) -> np.ndarray:
    """## Resizes a signed grid to a patch-sized int16 delta image"""
    values = np.asarray(values, dtype=np.float32)
    out_h, out_w = int(out_h), int(out_w)
    if values.shape == (out_h, out_w):
        return np.rint(values).astype(np.int16)
    resized = Image.fromarray(values).resize((out_w, out_h), RESAMPLE_FILTER, reducing_gap=RESAMPLE_REDUCING_GAP)
    out = np.asarray(resized, dtype=np.float32)
    return np.rint(out).astype(np.int16)


def signed_resample_cells(values, cell_size: int) -> np.ndarray:
    """## Resamples a patch residual to its cell grid size"""
    h, w = values.shape
    return signed_resample(values, ceil_div(h, cell_size), ceil_div(w, cell_size))


def apply_grid(canvas_layer, x: int, y: int, w: int, h: int, cell_size: int, values) -> None:
    """## Applies a signed patch grid into one canvas channel"""
    canvas_layer[y:y + h, x:x + w] += signed_resample(values, h, w).astype(np.int32)


def image_mse(reference, reconstructed) -> float:
    """## Returns mean squared error between two same-shaped image arrays"""
    a = np.ascontiguousarray(reference, dtype=np.float32)
    b = np.ascontiguousarray(reconstructed, dtype=np.float32)
    if a.shape != b.shape:
        raise ValueError(f"MSE arrays must have the same shape, got {a.shape} and {b.shape}")
    return float(_mse_kernel(a, b))


def final_mse(reference_image, reconstructed_image) -> float:
    """## Returns final image MSE from two PIL images or image-like arrays"""
    return image_mse(np.asarray(reference_image, dtype=np.float32), np.asarray(reconstructed_image, dtype=np.float32))


# ===== SEARCH OPS =====

def box_cell_bound(integral_arr, x: int, y: int, bw: int, bh: int, cell_size: int) -> float:
    """## Returns the best-case signed-error energy for a box/cell-size pair"""
    return float(_box_cell_bound_kernel(
        np.ascontiguousarray(integral_arr, dtype=np.int64),
        int(x),
        int(y),
        int(bw),
        int(bh),
        int(cell_size),
    ))


def top_anchors(visible_error_channel, top_k: int, block_size: int, channel: int) -> list[tuple[int, int, int]]:
    """## Returns the strongest error anchors as (channel, y, x) tuples"""
    h, w = visible_error_channel.shape
    block_size = max(1, int(block_size))
    if block_size == 1:
        flat = visible_error_channel.reshape(-1)
        k = min(int(top_k), flat.size)
        idx = np.argpartition(flat, -k)[-k:]
        idx = idx[np.argsort(flat[idx])[::-1]]
        return [(channel, int(i) // w, int(i) % w) for i in idx]

    scores, ys, xs = _anchor_block_scores_kernel(np.ascontiguousarray(visible_error_channel, dtype=np.float64), block_size)
    if scores.size == 0:
        return []
    k = min(int(top_k), scores.size)
    idx = np.argpartition(scores, -k)[-k:]
    order = idx[np.argsort(scores[idx])[::-1]]
    return [(channel, int(ys[i]), int(xs[i])) for i in order]


def select_top_indices(scores, keep: int) -> np.ndarray:
    """## Returns unordered indices of the top `keep` scores"""
    scores = np.asarray(scores)
    n = scores.size
    keep = max(1, int(keep))
    if keep >= n:
        return np.arange(n)
    cut = n - keep
    if cut < keep:
        drop = np.argpartition(scores, cut)[:cut]
        mask = np.ones(n, dtype=bool)
        mask[drop] = False
        return np.nonzero(mask)[0]
    return np.argpartition(scores, -keep)[-keep:]


def sample_box(rng, anchor, image_w: int, image_h: int, config) -> tuple[int, int, int, int, int, int, int]:
    """## Samples a candidate patch box around an anchor point"""
    c, ay, ax = anchor
    min_size = max(1, int(config.min_patch_size))
    max_w = max(min_size, min(int(config.max_patch_size), image_w))
    max_h = max(min_size, min(int(config.max_patch_size), image_h))
    w = int(round(2 ** rng.uniform(math.log2(min_size), math.log2(max_w))))
    h = int(round(2 ** rng.uniform(math.log2(min_size), math.log2(max_h))))
    w = min(max(1, w), image_w)
    h = min(max(1, h), image_h)
    x_min = max(0, ax - w + 1)
    x_max = min(ax, image_w - w)
    y_min = max(0, ay - h + 1)
    y_max = min(ay, image_h - h)
    x = int(rng.integers(x_min, x_max + 1)) if x_min <= x_max else max(0, min(ax, image_w - w))
    y = int(rng.integers(y_min, y_max + 1)) if y_min <= y_max else max(0, min(ay, image_h - h))
    return c, x, y, w, h, ax, ay


def base_cell_size(residual_patch, config) -> int:
    """## Returns the rough cell size suggested by local residual frequency"""
    if residual_patch.size == 0:
        return int(config.max_cell_size)
    return int(_base_cell_size_kernel(np.ascontiguousarray(residual_patch, dtype=np.float64), int(config.max_cell_size)))


# ===== PATCH OPS =====

def patch_bits_for(patch, channel_bits: int) -> int:
    """## Returns the serialized bit cost of a complete patch"""
    grid_bits = ceil_div(patch["w"], patch["cell_size"]) * ceil_div(patch["h"], patch["cell_size"]) * patch["bitcount"]
    return channel_bits + 64 + 16 + 1 + 10 + len(patch["mask"]) + 8 + 8 + 4 + grid_bits


def patch_header_bits(channel_bits: int, mask_size: int) -> int:
    """## Returns the fixed patch header bit cost before grid indices"""
    return channel_bits + 64 + 10 + mask_size + 8 + 8 + 4 + 16 + 1


def make_patch(channel: int, x: int, y: int, w: int, h: int, cell_size: int, residual, config, max_bitcount: int):
    """## Builds a generated-palette patch and the signed values it will apply"""
    small = signed_resample_cells(residual, cell_size)
    negative_max, positive_max = palette_bounds(small)
    mask = mask_from_values(small, config.mask_size, negative_max, positive_max, config.positive_bias)
    pal = palette_generator(mask, max_bitcount, negative_max, positive_max, config.positive_bias)
    indices = quantize_signed(np.clip(small, -negative_max, positive_max), pal)
    values = pal[indices]
    bitcount = resolve_palette_bitcount(mask, max_bitcount, negative_max, positive_max, config.positive_bias)
    return {
        "channel": channel,
        "x": x,
        "y": y,
        "w": w,
        "h": h,
        "cell_size": cell_size,
        "indices": indices,
        "palette_mode": PALETTE_GENERATED,
        "palette": None,
        "bitcount": bitcount,
        "mask": mask,
        "neg": negative_max,
        "pos": positive_max,
        "max_bitcount": max_bitcount,
    }, values



def debug_line(kind: str, **items) -> str:
    """## Returns one plain-text debug line with key=value fields"""
    return kind + " " + " ".join(f"{k}={v}" for k, v in items.items())