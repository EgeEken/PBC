
import math
import numpy as np
from numba import njit, uint32, uint64
from PIL import Image
from pbc3_trace import timed


RESAMPLE_FILTER = Image.Resampling.BICUBIC
RESAMPLE_REDUCING_GAP = None
PALETTE_GENERATED = 0


class SearchState:
    def __init__(self):
        self.target = self.canvas = None
        self.errors = {}
        self.totals = {}

    def error(self, target, canvas, channel):
        if target is not self.target or canvas is not self.canvas:
            self.target, self.canvas = target, canvas
            self.errors.clear()
            self.totals.clear()
        if channel not in self.errors:
            error = np.abs(target[:, :, channel] - np.clip(canvas[:, :, channel], 0, 255))
            self.errors[channel] = error
            self.totals[channel] = int(np.sum(error))
        return self.errors[channel]

    def score(self, target, canvas, channel):
        self.error(target, canvas, channel)
        return float(self.totals[channel])

    def update(self, target, canvas, channel, x, y, w, h):
        error = self.error(target, canvas, channel)
        region = error[y:y + h, x:x + w]
        previous = int(np.sum(region))
        region[:] = np.abs(target[y:y + h, x:x + w, channel] - np.clip(canvas[y:y + h, x:x + w, channel], 0, 255))
        self.totals[channel] += int(np.sum(region)) - previous


_UINT64_MASK = (1 << 64) - 1
_UINT32_SCALE = 1.0 / 4294967296.0


@njit(inline="always", cache=True)
def pcg_step(state):
    old_state = state
    state = uint64(old_state * uint64(6364136223846793005) + uint64(1442695040888963407))
    xorshifted = uint32(((old_state >> uint64(18)) ^ old_state) >> uint64(27))
    rot = uint32(old_state >> uint64(59))
    out = uint32((xorshifted >> rot) | (xorshifted << ((-rot) & uint32(31))))
    return state, out


class PBC3Rng:
    def __init__(self, seed: int):
        self.state = (int(seed) + 0x853C49E6748FEA9B) & _UINT64_MASK
        self.next_uint32()

    def next_uint32(self) -> int:
        old_state = self.state
        self.state = (old_state * 6364136223846793005 + 1442695040888963407) & _UINT64_MASK
        xorshifted = (((old_state >> 18) ^ old_state) >> 27) & 0xFFFFFFFF
        rot = (old_state >> 59) & 31
        return ((xorshifted >> rot) | ((xorshifted << ((-rot) & 31)) & 0xFFFFFFFF)) & 0xFFFFFFFF

    def uniform(self, low: float = 0.0, high: float = 1.0) -> float:
        return float(low) + (float(high) - float(low)) * (self.next_uint32() * _UINT32_SCALE)

    def integers(self, low: int, high: int = None) -> int:
        if high is None:
            low, high = 0, low
        low, high = int(low), int(high)
        if high <= low:
            return low
        return low + self.next_uint32() % (high - low)






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


def integral(a) -> np.ndarray:
    """## Returns a padded integral image for fast rectangle sums"""
    source = np.asarray(a)
    if source.ndim != 2:
        raise ValueError("integral input must be 2D")
    return _integral_kernel(source)


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
def _cell_mean_kernel(values, cell_size):
    h, w = values.shape
    ny = (h + cell_size - 1) // cell_size
    nx = (w + cell_size - 1) // cell_size
    out = np.empty((ny, nx), dtype=np.float32)
    for iy in range(ny):
        y0 = iy * cell_size
        y1 = min(h, y0 + cell_size)
        for ix in range(nx):
            x0 = ix * cell_size
            x1 = min(w, x0 + cell_size)
            total = 0.0
            for y in range(y0, y1):
                for x in range(x0, x1):
                    total += values[y, x]
            out[iy, ix] = total / ((y1 - y0) * (x1 - x0))
    return out


@njit(cache=True)
def _cell_representability_kernel(values, cell_size):
    h, w = values.shape
    total_energy = 0.0
    for y in range(h):
        for x in range(w):
            total_energy += values[y, x] * values[y, x]
    if total_energy <= 0.0:
        return 1.0

    ny = (h + cell_size - 1) // cell_size
    nx = (w + cell_size - 1) // cell_size
    captured = 0.0
    for iy in range(ny):
        y0 = iy * cell_size
        y1 = min(h, y0 + cell_size)
        for ix in range(nx):
            x0 = ix * cell_size
            x1 = min(w, x0 + cell_size)
            total = 0.0
            for y in range(y0, y1):
                for x in range(x0, x1):
                    total += values[y, x]
            captured += total * total / ((y1 - y0) * (x1 - x0))
    return min(1.0, max(0.0, captured / total_energy))


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


def palette_bounds(values) -> tuple[int, int]:
    """## Returns negative max (min or 0) and positive max (true max)
    The reason there is a "negative max" instead of just a "min" is that palette generation is symmetric, and the negative max is used to determine the number of negative bins in the mask."""
    return _palette_bounds_kernel(np.ascontiguousarray(values, dtype=np.int16))







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









def quantize_signed(values, palette):
    """## Returns the indices of the closest palette values for each value in the input array
    `quantize_signed([8,13,20], [5,30]) == [0,0,1]`"""
    vals = np.asarray(values, dtype=np.int16)
    pal = np.asarray(palette, dtype=np.int16)
    out = _quantize_signed_kernel(np.ascontiguousarray(vals), np.ascontiguousarray(pal))
    return out.reshape(vals.shape)





def signed_resample_cells(values, cell_size: int) -> np.ndarray:
    """## Resamples a patch residual to its cell grid size"""
    h, w = values.shape
    return signed_resample(values, ceil_div(h, cell_size), ceil_div(w, cell_size))


def project_residual_to_grid(values, cell_size: int, mode: str = "bicubic") -> np.ndarray:
    """Projects a signed residual patch onto the stored coarse grid."""
    cell_size = int(cell_size)
    if cell_size < 1:
        raise ValueError("cell_size must be >= 1")
    mode = str(mode).strip().lower()
    if mode == "bicubic":
        return signed_resample_cells(values, cell_size)
    if mode == "cell_mean":
        return _cell_mean_kernel(np.ascontiguousarray(values, dtype=np.float32), cell_size)
    raise ValueError(f"unknown residual projection mode: {mode}")


def cell_representability(residual_patch, cell_size: int) -> float:
    """Returns the fraction of residual energy representable by scalar cells."""
    cell_size = int(cell_size)
    if cell_size < 1:
        raise ValueError("cell_size must be >= 1")
    return float(_cell_representability_kernel(
        np.ascontiguousarray(residual_patch, dtype=np.float64), cell_size
    ))


def apply_grid(canvas_layer, x: int, y: int, w: int, h: int, cell_size: int, values, trace=None) -> None:
    """## Applies a signed patch grid into one canvas channel"""
    delta = signed_resample(values, h, w, trace=trace, purpose="selected_patch_application")
    canvas_layer[y:y + h, x:x + w] += delta


def apply_delta(canvas_layer, x: int, y: int, w: int, h: int, delta) -> None:
    """## Applies an already reconstructed signed patch delta"""
    delta = np.asarray(delta, dtype=np.int16)
    if delta.shape != (int(h), int(w)):
        raise ValueError(f"delta shape must be {(int(h), int(w))}, got {delta.shape}")
    canvas_layer[y:y + h, x:x + w] += delta


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

    scores, ys, xs = _anchor_block_scores_kernel(np.ascontiguousarray(visible_error_channel), block_size)
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


def search_boxes(target, canvas, config, rng, channel: int, depth: int, search_q: float, ordered=False, trace=None, abs_error=None):
    """Samples error-centered boxes and keeps the best prescores."""
    h_img, w_img, _ = target.shape
    with timed(trace, "action.search.visible_error"):
        if abs_error is None:
            visible = np.clip(canvas[:, :, channel], 0, 255).astype(np.int32)
            abs_error = np.abs(target[:, :, channel] - visible)
    with timed(trace, "action.search.integral_error"):
        integral_abs = integral(abs_error)
    with timed(trace, "action.search.anchors"):
        anchors = top_anchors(abs_error, config.top_k, config.anchor_block_size, channel)
    if not anchors:
        return []

    specs, sums, areas = [], [], []
    with timed(trace, "action.search.sample_boxes", depth=int(depth)):
        for i in range(max(1, int(depth))):
            c, x, y, bw, bh, _, _ = sample_box(rng, anchors[i % len(anchors)], w_img, h_img, config)
            total = integral_abs[y + bh, x + bw] - integral_abs[y, x + bw] - integral_abs[y + bh, x] + integral_abs[y, x]
            if total <= 0:
                continue
            sums.append(float(total))
            areas.append(float(bw * bh))
            specs.append((c, x, y, bw, bh))
    if not specs:
        return []

    with timed(trace, "action.search.prescore"):
        scores = search_q * norm(sums) - (1.0 - search_q) * norm(areas)
    with timed(trace, "action.search.select_top_boxes"):
        keep = select_top_indices(scores, config.proposal_depth)
    if ordered:
        keep = sorted(keep, key=lambda i: scores[i], reverse=True)
    return [specs[i] for i in keep]


def base_cell_size(residual_patch, config) -> int:
    """## Returns the rough cell size suggested by local residual frequency"""
    if residual_patch.size == 0:
        return int(config.max_cell_size)
    return int(_base_cell_size_kernel(np.ascontiguousarray(residual_patch, dtype=np.float64), int(config.max_cell_size)))


def patch_bits_for(patch, channel_bits: int) -> int:
    """## Returns the serialized bit cost of a complete patch"""
    grid_bits = ceil_div(patch["w"], patch["cell_size"]) * ceil_div(patch["h"], patch["cell_size"]) * patch["bitcount"]
    return channel_bits + 64 + 16 + 1 + 10 + len(patch["mask"]) + 8 + 8 + 4 + grid_bits


def patch_header_bits(channel_bits: int, mask_size: int) -> int:
    """## Returns the fixed patch header bit cost before grid indices"""
    return channel_bits + 64 + 10 + mask_size + 8 + 8 + 4 + 16 + 1


def make_patch(channel: int, x: int, y: int, w: int, h: int, cell_size: int, residual, config, max_bitcount: int, trace=None):
    """## Builds a generated-palette patch and the signed values it will apply"""
    mode = str(getattr(config, "residual_projection_mode", "bicubic")).strip().lower()
    if trace is not None:
        trace.count("candidates_built")
        trace.count("grid_coefficients_processed", ceil_div(h, cell_size) * ceil_div(w, cell_size))
        trace.count(f"projection.{mode}")
        if mode == "bicubic":
            trace.count("signed_resample.projection")
    small = np.rint(project_residual_to_grid(residual, cell_size, mode)).astype(np.int16)
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

from pbc3_decode_ops import bicubic_resample_2d, ceil_div, range_counts, range_for_mask_index, active_value_count, resolve_palette_bitcount, palette_generator, signed_resample
