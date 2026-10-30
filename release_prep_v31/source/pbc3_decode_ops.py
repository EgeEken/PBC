import math
import numpy as np
from PIL import Image

def bicubic_resample_2d(values, out_h: int, out_w: int) -> np.ndarray:
    """Resamples a signed grid with the production PIL bicubic filter."""
    arr = np.ascontiguousarray(values, dtype=np.float32)
    out_h, out_w = int(out_h), int(out_w)
    if arr.shape == (out_h, out_w):
        return arr.copy()
    return np.asarray(
        Image.frombuffer("F", (arr.shape[1], arr.shape[0]), arr, "raw", "F", 0, 1)
        .resize((out_w, out_h), Image.Resampling.BICUBIC),
        dtype=np.float32,
    )


def ceil_div(a: int, b: int) -> int:
    """## Returns ceil(a / b) for integer cell/grid math"""
    return (int(a) + int(b) - 1) // int(b)


def range_counts(mask_size, negative_max=255, positive_max=255, positive_bias=True) -> tuple[int, int]:
    """## Returns the number of positive and negative bins in the mask, given the mask size and the maximum positive and negative values.
    - The number of bins is determined by the relative sizes of the positive and negative ranges, and positive_bias is a tiebreaker
    - If for example mask size is 4, negative max is 10, positive max is 20, then the positive range is supposed to be twice as large as the negative range, so counting the 1 zero-bin, there should be 2 positive bins and 1 negative bin, representing ranges (-10,-1), (0), (1, 10), (11, 20)
    - If in that example mask size was 3 though, to make sure there is still a negative and a positive range, the positive range would have to be 1 bin"""
    side_bits = max(0, int(mask_size) - 1)
    negative_max = max(0, int(negative_max))
    positive_max = max(0, int(positive_max))
    if side_bits == 0 or (negative_max == 0 and positive_max == 0):
        return 0, 0
    if negative_max == 0:
        return min(side_bits, positive_max), 0
    if positive_max == 0:
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


def signed_resample(values, out_h: int, out_w: int, trace=None, purpose: str = None) -> np.ndarray:
    """## Resizes a signed grid to a patch-sized int16 delta image"""
    if trace is not None and purpose:
        trace.count(f"signed_resample.{purpose}")
    return np.rint(bicubic_resample_2d(values, int(out_h), int(out_w))).astype(np.int16)

