
# ====================================================================================================
#
#           PBC v2.4 - Probabilistic Brush Compression
#           Lossy Image Compression Algorithm by EgeEken (github.com/EgeEken)
#           2.4 Update - 2026-06 - Codebase Refactor and Optimization
#
# ====================================================================================================


# ======================================= IMPORTS ====================================================

from dataclasses import dataclass, replace
from typing import Optional, Union
from time import perf_counter

import numpy as np
from numba import njit, uint32, uint64
from PIL import Image
import matplotlib.pyplot as plt

# ====================================================================================================


# ================================== GLOBAL CONSTANTS / UTILS ========================================

MAGIC = b"PBC2"
VERSION = 4

RESAMPLE = {
    "nearest": Image.Resampling.NEAREST,
    "box": Image.Resampling.BOX,
    "bilinear": Image.Resampling.BILINEAR,
    "hamming": Image.Resampling.HAMMING,
    "bicubic": Image.Resampling.BICUBIC,
    "lanczos": Image.Resampling.LANCZOS,
}
def _resample_id(name: str) -> int:
    return list(RESAMPLE).index(name)
def _resample_name(idx: int) -> str:
    return list(RESAMPLE)[idx]


def rgb_to_ycbcr(img: np.ndarray) -> np.ndarray:
    xform = np.array([[0.299, 0.587, 0.114],
                      [-0.168736, -0.331264, 0.5],
                      [0.5, -0.418688, -0.081312]])
    ycbcr = img.astype(float).dot(xform.T)
    ycbcr[:, :, [1, 2]] += 128
    return np.clip(ycbcr, 0, 255).astype(np.uint8)


def ycbcr_to_rgb(img: np.ndarray) -> np.ndarray:
    xform = np.array([[1, 0, 1.402],
                      [1, -0.344136, -0.714136],
                      [1, 1.772, 0]])
    rgb = img.astype(float)
    rgb[:, :, [1, 2]] -= 128
    rgb = rgb.dot(xform.T)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _resize(img_pil: Image.Image, size_wh, resample) -> Image.Image:
    return img_pil.resize((int(size_wh[0]), int(size_wh[1])), resample)


def _downsample(img_pil: Image.Image, rate: float, resample) -> Image.Image:
    if rate <= 1:
        return img_pil
    return _resize(img_pil, (img_pil.width // rate, img_pil.height // rate), resample)


def _focus_bitcount(height: int, width: int, size: int, max_bitcount: int = 8) -> int:
    """Maximum number of focus-split bits usable for the given dimensions and brush size."""
    ch, cw = height, width
    bitcount = 0
    split_height = True
    while bitcount < max_bitcount:
        if split_height:
            if ch // 2 < size:
                break
            ch //= 2
        else:
            if cw // 2 < size:
                break
            cw //= 2
        split_height = not split_height
        bitcount += 1
    return bitcount


def _focus_region(height: int, width: int, code: int, bitcount: int):
    """Full bounding box of a focus code (no brush-size offset, no padding)."""
    r_s, r_e = 0, height
    c_s, c_e = 0, width
    split_height = True
    for b in range(bitcount - 1, -1, -1):
        bit = (code >> b) & 1
        if split_height:
            mid = (r_s + r_e) // 2
            r_e, r_s = (mid, r_s) if bit == 0 else (r_e, mid)
        else:
            mid = (c_s + c_e) // 2
            c_e, c_s = (mid, c_s) if bit == 0 else (c_e, mid)
        split_height = not split_height
    return r_s, c_s, r_e, c_e


def _select_focus(error_layer: np.ndarray, bitcount: int, criteria: str) -> int:
    """Selects the focus code maximizing error under the given criteria."""
    if bitcount == 0:
        return 0
    h, w = error_layer.shape
    best_err = -1.0
    best = 0
    for code in range(1 << bitcount):
        r_s, c_s, r_e, c_e = _focus_region(h, w, code, bitcount)
        region = error_layer[r_s:r_e, c_s:c_e]
        if region.size == 0:
            continue
        if criteria == "Max":
            e = float(region.max())
        elif criteria == "Min":
            e = float(region.min())
        else:
            e = float(region.sum())
        if e > best_err:
            best_err = e
            best = code
    return best


def _channel_cycle(full_error_layer: np.ndarray, strategy: str, criteria: str):
    """Returns the next 3-channel selection order based on per-channel error."""
    if strategy == "Default":
        return [0, 1, 2]
    errs = []
    for ch in range(3):
        layer = full_error_layer[:, :, ch]
        if criteria == "Max":
            errs.append(float(layer.max()))
        elif criteria == "Min":
            errs.append(float(layer.min()))
        elif criteria == "Median":
            errs.append(float(np.median(layer)))
        else:
            errs.append(float(layer.sum()))
    order = sorted(range(3), key=lambda x: errs[x], reverse=True)
    if strategy == "Strict":
        return [order[0]] * 3
    if strategy == "Balanced":
        return [order[0], order[0], order[1]]
    if strategy == "Smart":
        if errs[order[0]] > 2 * errs[order[1]]:
            return [order[0]] * 3
        if errs[order[1]] > 2 * errs[order[2]]:
            return [order[0], order[0], order[1]]
    return [0, 1, 2]


def _resolve_decay_value(val: float, length: int, kind: str):
    """Resolves an auto (-1) decay parameter; returns (was_auto, clamped_rounded_value)."""
    auto = (val == -1)
    if auto:
        val = 0.01 + (1 / 1.0000115) ** (length + 15000) if kind == "cutoff" else 0.5
    hi = 3.0 if kind == "cutoff" else 1.0
    return auto, round(float(np.clip(val, 0.0, hi)), 4)


def _decay_curve(length: int, start: int, end: int, cutoff: float, softness: float, progress: float) -> np.ndarray:
    """Brush size for each stroke index (even integers)."""
    x = np.arange(length, dtype=float)
    if cutoff <= 0:
        return np.full(length, end, dtype=int)
    lencut = length * cutoff
    if lencut >= length * 3:
        return np.full(length, start, dtype=int)
    lin = start + (x / (length * cutoff)) * (end - start)
    if softness <= 0:
        sig = np.where(x >= lencut / 2, end, start)
    else:
        k = 1.0 / softness * (np.sqrt(abs(end - start)) / length)
        sig = start + (end - start) / (1 + np.exp(-k * (x - lencut / 2)))
    curve = progress * sig + (1 - progress) * lin
    curve[x >= lencut] = end
    return (curve // 2 * 2).astype(int)

# ====================================================================================================


# ==================================== BIT WRITER/READERS ============================================

class BitWriter:
    """MSB-first bit writer backed by a growable byte buffer."""

    def __init__(self):
        self.buf = bytearray()
        self.acc = 0          # pending bits not yet flushed to a full byte
        self.n = 0            # number of pending bits in acc
        self.bits = 0         # total bits written

    def write(self, value: int, count: int) -> None:
        if count <= 0:
            return
        self.acc = (self.acc << count) | (int(value) & ((1 << count) - 1))
        self.n += count
        self.bits += count
        while self.n >= 8:
            self.n -= 8
            self.buf.append((self.acc >> self.n) & 0xFF)
        self.acc &= (1 << self.n) - 1

    def write_signed(self, value: int, count: int) -> None:
        self.write(0 if value < 0 else 1, 1)
        self.write(abs(int(value)), count - 1)

    def write_float(self, value: float, decimals: int = 4, count: int = 20) -> None:
        self.write(int(round(value * (10 ** decimals))), count)

    def write_array(self, arr: np.ndarray, bits: int) -> None:
        shift = 8 - bits
        for v in arr.reshape(-1):
            self.write(int(v) >> shift, bits)

    def finish(self):
        pad = (8 - self.n) % 8
        if self.n:
            self.buf.append((self.acc << pad) & 0xFF)
            self.acc = 0
            self.n = 0
        return bytes(self.buf), pad


class BitReader:
    """MSB-first bit reader over an immutable bytes payload."""

    def __init__(self, data: bytes):
        self.data = data
        self.i = 0
        self.bits = 0

    def read(self, count: int) -> int:
        if count <= 0:
            return 0
        v = 0
        i = self.i
        data = self.data
        for _ in range(count):
            v = (v << 1) | ((data[i >> 3] >> (7 - (i & 7))) & 1)
            i += 1
        self.i = i
        self.bits += count
        return v

    def read_signed(self, count: int) -> int:
        sign = self.read(1)
        v = self.read(count - 1)
        return v if sign else -v

    def read_float(self, decimals: int = 4, count: int = 20) -> float:
        return self.read(count) / (10 ** decimals)

    def read_array(self, shape, bits: int) -> np.ndarray:
        shift = 8 - bits
        arr = np.empty(int(np.prod(shape)), dtype=np.uint8)
        for i in range(arr.size):
            arr[i] = self.read(bits) << shift
        return arr.reshape(shape)

# ====================================================================================================


# =========================================== CONFIG =================================================

@dataclass
class PBC2Config:
    stroke_count: int = -1                                      # If -1, set from image size (20000 + 0.0015 * ((max_dim + 3200) ** 2))
    size_range: tuple[float, float] = (-1.0, -1.0)             # If (-1, -1), set from stroke_count
    mult_list: tuple[int, ...] = (-10, 0, 5, 20)               # Value options per stroke, encoded as an index into this list.
    start_mode: str = "Average"                                 # "Black", "White", "Custom", "Average", "Median"
    start_custom: tuple[int, int, int] = (128, 128, 128)        # Used when start_mode is "Custom".

    # Decay Function Parameters - Control how stroke sizes decrease over time.
    decay_cutoff: float = -1.0
    decay_softness: float = -1.0
    decay_progress: float = -1.0

    # Focus Area Parameters - Dynamic focus area prioritizing high-error regions for stroke placement.
    focus_strokes: int = 100
    focus_warmup: float = -1.0
    focus_max_bits: int = 8
    focus_padding: int = 4
    focus_criteria: str = "Sum"                                 # "Sum", "Max", "Min"

    # Channel Cycling Parameters.
    channel_cycle: Union[str, bool, None] = "Smart"             # "Smart", "Strict", "Balanced", "Default", False/None
    channel_cycle_strokes: int = 100
    channel_cycle_warmup: float = 0.9
    channel_cycle_criteria: str = "Min"                         # "Sum", "Max", "Min", "Median"

    color_space: str = "RGB"                                    # "RGB" or "YCbCr"
    seed: int = 28042003

    # Downsample Rate Parameters.
    downsample_rate: float = -1.0

    # Downsample Initialization Parameters.
    downsample_initialize: bool = True
    downsample_initialize_rate: float = 16.0
    downsample_initialize_bits: int = 8                         # Bits per channel for the init layer (8 = lossless).
    resample: str = "bicubic"

    # EXPERIMENTS (kept as config placeholders, not yet wired into the encode path)
    placement_mode: str = "random"
    final_only_clipping: bool = False
    rgb_native: bool = False


def _prepare_config(img_size, cfg: PBC2Config) -> PBC2Config:
    cfg = replace(cfg)
    if cfg.stroke_count == -1:
        cfg.stroke_count = int(20000 + 0.0015 * ((max(img_size) + 3200) ** 2))
    if cfg.downsample_rate == -1:
        cfg.downsample_rate = 1 if min(img_size) < 600 else min(img_size) / 500
    if cfg.downsample_initialize and cfg.downsample_initialize_rate < 32:
        if cfg.stroke_count > 20000:
            if cfg.decay_cutoff == -1:
                cfg.decay_cutoff = 0.3
            if cfg.size_range == (-1.0, -1.0):
                cfg.size_range = (0.05, 0.01)
            if cfg.focus_warmup == -1:
                cfg.focus_warmup = 0.1
        else:
            if cfg.decay_cutoff == -1:
                cfg.decay_cutoff = 0.7
            if cfg.size_range == (-1.0, -1.0):
                cfg.size_range = (0.1, 0.03)
            if cfg.focus_warmup == -1:
                cfg.focus_warmup = 0.7
    if cfg.focus_warmup == -1:
        v = 0.75 - (0.000014 * (cfg.stroke_count - 1000) ** 2) / 550000
        cfg.focus_warmup = max(0.0, min(v, 1.0))
    return cfg


@dataclass
class PBC2Result:
    image: Image.Image
    data: bytes
    config: PBC2Config
    losses: tuple[int, int, int]
    mse: float
    header_bits: int
    total_bits: int
    encode_seconds: float

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            f.write(self.data)

    def show(self) -> None:
        plt.imshow(self.image)
        plt.axis("off")
        plt.title(f"PBC2 Result - MSE: {self.mse:.2f}, Size: {self.total_bits / 8 / 1024:.2f} KB, "
                  f"Encode Time: {self.encode_seconds:.2f}s")
        plt.show()

# ====================================================================================================


# ========================================= NJIT FUNCTIONS ===========================================

@njit(inline='always')
def pcg_step(state):
    old_state = state
    state = uint64(old_state * 6364136223846793005 + 1442695040888963407)
    xorshifted = uint32(((old_state >> 18) ^ old_state) >> 27)
    rot = uint32(old_state >> 59)
    out = (xorshifted >> rot) | (xorshifted << ((-rot) & 31))
    return state, out


@njit(fastmath=True, cache=True)
def get_stroke_coords_rolling(state, r_start, c_start, r_end, c_end):
    if r_end <= r_start:
        r_end = r_start + 1
    if c_end <= c_start:
        c_end = c_start + 1
    range_h = uint32(r_end - r_start)
    range_w = uint32(c_end - c_start)
    state, rnd_row = pcg_step(state)
    state, rnd_col = pcg_step(state)
    row = r_start + int(rnd_row % range_h)
    col = c_start + int(rnd_col % range_w)
    return state, row, col


@njit(fastmath=True, cache=True)
def process_quadrant_int(height, width, size, q_int, bitcount, padding):
    row_start, row_end = 0, height
    col_start, col_end = 0, width
    split_height = True
    for b in range(bitcount - 1, -1, -1):
        bit = (q_int >> b) & 1
        if split_height:
            mid = (row_start + row_end) // 2
            if bit == 0:
                row_end = mid
            else:
                row_start = mid
        else:
            mid = (col_start + col_end) // 2
            if bit == 0:
                col_end = mid
            else:
                col_start = mid
        split_height = not split_height
    row_end -= size
    col_end -= size
    row_start = max(0, row_start - padding)
    col_start = max(0, col_start - padding)
    row_end = min(height - size, row_end + padding)
    col_end = min(width - size, col_end + padding)
    return row_start, col_start, row_end, col_end


@njit(inline='always')
def _quadrant_bounds(row, col, size, half, k, h, w):
    """Clamped (r_s, c_s, r_e, c_e) for quadrant k; r_e == r_s signals an empty quadrant."""
    r_s = row if k < 2 else row + half
    r_e = (row + half) if k < 2 else (row + size)
    c_s = col if (k & 1) == 0 else col + half
    c_e = (col + half) if (k & 1) == 0 else (col + size)
    if r_s >= h or c_s >= w or r_s >= r_e or c_s >= c_e:
        return r_s, c_s, r_s, c_s
    if r_e > h:
        r_e = h
    if c_e > w:
        c_e = w
    return r_s, c_s, r_e, c_e


@njit(fastmath=True, cache=True)
def stroke_numba(target_layer, canvas_layer, h, w, row, col, size, mult_arr):
    half = size // 2
    stroke_indices = np.zeros(4, dtype=np.int32)
    for k in range(4):
        r_s, c_s, r_e, c_e = _quadrant_bounds(row, col, size, half, k, h, w)
        if r_e == r_s:
            continue

        diff_sum = 0.0
        for rr in range(r_s, r_e):
            for cc in range(c_s, c_e):
                diff_sum += target_layer[rr, cc] - canvas_layer[rr, cc]
        mean_diff = diff_sum / ((r_e - r_s) * (c_e - c_s))

        best_idx = 0
        best_dist = abs(mult_arr[0] - mean_diff)
        for m in range(1, mult_arr.shape[0]):
            dist = abs(mult_arr[m] - mean_diff)
            if dist < best_dist:
                best_dist = dist
                best_idx = m

        best_mult = mult_arr[best_idx]
        if best_mult != 0:
            for rr in range(r_s, r_e):
                for cc in range(c_s, c_e):
                    val = canvas_layer[rr, cc] + best_mult
                    if val > 255:
                        val = 255
                    elif val < 0:
                        val = 0
                    canvas_layer[rr, cc] = val
        stroke_indices[k] = best_idx
    return stroke_indices, canvas_layer


@njit(fastmath=True, cache=True)
def stroke_numba_decompress(canvas_layer, h, w, row, col, size, mult_arr, stroke_indices):
    half = size // 2
    for k in range(4):
        r_s, c_s, r_e, c_e = _quadrant_bounds(row, col, size, half, k, h, w)
        if r_e == r_s:
            continue
        best_mult = mult_arr[stroke_indices[k]]
        if best_mult != 0:
            for rr in range(r_s, r_e):
                for cc in range(c_s, c_e):
                    val = canvas_layer[rr, cc] + best_mult
                    if val > 255:
                        val = 255
                    elif val < 0:
                        val = 0
                    canvas_layer[rr, cc] = val
    return canvas_layer

# ====================================================================================================


# ======================================= CORE ENCODE/DECODE =========================================

class PBC:
    """# Probabilistic Brush Compression (PBC) \n
    ---
    ### Developed by **Ege Eken** (https://github.com/EgeEken/PBC) \n
    Current Version: **V2.4** (2026) \n\n
    ---
    This is a lossy image compression algorithm that compresses images into a series of brush stroke instructions.\n
    For more information, visit the [GitHub Repository](https://github.com/EgeEken/PBC)
    """
    Config = PBC2Config
    Result = PBC2Result

    # ---------------------------------- public API ----------------------------------

    @staticmethod
    def compress(img, config: Optional[PBC2Config] = None, **overrides) -> PBC2Result:
        result = None
        for out in PBC._encode(img, config, overrides, stream_interval=None):
            result = out
        return result

    @staticmethod
    def compress_stream(img, config: Optional[PBC2Config] = None, stream_interval: int = 100, **overrides):
        yield from PBC._encode(img, config, overrides, stream_interval=stream_interval)

    @staticmethod
    def decompress(data: Union[bytes, bytearray, str], return_result: bool = False):
        if isinstance(data, str):
            with open(data, "rb") as f:
                data = f.read()
        data = bytes(data)
        if data[:4] != MAGIC:
            raise ValueError("Invalid PBC2 file (bad magic).")
        br = BitReader(data[6:])  # skip MAGIC(4) + VERSION(1) + pad(1)

        downsample_flag = br.read(1)
        original_w = original_h = -1
        if downsample_flag:
            original_w = br.read(16)
            original_h = br.read(16)
        ycbcr = br.read(1)
        resample_name = _resample_name(br.read(3))
        resample = RESAMPLE[resample_name]
        seed = br.read(32)
        h = br.read(16)
        w = br.read(16)
        stroke_count = br.read(20)
        start_color = (br.read(8), br.read(8), br.read(8))

        if br.read(1):  # downsample-initialize flag
            init_bits = br.read(4)
            n_h = br.read(10)
            n_w = br.read(10)
            canvas_rgb = br.read_array((n_h, n_w, 3), init_bits)
            canvas = np.array(_resize(Image.fromarray(canvas_rgb), (w, h), resample), dtype=np.int16)
            if ycbcr:
                canvas = rgb_to_ycbcr(canvas).astype(np.int16)
        else:
            canvas = np.full((h, w, 3), start_color, dtype=np.int16)

        size_start = br.read(16)
        size_end = br.read(16)
        c_auto = br.read(1); cv = br.read_float(4, 20)
        s_auto = br.read(1); sv = br.read_float(4, 20)
        p_auto = br.read(1); pv = br.read_float(4, 20)
        _, cutoff = _resolve_decay_value(-1 if c_auto else cv, stroke_count, "cutoff")
        _, softness = _resolve_decay_value(-1 if s_auto else sv, stroke_count, "softness")
        _, progress = _resolve_decay_value(-1 if p_auto else pv, stroke_count, "progress")
        sizes = _decay_curve(stroke_count, size_start, size_end, cutoff, softness, progress)

        len_multlist = br.read(9)
        mult_list = [br.read_signed(9) for _ in range(len_multlist)]
        mult_arr = np.array(mult_list)
        mb = int(np.ceil(np.log2(len(mult_arr)))) if len(mult_arr) > 1 else 0

        focus_warmup_strokes = br.read(20)
        focus_strokes = br.read(20)
        focus_max_bits = br.read(8)
        focus_padding = br.read(8)

        cycle = bool(br.read(1))
        if cycle:
            channel_cycle_strokes = br.read(20)
            channel_cycle_warmup_strokes = br.read(20)
        else:
            channel_cycle_strokes = channel_cycle_warmup_strokes = 0

        canvas = PBC._drain(PBC._run_loop(
            br, None, canvas, None, sizes, stroke_count, h, w, seed, mult_arr, mb,
            focus_warmup_strokes, focus_strokes, focus_max_bits, focus_padding, None,
            cycle, channel_cycle_strokes, channel_cycle_warmup_strokes, None, None, None,
        ))

        canvas = np.clip(canvas, 0, 255)
        final = ycbcr_to_rgb(canvas) if ycbcr else canvas.astype(np.uint8)
        final_pil = Image.fromarray(final)
        if downsample_flag:
            final_pil = _resize(final_pil, (original_w, original_h), resample)

        if not return_result:
            return final_pil
        cfg = PBC2Config(stroke_count=stroke_count, size_range=(size_start, size_end),
                         mult_list=tuple(mult_list), start_custom=start_color,
                         decay_cutoff=cutoff, decay_softness=softness, decay_progress=progress,
                         focus_strokes=focus_strokes, focus_max_bits=focus_max_bits,
                         focus_padding=focus_padding, channel_cycle=cycle,
                         channel_cycle_strokes=channel_cycle_strokes,
                         color_space="YCbCr" if ycbcr else "RGB", seed=seed, resample=resample_name)
        return PBC2Result(final_pil, data, cfg, (-1, -1, -1), -1.0, -1, br.bits, 0.0)

    # ---------------------------------- internals ----------------------------------

    @staticmethod
    def _drain(gen):
        try:
            while True:
                next(gen)
        except StopIteration as e:
            return e.value

    @staticmethod
    def _encode(img, config, overrides, stream_interval):
        t0 = perf_counter()

        if isinstance(img, str):
            img = Image.open(img)
        elif isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        img = img.convert("RGB")
        original_size = img.size  # (w, h)

        cfg = config or PBC2Config()
        if overrides:
            cfg = replace(cfg, **overrides)
        cfg = _prepare_config(original_size, cfg)

        ycbcr = cfg.color_space == "YCbCr"
        resample = RESAMPLE[cfg.resample]
        work = Image.fromarray(np.array(img.convert("YCbCr"))) if ycbcr else img

        bw = BitWriter()

        if cfg.downsample_rate > 1:
            ds = _downsample(work, cfg.downsample_rate, resample)
            arr = np.array(ds, dtype=np.int16)
            bw.write(1, 1)
            bw.write(original_size[0], 16)
            bw.write(original_size[1], 16)
        else:
            ds = None
            arr = np.array(work, dtype=np.int16)
            bw.write(0, 1)

        bw.write(1 if ycbcr else 0, 1)
        bw.write(_resample_id(cfg.resample), 3)
        bw.write(cfg.seed, 32)

        h, w = arr.shape[:2]
        bw.write(h, 16)
        bw.write(w, 16)
        bw.write(cfg.stroke_count, 20)

        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        if cfg.start_mode == "Black":
            start_color = (0, 0, 0) if not ycbcr else (0, 128, 128)
        elif cfg.start_mode == "White":
            start_color = (255, 255, 255) if not ycbcr else (255, 128, 128)
        elif cfg.start_mode == "Custom":
            start_color = tuple(cfg.start_custom)
        elif cfg.start_mode in ("Average", "Mean"):
            start_color = (int(np.mean(r)), int(np.mean(g)), int(np.mean(b)))
        elif cfg.start_mode == "Median":
            start_color = (int(np.median(r)), int(np.median(g)), int(np.median(b)))
        elif cfg.start_mode == "True Median":
            start_color = tuple(int(v) for v in np.median(arr.reshape(-1, 3), axis=0))
        else:
            start_color = (int(np.mean(r)), int(np.mean(g)), int(np.mean(b)))
        for c in start_color:
            bw.write(int(c), 8)

        if cfg.downsample_initialize:
            bw.write(1, 1)
            init_bits = cfg.downsample_initialize_bits
            n_h = int(h / cfg.downsample_initialize_rate)
            n_w = int(w / cfg.downsample_initialize_rate)
            bw.write(init_bits, 4)
            bw.write(n_h, 10)
            bw.write(n_w, 10)
            src = ds if cfg.downsample_rate > 1 else work
            canvas_rgb = np.array(_resize(src, (n_w, n_h), resample), dtype=np.uint8)
            if ycbcr:
                canvas_rgb = ycbcr_to_rgb(canvas_rgb)
            bw.write_array(canvas_rgb, init_bits)
            canvas = np.array(_resize(Image.fromarray(canvas_rgb), (w, h), resample), dtype=np.int16)
            if ycbcr:
                canvas = rgb_to_ycbcr(canvas).astype(np.int16)
        else:
            bw.write(0, 1)
            canvas = np.full((h, w, 3), start_color, dtype=np.int16)

        ss, se = cfg.size_range
        if ss == -1:
            ss = 0.3 + (1 / 1.00095) ** (cfg.stroke_count + 7000)
        if se == -1:
            se = 0.01 + (1 / 1.00015) ** (cfg.stroke_count + 10200)
        size_start = int(ss * (min(h, w) - 2)) + 2
        size_end = int(se * (min(h, w) - 2)) + 2
        bw.write(size_start, 16)
        bw.write(size_end, 16)

        c_auto, cutoff = _resolve_decay_value(cfg.decay_cutoff, cfg.stroke_count, "cutoff")
        s_auto, softness = _resolve_decay_value(cfg.decay_softness, cfg.stroke_count, "softness")
        p_auto, progress = _resolve_decay_value(cfg.decay_progress, cfg.stroke_count, "progress")
        bw.write(1 if c_auto else 0, 1); bw.write_float(cutoff, 4, 20)
        bw.write(1 if s_auto else 0, 1); bw.write_float(softness, 4, 20)
        bw.write(1 if p_auto else 0, 1); bw.write_float(progress, 4, 20)
        sizes = _decay_curve(cfg.stroke_count, size_start, size_end, cutoff, softness, progress)

        mult_arr = np.array(cfg.mult_list)
        mb = int(np.ceil(np.log2(len(mult_arr)))) if len(mult_arr) > 1 else 0
        bw.write(len(mult_arr), 9)
        for m in mult_arr:
            bw.write_signed(int(m), 9)

        focus_warmup_strokes = int(cfg.focus_warmup * cfg.stroke_count)
        bw.write(focus_warmup_strokes, 20)
        bw.write(cfg.focus_strokes, 20)
        bw.write(cfg.focus_max_bits, 8)
        bw.write(cfg.focus_padding, 8)

        cycle_name = cfg.channel_cycle
        cycle = bool(cycle_name)
        channel_cycle_warmup_strokes = int(cfg.channel_cycle_warmup * cfg.stroke_count)
        bw.write(1 if cycle else 0, 1)
        if cycle:
            bw.write(cfg.channel_cycle_strokes, 20)
            bw.write(channel_cycle_warmup_strokes, 20)

        header_bits = bw.bits

        canvas = yield from PBC._run_loop(
            None, bw, canvas, arr, sizes, cfg.stroke_count, h, w, cfg.seed, mult_arr, mb,
            focus_warmup_strokes, cfg.focus_strokes, cfg.focus_max_bits, cfg.focus_padding,
            cfg.focus_criteria, cycle, cfg.channel_cycle_strokes, channel_cycle_warmup_strokes,
            cycle_name if isinstance(cycle_name, str) else "Default", cfg.channel_cycle_criteria,
            (stream_interval, ycbcr),
        )

        canvas = np.clip(canvas, 0, 255)
        final = ycbcr_to_rgb(canvas) if ycbcr else canvas.astype(np.uint8)
        final_pil = Image.fromarray(final)
        if cfg.downsample_rate > 1:
            final_pil = _resize(final_pil, original_size, resample)

        payload, pad = bw.finish()
        data = MAGIC + bytes([VERSION, pad]) + payload

        final_arr = np.array(final_pil)
        orig_arr = np.array(img)
        losses = tuple(int(np.mean((orig_arr[:, :, c].astype(np.float32) -
                                    final_arr[:, :, c].astype(np.float32)) ** 2)) for c in range(3))
        mse = float(np.mean(losses))

        yield PBC2Result(final_pil, data, cfg, losses, mse, header_bits, bw.bits, perf_counter() - t0)

    @staticmethod
    def _run_loop(br, bw, canvas, arr, sizes, stroke_count, h, w, seed, mult_arr, mb,
                  focus_warmup_strokes, focus_strokes, focus_max_bits, focus_padding, focus_criteria,
                  cycle, channel_cycle_strokes, channel_cycle_warmup_strokes,
                  cycle_strategy, cycle_criteria, stream_ctx):
        """Shared stroke loop, used by both encode (bw set) and decode (br set).
        Generator: yields stream updates during encode; returns the final canvas."""
        encoding = bw is not None
        rng_state = np.uint64(seed)
        focus_codes = [None, None, None]
        focus_bits = [0, 0, 0]
        focus_counters = [focus_warmup_strokes // 3] * 3
        channel_selector = [0, 1, 2]
        cycle_timer = channel_cycle_warmup_strokes
        stream_interval = stream_ctx[0] if stream_ctx else None
        ycbcr = stream_ctx[1] if stream_ctx else False
        stream_timer = 0

        for i in range(stroke_count):
            ch = channel_selector[i % 3]
            canvas_layer = canvas[:, :, ch]
            size = int(sizes[i])

            if focus_counters[ch] <= 0:
                bc = _focus_bitcount(h, w, size, focus_max_bits)
                if bc > 0:
                    if encoding:
                        err = np.abs(arr[:, :, ch] - canvas_layer)
                        code = _select_focus(err, bc, focus_criteria)
                        bw.write(code, bc)
                    else:
                        code = br.read(bc)
                    focus_codes[ch] = code
                    focus_bits[ch] = bc
                else:
                    focus_codes[ch] = None
                    focus_bits[ch] = 0
                focus_counters[ch] = focus_strokes

            if cycle and cycle_timer <= 0:
                if encoding:
                    channel_selector = _channel_cycle(np.abs(arr - canvas), cycle_strategy, cycle_criteria)
                    for c in channel_selector:
                        bw.write(c, 2)
                else:
                    channel_selector = [br.read(2) for _ in range(3)]
                cycle_timer = channel_cycle_strokes

            if focus_codes[ch] is None:
                r_s, c_s = 0, 0
                r_e, c_e = h - size, w - size
            else:
                r_s, c_s, r_e, c_e = process_quadrant_int(h, w, size, focus_codes[ch], focus_bits[ch], focus_padding)
            rng_state, row, col = get_stroke_coords_rolling(rng_state, r_s, c_s, r_e, c_e)
            rng_state = np.uint64(rng_state)

            if encoding:
                indices, _ = stroke_numba(arr[:, :, ch], canvas_layer, h, w, row, col, size, mult_arr)
                for idx in indices:
                    bw.write(int(idx), mb)
            else:
                indices = np.array([br.read(mb) for _ in range(4)], dtype=np.int32)
                stroke_numba_decompress(canvas_layer, h, w, row, col, size, mult_arr, indices)

            focus_counters[ch] -= 1
            if cycle:
                cycle_timer -= 1

            if encoding and stream_interval:
                if stream_timer <= 0:
                    stream_timer = stream_interval - 1
                    interim = np.clip(canvas, 0, 255).astype(np.uint8)
                    if ycbcr:
                        interim = ycbcr_to_rgb(interim)
                    losses = [int(np.mean((interim[:, :, c].astype(np.float32) -
                                           arr[:, :, c].astype(np.float32)) ** 2)) for c in range(3)]
                    yield (Image.fromarray(interim),
                           f"Processed {i}/{stroke_count} strokes. {(i / stroke_count) * 100:.2f}%",
                           bw.bits, losses)
                else:
                    stream_timer -= 1

        return canvas

# ====================================================================================================
# ====================================================================================================

# Preloading numba when imported to prevent first run excess latency

def preload_numba() -> None:
    img = Image.fromarray(np.random.randint(0, 256, (16, 16, 3), dtype=np.uint8))
    PBC.compress(img, PBC2Config(stroke_count=10, downsample_initialize=False, focus_warmup=0.0))


preload_numba()
