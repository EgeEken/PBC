
# ====================================================================================================
#
#           PBC v3.0 - Probabilistic Brush Compression
#           Lossy Image Compression Algorithm by EgeEken (github.com/EgeEken)
#           3.0 Update - 2026-06 - Whole algorithm overhaul
#
# ====================================================================================================

from dataclasses import dataclass, field
import time
import math
import os
import lzma
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


class BitWriter:
    def __init__(self):
        self.data = bytearray()
        self.acc = 0
        self.nbits = 0

    def write(self, value, bitcount):
        value = int(value)
        if bitcount <= 0:
            return
        if value < 0 or value >= (1 << bitcount):
            raise ValueError(f"value {value} does not fit in {bitcount} bits")
        self.acc = (self.acc << bitcount) | value
        self.nbits += bitcount
        while self.nbits >= 8:
            shift = self.nbits - 8
            self.data.append((self.acc >> shift) & 255)
            self.acc &= (1 << shift) - 1
            self.nbits -= 8

    def finish(self):
        if self.nbits:
            self.data.append((self.acc << (8 - self.nbits)) & 255)
            self.acc = 0
            self.nbits = 0
        return bytes(self.data)


class BitReader:
    def __init__(self, data):
        self.data = data
        self.i = 0
        self.acc = 0
        self.nbits = 0

    def read(self, bitcount):
        while self.nbits < bitcount:
            if self.i >= len(self.data):
                raise EOFError("bitstream ended early")
            self.acc = (self.acc << 8) | self.data[self.i]
            self.i += 1
            self.nbits += 8
        shift = self.nbits - bitcount
        value = (self.acc >> shift) & ((1 << bitcount) - 1)
        self.acc &= (1 << shift) - 1
        self.nbits -= bitcount
        return value


@dataclass
class PBC3Config:
    patch_count: int = 20
    search_depth: int = 200
    proposal_depth: int = 50
    exact_depth: int = 10
    min_patch_size: int = 16
    max_patch_size: int = 400
    min_cell_size: int = 1
    max_cell_size: int = 64
    cell_sizes_per_candidate: int = 3
    top_k: int = 20
    search_q_start: float = 0.4
    search_q_end: float = 0.1
    q_init: float = 0.7
    q_start: float = 0.9
    q_end: float = 0.9
    color_space: str = "YCbCr"
    channel_cycle: str = "Sum"
    auto_downsample_init: bool = True
    init_search_depth: int = 7
    downsample_init_cell_size: int = 12
    downsample_palette_bitcount: int = 6
    downsample_rate: float = -1
    auto_downsample_max_pixels: int = 250_000
    patch_palette_bitcount: int = 2
    patch_bitcount_mode: str = "constant"     # "constant" | "dynamic"
    palette_mode: str = "generated"           # "generated" | "explicit" | "auto"
    palette_difference_threshold: int = 0
    palette_difference_threshold_mode: str = "constant"  # "constant" | "linear"
    explicit_palette_max_bitcount: int = 3
    quality_target_mae: float = 0.0  # >0: stop a channel once its mean abs error drops to/below this
    mask_size: int = 4
    anchor_block_size: int = 8
    dynamic_patch_bitcount_min: int = 2
    dynamic_patch_bitcount_max: int = 3
    positive_bias: bool = True
    random_seed: int = 2003
    debug_mode: bool = False
    debug_print: bool = False
    debug_path: str = None

    def __post_init__(self):
        self.channel_cycle = str(self.channel_cycle)
        self.patch_bitcount_mode = str(self.patch_bitcount_mode)

    @classmethod
    def fast(cls):
        return cls(
            patch_count=10,
            search_depth=100,
            proposal_depth=10,
            exact_depth=5,
            cell_sizes_per_candidate=1,
            search_q_start=0.35,
            q_init=0.5,
            init_search_depth=7,
            auto_downsample_max_pixels=200_000,
        )


@dataclass
class PBC3Result:
    image: Image.Image
    data: bytes
    config: PBC3Config
    mse: float
    encode_seconds: float
    total_bits: int
    original_width: int = None
    original_height: int = None
    working_width: int = None
    working_height: int = None
    timings: dict = field(default_factory=dict)
    debug_path: str = None
    channels: int = 3

    @property
    def original_bits(self):
        w = self.original_width or self.image.width
        h = self.original_height or self.image.height
        return w * h * self.channels * 8

    @property
    def compressed_kb(self):
        return self.total_bits / 8 / 1024

    @property
    def original_kb(self):
        return self.original_bits / 8 / 1024

    @property
    def compression_rate(self):
        return self.original_bits / self.total_bits if self.total_bits else float("inf")

    @property
    def compressed_percent(self):
        return self.total_bits / self.original_bits * 100 if self.original_bits else 0

    def save(self, path):
        if self.data is None:
            raise ValueError("result has no compressed data to save")
        with open(path, "wb") as f:
            f.write(self.data)

    def verify(self):
        if self.data is None:
            return False
        decoded = PBC3.decompress(self.data).image
        return np.array_equal(np.asarray(self.image), np.asarray(decoded))

    def show(self):
        fig = plt.figure(figsize=(8, 7.4), dpi=130)
        gs = fig.add_gridspec(3, 1, height_ratios=[0.09, 0.16, 1.0], hspace=0.04)
        title_ax = fig.add_subplot(gs[0])
        info_ax = fig.add_subplot(gs[1])
        image_ax = fig.add_subplot(gs[2])
        for ax in (title_ax, info_ax, image_ax):
            ax.axis("off")
        title_ax.text(0.5, 0.5, "PBC3 Result", ha="center", va="center", fontsize=16, fontweight="bold")
        mse = "N/A" if self.mse is None else f"{self.mse:.2f}"
        seconds = "N/A" if self.encode_seconds is None else f"{self.encode_seconds:.3f}s"
        debug = f"   |   Debug: {os.path.basename(self.debug_path)}" if self.debug_path else ""
        info = (
            f"MSE: {mse}   |   Compressed: {self.compressed_kb:.2f} KB   |   Original: {self.original_kb:.2f} KB\n"
            f"Compression: {self.compression_rate:.2f}x ({self.compressed_percent:.2f}%)   |   Time: {seconds}{debug}"
        )
        info_ax.text(0.5, 0.5, info, ha="center", va="center", color="white", fontsize=10, linespacing=1.35,
                     bbox=dict(boxstyle="round,pad=0.5", facecolor="black", alpha=0.72, edgecolor="none"))
        image_ax.imshow(self.image)
        plt.show()


class PBC3:
    MAGIC = b"PBC3"
    VERSION = 0
    PALETTE_GENERATED = 0
    PALETTE_EXPLICIT = 1
    ENTROPY_STORE = 0
    ENTROPY_LZMA = 2
    _LZMA_FILTERS = [{"id": lzma.FILTER_LZMA2, "preset": lzma.PRESET_EXTREME}]
    COLOR_SPACES = {"RGB": 0, "YCbCr": 1}
    COLOR_SPACE_NAMES = {0: "RGB", 1: "YCbCr"}
    RESAMPLE_FILTER = Image.Resampling.BICUBIC
    RESAMPLE_REDUCING_GAP = None
    USE_NUMBA_RESAMPLE = False  # opt-in numba bicubic (pbc3_resample.py); PIL by default

    @staticmethod
    def _to_image(image):
        if isinstance(image, Image.Image):
            return image
        if isinstance(image, str):
            return Image.open(image)
        arr = np.asarray(image)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        mode = "RGBA" if arr.ndim == 3 and arr.shape[-1] == 4 else "RGB"
        return Image.fromarray(arr, mode)

    @staticmethod
    def _has_alpha(img):
        return img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info)

    @staticmethod
    def _ceil_div(a, b):
        return (a + b - 1) // b

    @staticmethod
    def _add_time(timings, key, seconds):
        timings[key] = timings.get(key, 0.0) + seconds

    @staticmethod
    def _norm(values):
        arr = np.asarray(values, dtype=np.float64)
        rng = arr.max() - arr.min()
        if rng <= 0:
            return np.ones_like(arr)
        return (arr - arr.min()) / rng

    @staticmethod
    def _interp(start, end, step, count):
        if count <= 1:
            return float(end)
        p = (step - 1) / max(1, count - 1)
        return float(start) * (1 - p) + float(end) * p

    @classmethod
    def _entropy_pack(cls, body):
        x = lzma.compress(body, format=lzma.FORMAT_RAW, filters=cls._LZMA_FILTERS)
        if len(x) < len(body):
            return cls.ENTROPY_LZMA, x
        return cls.ENTROPY_STORE, body

    @classmethod
    def _entropy_unpack(cls, method, body):
        if method == cls.ENTROPY_STORE:
            return body
        if method == cls.ENTROPY_LZMA:
            return lzma.decompress(body, format=lzma.FORMAT_RAW, filters=cls._LZMA_FILTERS)
        raise ValueError(f"unknown entropy method {method}")

    @classmethod
    def _open_body(cls, data):
        if data[:4] != cls.MAGIC:
            raise ValueError("not a PBC3 file")
        version = data[4]
        if version != cls.VERSION:
            raise ValueError(f"unsupported PBC3 version {version}")
        return version, cls._entropy_unpack(data[5], data[6:])

    @classmethod
    def _auto_downsample_rate(cls, image_size, downsample_rate, max_pixels):
        if downsample_rate != -1:
            return float(downsample_rate)
        w, h = image_size
        pixels = w * h
        max_pixels = max(1, int(max_pixels))
        if pixels <= max_pixels:
            return 1.0
        return math.sqrt(pixels / max_pixels)

    @classmethod
    def _downsample_image(cls, img, rate):
        if rate <= 1:
            return img.copy()
        w = max(1, int(round(img.size[0] / rate)))
        h = max(1, int(round(img.size[1] / rate)))
        return img.resize((w, h), cls.RESAMPLE_FILTER, reducing_gap=cls.RESAMPLE_REDUCING_GAP)

    @staticmethod
    def _palette_bounds(values):
        min_value = int(np.min(values))
        max_value = int(np.max(values))
        return min(255, max(0, -min_value)), min(255, max(0, max_value))

    @classmethod
    def _range_counts(cls, mask_size, negative_max=255, positive_max=255, positive_bias=True):
        side_bits = max(0, mask_size - 1)
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

    @classmethod
    def _mask_index_for_value(cls, value, mask_size, negative_max=255, positive_max=255, positive_bias=True):
        if value == 0:
            return 0
        pos_count, neg_count = cls._range_counts(mask_size, negative_max, positive_max, positive_bias)
        if value > 0:
            if pos_count == 0 or positive_max <= 0:
                return None
            mag = min(int(value), positive_max)
            bin_i = min((mag - 1) * pos_count // positive_max, pos_count - 1)
            return 1 + bin_i
        if neg_count == 0 or negative_max <= 0:
            return None
        mag = min(int(-value), negative_max)
        bin_i = min((mag - 1) * neg_count // negative_max, neg_count - 1)
        return 1 + pos_count + bin_i

    @classmethod
    def _range_for_mask_index(cls, index, mask_size, negative_max=255, positive_max=255, positive_bias=True):
        pos_count, neg_count = cls._range_counts(mask_size, negative_max, positive_max, positive_bias)
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

    @classmethod
    def _mask_from_values(cls, values, mask_size, negative_max=255, positive_max=255, positive_bias=True):
        # Vectorized equivalent of looping _mask_index_for_value over every cell.
        mask = [0] * mask_size
        mask[0] = 1
        pos_count, neg_count = cls._range_counts(mask_size, negative_max, positive_max, positive_bias)
        flat = np.clip(np.rint(np.asarray(values)).astype(np.int32).ravel(), -negative_max, positive_max)
        if pos_count > 0 and positive_max > 0:
            pos = flat[flat > 0]
            if pos.size:
                bins = 1 + np.minimum((np.minimum(pos, positive_max) - 1) * pos_count // positive_max, pos_count - 1)
                for b in np.unique(bins):
                    if b < mask_size:
                        mask[int(b)] = 1
        if neg_count > 0 and negative_max > 0:
            neg = flat[flat < 0]
            if neg.size:
                mag = np.minimum(-neg, negative_max)
                bins = 1 + pos_count + np.minimum((mag - 1) * neg_count // negative_max, neg_count - 1)
                for b in np.unique(bins):
                    if b < mask_size:
                        mask[int(b)] = 1
        return mask

    @classmethod
    def _active_value_count(cls, mask, negative_max=255, positive_max=255, positive_bias=True):
        count = 0
        for i, bit in enumerate(mask):
            if bit:
                r = cls._range_for_mask_index(i, len(mask), negative_max, positive_max, positive_bias)
                if r is not None:
                    start, end = r
                    count += end - start + 1
        return max(1, count)

    @classmethod
    def resolve_palette_bitcount(cls, mask, max_bitcount, negative_max=255, positive_max=255, positive_bias=True):
        value_count = cls._active_value_count(mask, negative_max, positive_max, positive_bias)
        needed = max(1, math.ceil(math.log2(value_count)))
        return min(int(max_bitcount), needed)

    @classmethod
    def palette_generator(cls, mask, max_bitcount, negative_max=255, positive_max=255, positive_bias=True):
        bitcount = cls.resolve_palette_bitcount(mask, max_bitcount, negative_max, positive_max, positive_bias)
        size = 1 << bitcount
        active_ranges = []
        for i, bit in enumerate(mask):
            if bit:
                r = cls._range_for_mask_index(i, len(mask), negative_max, positive_max, positive_bias)
                if r is not None:
                    active_ranges.append(r)
        palette = []
        if mask and mask[0]:
            palette.append(0)
            active_ranges = [r for r in active_ranges if r != (0, 0)]
        value_count = cls._active_value_count(mask, negative_max, positive_max, positive_bias)
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

    @classmethod
    def _top_values_palette(cls, small, bitcount, threshold):
        size = 1 << bitcount
        flat = np.clip(np.rint(np.asarray(small)).astype(np.int32).ravel(), -255, 255)
        vals, counts = np.unique(flat, return_counts=True)
        centroids = [0.0]
        binw = max(1, int(threshold))
        agg = {}
        for v, ct in zip(np.round(vals / binw) * binw, counts):
            agg[float(v)] = agg.get(float(v), 0) + int(ct)
        for v in sorted((k for k in agg if k != 0.0), key=lambda k: agg[k], reverse=True):
            centroids.append(v)
            if len(centroids) >= size:
                break
        centroids = np.array(centroids, dtype=np.float64)
        if centroids.size > 1 and vals.size:
            w = counts.astype(np.float64)
            for _ in range(8):
                assign = np.argmin(np.abs(vals[:, None] - centroids[None, :]), axis=1)
                new = centroids.copy()
                for k in range(centroids.size):
                    sel = assign == k
                    wk = w[sel].sum()
                    if wk > 0:
                        new[k] = float((vals[sel] * w[sel]).sum() / wk)
                new[0] = 0.0
                if np.array_equal(np.rint(new), np.rint(centroids)):
                    centroids = new
                    break
                centroids = new
        pal = np.rint(centroids).astype(np.int16)
        if pal.size < size:
            pal = np.concatenate([pal, np.zeros(size - pal.size, dtype=np.int16)])
        return pal[:size]

    @staticmethod
    def quantize_signed(values, palette):
        vals = np.asarray(values, dtype=np.int16)
        pal = np.asarray(palette, dtype=np.int16)
        dist = np.abs(vals[..., None].astype(np.int32) - pal[None, None, :].astype(np.int32))
        return np.argmin(dist, axis=-1).astype(np.uint16)

    @classmethod
    def signed_resample(cls, values, out_h, out_w):
        values = np.asarray(values, dtype=np.float32)
        out_h, out_w = int(out_h), int(out_w)
        if values.shape == (out_h, out_w):
            return np.rint(values).astype(np.int16)
        if cls.USE_NUMBA_RESAMPLE:
            from pbc3_resample import resample_bicubic
            out = resample_bicubic(values, out_h, out_w)
        else:
            resized = Image.fromarray(values).resize((out_w, out_h), cls.RESAMPLE_FILTER, reducing_gap=cls.RESAMPLE_REDUCING_GAP)
            out = np.asarray(resized, dtype=np.float32)
        return np.rint(out).astype(np.int16)

    @classmethod
    def signed_resample_cells(cls, values, cell_size):
        h, w = values.shape
        return cls.signed_resample(values, cls._ceil_div(h, cell_size), cls._ceil_div(w, cell_size))

    @classmethod
    def apply_grid(cls, canvas_layer, x, y, w, h, cell_size, values):
        patch = cls.signed_resample(values, h, w).astype(np.int32)
        canvas_layer[y:y + h, x:x + w] += patch

    @staticmethod
    def _integral(a):
        return np.pad(a.astype(np.int64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))

    @classmethod
    def _cell_edges(cls, start, length, cell_size):
        n = cls._ceil_div(length, cell_size)
        edges = start + np.arange(n + 1) * cell_size
        edges[n] = start + length
        return edges

    @classmethod
    def _box_cell_bound(cls, integral, x, y, bw, bh, cell_size):
        xe = cls._cell_edges(x, bw, cell_size)
        ye = cls._cell_edges(y, bh, cell_size)
        corners = integral[np.ix_(ye, xe)].astype(np.float64)
        cell_sum = corners[1:, 1:] - corners[:-1, 1:] - corners[1:, :-1] + corners[:-1, :-1]
        counts = (np.diff(ye)[:, None] * np.diff(xe)[None, :]).astype(np.float64)
        return float(np.sum(cell_sum * cell_sum / counts))

    @classmethod
    def _write_grid(cls, bw, flat, bitcount):
        for value in flat:
            bw.write(int(value), bitcount)

    @classmethod
    def _read_grid(cls, br, n, bitcount):
        flat = np.zeros(n, dtype=np.uint16)
        for k in range(n):
            flat[k] = br.read(bitcount)
        return flat

    @classmethod
    def _patch_bits_for(cls, patch, channel_bits):
        w, h, cell = patch["w"], patch["h"], patch["cell_size"]
        bitcount = patch["bitcount"]
        grid_bits = cls._ceil_div(w, cell) * cls._ceil_div(h, cell) * bitcount
        base = channel_bits + 64 + 16 + 1
        if patch["palette_mode"] == cls.PALETTE_EXPLICIT:
            header = base + 4 + (1 << bitcount) * 9
        else:
            header = base + 10 + len(patch["mask"]) + 8 + 8 + 4
        return header + grid_bits

    @classmethod
    def _patch_header_bits(cls, channel_bits, mask_size):
        return channel_bits + 64 + 10 + mask_size + 8 + 8 + 4 + 16 + 1

    @classmethod
    def _palette_threshold(cls, config, step):
        base = int(config.palette_difference_threshold)
        if base <= 0:
            return 0
        if str(config.palette_difference_threshold_mode).lower() != "linear" or config.patch_count <= 1:
            return base
        progress = (step - 1) / max(1, config.patch_count - 1)
        if progress >= 0.9:
            return 0
        return int(round(base * (1 - progress / 0.9)))

    @classmethod
    def _palette_mode_options(cls, config, bitcount):
        mode = str(config.palette_mode).lower()
        if mode == "generated":
            return [cls.PALETTE_GENERATED]
        if mode == "explicit":
            return [cls.PALETTE_EXPLICIT]
        opts = [cls.PALETTE_GENERATED]
        if bitcount <= int(config.explicit_palette_max_bitcount):
            opts.append(cls.PALETTE_EXPLICIT)
        return opts

    @classmethod
    def _channel_error_score(cls, target, canvas, channel, mode):
        err = np.abs(target[:, :, channel] - np.clip(canvas[:, :, channel], 0, 255))
        if str(mode).lower() == "max":
            return float(np.max(err))
        return float(np.sum(err))

    @classmethod
    def _channel_mae(cls, target, canvas, channel):
        return float(np.mean(np.abs(target[:, :, channel] - np.clip(canvas[:, :, channel], 0, 255))))

    @classmethod
    def _choose_channel(cls, scores, step, channels, mode, done=None):
        allowed = [c for c in range(channels) if not (done and done[c])]
        if not allowed:
            return None
        mode = str(mode).lower()
        if mode in {"sum", "max"}:
            return max(allowed, key=lambda c: scores[c])
        return allowed[(step - 1) % len(allowed)]

    @classmethod
    def _write_patch(cls, bw, patch, channel_bits):
        bw.write(patch["channel"], channel_bits)
        bw.write(patch["x"], 16)
        bw.write(patch["y"], 16)
        bw.write(patch["w"], 16)
        bw.write(patch["h"], 16)
        pm = patch["palette_mode"]
        bw.write(pm, 1)
        bitcount = patch["bitcount"]
        if pm == cls.PALETTE_EXPLICIT:
            bw.write(bitcount, 4)
            for v in patch["palette"]:
                bw.write(int(v) & 0x1FF, 9)
        else:
            mask = patch["mask"]
            bw.write(len(mask), 10)
            for bit in mask:
                bw.write(bit, 1)
            bw.write(patch["neg"], 8)
            bw.write(patch["pos"], 8)
            bw.write(patch["max_bitcount"], 4)
        flat = patch["indices"].ravel().astype(np.int64)
        bw.write(patch["cell_size"], 16)
        cls._write_grid(bw, flat, bitcount)

    @classmethod
    def _read_patch(cls, br, channel_bits, positive_bias=True):
        channel = br.read(channel_bits)
        x = br.read(16)
        y = br.read(16)
        w = br.read(16)
        h = br.read(16)
        pm = br.read(1)
        if pm == cls.PALETTE_EXPLICIT:
            bitcount = br.read(4)
            size = 1 << bitcount
            palette = np.empty(size, dtype=np.int16)
            for i in range(size):
                raw = br.read(9)
                palette[i] = raw - 512 if raw >= 256 else raw
        else:
            mask_size = br.read(10)
            mask = [br.read(1) for _ in range(mask_size)]
            negative_max = br.read(8)
            positive_max = br.read(8)
            max_bitcount = br.read(4)
            bitcount = cls.resolve_palette_bitcount(mask, max_bitcount, negative_max, positive_max, positive_bias)
            palette = cls.palette_generator(mask, max_bitcount, negative_max, positive_max, positive_bias)
        cell_size = br.read(16)
        gw = cls._ceil_div(w, cell_size)
        gh = cls._ceil_div(h, cell_size)
        flat = cls._read_grid(br, gh * gw, bitcount)
        indices = flat.reshape(gh, gw)
        values = palette[indices]
        return channel, x, y, w, h, cell_size, values

    @classmethod
    def _make_patch(cls, channel, x, y, w, h, cell_size, residual, config, max_bitcount, palette_mode, threshold):
        small = cls.signed_resample_cells(residual, cell_size)
        if palette_mode == cls.PALETTE_EXPLICIT:
            bitcount = int(max_bitcount)
            palette = cls._top_values_palette(small, bitcount, threshold)
            indices = cls.quantize_signed(np.clip(small, -255, 255), palette)
            values = palette[indices]
            return {
                "channel": channel, "x": x, "y": y, "w": w, "h": h, "cell_size": cell_size,
                "indices": indices, "palette_mode": cls.PALETTE_EXPLICIT,
                "palette": palette, "bitcount": bitcount,
                "mask": None, "neg": 0, "pos": 0, "max_bitcount": bitcount,
            }, values
        negative_max, positive_max = cls._palette_bounds(small)
        mask = cls._mask_from_values(small, config.mask_size, negative_max, positive_max, config.positive_bias)
        palette = cls.palette_generator(mask, max_bitcount, negative_max, positive_max, config.positive_bias)
        indices = cls.quantize_signed(np.clip(small, -negative_max, positive_max), palette)
        values = palette[indices]
        bitcount = cls.resolve_palette_bitcount(mask, max_bitcount, negative_max, positive_max, config.positive_bias)
        return {
            "channel": channel, "x": x, "y": y, "w": w, "h": h, "cell_size": cell_size,
            "indices": indices, "palette_mode": cls.PALETTE_GENERATED,
            "palette": None, "bitcount": bitcount,
            "mask": mask, "neg": negative_max, "pos": positive_max, "max_bitcount": max_bitcount,
        }, values

    @classmethod
    def _top_anchors(cls, visible_error_channel, top_k, block_size, channel):
        h, w = visible_error_channel.shape
        block_size = max(1, int(block_size))
        if block_size == 1:
            flat = visible_error_channel.reshape(-1)
            k = min(int(top_k), flat.size)
            idx = np.argpartition(flat, -k)[-k:]
            idx = idx[np.argsort(flat[idx])[::-1]]
            return [(channel, int(i) // w, int(i) % w) for i in idx]
        anchors = []
        e = visible_error_channel
        ii = np.pad(e.cumsum(axis=0).cumsum(axis=1), ((1, 0), (1, 0)))
        for y0 in range(0, h, block_size):
            y1 = min(h, y0 + block_size)
            for x0 in range(0, w, block_size):
                x1 = min(w, x0 + block_size)
                s = ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0]
                anchors.append((float(s / ((y1 - y0) * (x1 - x0))), channel, (y0 + y1 - 1) // 2, (x0 + x1 - 1) // 2))
        if not anchors:
            return []
        k = min(int(top_k), len(anchors))
        idx = np.argpartition(np.array([a[0] for a in anchors]), -k)[-k:]
        selected = [anchors[i] for i in idx]
        selected.sort(key=lambda a: a[0], reverse=True)
        return [(c, y, x) for _, c, y, x in selected]

    @classmethod
    def _sample_box(cls, rng, anchor, image_w, image_h, config):
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

    @classmethod
    def _base_cell_size(cls, residual_patch, config):
        mean_abs = float(np.mean(np.abs(residual_patch)))
        if mean_abs <= 0:
            return int(config.max_cell_size)
        gx = float(np.mean(np.abs(np.diff(residual_patch, axis=1)))) if residual_patch.shape[1] > 1 else 0.0
        gy = float(np.mean(np.abs(np.diff(residual_patch, axis=0)))) if residual_patch.shape[0] > 1 else 0.0
        ratio = (gx + gy) / (mean_abs + 1.0)
        if ratio < 0.25:
            return 32
        if ratio < 0.5:
            return 16
        if ratio < 1.0:
            return 8
        return 4

    @classmethod
    def _candidate_cell_sizes(cls, base, config):
        offsets = [0, 1, -1, 2, -2, 3, -3]
        cells = []
        for off in offsets:
            if len(cells) >= max(1, int(config.cell_sizes_per_candidate)):
                break
            cell = int(round(base * (2 ** off)))
            cell = max(int(config.min_cell_size), min(int(config.max_cell_size), cell))
            if cell not in cells:
                cells.append(cell)
        return cells

    @classmethod
    def _patch_bitcounts(cls, config):
        if str(config.patch_bitcount_mode).lower() != "dynamic":
            return [int(config.patch_palette_bitcount)]
        lo = max(1, min(9, int(config.dynamic_patch_bitcount_min)))
        hi = max(lo, min(9, int(config.dynamic_patch_bitcount_max)))
        return list(range(lo, hi + 1))

    @classmethod
    def _auto_init_candidates(cls, residual, w, h, config):
        mean_abs = float(np.mean(np.abs(residual))) + 1.0
        gx = float(np.mean(np.abs(np.diff(residual, axis=1)))) if residual.shape[1] > 1 else 0.0
        gy = float(np.mean(np.abs(np.diff(residual, axis=0)))) if residual.shape[0] > 1 else 0.0
        freq = (gx + gy) / mean_abs
        std = float(np.std(residual))
        if freq >= 1.0:
            cell0 = 4
        elif freq >= 0.5:
            cell0 = 8
        elif freq >= 0.25:
            cell0 = 12
        elif freq >= 0.12:
            cell0 = 16
        else:
            cell0 = 24
        if std < 6:
            bits0 = 3
        elif std < 12:
            bits0 = 4
        elif std < 24:
            bits0 = 5
        else:
            bits0 = 6
        lo_c, hi_c = max(1, int(config.min_cell_size)), min(int(config.max_cell_size), max(w, h))
        max_b = int(config.downsample_palette_bitcount)
        clampc = lambda v: max(lo_c, min(hi_c, int(v)))
        clampb = lambda v: max(1, min(max_b, int(v)))
        raw = [(cell0, bits0), (cell0, bits0 - 1), (cell0, bits0 + 1),
               (cell0 // 2, bits0), (cell0 * 2, bits0), (cell0 // 2, bits0 - 1),
               (cell0 * 2, bits0 + 1), (cell0 // 4, bits0), (cell0 * 4, bits0),
               (cell0 // 2, bits0 + 1), (cell0 * 2, bits0 - 1), (cell0, bits0 + 2),
               (cell0, bits0 - 2), (cell0 // 4, bits0 + 1), (cell0 * 4, bits0 - 1)]
        out = []
        for cell, bits in raw:
            pair = (clampc(cell), clampb(bits))
            if pair not in out:
                out.append(pair)
        return out

    @classmethod
    def _select_init(cls, c, target, canvas, w, h, config, channel_bits):
        base_layer = canvas[:, :, c]
        residual = target[:, :, c] - base_layer
        before = target[:, :, c] - np.clip(base_layer, 0, 255)
        before_sse = float(np.sum(before.astype(np.int64) ** 2))
        cands = cls._auto_init_candidates(residual, w, h, config)[:max(1, int(config.init_search_depth))]
        reductions, bit_costs, built = [], [], []
        for cell, bits in cands:
            patch, values = cls._make_patch(c, 0, 0, w, h, cell, residual, config, bits, cls.PALETTE_GENERATED, 0)
            delta = cls.signed_resample(values, h, w).astype(np.int32)
            after = target[:, :, c] - np.clip(base_layer + delta, 0, 255)
            reductions.append(before_sse - float(np.sum(after.astype(np.int64) ** 2)))
            bit_costs.append(cls._patch_bits_for(patch, channel_bits))
            built.append((patch, values, cell, bits))
        q = float(config.q_init)
        scores = q * cls._norm(reductions) - (1.0 - q) * cls._norm(bit_costs)
        return built[int(np.argmax(scores))]

    @classmethod
    def _debug_line(cls, kind, **items):
        return kind + " " + " ".join(f"{k}={v}" for k, v in items.items())

    @classmethod
    def _select_patch(cls, target, canvas, config, rng, channel_bits, step, canvas_patches, debug_lines, timings, current_channel):
        t = time.perf_counter()
        visible_canvas_channel = np.clip(canvas[:, :, current_channel], 0, 255).astype(np.int32)
        visible_error = (target[:, :, current_channel] - visible_canvas_channel).astype(np.int64)
        abs_error = np.abs(visible_error)
        integral_signed = cls._integral(visible_error)
        integral_abs = cls._integral(abs_error)
        cls._add_time(timings, "visible_error", time.perf_counter() - t)

        q = cls._interp(config.q_start, config.q_end, step, config.patch_count)
        search_q = cls._interp(config.search_q_start, config.search_q_end, step, config.patch_count)
        threshold = cls._palette_threshold(config, step)

        t = time.perf_counter()
        anchors = cls._top_anchors(abs_error.astype(np.float32), config.top_k, config.anchor_block_size, current_channel)
        cls._add_time(timings, "anchors", time.perf_counter() - t)
        if not anchors:
            return None, None

        h, w, _ = target.shape
        box_sums, box_areas, box_specs = [], [], []
        t = time.perf_counter()
        for i in range(max(1, int(config.search_depth))):
            c, x, y, bw, bh, ax, ay = cls._sample_box(rng, anchors[i % len(anchors)], w, h, config)
            box_sum = integral_abs[y + bh, x + bw] - integral_abs[y, x + bw] - integral_abs[y + bh, x] + integral_abs[y, x]
            if box_sum <= 0:
                continue
            box_sums.append(float(box_sum))
            box_areas.append(float(bw * bh))
            box_specs.append((c, x, y, bw, bh))
        cls._add_time(timings, "search_prescore", time.perf_counter() - t)
        if not box_specs:
            return None, None
        pre_scores = search_q * cls._norm(box_sums) - (1.0 - search_q) * cls._norm(box_areas)
        keep = np.argsort(pre_scores)[::-1][:max(1, int(config.proposal_depth))]
        boxes = [box_specs[i] for i in keep]

        header_bits = cls._patch_header_bits(channel_bits, config.mask_size)
        mid_bitcount = int(config.patch_palette_bitcount)

        t = time.perf_counter()
        mid_bounds, mid_bits, mid_specs = [], [], []
        for (c, x, y, bw, bh) in boxes:
            hidden_residual = target[y:y + bh, x:x + bw, c] - canvas[y:y + bh, x:x + bw, c]
            base_cell = cls._base_cell_size(hidden_residual, config)
            for cell_size in cls._candidate_cell_sizes(base_cell, config):
                cell_size = max(1, min(cell_size, bw, bh))
                bound = cls._box_cell_bound(integral_signed, x, y, bw, bh, cell_size)
                if bound <= 0:
                    continue
                grid_cells = cls._ceil_div(bw, cell_size) * cls._ceil_div(bh, cell_size)
                mid_bounds.append(bound)
                mid_bits.append(header_bits + grid_cells * mid_bitcount)
                mid_specs.append((c, x, y, bw, bh, cell_size))
        cls._add_time(timings, "mid_score", time.perf_counter() - t)
        if not mid_specs:
            return None, None
        mid_scores = q * cls._norm(mid_bounds) - (1.0 - q) * cls._norm(mid_bits)
        keep = np.argsort(mid_scores)[::-1][:max(1, int(config.exact_depth))]
        mid_specs = [mid_specs[i] for i in keep]

        bitcounts = cls._patch_bitcounts(config)
        reductions, bit_costs, built = [], [], []
        t = time.perf_counter()
        for proposal_i, (c, x, y, bw, bh, cell_size) in enumerate(mid_specs):
            hidden_residual = target[y:y + bh, x:x + bw, c] - canvas[y:y + bh, x:x + bw, c]
            before = target[y:y + bh, x:x + bw, c] - np.clip(canvas[y:y + bh, x:x + bw, c], 0, 255)
            before_sse = float(np.sum(before.astype(np.int64) ** 2))
            for bitcount in bitcounts:
                for palette_mode in cls._palette_mode_options(config, bitcount):
                    patch, values = cls._make_patch(c, x, y, bw, bh, cell_size, hidden_residual, config, bitcount, palette_mode, threshold)
                    delta = cls.signed_resample(values, bh, bw).astype(np.int32)
                    after = target[y:y + bh, x:x + bw, c] - np.clip(canvas[y:y + bh, x:x + bw, c] + delta, 0, 255)
                    reduction = before_sse - float(np.sum(after.astype(np.int64) ** 2))
                    if reduction <= 0:
                        continue
                    reductions.append(reduction)
                    bit_costs.append(cls._patch_bits_for(patch, channel_bits))
                    built.append((patch, values))
                    if config.debug_mode:
                        debug_lines.append(cls._debug_line(
                            "CANDIDATE", patch_step=step, canvas_patches=canvas_patches, proposal=proposal_i,
                            channel=c, x=x, y=y, w=bw, h=bh, cell_size=cell_size, bitcount=bitcount,
                            palette_mode=palette_mode, reduction=f"{reduction:.4f}"))
        cls._add_time(timings, "fill_score", time.perf_counter() - t)
        if not built:
            return None, None

        scores = q * cls._norm(reductions) - (1.0 - q) * cls._norm(bit_costs)
        best_i = int(np.argmax(scores))
        best_patch, best_values = built[best_i]
        if config.debug_mode:
            debug_lines.append(cls._debug_line(
                "SELECTED", patch_step=step, canvas_patches=canvas_patches, channel=best_patch["channel"],
                x=best_patch["x"], y=best_patch["y"], w=best_patch["w"], h=best_patch["h"],
                cell_size=best_patch["cell_size"], bitcount=best_patch["bitcount"],
                palette_mode=best_patch["palette_mode"], score=f"{float(scores[best_i]):.6f}"))
        return best_patch, best_values

    @classmethod
    def _write_header(cls, bw, w, h, original_w, original_h, downsampled, color_id, channels, channel_bits, positive_bias, has_alpha, patch_count, base_values):
        bw.write(int(downsampled), 1)
        if downsampled:
            bw.write(original_w, 16)
            bw.write(original_h, 16)
        bw.write(w, 16)
        bw.write(h, 16)
        bw.write(color_id, 2)
        bw.write(channels, 8)
        bw.write(channel_bits, 4)
        bw.write(int(positive_bias), 1)
        bw.write(int(has_alpha), 1)
        bw.write(patch_count, 32)
        for base in base_values:
            bw.write(base, 8)

    @classmethod
    def _read_header(cls, br):
        downsampled = bool(br.read(1))
        original_w = br.read(16) if downsampled else None
        original_h = br.read(16) if downsampled else None
        w = br.read(16)
        h = br.read(16)
        color_id = br.read(2)
        channels = br.read(8)
        channel_bits = br.read(4)
        positive_bias = bool(br.read(1))
        has_alpha = bool(br.read(1))
        patch_count = br.read(32)
        color_space = cls.COLOR_SPACE_NAMES[color_id]
        base_values = [br.read(8) for _ in range(channels)]
        return downsampled, original_w, original_h, w, h, color_space, channels, channel_bits, positive_bias, has_alpha, patch_count, base_values

    @classmethod
    def compress(cls, image, config=None, **kwargs):
        if config is None:
            config = PBC3Config(**kwargs)
        elif kwargs:
            config = PBC3Config(**{**config.__dict__, **kwargs})

        t0 = time.perf_counter()
        timings = {}
        debug_lines = []

        t = time.perf_counter()
        src = cls._to_image(image)
        has_alpha = cls._has_alpha(src)
        if has_alpha:
            rgba = src.convert("RGBA")
            color_img = rgba.convert("RGB").convert(config.color_space)
            alpha_img = rgba.getchannel("A")
            orig_compare = rgba
        else:
            color_img = src.convert(config.color_space)
            alpha_img = None
            orig_compare = src.convert("RGB")
        original_w, original_h = color_img.size
        rate = cls._auto_downsample_rate(color_img.size, config.downsample_rate, config.auto_downsample_max_pixels)
        color_ds = cls._downsample_image(color_img, rate)
        downsampled = color_ds.size != color_img.size
        arr = np.asarray(color_ds, dtype=np.uint8)
        if has_alpha:
            alpha_ds = alpha_img.resize(color_ds.size, cls.RESAMPLE_FILTER, reducing_gap=cls.RESAMPLE_REDUCING_GAP) if downsampled else alpha_img
            arr = np.dstack([arr, np.asarray(alpha_ds, dtype=np.uint8)])
        target = arr.astype(np.int32)
        h, w, channels = arr.shape
        cls._add_time(timings, "setup_downsample", time.perf_counter() - t)

        if w > 65535 or h > 65535 or original_w > 65535 or original_h > 65535:
            raise ValueError("this prototype stores dimensions as uint16")
        if config.mask_size < 1 or config.mask_size > 1023:
            raise ValueError("mask_size must be in 1..1023")
        if config.auto_downsample_max_pixels < 1:
            raise ValueError("auto_downsample_max_pixels must be >= 1")
        if not (1 <= config.downsample_palette_bitcount <= 9 and 1 <= config.patch_palette_bitcount <= 9):
            raise ValueError("palette bitcounts must be in 1..9")
        if str(config.channel_cycle).lower() not in {"off", "sum", "max"}:
            raise ValueError('channel_cycle must be "Off", "Sum", or "Max"')
        if str(config.patch_bitcount_mode).lower() not in {"constant", "dynamic"}:
            raise ValueError('patch_bitcount_mode must be "constant" or "dynamic"')
        if str(config.palette_mode).lower() not in {"generated", "explicit", "auto"}:
            raise ValueError('palette_mode must be "generated", "explicit", or "auto"')

        color_id = cls.COLOR_SPACES[config.color_space]
        channel_bits = max(1, math.ceil(math.log2(channels)))
        base_values = [int(round(float(np.mean(arr[:, :, c])))) for c in range(channels)]
        canvas = np.zeros((h, w, channels), dtype=np.int32)
        for c, base in enumerate(base_values):
            canvas[:, :, c] = base

        patches = []
        t = time.perf_counter()
        for c in range(channels):
            if config.auto_downsample_init:
                patch, values, init_cell, init_bits = cls._select_init(c, target, canvas, w, h, config, channel_bits)
                if config.debug_print:
                    print(f"[auto-init] channel {c}: cell={init_cell}, bitcount={init_bits}")
            else:
                init_cell, init_bits = config.downsample_init_cell_size, config.downsample_palette_bitcount
                residual = target[:, :, c] - canvas[:, :, c]
                patch, values = cls._make_patch(c, 0, 0, w, h, init_cell, residual, config, init_bits, cls.PALETTE_GENERATED, 0)
            cls.apply_grid(canvas[:, :, c], 0, 0, w, h, init_cell, values)
            patches.append(patch)
            if config.debug_mode:
                debug_lines.append(cls._debug_line("INIT", stream_patch=len(patches), channel=c, x=0, y=0, w=w, h=h, cell_size=init_cell, bitcount=init_bits))
        cls._add_time(timings, "init_layer", time.perf_counter() - t)

        channel_scores = [cls._channel_error_score(target, canvas, c, config.channel_cycle) for c in range(channels)]
        quality_target = float(config.quality_target_mae)
        done = [False] * channels
        rng = np.random.default_rng(config.random_seed)
        t_patch_total = time.perf_counter()
        for step in range(1, max(0, int(config.patch_count)) + 1):
            if quality_target > 0:
                for c in range(channels):
                    if not done[c] and cls._channel_mae(target, canvas, c) <= quality_target:
                        done[c] = True
            current_channel = cls._choose_channel(channel_scores, step, channels, config.channel_cycle, done)
            if current_channel is None:
                break
            patch, values = cls._select_patch(target, canvas, config, rng, channel_bits, step, len(patches), debug_lines, timings, current_channel)
            if patch is None:
                break
            t = time.perf_counter()
            c = patch["channel"]
            cls.apply_grid(canvas[:, :, c], patch["x"], patch["y"], patch["w"], patch["h"], patch["cell_size"], values)
            patches.append(patch)
            channel_scores[c] = cls._channel_error_score(target, canvas, c, config.channel_cycle)
            cls._add_time(timings, "apply_selected", time.perf_counter() - t)
            if config.debug_mode:
                debug_lines.append(cls._debug_line("APPLIED", patch_step=step, stream_patch=len(patches), channel=c, channel_score=f"{channel_scores[c]:.4f}", x=patch["x"], y=patch["y"], w=patch["w"], h=patch["h"], cell_size=patch["cell_size"]))
            if config.debug_print:
                print("|", end="", flush=True)
        if config.debug_print:
            print()
        cls._add_time(timings, "patch_loop_total", time.perf_counter() - t_patch_total)

        t = time.perf_counter()
        bw = BitWriter()
        cls._write_header(bw, w, h, original_w, original_h, downsampled, color_id, channels, channel_bits, config.positive_bias, has_alpha, len(patches), base_values)
        for patch in patches:
            cls._write_patch(bw, patch, channel_bits)
        method, body = cls._entropy_pack(bw.finish())
        data = cls.MAGIC + bytes([cls.VERSION, method]) + body
        cls._add_time(timings, "serialize", time.perf_counter() - t)

        t = time.perf_counter()
        out_img = cls._canvas_to_image(canvas, config.color_space, has_alpha)
        if downsampled:
            out_img = out_img.resize((original_w, original_h), cls.RESAMPLE_FILTER, reducing_gap=cls.RESAMPLE_REDUCING_GAP)
        target_arr = np.asarray(orig_compare, dtype=np.float32)
        out_arr = np.asarray(out_img, dtype=np.float32)
        mse = float(np.mean((target_arr - out_arr) ** 2))
        cls._add_time(timings, "finalize_mse", time.perf_counter() - t)

        debug_path = None
        total_seconds = time.perf_counter() - t0
        timings["total"] = total_seconds
        if config.debug_mode:
            ts = time.strftime("%Y%m%d_%H%M%S")
            debug_path = config.debug_path or f"debug_{ts}.txt"
            with open(debug_path, "w", encoding="utf-8") as f:
                f.write(cls._debug_line("CONFIG", **{k: v for k, v in config.__dict__.items() if k not in {"debug_path"}}) + "\n")
                f.write(cls._debug_line("IMAGE", original_w=original_w, original_h=original_h, working_w=w, working_h=h,
                                        original_pixels=original_w * original_h, working_pixels=w * h,
                                        downsample_rate=f"{rate:.6f}", downsampled=int(downsampled), has_alpha=int(has_alpha)) + "\n")
                for k, v in timings.items():
                    f.write(cls._debug_line("TIMER", phase=k, seconds=f"{v:.6f}") + "\n")
                for line in debug_lines:
                    f.write(line + "\n")

        return PBC3Result(out_img, data, config, mse, total_seconds, len(data) * 8, original_w, original_h, w, h, timings, debug_path, channels=channels)

    @classmethod
    def _canvas_to_image(cls, canvas, color_space, has_alpha):
        arr = np.clip(canvas, 0, 255).astype(np.uint8)
        if has_alpha:
            color = Image.fromarray(arr[:, :, :3], color_space).convert("RGB").convert("RGBA")
            color.putalpha(Image.fromarray(arr[:, :, 3], "L"))
            return color
        return Image.fromarray(arr, color_space).convert("RGB")

    @classmethod
    def _decode_to_canvas(cls, data, max_patches=None):
        if isinstance(data, str):
            with open(data, "rb") as f:
                data = f.read()
        version, body = cls._open_body(data)
        br = BitReader(body)
        downsampled, original_w, original_h, w, h, color_space, channels, channel_bits, positive_bias, has_alpha, patch_count, base_values = cls._read_header(br)
        canvas = np.zeros((h, w, channels), dtype=np.int32)
        for c, base in enumerate(base_values):
            canvas[:, :, c] = base
        patches_to_read = patch_count if max_patches is None else min(int(max_patches), patch_count)
        for _ in range(patches_to_read):
            channel, x, y, pw, ph, cell_size, values = cls._read_patch(br, channel_bits, positive_bias)
            cls.apply_grid(canvas[:, :, channel], x, y, pw, ph, cell_size, values)
        return canvas, color_space, downsampled, original_w, original_h, w, h, has_alpha, channels, patch_count

    @classmethod
    def decompress(cls, data, max_patches=None):
        t0 = time.perf_counter()
        if isinstance(data, str):
            with open(data, "rb") as f:
                data = f.read()
        canvas, color_space, downsampled, original_w, original_h, w, h, has_alpha, channels, patch_count = cls._decode_to_canvas(data, max_patches=max_patches)
        img = cls._canvas_to_image(canvas, color_space, has_alpha)
        if downsampled:
            img = img.resize((original_w, original_h), cls.RESAMPLE_FILTER, reducing_gap=cls.RESAMPLE_REDUCING_GAP)
        cfg = PBC3Config(color_space=color_space)
        return PBC3Result(img, data, cfg, None, time.perf_counter() - t0, len(data) * 8, original_w or w, original_h or h, w, h, channels=channels)

    @classmethod
    def encode_file(cls, input_path, output_path, config=None, **kwargs):
        result = cls.compress(Image.open(input_path), config=config, **kwargs)
        with open(output_path, "wb") as f:
            f.write(result.data)
        return result

    @classmethod
    def decode_file(cls, input_path, output_path=None):
        image = cls.decompress(input_path).image
        if output_path is not None:
            image.save(output_path)
        return image


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("usage: python PBC3.py input_image output.pbc3")
    else:
        res = PBC3.encode_file(sys.argv[1], sys.argv[2])
        print(f"MSE: {res.mse:.2f} | Size: {len(res.data) / 1024:.2f} KB | Rate: {res.compression_rate:.2f}x | Time: {res.encode_seconds:.3f}s")