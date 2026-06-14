
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
import re
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle


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
    color_space: str = "YCbCr"
    downsample_rate: float = -1
    auto_downsample_max_pixels: int = 250_000
    downsample_cell_size: int = 12
    downsample_palette_bitcount: int = 6
    patch_palette_bitcount: int = 2
    patch_count_mode: str = "constant"
    dynamic_patch_bitcount_min: int = 2
    dynamic_patch_bitcount_max: int = 3
    mask_size: int = 4
    positive_bias: bool = True
    patch_count: int = 50
    search_depth: int = 400
    proposal_depth: int = 50
    top_k: int = 20
    anchor_block_size: int = 8
    min_patch_size: int = 16
    max_patch_size: int = 400
    min_cell_size: int = 1
    max_cell_size: int = 64
    cell_sizes_per_candidate: int = 3
    patch_slot_cost_start: int = 8000
    patch_slot_cost_end: int = 500
    channel_cycle: str = "Max"
    random_seed: int = 2003
    debug_mode: bool = False
    debug_print: bool = False
    debug_path: str = None
    palette_bitcount: int = None
    palette_max: int = None

    def __post_init__(self):
        if self.palette_bitcount is not None:
            self.downsample_palette_bitcount = self.palette_bitcount
            self.patch_palette_bitcount = self.palette_bitcount
        self.channel_cycle = str(self.channel_cycle)
        self.patch_count_mode = str(self.patch_count_mode)


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

    @property
    def original_bits(self):
        w = self.original_width or self.image.width
        h = self.original_height or self.image.height
        return w * h * 3 * 8

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

    def save(self, path: str) -> None:
        if self.data is None:
            raise ValueError("result has no compressed data to save")
        with open(path, "wb") as f:
            f.write(self.data)

    def verify(self) -> bool:
        if self.data is None:
            return False
        decoded = PBC3.decompress(self.data).image
        return np.array_equal(np.asarray(self.image), np.asarray(decoded))

    def show(self) -> None:
        fig = plt.figure(figsize=(8, 7.4), dpi=130)
        gs = fig.add_gridspec(3, 1, height_ratios=[0.09, 0.16, 1.0], hspace=0.04)
        title_ax = fig.add_subplot(gs[0])
        info_ax = fig.add_subplot(gs[1])
        image_ax = fig.add_subplot(gs[2])
        title_ax.axis("off")
        info_ax.axis("off")
        image_ax.axis("off")
        title_ax.text(0.5, 0.5, "PBC3 Result", ha="center", va="center", fontsize=16, fontweight="bold")
        mse = "N/A" if self.mse is None else f"{self.mse:.2f}"
        seconds = "N/A" if self.encode_seconds is None else f"{self.encode_seconds:.3f}s"
        debug = f"   |   Debug: {os.path.basename(self.debug_path)}" if self.debug_path else ""
        info = (
            f"MSE: {mse}   |   Compressed: {self.compressed_kb:.2f} KB   |   Original RGB: {self.original_kb:.2f} KB\n"
            f"Compression: {self.compression_rate:.2f}x ({self.compressed_percent:.2f}%)   |   Time: {seconds}{debug}"
        )
        info_ax.text(0.5, 0.5, info, ha="center", va="center", color="white", fontsize=10, linespacing=1.35,
                     bbox=dict(boxstyle="round,pad=0.5", facecolor="black", alpha=0.72, edgecolor="none"))
        image_ax.imshow(self.image)
        plt.show()


class PBC3:
    MAGIC = b"PBC3"
    VERSION = 1
    MODE_RAW = 0
    COLOR_SPACES = {"RGB": 0, "YCbCr": 1}
    COLOR_SPACE_NAMES = {0: "RGB", 1: "YCbCr"}
    RESAMPLE_FILTER = Image.Resampling.BICUBIC
    RESAMPLE_REDUCING_GAP = None

    @staticmethod
    def _to_image(image):
        if isinstance(image, Image.Image):
            return image
        if isinstance(image, str):
            return Image.open(image)
        arr = np.asarray(image)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr, "RGB")

    @staticmethod
    def _ceil_div(a, b):
        return (a + b - 1) // b

    @staticmethod
    def _add_time(timings, key, seconds):
        timings[key] = timings.get(key, 0.0) + seconds

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
            return img.copy(), img.size
        original_size = img.size
        w = max(1, int(round(original_size[0] / rate)))
        h = max(1, int(round(original_size[1] / rate)))
        return img.resize((w, h), cls.RESAMPLE_FILTER, reducing_gap=cls.RESAMPLE_REDUCING_GAP), original_size

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
        mask = [0] * mask_size
        mask[0] = 1
        flat = np.rint(values).astype(np.int32).ravel()
        for value in flat:
            value = int(np.clip(value, -negative_max, positive_max))
            idx = cls._mask_index_for_value(value, mask_size, negative_max, positive_max, positive_bias)
            if idx is not None and idx < mask_size:
                mask[idx] = 1
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

    @staticmethod
    def quantize_signed(values, palette):
        vals = np.asarray(values, dtype=np.int16)
        pal = np.asarray(palette, dtype=np.int16)
        dist = np.abs(vals[..., None].astype(np.int32) - pal[None, None, :].astype(np.int32))
        return np.argmin(dist, axis=-1).astype(np.uint16)

    @classmethod
    def signed_resample(cls, values, out_h, out_w):
        values = np.asarray(values, dtype=np.float32)
        if values.shape == (out_h, out_w):
            return np.rint(values).astype(np.int16)
        img = Image.fromarray(values)
        resized = img.resize((int(out_w), int(out_h)), cls.RESAMPLE_FILTER, reducing_gap=cls.RESAMPLE_REDUCING_GAP)
        return np.rint(np.asarray(resized, dtype=np.float32)).astype(np.int16)

    @classmethod
    def signed_resample_cells(cls, values, cell_size):
        h, w = values.shape
        return cls.signed_resample(values, cls._ceil_div(h, cell_size), cls._ceil_div(w, cell_size))

    @classmethod
    def apply_grid(cls, canvas_layer, x, y, w, h, cell_size, values):
        patch = cls.signed_resample(values, h, w).astype(np.int32)
        canvas_layer[y:y + h, x:x + w] += patch

    @classmethod
    def _patch_bits(cls, channel_bits, mask, negative_max, positive_max, max_bitcount, cell_size, w, h, positive_bias):
        bitcount = cls.resolve_palette_bitcount(mask, max_bitcount, negative_max, positive_max, positive_bias)
        grid_bits = cls._ceil_div(w, cell_size) * cls._ceil_div(h, cell_size) * bitcount
        return channel_bits + 64 + 10 + len(mask) + 8 + 8 + 4 + 2 + 16 + grid_bits

    @classmethod
    def _patch_slot_cost(cls, config, step):
        if config.patch_count <= 1:
            return float(config.patch_slot_cost_end)
        progress = (step - 1) / max(1, config.patch_count - 1)
        return float(config.patch_slot_cost_start) * (1 - progress) + float(config.patch_slot_cost_end) * progress

    @classmethod
    def _channel_error_score(cls, target, canvas, channel, mode):
        err = np.abs(target[:, :, channel] - np.clip(canvas[:, :, channel], 0, 255))
        if str(mode).lower() == "max":
            return float(np.max(err))
        return float(np.sum(err))

    @classmethod
    def _choose_channel(cls, scores, step, channels, mode):
        mode = str(mode).lower()
        if mode in {"sum", "max"}:
            return int(np.argmax(scores))
        return (step - 1) % channels

    @classmethod
    def _write_patch(cls, bw, channel, x, y, w, h, mask, negative_max, positive_max, max_bitcount, mode, cell_size, indices, channel_bits, positive_bias):
        bitcount = cls.resolve_palette_bitcount(mask, max_bitcount, negative_max, positive_max, positive_bias)
        bw.write(channel, channel_bits)
        bw.write(x, 16)
        bw.write(y, 16)
        bw.write(w, 16)
        bw.write(h, 16)
        bw.write(len(mask), 10)
        for bit in mask:
            bw.write(bit, 1)
        bw.write(negative_max, 8)
        bw.write(positive_max, 8)
        bw.write(max_bitcount, 4)
        bw.write(mode, 2)
        bw.write(cell_size, 16)
        for value in indices.ravel():
            bw.write(int(value), bitcount)

    @classmethod
    def _read_patch(cls, br, channel_bits, positive_bias=True):
        channel = br.read(channel_bits)
        x = br.read(16)
        y = br.read(16)
        w = br.read(16)
        h = br.read(16)
        mask_size = br.read(10)
        mask = [br.read(1) for _ in range(mask_size)]
        negative_max = br.read(8)
        positive_max = br.read(8)
        max_bitcount = br.read(4)
        bitcount = cls.resolve_palette_bitcount(mask, max_bitcount, negative_max, positive_max, positive_bias)
        mode = br.read(2)
        cell_size = br.read(16)
        gw = cls._ceil_div(w, cell_size)
        gh = cls._ceil_div(h, cell_size)
        indices = np.zeros((gh, gw), dtype=np.uint16)
        for gy in range(gh):
            for gx in range(gw):
                indices[gy, gx] = br.read(bitcount)
        palette = cls.palette_generator(mask, max_bitcount, negative_max, positive_max, positive_bias)
        values = palette[indices]
        return channel, x, y, w, h, cell_size, values, mode

    @classmethod
    def _make_patch(cls, channel, x, y, w, h, cell_size, residual, mask_size, max_bitcount, positive_bias):
        small = cls.signed_resample_cells(residual, cell_size)
        negative_max, positive_max = cls._palette_bounds(small)
        mask = cls._mask_from_values(small, mask_size, negative_max, positive_max, positive_bias)
        palette = cls.palette_generator(mask, max_bitcount, negative_max, positive_max, positive_bias)
        indices = cls.quantize_signed(np.clip(small, -negative_max, positive_max), palette)
        values = palette[indices]
        patch = (channel, x, y, w, h, mask, negative_max, positive_max, max_bitcount, cls.MODE_RAW, cell_size, indices)
        return patch, values

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

    @staticmethod
    def _pre_score(visible_error_patch):
        return float(np.mean(visible_error_patch))

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
        mode = str(config.patch_count_mode).lower()
        if mode != "dynamic":
            return [int(config.patch_palette_bitcount)]

        lo = max(1, min(9, int(config.dynamic_patch_bitcount_min)))
        hi = max(lo, min(9, int(config.dynamic_patch_bitcount_max)))
        return list(range(lo, hi + 1))

    @classmethod
    def _debug_line(cls, kind, **items):
        return kind + " " + " ".join(f"{k}={v}" for k, v in items.items())

    @classmethod
    def _select_patch(cls, target, canvas, config, rng, channel_bits, step, canvas_patches, debug_lines, timings, current_channel, channel_score):
        t = time.perf_counter()
        visible_canvas_channel = np.clip(canvas[:, :, current_channel], 0, 255).astype(np.int32)
        visible_error_channel = np.abs(target[:, :, current_channel] - visible_canvas_channel).astype(np.float32)
        cls._add_time(timings, "visible_error", time.perf_counter() - t)

        slot_cost = cls._patch_slot_cost(config, step)
        t = time.perf_counter()
        anchors = cls._top_anchors(visible_error_channel, config.top_k, config.anchor_block_size, current_channel)
        cls._add_time(timings, "anchors", time.perf_counter() - t)
        if not anchors:
            return None, None

        h, w, _ = target.shape
        boxes = []
        t = time.perf_counter()
        for i in range(max(1, int(config.search_depth))):
            c, x, y, bw, bh, ax, ay = cls._sample_box(rng, anchors[i % len(anchors)], w, h, config)
            score = cls._pre_score(visible_error_channel[y:y + bh, x:x + bw])
            if config.debug_mode:
                debug_lines.append(cls._debug_line(
                    "SEARCH", patch_step=step, canvas_patches=canvas_patches, search=i, channel=c, channel_score=f"{channel_score:.4f}",
                    anchor_x=ax, anchor_y=ay, x=x, y=y, w=bw, h=bh,
                    pre_score=f"{score:.6f}",
                ))
            if score > 0:
                boxes.append((score, (c, x, y, bw, bh)))
        cls._add_time(timings, "search_prescore", time.perf_counter() - t)
        if not boxes:
            return None, None
        boxes.sort(key=lambda item: item[0], reverse=True)
        boxes = boxes[:max(1, int(config.proposal_depth))]

        bitcounts = cls._patch_bitcounts(config)

        best_score = 0.0
        best_patch = None
        best_values = None
        t = time.perf_counter()

        for proposal_i, (pre_score, (c, x, y, bw, bh)) in enumerate(boxes):
            hidden_residual = target[y:y + bh, x:x + bw, c] - canvas[y:y + bh, x:x + bw, c]
            base_cell = cls._base_cell_size(hidden_residual, config)
            before = target[y:y + bh, x:x + bw, c] - np.clip(canvas[y:y + bh, x:x + bw, c], 0, 255)

            for cell_size in cls._candidate_cell_sizes(base_cell, config):
                cell_size = max(1, min(cell_size, bw, bh))

                for bitcount in bitcounts:
                    patch, values = cls._make_patch(
                        c,
                        x,
                        y,
                        bw,
                        bh,
                        cell_size,
                        hidden_residual,
                        config.mask_size,
                        bitcount,
                        config.positive_bias,
                    )

                    delta = cls.signed_resample(values, bh, bw).astype(np.int32)
                    after = target[y:y + bh, x:x + bw, c] - np.clip(canvas[y:y + bh, x:x + bw, c] + delta, 0, 255)

                    reduction = float(
                        np.sum(before.astype(np.int64) ** 2)
                        - np.sum(after.astype(np.int64) ** 2)
                    )

                    bits = cls._patch_bits(
                        channel_bits,
                        patch[5],
                        patch[6],
                        patch[7],
                        patch[8],
                        patch[10],
                        bw,
                        bh,
                        config.positive_bias,
                    )

                    raw_score = reduction / max(1, bits) if reduction > 0 else 0.0
                    score = reduction / max(1.0, bits + slot_cost) if reduction > 0 else 0.0

                    if config.debug_mode:
                        debug_lines.append(cls._debug_line(
                            "CANDIDATE",
                            patch_step=step,
                            canvas_patches=canvas_patches,
                            proposal=proposal_i,
                            channel=c,
                            x=x,
                            y=y,
                            w=bw,
                            h=bh,
                            cell_size=cell_size,
                            mask_size=config.mask_size,
                            bitcount=bitcount,
                            neg_min=-patch[6],
                            pos_max=patch[7],
                            bits=bits,
                            slot_cost=f"{slot_cost:.2f}",
                            reduction=f"{reduction:.4f}",
                            raw_score=f"{raw_score:.8f}",
                            score=f"{score:.8f}",
                            pre_score=f"{pre_score:.6f}",
                        ))

                    if score > best_score:
                        best_score = score
                        best_patch = patch
                        best_values = values
        cls._add_time(timings, "fill_score", time.perf_counter() - t)
        if config.debug_mode and best_patch is not None:
            c, x, y, bw, bh = best_patch[:5]
            debug_lines.append(cls._debug_line(
                "SELECTED", patch_step=step, canvas_patches=canvas_patches, channel=c,
                x=x, y=y, w=bw, h=bh, cell_size=best_patch[10], bitcount=best_patch[8], slot_cost=f"{slot_cost:.2f}", score=f"{best_score:.8f}",
            ))
        return best_patch, best_values

    @classmethod
    def _write_header(cls, bw, w, h, original_w, original_h, downsampled, color_id, channels, channel_bits, positive_bias, patch_count, base_values):
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
        bw.write(patch_count, 32)
        for base in base_values:
            bw.write(base, 8)

    @classmethod
    def _read_header(cls, br, version):
        if version == 0:
            downsampled = False
            original_w = original_h = None
        else:
            downsampled = bool(br.read(1))
            original_w = br.read(16) if downsampled else None
            original_h = br.read(16) if downsampled else None
        w = br.read(16)
        h = br.read(16)
        color_id = br.read(2)
        channels = br.read(8)
        channel_bits = br.read(4)
        positive_bias = bool(br.read(1))
        patch_count = br.read(32)
        color_space = cls.COLOR_SPACE_NAMES[color_id]
        base_values = [br.read(8) for _ in range(channels)]
        return downsampled, original_w, original_h, w, h, color_space, channels, channel_bits, positive_bias, patch_count, base_values

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
        original_img = cls._to_image(image).convert(config.color_space)
        original_w, original_h = original_img.size
        rate = cls._auto_downsample_rate(
            original_img.size,
            config.downsample_rate,
            config.auto_downsample_max_pixels,
        )
        img, original_size = cls._downsample_image(original_img, rate)
        downsampled = img.size != original_img.size
        arr = np.asarray(img, dtype=np.uint8)
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
        if str(config.patch_count_mode).lower() not in {"constant", "dynamic"}:
            raise ValueError('patch_count_mode must be "constant" or "dynamic"')

        color_id = cls.COLOR_SPACES[config.color_space]
        channel_bits = max(1, math.ceil(math.log2(channels)))
        base_values = [int(round(float(np.mean(arr[:, :, c])))) for c in range(channels)]
        canvas = np.zeros((h, w, channels), dtype=np.int32)
        for c, base in enumerate(base_values):
            canvas[:, :, c] = base

        patches = []
        t = time.perf_counter()
        for c in range(channels):
            residual = target[:, :, c] - canvas[:, :, c]
            patch, values = cls._make_patch(c, 0, 0, w, h, config.downsample_cell_size, residual, config.mask_size, config.downsample_palette_bitcount, config.positive_bias)
            cls.apply_grid(canvas[:, :, c], 0, 0, w, h, config.downsample_cell_size, values)
            patches.append(patch)
            if config.debug_mode:
                debug_lines.append(cls._debug_line("INIT", stream_patch=len(patches), channel=c, x=0, y=0, w=w, h=h, cell_size=config.downsample_cell_size))
        cls._add_time(timings, "init_layer", time.perf_counter() - t)

        channel_scores = [cls._channel_error_score(target, canvas, c, config.channel_cycle) for c in range(channels)]
        rng = np.random.default_rng(config.random_seed)
        t_patch_total = time.perf_counter()
        for step in range(1, max(0, int(config.patch_count)) + 1):
            current_channel = cls._choose_channel(channel_scores, step, channels, config.channel_cycle)
            patch, values = cls._select_patch(target, canvas, config, rng, channel_bits, step, len(patches), debug_lines, timings, current_channel, channel_scores[current_channel])
            if patch is None:
                break
            t = time.perf_counter()
            c, x, y, pw, ph = patch[:5]
            cls.apply_grid(canvas[:, :, c], x, y, pw, ph, patch[10], values)
            patches.append(patch)
            channel_scores[c] = cls._channel_error_score(target, canvas, c, config.channel_cycle)
            cls._add_time(timings, "apply_selected", time.perf_counter() - t)
            if config.debug_mode:
                debug_lines.append(cls._debug_line("APPLIED", patch_step=step, stream_patch=len(patches), channel=c, channel_score=f"{channel_scores[c]:.4f}", x=x, y=y, w=pw, h=ph, cell_size=patch[10]))
            if config.debug_print:
                print("|", end="", flush=True)
        if config.debug_print:
            print()
        cls._add_time(timings, "patch_loop_total", time.perf_counter() - t_patch_total)

        t = time.perf_counter()
        bw = BitWriter()
        cls._write_header(bw, w, h, original_w, original_h, downsampled, color_id, channels, channel_bits, config.positive_bias, len(patches), base_values)
        for patch in patches:
            cls._write_patch(bw, *patch, channel_bits=channel_bits, positive_bias=config.positive_bias)
        data = cls.MAGIC + bytes([cls.VERSION]) + bw.finish()
        cls._add_time(timings, "serialize", time.perf_counter() - t)

        t = time.perf_counter()
        out_arr = np.clip(canvas, 0, 255).astype(np.uint8)
        out_img = Image.fromarray(out_arr, config.color_space).convert("RGB")
        if downsampled:
            out_img = out_img.resize((original_w, original_h), cls.RESAMPLE_FILTER, reducing_gap=cls.RESAMPLE_REDUCING_GAP)
        target_rgb = np.asarray(original_img.convert("RGB"), dtype=np.float32)
        out_rgb = np.asarray(out_img, dtype=np.float32)
        mse = float(np.mean((target_rgb - out_rgb) ** 2))
        cls._add_time(timings, "finalize_mse", time.perf_counter() - t)

        debug_path = None
        total_seconds = time.perf_counter() - t0
        timings["total"] = total_seconds
        if config.debug_mode:
            ts = time.strftime("%Y%m%d_%H%M%S")
            debug_path = config.debug_path or f"debug_{ts}.txt"
            with open(debug_path, "w", encoding="utf-8") as f:
                f.write(cls._debug_line("CONFIG", **{k: v for k, v in config.__dict__.items() if k not in {"debug_path"}}) + "\n")
                f.write(cls._debug_line(
                    "IMAGE",
                    original_w=original_w,
                    original_h=original_h,
                    working_w=w,
                    working_h=h,
                    original_pixels=original_w * original_h,
                    working_pixels=w * h,
                    auto_downsample_max_pixels=config.auto_downsample_max_pixels,
                    downsample_rate=f"{rate:.6f}",
                    downsampled=int(downsampled),
                ) + "\n")
                for k, v in timings.items():
                    f.write(cls._debug_line("TIMER", phase=k, seconds=f"{v:.6f}") + "\n")
                for line in debug_lines:
                    f.write(line + "\n")

        return PBC3Result(out_img, data, config, mse, total_seconds, len(data) * 8, original_w, original_h, w, h, timings, debug_path)

    @classmethod
    def _decode_to_canvas(cls, data, max_patches=None):
        if isinstance(data, str):
            with open(data, "rb") as f:
                data = f.read()
        if data[:4] != cls.MAGIC:
            raise ValueError("not a PBC3 file")
        version = data[4]
        if version not in (0, cls.VERSION):
            raise ValueError(f"unsupported PBC3 version {version}")
        br = BitReader(data[5:])
        downsampled, original_w, original_h, w, h, color_space, channels, channel_bits, positive_bias, patch_count, base_values = cls._read_header(br, version)
        canvas = np.zeros((h, w, channels), dtype=np.int32)
        for c, base in enumerate(base_values):
            canvas[:, :, c] = base
        patches_to_read = patch_count if max_patches is None else min(int(max_patches), patch_count)
        for _ in range(patches_to_read):
            channel, x, y, pw, ph, cell_size, values, mode = cls._read_patch(br, channel_bits, positive_bias)
            if mode != cls.MODE_RAW:
                raise ValueError(f"unsupported patch mode {mode}")
            cls.apply_grid(canvas[:, :, channel], x, y, pw, ph, cell_size, values)
        return canvas, color_space, downsampled, original_w, original_h, w, h, patch_count

    @classmethod
    def decompress(cls, data, max_patches=None):
        t0 = time.perf_counter()
        if isinstance(data, str):
            with open(data, "rb") as f:
                data = f.read()
        canvas, color_space, downsampled, original_w, original_h, w, h, patch_count = cls._decode_to_canvas(data, max_patches=max_patches)
        arr = np.clip(canvas, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr, color_space).convert("RGB")
        if downsampled:
            img = img.resize((original_w, original_h), cls.RESAMPLE_FILTER, reducing_gap=cls.RESAMPLE_REDUCING_GAP)
        cfg = PBC3Config(color_space=color_space)
        return PBC3Result(img, data, cfg, None, time.perf_counter() - t0, len(data) * 8, original_w or w, original_h or h, w, h)

    @staticmethod
    def parse_debug_line(line):
        parts = line.strip().split()
        if not parts:
            return None
        out = {"kind": parts[0]}
        for part in parts[1:]:
            if "=" not in part:
                continue
            k, v = part.split("=", 1)
            if re.fullmatch(r"-?\d+", v):
                out[k] = int(v)
            else:
                try:
                    out[k] = float(v)
                except ValueError:
                    out[k] = v
        return out

    @classmethod
    def show_debug_line(cls, line, original_image=None, data=None):
        d = cls.parse_debug_line(line)
        if not d:
            raise ValueError("could not parse debug line")
        if not all(k in d for k in ("x", "y", "w", "h")):
            raise ValueError("debug line has no bounding box")

        images = []
        titles = []
        if original_image is not None:
            img = cls._to_image(original_image).convert("RGB")
            images.append(np.asarray(img))
            titles.append("Original")

        canvas_before = None
        color_space = "RGB"
        if data is not None and "canvas_patches" in d:
            canvas, color_space, downsampled, original_w, original_h, _, _, _ = cls._decode_to_canvas(data, max_patches=int(d["canvas_patches"]))
            canvas_before = np.clip(canvas, 0, 255).astype(np.uint8)
            img = Image.fromarray(canvas_before, color_space).convert("RGB")
            if downsampled:
                img = img.resize((original_w, original_h), cls.RESAMPLE_FILTER, reducing_gap=cls.RESAMPLE_REDUCING_GAP)
            images.append(np.asarray(img))
            titles.append(f"Canvas after {d['canvas_patches']} patches")

        if canvas_before is not None and original_image is not None and d["kind"] == "CANDIDATE":
            work_original = cls._to_image(original_image).convert(color_space)
            if canvas_before.shape[1] != work_original.width or canvas_before.shape[0] != work_original.height:
                work_original = work_original.resize((canvas_before.shape[1], canvas_before.shape[0]), cls.RESAMPLE_FILTER, reducing_gap=cls.RESAMPLE_REDUCING_GAP)
            target = np.asarray(work_original, dtype=np.uint8).astype(np.int32)
            x, y, w, h, c = int(d["x"]), int(d["y"]), int(d["w"]), int(d["h"]), int(d["channel"])
            hidden = target[y:y + h, x:x + w, c] - canvas_before[y:y + h, x:x + w, c].astype(np.int32)
            patch, values = cls._make_patch(c, x, y, w, h, int(d["cell_size"]), hidden, int(d.get("mask_size", 9)), int(d.get("bitcount", 3)), True)
            after = canvas_before.astype(np.int32)
            cls.apply_grid(after[:, :, c], x, y, w, h, int(d["cell_size"]), values)
            images.append(np.clip(after, 0, 255).astype(np.uint8))
            titles.append("Candidate applied")

        if not images:
            raise ValueError("provide original_image and/or data to visualize this line")

        fig, axes = plt.subplots(1, len(images), figsize=(5 * len(images), 5), dpi=130)
        if len(images) == 1:
            axes = [axes]
        for ax, img, title in zip(axes, images, titles):
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(title)
            scale_x = img.shape[1] / (d.get("working_w", img.shape[1]))
            scale_y = img.shape[0] / (d.get("working_h", img.shape[0]))
            x = d["x"] * scale_x
            y = d["y"] * scale_y
            w = d["w"] * scale_x
            h = d["h"] * scale_y
            ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor="red", linewidth=2))
        fig.suptitle(" | ".join(f"{k}: {v}" for k, v in d.items() if k != "kind"), fontsize=9)
        plt.tight_layout()
        plt.show()

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
