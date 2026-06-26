
# ====================================================================================================
#
#           PBC v3.0 - Probabilistic Brush Compression
#           Lossy Image Compression Algorithm by EgeEken (github.com/EgeEken)
#           3.0 Update - 2026-06 - Whole algorithm overhaul
#
# ====================================================================================================
import time
import math
import lzma
import numpy as np
from PIL import Image

from pbc3_types import BitWriter, BitReader, PBC3Config, PBC3Result
from pbc3_kernels import NUMBA_AVAILABLE as _NUMBA, box_cell_bound as _nb_box_cell_bound, base_cell_size as _nb_base_cell_size, anchor_block_scores as _nb_anchor_block_scores
from pbc3_heads import ChannelState, DownsampleInitHead, SearchHead, FillerHead

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
    USE_NUMBA_RESAMPLE = False

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
        return (int(a) + int(b) - 1) // int(b)

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
    def _entropy_pack(cls, body, use_lzma=True):
        if not use_lzma:
            return cls.ENTROPY_STORE, body
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

    @classmethod
    def _resize_canvas(cls, canvas, new_w, new_h):
        h, w, ch = canvas.shape
        if (w, h) == (new_w, new_h):
            return canvas
        out = np.empty((new_h, new_w, ch), dtype=np.int32)
        for c in range(ch):
            layer = Image.fromarray(canvas[:, :, c].astype(np.float32), mode="F")
            layer = layer.resize((new_w, new_h), cls.RESAMPLE_FILTER)
            out[:, :, c] = np.rint(np.asarray(layer, dtype=np.float64)).astype(np.int32)
        return out

    @classmethod
    def _warmup_plan(cls, config, original_size, init_rate):
        ratio = config.warmup_ratio
        if ratio is None or ratio <= 0:
            return None
        warm_max = int(config.warm_downsample_max_pixels)
        warm_rate = 1.0 if warm_max <= 0 else cls._auto_downsample_rate(original_size, -1, warm_max)
        if warm_rate >= init_rate:
            print(f"[warmup] warm target rate {warm_rate:.3f} is not higher-res than initial rate {init_rate:.3f}; ignoring warmup.", flush=True)
            return None
        k = int(round(float(ratio) * int(config.patch_count)))
        if k <= 0 or k >= int(config.patch_count):
            return None
        return warm_rate, k

    @staticmethod
    def _palette_bounds(values):
        min_value = int(np.min(values)); max_value = int(np.max(values))
        return min(255, max(0, -min_value)), min(255, max(0, max_value))

    @classmethod
    def _range_counts(cls, mask_size, negative_max=255, positive_max=255, positive_bias=True):
        side_bits = max(0, int(mask_size) - 1)
        negative_max = max(0, int(negative_max)); positive_max = max(0, int(positive_max))
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
            neg_count = 1; pos_count = max(1, pos_count - 1)
        return pos_count, neg_count

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
        mask = [0] * int(mask_size); mask[0] = 1
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
                    start, end = r; count += end - start + 1
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
            palette.append(0); active_ranges = [r for r in active_ranges if r != (0, 0)]
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
        vals = np.asarray(values, dtype=np.int16); pal = np.asarray(palette, dtype=np.int16)
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
        canvas_layer[y:y + h, x:x + w] += cls.signed_resample(values, h, w).astype(np.int32)

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
        if _NUMBA:
            return _nb_box_cell_bound(np.ascontiguousarray(integral, dtype=np.int64), int(x), int(y), int(bw), int(bh), int(cell_size))
        xe = cls._cell_edges(x, bw, cell_size); ye = cls._cell_edges(y, bh, cell_size)
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
        grid_bits = cls._ceil_div(patch["w"], patch["cell_size"]) * cls._ceil_div(patch["h"], patch["cell_size"]) * patch["bitcount"]
        return channel_bits + 64 + 16 + 1 + 10 + len(patch["mask"]) + 8 + 8 + 4 + grid_bits

    @classmethod
    def _patch_header_bits(cls, channel_bits, mask_size):
        return channel_bits + 64 + 10 + mask_size + 8 + 8 + 4 + 16 + 1

    @classmethod
    def _patch_bitcounts(cls, config):
        return [int(config.patch_palette_bitcount)]

    @classmethod
    def _palette_mode_options(cls, config, bitcount):
        return [cls.PALETTE_GENERATED]

    @classmethod
    def _palette_threshold(cls, config, step):
        return 0

    @classmethod
    def _channel_error_score(cls, target, canvas, channel, mode):
        return ChannelState(channel, target, canvas).score

    @classmethod
    def _choose_channel(cls, scores, step, channels, mode):
        return (step - 1) % channels if str(mode).lower() == "mod" else int(max(range(channels), key=lambda c: scores[c]))

    @classmethod
    def _write_patch(cls, bw, patch, channel_bits):
        bw.write(patch["channel"], channel_bits)
        bw.write(patch["x"], 16); bw.write(patch["y"], 16); bw.write(patch["w"], 16); bw.write(patch["h"], 16)
        bw.write(cls.PALETTE_GENERATED, 1)
        mask = patch["mask"]
        bw.write(len(mask), 10)
        for bit in mask:
            bw.write(bit, 1)
        bw.write(patch["neg"], 8); bw.write(patch["pos"], 8); bw.write(patch["max_bitcount"], 4)
        bw.write(patch["cell_size"], 16)
        cls._write_grid(bw, patch["indices"].ravel().astype(np.int64), patch["bitcount"])

    @classmethod
    def _read_patch(cls, br, channel_bits, positive_bias=True):
        channel = br.read(channel_bits)
        x = br.read(16); y = br.read(16); w = br.read(16); h = br.read(16)
        pm = br.read(1)
        if pm != cls.PALETTE_GENERATED:
            raise ValueError("explicit palette patches were removed in PBC3 3.0 release cleanup")
        mask_size = br.read(10)
        mask = [br.read(1) for _ in range(mask_size)]
        negative_max = br.read(8); positive_max = br.read(8); max_bitcount = br.read(4)
        bitcount = cls.resolve_palette_bitcount(mask, max_bitcount, negative_max, positive_max, positive_bias)
        palette = cls.palette_generator(mask, max_bitcount, negative_max, positive_max, positive_bias)
        cell_size = br.read(16)
        gw = cls._ceil_div(w, cell_size); gh = cls._ceil_div(h, cell_size)
        indices = cls._read_grid(br, gh * gw, bitcount).reshape(gh, gw)
        return channel, x, y, w, h, cell_size, palette[indices], bitcount

    @classmethod
    def _make_patch(cls, channel, x, y, w, h, cell_size, residual, config, max_bitcount, palette_mode=0, threshold=0):
        small = cls.signed_resample_cells(residual, cell_size)
        negative_max, positive_max = cls._palette_bounds(small)
        mask = cls._mask_from_values(small, config.mask_size, negative_max, positive_max, config.positive_bias)
        palette = cls.palette_generator(mask, max_bitcount, negative_max, positive_max, config.positive_bias)
        indices = cls.quantize_signed(np.clip(small, -negative_max, positive_max), palette)
        values = palette[indices]
        bitcount = cls.resolve_palette_bitcount(mask, max_bitcount, negative_max, positive_max, config.positive_bias)
        return {"channel": channel, "x": x, "y": y, "w": w, "h": h, "cell_size": cell_size, "indices": indices,
                "palette_mode": cls.PALETTE_GENERATED, "palette": None, "bitcount": bitcount, "mask": mask,
                "neg": negative_max, "pos": positive_max, "max_bitcount": max_bitcount}, values

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
        if _NUMBA:
            scores, ys, xs = _nb_anchor_block_scores(np.ascontiguousarray(visible_error_channel, dtype=np.float64), block_size)
            if scores.size == 0:
                return []
            k = min(int(top_k), scores.size)
            idx = np.argpartition(scores, -k)[-k:]
            order = idx[np.argsort(scores[idx])[::-1]]
            return [(channel, int(ys[i]), int(xs[i])) for i in order]
        anchors = []
        ii = np.pad(visible_error_channel.cumsum(axis=0).cumsum(axis=1), ((1, 0), (1, 0)))
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

    @staticmethod
    def _select_top_indices(scores, keep):
        scores = np.asarray(scores); n = scores.size; keep = max(1, int(keep))
        if keep >= n:
            return np.arange(n)
        cut = n - keep
        if cut < keep:
            drop = np.argpartition(scores, cut)[:cut]
            mask = np.ones(n, dtype=bool); mask[drop] = False
            return np.nonzero(mask)[0]
        return np.argpartition(scores, -keep)[-keep:]

    @classmethod
    def _sample_box(cls, rng, anchor, image_w, image_h, config):
        c, ay, ax = anchor
        min_size = max(1, int(config.min_patch_size))
        max_w = max(min_size, min(int(config.max_patch_size), image_w)); max_h = max(min_size, min(int(config.max_patch_size), image_h))
        w = int(round(2 ** rng.uniform(math.log2(min_size), math.log2(max_w))))
        h = int(round(2 ** rng.uniform(math.log2(min_size), math.log2(max_h))))
        w = min(max(1, w), image_w); h = min(max(1, h), image_h)
        x_min = max(0, ax - w + 1); x_max = min(ax, image_w - w)
        y_min = max(0, ay - h + 1); y_max = min(ay, image_h - h)
        x = int(rng.integers(x_min, x_max + 1)) if x_min <= x_max else max(0, min(ax, image_w - w))
        y = int(rng.integers(y_min, y_max + 1)) if y_min <= y_max else max(0, min(ay, image_h - h))
        return c, x, y, w, h, ax, ay

    @classmethod
    def _base_cell_size(cls, residual_patch, config):
        if _NUMBA and residual_patch.size:
            return int(_nb_base_cell_size(np.ascontiguousarray(residual_patch, dtype=np.float64), int(config.max_cell_size)))
        mean_abs = float(np.mean(np.abs(residual_patch))) if residual_patch.size else 0.0
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
    def _select_init(cls, c, target, canvas, w, h, config, channel_bits):
        return DownsampleInitHead(cls).select(c, target, canvas, w, h, config, channel_bits)

    @classmethod
    def _debug_line(cls, kind, **items):
        return kind + " " + " ".join(f"{k}={v}" for k, v in items.items())

    @classmethod
    def search_candidates(cls, target, canvas, config, rng, step, current_channel, depth, search_q, timings):
        return SearchHead(cls).search(target, canvas, config, rng, current_channel, depth, search_q, timings)

    @classmethod
    def _select_patch(cls, target, canvas, config, rng, channel_bits, step, canvas_patches, debug_lines, timings, current_channel):
        boxes = SearchHead(cls).propose(target, canvas, config, rng, step, current_channel, timings)
        return FillerHead(cls).select_heuristic(target, canvas, config, channel_bits, step, boxes, canvas_patches, debug_lines, timings)

    @classmethod
    def _write_header(cls, bw, w, h, original_w, original_h, downsampled, color_id, channels, channel_bits, positive_bias, has_alpha, patch_count, base_values, warmup=None):
        bw.write(int(downsampled), 1)
        if downsampled:
            bw.write(original_w, 16); bw.write(original_h, 16)
        bw.write(w, 16); bw.write(h, 16); bw.write(color_id, 2); bw.write(channels, 8); bw.write(channel_bits, 4)
        bw.write(int(positive_bias), 1); bw.write(int(has_alpha), 1); bw.write(patch_count, 32)
        for base in base_values:
            bw.write(base, 8)
        bw.write(int(warmup is not None), 1)
        if warmup is not None:
            wm_w, wm_h, wm_split = warmup
            bw.write(wm_w, 16); bw.write(wm_h, 16); bw.write(wm_split, 32)

    @classmethod
    def _read_header(cls, br):
        downsampled = bool(br.read(1))
        original_w = br.read(16) if downsampled else None
        original_h = br.read(16) if downsampled else None
        w = br.read(16); h = br.read(16); color_id = br.read(2); channels = br.read(8); channel_bits = br.read(4)
        positive_bias = bool(br.read(1)); has_alpha = bool(br.read(1)); patch_count = br.read(32)
        base_values = [br.read(8) for _ in range(channels)]
        warmup_on = bool(br.read(1)); warm_w = warm_h = warmup_split = None
        if warmup_on:
            warm_w = br.read(16); warm_h = br.read(16); warmup_split = br.read(32)
        return downsampled, original_w, original_h, w, h, cls.COLOR_SPACE_NAMES[color_id], channels, channel_bits, positive_bias, has_alpha, patch_count, base_values, warmup_on, warm_w, warm_h, warmup_split

    @classmethod
    def prepare(cls, image, config=None, **kwargs):
        if config is None:
            config = PBC3Config(**kwargs)
        elif kwargs:
            config = PBC3Config(**{**config.__dict__, **kwargs})
        src = cls._to_image(image)
        has_alpha = cls._has_alpha(src)
        if has_alpha:
            rgba = src.convert("RGBA"); color_img = rgba.convert("RGB").convert(config.color_space); alpha_img = rgba.getchannel("A"); orig_compare = rgba
        else:
            color_img = src.convert(config.color_space); alpha_img = None; orig_compare = src.convert("RGB")
        original_w, original_h = color_img.size
        rate = cls._auto_downsample_rate(color_img.size, config.downsample_rate, config.auto_downsample_max_pixels)
        color_ds = cls._downsample_image(color_img, rate)
        downsampled = color_ds.size != color_img.size
        arr = np.asarray(color_ds, dtype=np.uint8)
        if has_alpha:
            alpha_ds = alpha_img.resize(color_ds.size, cls.RESAMPLE_FILTER, reducing_gap=cls.RESAMPLE_REDUCING_GAP) if downsampled else alpha_img
            arr = np.dstack([arr, np.asarray(alpha_ds, dtype=np.uint8)])
        warm_plan = cls._warmup_plan(config, color_img.size, rate)
        warm_w = warm_h = warmup_patches = warm_target = None
        if warm_plan is not None:
            warm_rate, warmup_patches = warm_plan
            warm_color_ds = cls._downsample_image(color_img, warm_rate)
            warm_w, warm_h = warm_color_ds.size
            warm_arr = np.asarray(warm_color_ds, dtype=np.uint8)
            if has_alpha:
                warm_alpha = alpha_img.resize(warm_color_ds.size, cls.RESAMPLE_FILTER, reducing_gap=cls.RESAMPLE_REDUCING_GAP)
                warm_arr = np.dstack([warm_arr, np.asarray(warm_alpha, dtype=np.uint8)])
            warm_target = warm_arr.astype(np.int32)
        h, w, channels = arr.shape
        return {"arr": arr, "target": arr.astype(np.int32), "h": h, "w": w, "channels": channels, "original_w": original_w, "original_h": original_h, "downsampled": downsampled, "has_alpha": has_alpha, "orig_compare": orig_compare, "rate": rate, "color_id": cls.COLOR_SPACES[config.color_space], "color_space": config.color_space, "warm_w": warm_w, "warm_h": warm_h, "warmup_patches": warmup_patches, "warm_target": warm_target}

    @classmethod
    def compress(cls, image, config=None, *, reuse=None, **kwargs):
        result = None
        for ev in cls.compress_stream(image, config, reuse=reuse, frame_every=0, **kwargs):
            if ev["event"] == "done":
                result = ev["result"]
        return result

    @classmethod
    def compress_stream(cls, image, config=None, *, reuse=None, frame_every=25, **kwargs):
        if config is None:
            config = PBC3Config(**kwargs)
        elif kwargs:
            config = PBC3Config(**{**config.__dict__, **kwargs})
        t0 = time.perf_counter(); timings = {}; debug_lines = []
        t = time.perf_counter(); prep = reuse if reuse is not None else cls.prepare(image, config)
        arr, target = prep["arr"], prep["target"]
        h, w, channels = prep["h"], prep["w"], prep["channels"]
        original_w, original_h = prep["original_w"], prep["original_h"]
        downsampled, has_alpha = prep["downsampled"], prep["has_alpha"]
        orig_compare, color_id, rate = prep["orig_compare"], prep["color_id"], prep["rate"]
        warm_w, warm_h = prep.get("warm_w"), prep.get("warm_h")
        warmup_patches, warm_target = prep.get("warmup_patches"), prep.get("warm_target")
        warmup_on = warmup_patches is not None; did_warmup = False; warmup_split = None
        cls._add_time(timings, "setup_downsample", time.perf_counter() - t)
        if w > 65535 or h > 65535 or original_w > 65535 or original_h > 65535:
            raise ValueError("this prototype stores dimensions as uint16")
        if config.mask_size < 1 or config.mask_size > 1023:
            raise ValueError("mask_size must be in 1..1023")
        if config.auto_downsample_max_pixels < 1:
            raise ValueError("auto_downsample_max_pixels must be >= 1")
        if not (1 <= config.downsample_palette_bitcount <= 9 and 1 <= config.patch_palette_bitcount <= 9):
            raise ValueError("palette bitcounts must be in 1..9")
        if str(config.channel_cycle).lower() not in {"sum", "mod"}:
            raise ValueError('channel_cycle must be "Sum" or "Mod"')
        channel_bits = max(1, math.ceil(math.log2(channels)))
        base_values = [int(round(float(np.mean(arr[:, :, c])))) for c in range(channels)]
        canvas = np.zeros((h, w, channels), dtype=np.int32)
        for c, base in enumerate(base_values):
            canvas[:, :, c] = base
        if frame_every:
            yield {"event": "frame", "step": 0, "total": int(config.patch_count), "image": cls._canvas_to_image(canvas, config.color_space, has_alpha)}
        patches = []
        t = time.perf_counter(); init_head = DownsampleInitHead(cls)
        for c in range(channels):
            patch, values, init_cell, init_bits = init_head.select(c, target, canvas, w, h, config, channel_bits)
            if config.debug_print:
                print(f"[auto-init] channel {c}: cell={init_cell}, bitcount={init_bits}")
            cls.apply_grid(canvas[:, :, c], 0, 0, w, h, init_cell, values)
            patches.append(patch)
            if config.debug_mode:
                debug_lines.append(cls._debug_line("INIT", stream_patch=len(patches), channel=c, x=0, y=0, w=w, h=h, cell_size=init_cell, bitcount=init_bits))
        cls._add_time(timings, "init_layer", time.perf_counter() - t)
        if frame_every:
            yield {"event": "frame", "step": 0, "total": int(config.patch_count), "image": cls._canvas_to_image(canvas, config.color_space, has_alpha)}
        channel_scores = [
            float(np.sum(np.abs(target[:, :, c] - np.clip(canvas[:, :, c], 0, 255))))
            for c in range(channels)
        ]
        quality_target = float(config.quality_target_mae)
        filler = FillerHead(cls, config, channel_bits, (h, w, channels), patches, (original_w, original_h))
        search = SearchHead(cls)
        rng = np.random.default_rng(config.random_seed)
        applied = 0; t_patch_total = time.perf_counter()
        for step in range(1, max(0, int(config.patch_count)) + 1):
            current_channel = cls._choose_channel(channel_scores, step, channels, config.channel_cycle)
            boxes = None if filler.learned is not None else search.propose(target, canvas, config, rng, step, current_channel, timings)
            patch, values = filler.select(target, canvas, config, rng, channel_bits, step, current_channel, boxes, len(patches), debug_lines, timings)
            # TODO: CHECK THIS AGAIN AFTER BENCHMARK CHECK
            #if patch is None and filler.learned is not None:
            #    boxes = search.propose(target, canvas, config, rng, step, current_channel, timings)
            #    patch, values = filler.select_heuristic(target, canvas, config, channel_bits, step, boxes, len(patches), debug_lines, timings)
            if patch is None:
                break
            t = time.perf_counter(); c = patch["channel"]
            cls.apply_grid(canvas[:, :, c], patch["x"], patch["y"], patch["w"], patch["h"], patch["cell_size"], values)
            patches.append(patch)
            channel_scores[c] = float(np.sum(np.abs(target[:, :, c] - np.clip(canvas[:, :, c], 0, 255))))
            cls._add_time(timings, "apply_selected", time.perf_counter() - t)
            applied += 1
            if warmup_on and not did_warmup and applied == warmup_patches:
                ts = time.perf_counter(); canvas = cls._resize_canvas(canvas, warm_w, warm_h); target = warm_target
                channel_scores = [
                    float(np.sum(np.abs(target[:, :, c] - np.clip(canvas[:, :, c], 0, 255))))
                    for c in range(channels)
                ]
                warmup_split = len(patches); did_warmup = True
                cls._add_time(timings, "warmup_switch", time.perf_counter() - ts)
            if config.debug_mode:
                debug_lines.append(cls._debug_line("APPLIED", patch_step=step, stream_patch=len(patches), channel=c, channel_score=f"{channel_scores[c]:.4f}", x=patch["x"], y=patch["y"], w=patch["w"], h=patch["h"], cell_size=patch["cell_size"]))
            if config.debug_print:
                print("|", end="", flush=True)
            if frame_every and applied % frame_every == 0:
                yield {"event": "frame", "step": step, "total": int(config.patch_count), "image": cls._canvas_to_image(canvas, config.color_space, has_alpha)}
            if quality_target > 0 and float(np.mean(np.abs(target - np.clip(canvas, 0, 255)))) <= quality_target:
                break
        if config.debug_print:
            print()
        cls._add_time(timings, "patch_loop_total", time.perf_counter() - t_patch_total)
        t = time.perf_counter(); bw = BitWriter()
        cls._write_header(bw, w, h, original_w, original_h, downsampled, color_id, channels, channel_bits, config.positive_bias, has_alpha, len(patches), base_values, warmup=(warm_w, warm_h, warmup_split) if did_warmup else None)
        for patch in patches:
            cls._write_patch(bw, patch, channel_bits)
        method, body = cls._entropy_pack(bw.finish(), config.use_lzma)
        data = cls.MAGIC + bytes([cls.VERSION, method]) + body
        cls._add_time(timings, "serialize", time.perf_counter() - t)
        t = time.perf_counter(); out_img = cls._canvas_to_image(canvas, config.color_space, has_alpha)
        if out_img.size != (original_w, original_h):
            out_img = out_img.resize((original_w, original_h), cls.RESAMPLE_FILTER, reducing_gap=cls.RESAMPLE_REDUCING_GAP)
        mse = float(np.mean((np.asarray(orig_compare, dtype=np.float32) - np.asarray(out_img, dtype=np.float32)) ** 2))
        cls._add_time(timings, "finalize_mse", time.perf_counter() - t)
        debug_path = None; total_seconds = time.perf_counter() - t0; timings["total"] = total_seconds
        if config.debug_mode:
            ts = time.strftime("%Y%m%d_%H%M%S"); debug_path = config.debug_path or f"debug_{ts}.txt"
            with open(debug_path, "w", encoding="utf-8") as f:
                f.write(cls._debug_line("CONFIG", **{k: v for k, v in config.__dict__.items() if k not in {"debug_path"}}) + "\n")
                f.write(cls._debug_line("IMAGE", original_w=original_w, original_h=original_h, working_w=w, working_h=h, original_pixels=original_w * original_h, working_pixels=w * h, downsample_rate=f"{rate:.6f}", downsampled=int(downsampled), has_alpha=int(has_alpha)) + "\n")
                for k, v in timings.items():
                    f.write(cls._debug_line("TIMER", phase=k, seconds=f"{v:.6f}") + "\n")
                for line in debug_lines:
                    f.write(line + "\n")
        yield {"event": "done", "result": PBC3Result(out_img, data, config, mse, total_seconds, len(data) * 8, original_w, original_h, canvas.shape[1], canvas.shape[0], timings, debug_path, channels=channels)}

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
        downsampled, original_w, original_h, w, h, color_space, channels, channel_bits, positive_bias, has_alpha, patch_count, base_values, warmup_on, warm_w, warm_h, warmup_split = cls._read_header(br)
        canvas = np.zeros((h, w, channels), dtype=np.int32)
        for c, base in enumerate(base_values):
            canvas[:, :, c] = base
        patches_to_read = patch_count if max_patches is None else min(int(max_patches), patch_count)
        for idx in range(patches_to_read):
            if warmup_on and idx == warmup_split:
                canvas = cls._resize_canvas(canvas, warm_w, warm_h)
            channel, x, y, pw, ph, cell_size, values, _ = cls._read_patch(br, channel_bits, positive_bias)
            cls.apply_grid(canvas[:, :, channel], x, y, pw, ph, cell_size, values)
        return canvas, color_space, downsampled, original_w, original_h, canvas.shape[1], canvas.shape[0], has_alpha, channels, patch_count

    @classmethod
    def decompress(cls, data, max_patches=None):
        t0 = time.perf_counter()
        if isinstance(data, str):
            with open(data, "rb") as f:
                data = f.read()
        canvas, color_space, downsampled, original_w, original_h, w, h, has_alpha, channels, patch_count = cls._decode_to_canvas(data, max_patches=max_patches)
        img = cls._canvas_to_image(canvas, color_space, has_alpha)
        if downsampled and img.size != (original_w, original_h):
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


def preload_numba() -> None:
    img = Image.fromarray(np.random.randint(0, 256, (16, 16, 3), dtype=np.uint8))
    PBC3.compress(img, PBC3Config(patch_count=10, auto_downsample_init=True, learned_filler_enabled=False))


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("usage: python PBC3.py input_image output.pbc3")
    else:
        res = PBC3.encode_file(sys.argv[1], sys.argv[2])
        print(f"MSE: {res.mse:.2f} | Size: {len(res.data) / 1024:.2f} KB | Rate: {res.compression_rate:.2f}x | Time: {res.encode_seconds:.3f}s")