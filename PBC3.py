from dataclasses import dataclass
import time
import math
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
    color_space: str = "RGB"
    downsample_cell_size: int = 8
    mask_size: int = 9
    palette_bitcount: int = 3
    positive_bias: bool = True
    patch_count: int = 0
    random_seed: int = 2003
    palette_max: int = None


@dataclass
class PBC3Result:
    image: Image.Image
    data: bytes
    config: PBC3Config
    mse: float
    encode_seconds: float
    total_bits: int

    @property
    def original_bits(self):
        return self.image.width * self.image.height * 3 * 8

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
        info = (
            f"MSE: {mse}   |   "
            f"Compressed: {self.compressed_kb:.2f} KB   |   "
            f"Original RGB: {self.original_kb:.2f} KB\n"
            f"Compression: {self.compression_rate:.2f}x ({self.compressed_percent:.2f}%)   |   "
            f"Time: {seconds}"
        )
        info_ax.text(
            0.5, 0.5, info,
            ha="center", va="center",
            color="white",
            fontsize=10,
            linespacing=1.35,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="black", alpha=0.72, edgecolor="none"),
        )

        image_ax.imshow(self.image)
        plt.show()


class PBC3:
    MAGIC = b"PBC3"
    VERSION = 0
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
    def _palette_bounds(values):
        min_value = int(np.min(values))
        max_value = int(np.max(values))
        negative_max = min(255, max(0, -min_value))
        positive_max = min(255, max(0, max_value))
        return negative_max, positive_max

    @classmethod
    def _range_counts(cls, mask_size, negative_max=255, positive_max=255, positive_bias=True):
        side_bits = max(0, mask_size - 1)
        negative_max = max(0, int(negative_max))
        positive_max = max(0, int(positive_max))
        if side_bits == 0 or (negative_max == 0 and positive_max == 0):
            return 0, 0
        if negative_max == 0:
            return side_bits, 0
        if positive_max == 0:
            return 0, side_bits
        raw_pos = side_bits * positive_max / (positive_max + negative_max)
        pos_count = math.ceil(raw_pos) if positive_bias else math.floor(raw_pos)
        pos_count = min(side_bits - 1, max(1, pos_count))
        return pos_count, side_bits - pos_count

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
            return start, end
        bin_i = index - 1 - pos_count
        if 0 <= bin_i < neg_count:
            low_mag = 1 + (bin_i * negative_max) // neg_count
            high_mag = ((bin_i + 1) * negative_max) // neg_count
            return -high_mag, -low_mag
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
            if not bit:
                continue
            r = cls._range_for_mask_index(i, len(mask), negative_max, positive_max, positive_bias)
            if r is None:
                continue
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
            if not bit:
                continue
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
            if count <= 0:
                continue
            if count == 1:
                palette.append(int(round((start + end) / 2)))
            else:
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
    def _make_patch(cls, channel, x, y, w, h, cell_size, residual, config):
        small = cls.signed_resample_cells(residual, cell_size)
        negative_max, positive_max = cls._palette_bounds(small)
        mask = cls._mask_from_values(small, config.mask_size, negative_max, positive_max, config.positive_bias)
        palette = cls.palette_generator(mask, config.palette_bitcount, negative_max, positive_max, config.positive_bias)
        indices = cls.quantize_signed(np.clip(small, -negative_max, positive_max), palette)
        values = palette[indices]
        return (channel, x, y, w, h, mask, negative_max, positive_max, config.palette_bitcount, cls.MODE_RAW, cell_size, indices), values

    @classmethod
    def compress(cls, image, config=None, **kwargs):
        if config is None:
            config = PBC3Config(**kwargs)
        elif kwargs:
            config = PBC3Config(**{**config.__dict__, **kwargs})

        t0 = time.perf_counter()
        img = cls._to_image(image).convert(config.color_space)
        arr = np.asarray(img, dtype=np.uint8)
        h, w, channels = arr.shape
        if w > 65535 or h > 65535:
            raise ValueError("this prototype stores width/height as uint16")
        if config.mask_size < 1 or config.mask_size > 1023:
            raise ValueError("mask_size must be in 1..1023")
        if config.palette_bitcount < 1 or config.palette_bitcount > 9:
            raise ValueError("palette_bitcount must be in 1..9")

        color_id = cls.COLOR_SPACES[config.color_space]
        channel_bits = max(1, math.ceil(math.log2(channels)))
        base_values = [int(round(float(np.mean(arr[:, :, c])))) for c in range(channels)]
        canvas = np.zeros((h, w, channels), dtype=np.int32)
        for c, base in enumerate(base_values):
            canvas[:, :, c] = base

        patches = []
        for c in range(channels):
            residual = arr[:, :, c].astype(np.int16) - base_values[c]
            patch, values = cls._make_patch(c, 0, 0, w, h, config.downsample_cell_size, residual, config)
            cls.apply_grid(canvas[:, :, c], 0, 0, w, h, config.downsample_cell_size, values)
            patches.append(patch)

        rng = np.random.default_rng(config.random_seed)
        for _ in range(config.patch_count):
            c = int(rng.integers(0, channels))
            pw = int(rng.integers(max(1, w // 16), max(2, w // 3)))
            ph = int(rng.integers(max(1, h // 16), max(2, h // 3)))
            x = int(rng.integers(0, max(1, w - pw + 1)))
            y = int(rng.integers(0, max(1, h - ph + 1)))
            cell = int(rng.choice([1, 2, 4, 8, 16, 32]))
            cell = max(1, min(cell, pw, ph))
            residual = arr[y:y + ph, x:x + pw, c].astype(np.int16) - np.clip(canvas[y:y + ph, x:x + pw, c], 0, 255).astype(np.int16)
            patch, values = cls._make_patch(c, x, y, pw, ph, cell, residual, config)
            cls.apply_grid(canvas[:, :, c], x, y, pw, ph, cell, values)
            patches.append(patch)

        bw = BitWriter()
        bw.write(w, 16)
        bw.write(h, 16)
        bw.write(color_id, 2)
        bw.write(channels, 8)
        bw.write(channel_bits, 4)
        bw.write(int(config.positive_bias), 1)
        bw.write(len(patches), 32)
        for base in base_values:
            bw.write(base, 8)
        for patch in patches:
            cls._write_patch(bw, *patch, channel_bits=channel_bits, positive_bias=config.positive_bias)

        data = cls.MAGIC + bytes([cls.VERSION]) + bw.finish()
        out_arr = np.clip(canvas, 0, 255).astype(np.uint8)
        out_img = Image.fromarray(out_arr, config.color_space).convert("RGB")
        target_rgb = np.asarray(img.convert("RGB"), dtype=np.float32)
        out_rgb = np.asarray(out_img, dtype=np.float32)
        mse = float(np.mean((target_rgb - out_rgb) ** 2))
        return PBC3Result(out_img, data, config, mse, time.perf_counter() - t0, len(data) * 8)

    @classmethod
    def decompress(cls, data):
        if isinstance(data, str):
            with open(data, "rb") as f:
                data = f.read()
        if data[:4] != cls.MAGIC:
            raise ValueError("not a PBC3 file")
        version = data[4]
        if version != cls.VERSION:
            raise ValueError(f"unsupported PBC3 version {version}")
        t0 = time.perf_counter()

        br = BitReader(data[5:])
        w = br.read(16)
        h = br.read(16)
        color_id = br.read(2)
        channels = br.read(8)
        channel_bits = br.read(4)
        positive_bias = bool(br.read(1))
        patch_count = br.read(32)
        color_space = cls.COLOR_SPACE_NAMES[color_id]
        base_values = [br.read(8) for _ in range(channels)]
        canvas = np.zeros((h, w, channels), dtype=np.int32)
        for c, base in enumerate(base_values):
            canvas[:, :, c] = base

        for _ in range(patch_count):
            channel, x, y, pw, ph, cell_size, values, mode = cls._read_patch(br, channel_bits, positive_bias)
            if mode != cls.MODE_RAW:
                raise ValueError(f"unsupported patch mode {mode}")
            cls.apply_grid(canvas[:, :, channel], x, y, pw, ph, cell_size, values)

        arr = np.clip(canvas, 0, 255).astype(np.uint8)
        cfg = PBC3Config(color_space=color_space, positive_bias=positive_bias)
        return PBC3Result(Image.fromarray(arr, color_space).convert("RGB"), data, cfg, None, time.perf_counter() - t0, len(data) * 8)

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