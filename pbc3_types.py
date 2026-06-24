from dataclasses import dataclass, field
import numpy as np
from PIL import Image


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
    patch_count: int = 50
    search_depth: int = 200
    proposal_depth: int = 50
    exact_depth: int = 10
    min_patch_size: int = 16
    max_patch_size: int = 400
    min_cell_size: int = 1
    max_cell_size: int = 64
    cell_sizes_per_candidate: int = 3
    top_k: int = 20
    search_q_start: float = 0.5
    search_q_end: float = 0.2
    q_init: float = 0.7
    q_start: float = 0.8
    q_end: float = 0.8
    color_space: str = "YCbCr"
    channel_cycle: str = "Sum"
    auto_downsample_init: bool = True
    init_search_depth: int = 3
    downsample_init_cell_size: int = 12
    downsample_palette_bitcount: int = 6
    downsample_rate: float = -1
    auto_downsample_max_pixels: int = 250_000
    warmup_ratio: float = -1
    warm_downsample_max_pixels: int = 750_000
    patch_palette_bitcount: int = 2
    quality_target_mae: float = 0.0
    mask_size: int = 4
    anchor_block_size: int = 8
    positive_bias: bool = True
    learned_filler_enabled: bool = True
    learned_filler_model_path: str = "patch_policy.npz"
    learned_filler_top_k: int = 1
    learned_filler_q: float = 0.6
    learned_filler_candidates: int = 1
    use_lzma: bool = True
    random_seed: int = 2003
    debug_mode: bool = False
    debug_print: bool = False
    debug_path: str = None

    def __post_init__(self):
        cycle = str(self.channel_cycle).strip().lower().replace("_", " ")
        if cycle in {"off", "cycle", "round robin", "roundrobin"}:
            self.channel_cycle = "Mod"
        elif cycle in {"sum", "sum target", "target", "max", "max sum"}:
            self.channel_cycle = "Sum"
        elif cycle == "mod":
            self.channel_cycle = "Mod"
        else:
            self.channel_cycle = str(self.channel_cycle)

    @classmethod
    def _preset(cls, **values):
        values.update(values.pop("overrides", {}))
        return cls(**values)

    @classmethod
    def compression(cls, **kwargs):
        return cls._preset(patch_count=50, search_q_start=0.5, search_q_end=0.2, init_search_depth=3,
                           q_init=0.7, q_start=0.8, q_end=0.8, quality_target_mae=0.0,
                           learned_filler_enabled=True, learned_filler_q=0.4, overrides=kwargs)

    @classmethod
    def balanced(cls, **kwargs):
        return cls._preset(patch_count=50, search_q_start=0.5, search_q_end=0.2, init_search_depth=3,
                           q_init=0.7, q_start=0.8, q_end=0.8, quality_target_mae=0.0,
                           learned_filler_enabled=True, learned_filler_q=0.6, overrides=kwargs)

    @classmethod
    def quality(cls, **kwargs):
        return cls._preset(patch_count=50, search_q_start=0.5, search_q_end=0.2, init_search_depth=3,
                           q_init=0.7, q_start=0.8, q_end=0.8, quality_target_mae=0.0,
                           learned_filler_enabled=True, learned_filler_q=0.8, overrides=kwargs)

    @classmethod
    def high_quality(cls, **kwargs):
        return cls._preset(patch_count=20, search_q_start=0.7, search_q_end=0.2, init_search_depth=3,
                           q_init=0.7, q_start=0.8, q_end=0.8, quality_target_mae=0.0,
                           learned_filler_enabled=True, learned_filler_q=0.95, overrides=kwargs)


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
    def time(self):
        return self.encode_seconds

    @property
    def encode_time(self):
        return self.encode_seconds

    @property
    def decode_time(self):
        return self.encode_seconds

    @property
    def decode_seconds(self):
        return self.encode_seconds

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
        from PBC3 import PBC3
        if self.data is None:
            return False
        decoded = PBC3.decompress(self.data).image
        return np.array_equal(np.asarray(self.image), np.asarray(decoded))

    def show(self, subtitle=None):
        import os
        from matplotlib import pyplot as plt

        fig = plt.figure(figsize=(8, 7.4), dpi=130)
        gs = fig.add_gridspec(3, 1, height_ratios=[0.09, 0.16, 1.0], hspace=0.04)
        title_ax = fig.add_subplot(gs[0])
        info_ax = fig.add_subplot(gs[1])
        image_ax = fig.add_subplot(gs[2])
        for ax in (title_ax, info_ax, image_ax):
            ax.axis("off")
        title_ax.text(0.5, 0.5, "PBC3 Result" if subtitle is None else f"PBC3 Result\n{subtitle}", ha="center", va="center", fontsize=16, fontweight="bold")
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