# PBC3 supporting types: bit I/O, config, and result container.
# Split out of PBC3.py to keep that module under size limits and improve structure.

from dataclasses import dataclass, field
import os
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
    patch_bitcount_mode: str = "constant"
    palette_mode: str = "generated"
    palette_difference_threshold: int = 0
    palette_difference_threshold_mode: str = "constant"
    explicit_palette_max_bitcount: int = 3
    quality_target_mae: float = 0.0
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
        from PBC3 import PBC3
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
