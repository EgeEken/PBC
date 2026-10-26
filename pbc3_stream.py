"""PBC3 bitstream framing, separate from the image algorithm."""

import numpy as np

import pbc3_ops as ops
from pbc3_types import BitReader, BitWriter


def write_grid(bw: BitWriter, values, bitcount: int) -> None:
    """Writes a flat grid of palette indices."""
    acc = bw.acc
    nbits = bw.nbits
    data = bw.data
    bitmask = (1 << bitcount) - 1
    for value in values:
        acc = (acc << bitcount) | (int(value) & bitmask)
        nbits += bitcount
        while nbits >= 8:
            shift = nbits - 8
            data.append((acc >> shift) & 255)
            acc &= (1 << shift) - 1
            nbits = shift
    bw.acc = acc
    bw.nbits = nbits


def read_grid(br: BitReader, count: int, bitcount: int) -> np.ndarray:
    """Reads a flat grid of palette indices."""
    values = np.empty(count, dtype=np.uint16)
    data = br.data
    data_len = len(data)
    data_index = br.i
    acc = br.acc
    nbits = br.nbits
    bitmask = (1 << bitcount) - 1
    for i in range(count):
        while nbits < bitcount:
            if data_index >= data_len:
                raise EOFError("bitstream ended early")
            acc = (acc << 8) | data[data_index]
            data_index += 1
            nbits += 8
        shift = nbits - bitcount
        values[i] = (acc >> shift) & bitmask
        acc &= (1 << shift) - 1
        nbits = shift
    br.i = data_index
    br.acc = acc
    br.nbits = nbits
    return values


def write_header(bw, w, h, original_w, original_h, downsampled, color_id, channels,
                 channel_bits, positive_bias, has_alpha, patch_count, base_values, warmup=None):
    """Writes the image-level stream header."""
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
    bw.write(int(warmup is not None), 1)
    if warmup is not None:
        warm_w, warm_h, warm_split = warmup
        bw.write(warm_w, 16)
        bw.write(warm_h, 16)
        bw.write(warm_split, 32)


def read_header(br, color_space_names):
    """Reads the image-level stream header."""
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
    base_values = [br.read(8) for _ in range(channels)]
    warmup_on = bool(br.read(1))
    warm_w = warm_h = warmup_split = None
    if warmup_on:
        warm_w = br.read(16)
        warm_h = br.read(16)
        warmup_split = br.read(32)
    return (
        downsampled, original_w, original_h, w, h, color_space_names[color_id], channels,
        channel_bits, positive_bias, has_alpha, patch_count, base_values, warmup_on, warm_w,
        warm_h, warmup_split,
    )


def write_patch(bw, patch, channel_bits: int) -> None:
    """Writes one generated-palette patch."""
    bw.write(patch["channel"], channel_bits)
    bw.write(patch["x"], 16)
    bw.write(patch["y"], 16)
    bw.write(patch["w"], 16)
    bw.write(patch["h"], 16)
    bw.write(0, 1)
    mask = patch["mask"]
    bw.write(len(mask), 10)
    for bit in mask:
        bw.write(bit, 1)
    bw.write(patch["neg"], 8)
    bw.write(patch["pos"], 8)
    bw.write(patch["max_bitcount"], 4)
    bw.write(patch["cell_size"], 16)
    write_grid(bw, patch["indices"].ravel().astype(np.int64), patch["bitcount"])


def read_patch(br, channel_bits: int, positive_bias: bool = True):
    """Reads one generated-palette patch and returns its decoded values."""
    channel = br.read(channel_bits)
    x, y = br.read(16), br.read(16)
    w, h = br.read(16), br.read(16)
    if br.read(1) != 0:
        raise ValueError("explicit palette patches were removed in PBC3 3.0 release cleanup")
    mask = [br.read(1) for _ in range(br.read(10))]
    negative_max, positive_max = br.read(8), br.read(8)
    max_bitcount = br.read(4)
    bitcount = ops.resolve_palette_bitcount(mask, max_bitcount, negative_max, positive_max, positive_bias)
    palette = ops.palette_generator(mask, max_bitcount, negative_max, positive_max, positive_bias)
    cell_size = br.read(16)
    gw, gh = ops.ceil_div(w, cell_size), ops.ceil_div(h, cell_size)
    indices = read_grid(br, gh * gw, bitcount).reshape(gh, gw)
    return channel, x, y, w, h, cell_size, palette[indices], bitcount
