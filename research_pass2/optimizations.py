from contextlib import contextmanager

import numpy as np
from numba import njit


@njit(cache=True)
def channel_sum(target, canvas, channel):
    total = np.int64(0)
    for y in range(target.shape[0]):
        for x in range(target.shape[1]):
            value = min(255, max(0, int(canvas[y, x, channel])))
            total += abs(int(target[y, x, channel]) - value)
    return float(total)


def read_grid(br, count, bitcount):
    if not count or not bitcount:
        return np.zeros(count, dtype=np.uint16)
    start = br.i * 8 - br.nbits
    end = start + count * bitcount
    byte_end = (end + 7) // 8
    if byte_end > len(br.data):
        raise EOFError('bitstream ended early')
    packed = np.frombuffer(br.data, dtype=np.uint8, count=byte_end - start // 8, offset=start // 8)
    bits = np.unpackbits(packed)[start % 8:start % 8 + count * bitcount]
    weights = (1 << np.arange(bitcount - 1, -1, -1, dtype=np.uint16))
    values = (bits.reshape(count, bitcount) @ weights).astype(np.uint16)
    br.i = byte_end
    br.nbits = byte_end * 8 - end
    br.acc = br.data[byte_end - 1] & ((1 << br.nbits) - 1)
    return values


def write_grid(bw, values, bitcount):
    if not bitcount:
        return
    values = np.asarray(values, dtype=np.uint16)
    shifts = np.arange(bitcount - 1, -1, -1, dtype=np.uint16)
    bits = ((values[:, None] >> shifts) & 1).astype(np.uint8).ravel()
    if bw.nbits:
        prefix = ((bw.acc >> np.arange(bw.nbits - 1, -1, -1)) & 1).astype(np.uint8)
        bits = np.concatenate((prefix, bits))
    whole = len(bits) // 8 * 8
    bw.data.extend(np.packbits(bits[:whole]).tobytes())
    bw.nbits = len(bits) - whole
    bw.acc = 0
    for bit in bits[whole:]:
        bw.acc = (bw.acc << 1) | int(bit)


@contextmanager
def enabled(name):
    from PBC3 import PBC3
    import pbc3_stream as stream
    old_read, old_write = stream.read_grid, stream.write_grid
    old_class_read = PBC3._read_grid
    old_sum = PBC3._channel_sum_error
    try:
        if name == 'grid_numpy':
            stream.read_grid, stream.write_grid = read_grid, write_grid
            PBC3._read_grid = staticmethod(read_grid)
        elif name == 'channel_sum':
            PBC3._channel_sum_error = staticmethod(channel_sum)
        else:
            raise ValueError(name)
        yield
    finally:
        stream.read_grid, stream.write_grid = old_read, old_write
        PBC3._read_grid = staticmethod(old_class_read)
        PBC3._channel_sum_error = staticmethod(old_sum)
