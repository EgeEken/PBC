import operator

import numpy as np


def read_grid(br, count, bitcount):
    count, bitcount = operator.index(count), operator.index(bitcount)
    if count < 0 or bitcount < 0:
        raise ValueError('negative grid count or bit width')
    if bitcount > 16:
        raise ValueError('grid bit width exceeds uint16')
    if not count or not bitcount:
        return np.zeros(count, dtype=np.uint16)
    start = br.i * 8 - br.nbits
    end = start + count * bitcount
    byte_end = (end + 7) // 8
    if byte_end > len(br.data):
        raise EOFError('bitstream ended early')
    data = br.data if isinstance(br.data, (bytes, bytearray)) else bytes(br.data)
    packed = np.frombuffer(data, dtype=np.uint8, count=byte_end - start // 8, offset=start // 8)
    bits = np.unpackbits(packed)[start % 8:start % 8 + count * bitcount]
    weights = 1 << np.arange(bitcount - 1, -1, -1, dtype=np.uint16)
    values = (bits.reshape(count, bitcount) @ weights).astype(np.uint16)
    br.i = byte_end
    br.nbits = byte_end * 8 - end
    br.acc = br.data[byte_end - 1] & ((1 << br.nbits) - 1)
    return values


def write_grid(bw, values, bitcount):
    bitcount = operator.index(bitcount)
    if not 0 <= bitcount <= 16:
        raise ValueError('grid bit width must be in 0..16')
    if not bitcount:
        return
    values = np.asarray(values)
    if values.ndim != 1 or values.dtype.kind not in 'iub':
        raise ValueError('grid values must be a flat integer array')
    values = values.astype(np.uint16, copy=False)
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
