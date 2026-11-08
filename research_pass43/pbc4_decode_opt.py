import numpy as np

import pbc4_codec as ref
from pbc4_stream import unpack_bits, unpack_file

_XY = np.arange(16, dtype=np.int64) * 256 // 15
_SURFACE_WEIGHTS = np.asarray([
    ((256 - x) * (256 - y), x * (256 - y), (256 - x) * y, x * y)
    for y in _XY for x in _XY
], dtype=np.int64)


def _surface_from_controls(controls):
    controls = np.asarray(controls, dtype=np.int64).reshape(4, 3)
    values = ref._round_div(_SURFACE_WEIGHTS @ controls, 65536)
    return values.reshape(16, 16, 3).astype(np.int16)


def decode(data, return_info=False):
    header, payload = unpack_file(data)
    pos = 0
    palette_count = payload[pos]
    pos += 1
    if palette_count != header['palette_count']:
        raise ValueError('palette count mismatch')
    palette_bytes = palette_count * 3
    if pos + palette_bytes > len(payload):
        raise ValueError('truncated palette')
    palette = np.frombuffer(payload[pos:pos + palette_bytes], dtype=np.uint8).reshape(palette_count, 3)
    pos += palette_bytes
    index_count = header['init_width'] * header['init_height']
    index_bits = max(1, (palette_count - 1).bit_length())
    index_bytes = (index_count * index_bits + 7) // 8
    if pos + index_bytes > len(payload):
        raise ValueError('truncated initialization indices')
    indices = np.asarray(unpack_bits(payload[pos:pos + index_bytes], index_count, index_bits), dtype=np.uint8)
    if indices.size and indices.max() >= palette_count:
        raise ValueError('palette index out of range')
    pos += index_bytes
    init = palette[indices.reshape(header['init_height'], header['init_width'])]
    base = ref._resize_bilinear(init, header['width'], header['height'])
    rows, cols = (header['height'] + 15) // 16, (header['width'] + 15) // 16
    mode_count = rows * cols
    map_bytes = (mode_count * 2 + 7) // 8
    if pos + map_bytes > len(payload):
        raise ValueError('truncated region mode map')
    modes = np.asarray(unpack_bits(payload[pos:pos + map_bytes], mode_count, 2), dtype=np.uint8).reshape(rows, cols)
    if np.any(modes == 3):
        raise ValueError('reserved region mode')
    pos += map_bytes
    output = base.copy()
    counts = np.bincount(modes.ravel(), minlength=3)
    transform_step = header['transform_step']
    for row in range(rows):
        for col in range(cols):
            mode = int(modes[row, col])
            if mode == 0:
                continue
            if mode == 1:
                if pos >= len(payload):
                    raise ValueError('truncated procedural mode')
                subtype = payload[pos]
                pos += 1
                if subtype == 0:
                    size = 6
                    if pos + size > len(payload):
                        raise ValueError('truncated constant correction')
                    correction = np.frombuffer(payload[pos:pos + size], dtype='<i2').astype(np.int16)
                    pos += size
                    correction = np.broadcast_to(correction, (16, 16, 3)).copy()
                elif subtype == 1:
                    size = 24
                    if pos + size > len(payload):
                        raise ValueError('truncated surface correction')
                    controls = np.frombuffer(payload[pos:pos + size], dtype='<i2').astype(np.int16).reshape(4, 3)
                    pos += size
                    correction = _surface_from_controls(controls)
                else:
                    raise ValueError('invalid procedural subtype')
            else:
                correction = np.zeros((16, 16, 3), dtype=np.int16)
                for by in range(2):
                    for bx in range(2):
                        for channel in range(3):
                            coefficients, pos = ref._read_coefficients(payload, pos)
                            matrix = np.zeros((8, 8), dtype=np.int16)
                            matrix[ref.ZIGZAG[:, 0], ref.ZIGZAG[:, 1]] = coefficients
                            transformed = np.matmul(np.matmul(ref.DCT.T, matrix.astype(np.int64) * transform_step), ref.DCT)
                            block = np.clip(ref._round_div(transformed, ref.Q28), -4096, 4096).astype(np.int16)
                            correction[by * 8:(by + 1) * 8, bx * 8:(bx + 1) * 8, channel] = block
            if header['taper']:
                correction = ref._taper_correction(correction, modes, row, col)
            y0, x0 = row * 16, col * 16
            hh = min(16, header['height'] - y0)
            ww = min(16, header['width'] - x0)
            output[y0:y0 + hh, x0:x0 + ww] = np.clip(
                output[y0:y0 + hh, x0:x0 + ww].astype(np.int32) + correction[:hh, :ww].astype(np.int32),
                0, 255).astype(np.uint8)
    if pos != len(payload):
        raise ValueError('unexpected trailing payload bytes')
    info = {'mode_counts': {'skip': int(counts[0]), 'procedural': int(counts[1]),
                            'transform': int(counts[2])},
            'skip_exact': ref._verify_skip(output, base, modes)}
    return (output, info) if return_info else output
