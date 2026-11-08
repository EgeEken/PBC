import hashlib
import math
import pathlib
import struct
import sys

import numpy as np
from PIL import Image

HERE = pathlib.Path(__file__).resolve().parent
FROZEN = HERE.parent / 'research_pass42' / 'outputs' / 'v31_frozen'
sys.path.insert(0, str(FROZEN))
from PBC3 import PBC3
from pbc3_types import BitReader

import pbc4_codec as v1
from pbc4_stream import unpack_bits
from pbc4_v2_stream import pack_v2, unpack_v2
from pbc4_decode_opt import _surface_from_controls

POINTS = v1.POINTS
VARIANTS = v1.VARIANTS


def unpack_p3_prefix(prefix_data):
    _, raw_body = PBC3._open_body(prefix_data)
    header = PBC3._read_header(BitReader(raw_body))
    (downsampled, original_w, original_h, working_w, working_h, color_space,
     channels, channel_bits, positive_bias, has_alpha, patch_count, base_values,
     warmup_on, warm_w, warm_h, warmup_split) = header
    return raw_body, {
        'working_width': working_w, 'working_height': working_h,
        'patch_count': patch_count, 'color_space': color_space,
        'downsampled': downsampled,
    }


def decode_coarse_body(raw_body):
    store_stream = PBC3.MAGIC + bytes((PBC3.VERSION, PBC3.ENTROPY_STORE)) + raw_body
    result = PBC3.decompress(store_stream)
    return np.asarray(result.image.convert('RGB'), dtype=np.uint8).copy()


def _decode_v2(data, return_info=False):
    header, payload = unpack_v2(data)
    if len(payload) < 4:
        raise ValueError('truncated coarse-stage length')
    coarse_bytes = struct.unpack_from('<I', payload, 0)[0]
    coarse_end = 4 + coarse_bytes
    if coarse_end > len(payload):
        raise ValueError('truncated coarse-stage stream')
    base = decode_coarse_body(payload[4:coarse_end])
    if base.shape[:2] != (header['height'], header['width']):
        raise ValueError('coarse reconstruction dimensions mismatch')
    pos = coarse_end
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
                            coefficients, pos = v1._read_coefficients(payload, pos)
                            matrix = np.zeros((8, 8), dtype=np.int16)
                            matrix[v1.ZIGZAG[:, 0], v1.ZIGZAG[:, 1]] = coefficients
                            transformed = np.matmul(np.matmul(v1.DCT.T, matrix.astype(np.int64) * transform_step), v1.DCT)
                            block = np.clip(v1._round_div(transformed, v1.Q28), -4096, 4096).astype(np.int16)
                            correction[by * 8:(by + 1) * 8, bx * 8:(bx + 1) * 8, channel] = block
            y0, x0 = row * 16, col * 16
            hh = min(16, header['height'] - y0)
            ww = min(16, header['width'] - x0)
            output[y0:y0 + hh, x0:x0 + ww] = np.clip(
                output[y0:y0 + hh, x0:x0 + ww].astype(np.int32) + correction[:hh, :ww].astype(np.int32),
                0, 255).astype(np.uint8)
    if pos != len(payload):
        raise ValueError('unexpected trailing payload bytes')
    info = {'mode_counts': {'skip': int(counts[0]), 'procedural': int(counts[1]), 'transform': int(counts[2])},
            'skip_exact': v1._verify_skip(output, base, modes),
            'base_sha256': hashlib.sha256(base.tobytes()).hexdigest()}
    return (output, info) if return_info else output


def decode(data, return_info=False):
    return _decode_v2(data, return_info)


def encode_from_prefix(image, prefix_data, variant='D', point='low'):
    if variant not in VARIANTS or point not in POINTS:
        raise ValueError('unknown codec variant or operating point')
    source = v1.load_rgb(image)
    raw_body, coarse_info = unpack_p3_prefix(prefix_data)
    base = decode_coarse_body(raw_body)
    if base.shape != source.shape:
        raise ValueError('coarse base and source dimensions differ')
    params = POINTS[point]
    procedural, transform = VARIANTS[variant]
    encoded = v1._encode_modes(source, base, params['transform_step'], params['procedure_step'],
                               params['rd_lambda'], procedural, transform)
    coarse_field = struct.pack('<I', len(raw_body)) + raw_body
    payload, byte_stats = v1._serialize_payload(coarse_field, encoded)
    data = pack_v2(source.shape[1], source.shape[0], coarse_info['working_width'],
                   coarse_info['working_height'], params['procedure_step'],
                   params['transform_step'], params['rd_lambda'], payload)
    reconstruction, info = decode(data, return_info=True)
    delta = source.astype(np.int32) - reconstruction.astype(np.int32)
    sse = int(np.sum(delta.astype(np.int64) ** 2))
    mse = sse / (source.shape[0] * source.shape[1] * 3)
    counts = np.bincount(encoded['modes'].ravel(), minlength=3)
    return {
        'data': data, 'reconstruction': reconstruction, 'base': base,
        'bytes': len(data), 'bpp': len(data) * 8 / (source.shape[0] * source.shape[1]),
        'mse': mse, 'sse': sse, 'psnr': float('inf') if mse == 0 else 10 * math.log10(255**2 / mse),
        'score': sse + params['rd_lambda'] * len(data),
        'file_sha256': hashlib.sha256(data).hexdigest(),
        'pixel_sha256': hashlib.sha256(reconstruction.tobytes()).hexdigest(),
        'base_sha256': hashlib.sha256(base.tobytes()).hexdigest(),
        'base_mse': float(np.mean((source.astype(np.int32) - base.astype(np.int32)).astype(np.int64) ** 2)),
        'coarse': coarse_info, 'mode_counts': {'skip': int(counts[0]), 'procedural': int(counts[1]), 'transform': int(counts[2])},
        'byte_accounting': {'header_bytes': 32, 'xz_bytes': len(data) - 32,
                            'coarse_raw_bytes': len(raw_body), **byte_stats},
        'skip_exact': info['skip_exact'], 'variant': variant, 'point': point,
    }


def encode_v1_control(image, variant='D', point='low'):
    return v1.encode(image, variant=variant, point=point)
