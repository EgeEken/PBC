import hashlib
import math
import struct
import time

import numpy as np
from PIL import Image, ImageOps

from pbc4_stream import HEADER, pack_bits, pack_file, put_uleb, put_sleb_zigzag, unpack_bits, unpack_file

REGION = 16
DCT = np.array([
    [5793,5793,5793,5793,5793,5793,5793,5793],
    [8035,6811,4551,1598,-1598,-4551,-6811,-8035],
    [7568,3135,-3135,-7568,-7568,-3135,3135,7568],
    [6811,-1598,-8035,-4551,4551,8035,1598,-6811],
    [5793,-5793,-5793,5793,5793,-5793,-5793,5793],
    [4551,-8035,1598,6811,-6811,-1598,8035,-4551],
    [3135,-7568,7568,-3135,-3135,7568,-7568,3135],
    [1598,-4551,6811,-8035,8035,-6811,4551,-1598],
], dtype=np.int64)
Q28 = 1 << 28
ZIGZAG = np.array([
    (0,0),(0,1),(1,0),(2,0),(1,1),(0,2),(0,3),(1,2),
    (2,1),(3,0),(4,0),(3,1),(2,2),(1,3),(0,4),(0,5),
    (1,4),(2,3),(3,2),(4,1),(5,0),(6,0),(5,1),(4,2),
    (3,3),(2,4),(1,5),(0,6),(0,7),(1,6),(2,5),(3,4),
    (4,3),(5,2),(6,1),(7,0),(7,1),(6,2),(5,3),(4,4),
    (3,5),(2,6),(1,7),(2,7),(3,6),(4,5),(5,4),(6,3),
    (7,2),(7,3),(6,4),(5,5),(4,6),(3,7),(4,7),(5,6),
    (6,5),(7,4),(7,5),(6,6),(5,7),(6,7),(7,6),(7,7),
], dtype=np.int8)
POINTS = {
    'low': {'transform_step': 16, 'procedure_step': 8, 'rd_lambda': 1000},
    'intermediate': {'transform_step': 4, 'procedure_step': 2, 'rd_lambda': 100},
    'high': {'transform_step': 1, 'procedure_step': 1, 'rd_lambda': 0},
}
VARIANTS = {'A': (False, False), 'B': (True, False),
            'C': (False, True), 'D': (True, True)}


def _round_div(values, denominator):
    values = np.asarray(values, dtype=np.int64)
    half = denominator // 2
    return np.where(values >= 0, (values + half) // denominator,
                    -((-values + half) // denominator))


def _round_scalar(value, denominator):
    value = int(value)
    half = denominator // 2
    return (value + half) // denominator if value >= 0 else -((-value + half) // denominator)


def load_rgb(image):
    if isinstance(image, np.ndarray):
        arr = np.asarray(image)
        if arr.dtype != np.uint8 or arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError('input array must be HxWx3 uint8 RGB')
        return np.ascontiguousarray(arr)
    with Image.open(image) as source:
        return np.asarray(ImageOps.exif_transpose(source).convert('RGB'), dtype=np.uint8).copy()


def _init_size(width, height, cap):
    if width * height <= cap:
        return width, height
    scale = math.sqrt(cap / (width * height))
    iw = max(1, int(round(width * scale)))
    ih = max(1, int(round(height * scale)))
    while iw * ih > cap:
        if iw / width >= ih / height:
            iw -= 1
        else:
            ih -= 1
    return iw, ih


def _make_initialization(source, cap):
    height, width = source.shape[:2]
    iw, ih = _init_size(width, height, cap)
    image = Image.fromarray(source)
    if (iw, ih) != (width, height):
        image = image.resize((iw, ih), Image.Resampling.BICUBIC)
    palette_image = image.quantize(colors=64, method=Image.Quantize.MEDIANCUT,
                                   dither=Image.Dither.NONE)
    indices = np.asarray(palette_image, dtype=np.uint8)
    palette_count = int(indices.max()) + 1
    if palette_count < 1 or palette_count > 64:
        raise ValueError('invalid generated palette')
    palette = np.asarray(palette_image.getpalette()[:palette_count * 3], dtype=np.uint8).reshape(-1, 3)
    bits = max(1, (palette_count - 1).bit_length())
    packed_indices = pack_bits(indices.ravel(), bits)
    init_rgb = palette[indices]
    base = _resize_bilinear(init_rgb, width, height)
    payload = bytes((palette_count,)) + palette.tobytes() + packed_indices
    return iw, ih, palette_count, base, payload, {
        'palette_bytes': palette_count * 3,
        'index_raw_bytes': len(packed_indices),
        'init_samples': iw * ih,
        'init_cap': cap,
    }


def _resize_bilinear(source, out_w, out_h):
    in_h, in_w = source.shape[:2]
    if (in_w, in_h) == (out_w, out_h):
        return np.ascontiguousarray(source.copy())
    scale = 1 << 16
    xfp = np.arange(out_w, dtype=np.int64) * max(0, in_w - 1) * scale // max(1, out_w - 1)
    x0 = xfp >> 16
    x1 = np.minimum(x0 + 1, in_w - 1)
    wx = xfp & 65535
    out = np.empty((out_h, out_w, 3), dtype=np.uint8)
    for y in range(out_h):
        yfp = y * max(0, in_h - 1) * scale // max(1, out_h - 1)
        y0 = yfp >> 16
        y1 = min(y0 + 1, in_h - 1)
        wy = yfp & 65535
        top = (source[y0, x0].astype(np.int64) * (scale - wx[:, None]) +
               source[y0, x1].astype(np.int64) * wx[:, None] + 32768) >> 16
        bottom = (source[y1, x0].astype(np.int64) * (scale - wx[:, None]) +
                  source[y1, x1].astype(np.int64) * wx[:, None] + 32768) >> 16
        out[y] = ((top * (scale - wy) + bottom * wy + 32768) >> 16).astype(np.uint8)
    return out


def _pad(array, height, width):
    out = np.zeros((height, width, array.shape[2]), dtype=array.dtype)
    out[:array.shape[0], :array.shape[1]] = array
    return out


def _blocks(array):
    h, w = array.shape[:2]
    return array.reshape(h // 8, 8, w // 8, 8, 3).transpose(0, 2, 4, 1, 3).reshape(-1, 3, 8, 8)


def _blocks_to_image(block_array, block_rows, block_cols):
    return block_array.reshape(block_rows, block_cols, 3, 8, 8).transpose(0, 3, 1, 4, 2).reshape(block_rows * 8, block_cols * 8, 3)


def _forward_dct_quantized(residual, step):
    block_rows, block_cols = residual.shape[0] // 8, residual.shape[1] // 8
    blocks = _blocks(residual)
    quantized = np.empty(blocks.shape, dtype=np.int16)
    for start in range(0, len(blocks), 512):
        chunk = blocks[start:start + 512].astype(np.int64)
        transformed = np.matmul(np.matmul(DCT, chunk), DCT.T)
        coeff = _round_div(transformed, Q28)
        qcoeff = _round_div(coeff, step)
        if np.any(np.abs(qcoeff) > 32767):
            raise ValueError('transform coefficient overflow')
        quantized[start:start + len(chunk)] = qcoeff.astype(np.int16)
    return quantized, block_rows, block_cols


def _inverse_dct_blocks(quantized, step, block_rows, block_cols):
    reconstructed = np.empty(quantized.shape, dtype=np.int16)
    for start in range(0, len(quantized), 512):
        chunk = quantized[start:start + 512].astype(np.int64) * int(step)
        transformed = np.matmul(np.matmul(DCT.T, chunk), DCT)
        values = _round_div(transformed, Q28)
        reconstructed[start:start + len(chunk)] = np.clip(values, -4096, 4096).astype(np.int16)
    return _blocks_to_image(reconstructed, block_rows, block_cols)


def _regional_view(array):
    rows, cols = array.shape[0] // REGION, array.shape[1] // REGION
    return array.reshape(rows, REGION, cols, REGION, array.shape[2]).transpose(0, 2, 1, 3, 4)


def _sse_by_region(source, base, correction, valid):
    rows, cols = source.shape[0] // REGION, source.shape[1] // REGION
    output = np.empty((rows, cols), dtype=np.int64)
    for row in range(rows):
        y0 = row * REGION
        src = source[y0:y0 + REGION]
        b = base[y0:y0 + REGION].astype(np.int16)
        corr = correction[y0:y0 + REGION].astype(np.int16)
        recon = np.clip(b + corr, 0, 255).astype(np.uint8)
        error = (src.astype(np.int16) - recon.astype(np.int16)).astype(np.int32)
        error *= valid[y0:y0 + REGION]
        output[row] = (error * error).reshape(REGION, cols, REGION, 3).sum(
            axis=(0, 2, 3), dtype=np.int64)
    return output


def _quantized_controls(values, step):
    return (_round_div(_round_div(values, 1), step) * step).astype(np.int16)


def _proc_surface(residual_regions, step):
    corners = np.stack([
        _round_div(residual_regions[:, :, 0:2, 0:2].sum(axis=(2, 3), dtype=np.int64), 4),
        _round_div(residual_regions[:, :, 0:2, 14:16].sum(axis=(2, 3), dtype=np.int64), 4),
        _round_div(residual_regions[:, :, 14:16, 0:2].sum(axis=(2, 3), dtype=np.int64), 4),
        _round_div(residual_regions[:, :, 14:16, 14:16].sum(axis=(2, 3), dtype=np.int64), 4),
    ], axis=2)
    return _quantized_controls(corners, step)


def _surface_sse_by_region(source, base, valid, corners):
    rows, cols = corners.shape[:2]
    output = np.empty((rows, cols), dtype=np.int64)
    xs = np.arange(16, dtype=np.int64) * 256 // 15
    for region_row in range(rows):
        controls = corners[region_row].astype(np.int64)
        correction = np.empty((cols, 16, 16, 3), dtype=np.int16)
        for y in range(16):
            wy = y * 256 // 15
            top = controls[:, 0, None, :] * (256 - xs[None, :, None]) + controls[:, 1, None, :] * xs[None, :, None]
            bottom = controls[:, 2, None, :] * (256 - xs[None, :, None]) + controls[:, 3, None, :] * xs[None, :, None]
            correction[:, y] = _round_div(top * (256 - wy) + bottom * wy, 65536).astype(np.int16)
        y0 = region_row * 16
        corr = correction.transpose(1, 0, 2, 3).reshape(16, source.shape[1], 3)
        b = base[y0:y0 + 16].astype(np.int16)
        recon = np.clip(b + corr, 0, 255).astype(np.uint8)
        error = (source[y0:y0 + 16].astype(np.int16) - recon.astype(np.int16)).astype(np.int32)
        error *= valid[y0:y0 + 16]
        output[region_row] = (error * error).reshape(16, cols, 16, 3).sum(
            axis=(0, 2, 3), dtype=np.int64)
    return output


def _zigzag_coefficients(block):
    return np.asarray(block)[ZIGZAG[:, 0], ZIGZAG[:, 1]]


def _write_coefficients(out, coeffs):
    coeffs = np.asarray(coeffs, dtype=np.int16)
    position = 0
    while position < 64:
        run = 0
        while position + run < 64 and coeffs[position + run] == 0:
            run += 1
        put_uleb(out, run)
        position += run
        if position == 64:
            break
        put_sleb_zigzag(out, int(coeffs[position]))
        position += 1


def _read_coefficients(data, position):
    out = np.zeros(64, dtype=np.int16)
    cursor = 0
    while cursor < 64:
        run, position = _read_uleb(data, position, 64)
        cursor += run
        if cursor > 64:
            raise ValueError('coefficient zero run exceeds block')
        if cursor == 64:
            break
        value, position = _read_signed(data, position, 32767)
        if value == 0:
            raise ValueError('zero coefficient must be represented by its run')
        out[cursor] = value
        cursor += 1
    return out, position


def _read_uleb(data, position, limit):
    from pbc4_stream import get_uleb
    return get_uleb(data, position, limit)


def _read_signed(data, position, limit):
    from pbc4_stream import get_sleb_zigzag
    return get_sleb_zigzag(data, position, limit)


def _encode_modes(source, base, transform_step, procedure_step, rd_lambda,
                  procedural, transform):
    h, w = source.shape[:2]
    ph, pw = (h + 15) // 16 * 16, (w + 15) // 16 * 16
    sp, bp = _pad(source, ph, pw), _pad(base, ph, pw)
    valid = np.zeros((ph, pw, 1), dtype=np.uint8)
    valid[:h, :w] = 1
    residual = sp.astype(np.int16) - bp.astype(np.int16)
    residual_regions = _regional_view(residual)
    region_rows, region_cols = residual_regions.shape[:2]
    skip_sse = _sse_by_region(sp, bp, np.zeros_like(sp, dtype=np.int16), valid)
    best_score = skip_sse.astype(np.float64)
    modes = np.zeros((region_rows, region_cols), dtype=np.uint8)
    subtypes = np.zeros((region_rows, region_cols), dtype=np.uint8)
    const_params = np.zeros((region_rows, region_cols, 3), dtype=np.int16)
    surface_params = None
    dct_q = dct_recon = None
    dct_cost = None
    dct_sse = None

    if procedural:
        valid_regions = _regional_view(valid.astype(np.uint8)).sum(axis=(2, 3, 4)).astype(np.int64)
        sums = residual_regions.sum(axis=(2, 3), dtype=np.int64)
        means = _round_div(sums, np.maximum(1, valid_regions[:, :, None]))
        const_params = _quantized_controls(means, procedure_step)
        const_corr = np.broadcast_to(const_params[:, :, None, None, :],
                                     (region_rows, region_cols, 16, 16, 3)).reshape(ph, pw, 3)
        const_sse = _sse_by_region(sp, bp, const_corr, valid)
        corners = _proc_surface(residual_regions, procedure_step)
        surface_sse = _surface_sse_by_region(sp, bp, valid, corners)
        const_score = const_sse + rd_lambda * 7
        surface_score = surface_sse + rd_lambda * 25
        use_surface = surface_score < const_score
        proc_score = np.minimum(const_score, surface_score)
        proc_score = proc_score.astype(np.float64)
        take = proc_score < best_score
        modes[take] = 1
        subtypes[take] = use_surface[take].astype(np.uint8)
        best_score[take] = proc_score[take]
        surface_params = corners
        proc_metadata = {
            'const_sse': const_sse,
            'surface_sse': surface_sse,
            'const_params': const_params,
            'surface_params': surface_params,
        }
    else:
        proc_metadata = None

    if transform:
        dct_q, block_rows, block_cols = _forward_dct_quantized(residual, transform_step)
        dct_recon = _inverse_dct_blocks(dct_q, transform_step, block_rows, block_cols)
        dct_sse = _sse_by_region(sp, bp, dct_recon, valid)
        qregions = dct_q.reshape(block_rows, block_cols, 3, 8, 8).reshape(
            region_rows, 2, region_cols, 2, 3, 8, 8).transpose(0, 2, 1, 3, 4, 5, 6)
        nz = np.count_nonzero(qregions, axis=(2, 3, 4, 5, 6))
        dct_cost = 12 + 2 * nz
        dct_score = dct_sse + rd_lambda * dct_cost
        take = dct_score < best_score
        modes[take] = 2
        best_score[take] = dct_score[take]
    else:
        block_rows = block_cols = None
        qregions = None

    return {
        'modes': modes, 'subtypes': subtypes, 'const_params': const_params,
        'surface_params': surface_params, 'proc_metadata': proc_metadata,
        'dct_q': dct_q, 'dct_recon': dct_recon, 'qregions': qregions,
        'sse': best_score, 'skip_sse': skip_sse, 'dct_sse': dct_sse,
        'transform_step': transform_step, 'procedure_step': procedure_step,
        'region_rows': region_rows, 'region_cols': region_cols,
        'height': h, 'width': w, 'padded_height': ph, 'padded_width': pw,
        'block_rows': block_rows, 'block_cols': block_cols,
    }


def _serialize_payload(init_payload, encoded):
    modes = encoded['modes'].ravel()
    payload = bytearray(init_payload)
    map_bytes = pack_bits(modes, 2)
    payload.extend(map_bytes)
    proc_bytes = transform_bytes = 0
    qregions = encoded['qregions']
    for index, mode in enumerate(modes):
        row, col = divmod(index, encoded['region_cols'])
        if mode == 1:
            subtype = int(encoded['subtypes'][row, col])
            record = bytearray((subtype,))
            if subtype == 0:
                for value in encoded['const_params'][row, col]:
                    record.extend(struct.pack('<h', int(value)))
            else:
                for value in encoded['surface_params'][row, col].ravel():
                    record.extend(struct.pack('<h', int(value)))
            payload.extend(record)
            proc_bytes += len(record)
        elif mode == 2:
            region = qregions[row, col]
            record = bytearray()
            for block_y in range(2):
                for block_x in range(2):
                    for channel in range(3):
                        coeff = _zigzag_coefficients(region[block_y, block_x, channel])
                        _write_coefficients(record, coeff)
            payload.extend(record)
            transform_bytes += len(record)
    return bytes(payload), {
        'mode_map_raw_bytes': len(map_bytes),
        'procedural_raw_bytes': proc_bytes,
        'transform_raw_bytes': transform_bytes,
        'raw_payload_bytes': len(payload),
    }


def encode(image, variant='D', point='intermediate', init_caps=(250_000, 750_000), taper=False):
    if variant not in VARIANTS or point not in POINTS:
        raise ValueError('unknown codec variant or operating point')
    source = load_rgb(image)
    h, w = source.shape[:2]
    params = POINTS[point]
    procedural, transform = VARIANTS[variant]
    init_summaries = []
    selected = None
    for cap in dict.fromkeys(init_caps):
        iw, ih, palette_count, base, init_payload, init_stats = _make_initialization(source, cap)
        encoded = _encode_modes(source, base, params['transform_step'],
                                params['procedure_step'], params['rd_lambda'],
                                procedural, transform)
        payload, byte_stats = _serialize_payload(init_payload, encoded)
        data = pack_file(w, h, iw, ih, palette_count, params['procedure_step'],
                         params['transform_step'], params['rd_lambda'], payload, taper=taper)
        counts = np.bincount(encoded['modes'].ravel(), minlength=3)
        mode_counts = {'skip': int(counts[0]), 'procedural': int(counts[1]),
                       'transform': int(counts[2])}
        del encoded, payload, init_payload, base
        reconstruction, decoded_info = decode(data, return_info=True)
        diff = source.astype(np.int32) - reconstruction.astype(np.int32)
        sse = int(np.sum(diff.astype(np.int64) ** 2))
        mse = sse / (h * w * 3)
        score = sse + params['rd_lambda'] * len(data)
        file_hash = hashlib.sha256(data).hexdigest()
        pixel_hash = hashlib.sha256(reconstruction.tobytes()).hexdigest()
        byte_accounting = {'header_bytes': HEADER.size,
                           'xz_payload_bytes': len(data) - HEADER.size,
                           **init_stats, **byte_stats}
        summary = {
            'init_cap': cap, 'init_width': iw, 'init_height': ih,
            'palette_count': palette_count, 'bytes': len(data), 'mse': mse,
            'sse': sse, 'selection_score': score, 'file_sha256': file_hash,
            'pixel_sha256': pixel_hash, 'mode_counts': mode_counts,
            'byte_accounting': byte_accounting,
        }
        init_summaries.append({key: value for key, value in summary.items() if key != 'selection_score'})
        candidate = {
            'data': data, 'reconstruction': reconstruction, 'mse': mse,
            'rgb_psnr': float('inf') if mse == 0 else 10 * math.log10(255.0 ** 2 / mse),
            'sse': sse, 'selection_score': score, 'width': w, 'height': h,
            'init_width': iw, 'init_height': ih, 'init_cap': cap,
            'palette_count': palette_count, 'variant': variant, 'point': point,
            'mode_counts': mode_counts, 'byte_accounting': byte_accounting,
            'pixel_sha256': pixel_hash, 'file_sha256': file_hash,
            'decoder_mode_counts': decoded_info['mode_counts'],
            'skip_exact': decoded_info['skip_exact'],
        }
        if selected is None or (score, len(data)) < (selected['selection_score'], len(selected['data'])):
            selected = candidate
        del diff, reconstruction, data, decoded_info, candidate
    selected['init_candidate_results'] = init_summaries
    return selected


def _taper_correction(correction, modes, row, col):
    weights_y = np.full(16, 256, dtype=np.int64)
    weights_x = np.full(16, 256, dtype=np.int64)
    if row > 0 and modes[row - 1, col] == 0:
        weights_y[:2] = (64, 192)
    if row + 1 < modes.shape[0] and modes[row + 1, col] == 0:
        weights_y[-2:] = (192, 64)
    if col > 0 and modes[row, col - 1] == 0:
        weights_x[:2] = (64, 192)
    if col + 1 < modes.shape[1] and modes[row, col + 1] == 0:
        weights_x[-2:] = (192, 64)
    weight = np.minimum(weights_y[:, None], weights_x[None, :])[:, :, None]
    return _round_div(correction.astype(np.int64) * weight, 256).astype(np.int16)


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
    base = _resize_bilinear(init, header['width'], header['height'])
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
                            coefficients, pos = _read_coefficients(payload, pos)
                            matrix = np.zeros((8, 8), dtype=np.int16)
                            matrix[ZIGZAG[:, 0], ZIGZAG[:, 1]] = coefficients
                            transformed = np.matmul(np.matmul(DCT.T, matrix.astype(np.int64) * transform_step), DCT)
                            block = np.clip(_round_div(transformed, Q28), -4096, 4096).astype(np.int16)
                            correction[by * 8:(by + 1) * 8, bx * 8:(bx + 1) * 8, channel] = block
            if header['taper']:
                correction = _taper_correction(correction, modes, row, col)
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
            'skip_exact': _verify_skip(output, base, modes)}
    return (output, info) if return_info else output


def _surface_from_controls(controls):
    controls = np.asarray(controls, dtype=np.int64).reshape(4, 3)
    xs = np.arange(16, dtype=np.int64) * 256 // 15
    out = np.empty((16, 16, 3), dtype=np.int16)
    for y in range(16):
        wy = y * 256 // 15
        top = controls[0] * (256 - xs[:, None]) + controls[1] * xs[:, None]
        bottom = controls[2] * (256 - xs[:, None]) + controls[3] * xs[:, None]
        out[y] = _round_div(top * (256 - wy) + bottom * wy, 65536).astype(np.int16)
    return out


def _verify_skip(output, base, modes):
    for row, col in np.argwhere(modes == 0):
        y0, x0 = int(row) * 16, int(col) * 16
        hh, ww = min(16, output.shape[0] - y0), min(16, output.shape[1] - x0)
        if not np.array_equal(output[y0:y0 + hh, x0:x0 + ww], base[y0:y0 + hh, x0:x0 + ww]):
            return False
    return True

