from __future__ import annotations

import hashlib
import lzma
import math
import struct
import time

import numpy as np
from scipy.fft import dctn, idctn

CELL = 16
DC_STEP = 8
DETAIL_STEP = 16
QUANT_STEPS = (16, 32, 64)
MODE_BITS = 3
HEADER = struct.Struct('<4sBIIHBBBBIII')
MAGIC = b'P33S'
MAX_PIXELS = 50_000_000
MAX_RAW = 128 * 1024 * 1024
MAX_XZ = 64 * 1024 * 1024
MAX_BLOCKS = 8
SHORTLIST = 16

_BASIS = idctn(np.eye(64, dtype=np.float64).reshape(64, 8, 8), axes=(-2, -1), type=2, norm='ortho', workers=1)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def round_div_away(values: np.ndarray, divisor: int) -> np.ndarray:
    values = np.asarray(values, np.int64)
    return np.where(values < 0, -((-values + divisor // 2) // divisor), (values + divisor // 2) // divisor)


def pack_modes(values: np.ndarray) -> bytes:
    out = bytearray()
    acc = bits = 0
    for value in np.asarray(values, np.uint8).reshape(-1):
        if int(value) > 7:
            raise ValueError('invalid mode')
        acc = (acc << MODE_BITS) | int(value)
        bits += MODE_BITS
        while bits >= 8:
            bits -= 8
            out.append((acc >> bits) & 255)
            acc &= (1 << bits) - 1 if bits else 0
    if bits:
        out.append((acc << (8 - bits)) & 255)
    return bytes(out)


def unpack_modes(data: bytes, count: int) -> np.ndarray:
    out = np.empty(count, np.uint8)
    acc = bits = pos = 0
    for i in range(count):
        while bits < MODE_BITS:
            if pos >= len(data):
                raise ValueError('truncated mode stream')
            acc = (acc << 8) | data[pos]
            pos += 1
            bits += 8
        bits -= MODE_BITS
        out[i] = (acc >> bits) & 7
        acc &= (1 << bits) - 1 if bits else 0
    if pos != len(data) or acc or np.any(~np.isin(out, (0, 1, 2, 4))):
        raise ValueError('invalid mode stream or padding')
    return out


def qvector(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, np.int64).reshape(-1, 3)
    levels = round_div_away(values.sum(0), values.shape[0] * DC_STEP)
    return np.clip(levels, -128, 127).astype(np.int8)


def detail_units(residual: np.ndarray) -> np.ndarray:
    h, w = residual.shape[:2]
    grid = np.empty((4, 4, 3), np.int8)
    for gy in range(4):
        for gx in range(4):
            y0, x0 = gy * 4, gx * 4
            block = residual[y0:min(y0 + 4, h), x0:min(x0 + 4, w)].astype(np.int64)
            if block.size:
                levels = round_div_away(block.sum((0, 1)), block.shape[0] * block.shape[1] * DETAIL_STEP)
                grid[gy, gx] = np.clip(levels, -128, 127).astype(np.int8)
            else:
                grid[gy, gx] = grid[min(gy, max(0, math.ceil(h / 4) - 1)), min(gx, max(0, math.ceil(w / 4) - 1))]
    return grid


def interpolate_grid(units: np.ndarray) -> np.ndarray:
    grid = np.asarray(units, np.int16).astype(np.int32) * DETAIL_STEP
    coord = np.arange(CELL)
    y0, fy = coord // 5, coord % 5
    x0, fx = coord // 5, coord % 5
    y1, x1 = np.minimum(y0 + 1, 3), np.minimum(x0 + 1, 3)
    fy2=(5-fy)[:,None,None]; fx2=(5-fx)[None,:,None]
    fy1=fy[:,None,None]; fx1=fx[None,:,None]
    total=(grid[y0[:,None],x0[None,:]]*fy2*fx2 + grid[y0[:,None],x1[None,:]]*fy2*fx1 +
           grid[y1[:,None],x0[None,:]]*fy1*fx2 + grid[y1[:,None],x1[None,:]]*fy1*fx1)
    return round_div_away(total, 25).astype(np.int16)


def cell_slices(height: int, width: int, index: int) -> tuple[int, int, int, int]:
    cols = math.ceil(width / CELL)
    cy, cx = divmod(index, cols)
    y, x = cy * CELL, cx * CELL
    return y, x, min(CELL, height - y), min(CELL, width - x)


def _zigzag(value: int) -> int:
    return (value << 1) ^ (value >> 15)


def _varint(value: int) -> bytes:
    if not 0 <= value <= 65535:
        raise ValueError('coefficient outside int16 zigzag range')
    out = bytearray()
    while value >= 128:
        out.append((value & 127) | 128)
        value >>= 7
    out.append(value)
    return bytes(out)


def _read_varint(data: bytes, pos: int) -> tuple[int, int]:
    value = shift = 0
    start = pos
    for _ in range(3):
        if pos >= len(data):
            raise ValueError('truncated coefficient varint')
        byte = data[pos]
        pos += 1
        value |= (byte & 127) << shift
        if byte < 128:
            if pos - start > 1 and byte == 0:
                raise ValueError('noncanonical coefficient varint')
            if value > 65535:
                raise ValueError('coefficient varint overflow')
            return value, pos
        shift += 7
    raise ValueError('coefficient varint too long')


def _coefficient_record(position: int, value: int) -> bytes:
    if not 0 <= position < 192 or not -32768 <= value <= 32767 or value == 0:
        raise ValueError('invalid sparse coefficient')
    return bytes((position,)) + _varint(_zigzag(value))


def _block_options(source: np.ndarray, base: np.ndarray, step: int) -> list[tuple[int, list[tuple[int, int]]]]:
    h, w = source.shape[:2]
    if h <= 0 or w <= 0:
        return [(0, [])]
    residual = np.zeros((8, 8, 3), np.float64)
    residual[:h, :w] = source.astype(np.float64) - base.astype(np.float64)
    coeff = dctn(residual, axes=(0, 1), type=2, norm='ortho', workers=1)
    quantized = np.rint(coeff / step).astype(np.int32)
    if np.any(quantized < -32768) or np.any(quantized > 32767):
        raise ValueError('quantized coefficient exceeds int16')
    base64 = base.astype(np.float64)
    source64 = source.astype(np.int32)
    base_sse = int(np.square(source64 - base.astype(np.int32), dtype=np.int64).sum())
    proposals = []
    for channel in range(3):
        for u in range(8):
            for v in range(8):
                value = int(quantized[u, v, channel])
                if value:
                    position = channel * 64 + u * 8 + v
                    record = _coefficient_record(position, value)
                    energy = float(np.square(_BASIS[u * 8 + v, :h, :w] * (step * value)).sum())
                    proposals.append((energy / (8 * len(record)), position, value))
    proposals.sort(key=lambda item: (-item[0], item[1]))
    shortlist = proposals[:SHORTLIST]
    measured = []
    for _, position, value in shortlist:
        channel, rem = divmod(position, 64)
        u, v = divmod(rem, 8)
        corrected = np.clip(np.rint(base64[:, :, channel] + _BASIS[u * 8 + v, :h, :w] * (step * value)), 0, 255)
        test = source64.copy()
        test[:, :, channel] = corrected.astype(np.int32)
        gain = base_sse - int(np.square(source64 - test, dtype=np.int64).sum())
        if gain > 0:
            measured.append((gain / (8 * len(_coefficient_record(position, value))), position, value))
    measured.sort(key=lambda item: (-item[0], item[1]))
    selected = measured[:MAX_BLOCKS]
    options = [(base_sse, [])]
    selected_rows = []
    for _, position, value in selected:
        channel, rem = divmod(position, 64)
        u, v = divmod(rem, 8)
        selected_rows.append((position, value))
        delta = np.zeros((8, 8, 3), np.float64)
        for selected_position, selected_value in selected_rows:
            selected_channel, selected_rem = divmod(selected_position, 64)
            selected_u, selected_v = divmod(selected_rem, 8)
            delta[:, :, selected_channel] += _BASIS[selected_u * 8 + selected_v] * (step * selected_value)
        rec = np.clip(np.rint(base64 + delta[:h, :w]), 0, 255).astype(np.int32)
        sse = int(np.square(source64 - rec, dtype=np.int64).sum())
        options.append((sse, sorted(selected_rows)))
    return options


def _block_param_length(rows: list[tuple[int, int]]) -> int:
    return 0 if not rows else 1 + sum(len(_coefficient_record(position, value)) for position, value in rows)


def _sparse_cell(source_cell: np.ndarray, base_cell: np.ndarray, block_options: list[list[tuple[int, list[tuple[int, int]]]]], lam: float, step: int) -> tuple[int, bytes, np.ndarray, int, list[tuple[int, int]]]:
    chosen = []
    for options in block_options:
        best = min(range(len(options)), key=lambda i: (options[i][0] + lam * 8 * _block_param_length(options[i][1]), i))
        chosen.append(options[best][1])
    mask = sum((1 << i) for i, rows in enumerate(chosen) if rows)
    if not mask:
        return int(np.square(source_cell.astype(np.int32) - base_cell.astype(np.int32), dtype=np.int64).sum()), b'', base_cell.copy(), 0, []
    h, w = source_cell.shape[:2]
    correction = np.zeros((CELL, CELL, 3), np.float64)
    positions = []
    param = bytearray((mask,))
    active_blocks = coeff_count = 0
    for block_index, rows in enumerate(chosen):
        if not rows:
            continue
        active_blocks += 1
        coeff = np.zeros((8, 8, 3), np.float64)
        param.append(len(rows))
        for position, value in rows:
            channel, rem = divmod(position, 64)
            u, v = divmod(rem, 8)
            coeff[u, v, channel] = value
            param.extend(_coefficient_record(position, value))
            positions.append((u, v))
            coeff_count += 1
        delta = idctn(coeff * step, axes=(0, 1), type=2, norm='ortho', workers=1)
        by, bx = divmod(block_index, 2)
        vh, vw = max(0, min(8, h - by * 8)), max(0, min(8, w - bx * 8))
        correction[by * 8:by * 8 + vh, bx * 8:bx * 8 + vw] = delta[:vh, :vw]
    rec = np.clip(np.rint(base_cell.astype(np.float64) + correction[:h, :w]), 0, 255).astype(np.uint8)
    err = source_cell.astype(np.int32) - rec.astype(np.int32)
    return int(np.square(err, dtype=np.int64).sum()), bytes(param), rec, active_blocks, positions


def _emit(result: dict, width: int, height: int, qstep: int) -> bytes:
    modes = result['modes']
    params = bytes(result['params'])
    raw = pack_modes(modes) + params
    if len(raw) > MAX_RAW:
        raise ValueError('raw enhancement exceeds limit')
    compressed = lzma.compress(raw, format=lzma.FORMAT_XZ, preset=6)
    header = HEADER.pack(MAGIC, 1, width, height, CELL, DC_STEP, DETAIL_STEP, qstep, MODE_BITS,
                         len(modes), len(raw), len(compressed))
    return header + compressed


def encode_family(source: np.ndarray, base: np.ndarray, config: str, lambdas=(0.0, 20.0, 100.0), qstep: int = 0) -> list[dict]:
    if config not in ('A', 'B', 'S') or source.dtype != np.uint8 or base.dtype != np.uint8 or source.shape != base.shape or source.ndim != 3 or source.shape[2] != 3:
        raise ValueError('expected config A/B/S and matching uint8 RGB arrays')
    if source.size == 0 or any(not math.isfinite(lam) or lam < 0 for lam in lambdas):
        raise ValueError('invalid image dimensions or lambda')
    if (config == 'S') != (qstep in QUANT_STEPS):
        raise ValueError('sparse transform requires qstep 16, 32, or 64; controls use 0')
    height, width = source.shape[:2]
    n = math.ceil(height / CELL) * math.ceil(width / CELL)
    outputs = [{'lambda': float(lam), 'modes': np.zeros(n, np.uint8), 'params': bytearray(), 'reconstruction': np.empty_like(base),
                'mode_counts': [0, 0, 0], 'active_blocks': 0, 'coefficients': 0, 'frequencies': [0] * 15,
                'rate_bits': 0} for lam in lambdas]
    start = time.perf_counter()
    for cell_index in range(n):
        y, x, ch, cw = cell_slices(height, width, cell_index)
        src = source[y:y + ch, x:x + cw]
        b = base[y:y + ch, x:x + cw]
        target = src.astype(np.int16) - b.astype(np.int16)
        skip_sse = int(np.square(target.astype(np.int32), dtype=np.int64).sum())
        dc = qvector(target)
        dc_rec = np.clip(b.astype(np.int16) + dc.astype(np.int16)[None, None, :] * DC_STEP, 0, 255).astype(np.uint8)
        dc_sse = int(np.square(src.astype(np.int32) - dc_rec.astype(np.int32), dtype=np.int64).sum())
        dc_param = dc.tobytes()
        detail_param = b''
        detail_rec = None
        detail_sse = None
        if config == 'B':
            units = detail_units(target)
            correction = interpolate_grid(units)[:ch, :cw]
            detail_rec = np.clip(b.astype(np.int32) + correction.astype(np.int32), 0, 255).astype(np.uint8)
            detail_sse = int(np.square(src.astype(np.int32) - detail_rec.astype(np.int32), dtype=np.int64).sum())
            detail_param = units.tobytes()
        block_options = None
        if config == 'S':
            block_options = []
            for by in range(2):
                for bx in range(2):
                    sy, sx = by * 8, bx * 8
                    vh, vw = min(8, ch - sy), min(8, cw - sx)
                    if vh <= 0 or vw <= 0:
                        block_options.append([(0, [])])
                    else:
                        block_options.append(_block_options(src[sy:sy + vh, sx:sx + vw], b[sy:sy + vh, sx:sx + vw], qstep))
        for output in outputs:
            lam = output['lambda']
            candidates = [(skip_sse + lam * MODE_BITS, 0, b'', b.copy(), 0, [])]
            candidates.append((dc_sse + lam * (MODE_BITS + len(dc_param) * 8), 1, dc_param, dc_rec, 0, []))
            if config == 'B':
                candidates.append((detail_sse + lam * (MODE_BITS + len(detail_param) * 8), 4, detail_param, detail_rec, 0, []))
            elif config == 'S':
                ss, sp, sr, active, positions = _sparse_cell(src, b, block_options, lam, qstep)
                if sp:
                    candidates.append((ss + lam * (MODE_BITS + len(sp) * 8), 2, sp, sr, active, positions))
            _, mode, param, rec, active, positions = min(candidates, key=lambda item: (item[0], item[1]))
            output['modes'][cell_index] = mode
            output['params'].extend(param)
            output['reconstruction'][y:y + ch, x:x + cw] = rec
            output['mode_counts'][{0: 0, 1: 1, 2: 2, 4: 2}[mode]] += 1
            output['rate_bits'] += MODE_BITS + 8 * len(param)
            output['active_blocks'] += active
            output['coefficients'] += len(positions)
            for u, v in positions:
                output['frequencies'][u + v] += 1
    analysis_seconds = time.perf_counter() - start
    results = []
    for output in outputs:
        stream = _emit(output, width, height, qstep)
        result = dict(output)
        result['params'] = bytes(result['params'])
        result['data'] = stream
        result['encode_seconds'] = analysis_seconds / len(outputs)
        result['family_analysis_seconds'] = analysis_seconds
        result['mode_bytes'] = math.ceil(n * MODE_BITS / 8)
        result['parameter_bytes'] = len(result['params'])
        result['raw_bytes'] = result['mode_bytes'] + result['parameter_bytes']
        result['xz_bytes'] = len(stream) - HEADER.size
        result['header_bytes'] = HEADER.size
        result['active_cells'] = int(np.count_nonzero(result['modes']))
        results.append(result)
    return results


def decode(data: bytes, base: np.ndarray, profile: dict | None = None) -> tuple[np.ndarray, np.ndarray, bytes]:
    total_start = time.perf_counter()
    if len(data) < HEADER.size:
        raise ValueError('short P33S header')
    magic, version, width, height, cell, dc_step, detail_step, qstep, mode_bits, n, raw_len, xz_len = HEADER.unpack_from(data)
    if (magic != MAGIC or version != 1 or cell != CELL or dc_step != DC_STEP or detail_step != DETAIL_STEP or
        qstep not in (0,) + QUANT_STEPS or mode_bits != MODE_BITS or width < 1 or height < 1 or
        width > 16384 or height > 16384 or width * height > MAX_PIXELS or
        n != math.ceil(height / CELL) * math.ceil(width / CELL) or base.shape != (height, width, 3) or
        base.dtype != np.uint8 or xz_len > MAX_XZ or len(data) != HEADER.size + xz_len or
        raw_len > MAX_RAW or raw_len < math.ceil(n * MODE_BITS / 8)):
        raise ValueError('invalid P33S header or base dimensions')
    entropy_start = time.perf_counter()
    decoder = lzma.LZMADecompressor(format=lzma.FORMAT_XZ, memlimit=64 * 1024 * 1024)
    raw = decoder.decompress(data[HEADER.size:], max_length=raw_len + 1)
    if not decoder.eof or decoder.unused_data or len(raw) != raw_len:
        raise ValueError('invalid, truncated, trailing, or oversized XZ payload')
    entropy_seconds = time.perf_counter() - entropy_start
    mode_len = math.ceil(n * MODE_BITS / 8)
    modes = unpack_modes(raw[:mode_len], n)
    if (qstep == 0 and np.any(modes == 2)) or (qstep != 0 and np.any(modes == 4)):
        raise ValueError('mode is incompatible with stream quantizer')
    params = raw[mode_len:]
    out = base.copy()
    cursor = 0
    sparse_seconds = correction_seconds = 0.0
    height, width = base.shape[:2]
    for index, mode_v in enumerate(modes):
        mode = int(mode_v)
        y, x, ch, cw = cell_slices(height, width, index)
        if mode == 0:
            continue
        if mode == 1:
            if cursor + 3 > len(params):
                raise ValueError('truncated DC parameters')
            vec = np.frombuffer(params[cursor:cursor + 3], dtype=np.int8).astype(np.int16)
            cursor += 3
            start = time.perf_counter()
            out[y:y + ch, x:x + cw] = np.clip(base[y:y + ch, x:x + cw].astype(np.int16) + vec[None, None, :] * DC_STEP, 0, 255).astype(np.uint8)
            correction_seconds += time.perf_counter() - start
        elif mode == 2:
            if qstep == 0 or cursor >= len(params):
                raise ValueError('sparse mode without mask')
            mask = params[cursor]
            cursor += 1
            if not mask or mask > 15:
                raise ValueError('invalid active-block mask')
            for block_index in range(4):
                if not (mask & (1 << block_index)):
                    continue
                if cursor >= len(params):
                    raise ValueError('truncated sparse block count')
                count = params[cursor]
                cursor += 1
                if not 1 <= count <= MAX_BLOCKS:
                    raise ValueError('invalid sparse coefficient count')
                coeff = np.zeros((8, 8, 3), np.float64)
                seen = set()
                for _ in range(count):
                    if cursor >= len(params):
                        raise ValueError('truncated coefficient position')
                    position = params[cursor]
                    cursor += 1
                    if position >= 192 or position in seen:
                        raise ValueError('invalid or duplicate coefficient position')
                    seen.add(position)
                    encoded, cursor = _read_varint(params, cursor)
                    value = (encoded >> 1) ^ -(encoded & 1)
                    if value == 0 or not -32768 <= value <= 32767:
                        raise ValueError('invalid zero or out-of-range coefficient')
                    channel, rem = divmod(position, 64)
                    u, v = divmod(rem, 8)
                    coeff[u, v, channel] = value
                by, bx = divmod(block_index, 2)
                sy, sx = y + by * 8, x + bx * 8
                vh, vw = min(8, height - sy), min(8, width - sx)
                stage_start = time.perf_counter()
                delta = idctn(coeff * qstep, axes=(0, 1), type=2, norm='ortho', workers=1)
                sparse_seconds += time.perf_counter() - stage_start
                stage_start = time.perf_counter()
                out[sy:sy + vh, sx:sx + vw] = np.clip(np.rint(base[sy:sy + vh, sx:sx + vw].astype(np.float64) + delta[:vh, :vw]), 0, 255).astype(np.uint8)
                correction_seconds += time.perf_counter() - stage_start
        elif mode == 4:
            if cursor + 48 > len(params) or qstep != 0:
                raise ValueError('invalid interpolated detail parameters')
            units = np.frombuffer(params[cursor:cursor + 48], dtype=np.int8).reshape(4, 4, 3)
            cursor += 48
            start = time.perf_counter()
            correction = interpolate_grid(units)[:ch, :cw]
            out[y:y + ch, x:x + cw] = np.clip(base[y:y + ch, x:x + cw].astype(np.int32) + correction.astype(np.int32), 0, 255).astype(np.uint8)
            correction_seconds += time.perf_counter() - start
    if cursor != len(params):
        raise ValueError('trailing enhancement parameters')
    if profile is not None:
        profile.update({'entropy_decode_seconds': entropy_seconds, 'sparse_idct_seconds': sparse_seconds,
                        'correction_seconds': correction_seconds, 'total_decode_seconds': time.perf_counter() - total_start})
    return out, modes, bytes(params)
