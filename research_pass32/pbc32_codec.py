from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import math
import struct
import time
from pathlib import Path

import numpy as np

CELL = 16
DC_STEP = 8
DETAIL_STEP = 16
MODE_BITS = 3
HEADER = struct.Struct('<4sBIIHBBBBIII')
MAGIC = b'P31R'
MASKS = {'A': 0b0000011, 'B': 0b0010011, 'C': 0b1101111, 'D0': 0b0011111, 'D1': 0b1110011, 'D': 0b1111111}
RATE_BITS = {0: 3, 1: 27, 2: 3, 3: 3, 4: 387, 5: 27, 6: 27}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def round_div_away(values: np.ndarray, divisor: int) -> np.ndarray:
    a = np.asarray(values, np.int64)
    magnitude = np.abs(a)
    rounded = (magnitude + divisor // 2) // divisor
    return np.where(a < 0, -rounded, rounded)


def pack_bits(values: np.ndarray, width: int) -> bytes:
    out = bytearray()
    acc = 0
    bits = 0
    for value in np.asarray(values, np.uint8).reshape(-1):
        if int(value) >= (1 << width):
            raise ValueError('symbol outside bit width')
        acc = (acc << width) | int(value)
        bits += width
        while bits >= 8:
            bits -= 8
            out.append((acc >> bits) & 255)
            acc &= (1 << bits) - 1 if bits else 0
    if bits:
        out.append((acc << (8 - bits)) & 255)
    return bytes(out)


def unpack_bits(data: bytes, count: int, width: int) -> np.ndarray:
    out = np.empty(count, np.uint8)
    acc = bits = pos = 0
    for i in range(count):
        while bits < width:
            if pos >= len(data):
                raise ValueError('truncated symbol stream')
            acc = (acc << 8) | data[pos]
            pos += 1
            bits += 8
        bits -= width
        out[i] = (acc >> bits) & ((1 << width) - 1)
        acc &= (1 << bits) - 1 if bits else 0
    if pos != len(data) or acc:
        raise ValueError('bad symbol padding or length')
    return out


def qvector(values: np.ndarray, step: int) -> np.ndarray:
    v = np.asarray(values, np.int64).reshape(-1, 3)
    levels = round_div_away(v.sum(0), v.shape[0] * step)
    return np.clip(levels, -128, 127).astype(np.int8)


def detail_units(residual: np.ndarray, step: int) -> np.ndarray:
    h, w = residual.shape[:2]
    if h == CELL and w == CELL:
        blocks = residual.reshape(4, 4, 4, 4, 3).astype(np.int64)
        levels = round_div_away(blocks.sum((1, 3)), 16 * step)
        return np.clip(levels, -128, 127).astype(np.int8)
    grid = np.empty((4, 4, 3), np.int8)
    for gy in range(4):
        for gx in range(4):
            y0, x0 = gy * 4, gx * 4
            block = residual[y0:min(y0 + 4, h), x0:min(x0 + 4, w)].astype(np.int64)
            if block.size:
                levels = round_div_away(block.sum((0, 1)), block.shape[0] * block.shape[1] * step)
                grid[gy, gx] = np.clip(levels, -128, 127).astype(np.int8)
            else:
                grid[gy, gx] = grid[min(gy, max(0, math.ceil(h / 4) - 1)), min(gx, max(0, math.ceil(w / 4) - 1))]
    return grid


def interpolate_grid(units: np.ndarray, step: int) -> np.ndarray:
    grid = np.asarray(units, np.int16).astype(np.int32) * step
    coord = np.arange(CELL)
    y0, fy = coord // 5, coord % 5
    x0, fx = coord // 5, coord % 5
    y1, x1 = np.minimum(y0 + 1, 3), np.minimum(x0 + 1, 3)
    fy2=(5-fy)[:,None,None]; fx2=(5-fx)[None,:,None]
    fy1=fy[:,None,None]; fx1=fx[None,:,None]
    total=(grid[y0[:,None],x0[None,:]]*fy2*fx2 + grid[y0[:,None],x1[None,:]]*fy2*fx1 +
           grid[y1[:,None],x0[None,:]]*fy1*fx2 + grid[y1[:,None],x1[None,:]]*fy1*fx1)
    return round_div_away(total,25).astype(np.int16)


def cell_sse(source: np.ndarray, base: np.ndarray, residual: np.ndarray) -> tuple[int, np.ndarray]:
    rec = np.clip(base.astype(np.int32) + residual.astype(np.int32), 0, 255).astype(np.uint8)
    err = source.astype(np.int32) - rec.astype(np.int32)
    return int(np.square(err, dtype=np.int64).sum()), rec


def cell_slices(height: int, width: int, index: int) -> tuple[int, int, int, int]:
    cols = math.ceil(width / CELL)
    cy, cx = divmod(index, cols)
    y, x = cy * CELL, cx * CELL
    return y, x, min(CELL, height - y), min(CELL, width - x)


def encode(source: np.ndarray, base: np.ndarray, config: str, lam: float, rate_estimator: str = 'R0', rate_model=None) -> dict:
    if source.dtype != np.uint8 or base.dtype != np.uint8 or source.shape != base.shape or source.ndim != 3 or source.shape[2] != 3:
        raise ValueError('source and base must be matching uint8 RGB arrays')
    if not math.isfinite(lam) or lam < 0:
        raise ValueError('lambda must be finite and nonnegative')
    if rate_estimator not in ('R0', 'R1') or (rate_estimator == 'R1' and rate_model is None):
        raise ValueError('R1 requires a frozen rate model')
    h, w = source.shape[:2]
    n = math.ceil(h / CELL) * math.ceil(w / CELL)
    mask = MASKS[config]
    modes = np.zeros(n, np.uint8)
    params = bytearray()
    chosen_sse = np.zeros(n, np.int64)
    skip_sse = np.zeros(n, np.int64)
    best_direction_sse = np.full(n, -1, np.int64)
    above = np.zeros((w, 3), np.int16)
    left = np.zeros((h, 3), np.int16)
    recon = np.empty_like(base)
    estimated_mode_bits = 0.0
    estimated_parameter_bits = 0.0
    previous_mode = 7
    mode_counts = np.zeros(7, np.int64)
    direction_gain_sse = 0
    start = time.perf_counter()

    def evaluate(src_cell: np.ndarray, base_cell: np.ndarray, correction: np.ndarray, mode: int,
                 param: bytes, context: int) -> tuple[float, np.ndarray, int, bytes, float, float]:
        sse, pixels = cell_sse(src_cell, base_cell, correction)
        if rate_estimator == 'R0':
            mode_bits = 3.0
            parameter_bits = float(RATE_BITS[mode] - 3)
        else:
            mode_bits, parameter_bits = rate_model.cost_parts(mode, previous_mode, context, param)
        return sse + lam * (mode_bits + parameter_bits), pixels, sse, param, mode_bits, parameter_bits

    for i in range(n):
        y, x, ch, cw = cell_slices(h, w, i)
        src_cell = source[y:y + ch, x:x + cw]
        base_cell = base[y:y + ch, x:x + cw]
        target = src_cell.astype(np.int16) - base_cell.astype(np.int16)
        zero = np.zeros((ch, cw, 3), np.int16)
        cy, cx = divmod(i, math.ceil(w / CELL))
        context = 0 if cy == 0 and cx == 0 else 1 if cy == 0 else 2 if cx == 0 else 3
        skip_obj = evaluate(src_cell, base_cell, zero, 0, b'', context)
        skip_sse[i] = skip_obj[2]
        best = (0, skip_obj)

        if mask & (1 << 1):
            vec = qvector(target, DC_STEP)
            correction = np.broadcast_to(vec.astype(np.int16) * DC_STEP, (ch, cw, 3))
            best_dc = evaluate(src_cell, base_cell, correction, 1, vec.tobytes(), context)
            if best_dc[0] < best[1][0]:
                best = (1, best_dc)

        predictor_specs = []
        if cy and mask & ((1 << 2) | (1 << 5)):
            pred_line = above[x:x + cw]
            predictor_specs.append((2, np.broadcast_to(pred_line[None], (ch, cw, 3))))
        if cx and mask & ((1 << 3) | (1 << 6)):
            pred_line = left[y:y + ch]
            predictor_specs.append((3, np.broadcast_to(pred_line[:, None], (ch, cw, 3))))
        local_best_direction = None
        for mode, pred in predictor_specs:
            pred_i32 = pred.astype(np.int32)
            offset_mode = mode + 3
            if mask & (1 << mode):
                obj = evaluate(src_cell, base_cell, pred_i32, mode, b'', context)
                if local_best_direction is None or obj[2] < local_best_direction:
                    local_best_direction = obj[2]
                if obj[0] < best[1][0]:
                    best = (mode, obj)
            if mask & (1 << offset_mode):
                offset = qvector(target.astype(np.int32) - pred_i32, DC_STEP)
                correction = pred_i32 + offset.astype(np.int32)[None, None, :] * DC_STEP
                obj = evaluate(src_cell, base_cell, correction, offset_mode, offset.tobytes(), context)
                if local_best_direction is None or obj[2] < local_best_direction:
                    local_best_direction = obj[2]
                if obj[0] < best[1][0]:
                    best = (offset_mode, obj)
        if local_best_direction is not None:
            best_direction_sse[i] = local_best_direction
            direction_gain_sse += max(0, int(skip_sse[i]) - local_best_direction)

        if mask & (1 << 4):
            units = detail_units(target, DETAIL_STEP)
            correction = interpolate_grid(units, DETAIL_STEP)[:ch, :cw]
            detail_obj = evaluate(src_cell, base_cell, correction, 4, units.tobytes(), context)
            if detail_obj[0] < best[1][0]:
                best = (4, detail_obj)

        mode, obj = best
        modes[i] = mode
        chosen_sse[i] = obj[2]
        mode_counts[mode] += 1
        estimated_mode_bits += obj[4]
        estimated_parameter_bits += obj[5]
        params.extend(obj[3])
        previous_mode = mode
        recon[y:y + ch, x:x + cw] = obj[1]
        actual_residual = obj[1].astype(np.int16) - base_cell.astype(np.int16)
        above[x:x + cw] = actual_residual[-1]
        left[y:y + ch] = actual_residual[:, -1]

    raw_modes = pack_bits(modes, MODE_BITS)
    raw = raw_modes + params
    entropy = lzma.compress(raw, format=lzma.FORMAT_XZ, preset=6)
    header = HEADER.pack(MAGIC, 1, w, h, CELL, DC_STEP, DETAIL_STEP, mask, MODE_BITS, n, len(raw), len(entropy))
    data = header + entropy
    framing_bits = 8 * (HEADER.size + len(lzma.compress(b'', format=lzma.FORMAT_XZ, preset=6))) if rate_estimator == 'R1' else 0
    estimated_payload_bits = estimated_mode_bits + estimated_parameter_bits
    return {'data': data, 'modes': modes, 'params': bytes(params), 'reconstruction': recon,
            'rate_estimator': rate_estimator, 'estimated_bits': int(round(estimated_payload_bits + framing_bits)),
            'estimated_payload_bits': float(estimated_payload_bits), 'estimated_mode_bits': float(estimated_mode_bits),
            'estimated_parameter_bits': float(estimated_parameter_bits), 'estimated_framing_bits': int(framing_bits),
            'mode_counts': mode_counts.tolist(),
            'chosen_sse': chosen_sse, 'skip_sse': skip_sse, 'best_direction_sse': best_direction_sse,
            'direction_gain_sse': int(direction_gain_sse), 'encode_seconds': time.perf_counter() - start,
            'raw_payload_bytes': len(raw), 'entropy_payload_bytes': len(entropy), 'header_bytes': len(header)}


def decode(data: bytes, base: np.ndarray, profile: dict | None = None) -> tuple[np.ndarray, np.ndarray, bytes]:
    total_start = time.perf_counter()
    if len(data) < HEADER.size:
        raise ValueError('short P31R header')
    magic, version, width, height, cell, qdc, qdetail, mask, mode_bits, n, raw_len, xz_len = HEADER.unpack_from(data)
    if (magic != MAGIC or version != 1 or cell != CELL or qdc != DC_STEP or qdetail != DETAIL_STEP or
        mask not in MASKS.values() or mode_bits != MODE_BITS or width < 1 or height < 1 or
        width > 16384 or height > 16384 or width * height > 50_000_000 or
        n != math.ceil(height / CELL) * math.ceil(width / CELL) or base.shape != (height, width, 3) or
        xz_len > 64 * 1024 * 1024 or len(data) != HEADER.size + xz_len):
        raise ValueError('invalid P31R header or base dimensions')
    mode_len = math.ceil(n * mode_bits / 8)
    if raw_len < mode_len or raw_len > mode_len + 48 * n or raw_len > 64 * 1024 * 1024:
        raise ValueError('P31R payload exceeds resource limit')
    entropy_start = time.perf_counter()
    dec = lzma.LZMADecompressor(format=lzma.FORMAT_XZ, memlimit=64 * 1024 * 1024)
    raw = dec.decompress(data[HEADER.size:], max_length=raw_len + 1)
    if not dec.eof or dec.unused_data or len(raw) != raw_len:
        raise ValueError('invalid, truncated, trailing, or oversized XZ payload')
    modes = unpack_bits(raw[:mode_len], n, mode_bits)
    if np.any(modes > 6) or any(not (mask & (1 << int(mode))) for mode in modes):
        raise ValueError('mode not permitted by header')
    entropy_seconds = time.perf_counter() - entropy_start
    cell_start = time.perf_counter()
    params = raw[mode_len:]
    cursor = 0
    out = np.empty_like(base)
    h, w = height, width
    above = np.zeros((w, 3), np.int16)
    left = np.zeros((h, 3), np.int16)

    def read_dc() -> np.ndarray:
        nonlocal cursor
        if cursor + 3 > len(params):
            raise ValueError('truncated DC parameters')
        value = np.frombuffer(params[cursor:cursor + 3], np.int8).astype(np.int16) * qdc
        cursor += 3
        return value

    for i, mode in enumerate(modes):
        y, x, ch, cw = cell_slices(h, w, i)
        base_cell = base[y:y + ch, x:x + cw]
        if mode == 0:
            residual = np.zeros((ch, cw, 3), np.int16)
        elif mode == 1:
            residual = np.broadcast_to(read_dc()[None, None, :], (ch, cw, 3))
        elif mode in (2, 3, 5, 6):
            cy, cx = divmod(i, math.ceil(w / CELL))
            if mode in (2, 5):
                if not cy:
                    raise ValueError('vertical predictor at top boundary')
                pred = np.broadcast_to(above[x:x + cw][None], (ch, cw, 3)).astype(np.int32)
            else:
                if not cx:
                    raise ValueError('horizontal predictor at left boundary')
                pred = np.broadcast_to(left[y:y + ch][:, None], (ch, cw, 3)).astype(np.int32)
            residual = pred if mode in (2,3) else pred + read_dc().astype(np.int32)[None,None,:]
        else:
            if cursor + 48 > len(params):
                raise ValueError('truncated detail grid')
            units = np.frombuffer(params[cursor:cursor + 48], np.int8).reshape(4, 4, 3).copy()
            cursor += 48
            residual = interpolate_grid(units, qdetail)[:ch, :cw]
        pixels = np.clip(base_cell.astype(np.int32) + residual.astype(np.int32), 0, 255).astype(np.uint8)
        out[y:y + ch, x:x + cw] = pixels
        actual_residual = pixels.astype(np.int16) - base_cell.astype(np.int16)
        above[x:x + cw] = actual_residual[-1]
        left[y:y + ch] = actual_residual[:, -1]
    if cursor != len(params):
        raise ValueError('unexpected trailing parameters')
    if profile is not None:
        cell_seconds = time.perf_counter() - cell_start
        total_seconds = time.perf_counter() - total_start
        profile.update({'entropy_decode_seconds': entropy_seconds, 'cell_reconstruction_seconds': cell_seconds,
                        'other_parse_seconds': max(0.0, total_seconds - entropy_seconds - cell_seconds),
                        'total_decode_seconds': total_seconds})
    return out, modes, params


def decode_main(args: argparse.Namespace) -> None:
    base = np.load(args.base, allow_pickle=False)
    data = Path(args.enhancement).read_bytes()
    image, modes, params = decode(data, base)
    np.save(args.output, image, allow_pickle=False)
    print(json.dumps({'pixels_sha256': sha(image.tobytes()), 'modes_sha256': sha(modes.tobytes()),
                      'params_sha256': sha(params), 'mode_counts': np.bincount(modes, minlength=7).tolist()}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--decode', action='store_true')
    parser.add_argument('--enhancement')
    parser.add_argument('--base')
    parser.add_argument('--output')
    args = parser.parse_args()
    if args.decode:
        decode_main(args)
