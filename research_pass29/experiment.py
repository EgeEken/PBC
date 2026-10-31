from __future__ import annotations

import argparse
import platform
import subprocess
import sys
import hashlib
import json
import lzma
import math
import os
import struct
import time
import tracemalloc
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"

import numba
import numpy as np
from PIL import Image, ImageDraw, ImageOps, __version__ as PILLOW_VERSION

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PBC3 import PBC3
from pbc3_types import PBC3Config
OUT = Path(__file__).resolve().parent / "outputs"
VISUALS = Path(__file__).resolve().parent / "visuals"
IMAGES = ("googlewiki.png", "mario64.png", "flower.jpg", "cat.jpg")
CELL = 16
BUDGETS = (1, 16, 64, 256)
MAX_PALETTE_SAMPLES = 65536
PROXY_SAMPLE_CELLS = 2048
MASK64 = (1 << 64) - 1
HEADER = "<4sBBBBHHHII"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@numba.njit(cache=True)
def _permutation_numba(n: int, seed: int, group: int) -> np.ndarray:
    values = np.arange(n, dtype=np.int32)
    state = np.uint64(((seed + 1) * 0x9E3779B97F4A7C15 + (group + 1) * 0xD1B54A32D192ED03) & MASK64)
    if state == 0:
        state = np.uint64(0xA5A5A5A5A5A5A5A5)
    for i in range(n - 1, 0, -1):
        state ^= state >> 12
        state ^= state << 25
        state ^= state >> 27
        word = state * np.uint64(0x2545F4914F6CDD1D)
        j = int(word % np.uint64(i + 1))
        values[i], values[j] = values[j], values[i]
    return values


def permutation(n: int, seed: int, group: int) -> np.ndarray:
    return _permutation_numba(n, seed, group)


def pack2(values: np.ndarray) -> bytes:
    v = np.asarray(values, np.uint8).reshape(-1)
    out = np.zeros((len(v) + 3) // 4, np.uint8)
    for shift in range(4):
        a = v[shift::4]
        out[:len(a)] |= (a & 3) << (6 - 2 * shift)
    return out.tobytes()


def pack4(values: np.ndarray) -> bytes:
    v = np.asarray(values, np.uint8).reshape(-1)
    out = np.zeros((len(v) + 1) // 2, np.uint8)
    out[:] = (v[0::2] & 15) << 4
    if len(v) > 1:
        out[:len(v[1::2])] |= v[1::2] & 15
    return out.tobytes()


def unpack4(data: bytes, count: int) -> np.ndarray:
    src = np.frombuffer(data, np.uint8)
    out = np.empty(count, np.uint8)
    out[0::2] = src[:len(out[0::2])] >> 4
    out[1::2] = src[:len(out[1::2])] & 15
    return out


def unpack2(data: bytes, count: int) -> np.ndarray:
    src = np.frombuffer(data, np.uint8)
    out = np.empty(count, np.uint8)
    for shift in range(4):
        dst = out[shift::4]
        dst[:] = (src[:len(dst)] >> (6 - 2 * shift)) & 3
    return out


def pack_small(value: int, bits: int) -> bytes:
    if not bits:
        return b""
    return bytes([value << (8 - bits)])


def unpack_small(data: bytes, bits: int) -> int:
    return data[0] >> (8 - bits) if bits else 0


def cells_of(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    h, w, _ = image.shape
    rows, cols = math.ceil(h / CELL), math.ceil(w / CELL)
    cells = np.zeros((rows * cols, CELL, CELL, 3), np.int16)
    masks = np.zeros((rows * cols, CELL, CELL), bool)
    for cy in range(rows):
        for cx in range(cols):
            i, y, x = cy * cols + cx, cy * CELL, cx * CELL
            ch, cw = min(CELL, h - y), min(CELL, w - x)
            cells[i, :ch, :cw] = image[y:y + ch, x:x + cw]
            masks[i, :ch, :cw] = True
    return cells, masks, (rows, cols)


def features_of(residual: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cells, masks, _ = cells_of(residual)
    distribution, structure = [], []
    edges = np.linspace(-384, 384, 9)
    for cell, mask in zip(cells, masks):
        pix = cell[mask].astype(np.float32)
        hist, _ = np.histogram(pix.mean(1), bins=edges)
        hist = hist.astype(np.float32) / len(pix)
        distribution.append(np.r_[pix.mean(0), pix.var(0), hist])
        ch, cw = int(mask.any(1).sum()), int(mask.any(0).sum())
        core = cell[:ch, :cw].astype(np.float32)
        gx = float(np.abs(np.diff(core, axis=1)).mean()) if cw > 1 else 0.0
        gy = float(np.abs(np.diff(core, axis=0)).mean()) if ch > 1 else 0.0
        structure.append((gx, gy, math.hypot(gx, gy)))
    return np.asarray(distribution, np.float64), np.asarray(structure, np.float64)


def cluster(x: np.ndarray, k: int = 4) -> np.ndarray:
    x = np.asarray(x, np.float64)
    scale = x.std(0)
    scale[scale < 1e-9] = 1
    z = (x - x.mean(0)) / scale
    centers = [0]
    dmin = ((z - z[0]) ** 2).sum(1)
    for _ in range(1, k):
        j = int(np.argmax(dmin))
        centers.append(j)
        dmin = np.minimum(dmin, ((z - z[j]) ** 2).sum(1))
    c = z[centers].copy()
    for _ in range(24):
        labels = ((z[:, None] - c[None]) ** 2).sum(2).argmin(1)
        new = c.copy()
        for g in range(k):
            if np.any(labels == g):
                new[g] = z[labels == g].mean(0)
        if np.array_equal(new, c):
            break
        c = new
    labels = ((z[:, None] - c[None]) ** 2).sum(2).argmin(1)
    order = sorted(range(k), key=lambda g: tuple(np.round(c[g], 12)))
    remap = np.empty(k, np.uint8)
    for new, old in enumerate(order):
        remap[old] = new
    return remap[labels]


def palette_kmeans(pixels: np.ndarray, k: int) -> np.ndarray:
    x = np.asarray(pixels, np.float64).reshape(-1, 3)
    if len(x) > MAX_PALETTE_SAMPLES:
        x = x[::math.ceil(len(x) / MAX_PALETTE_SAMPLES)]
    if not len(x):
        return np.zeros((k, 3), np.int16)
    centers = [int(np.argmin(((x - x.mean(0)) ** 2).sum(1)))]
    near = ((x - x[centers[0]]) ** 2).sum(1)
    for _ in range(1, k):
        j = int(np.argmax(near))
        centers.append(j)
        near = np.minimum(near, ((x - x[j]) ** 2).sum(1))
    c = x[centers].copy()
    for _ in range(20):
        labels = ((x[:, None] - c[None]) ** 2).sum(2).argmin(1)
        new = c.copy()
        for g in range(k):
            if np.any(labels == g):
                new[g] = x[labels == g].mean(0)
        if np.array_equal(new, c):
            break
        c = new
    return np.clip(np.rint(c), -32768, 32767).astype(np.int16)


def grouped_palette(residual: np.ndarray, assign: np.ndarray) -> np.ndarray:
    cells, valid, _ = cells_of(residual)
    return np.stack([
        palette_kmeans(cells[assign == g][valid[assign == g]].reshape(-1, 3), 4)
        for g in range(4)
    ])


def palettes_for(residual: np.ndarray, assign: np.ndarray) -> dict[str, np.ndarray]:
    cells, valid, _ = cells_of(residual)
    return {"global4": palette_kmeans(cells[valid], 4),
            "global16": palette_kmeans(cells[valid], 16),
            "group4": grouped_palette(residual, assign)}


def quantize(residual: np.ndarray, palette: np.ndarray, assignment: np.ndarray | None) -> np.ndarray:
    cells, _, _ = cells_of(residual)
    out = np.empty((len(cells), CELL * CELL), np.uint8)
    for i, cell in enumerate(cells):
        p = palette if palette.ndim == 2 else palette[int(assignment[i])]
        d = ((cell.reshape(-1, 3).astype(np.int32)[:, None] - p.astype(np.int32)[None]) ** 2).sum(2)
        out[i] = d.argmin(1)
    return out


def reconstruct(base: np.ndarray, indices: np.ndarray, palette: np.ndarray,
                assignment: np.ndarray | None) -> np.ndarray:
    h, w, _ = base.shape
    rows, cols = math.ceil(h / CELL), math.ceil(w / CELL)
    if palette.ndim == 3:
        cell_palettes = palette[assignment]
    else:
        cell_palettes = np.broadcast_to(palette, (len(indices),) + palette.shape)
    colors = np.take_along_axis(cell_palettes, indices.astype(np.intp)[:, :, None], axis=1)
    padded = colors.reshape(rows, cols, CELL, CELL, 3).transpose(0, 2, 1, 3, 4)
    correction = padded.reshape(rows * CELL, cols * CELL, 3)[:h, :w]
    return np.clip(base.astype(np.int16) + correction, 0, 255).astype(np.uint8)


def repeated_timing(fn, repeats: int = 3) -> dict:
    fn()
    samples = []
    for _ in range(repeats):
        t = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t)
    return {"mean": float(np.mean(samples)), "median": float(np.median(samples)), "samples": samples}


def distortion(src: np.ndarray, rec: np.ndarray) -> dict:
    err = (src.astype(np.float32) - rec.astype(np.float32)) ** 2
    lum = src.astype(np.float32).mean(2)
    gy, gx = np.gradient(lum)
    edge = np.hypot(gx, gy)
    return {"mse": float(err.mean()),
            "edge_top10_mse": float(err[edge >= np.quantile(edge, .9)].mean()),
            "smooth_bottom50_mse": float(err[edge <= np.quantile(edge, .5)].mean())}


def groupmap(assign: np.ndarray) -> bytes:
    return pack2(assign)


def ordering(indices: np.ndarray, assign: np.ndarray, strategy: int, seed: int,
             desc: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    if strategy == 3:
        return np.arange(len(indices), dtype=np.uint32), indices.reshape(-1)
    positions, chunks = [], []
    for g in range(4):
        ids = np.flatnonzero(assign == g)
        if strategy == 1:
            ids = ids[permutation(len(ids), seed, g)]
        elif strategy == 2:
            d = desc[ids]
            scale = d.std(0)
            scale[scale < 1e-9] = 1
            z = (d - d.mean(0)) / scale
            ids = ids[np.lexsort(tuple(z[:, j] for j in range(z.shape[1] - 1, -1, -1)))]
        positions.extend(ids.tolist())
        chunks.append(indices[ids].reshape(-1))
    return np.asarray(positions, np.uint32), np.concatenate(chunks)


def virtual_file(indices: np.ndarray, assign: np.ndarray, palette: np.ndarray,
                 width: int, height: int, strategy: int = 0, seed: int = 0,
                 seed_bits: int = 0, desc: np.ndarray | None = None) -> tuple[bytes, dict]:
    positions, ordered = ordering(indices, assign, strategy, seed, desc)
    index_bits = 2 if palette.ndim == 3 or len(palette) <= 4 else 4
    raw = pack2(ordered) if index_bits == 2 else pack4(ordered)
    entropy = lzma.compress(raw, format=lzma.FORMAT_XZ, preset=6)
    has_map = int(palette.ndim == 3 or strategy in (1, 2))
    flags = has_map | (2 if index_bits == 4 else 0)
    amap = groupmap(assign) if has_map else b""
    pbytes = np.asarray(palette, dtype="<i2").tobytes()
    omap = positions.astype("<u4").tobytes() if strategy == 2 else b""
    sbits = pack_small(seed, seed_bits)
    header = struct.pack(HEADER, b"P29V", 1, strategy, seed_bits, flags,
                         width, height, CELL, len(assign), len(palette.reshape(-1, 3)))
    data = header + sbits + amap + pbytes + omap + entropy
    return data, {"header": len(header), "seed_index_bits": seed_bits,
                  "seed_padding_bits": len(sbits) * 8 - seed_bits, "seed_signal": len(sbits),
                  "group_map": len(amap), "palette": len(pbytes), "order_map": len(omap),
                  "index_bits": index_bits, "index_packed_raw": len(raw), "xz_stream": len(entropy), "complete": len(data)}


def decompress_xz_exact(data: bytes) -> bytes:
    dec = lzma.LZMADecompressor(format=lzma.FORMAT_XZ)
    raw = dec.decompress(data)
    if not dec.eof or dec.unused_data:
        raise ValueError("invalid or trailing XZ data")
    return raw


def read_virtual(data: bytes) -> tuple[np.ndarray, dict]:
    nheader = struct.calcsize(HEADER)
    magic, ver, strategy, seed_bits, flags, width, height, cell, ncell, npal = struct.unpack(HEADER, data[:nheader])
    has_map, index_bits = flags & 1, (4 if flags & 2 else 2)
    if (magic != b"P29V" or ver != 1 or strategy not in (0, 1, 2, 3) or cell != CELL
            or flags & ~3 or seed_bits > 8 or width < 1 or height < 1 or ncell < 1
            or npal not in (4, 16) or (has_map and npal != 16)
            or (index_bits == 4 and npal != 16) or (strategy in (1, 2) and not has_map)):
        raise ValueError("bad P29V header")
    at = nheader
    seed_len = (seed_bits + 7) // 8
    seed = unpack_small(data[at:at + seed_len], seed_bits)
    if seed_bits and data[at] & ((1 << (8 - seed_bits)) - 1):
        raise ValueError("nonzero seed padding")
    at += seed_len
    mlen = (ncell + 3) // 4
    assign = unpack2(data[at:at + mlen], ncell) if has_map else np.zeros(ncell, np.uint8)
    at += mlen if has_map else 0
    pbytes = npal * 6
    palette = np.frombuffer(data[at:at + pbytes], dtype="<i2").reshape(-1, 3).copy()
    at += pbytes
    if has_map and npal == 16:
        palette = palette.reshape(4, 4, 3)
    assignment = assign if has_map else None
    rows, cols = math.ceil(height / CELL), math.ceil(width / CELL)
    if rows * cols != ncell:
        raise ValueError("dimension mismatch")
    if strategy == 2:
        olen = ncell * 4
        pos = np.frombuffer(data[at:at + olen], dtype="<u4").astype(np.int64)
        at += olen
        if len(np.unique(pos)) != ncell or np.any(pos >= ncell):
            raise ValueError("bad ordering map")
    else:
        pos = None
    packed = decompress_xz_exact(data[at:])
    ordered = (unpack2(packed, ncell * CELL * CELL) if index_bits == 2 else
               unpack4(packed, ncell * CELL * CELL)).reshape(ncell, -1)
    if strategy == 3:
        restored = ordered
        return restored, {"width": width, "height": height, "seed": seed,
                          "palette": palette, "assignments": assignment, "has_group_map": bool(has_map)}
    restored = np.empty_like(ordered)
    cursor = 0
    for g in range(4):
        raster = np.flatnonzero(assign == g)
        count = len(raster)
        if strategy == 1:
            dst = raster[permutation(count, seed, g)]
        elif strategy == 2:
            dst = pos[cursor:cursor + count]
            if np.any(assign[dst] != g):
                raise ValueError("cross-group mapping")
        else:
            dst = raster
        restored[dst] = ordered[cursor:cursor + count]
        cursor += count
    return restored, {"width": width, "height": height, "seed": seed,
                      "palette": palette, "assignments": assignment, "has_group_map": bool(has_map)}


def decode_enhancement(data: bytes, base: np.ndarray) -> np.ndarray:
    indices, metadata = read_virtual(data)
    if base.shape[:2] != (metadata["height"], metadata["width"]):
        raise ValueError("decoded base dimensions do not match enhancement")
    return reconstruct(base, indices, metadata["palette"], metadata["assignments"])


def template_file(indices: np.ndarray, assign: np.ndarray, palette: np.ndarray,
                  width: int, height: int) -> tuple[bytes, dict]:
    reps, delta = [], []
    for g in range(4):
        ids = np.flatnonzero(assign == g)
        ref = indices[ids[0]] if len(ids) else np.zeros(CELL * CELL, np.uint8)
        reps.append(pack2(ref))
        delta.append(((indices[ids].astype(np.int16) - ref[None]) & 3).astype(np.uint8).reshape(-1))
    dbytes = pack2(np.concatenate(delta))
    xz = lzma.compress(dbytes, format=lzma.FORMAT_XZ, preset=6)
    amap, pbytes, representatives = groupmap(assign), np.asarray(palette, dtype="<i2").tobytes(), b"".join(reps)
    header = struct.pack("<4sBHHHI", b"P29T", 1, width, height, CELL, len(assign))
    data = header + amap + pbytes + representatives + xz
    return data, {"header": len(header), "group_map": len(amap), "palette": len(pbytes),
                  "representatives": len(representatives), "delta_xz": len(xz), "complete": len(data)}


def read_template(data: bytes) -> np.ndarray:
    fmt = "<4sBHHHI"
    nh = struct.calcsize(fmt)
    magic, ver, w, h, cell, n = struct.unpack(fmt, data[:nh])
    if (magic != b"P29T" or ver != 1 or cell != CELL or w < 1 or h < 1 or n < 1
            or math.ceil(w / CELL) * math.ceil(h / CELL) != n):
        raise ValueError("bad P29T header")
    at = nh
    assign = unpack2(data[at:at + (n + 3) // 4], n)
    at += (n + 3) // 4 + 4 * 4 * 6
    reps = [unpack2(data[at + g * 64:at + (g + 1) * 64], CELL * CELL) for g in range(4)]
    at += 4 * 64
    delta = unpack2(decompress_xz_exact(data[at:]), n * CELL * CELL).reshape(n, -1)
    out = np.empty_like(delta)
    cursor = 0
    for g in range(4):
        ids = np.flatnonzero(assign == g)
        count = len(ids)
        out[ids] = (reps[g].astype(np.int16)[None] + delta[cursor:cursor + count]) & 3
        cursor += count
    return out


def decode_template(data: bytes, base: np.ndarray) -> np.ndarray:
    fmt = "<4sBHHHI"
    nh = struct.calcsize(fmt)
    magic, version, width, height, cell, ncell = struct.unpack(fmt, data[:nh])
    if magic != b"P29T" or version != 1 or cell != CELL or base.shape[:2] != (height, width):
        raise ValueError("bad P29T header or base dimensions")
    at = nh + (ncell + 3) // 4
    palette = np.frombuffer(data[at:at + 96], dtype="<i2").reshape(4, 4, 3).copy()
    assignments = unpack2(data[nh:nh + (ncell + 3) // 4], ncell)
    return reconstruct(base, read_template(data), palette, assignments)


def verify(data: bytes, expected: np.ndarray, strategy: int) -> None:
    got, _ = read_virtual(data)
    if not np.array_equal(got, expected):
        raise AssertionError(f"virtual round-trip {strategy}")
    for bad in (data[:5], data[:-3], data + b"trailing"):

        try:
            read_virtual(bad)
        except Exception:
            continue
        raise AssertionError("truncated stream accepted")


def encode_base(name: str) -> dict:
    path = OUT / "bases" / f"{name}.pbc3"
    path.parent.mkdir(parents=True, exist_ok=True)
    im = ImageOps.exif_transpose(Image.open(ROOT / "hpt_data" / name)).convert("RGB")
    src = np.asarray(im, np.uint8)
    elapsed = None
    if path.exists():
        stream = path.read_bytes()
    else:
        t = time.perf_counter()
        result = PBC3.compress(im, PBC3Config.quality(), rng_mode="release")
        elapsed = time.perf_counter() - t
        stream = result.data
        path.write_bytes(stream)
    base = np.asarray(PBC3.decompress(stream).image.convert("RGB"), np.uint8)
    if src.shape != base.shape:
        raise AssertionError("source and PBC base dimensions differ")
    return {"src": src, "base": base, "stream": stream, "source_sha256": sha((ROOT / "hpt_data" / name).read_bytes()), "encode_seconds": elapsed,
            "stream_bytes": len(stream), "stream_sha256": sha(stream)}


def proxy(indices: np.ndarray, assign: np.ndarray, desc: np.ndarray, seed: int) -> float:
    score = 0.0
    for g in range(4):
        ids = np.flatnonzero(assign == g)
        path = ids[permutation(len(ids), seed, g)]
        if len(path) > PROXY_SAMPLE_CELLS + 1:
            path = path[np.linspace(0, len(path) - 1, PROXY_SAMPLE_CELLS + 1, dtype=np.int64)]
        if len(path) > 1:
            score += float(np.abs(desc[path[1:]] - desc[path[:-1]]).sum())
            score += .01 * float(np.count_nonzero(indices[path[1:], -CELL:] != indices[path[:-1], :CELL]))
    return score


def rank_correlation(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return None
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def visualize(name: str, assign: np.ndarray, rows: int, cols: int) -> None:
    colors = ((60, 110, 200), (230, 130, 40), (50, 160, 95), (190, 70, 90))
    tile = 5
    im = Image.new("RGB", (cols * tile * 2 + 28, rows * tile + 30), "white")
    d = ImageDraw.Draw(im)
    d.text((2, 2), f"{name}: raster cells / grouped virtual order", fill="black")
    for i, g in enumerate(assign):
        y, x = divmod(i, cols)
        d.rectangle((x * tile, 20 + y * tile, (x + 1) * tile - 1, 20 + (y + 1) * tile - 1), fill=colors[int(g)])
    order = np.concatenate([np.flatnonzero(assign == g) for g in range(4)])
    for j, i in enumerate(order):
        y, x = divmod(j, cols)
        x0 = cols * tile + 28 + x * tile
        d.rectangle((x0, 20 + y * tile, x0 + tile - 1, 20 + (y + 1) * tile - 1), fill=colors[int(assign[i])])
    VISUALS.mkdir(parents=True, exist_ok=True)
    im.save(VISUALS / f"{name}.layout.png", optimize=True)


def process(name: str) -> dict:
    tracemalloc.start()
    started = time.perf_counter()
    base_info = encode_base(name)
    src, base = base_info["src"], base_info["base"]
    residual = src.astype(np.int16) - base.astype(np.int16)
    t = time.perf_counter()
    dist_features, spatial_features = features_of(residual)
    distribution = cluster(dist_features)
    structured = cluster(np.column_stack((dist_features, spatial_features)))
    group_seconds = time.perf_counter() - t
    desc_dist = (dist_features - dist_features.mean(0)) / np.where(dist_features.std(0) < 1e-9, 1, dist_features.std(0))
    desc_struct = np.column_stack((dist_features, spatial_features))
    t = time.perf_counter()
    palette = palettes_for(residual, structured)
    palette_seconds = time.perf_counter() - t
    h, w, _ = src.shape
    rows, cols = math.ceil(h / CELL), math.ceil(w / CELL)
    results = {
        "image": name, "dimensions": [w, h],
        "base": {"bytes": base_info["stream_bytes"], "sha256": base_info["stream_sha256"],
                 "source_sha256": base_info["source_sha256"],
                 "encode_seconds": base_info["encode_seconds"]},
        "base_mse": distortion(src, base)["mse"],
        "timings": {"feature_grouping_seconds": group_seconds, "palette_quantize_setup_seconds": palette_seconds},
        "quantization": {}, "seed_search": {}, "representations": []
    }
    retained = {}
    for mode, assign, desc in (("distribution", distribution, desc_dist),
                               ("distribution_structure", structured, desc_struct)):
        gp = grouped_palette(residual, assign)
        ix = quantize(residual, gp, assign)
        retained[mode] = (assign.copy(), gp.copy(), ix.copy())
        q = distortion(src, reconstruct(base, ix, gp, assign))
        results["quantization"][mode] = {**q, "group_map_bytes": len(groupmap(assign))}
        # Evaluate all fixed seeds cheaply; entropy-code only bounded proxy leaders and fixed controls.
        permutation(int(np.bincount(assign, minlength=4).max()), 0, 0)
        t = time.perf_counter()
        scores_list, proxy_budget_seconds = [], {}
        for seed in range(256):
            scores_list.append(proxy(ix, assign, desc, seed))
            if seed + 1 in BUDGETS:
                proxy_budget_seconds[seed + 1] = time.perf_counter() - t
        scores = np.asarray(scores_list)
        proxy_seconds = time.perf_counter() - t
        budgets = []
        for budget in BUDGETS:
            leaders = np.argsort(scores[:budget], kind="stable")[:min(3, budget)]
            fixed = np.unique(np.linspace(0, budget - 1, min(2, budget), dtype=int))
            candidates = sorted(set(leaders.tolist()) | set(fixed.tolist()))
            actual, candidate_scores = [], []
            t = time.perf_counter()
            for seed in candidates:
                blob, _ = virtual_file(ix, assign, gp, w, h, 1, seed, math.ceil(math.log2(budget)))
                actual.append(len(blob))
                candidate_scores.append(scores[seed])
                verify(blob, ix, 1)
            search_encode = time.perf_counter() - t
            selected = int(candidates[int(np.argmin(actual))])
            budget_row = {"budget": budget, "candidate_bits": math.ceil(math.log2(budget)),
                          "selected_seed_index": selected, "proxy_score": float(scores[selected]),
                          "seeded_bytes": int(min(actual)),
                          "proxy_seconds_budget": proxy_budget_seconds[budget],
                          "proxy_seconds_256": proxy_seconds, "coded_candidates_seconds": search_encode,
                          "sampled_spearman": rank_correlation(np.asarray(candidate_scores), np.asarray(actual))}
            budgets.append(budget_row)
        results["seed_search"][mode] = budgets

    assign = structured
    gp = grouped_palette(residual, assign)
    variant_rows = []
    for pname, pal, a in (("global4", palette["global4"], None),
                          ("global16", palette["global16"], None),
                          ("group4", gp, assign)):
        ix = quantize(residual, pal, a)
        q = distortion(src, reconstruct(base, ix, pal, a))
        map_assign = np.zeros(len(ix), np.uint8) if a is None else a
        tenc = time.perf_counter()
        raster_strategy = 3 if pname == "group4" else 0
        blob, parts = virtual_file(ix, map_assign, pal, w, h, raster_strategy)
        enc_elapsed = time.perf_counter() - tenc
        tdec = time.perf_counter()
        decoded = decode_enhancement(blob, base)
        dec_elapsed = time.perf_counter() - tdec
        if not np.array_equal(decoded, reconstruct(base, ix, pal, a)):
            raise AssertionError("file-only virtual image reconstruction")
        variant_rows.append({"variant": f"original_raster_{pname}", "bytes": len(blob), "parts": parts,
                             "encode_seconds": enc_elapsed, "decode_seconds": dec_elapsed,
                             "encode_timing_3": repeated_timing(lambda: virtual_file(ix, map_assign, pal, w, h, raster_strategy)[0]),
                             "decode_timing_3": repeated_timing(lambda: decode_enhancement(blob, base)), **q})
        if pname == "group4":
            verify(blob, ix, 3)
            tvirtual_encode = time.perf_counter()
            virtual, virtual_parts = virtual_file(ix, a, pal, w, h, 0)
            virtual_encode = time.perf_counter() - tvirtual_encode
            verify(virtual, ix, 0)
            virtual_decode_start = time.perf_counter()
            virtual_rec = decode_enhancement(virtual, base)
            virtual_decode = time.perf_counter() - virtual_decode_start
            if not np.array_equal(virtual_rec, decoded):
                raise AssertionError("grouped virtual baseline changed reconstruction")
            selected_seed = results["seed_search"]["distribution_structure"][-1]["selected_seed_index"]
            selected_blob, selected_parts = virtual_file(ix, a, pal, w, h, 1, selected_seed, 8)
            verify(selected_blob, ix, 1)
            selected_dec_start = time.perf_counter()
            selected_rec = decode_enhancement(selected_blob, base)
            selected_decode = time.perf_counter() - selected_dec_start
            if not np.array_equal(selected_rec, decoded):
                raise AssertionError("best seed changed reconstruction")
            trand_encode = time.perf_counter()
            rand, randparts = virtual_file(ix, a, pal, w, h, 1, 0)
            rand_encode = time.perf_counter() - trand_encode
            verify(rand, ix, 1)
            trand = time.perf_counter()
            rand_rec = decode_enhancement(rand, base)
            rand_decode = time.perf_counter() - trand
            tcontent_encode = time.perf_counter()
            content, contparts = virtual_file(ix, a, pal, w, h, 2, desc=desc_struct)
            content_encode = time.perf_counter() - tcontent_encode
            verify(content, ix, 2)
            tcontent = time.perf_counter()
            content_rec = decode_enhancement(content, base)
            content_decode = time.perf_counter() - tcontent
            if not np.array_equal(rand_rec, decoded) or not np.array_equal(content_rec, decoded):
                raise AssertionError("ordering changed reconstruction")
            ttemplate = time.perf_counter()
            template, tparts = template_file(ix, a, pal, w, h)
            template_encode = time.perf_counter() - ttemplate
            if not np.array_equal(read_template(template), ix):
                raise AssertionError("template index recovery")
            ttemplate_decode = time.perf_counter()
            template_rec = decode_template(template, base)
            template_decode = time.perf_counter() - ttemplate_decode
            if not np.array_equal(template_rec, decoded):
                raise AssertionError("file-only template reconstruction")
            variant_rows.extend([
                {"variant": "grouped_virtual_original_order", "bytes": len(virtual), "parts": virtual_parts,
                 "encode_seconds": virtual_encode, "decode_seconds": virtual_decode,
                 "encode_timing_3": repeated_timing(lambda: virtual_file(ix, a, pal, w, h, 0)[0]),
                 "decode_timing_3": repeated_timing(lambda: decode_enhancement(virtual, base)), **q},
                {"variant": "best_seeded_virtual_256", "bytes": len(selected_blob), "parts": selected_parts,
                 "encode_seconds": None, "decode_seconds": selected_decode,
                 "encode_timing_3": repeated_timing(lambda: virtual_file(ix, a, pal, w, h, 1, selected_seed, 8)[0]),
                 "decode_timing_3": repeated_timing(lambda: decode_enhancement(selected_blob, base)), **q},
                {"variant": "random_seed0", "bytes": len(rand), "parts": randparts,
                 "encode_seconds": rand_encode, "decode_seconds": rand_decode,
                 "encode_timing_3": repeated_timing(lambda: virtual_file(ix, a, pal, w, h, 1, 0)[0]),
                 "decode_timing_3": repeated_timing(lambda: decode_enhancement(rand, base)), **q},
                {"variant": "content_order_signaled", "bytes": len(content), "parts": contparts,
                 "encode_seconds": content_encode, "decode_seconds": content_decode,
                 "encode_timing_3": repeated_timing(lambda: virtual_file(ix, a, pal, w, h, 2, desc=desc_struct)[0]),
                 "decode_timing_3": repeated_timing(lambda: decode_enhancement(content, base)), **q},
                {"variant": "content_order_no_map_upper_bound", "bytes": len(content) - contparts["order_map"], "parts": contparts, **q},
                {"variant": "grouped_template", "bytes": len(template), "parts": tparts,
                 "encode_seconds": template_encode, "decode_seconds": template_decode,
                 "encode_timing_3": repeated_timing(lambda: template_file(ix, a, pal, w, h)[0]),
                 "decode_timing_3": repeated_timing(lambda: decode_template(template, base)), **q},
            ])
    results["representations"] = variant_rows
    arrays = OUT / "arrays"
    arrays.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arrays / f"{name}.npz",
                        distribution_assignments=retained["distribution"][0],
                        distribution_palette=retained["distribution"][1],
                        distribution_indices=retained["distribution"][2],
                        structure_assignments=retained["distribution_structure"][0],
                        structure_palette=retained["distribution_structure"][1],
                        structure_indices=retained["distribution_structure"][2])
    results["group_assignment_counts"] = np.bincount(structured, minlength=4).tolist()
    results["group_palette"] = gp.astype(int).tolist()
    results["timings"]["total_enhancement_seconds"] = time.perf_counter() - started - (base_info["encode_seconds"] or 0)
    results["peak_traced_bytes"] = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    visualize(name, structured, rows, cols)
    return results


def visualize_rd(rows: list[dict]) -> None:
    VISUALS.mkdir(parents=True, exist_ok=True)
    names = ("original_raster_global4", "original_raster_global16", "original_raster_group4",
             "grouped_virtual_original_order", "best_seeded_virtual_256", "content_order_signaled",
             "grouped_template")
    labels = ("Global 4", "Global 16", "Group palette", "Grouped", "Seeded", "Content", "Template")
    colors = ((50, 100, 190), (200, 55, 50), (40, 150, 90), (210, 140, 30),
              (125, 70, 170), (40, 155, 175), (90, 90, 90))
    canvas = Image.new("RGB", (900, 610), "white")
    draw = ImageDraw.Draw(canvas)
    for ix, row in enumerate(rows):
        col, r = ix % 2, ix // 2
        x0, y0 = 55 + col * 430, 45 + r * 185
        x1, y1 = x0 + 350, y0 + 125
        draw.text((x0, y0 - 20), row["image"], fill="black")
        draw.line((x0, y0, x0, y1, x1, y1), fill="black", width=1)
        available = {v["variant"]: v for v in row["representations"]}
        values = [available[n] for n in names if n in available]
        max_bpp = max(v["bytes"] * 8 / (row["dimensions"][0] * row["dimensions"][1]) for v in values) or 1
        max_mse = max(v["mse"] for v in values) or 1
        for name, color in zip(names, colors):
            v = available.get(name)
            if v:
                bpp = v["bytes"] * 8 / (row["dimensions"][0] * row["dimensions"][1])
                x = x0 + int(340 * bpp / max_bpp)
                y = y1 - int(110 * v["mse"] / max_mse)
                draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)
        draw.text((x0 + 10, y1 + 5), "Enhancement rate (bpp)", fill="black")
        draw.text((x0 - 48, y0 + 5), "RGB MSE", fill="black")
    for i, (label, color) in enumerate(zip(labels, colors)):
        x = 18 + (i % 4) * 220
        y = 570 + (i // 4) * 18
        draw.ellipse((x, y + 3, x + 8, y + 11), fill=color)
        draw.text((x + 12, y), label, fill="black")
    canvas.save(VISUALS / "rate_distortion.png", optimize=True)


def refresh_decode_timings() -> None:
    path = OUT / "results.json"
    results = json.loads(path.read_text())
    for row in results["images"]:
        name = row["image"]
        base_info = encode_base(name)
        src, base = base_info["src"], base_info["base"]
        residual = src.astype(np.int16) - base.astype(np.int16)
        t = time.perf_counter()
        dist_features, spatial = features_of(residual)
        feature_seconds = time.perf_counter() - t
        desc = np.column_stack((dist_features, spatial))
        t = time.perf_counter()
        assignments = cluster(desc)
        grouping_seconds = time.perf_counter() - t
        t = time.perf_counter()
        palettes = palettes_for(residual, assignments)
        palette_seconds = time.perf_counter() - t
        row["timings"]["feature_extraction_seconds"] = feature_seconds
        row["timings"]["grouping_seconds"] = grouping_seconds
        row["timings"]["palette_quantize_setup_seconds"] = palette_seconds
        h, w, _ = src.shape
        group_palette = palettes["group4"]
        cases = {}
        archive = np.load(OUT / "arrays" / f"{name}.npz")
        proxy_modes = (
            ("distribution", archive["distribution_assignments"], archive["distribution_indices"],
             (dist_features - dist_features.mean(0)) / np.where(dist_features.std(0) < 1e-9, 1, dist_features.std(0))),
            ("distribution_structure", archive["structure_assignments"], archive["structure_indices"], desc),
        )
        for mode, assignment, indices_for_proxy, proxy_desc in proxy_modes:
            permutation(int(np.bincount(assignment, minlength=4).max()), 0, 0)
            tproxy = time.perf_counter()
            proxy_times = {}
            for seed_index in range(256):
                proxy(indices_for_proxy, assignment, proxy_desc, seed_index)
                if seed_index + 1 in BUDGETS:
                    proxy_times[seed_index + 1] = time.perf_counter() - tproxy
            for budget_row in row["seed_search"][mode]:
                budget_row["proxy_seconds_budget"] = proxy_times[budget_row["budget"]]
                budget_row["proxy_seconds_256"] = proxy_times[256]
        for palette_name in ("global4", "global16"):
            palette = palettes[palette_name]
            indices = quantize(residual, palette, None)
            blob, _ = virtual_file(indices, np.zeros(len(indices), np.uint8), palette, w, h)
            cases[f"original_raster_{palette_name}"] = lambda blob=blob: decode_enhancement(blob, base)
        indices = quantize(residual, group_palette, assignments)
        raster_grouped = virtual_file(indices, assignments, group_palette, w, h, 3)[0]
        cases["original_raster_group4"] = lambda: decode_enhancement(raster_grouped, base)
        grouped_blob = virtual_file(indices, assignments, group_palette, w, h, 0)[0]
        cases["grouped_virtual_original_order"] = lambda: decode_enhancement(grouped_blob, base)
        selected_seed = row["seed_search"]["distribution_structure"][-1]["selected_seed_index"]
        seeded = virtual_file(indices, assignments, group_palette, w, h, 1, selected_seed, 8)[0]
        cases["best_seeded_virtual_256"] = lambda: decode_enhancement(seeded, base)
        random_blob = virtual_file(indices, assignments, group_palette, w, h, 1, 0)[0]
        cases["random_seed0"] = lambda: decode_enhancement(random_blob, base)
        content = virtual_file(indices, assignments, group_palette, w, h, 2, desc=desc)[0]
        cases["content_order_signaled"] = lambda: decode_enhancement(content, base)
        template = template_file(indices, assignments, group_palette, w, h)[0]
        cases["grouped_template"] = lambda: decode_template(template, base)
        for rep in row["representations"]:
            fn = cases.get(rep["variant"])
            if fn is None:
                continue
            decoded = fn()
            if decoded.shape != src.shape:
                raise AssertionError(f"decode dimensions changed for {name}/{rep['variant']}")
            measured_mse = distortion(src, decoded)["mse"]
            if not np.isclose(measured_mse, rep["mse"], rtol=0, atol=1e-7):
                raise AssertionError(f"reconstructed MSE changed for {name}/{rep['variant']}")
            rep["decode_timing_3"] = repeated_timing(fn)
            rep["decode_seconds"] = rep["decode_timing_3"]["median"]
    results["runtime"]["decode_timing_repeats"] = 3
    path.write_text(json.dumps(results, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", nargs="*", default=list(IMAGES))
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--refresh-decode-timings", action="store_true")
    args = parser.parse_args()
    numba.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    if args.refresh_decode_timings:
        refresh_decode_timings()
        visualize_rd(json.loads((OUT / "results.json").read_text())["images"])
        return
    partial = OUT / "partial.json"
    rows = json.loads(partial.read_text()) if partial.exists() and not args.rerun else []
    completed = {r["image"] for r in rows}
    if not args.finalize:
        for name in args.images:
            if name in completed:
                continue
            print(f"processing {name}", flush=True)
            rows.append(process(name))
            completed.add(name)
            partial.write_text(json.dumps(rows, indent=2))
    config = PBC3Config.quality()
    model = ROOT / "patch_policy.npz"
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    source_files = ("PBC3.py", "pbc3_types.py", "pbc3_ops.py", "pbc3_heads.py",
                    "pbc3_features.py", "learned_filler.py", "patch_policy.npz")
    source_hashes = {f: sha((ROOT / f).read_bytes()) for f in source_files}
    obj = {"head": commit, "runtime_source_sha256": source_hashes,
           "runtime": {"python": sys.version, "platform": platform.platform(),
                       "numpy": np.__version__, "numba": numba.__version__, "pillow": PILLOW_VERSION,
                       "cpu": platform.processor(), "logical_cpus": os.cpu_count(),
                       "thread_environment": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS")},
                       "threads": {k: __import__(k).get_num_threads() for k in ("numba",)},
                       "model_sha256": sha(model.read_bytes()), "preset": config.__dict__, "rng_mode": "release"},
           "cell_size": CELL, "groups": 4, "seed_budgets": BUDGETS, "proxy_sample_cells_per_group": PROXY_SAMPLE_CELLS, "images": rows}
    (OUT / "results.json").write_text(json.dumps(obj, indent=2))
    visualize_rd(rows)
    print(OUT / "results.json")


if __name__ == "__main__":
    main()
