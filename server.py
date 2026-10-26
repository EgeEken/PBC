import ast
import base64
import glob
import io
import json
import math
import os
_HERE = os.path.dirname(os.path.abspath(__file__))
import time
from typing import List

import tempfile
from fastapi import FastAPI, File, Form, Request, UploadFile, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse, Response, FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps
try:
    import pillow_jxl  # noqa: F401 — registers JXL with PIL
except Exception:
    pass
import numpy as np
import cv2

from PBC2_4 import PBC, PBC2Config, PBC2Result
from PBC3 import PBC3, PBC3Config, preload_numba
from PBC3_animation import animate_pbc3
import pbc3_sweep
import pbc3_quick_rd
import pbc3_benchmark
import train_api

import urllib.request

import threading

BENCHMARK_LOCK = threading.Lock()
BENCHMARK_OWNER = {"name": None}
BENCHMARK_OWNER_LOCK = threading.Lock()


def _release_benchmark_lock(name):
    with BENCHMARK_OWNER_LOCK:
        if BENCHMARK_OWNER.get("name") != name:
            return
        BENCHMARK_OWNER["name"] = None
        BENCHMARK_LOCK.release()


def _benchmark_busy():
    return bool(
        pbc3_sweep.status().get("running")
        or pbc3_quick_rd.status().get("running")
        or pbc3_benchmark.status().get("running")
        or train_api.status().get("running")
    )


def _guarded_benchmark_start(name, start_fn, status_fn, *args):
    if _benchmark_busy():
        return {"ok": False, "error": "Another benchmark is already running."}
    if not BENCHMARK_LOCK.acquire(blocking=False):
        return {"ok": False, "error": "Another benchmark is already running."}

    with BENCHMARK_OWNER_LOCK:
        BENCHMARK_OWNER["name"] = name

    try:
        result = start_fn(*args)
    except Exception:
        _release_benchmark_lock(name)
        raise

    if isinstance(result, dict) and result.get("error"):
        _release_benchmark_lock(name)
        return result

    def monitor():
        deadline = time.time() + 5.0
        while time.time() < deadline and not status_fn().get("running"):
            time.sleep(0.05)
        while status_fn().get("running"):
            time.sleep(0.25)
        _release_benchmark_lock(name)

    threading.Thread(target=monitor, daemon=True).start()
    return result

import importlib.util
from PIL import features, __version__ as PIL_VERSION
print(f"[startup] Pillow {PIL_VERSION} · native AVIF={features.check('avif')} · "
      f"plugin_installed={importlib.util.find_spec('pillow_avif') is not None}", flush=True)

app = FastAPI(title="PBC Compression Demo")

CANVAS = 256

TOPICS = ["nature", "city", "animals", "food", "architecture", "landscape",
          "technology", "sports", "flowers", "beach", "mountains", "street", "cats"]

PROVIDERS = [
    lambda s: f"https://picsum.photos/seed/{s}/{CANVAS}/{CANVAS}",            # dead from HF, last resort
    lambda s: f"https://loremflickr.com/{CANVAS}/{CANVAS}/{TOPICS[s % len(TOPICS)]}?lock={s}",
    lambda s: f"https://loremflickr.com/{CANVAS}/{CANVAS}?lock={s}",          # untagged backup
]
_good = {"i": 0}

def _fetch(url, timeout=4):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read(), r.headers.get("Content-Type", "image/jpeg")

@app.on_event("startup")
def _probe_providers():
    for i, p in enumerate(PROVIDERS):
        try:
            _fetch(p(1)); _good["i"] = i
            print(f"[random_photo] reachable via provider {i}", flush=True); return
        except Exception as e:
            print(f"[random_photo] provider {i} unreachable: {e!r}", flush=True)
    print("[random_photo] NO provider reachable -> HF egress is blocked", flush=True)

@app.get("/api/random_photo")
def random_photo(seed: int):
    order = [_good["i"]] + [i for i in range(len(PROVIDERS)) if i != _good["i"]]
    for i in order:
        try:
            data, ctype = _fetch(PROVIDERS[i](seed))
            _good["i"] = i
            return Response(content=data, media_type=ctype, headers={"Cache-Control": "no-store"})
        except Exception as e:
            print(f"[random_photo] provider {i} failed: {e!r}", flush=True)
    raise HTTPException(status_code=502, detail="all providers failed")

# Maps the UI's resampling labels to PBC2.4 config resample names.
RESAMPLE = {
    "Lanczos": "lanczos",
    "Bicubic": "bicubic",
    "Bilinear": "bilinear",
    "Nearest": "nearest",
    "Box": "box",
}

def mse_metric(a, b):
    a = np.asarray(a.convert("RGB") if isinstance(a, Image.Image) else a, dtype=np.float64)
    b = np.asarray(b.convert("RGB") if isinstance(b, Image.Image) else b, dtype=np.float64)
    return float(np.mean((a - b) ** 2))

def _aligned_mse(orig, rec):
    a = np.asarray(orig, dtype=np.float64)
    b = np.asarray(rec.convert(orig.mode), dtype=np.float64)
    return float(np.mean((a - b) ** 2))


def _ssim_maps(a, b):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    mu_a = cv2.GaussianBlur(a, (11, 11), 1.5)
    mu_b = cv2.GaussianBlur(b, (11, 11), 1.5)

    mu_a2 = mu_a * mu_a
    mu_b2 = mu_b * mu_b
    mu_ab = mu_a * mu_b

    sa = cv2.GaussianBlur(a * a, (11, 11), 1.5) - mu_a2
    sb = cv2.GaussianBlur(b * b, (11, 11), 1.5) - mu_b2
    sab = cv2.GaussianBlur(a * b, (11, 11), 1.5) - mu_ab

    cs = (2 * sab + C2) / (sa + sb + C2)
    ssim = ((2 * mu_ab + C1) / (mu_a2 + mu_b2 + C1)) * cs

    return float(ssim.mean()), float(cs.mean())


def _ms_ssim_2d(a, b):
    weights = np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    a = a.astype(np.float64)
    b = b.astype(np.float64)

    mssim = []
    mcs = []

    for i in range(len(weights)):
        s, cs = _ssim_maps(a, b)
        mssim.append(s)
        mcs.append(cs)

        if i < len(weights) - 1:
            h = max(1, a.shape[0] // 2)
            w = max(1, a.shape[1] // 2)
            a = cv2.resize(a, (w, h), interpolation=cv2.INTER_AREA)
            b = cv2.resize(b, (w, h), interpolation=cv2.INTER_AREA)

    mssim = np.clip(np.array(mssim), 1e-8, 1.0)
    mcs = np.clip(np.array(mcs), 1e-8, 1.0)

    return float(np.prod(mcs[:-1] ** weights[:-1]) * (mssim[-1] ** weights[-1]))


def ms_ssim_rgb(a, b):
    a = np.asarray(a.convert("RGB") if isinstance(a, Image.Image) else a)
    b = np.asarray(b.convert("RGB") if isinstance(b, Image.Image) else b)
    return float(np.mean([_ms_ssim_2d(a[:, :, c], b[:, :, c]) for c in range(3)]))


def edge_similarity(a, b):
    a = np.asarray(a.convert("RGB") if isinstance(a, Image.Image) else a)
    b = np.asarray(b.convert("RGB") if isinstance(b, Image.Image) else b)

    a = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float64)
    b = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY).astype(np.float64)

    ax = cv2.Sobel(a, cv2.CV_64F, 1, 0, ksize=3)
    ay = cv2.Sobel(a, cv2.CV_64F, 0, 1, ksize=3)
    bx = cv2.Sobel(b, cv2.CV_64F, 1, 0, ksize=3)
    by = cv2.Sobel(b, cv2.CV_64F, 0, 1, ksize=3)

    ga = np.sqrt(ax * ax + ay * ay)
    gb = np.sqrt(bx * bx + by * by)

    return float((2 * np.mean(ga * gb) + 1e-6) / (np.mean(ga * ga) + np.mean(gb * gb) + 1e-6))


def laplacian_similarity(a, b):
    a = np.asarray(a.convert("RGB") if isinstance(a, Image.Image) else a)
    b = np.asarray(b.convert("RGB") if isinstance(b, Image.Image) else b)

    a = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float64)
    b = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY).astype(np.float64)

    la = cv2.Laplacian(a, cv2.CV_64F, ksize=3)
    lb = cv2.Laplacian(b, cv2.CV_64F, ksize=3)

    return float((2 * np.mean(la * lb) + 1e-6) / (np.mean(la * la) + np.mean(lb * lb) + 1e-6))


def composite_quality(a, b):
    mse = mse_metric(a, b)

    m = np.clip(ms_ssim_rgb(a, b), 0.0, 1.0)
    e = np.clip(edge_similarity(a, b), 0.0, 1.0)
    l = np.clip(laplacian_similarity(a, b), 0.0, 1.0)

    mse_quality = math.exp(-mse / 140.0)

    return float(0.40 * m + 0.25 * e + 0.25 * l + 0.10 * mse_quality)


def generate_multlist(bit_count, min_val, max_val, mode="Stable_Uniform"):
    if mode in ("PBC Default", "PBC_Default", "PBCDefault"):
        return [-10, 0, 5, 20]

    if min_val > max_val:
        min_val, max_val = max_val, min_val
    if min_val == max_val:
        max_val = min_val + 1
    count = 2 ** int(bit_count)
    if mode == "Random":
        vals = sorted(np.random.default_rng(28042003).integers(min_val, max_val + 1, size=count).tolist())
    else:
        vals = np.linspace(min_val, max_val, count, dtype=int).tolist()
    if mode == "Stable_Uniform":
        closest = min(vals, key=lambda x: abs(x))
        if abs(closest) > 1:
            vals.remove(max(vals, key=lambda x: abs(x)))
            vals.append(0)

    return sorted(set(int(v) for v in vals)) or [0]


# High-resolution multiplier list (bit count 7, range -255..255) for the landing
# teaser. Compression rate is irrelevant there, so a richer palette just makes the
# streamed build look better without costing extra time.
DEMO_MULT_LIST = generate_multlist(7, -255, 255, "Stable_Uniform")


def _png_b64(img: Image.Image, compress_level: int = 6) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG", compress_level=compress_level)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _f(v, d):
    try:
        return float(v) if v not in (None, "") else d
    except ValueError:
        return d


def _i(v, d):
    return int(_f(v, d))


def _truthy(v):
    return str(v).lower() == "true"

PBC3_PRESETS = {
    "compression": PBC3Config.compression,
    "balanced": PBC3Config.balanced,
    "quality": PBC3Config.quality,
    "high_quality": PBC3Config.high_quality,
}
LEARNED_Q_DEFAULTS = {"compression": 0.4, "balanced": 0.6, "quality": 0.8, "high_quality": 0.95}

# Every PBC3Config field the UI can submit, with the caster used to parse its form value.
PBC3_FIELDS = {
    "patch_count": int, "search_depth": int, "proposal_depth": int, "exact_depth": int,
    "min_patch_size": int, "max_patch_size": int, "min_cell_size": int, "max_cell_size": int,
    "cell_sizes_per_candidate": int, "top_k": int,
    "search_q_start": float, "search_q_end": float, "q_init": float, "q_start": float, "q_end": float,
    "color_space": str, "channel_cycle": str,
    "auto_downsample_init": _truthy, "init_search_depth": int, "downsample_init_cell_size": int,
    "downsample_palette_bitcount": int, "downsample_rate": float, "auto_downsample_max_pixels": int,
    "warmup_ratio": float, "warm_downsample_max_pixels": int,
    "patch_palette_bitcount": int, "quality_target_mae": float, "mask_size": int,
    "anchor_block_size": int,
    "positive_bias": _truthy, "use_lzma": _truthy, "random_seed": int,
    "debug_mode": _truthy, "debug_print": _truthy,
    "learned_filler_enabled": _truthy, "learned_filler_q": float,
    "learned_filler_top_k": int, "learned_filler_candidates": int,
}

@app.get("/api/multlist")
def multlist(bit_count: int = 2, min: int = -10, max: int = 20, mode: str = "Stable_Uniform"):
    return {"list": list(generate_multlist(bit_count, min, max, mode))}


@app.post("/api/compress")
async def compress(request: Request):
    stages = {}
    t = time.perf_counter()
    def mark(name):
        nonlocal t
        stages[name] = round(time.perf_counter() - t, 3)
        t = time.perf_counter()

    form = await request.form()
    upload = form.get("image")
    if upload is None:
        return JSONResponse({"error": "No image provided"}, status_code=400)
    try:
        raw_bytes = await upload.read()
        src = ImageOps.exif_transpose(Image.open(io.BytesIO(raw_bytes)))
        img = src.convert("RGBA") if PBC3._has_alpha(src) else src.convert("RGB")
    except Exception as exc:
        return JSONResponse({"error": f"Could not read image: {exc}"}, status_code=400)
    w, h = img.size
    mark("read_decode")

    mode = form.get("mode", "Auto")
    mode = mode if mode in {"Auto", "Semi", "Manual"} else "Manual"
    if mode == "Auto":
        preset = form.get("auto_config", "quality")
        config = PBC3_PRESETS.get(preset, PBC3Config.quality)()
        kwargs = {"auto_config": preset}
    else:
        preset = None
        kwargs = {}
        for k, caster in PBC3_FIELDS.items():
            v = form.get(k)
            if v in (None, ""):
                continue
            try:
                kwargs[k] = caster(v)
            except (ValueError, TypeError):
                pass
        config = PBC3Config(**kwargs)

    le = form.get("learned_filler_enabled")
    if le is not None:
        config.learned_filler_enabled = _truthy(le)
    if config.learned_filler_enabled:
        config.learned_filler_candidates = 1
        config.learned_filler_top_k = 1
        lq = form.get("learned_filler_q")
        if lq not in (None, ""):
            try:
                config.learned_filler_q = float(lq)
            except ValueError:
                pass
        elif preset:
            config.learned_filler_q = LEARNED_Q_DEFAULTS.get(preset, 0.7)
        if not os.path.isabs(config.learned_filler_model_path):
            config.learned_filler_model_path = os.path.join(_HERE, "patch_policy.npz")
    mark("build_config")
    if getattr(config, "learned_filler_enabled", False) and not os.path.isabs(config.learned_filler_model_path):
        config.learned_filler_model_path = os.path.join(_HERE, config.learned_filler_model_path)

    try:
        result = PBC3.compress(img, config=config)
    except Exception as exc:
        return JSONResponse({"error": f"Compression failed: {exc}"}, status_code=400)
    reconstructed = result.image                      # keep alpha if present
    mark("compress")

    mse = float(result.mse)
    mark("mse")

    recon_b64 = _png_b64(reconstructed, 1)               # PNG preserves the alpha channel
    mark("png_reconstructed")

    pbc_b64 = base64.b64encode(result.data).decode()
    mark("encode_payload")

    original_raw = w * h * len(img.getbands())
    compressed = len(result.data)
    print(f"[compress] encode={result.encode_seconds:.3f}s  stages={stages}", flush=True)

    return JSONResponse({
        "reconstructed_image": recon_b64,
        "pbc_base64": pbc_b64,
        "width": w,
        "height": h,
        "original_raw_kb": round(original_raw / 1024, 2),
        "original_file_kb": round(len(raw_bytes) / 1024, 2),
        "compressed_kb": round(compressed / 1024, 2),
        "compression_rate": round(original_raw / compressed, 2) if compressed else 0,
        "compression_percent": round(compressed / original_raw * 100, 2) if original_raw else 0,
        "mse": round(mse, 2),
        "time_seconds": round(result.encode_seconds, 2),
        "stage_timings": stages,
        "params": {"mode": mode, **{k: (list(v) if isinstance(v, tuple) else v) for k, v in kwargs.items()}},
    })

@app.post("/api/stream_compress")
async def stream_compress(image: UploadFile = File(...), downsample_initialize: str = Form("false")):
    raw = await image.read()
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")

    def gen():
        for ev in PBC3.compress_stream(
            img,
            config=PBC3Config.high_quality(
                auto_downsample_init=_truthy(downsample_initialize),
                patch_count=150,
                max_patch_size=48,
                min_cell_size=1,
                max_cell_size=16,
                q_init=0.5, q_start=0.7, q_end=0.6,
                search_q_start=0.6, search_q_end=0.4,
            ),
            frame_every=1,
        ):
            if ev["event"] == "frame":
                yield json.dumps({"image": _png_b64(ev["image"])}) + "\n"

    return StreamingResponse(gen(), media_type="application/x-ndjson")


@app.post("/api/decode")
async def decode(file: UploadFile = File(...)):
    raw = await file.read()

    try:
        dec_res = PBC3.decompress(bytes(raw))
        img = dec_res.image                            # keep alpha
        elapsed = dec_res.encode_seconds
    except Exception as exc:
        return JSONResponse({"error": f"Decode failed: {exc}"}, status_code=400)

    w, h = img.size
    original_raw = w * h * len(img.getbands())
    compressed = len(raw)

    return JSONResponse({
        "reconstructed_image": _png_b64(img, 1),
        "pbc_base64": base64.b64encode(raw).decode(),
        "width": w,
        "height": h,
        "original_raw_kb": round(original_raw / 1024, 2),
        "compressed_kb": round(compressed / 1024, 2),
        "compression_rate": round(original_raw / compressed, 2) if compressed else 0,
        "compression_percent": round(compressed / original_raw * 100, 2) if original_raw else 0,
        "time_seconds": round(elapsed, 2),
        "decoded": True,
    })


# ====================================================================================================
#  CODEC MATCHING  -  encode an image to (as close as possible to) a target bpp in very few tries.
# ====================================================================================================

# Rough quality→bpp guidelines (per the project's reference images). Used only to seed the search
# with a sensible starting quality, so matching a target bpp takes 1-3 encodes instead of a full sweep.
BPP_GUIDE = {
    "JPEG": [(1, 0.17), (5, 0.21), (10, 0.29), (20, 0.44), (40, 0.65), (60, 0.85), (80, 1.26), (95, 2.77)],
    "AVIF": [(0, 0.01), (2, 0.02), (5, 0.04), (10, 0.06), (20, 0.12), (40, 0.30), (60, 0.68), (80, 1.16), (95, 2.69)],
    "WEBP": [(0, 0.06), (5, 0.13), (10, 0.17), (20, 0.25), (40, 0.40), (60, 0.55), (80, 0.85), (95, 1.80)],
    "JPEG2000": [(0, 0.03), (5, 0.05), (10, 0.08), (20, 0.14), (40, 0.30), (60, 0.65), (80, 1.20), (95, 2.50)],
    "JXL":  [(0, 0.02), (2, 0.03), (5, 0.05), (10, 0.08), (20, 0.15), (40, 0.34), (60, 0.70), (80, 1.20), (95, 2.60)],
}
# Quality samples for the tradeoff baselines (AVIF goes down to 0, JPEG stops at 1).
JPEG_QUALITIES = (1, 2, 3, 5, 8, 10, 20, 40, 60, 80, 95)
AVIF_QUALITIES = (0, 1, 2, 3, 5, 8, 10, 20, 40, 60, 80, 95)
WEBP_QUALITIES = (0, 1, 2, 3, 5, 8, 10, 20, 40, 60, 80, 95)
JP2_QUALITIES = (0, 1, 2, 3, 5, 8, 10, 20, 40, 60, 80, 95)
JXL_QUALITIES = (0, 1, 2, 3, 5, 8, 10, 20, 40, 60, 80, 95)

def _q_bounds(fmt):
    return (1, 95) if fmt == "JPEG" else (0, 95)


def _jp2_rate(q):
    q = max(0.0, min(95.0, float(q))) / 95.0
    return 200.0 ** (1.0 - q)


def _encode_codec_bytes(img, fmt, q):
    buf = io.BytesIO()
    if fmt == "JPEG2000":
        img.save(buf, format="JPEG2000", quality_mode="rates", quality_layers=[_jp2_rate(q)])
    else:
        img.save(buf, format=fmt, quality=int(q))
    return buf.getvalue()


def _guess_quality(fmt, target_bpp):
    """Inverse-interpolate the guideline table to pick a starting quality for `target_bpp`."""
    g = BPP_GUIDE[fmt]
    qs = [q for q, _ in g]
    bs = [b for _, b in g]
    if target_bpp <= bs[0]:
        return qs[0]
    if target_bpp >= bs[-1]:
        return qs[-1]
    for i in range(1, len(g)):
        if target_bpp <= bs[i]:
            f = (target_bpp - bs[i - 1]) / (bs[i] - bs[i - 1])
            return int(round(qs[i - 1] + f * (qs[i] - qs[i - 1])))
    return qs[-1]


def _match_codec_gen(img, fmt, target_bpp, pixels, max_iters=12):
    """Yield {'q','bpp'} per new encode, then {'best': {...}} — the quality whose bpp is
    closest to target_bpp (bits/pixel). bpp rises ~monotonically with quality, so probe the
    floor, gallop upward (1,2,4,…) to bracket the target with proportional steps (no jump to
    the max), then interpolation-search inside the bracket. Ties prefer the lower quality."""
    qmin, qmax = _q_bounds(fmt)
    tried = {}
    best = {"ref": None}

    def better(r):
        b = best["ref"]
        if b is None:
            return True
        da, db = abs(r["bpp"] - target_bpp), abs(b["bpp"] - target_bpp)
        return da < db or (da == db and r["q"] < b["q"])

    def enc(q):
        q = int(max(qmin, min(qmax, round(q))))
        if q in tried:
            return tried[q], False
        try:
            data = _encode_codec_bytes(img, fmt, q)
        except Exception:
            tried[q] = None
            return None, False
        r = {"q": q, "bpp": len(data) * 8 / pixels, "data": data}
        tried[q] = r
        if better(r):
            best["ref"] = r
        return r, True

    lo, _ = enc(qmin)
    if lo:
        yield {"q": lo["q"], "bpp": lo["bpp"]}
    if lo is None or lo["bpp"] >= target_bpp:
        yield {"best": best["ref"]}            # nothing smaller than the floor exists
        return

    hi = None
    step = 1
    while lo["q"] < qmax:                       # gallop up to bracket the target
        r, isnew = enc(min(qmax, lo["q"] + step))
        if r is None:
            break
        if isnew:
            yield {"q": r["q"], "bpp": r["bpp"]}
        if r["bpp"] >= target_bpp:
            hi = r
            break
        lo = r
        step *= 2
    if hi is None:
        yield {"best": best["ref"]}            # even max quality is under target
        return

    for _ in range(max_iters):                  # interpolation search inside [lo, hi]
        if hi["q"] - lo["q"] <= 1:
            break
        span = hi["bpp"] - lo["bpp"]
        frac = (target_bpp - lo["bpp"]) / span if span > 0 else 0.5
        q = min(max(int(round(lo["q"] + frac * (hi["q"] - lo["q"]))), lo["q"] + 1), hi["q"] - 1)
        r, isnew = enc(q)
        if r is None:
            break
        if isnew:
            yield {"q": r["q"], "bpp": r["bpp"]}
        if target_bpp > 0 and abs(r["bpp"] - target_bpp) / target_bpp < 0.02:
            break
        if r["bpp"] > target_bpp:
            hi = r
        else:
            lo = r

    yield {"best": best["ref"]}


def _match_codec(img, fmt, target_bpp, pixels):
    best = None
    for m in _match_codec_gen(img, fmt, target_bpp, pixels):
        if "best" in m:
            best = m["best"]
    return best


# ====================================================================================================
#  SWEEP ANALYZER  -  read Optuna SQLite studies, compute Pareto fronts, codec baselines, artifacts.
# ====================================================================================================

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(PROJECT_DIR, "_sweep_uploads")
DEFAULT_DB = "pbc_hpt.db"
IMAGE_EXTS = ("*.png", "*.jpg", "*.jpeg", "*.webp")


def _nd(obj):
    return json.dumps(obj) + "\n"


def _safe_path(*parts):
    p = os.path.abspath(os.path.join(PROJECT_DIR, *parts))
    if not p.startswith(PROJECT_DIR + os.sep) and p != PROJECT_DIR:
        raise ValueError("Path escapes the project directory.")
    return p

def _pareto_mask(values):
    """values: (n, 3) array in 'maximize' orientation [-speed, -bpp, quality]."""
    n = len(values)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        if not mask[i]:
            continue
        dominated = np.all(values <= values[i], axis=1) & np.any(values < values[i], axis=1)
        dominated[i] = False
        mask[dominated] = False
    return mask

def _in_range(mp, lo, hi):
    return (lo is None or mp >= lo) and (hi is None or mp <= hi)


def _pbc3_trials(study, mp_min=None, mp_max=None):
    import optuna
    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE and t.values and len(t.values) == 3]
    rows = []
    for t in completed:
        cfg = t.user_attrs.get("config") or {}
        kind = t.user_attrs.get("kind") or ("baseline" if cfg.get("codec") else "pbc3")

        if mp_min is not None or mp_max is not None:
            sub = [r for r in (t.user_attrs.get("per_image") or []) if _in_range(r["mp"], mp_min, mp_max)]
            if not sub:
                continue
            seconds = float(np.mean([r["seconds"] for r in sub]))
            bpp = float(np.mean([r["bpp"] for r in sub]))
            mse = float(np.mean([r["mse"] for r in sub]))
        else:
            seconds, bpp, mse = t.values

        rows.append((t, seconds, bpp, mse, kind, cfg))

    pareto = [False] * len(rows)
    pbc_rows = [(i, s, b, m) for i, (_, s, b, m, kind, _) in enumerate(rows) if kind == "pbc3"]
    if pbc_rows:
        orient = np.array([[-s, -b, -m] for _, s, b, m in pbc_rows], dtype=np.float64)
        mask = _pareto_mask(orient)
        for (i, _, _, _), is_pareto in zip(pbc_rows, mask):
            pareto[i] = bool(is_pareto)

    out = []
    for i, (t, s, b, m, kind, cfg) in enumerate(rows):
        out.append({
            "number": t.number, "speed": s, "bpp": b, "mse": m,
            "pareto": pareto[i], "baseline": t.user_attrs.get("baseline"),
            "preset": t.user_attrs.get("preset"),
            "kind": kind,
            "codec": cfg.get("codec"), "q": cfg.get("q"), "params": cfg,
        })
    return out

@app.get("/api/presets")
def presets():
    return {name: vars(fn()) for name, fn in PBC3_PRESETS.items()}

@app.get("/api/sweeps/study")
def sweeps_study(mp_min: float = None, mp_max: float = None):
    import optuna
    if not os.path.exists(pbc3_sweep.DB_PATH):
        return {"exists": False, "metric_names": list(pbc3_sweep.METRIC_NAMES), "trials": [], "completed": 0, "pareto": 0}
    study = optuna.load_study(study_name=pbc3_sweep.STUDY, storage=pbc3_sweep.STORAGE)
    trials = _pbc3_trials(study, mp_min, mp_max)
    return {"exists": True, "study_name": pbc3_sweep.STUDY,
            "metric_names": list(pbc3_sweep.METRIC_NAMES),
            "completed": len(trials), "pareto": sum(1 for t in trials if t["pareto"]),
            "trials": trials}

@app.post("/api/sweeps/run/start")
async def sweeps_run_start(request: Request):
    body = await request.json()
    return _guarded_benchmark_start(
        "sweep",
        pbc3_sweep.start,
        pbc3_sweep.status,
        body.get("mode", "optimizer"),
        body.get("spec"),
    )


@app.post("/api/sweeps/run/stop")
def sweeps_run_stop():
    return pbc3_sweep.stop()


@app.get("/api/sweeps/run/status")
def sweeps_run_status():
    return pbc3_sweep.status()


@app.get("/api/quick_rd/status")
def quick_rd_status():
    return pbc3_quick_rd.status()


@app.post("/api/quick_rd/start")
def quick_rd_start():
    return _guarded_benchmark_start("quick_rd", pbc3_quick_rd.start, pbc3_quick_rd.status)


@app.post("/api/quick_rd/stop")
def quick_rd_stop():
    return pbc3_quick_rd.stop()


@app.get("/api/quick_rd/results")
def quick_rd_results(mp_min: float = None, mp_max: float = None):
    return pbc3_quick_rd.results(mp_min, mp_max)


@app.get("/api/sweeps/run/dataset")
def sweeps_run_dataset():
    paths = sorted(p for ext in pbc3_sweep.IMAGE_EXTS
                   for p in glob.glob(os.path.join(pbc3_sweep.DATA_DIR, ext)))
    images = []
    for p in paths:
        try:
            with Image.open(p) as im:
                w, h = ImageOps.exif_transpose(im).size
            images.append({"name": os.path.basename(p), "mp": round(w * h / 1e6, 3)})
        except Exception:
            pass
    return {"count": len(images), "images": images}


@app.get("/api/sweeps/run/params")
def sweeps_run_params():
    out = []
    for name, kind, *a in pbc3_sweep.SEARCH_SPACE:
        if kind == "cat":
            out.append({"name": name, "kind": "cat", "options": a[0]})
        else:
            out.append({"name": name, "kind": kind, "min": a[0], "max": a[1]})
    return {"params": out}


@app.get("/api/sweeps/download_db")
def sweeps_download_db():
    if not os.path.exists(pbc3_sweep.DB_PATH):
        return JSONResponse({"error": "No sweep database yet."}, status_code=404)
    return FileResponse(pbc3_sweep.DB_PATH, filename="pbc3_sweep.db", media_type="application/octet-stream")


@app.post("/api/sweeps/run/upload_db")
async def sweeps_run_upload_db(file: UploadFile = File(...)):
    if pbc3_sweep.status()["running"]:
        return JSONResponse({"error": "Stop the sweep before replacing the database."}, status_code=409)
    with open(pbc3_sweep.DB_PATH, "wb") as f:
        f.write(await file.read())
    return {"ok": True}

@app.post("/api/sweeps/upload_db")
async def sweeps_upload_db(file: UploadFile = File(...)):
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    name = os.path.basename(file.filename or "upload.db")
    if not name.endswith(".db"):
        name += ".db"
    dest = os.path.join(UPLOAD_DIR, name)
    with open(dest, "wb") as f:
        f.write(await file.read())
    return {"db_path": os.path.relpath(dest, PROJECT_DIR), "db_name": name}


@app.get("/api/benchmark/dataset")
def benchmark_dataset():
    return pbc3_benchmark.dataset_summary()


@app.get("/api/benchmark/status")
def benchmark_status():
    return pbc3_benchmark.status()


@app.post("/api/benchmark/start")
async def benchmark_start(request: Request):
    body = await request.json()
    return _guarded_benchmark_start(
        "benchmark",
        pbc3_benchmark.start,
        pbc3_benchmark.status,
        body.get("codecs"),
        body.get("n_trials", 1),
    )


@app.post("/api/benchmark/stop")
def benchmark_stop():
    return pbc3_benchmark.stop()


@app.get("/api/benchmark/results")
def benchmark_results(mp_min: float = None, mp_max: float = None):
    return pbc3_benchmark.results(mp_min, mp_max)


@app.post("/api/benchmark/reset_pbc")
def benchmark_reset_pbc():
    result = pbc3_benchmark.reset_pbc()
    if result.get("error"):
        return JSONResponse(result, status_code=409)
    return result


@app.get("/api/benchmark/download_db")
def benchmark_download_db():
    if not os.path.exists(pbc3_benchmark.DB_PATH):
        return JSONResponse({"error": "No benchmark database yet."}, status_code=404)
    return FileResponse(pbc3_benchmark.DB_PATH, filename="pbc3_benchmark.db", media_type="application/octet-stream")


@app.post("/api/benchmark/upload_db")
async def benchmark_upload_db(file: UploadFile = File(...)):
    if pbc3_benchmark.status()["running"]:
        return JSONResponse({"error": "Stop the benchmark before replacing the database."}, status_code=409)
    with open(pbc3_benchmark.DB_PATH, "wb") as f:
        f.write(await file.read())
    return {"ok": True}


@app.post("/api/match_codec")
async def match_codec(image: UploadFile = File(...), codec: str = Form("jpeg"), target_bpp: float = Form(...)):
    try:
        src = ImageOps.exif_transpose(Image.open(io.BytesIO(await image.read())))
        has_alpha = PBC3._has_alpha(src)
        img = src.convert("RGBA") if has_alpha else src.convert("RGB")
    except Exception as exc:
        return JSONResponse({"error": f"Could not read image: {exc}"}, status_code=400)
    fmt = {"jpeg": "JPEG", "jp2": "JPEG2000", "jpeg2000": "JPEG2000", "avif": "AVIF", "webp": "WEBP", "jxl": "JXL"}.get(codec.lower(), "JPEG")
    pixels = img.size[0] * img.size[1]
    enc_img = img if (has_alpha and fmt in ("WEBP", "AVIF", "JXL", "JPEG2000")) else img.convert("RGB")

    def gen():
        best = None
        for m in _match_codec_gen(enc_img, fmt, target_bpp, pixels):
            if "best" in m:
                best = m["best"]
            else:
                print(f"[match_codec] {fmt} q{m['q']:>3} -> bpp {m['bpp']:.5f}  (target {target_bpp:.5f})", flush=True)
                yield _nd({"q": m["q"], "bpp": m["bpp"]})
        if not best:
            yield _nd({"error": f"{fmt} encoding unavailable."})
            return
        rec = Image.open(io.BytesIO(best["data"]))
        print(f"[match_codec] {fmt} BEST q{best['q']} -> bpp {best['bpp']:.5f}  {len(best['data'])/1024:.2f} KB", flush=True)
        yield _nd({"done": True, "image": _png_b64(rec.convert("RGBA") if has_alpha else rec.convert("RGB")),
                   "q": best["q"], "bpp": best["bpp"],
                   "mse": round(_aligned_mse(img, rec), 2),
                   "size_kb": round(len(best["data"]) / 1024, 2)})

    return StreamingResponse(gen(), media_type="application/x-ndjson")

@app.post("/api/animate")
async def animate(file: UploadFile = File(...), original: UploadFile = File(None)):
    import traceback
    raw = await file.read()
    orig_img = None
    if original is not None:
        try:
            orig_img = ImageOps.exif_transpose(Image.open(io.BytesIO(await original.read()))).convert("RGB")
        except Exception:
            orig_img = None
    fd, base = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    out_path = None
    try:
        out_path = animate_pbc3(
            bytes(raw), output_path=base, fps=3,
            separated_channels=True, show_errors=orig_img is not None,
            original_image=orig_img, output_size=(1920, 1080),
        )
        with open(out_path, "rb") as f:
            data = f.read()
        media = "image/gif" if out_path.lower().endswith(".gif") else "video/mp4"
        print(f"[animate] ok -> {os.path.basename(out_path)} ({len(data)/1024:.1f} KB)", flush=True)
        return Response(content=data, media_type=media)
    except Exception as exc:
        traceback.print_exc()
        print(f"[animate] FAILED: {exc}", flush=True)
        return JSONResponse({"error": f"Animation failed: {exc}"}, status_code=400)
    finally:
        for p in {base, out_path}:
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass

BOOT_ID = f"{time.time():.3f}"


@app.on_event("startup")
def _startup():
    print("[startup] warming production PBC3 paths...", flush=True)
    preload_numba(os.path.join(_HERE, "patch_policy.npz"))
    print("[startup] done.", flush=True)
    print(f"[startup] BOOT_ID={BOOT_ID}", flush=True)


@app.get("/api/health")
def health():
    return {"ok": True, "boot_id": BOOT_ID}


@app.head("/")
def head_root():
    return PlainTextResponse("")


@app.get("/healthz")
def healthz():
    return PlainTextResponse("ok")


@app.head("/healthz")
def head_healthz():
    return PlainTextResponse("")


@app.post("/api/sweeps/run/reset_status")
def reset_sweep_status():
    pbc3_sweep.reset_status()
    return pbc3_sweep.status()


class NoCacheStaticFiles(StaticFiles):
    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        if path.endswith((".html", ".js", ".css")):
            response.headers["Cache-Control"] = "no-store, max-age=0, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response

train_api.register(app)
app.mount("/", NoCacheStaticFiles(directory="static", html=True), name="static")
