import glob
import io
import math
import os
import sqlite3
import threading
import time

import numpy as np
from PIL import Image, ImageOps

from PBC3 import PBC3, preload_numba
from pbc3_types import PBC3Config

try:
    import pillow_jxl  # noqa: F401
except Exception:
    pillow_jxl = None

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "hpt_data")
DB_PATH = os.path.join(PROJECT_DIR, "pbc3_benchmark.db")
IMAGE_EXTS = ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp")

PBC3_PRESETS = ("compression", "balanced", "quality", "high_quality")
JPEG_QUALITIES = (1, 3, 5, 10, 20, 40, 70, 95)
JP2_QUALITIES = (0, 1, 3, 5, 10, 20, 40, 70, 95)
WEBP_QUALITIES = (0, 1, 3, 5, 10, 20, 40, 70, 95)
JXL_QUALITIES = (0, 1, 3, 5, 10, 20, 40, 70, 95)
AVIF_QUALITIES = (0, 1, 3, 5, 10, 20, 40, 70, 95)
CODECS = ("PBC3", "JPEG", "JPEG2000", "WEBP", "JXL", "AVIF", "PNG")

_LOCK = threading.Lock()
_STOP = threading.Event()
_THREAD = None
_STATE = {"running": False, "done": 0, "total": None, "current": None, "started": None, "error": None, "log": []}


def _log(msg):
    with _LOCK:
        _STATE["log"] = (_STATE["log"] + [f"{time.strftime('%H:%M:%S')} {msg}"])[-5000:]
    print(f"[benchmark] {msg}", flush=True)


def _new_bar():
    with _LOCK:
        _STATE["log"] = (_STATE["log"] + [""])[-5000:]


def _bar_tick():
    with _LOCK:
        if _STATE["log"]:
            _STATE["log"][-1] += "|"


def status():
    with _LOCK:
        s = dict(_STATE)
        s["log"] = list(_STATE["log"])
        return s


def _connect():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    con.execute("""
        CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL NOT NULL,
            run_id TEXT NOT NULL,
            trial INTEGER NOT NULL,
            codec TEXT NOT NULL,
            variant TEXT NOT NULL,
            q REAL,
            image TEXT NOT NULL,
            mp REAL NOT NULL,
            encode_seconds REAL NOT NULL,
            decode_seconds REAL NOT NULL,
            bpp REAL NOT NULL,
            mse REAL NOT NULL,
            psnr REAL NOT NULL
        )
    """)
    con.execute("CREATE INDEX IF NOT EXISTS idx_results_codec ON results(codec, variant, q)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_results_mp ON results(mp)")
    con.commit()
    return con


def dataset():
    paths = sorted(p for ext in IMAGE_EXTS for p in glob.glob(os.path.join(DATA_DIR, ext)))
    images = []
    for p in paths:
        try:
            img = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
            arr = np.asarray(img)
            h, w = arr.shape[:2]
            images.append({"name": os.path.basename(p), "mp": round(w * h / 1e6, 4), "arr": arr})
        except Exception as e:
            _log(f"skipped {os.path.basename(p)}: {e}")
    return images


def dataset_summary():
    return {"count": len(dataset()), "images": [{"name": im["name"], "mp": im["mp"]} for im in dataset()]}


def _mse(a, b):
    return float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))


def _psnr(mse):
    if mse <= 0:
        return 99.0
    return float(10.0 * math.log10((255.0 * 255.0) / mse))


def _jp2_rate(q):
    q = max(0.0, min(95.0, float(q))) / 95.0
    return 200.0 ** (1.0 - q)


def _encode_codec_bytes(img, fmt, q):
    buf = io.BytesIO()
    if fmt == "JPEG2000":
        img.save(buf, format="JPEG2000", quality_mode="rates", quality_layers=[_jp2_rate(q)])
    elif fmt == "PNG":
        img.save(buf, format="PNG")
    else:
        img.save(buf, format=fmt, quality=int(q))
    return buf.getvalue()


def _eval_pbc3(arr, preset):
    pixels = arr.shape[0] * arr.shape[1]
    cfg = getattr(PBC3Config, preset)()
    t = time.perf_counter()
    res = PBC3.compress(Image.fromarray(arr), config=cfg)
    enc = time.perf_counter() - t
    t = time.perf_counter()
    dec = PBC3.decompress(res.data)
    dec_s = time.perf_counter() - t
    recon = np.asarray(dec.image.convert("RGB").resize((arr.shape[1], arr.shape[0])))
    mse = _mse(arr, recon)
    return enc, dec_s, float(len(res.data) * 8 / pixels), mse, _psnr(mse)


def _eval_codec(arr, fmt, q):
    pixels = arr.shape[0] * arr.shape[1]
    img = Image.fromarray(arr)
    t = time.perf_counter()
    data = _encode_codec_bytes(img, fmt, q)
    enc = time.perf_counter() - t
    t = time.perf_counter()
    rec_img = Image.open(io.BytesIO(data)).convert("RGB")
    rec_img.load()
    dec_s = time.perf_counter() - t
    recon = np.asarray(rec_img.resize((arr.shape[1], arr.shape[0])))
    mse = _mse(arr, recon)
    return enc, dec_s, float(len(data) * 8 / pixels), mse, _psnr(mse)


def _jobs(codecs):
    jobs = []
    selected = set(codecs or CODECS)
    if "PBC3" in selected:
        jobs += [("PBC3", p, None) for p in PBC3_PRESETS]
    for codec, fmt, qs in (
        ("JPEG", "JPEG", JPEG_QUALITIES),
        ("JPEG2000", "JPEG2000", JP2_QUALITIES),
        ("WEBP", "WEBP", WEBP_QUALITIES),
        ("JXL", "JXL", JXL_QUALITIES),
        ("AVIF", "AVIF", AVIF_QUALITIES),
    ):
        if codec in selected:
            jobs += [(codec, f"q{q}", q) for q in qs]
    if "PNG" in selected:
        jobs.append(("PNG", "lossless", 0))
    return jobs


def start(codecs=None, n_trials=1):
    global _THREAD
    with _LOCK:
        if _STATE["running"]:
            return {"error": "A benchmark is already running."}
    if not os.path.isdir(DATA_DIR):
        return {"error": "Dataset folder missing: hpt_data/"}
    n_trials = max(1, min(10, int(n_trials or 1)))
    selected = [c for c in (codecs or CODECS) if c in CODECS]
    if not selected:
        return {"error": "Select at least one codec."}
    _STOP.clear()
    _THREAD = threading.Thread(target=_run, args=(selected, n_trials), daemon=True)
    _THREAD.start()
    return {"started": True, "codecs": selected, "n_trials": n_trials}


def stop():
    _STOP.set()
    return {"stopping": True}


def _insert(con, row):
    con.execute("""
        INSERT INTO results
        (ts, run_id, trial, codec, variant, q, image, mp, encode_seconds, decode_seconds, bpp, mse, psnr)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, row)


def _run(codecs, n_trials):
    with _LOCK:
        _STATE.update(running=True, done=0, total=None, current=None, started=time.time(), error=None, log=[])
    try:
        images = dataset()
        if not images:
            raise RuntimeError("No images found in hpt_data/")
        preload_numba(os.path.join(PROJECT_DIR, "patch_policy.npz"))
        jobs = _jobs(codecs)
        total = n_trials * len(jobs) * len(images)
        run_id = time.strftime("%Y%m%d_%H%M%S")
        with _LOCK:
            _STATE["total"] = total
        _log(f"loaded {len(images)} images · {len(jobs)} data points · {n_trials} trial(s)")
        done = 0
        con = _connect()
        try:
            for trial in range(1, n_trials + 1):
                for codec, variant, q in jobs:
                    if _STOP.is_set():
                        return
                    with _LOCK:
                        _STATE["current"] = {"trial": trial, "codec": codec, "variant": variant}
                    _log(f"trial {trial}/{n_trials} {codec} {variant} started")
                    _new_bar()
                    for im in images:
                        if _STOP.is_set():
                            return
                        try:
                            if codec == "PBC3":
                                enc, dec, bpp, mse, psnr = _eval_pbc3(im["arr"], variant)
                            else:
                                fmt = "PNG" if codec == "PNG" else codec
                                enc, dec, bpp, mse, psnr = _eval_codec(im["arr"], fmt, q)
                            _insert(con, (time.time(), run_id, trial, codec, variant, q, im["name"], im["mp"], enc, dec, bpp, mse, psnr))
                            con.commit()
                        except Exception as e:
                            _log(f"{codec} {variant} failed on {im['name']}: {e}")
                        done += 1
                        with _LOCK:
                            _STATE["done"] = done
                        _bar_tick()
                    _log(f"trial {trial}/{n_trials} {codec} {variant} done")
        finally:
            con.close()
    except Exception as e:
        with _LOCK:
            _STATE["error"] = str(e)
        _log(f"ERROR: {e}")
    finally:
        with _LOCK:
            _STATE["running"] = False
            _STATE["current"] = None
        _log("benchmark stopped")


def reset_pbc():
    if status()["running"]:
        return {"error": "Stop the benchmark before resetting PBC3 rows."}
    if not os.path.exists(DB_PATH):
        return {"ok": True, "deleted": 0}
    con = _connect()
    try:
        cur = con.execute("DELETE FROM results WHERE codec = ?", ("PBC3",))
        con.commit()
        return {"ok": True, "deleted": cur.rowcount}
    finally:
        con.close()


def results(mp_min=None, mp_max=None):
    if not os.path.exists(DB_PATH):
        return {"exists": False, "rows": [], "points": []}
    con = _connect()
    try:
        where, args = [], []
        if mp_min is not None:
            where.append("mp >= ?"); args.append(float(mp_min))
        if mp_max is not None:
            where.append("mp <= ?"); args.append(float(mp_max))
        w = "WHERE " + " AND ".join(where) if where else ""
        rows = [dict(r) for r in con.execute(f"SELECT * FROM results {w} ORDER BY id", args)]
        points = [dict(r) for r in con.execute(f"""
            SELECT codec, variant, q, COUNT(*) AS count, COUNT(DISTINCT image) AS images,
                   AVG(encode_seconds) AS encode_seconds, AVG(decode_seconds) AS decode_seconds,
                   AVG(bpp) AS bpp, AVG(mse) AS mse, AVG(psnr) AS psnr
            FROM results {w}
            GROUP BY codec, variant, q
            ORDER BY codec, q, variant
        """, args)]
        return {"exists": True, "rows": rows, "points": points, "count": len(rows)}
    finally:
        con.close()
