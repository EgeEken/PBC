"""PBC3 sweep runner + store, designed to run headless on the HF Space CPU.

One mixed Optuna study ("pbc3") in pbc3_sweep.db, three MINIMIZED objectives
(seconds, bpp, mse) averaged over every image in hpt_data/. Per-image rows and the
resolved config live in trial user_attrs so the analyzer can re-aggregate by
megapixel range. JPEG/AVIF/WEBP baselines are logged once as tagged trials.
Start/stop controlled from the web Sweep Runner."""

import glob, io, itertools, math, os, threading, time

import numpy as np
import optuna
from PIL import Image, ImageOps

from PBC3 import PBC3, PBC3Config
try:
    import pillow_jxl  # noqa: F401 — registers JXL with PIL
except Exception:
    pillow_jxl = None

optuna.logging.set_verbosity(optuna.logging.WARNING)

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "hpt_data")
DB_PATH = os.path.join(PROJECT_DIR, "pbc3_sweep.db")
STORAGE = f"sqlite:///{DB_PATH}"
STUDY = "pbc3"
DIRECTIONS = ("minimize", "minimize", "minimize")
METRIC_NAMES = ("Seconds", "Compression (bpp)", "MSE")
IMAGE_EXTS = ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp")

# JPEG is a 1-95 scale (q=0 clamps to 1); AVIF/WEBP are true 0-100.
JPEG_QUALITIES = (1, 3, 5, 10, 20, 40, 70, 95)
AVIF_QUALITIES = (0, 1, 3, 5, 10, 20, 40, 70, 95)
WEBP_QUALITIES = (0, 1, 3, 5, 10, 20, 40, 70, 95)
JP2_QUALITIES = (0, 1, 3, 5, 10, 20, 40, 70, 95)
JXL_QUALITIES = (0, 1, 3, 5, 10, 20, 40, 70, 95)
DOWNSAMPLE_FACTORS = (1, 1.5, 2, 3, 4, 5, 8, 10, 15, 20, 32, 64, 128)

SEARCH_SPACE = [
    ("patch_count", "cat", [1, 5, 10, 25, 50, 100, 200], True),
    ("search_depth", "cat", [20, 50, 100, 500, 1000], True),
    ("top_k", "cat", [1, 3, 5, 10, 20], False),
    ("anchor_block_size", "cat", [2, 4, 8, 16, 32], False),
    ("cell_sizes_per_candidate", "cat", [1, 2, 3, 4], False),
    ("q_init", "cat", [0.1, 0.3, 0.5, 0.7, 0.9], False),
    ("q_start", "cat", [0.1, 0.3, 0.5, 0.7, 0.9], False),
    ("q_end", "cat", [0.1, 0.3, 0.5, 0.7, 0.9], False),
    ("patch_palette_bitcount", "cat", [1, 2, 3, 4, 5], False),
    ("mask_size", "cat", [4, 6, 8, 16], False),
    ("auto_downsample_max_pixels", "cat", [50_000, 100_000, 250_000], True),
    ("warmup_ratio", "cat", [0.0, 0.2, 0.5, 0.8, 1.0]),
    ("warm_downsample_max_pixels", "cat", [250_000, 500_000, 1_000_000], True),
    ("learned_filler_q", "float", 0.01, 0.99),
]

_DEFAULTS = PBC3Config().__dict__
_bad = [n for n, *_ in SEARCH_SPACE if n not in _DEFAULTS]
assert not _bad, f"SEARCH_SPACE keys not in PBC3Config: {_bad}"
# name -> (base PBC3Config preset, fixed learned_filler_q)
PRESETS = ("compression", "balanced", "quality", "high_quality")
LEARNED_MODEL = os.path.join(PROJECT_DIR, "patch_policy.npz")

_LOCK = threading.Lock()
_STOP = threading.Event()
_THREAD = None
_STATE = {"running": False, "mode": None, "done": 0, "total": None,
          "current": None, "started": None, "error": None, "log": []}


def _log(msg):
    with _LOCK:
        _STATE["log"] = (_STATE["log"] + [f"{time.strftime('%H:%M:%S')} {msg}"])[-5000:]
    print(f"[sweep] {msg}", flush=True)


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


def dataset():
    paths = sorted(p for ext in IMAGE_EXTS for p in glob.glob(os.path.join(DATA_DIR, ext)))
    images = []
    for p in paths:
        img = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
        arr = np.asarray(img)
        h, w = arr.shape[:2]
        images.append({"name": os.path.basename(p), "mp": round(w * h / 1e6, 4), "arr": arr})
    return images


def _mse(a, b):
    return float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))


def _avg(rows):
    return [float(np.mean([r[k] for r in rows])) for k in ("seconds", "bpp", "mse")]


def _is_pbc3_trial(t):
    cfg = t.user_attrs.get("config") or {}
    kind = t.user_attrs.get("kind") or ("baseline" if cfg.get("codec") else "pbc3")
    return kind == "pbc3"


def _is_pareto_candidate(study, values):
    vals = np.asarray(values, dtype=np.float64)
    for t in study.trials:
        if not _is_pbc3_trial(t) or t.state != optuna.trial.TrialState.COMPLETE or not t.values or len(t.values) != 3:
            continue
        other = np.asarray(t.values, dtype=np.float64)
        if np.all(other <= vals) and np.any(other < vals):
            return False
    return True


def _log_result(prefix, values, pareto=None):
    seconds, bpp, mse = values
    msg = f"{prefix} | MSE {mse:.3f} | bpp {bpp:.5f} | speed {seconds:.3f}s"
    if pareto is not None:
        msg += f" | pareto {'yes' if pareto else 'no'}"
    _log(msg)


def suggest_config(trial):
    cfg = {}
    for name, kind, *a in SEARCH_SPACE:
        if kind == "int":
            cfg[name] = trial.suggest_int(name, a[0], a[1], log=a[2])
        elif kind == "float":
            cfg[name] = trial.suggest_float(name, a[0], a[1])
        else:
            cfg[name] = trial.suggest_categorical(name, a[0])
    return cfg


def _coerce(params):
    out = {}
    for k, v in params.items():
        d = _DEFAULTS.get(k)
        if isinstance(d, bool):
            out[k] = v if isinstance(v, bool) else str(v).lower() in ("true", "1")
        elif isinstance(d, int) and not isinstance(d, bool):
            out[k] = int(v)
        elif isinstance(d, float):
            out[k] = float(v)
        else:
            out[k] = v
    return out


def _eval_pbc3(cfg, images, on_image=None):
    rows = []
    for im in images:
        arr = im["arr"]
        pixels = arr.shape[0] * arr.shape[1]
        res = PBC3.compress(Image.fromarray(arr), config=PBC3Config(**cfg))
        recon = np.asarray(res.image.convert("RGB").resize((arr.shape[1], arr.shape[0])))
        rows.append({"name": im["name"], "mp": im["mp"], "seconds": float(res.encode_seconds),
                     "bpp": float(res.total_bits / pixels), "mse": _mse(arr, recon)})
        if on_image:
            on_image()
    return rows


def _jp2_rate(q):
    q = max(0.0, min(95.0, float(q))) / 95.0
    return 200.0 ** (1.0 - q)


def _encode_codec_bytes(img, fmt, q):
    buf = io.BytesIO()
    if fmt == "JPEG2000":
        img.save(buf, format="JPEG2000", quality_mode="rates", quality_layers=[_jp2_rate(q)])
    else:
        img.save(buf, format=fmt, quality=q)
    return buf.getvalue()


def _eval_codec(fmt, q, images, on_image=None):
    rows = []
    for im in images:
        if _STOP.is_set():
            raise InterruptedError("sweep stopped")
        arr = im["arr"]
        pixels = arr.shape[0] * arr.shape[1]
        t = time.perf_counter()
        data = _encode_codec_bytes(Image.fromarray(arr), fmt, q)
        secs = time.perf_counter() - t
        recon = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
        rows.append({"name": im["name"], "mp": im["mp"], "seconds": secs,
                     "bpp": len(data) * 8 / pixels, "mse": _mse(arr, recon)})
        if on_image:
            on_image()
    return rows

def _eval_downsample(factor, images, on_image=None):
    rows = []
    for im in images:
        if _STOP.is_set():
            raise InterruptedError("sweep stopped")
        arr = im["arr"]
        h, w = arr.shape[:2]
        pixels = h * w
        t = time.perf_counter()
        if factor == 1:
            recon, nw, nh = arr, w, h
        else:
            nw, nh = max(1, round(w / factor)), max(1, round(h / factor))
            small = Image.fromarray(arr).resize((nw, nh), Image.BICUBIC)
            recon = np.asarray(small.resize((w, h), Image.BICUBIC))
        secs = time.perf_counter() - t
        rows.append({"name": im["name"], "mp": im["mp"], "seconds": secs,
                     "bpp": nw * nh * 24 / pixels, "mse": _mse(arr, recon)})
        if on_image:
            on_image()
    return rows


def get_study():
    study = optuna.create_study(
        study_name=STUDY, storage=STORAGE, directions=list(DIRECTIONS),
        sampler=optuna.samplers.NSGAIISampler(seed=2003), load_if_exists=True)
    try:
        study.set_metric_names(list(METRIC_NAMES))
    except Exception:
        pass
    return study


def _add_completed(study, values, attrs):
    study.add_trial(optuna.trial.create_trial(
        state=optuna.trial.TrialState.COMPLETE, values=list(values), user_attrs=attrs))


def ensure_baselines(study, images):
    have = {t.user_attrs.get("baseline") for t in study.trials}
    for fmt, qs in (("JPEG", JPEG_QUALITIES), ("JPEG2000", JP2_QUALITIES), ("AVIF", AVIF_QUALITIES),
                    ("WEBP", WEBP_QUALITIES), ("JXL", JXL_QUALITIES)):
        for q in qs:
            if _STOP.is_set():
                return
            tag = f"{fmt.lower()}_q{q}"
            if tag in have:
                continue
            _log(f"baseline {tag} started")
            try:
                _new_bar()
                rows = _eval_codec(fmt, q, images, on_image=_bar_tick)
            except InterruptedError:
                _log(f"baseline {tag} stopped")
                return
            except Exception as e:
                _log(f"baseline {tag} failed: {e}")
                continue
            values = _avg(rows)
            _add_completed(study, values, {
                "baseline": tag, "kind": "baseline",
                "config": {"codec": fmt, "q": q}, "per_image": rows})
            _log_result(f"baseline {tag} done", values)
    for f in DOWNSAMPLE_FACTORS:
        if _STOP.is_set():
            return
        tag = f"downsample_f{f}"
        if tag in have:
            continue
        _log(f"baseline {tag} started")
        try:
            _new_bar()
            rows = _eval_downsample(f, images, on_image=_bar_tick)
        except InterruptedError:
            _log(f"baseline {tag} stopped")
            return
        except Exception as e:
            _log(f"baseline {tag} failed: {e}")
            continue
        values = _avg(rows)
        _add_completed(study, values, {
            "baseline": tag, "kind": "baseline",
            "config": {"codec": "DOWNSAMPLE", "q": f}, "per_image": rows})
        _log_result(f"baseline {tag} done", values)
    if "png" not in have and not _STOP.is_set():
        _log("baseline png started")
        try:
            _new_bar()
            rows = _eval_codec("PNG", 0, images, on_image=_bar_tick)
        except InterruptedError:
            _log("baseline png stopped")
            return
        except Exception as e:
            _log(f"baseline png failed: {e}")
        else:
            values = _avg(rows)
            _add_completed(study, values, {
                "baseline": "png", "kind": "baseline",
                "config": {"codec": "PNG", "q": 0}, "per_image": rows})
            _log_result("baseline png done", values)


def ensure_presets(study, images):
    if not os.path.exists(LEARNED_MODEL):
        _log("presets skipped: patch_policy.npz not found")
        return
    have = {t.user_attrs.get("baseline") for t in study.trials}
    for name in PRESETS:
        if _STOP.is_set():
            return
        tag = f"preset_{name}"
        if tag in have:
            continue
        cfg = dict(vars(getattr(PBC3Config, name)()))
        cfg.update(learned_filler_candidates=1, learned_filler_top_k=1,
                   learned_filler_model_path=LEARNED_MODEL)
        _log(f"preset {name} started")
        try:
            _new_bar()
            rows = _eval_pbc3(cfg, images, on_image=_bar_tick)
        except Exception as e:
            _log(f"preset {name} failed: {e}")
            continue
        values = _avg(rows)
        _add_completed(study, values,
                       {"kind": "pbc3", "baseline": tag, "preset": name,
                        "config": cfg, "per_image": rows})
        _log_result(f"preset {name} done", values, _is_pareto_candidate(study, values))


def _grid_trials(spec):
    grid = {k: v for k, v in (spec.get("grid") or {}).items() if v}
    keys = list(grid)
    combos = [dict(zip(keys, vals)) for vals in itertools.product(*[grid[k] for k in keys])]
    return combos or [{}]


def start(mode, spec=None):
    global _THREAD
    with _LOCK:
        if _STATE["running"]:
            return {"error": "A sweep is already running."}
    if not os.path.isdir(DATA_DIR):
        return {"error": "Dataset folder missing: hpt_data/"}
    _STOP.clear()
    _THREAD = threading.Thread(target=_run, args=(mode, spec or {}), daemon=True)
    _THREAD.start()
    return {"started": True, "mode": mode}


def stop():
    _STOP.set()
    return {"stopping": True}

def reset_status():
    _STOP.set()
    with _LOCK:
        _STATE.update(
            running=False,
            mode=None,
            done=0,
            total=None,
            current=None,
            started=None,
            error=None,
            log=[],
        )


def _run(mode, spec):
    with _LOCK:
        _STATE.update(running=True, mode=mode, done=0, total=None,
                      current=None, started=time.time(), error=None)
    try:
        images = dataset()
        if not images:
            raise RuntimeError("No images found in hpt_data/")
        _log(f"loaded {len(images)} images")
        study = get_study()
        ensure_baselines(study, images)
        if _STOP.is_set():
            return
        ensure_presets(study, images)
        if _STOP.is_set():
            return
        if mode == "grid":
            trials = _grid_trials(spec)
            with _LOCK:
                _STATE["total"] = len(trials)
            for i, params in enumerate(trials):
                if _STOP.is_set():
                    break
                cfg = _coerce(params)
                with _LOCK:
                    _STATE["current"] = cfg
                _log(f"grid trial {i + 1}/{len(trials)} started")
                _new_bar()
                rows = _eval_pbc3(cfg, images, on_image=_bar_tick)
                values = _avg(rows)
                _add_completed(study, values,
                               {"kind": "pbc3", "grid": True, "config": cfg, "per_image": rows})
                _log_result(f"grid trial {i + 1}/{len(trials)} done", values, _is_pareto_candidate(study, values))
                with _LOCK:
                    _STATE["done"] = i + 1
        else:
            def objective(trial):
                cfg = suggest_config(trial)
                cfg.update(learned_filler_enabled=True, learned_filler_candidates=1,
                           learned_filler_top_k=1, learned_filler_model_path=LEARNED_MODEL)
                with _LOCK:
                    _STATE["current"] = cfg
                _log(f"optimizer trial {trial.number} started")
                _new_bar()
                rows = _eval_pbc3(cfg, images, on_image=_bar_tick)
                values = _avg(rows)
                trial.set_user_attr("kind", "pbc3")
                trial.set_user_attr("config", cfg)
                trial.set_user_attr("per_image", rows)
                _log_result(f"optimizer trial {trial.number} done", values, _is_pareto_candidate(study, values))
                return values
            n = 0
            while not _STOP.is_set():
                study.optimize(objective, n_trials=1, catch=(Exception,))
                n += 1
                with _LOCK:
                    _STATE["done"] = n
    except Exception as e:
        with _LOCK:
            _STATE["error"] = str(e)
        _log(f"ERROR: {e}")
    finally:
        with _LOCK:
            _STATE["running"] = False
            _STATE["current"] = None
        _log("sweep stopped")