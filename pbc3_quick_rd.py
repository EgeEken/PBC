import json
import os
import subprocess
import sys
import threading
import time

import numpy as np
import optuna
from PIL import Image, ImageOps

import pbc3_sweep
from PBC3 import PBC3, preload_numba
from pbc3_types import PBC3Config

RESULTS_PATH = os.path.join(pbc3_sweep.PROJECT_DIR, "pbc3_quick_rd.json")
OLD_PBC_DIR = os.path.join(pbc3_sweep.PROJECT_DIR, "old_pbc")
PRESETS = ("compression", "balanced", "quality", "high_quality")

_LOCK = threading.Lock()
_STOP = threading.Event()
_THREAD = None
_STATE = {"running": False, "done": 0, "total": None, "current": None, "started": None, "error": None, "log": []}


def _log(msg):
    with _LOCK:
        _STATE["log"] = (_STATE["log"] + [f"{time.strftime('%H:%M:%S')} {msg}"])[-5000:]
    print(f"[quick-rd] {msg}", flush=True)


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


def stop():
    _STOP.set()
    return {"stopping": True}


def start(n_trials=1, run_old=True):
    global _THREAD
    with _LOCK:
        if _STATE["running"]:
            return {"error": "A quick RD benchmark is already running."}
    if not os.path.isdir(pbc3_sweep.DATA_DIR):
        return {"error": "Dataset folder missing: hpt_data/"}
    n_trials = max(1, min(10, int(n_trials or 1)))
    run_old = bool(run_old)
    if run_old and not os.path.exists(os.path.join(OLD_PBC_DIR, "PBC3.py")):
        return {"error": "old_pbc/PBC3.py not found. Add the old runtime files or disable live old comparison."}
    _STOP.clear()
    _THREAD = threading.Thread(target=_run, args=(n_trials, run_old), daemon=True)
    _THREAD.start()
    return {"started": True, "n_trials": n_trials, "run_old": run_old}


def _mse(a, b):
    return float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))


def _avg(rows):
    if not rows:
        return {"speed": None, "bpp": None, "mse": None}
    return {
        "speed": float(np.mean([r["seconds"] for r in rows])),
        "bpp": float(np.mean([r["bpp"] for r in rows])),
        "mse": float(np.mean([r["mse"] for r in rows])),
    }


def _eval_current_preset(name, images, trial):
    rows = []
    cfg = getattr(PBC3Config, name)()
    for im in images:
        if _STOP.is_set():
            raise InterruptedError("quick RD stopped")
        arr = im["arr"]
        pixels = arr.shape[0] * arr.shape[1]
        res = PBC3.compress(Image.fromarray(arr), config=cfg)
        recon = np.asarray(res.image.convert("RGB").resize((arr.shape[1], arr.shape[0])))
        rows.append({
            "name": im["name"], "mp": im["mp"], "trial": trial,
            "seconds": float(res.encode_seconds), "bpp": float(res.total_bits / pixels), "mse": _mse(arr, recon),
        })
        _bar_tick()
    return rows


def _eval_old_preset_subprocess(name, images, trial):
    payload = [{
        "name": im["name"],
        "path": im.get("path") or os.path.join(pbc3_sweep.DATA_DIR, im["name"]),
        "mp": im["mp"],
    } for im in images]
    code = r'''
import json, os, sys, time
import numpy as np
from PIL import Image, ImageOps
sys.path.insert(0, sys.argv[1])
from PBC3 import PBC3
try:
    from pbc3_types import PBC3Config
except Exception:
    from pbc_types import PBC3Config
from PBC3 import preload_numba
preset = sys.argv[2]
trial = int(sys.argv[3])
images = json.loads(sys.stdin.read())
preload_numba()
out = []
for im in images:
    img = ImageOps.exif_transpose(Image.open(im["path"])).convert("RGB")
    arr = np.asarray(img)
    pixels = arr.shape[0] * arr.shape[1]
    cfg = getattr(PBC3Config, preset)()
    res = PBC3.compress(img, config=cfg)
    recon = np.asarray(res.image.convert("RGB").resize((arr.shape[1], arr.shape[0])))
    mse = float(np.mean((arr.astype(np.float32) - recon.astype(np.float32)) ** 2))
    out.append({"name": im["name"], "mp": im["mp"], "trial": trial, "seconds": float(res.encode_seconds), "bpp": float(res.total_bits / pixels), "mse": mse})
print(json.dumps(out))
'''
    env = os.environ.copy()
    env["PYTHONPATH"] = OLD_PBC_DIR + os.pathsep + env.get("PYTHONPATH", "")
    p = subprocess.run(
        [sys.executable, "-c", code, OLD_PBC_DIR, name, str(trial)],
        input=json.dumps(payload), text=True, capture_output=True, cwd=pbc3_sweep.PROJECT_DIR, env=env,
    )
    if p.returncode != 0:
        raise RuntimeError(f"old_pbc {name} failed: {p.stderr[-1200:]}")
    for _ in images:
        _bar_tick()
    return json.loads(p.stdout)


def _run(n_trials, run_old):
    total = len(PRESETS) * n_trials * (2 if run_old else 1)
    with _LOCK:
        _STATE.update(running=True, done=0, total=total, current=None, started=time.time(), error=None, log=[])
    done = 0
    try:
        images = pbc3_sweep.dataset()
        if not images:
            raise RuntimeError("No images found in hpt_data/")
        preload_numba(os.path.join(pbc3_sweep.PROJECT_DIR, "patch_policy.npz"))
        _log(f"loaded {len(images)} images · {n_trials} trial(s)" + (" · live old enabled" if run_old else ""))
        out = {"created": time.time(), "n_trials": n_trials, "presets": []}
        for trial in range(1, n_trials + 1):
            for source, evaluator in (("old_live", _eval_old_preset_subprocess), ("new", _eval_current_preset)):
                if source == "old_live" and not run_old:
                    continue
                for name in PRESETS:
                    if _STOP.is_set():
                        raise InterruptedError("quick RD stopped")
                    with _LOCK:
                        _STATE["current"] = {"source": source, "preset": name, "trial": trial}
                    _log(f"trial {trial}/{n_trials} {source} {name} started")
                    _new_bar()
                    rows = evaluator(name, images, trial)
                    values = _avg(rows)
                    out["presets"].append({"source": source, "kind": "preset", "preset": name, "trial": trial, "rows": rows, **values})
                    done += 1
                    with _LOCK:
                        _STATE["done"] = done
                    _log(f"trial {trial}/{n_trials} {source} {name} done | MSE {values['mse']:.3f} | bpp {values['bpp']:.5f} | speed {values['speed']:.3f}s")
                    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
                        json.dump(out, f)
    except Exception as e:
        with _LOCK:
            _STATE["error"] = str(e)
        _log(f"ERROR: {e}")
    finally:
        with _LOCK:
            _STATE["running"] = False
            _STATE["current"] = None
        _log("quick RD stopped")


def _trial_row(t, mp_min=None, mp_max=None):
    ua = t.user_attrs
    cfg = ua.get("config") or {}
    rows = ua.get("per_image") or []
    if mp_min is not None:
        rows = [r for r in rows if float(r.get("mp", 0)) >= float(mp_min)]
    if mp_max is not None:
        rows = [r for r in rows if float(r.get("mp", 0)) <= float(mp_max)]
    if rows:
        vals = _avg(rows)
    elif t.values and len(t.values) == 3:
        vals = {"speed": float(t.values[0]), "bpp": float(t.values[1]), "mse": float(t.values[2])}
    else:
        return None
    kind = ua.get("kind")
    if ua.get("preset"):
        return {"source": "old", "kind": "preset", "preset": ua.get("preset"), **vals}
    if cfg.get("codec"):
        return {"source": "codec", "kind": "codec", "codec": cfg.get("codec"), "q": cfg.get("q"), **vals}
    if kind == "baseline" and ua.get("baseline"):
        return {"source": "codec", "kind": "codec", "codec": str(ua.get("baseline")).split("_")[0].upper(), "q": cfg.get("q"), **vals}
    return None


def _load_old_db(mp_min=None, mp_max=None):
    if not os.path.exists(pbc3_sweep.DB_PATH):
        return {"exists": False, "rows": []}
    try:
        study = optuna.load_study(study_name=pbc3_sweep.STUDY, storage=pbc3_sweep.STORAGE)
    except Exception:
        return {"exists": False, "rows": []}
    rows = []
    for t in study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue
        ua = t.user_attrs
        baseline = ua.get("baseline") or ""
        cfg = ua.get("config") or {}
        if ua.get("preset") or baseline.startswith("preset_") or cfg.get("codec"):
            r = _trial_row(t, mp_min, mp_max)
            if r:
                rows.append(r)
    return {"exists": True, "rows": rows}


def _load_live(mp_min=None, mp_max=None):
    if not os.path.exists(RESULTS_PATH):
        return []
    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    grouped = {}
    for p in data.get("presets", []):
        rows = p.get("rows") or []
        if mp_min is not None:
            rows = [r for r in rows if float(r.get("mp", 0)) >= float(mp_min)]
        if mp_max is not None:
            rows = [r for r in rows if float(r.get("mp", 0)) <= float(mp_max)]
        key = (p.get("source"), p.get("preset"))
        grouped.setdefault(key, []).extend(rows)
    out = []
    for (source, preset), rows in grouped.items():
        out.append({"source": source, "kind": "preset", "preset": preset, **_avg(rows)})
    return out


def results(mp_min=None, mp_max=None):
    old = _load_old_db(mp_min, mp_max)
    live = _load_live(mp_min, mp_max)
    return {
        "old_db_exists": old["exists"],
        "new_exists": os.path.exists(RESULTS_PATH),
        "old_pbc_exists": os.path.exists(os.path.join(OLD_PBC_DIR, "PBC3.py")),
        "rows": old["rows"] + live,
    }
