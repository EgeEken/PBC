import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from fastapi import File, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse


ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "training_space" / "runs"
RUNS.mkdir(parents=True, exist_ok=True)
IMAGE_FOLDERS = {"train": ROOT / "hpt_data", "validation": ROOT / "hpt_data_val"}
MODEL_FOLDERS = (ROOT / "pbc3_students", RUNS)
LOCK = threading.Lock()
PROCESS = None
CURRENT = {}


def _safe_model_path(value):
    path = (ROOT / value).resolve() if not os.path.isabs(value) else Path(value).resolve()
    if not any(path == folder.resolve() or folder.resolve() in path.parents for folder in MODEL_FOLDERS):
        return None
    return path if path.is_file() else None


def _output_path(value):
    name = Path(value or "rl_training.npz").name
    if not name.endswith(".npz"):
        name += ".npz"
    return (RUNS / name).resolve()


def _json_lines(log_path):
    rows = []
    if not log_path.exists():
        return rows
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        candidate = line.strip()
        if candidate.startswith("initial "):
            candidate = candidate[8:]
        try:
            row = json.loads(candidate)
        except Exception:
            continue
        if "epoch" in row or "reward" in row:
            rows.append(row)
    return rows


def _refresh_process():
    global PROCESS
    if PROCESS is not None and PROCESS.poll() is not None:
        CURRENT["return_code"] = PROCESS.returncode
        PROCESS = None


def status():
    with LOCK:
        _refresh_process()
        output = Path(CURRENT["output"]) if CURRENT.get("output") else None
        log = Path(CURRENT["log"]) if CURRENT.get("log") else None
        history_path = Path(f"{output}.json") if output else None
        manifest_path = Path(f"{output}.checkpoints.json") if output else None
        history = []
        checkpoint_groups = {"mse": [], "bpp": [], "work": [], "overall": []}
        if history_path and history_path.exists():
            try:
                saved = json.loads(history_path.read_text(encoding="utf-8"))
                initial = saved.get("initial")
                if initial:
                    history.append(initial if "validation" in initial else {"validation": initial})
                history.extend(saved.get("history", []))
            except Exception:
                pass
        if not history:
            history = _json_lines(log) if log else []
        if manifest_path and manifest_path.exists():
            try:
                checkpoint_groups = json.loads(manifest_path.read_text(encoding="utf-8")).get("groups", checkpoint_groups)
            except Exception:
                pass
        return {
            "running": PROCESS is not None,
            "return_code": CURRENT.get("return_code"),
            "output": str(output.relative_to(ROOT)) if output else None,
            "log": str(log.relative_to(ROOT)) if log else None,
            "started": CURRENT.get("started"),
            "spec": CURRENT.get("spec", {}),
            "history": history,
            "checkpoint_groups": checkpoint_groups,
            "log_tail": log.read_text(encoding="utf-8", errors="replace")[-10000:] if log and log.exists() else "",
        }


def start(spec):
    global PROCESS
    with LOCK:
        _refresh_process()
        if PROCESS is not None:
            return {"error": "A training run is already active."}
        presets = str(spec.get("presets", "high_quality"))
        selected = [p.strip() for p in presets.split(",") if p.strip()]
        allowed = {"compression", "balanced", "quality", "high_quality"}
        if not selected or any(p not in allowed for p in selected):
            return {"error": "Presets must be compression, balanced, quality, or high_quality."}
        init = _safe_model_path(str(spec.get("init", "pbc3_students/patch_policy_f26_a20_h512_l2_e1200.npz")))
        if init is None:
            return {"error": "Initial model was not found in pbc3_students or training_space/runs."}
        default_output = f"rl_{'_'.join(selected)}_top2.npz"
        output = _output_path(spec.get("output") or default_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        log = Path(f"{output}.log")
        resume = Path(f"{output}.resume.pt")
        command = [
            sys.executable, "pbc3_rl_a20.py",
            "--presets", ",".join(selected),
            "--epochs", str(int(spec.get("epochs", 100))),
            "--batch", str(int(spec.get("batch", 4))),
            "--init", os.fspath(init),
            "--out", os.fspath(output),
            "--rate-weight", str(float(spec.get("rate_weight", 1.5))),
            "--speed-weight", str(float(spec.get("speed_weight", 0.15))),
            "--temperature", str(float(spec.get("temperature", 0.9))),
            "--entropy-weight", str(float(spec.get("entropy_weight", 0.003))),
            "--kl-weight", str(float(spec.get("kl_weight", 0.015))),
            "--quality-weight", str(float(spec.get("quality_weight", 2.0))),
            "--worse-quality-weight", str(float(spec.get("worse_quality_weight", 7.0))),
        ]
        if bool(spec.get("resume", True)) and resume.exists():
            command.extend(["--resume", os.fspath(resume)])
        log.write_text("", encoding="utf-8")
        handle = log.open("a", encoding="utf-8")
        PROCESS = subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
        handle.close()
        CURRENT.clear()
        CURRENT.update({"output": str(output), "log": str(log), "started": time.time(), "spec": spec, "return_code": None})
        return {"ok": True, "output": str(output.relative_to(ROOT)), "resuming": resume.exists()}


def stop():
    with LOCK:
        _refresh_process()
        if PROCESS is None:
            return {"ok": False, "error": "No training run is active."}
        PROCESS.terminate()
        return {"ok": True}


def checkpoints():
    rows = []
    for folder in MODEL_FOLDERS:
        for path in sorted(folder.glob("*.npz"), key=lambda p: p.stat().st_mtime, reverse=True):
            rows.append({"name": path.name, "path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "modified": path.stat().st_mtime})
    return {"checkpoints": rows}


def download_checkpoint(path):
    safe = _safe_model_path(path)
    if safe is None or safe.suffix != ".npz":
        return JSONResponse({"error": "Checkpoint not found or not allowed."}, status_code=404)
    return FileResponse(safe, filename=safe.name, media_type="application/octet-stream")


async def upload_checkpoint(file: UploadFile):
    name = Path(file.filename or "checkpoint.npz").name
    if Path(name).suffix.lower() != ".npz":
        return JSONResponse({"error": "Only .npz checkpoints are supported."}, status_code=400)
    stem = Path(name).stem.replace(" ", "_")
    path = (RUNS / f"uploaded_{int(time.time())}_{stem}.npz").resolve()
    path.write_bytes(await file.read())
    return {"path": str(path.relative_to(ROOT)), "name": path.name}


def register(app):
    @app.get("/api/train/status")
    def train_status():
        return status()

    @app.post("/api/train/start")
    async def train_start(request: Request):
        result = start(await request.json())
        return JSONResponse(result, status_code=400 if result.get("error") else 200)

    @app.post("/api/train/stop")
    def train_stop():
        return stop()

    @app.get("/api/train/checkpoints")
    def train_checkpoints():
        return checkpoints()

    @app.get("/api/train/download_checkpoint")
    def train_download_checkpoint(path: str):
        return download_checkpoint(path)

    @app.post("/api/train/upload_checkpoint")
    async def train_upload_checkpoint(file: UploadFile = File(...)):
        return await upload_checkpoint(file)

    @app.get("/api/train/log")
    def train_log():
        data = status()
        path = ROOT / data["log"] if data.get("log") else None
        if not path or not path.exists():
            return PlainTextResponse("No training log is available.")
        return FileResponse(path, filename=path.name, media_type="text/plain")
