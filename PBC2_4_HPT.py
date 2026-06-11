"""
PBC2.4 Hyperparameter Tuning (HPT)
===================================

Multi-objective optimization of PBC2.4 compression parameters over three axes:

    - Speed        : encode seconds per megapixel        (minimize)
    - Compression  : bits per pixel (bpp)                 (minimize)
    - Quality      : MS-SSIM                              (maximize)

The result is a Pareto front per target resolution. Studies are persisted in a
SQLite database so runs can be stopped and resumed freely.

Workflow
--------
    # 1. Download a DIV8K subset once and pre-render it to each target resolution.
    python PBC2_4_HPT.py prepare --dataset-size 200

    # 2. Run / resume the sweep (sequential trials, isolated timing).
    python PBC2_4_HPT.py run --trials 1000 --timeout 14400

    # 3. Build Pareto sets, CSVs and interactive plots.
    python PBC2_4_HPT.py analyze

Monitor live with:  optuna-dashboard sqlite:///pbc_hpt.db
"""

import argparse
import glob
import json
import math
import os
import time

import numpy as np
import cv2
import pandas as pd
from PIL import Image
import optuna

from PBC2_4 import PBC, PBC2Config

optuna.logging.set_verbosity(optuna.logging.WARNING)

OBJECTIVES = ("speed_s_per_mp", "bpp", "quality")
DIRECTIONS = ("minimize", "minimize", "maximize")
DEFAULT_TARGETS = (0.5, 2.0, 6.0, 12.0, 18.0)
DEFAULT_SOURCE = "Iceclear/DIV8K_TrainingSet"
DEFAULT_DB = "sqlite:///pbc_hpt.db"
DEFAULT_DATA_DIR = "hpt_data"
DEFAULT_OUT_DIR = "hpt_results"


# =============================== METRICS ===========================================

def mse_metric(a, b):
    a = np.asarray(a.convert("RGB") if isinstance(a, Image.Image) else a, dtype=np.float64)
    b = np.asarray(b.convert("RGB") if isinstance(b, Image.Image) else b, dtype=np.float64)
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
    if min_val > max_val:
        min_val, max_val = max_val, min_val
    if min_val == max_val:
        max_val = min_val + 1
    count = 2 ** bit_count
    vals = np.linspace(min_val, max_val, count, dtype=int).tolist()
    if mode == "Stable_Uniform":
        closest = min(vals, key=lambda x: abs(x))
        if abs(closest) > 1:
            vals.remove(max(vals, key=lambda x: abs(x)))
            vals.append(0)
    return sorted(set(vals)) or [0]


# =============================== DATA PREP =========================================

def _load_source_images(source, raw_dir, n):
    os.makedirs(raw_dir, exist_ok=True)
    if os.path.isdir(source):
        paths = sorted(glob.glob(os.path.join(source, "*")))[:n]
        if not paths:
            raise SystemExit(f"No images found in source dir: {source}")
        return paths

    from datasets import load_dataset
    print(f"Streaming up to {n} images from HF dataset '{source}' ...")
    ds = load_dataset(source, split="train", streaming=True)
    paths = []
    for i, ex in enumerate(ds):
        if len(paths) >= n:
            break
        img = ex.get("image") or ex[next(iter(ex))]
        p = os.path.join(raw_dir, f"{i:04d}.png")
        img.convert("RGB").save(p)
        paths.append(p)
        if (i + 1) % 25 == 0:
            print(f"  fetched {len(paths)} ...")
    return paths


def _resize_to_mp(img, mp):
    w, h = img.size
    scale = math.sqrt((mp * 1e6) / (w * h))
    if scale >= 1.0:  # never upsample (would inject resampling bias)
        return img
    return img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)


def prepare_data(source, data_dir, n, targets):
    raw_dir = os.path.join(data_dir, "_raw")
    paths = _load_source_images(source, raw_dir, n)
    for mp in targets:
        out = os.path.join(data_dir, f"{mp}MP")
        os.makedirs(out, exist_ok=True)
        for i, p in enumerate(paths):
            img = _resize_to_mp(Image.open(p).convert("RGB"), mp)
            img.save(os.path.join(out, f"{i:04d}.png"))
        print(f"Rendered {len(paths)} images at {mp} MP -> {out}")


def _load_target_images(data_dir, mp, k):
    folder = os.path.join(data_dir, f"{mp}MP")
    paths = sorted(glob.glob(os.path.join(folder, "*.png")))
    if not paths:
        raise SystemExit(f"No prepared images at {folder}. Run `prepare` first.")
    return [(p, np.asarray(Image.open(p).convert("RGB"))) for p in paths[:k]]

def _jsonable(v):
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, tuple):
        return list(v)
    if isinstance(v, list):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    return v


# =============================== SEARCH SPACE / EVAL ===============================

def suggest_config(trial):
    ss = trial.suggest_float("size_start", 0.02, 0.5)
    se = trial.suggest_float("size_end", 0.005, ss)
    cfg = dict(
        stroke_count=trial.suggest_int("stroke_count", 1000, 150000, log=True),
        size_range=(ss, se),
        decay_cutoff=trial.suggest_float("decay_cutoff", 0.0, 1.0),
        decay_softness=trial.suggest_float("decay_softness", 0.0, 1.0),
        decay_progress=trial.suggest_float("decay_progress", 0.0, 1.0),
        focus_strokes=trial.suggest_int("focus_strokes", 50, 500, log=True),
        focus_warmup=trial.suggest_float("focus_warmup", 0.5, 1.0),
        focus_max_bits=trial.suggest_int("focus_max_bits", 0, 12),
        focus_padding=trial.suggest_int("focus_padding", 0, 16),
        focus_criteria=trial.suggest_categorical("focus_criteria", ["Sum", "Max", "Min"]),
        color_space=trial.suggest_categorical("color_space", ["RGB", "YCbCr"]),
        downsample_rate=trial.suggest_float("downsample_rate", 1.0, 8.0),
        resample=trial.suggest_categorical("resample", ["bicubic", "lanczos", "bilinear", "box"]),
    )

    cycle = trial.suggest_categorical("channel_cycle", ["Smart", "Strict", "Balanced", "Default", "Off"])
    if cycle == "Off":
        cfg["channel_cycle"] = False
    else:
        cfg["channel_cycle"] = cycle
        cfg["channel_cycle_strokes"] = trial.suggest_int("channel_cycle_strokes", 20, 500, log=True)
        cfg["channel_cycle_warmup"] = trial.suggest_float("channel_cycle_warmup", 0.6, 1.0)
        cfg["channel_cycle_criteria"] = trial.suggest_categorical("channel_cycle_criteria", ["Sum", "Max", "Min", "Median"])

    # downsample_initialize seeds the canvas and fully overrides the start color,
    # so start_mode is only a live parameter when initialization is off.
    if trial.suggest_categorical("downsample_initialize", [True, False]):
        cfg["downsample_initialize"] = True
        cfg["downsample_initialize_rate"] = trial.suggest_float("downsample_initialize_rate", 4.0, 32.0)
        cfg["downsample_initialize_bits"] = trial.suggest_int("downsample_initialize_bits", 4, 8)
        cfg["start_mode"] = "Average"
    else:
        cfg["downsample_initialize"] = False
        cfg["start_mode"] = trial.suggest_categorical("start_mode", ["Average", "Median", "True Median", "Black", "White"])

    # PBC_Default uses a fixed multiplier list that overrides bit-count/min/max,
    # so those are only live parameters for the generated modes.
    mult_mode = trial.suggest_categorical("mult_mode", ["Stable_Uniform", "Uniform", "PBC_Default"])
    if mult_mode in {"PBC_Default", "PBC Default", "PBCDefault"}:
        cfg["mult_list"] = (-10, 0, 5, 20)
    else:
        mult_bit_count = trial.suggest_int("mult_bit_count", 1, 7)
        mult_min = trial.suggest_int("mult_min", -60, 0)
        mult_max = trial.suggest_int("mult_max", 0, 60)
        cfg["mult_list"] = generate_multlist(mult_bit_count, mult_min, mult_max, mult_mode)
    return cfg

def _resolved_decay_value(val, stroke_count, kind):
    if val != -1:
        return float(val)
    if kind == "cutoff":
        return float(round(0.01 + (1 / 1.0000115) ** (stroke_count + 15000), 4))
    return 0.5


def _auto_trial_params(images):
    arr = images[0][1] if isinstance(images[0], tuple) else images[0]
    res = PBC.compress(arr)
    cfg = res.config

    params = {
        "stroke_count": int(cfg.stroke_count),
        "size_start": float(cfg.size_range[0]),
        "size_end": float(cfg.size_range[1]),
        "decay_cutoff": _resolved_decay_value(cfg.decay_cutoff, cfg.stroke_count, "cutoff"),
        "decay_softness": _resolved_decay_value(cfg.decay_softness, cfg.stroke_count, "softness"),
        "decay_progress": _resolved_decay_value(cfg.decay_progress, cfg.stroke_count, "progress"),
        "focus_strokes": int(cfg.focus_strokes),
        "focus_warmup": float(cfg.focus_warmup),
        "focus_max_bits": int(cfg.focus_max_bits),
        "focus_padding": int(cfg.focus_padding),
        "focus_criteria": cfg.focus_criteria,
        "color_space": cfg.color_space,
        "downsample_rate": float(cfg.downsample_rate),
        "resample": cfg.resample,
        "start_mode": cfg.start_mode,
        "channel_cycle": "Off" if not cfg.channel_cycle else cfg.channel_cycle,
        "downsample_initialize": bool(cfg.downsample_initialize),
        "mult_bit_count": 2,
        "mult_min": -10,
        "mult_max": 20,
        "mult_mode": "Stable_Uniform",
    }

    if cfg.channel_cycle:
        params.update({
            "channel_cycle_strokes": int(cfg.channel_cycle_strokes),
            "channel_cycle_warmup": float(cfg.channel_cycle_warmup),
            "channel_cycle_criteria": cfg.channel_cycle_criteria,
        })

    if cfg.downsample_initialize:
        params.update({
            "downsample_initialize_rate": float(cfg.downsample_initialize_rate),
            "downsample_initialize_bits": int(cfg.downsample_initialize_bits),
        })

    return params


def _has_auto_baseline(study):
    return any(
        t.user_attrs.get("baseline") == "auto" and t.state == optuna.trial.TrialState.COMPLETE
        for t in study.trials
    )

DOWNSAMPLE_BASELINE_RATES = (1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 6.0, 8.0, 16.0, 32.0)

def _has_baseline(study, kind):
    return any(
        t.user_attrs.get("baseline") == kind and t.state == optuna.trial.TrialState.COMPLETE
        for t in study.trials
    )


def _downsample_baseline_params(base, rate, bits=8):
    params = dict(base)
    params["stroke_count"] = 1
    params["downsample_initialize"] = True
    params["downsample_initialize_rate"] = float(rate)
    params["downsample_initialize_bits"] = int(bits)
    return params

def evaluate(cfg, images, n_repeats, artifact_dir=None):
    if artifact_dir:
        os.makedirs(artifact_dir, exist_ok=True)
        with open(os.path.join(artifact_dir, "config.json"), "w") as f:
            json.dump(_jsonable(cfg), f, indent=2)

    times, bpps, qualities = [], [], []
    rows = []

    for idx, (path, arr) in enumerate(images):
        pixels = arr.shape[0] * arr.shape[1]

        res = PBC.compress(arr, **cfg)
        recon = np.asarray(res.image.convert("RGB"))

        bpp = res.total_bits / pixels
        q = composite_quality(arr, recon)

        bpps.append(bpp)
        qualities.append(q)

        samples = [res.encode_seconds]
        for _ in range(n_repeats - 1):
            samples.append(PBC.compress(arr, **cfg).encode_seconds)

        speed = float(np.median(samples)) / (pixels / 1e6)
        times.append(speed)

        if artifact_dir:
            stem = f"img_{idx:02d}_{os.path.splitext(os.path.basename(path))[0]}"
            Image.fromarray(arr).save(os.path.join(artifact_dir, f"{stem}_input.png"))
            res.image.save(os.path.join(artifact_dir, f"{stem}_recon.png"))
            with open(os.path.join(artifact_dir, f"{stem}.pbc"), "wb") as f:
                f.write(res.data)

        rows.append({
            "image": path,
            "speed_s_per_mp": speed,
            "bpp": bpp,
            "quality": q,
            "mse": float(np.mean((arr.astype(np.float64) - recon.astype(np.float64)) ** 2)),
        })

    if artifact_dir:
        pd.DataFrame(rows).to_csv(os.path.join(artifact_dir, "per_image.csv"), index=False)

    return float(np.mean(times)), float(np.mean(bpps)), float(np.mean(qualities))


def make_objective(images, n_repeats, artifact_root=None, mp=None):
    def objective(trial):
        cfg = suggest_config(trial)

        if trial.user_attrs.get("baseline") == "auto":
            cfg["mult_list"] = (-10, 0, 5, 20)
            print(f"  running automatic baseline as trial {trial.number}")

        artifact_dir = None
        ...
        if artifact_root:
            artifact_dir = os.path.join(artifact_root, f"{mp}MP", f"trial_{trial.number:05d}")
            os.makedirs(artifact_dir, exist_ok=True)
            with open(os.path.join(artifact_dir, "params.txt"), "w") as f:
                for k, v in trial.params.items():
                    f.write(f"{k} {v}\n")
                f.write(f"mult_list {json.dumps(cfg['mult_list'])}\n")

        try:
            speed, bpp, quality = evaluate(cfg, images, n_repeats, artifact_dir)
        except Exception as exc:
            print(f"  trial {trial.number} failed: {exc}")
            raise optuna.TrialPruned()

        trial.set_user_attr("mult_list", cfg["mult_list"])
        trial.set_user_attr("config", _jsonable(cfg))
        return speed, bpp, quality

    return objective


# =============================== RUN ===============================================

def study_for(mp, storage):
    study = optuna.create_study(
        study_name=f"pbc_{mp}MP",
        storage=storage,
        directions=list(DIRECTIONS),
        sampler=optuna.samplers.NSGAIISampler(population_size=48, seed=28042003),
        load_if_exists=True,
    )
    study.set_metric_names(["Speed (s/MP)", "Compression (bpp)", "Quality (composite)"])
    return study


def run(data_dir, storage, targets, k, n_repeats, trials, timeout, warmup, artifact_root, auto_baseline):
    per_target_budget = timeout / len(targets) if timeout else None

    for mp in targets:
        print(f"\n=== Tuning {mp} MP (k={k}, n_repeats={n_repeats}) ===")
        images = _load_target_images(data_dir, mp, k)
        study = study_for(mp, storage)

        if auto_baseline:
            base = _auto_trial_params(images)
            if not _has_auto_baseline(study):
                study.enqueue_trial(dict(base), user_attrs={"baseline": "auto"}, skip_if_exists=True)
                print("  queued automatic baseline trial")
            for rate in DOWNSAMPLE_BASELINE_RATES:
                kind = f"downsample_{rate:g}"
                if not _has_baseline(study, kind):
                    study.enqueue_trial(
                        _downsample_baseline_params(base, rate),
                        user_attrs={"baseline": kind}, skip_if_exists=True,
                    )
            print(f"  queued downsample baselines (stroke_count=1): {DOWNSAMPLE_BASELINE_RATES}")

        if warmup and len(study.trials) == 0:
            for v in np.unique(np.logspace(np.log10(1000), np.log10(150000), warmup).astype(int)):
                study.enqueue_trial({"stroke_count": int(v)})

        t0 = time.time()
        study.optimize(
            make_objective(images, n_repeats, artifact_root, mp),
            n_trials=trials,
            timeout=per_target_budget,
        )
        print(f"  {len(study.trials)} total trials, {time.time() - t0:.0f}s this session")


# =============================== ANALYZE ===========================================

def _pareto_mask(values):
    """values: (n, 3) in 'maximize' orientation. Returns the non-dominated mask."""
    n = len(values)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        if not mask[i]:
            continue
        # remove points strictly dominated BY i (i is >= in all objectives, > in at least one)
        dominated = np.all(values <= values[i], axis=1) & np.any(values < values[i], axis=1)
        mask[dominated] = False
    return mask


def _trials_df(study):
    rows = []
    for t in study.trials:
        if t.values is None:
            continue
        row = {"number": t.number,
               "speed_s_per_mp": t.values[0], "bpp": t.values[1], "quality": t.values[2]}
        row.update(t.params)
        row["mult_list"] = json.dumps(t.user_attrs.get("mult_list", []))
        rows.append(row)
    return pd.DataFrame(rows)


def analyze(data_dir, storage, targets, out_dir, jpeg):
    import plotly.graph_objects as go
    import plotly.express as px

    os.makedirs(out_dir, exist_ok=True)
    for mp in targets:
        study = optuna.load_study(study_name=f"pbc_{mp}MP", storage=storage)
        df = _trials_df(study)
        if df.empty:
            print(f"{mp} MP: no completed trials, skipping.")
            continue
        out = os.path.join(out_dir, f"{mp}MP")
        os.makedirs(out, exist_ok=True)

        # Pareto orientation: maximize quality, minimize the other two -> negate the minims.
        orient = np.column_stack([-df["speed_s_per_mp"], -df["bpp"], df["quality"]])
        df["pareto"] = _pareto_mask(orient)
        df.to_csv(os.path.join(out, "trials.csv"), index=False)
        df[df["pareto"]].to_csv(os.path.join(out, "pareto.csv"), index=False)

        front = df[df["pareto"]]
        # 3D scatter (all trials grey, Pareto colored by quality).
        fig = go.Figure()
        fig.add_trace(go.Scatter3d(x=df["speed_s_per_mp"], y=df["bpp"], z=df["quality"],
                                   mode="markers", name="all",
                                   marker=dict(size=2, color="lightgrey", opacity=0.4)))
        fig.add_trace(go.Scatter3d(x=front["speed_s_per_mp"], y=front["bpp"], z=front["quality"],
                                   mode="markers", name="Pareto",
                                   marker=dict(size=5, color=front["quality"], colorscale="Viridis"),
                                   text=[f"#{n}" for n in front["number"]]))
        fig.update_layout(title=f"PBC2.4 Pareto front — {mp} MP",
                          scene=dict(xaxis_title="sec / MP", yaxis_title="bpp", zaxis_title="MS-SSIM"))
        fig.write_html(os.path.join(out, "pareto_3d.html"))

        # 2D projections with the front highlighted.
        for xa, ya in (("speed_s_per_mp", "quality"), ("bpp", "quality"), ("speed_s_per_mp", "bpp")):
            f2 = go.Figure()
            f2.add_trace(go.Scatter(x=df[xa], y=df[ya], mode="markers", name="all",
                                    marker=dict(size=4, color="lightgrey")))
            fr = front.sort_values(xa)
            f2.add_trace(go.Scatter(x=fr[xa], y=fr[ya], mode="markers+lines", name="Pareto",
                                    marker=dict(size=7, color="crimson")))
            f2.update_layout(title=f"{ya} vs {xa} — {mp} MP", xaxis_title=xa, yaxis_title=ya)
            f2.write_html(os.path.join(out, f"proj_{xa}__{ya}.html"))

        # Parallel coordinates over numeric params, colored by quality.
        numeric = [c for c in df.columns if df[c].dtype.kind in "if" and c not in ("number",)]
        px.parallel_coordinates(df, dimensions=numeric, color="quality",
                                 color_continuous_scale=px.colors.diverging.Tealrose).write_html(
            os.path.join(out, "parallel_coordinates.html"))

        # Per-objective parameter importances (fANOVA).
        for i, name in enumerate(OBJECTIVES):
            try:
                optuna.visualization.plot_param_importances(
                    study, target=lambda t, i=i: t.values[i], target_name=name
                ).write_html(os.path.join(out, f"importance_{name}.html"))
            except Exception as exc:
                print(f"  importance ({name}) skipped: {exc}")

        if jpeg:
            _jpeg_reference(data_dir, mp, out)

        print(f"{mp} MP: {len(df)} trials, {int(df['pareto'].sum())} on the Pareto front -> {out}")


def _jpeg_reference(data_dir, mp, out, k=5):
    import io
    rows = []
    images = _load_target_images(data_dir, mp, k)
    for q in (1, 5, 10, 20, 40, 60, 80, 95):
        bpps, quals = [], []
        for arr in images:
            buf = io.BytesIO()
            Image.fromarray(arr).save(buf, format="JPEG", quality=q)
            recon = np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"))
            bpps.append(len(buf.getvalue()) * 8 / (arr.shape[0] * arr.shape[1]))
            quals.append(composite_quality(arr, recon))
        rows.append({"quality": q, "bpp": float(np.mean(bpps)), "quality": float(np.mean(quals))})
    pd.DataFrame(rows).to_csv(os.path.join(out, "jpeg_reference.csv"), index=False)


# =============================== CLI ===============================================

def _parse_targets(s):
    return [float(x) for x in s.split(",")] if s else list(DEFAULT_TARGETS)


def main():
    p = argparse.ArgumentParser(description="PBC2.4 hyperparameter tuning")
    p.add_argument("command", choices=["prepare", "run", "analyze", "all"])
    p.add_argument("--source", default=DEFAULT_SOURCE, help="HF dataset id or local image folder")
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--out", default=DEFAULT_OUT_DIR)
    p.add_argument("--storage", default=DEFAULT_DB)
    p.add_argument("--targets", default="", help="comma-separated MP values")
    p.add_argument("--dataset-size", type=int, default=200)
    p.add_argument("--k", type=int, default=5, help="images per trial (tuning subset)")
    p.add_argument("--n-repeats", type=int, default=2, help="timing repeats per image")
    p.add_argument("--trials", type=int, default=1000)
    p.add_argument("--timeout", type=float, default=None, help="total seconds across all targets")
    p.add_argument("--warmup", type=int, default=8, help="stroke_count-spanning seed trials per study")
    p.add_argument("--jpeg", action="store_true", help="also record a JPEG reference frontier")
    p.add_argument("--artifacts", default=None, help="dir to save per-trial .pbc files (empty to disable)")
    p.add_argument("--no-auto-baseline", action="store_true", help="do not seed each study with PBC's automatic default config")
    args = p.parse_args()

    targets = _parse_targets(args.targets)

    if args.command in ("prepare", "all"):
        prepare_data(args.source, args.data_dir, args.dataset_size, targets)
    if args.command in ("run", "all"):
        run(args.data_dir, args.storage, targets, args.k, args.n_repeats,
            args.trials, args.timeout, args.warmup, args.artifacts or None, not args.no_auto_baseline)
    if args.command in ("analyze", "all"):
        analyze(args.data_dir, args.storage, targets, args.out, args.jpeg)


if __name__ == "__main__":
    main()
