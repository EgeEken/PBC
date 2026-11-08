import hashlib
import json
import lzma
import math
import os
import pathlib
import statistics
import shutil
import struct
import subprocess
import sys
import time

os.environ["NUMBA_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = pathlib.Path(__file__).resolve().parents[2]
PASS = pathlib.Path(__file__).resolve().parents[1]
OUT = PASS / "outputs"
BASE = OUT / "baseline"
VIS = OUT / "visuals"
PUBLISHED = ROOT / "research_pass45" / "outputs" / "published_release_python"
ASSETS = json.loads((ROOT / "research" / "assets.json").read_text())
SSIMULACRA2 = pathlib.Path(os.environ["SSIMULACRA2"]) if os.environ.get("SSIMULACRA2") else shutil.which("ssimulacra2")
if SSIMULACRA2 is None:
    raise SystemExit("Set SSIMULACRA2 or add ssimulacra2 to PATH")
IMAGES = ("coast.jpg", "googlewiki.png", "mario64.png")
STROKE = struct.Struct("<4HB3b")
PACKET_HEADER = struct.Struct("<4sI")

sys.path.insert(0, str(PUBLISHED))
sys.path.insert(0, str(PUBLISHED / "pbc3"))
import pbc3_ops as pbc_ops
import pbc3_stream as pbc_stream
from pbc3_types import BitWriter, PBC3Config


def sha(data):
    return hashlib.sha256(data).hexdigest()


def luma(rgb):
    a = rgb.astype(np.int32)
    return ((77 * a[:, :, 0] + 150 * a[:, :, 1] + 29 * a[:, :, 2] + 128) >> 8).astype(np.int16)


def edge_strength(gray):
    a = gray.astype(np.int16)
    gx = np.zeros(a.shape, dtype=np.int16)
    gy = np.zeros(a.shape, dtype=np.int16)
    gx[:, 1:] = a[:, 1:] - a[:, :-1]
    gx[:, 0] = gx[:, 1]
    gy[1:, :] = a[1:, :] - a[:-1, :]
    gy[0, :] = gy[1, :]
    return np.abs(gx.astype(np.int32)) + np.abs(gy.astype(np.int32))


def find_lines(source, base, limit=64):
    sy, by = luma(source), luma(base)
    residual = np.abs(sy.astype(np.int32) - by.astype(np.int32))
    edge = edge_strength(sy)
    eligible = (residual >= 24) & (edge >= 20)
    flat = np.flatnonzero(eligible)
    if not len(flat):
        return [], {"edge_points": 0, "hough_peaks": 0}
    score = residual.ravel()[flat].astype(np.int64) * edge.ravel()[flat].astype(np.int64)
    if len(flat) > 5000:
        select = np.argpartition(score, -5000)[-5000:]
        flat, score = flat[select], score[select]
    order = np.lexsort((flat, -score))
    flat, score = flat[order], score[order]
    height, width = sy.shape
    ys, xs = np.unravel_index(flat, sy.shape)
    xs = xs.astype(np.float64)
    ys = ys.astype(np.float64)
    diag_bins = int(math.ceil(math.hypot(width, height) / 4))
    rho_bins = 2 * diag_bins + 1
    angles = np.arange(0, 180, 6, dtype=np.float64)
    trig = np.deg2rad(angles)
    acc = np.zeros((len(angles), rho_bins), dtype=np.float64)
    for ai, theta in enumerate(trig):
        rho = np.rint((xs * math.cos(theta) + ys * math.sin(theta)) / 4).astype(np.int32) + diag_bins
        acc[ai] = np.bincount(rho, weights=score, minlength=rho_bins)

    peaks = []
    lines = []
    for _ in range(limit * 8):
        flat_peak = int(np.argmax(acc))
        strength = float(acc.ravel()[flat_peak])
        if strength <= 0:
            break
        ai, ri = np.unravel_index(flat_peak, acc.shape)
        rho0 = (int(ri) - diag_bins) * 4.0
        theta = trig[ai]
        dist = np.abs(xs * math.cos(theta) + ys * math.sin(theta) - rho0)
        ids = np.flatnonzero(dist <= 2.5)
        # Suppress neighboring accumulator cells before considering the next maximum.
        for dai in (-1, 0, 1):
            aj = (ai + dai) % len(angles)
            lo, hi = max(0, ri - 2), min(rho_bins, ri + 3)
            acc[aj, lo:hi] = 0
        if len(ids) < 8:
            continue
        along = -xs[ids] * math.sin(theta) + ys[ids] * math.cos(theta)
        lo, hi = np.percentile(along, [5, 95])
        if hi - lo < 24:
            continue
        x0 = int(np.rint(rho0 * math.cos(theta) - lo * math.sin(theta)))
        y0 = int(np.rint(rho0 * math.sin(theta) + lo * math.cos(theta)))
        x1 = int(np.rint(rho0 * math.cos(theta) - hi * math.sin(theta)))
        y1 = int(np.rint(rho0 * math.sin(theta) + hi * math.cos(theta)))
        x0, x1 = np.clip([x0, x1], 0, width - 1).tolist()
        y0, y1 = np.clip([y0, y1], 0, height - 1).tolist()
        length = math.hypot(x1 - x0, y1 - y0)
        if length < 24:
            continue
        line = {"x0": int(x0), "y0": int(y0), "x1": int(x1), "y1": int(y1), "length": float(length), "support": int(len(ids)), "hough_weight": strength}
        if any(math.hypot(line["x0"] - old["x0"], line["y0"] - old["y0"]) < 12 and math.hypot(line["x1"] - old["x1"], line["y1"] - old["y1"]) < 12 for old in lines):
            continue
        lines.append(line)
        if len(lines) >= limit:
            break
    return lines, {"edge_points": int(len(flat)), "hough_peaks": len(lines)}


def stroke_pixels(line, width, shape):
    x0, y0, x1, y1 = (line[k] for k in ("x0", "y0", "x1", "y1"))
    dx, dy = x1 - x0, y1 - y0
    denom = dx * dx + dy * dy
    if denom == 0:
        return np.empty(0, np.int32), np.empty(0, np.int32), np.empty(0, np.uint8)
    height, image_width = shape
    pad = width + 2
    xs, ys = [], []
    if abs(dx) >= abs(dy):
        for x in range(max(0, min(x0, x1) - pad), min(image_width, max(x0, x1) + pad + 1)):
            t = min(1.0, max(0.0, ((x - x0) * dx) / denom))
            yc = y0 + t * dy
            for y in range(max(0, int(math.floor(yc)) - pad), min(height, int(math.ceil(yc)) + pad + 1)):
                xs.append(x)
                ys.append(y)
    else:
        for y in range(max(0, min(y0, y1) - pad), min(height, max(y0, y1) + pad + 1)):
            t = min(1.0, max(0.0, ((y - y0) * dy) / denom))
            xc = x0 + t * dx
            for x in range(max(0, int(math.floor(xc)) - pad), min(image_width, int(math.ceil(xc)) + pad + 1)):
                xs.append(x)
                ys.append(y)
    xx = np.asarray(xs, dtype=np.int32)
    yy = np.asarray(ys, dtype=np.int32)
    t = np.clip(((xx - x0) * dx + (yy - y0) * dy) / denom, 0.0, 1.0)
    cx = x0 + t * dx
    cy = y0 + t * dy
    dist = np.hypot(xx - cx, yy - cy)
    coverage = np.clip(np.rint((width / 2 + 0.5 - dist) * 255), 0, 255).astype(np.uint8)
    keep = coverage > 0
    return yy[keep], xx[keep], coverage[keep]


def fit_stroke(source, base, line, width):
    yy, xx, cov = stroke_pixels(line, width, source.shape[:2])
    if not len(cov):
        return None
    residual = source[yy, xx].astype(np.int32) - base[yy, xx].astype(np.int32)
    c = cov.astype(np.float64) / 255.0
    denom = float(np.sum(c * c))
    amplitudes = np.rint(np.sum(residual * c[:, None], axis=0) / denom).clip(-127, 127).astype(np.int16)
    delta = np.rint(c[:, None] * amplitudes[None, :]).astype(np.int32)
    before = source[yy, xx].astype(np.int32) - base[yy, xx].astype(np.int32)
    after = source[yy, xx].astype(np.int32) - np.clip(base[yy, xx].astype(np.int32) + delta, 0, 255)
    gain = int(np.sum(before * before, dtype=np.int64) - np.sum(after * after, dtype=np.int64))
    return {**line, "width": int(width), "amplitudes": [int(v) for v in amplitudes], "support_pixels": int(len(cov)), "gain": gain, "yy": yy, "xx": xx, "coverage": cov}


def apply_stroke(canvas, stroke):
    out = canvas.copy()
    yy, xx = stroke["yy"], stroke["xx"]
    cov = stroke["coverage"].astype(np.float64) / 255.0
    delta = np.rint(cov[:, None] * np.asarray(stroke["amplitudes"], dtype=np.int16)[None, :]).astype(np.int32)
    out[yy, xx] = np.clip(out[yy, xx].astype(np.int32) + delta, 0, 255).astype(np.uint8)
    return out


def mse_sse(source, decoded):
    delta = source.astype(np.int32) - decoded.astype(np.int32)
    return int(np.sum(delta * delta, dtype=np.int64))


def stroke_record(stroke):
    return STROKE.pack(stroke["x0"], stroke["y0"], stroke["x1"], stroke["y1"], stroke["width"], *stroke["amplitudes"])


def stroke_packet(strokes):
    raw = PACKET_HEADER.pack(b"P46S", len(strokes)) + b"".join(stroke_record(s) for s in strokes)
    return raw, lzma.compress(raw, format=lzma.FORMAT_XZ, preset=6)


def segment_boxes(stroke, width, height):
    x0, y0, x1, y1 = (stroke[k] for k in ("x0", "y0", "x1", "y1"))
    span = max(abs(x1 - x0), abs(y1 - y0))
    steps = max(1, int(math.ceil(span / 380)))
    boxes = []
    for i in range(steps):
        a, b = i / steps, (i + 1) / steps
        ax, ay = int(round(x0 + a * (x1 - x0))), int(round(y0 + a * (y1 - y0)))
        bx, by = int(round(x0 + b * (x1 - x0))), int(round(y0 + b * (y1 - y0)))
        pad = 8 + stroke["width"]
        xa, xb = max(0, min(ax, bx) - pad), min(width, max(ax, bx) + pad + 1)
        ya, yb = max(0, min(ay, by) - pad), min(height, max(ay, by) + pad + 1)
        if xb - xa > 400 or yb - ya > 400:
            raise ValueError("native comparison patch exceeds PBC's 400-pixel patch bound")
        boxes.append((xa, ya, xb - xa, yb - ya))
    return boxes


def native_control(source, base, strokes, cell_size, max_bits):
    height, width = base.shape[:2]
    canvas = base.astype(np.int32).copy()
    config = PBC3Config.compression()
    patches = []
    for stroke in strokes:
        for x, y, w, h in segment_boxes(stroke, width, height):
            for channel in range(3):
                residual = source[y:y+h, x:x+w, channel].astype(np.int16) - canvas[y:y+h, x:x+w, channel].astype(np.int16)
                patch, values = pbc_ops.make_patch(channel, x, y, w, h, cell_size, residual, config, max_bits)
                pbc_ops.apply_grid(canvas[:, :, channel], x, y, w, h, cell_size, values)
                patches.append(patch)
    bw = BitWriter()
    for patch in patches:
        pbc_stream.write_patch(bw, patch, 2)
    raw = PACKET_HEADER.pack(b"P46N", len(patches)) + bw.finish()
    compressed = lzma.compress(raw, format=lzma.FORMAT_XZ, preset=6)
    decoded = np.clip(canvas, 0, 255).astype(np.uint8)
    return {"cell_size": cell_size, "max_bits": max_bits, "patch_count": len(patches), "raw_packet_bytes": len(raw), "compressed_packet_bytes": len(compressed), "gain_sse": mse_sse(source, base) - mse_sse(source, decoded), "decoded": decoded, "packet_sha256": sha(compressed)}


def save_scalar(path, values, high=None):
    a = values.astype(np.float32)
    if high is None:
        high = float(np.percentile(a, 99)) if np.any(a > 0) else 1.0
    pixels = np.clip(a / max(high, 1e-9) * 255, 0, 255).astype(np.uint8)
    Image.fromarray(pixels).save(path)


def make_diagnostic_visuals(name, source, base, corrected, stroke=None):
    VIS.mkdir(parents=True, exist_ok=True)
    stem = pathlib.Path(name).stem
    src_y, base_y = luma(source), luma(base)
    src_edge, rec_edge = edge_strength(src_y), edge_strength(base_y)
    error = np.mean((source.astype(np.int16) - base.astype(np.int16)).astype(np.float32) ** 2, axis=2)
    edge_cut = float(np.percentile(src_edge, 90))
    miss = (src_edge >= edge_cut) & (rec_edge < src_edge * 0.5)
    smooth_cut = float(np.percentile(src_edge, 50))
    smooth_error = np.where(src_edge <= smooth_cut, error, 0)
    save_scalar(VIS / f"{stem}_rgb_error.png", error)
    save_scalar(VIS / f"{stem}_source_edge.png", src_edge)
    save_scalar(VIS / f"{stem}_reconstructed_edge.png", rec_edge)
    save_scalar(VIS / f"{stem}_missed_edges.png", miss.astype(np.float32), 1)
    save_scalar(VIS / f"{stem}_smooth_error.png", smooth_error)
    # A compact, source-aligned diagnostic sheet; each panel is reduced by the same factor.
    panels = []
    for image in (source, base, corrected):
        im = Image.fromarray(image)
        im.thumbnail((1200, 900), Image.Resampling.LANCZOS)
        panels.append(im)
    maps = []
    for array in (error, src_edge, rec_edge, miss.astype(np.float32), smooth_error):
        im = Image.fromarray(np.clip(array / max(float(array.max()), 1e-9) * 255, 0, 255).astype(np.uint8))
        im.thumbnail((1200, 900), Image.Resampling.NEAREST)
        maps.append(im)
    widths = [p.width for p in panels + maps]
    heights = [p.height for p in panels + maps]
    cell_w, cell_h = max(widths), max(heights) + 24
    sheet = Image.new("RGB", (cell_w * 4, cell_h * 2), "white")
    labels = ["Source", "PBC V3.1 base", "Stroke candidate", "Base error", "Source edges", "Base edges", "Missed high edges", "Smooth-region error"]
    all_panels = panels + maps
    draw = ImageDraw.Draw(sheet)
    for i, (im, label) in enumerate(zip(all_panels, labels)):
        col, row = i % 4, i // 4
        sheet.paste(im, (col * cell_w, row * cell_h + 24))
        draw.text((col * cell_w + 4, row * cell_h + 4), label, fill="black")
    sheet.save(VIS / f"{stem}_baseline_diagnostics.png")
    if stroke:
        pad = 32
        x0, y0, x1, y1 = stroke["x0"], stroke["y0"], stroke["x1"], stroke["y1"]
        left, right = max(0, min(x0, x1) - pad), min(source.shape[1], max(x0, x1) + pad + 1)
        top, bottom = max(0, min(y0, y1) - pad), min(source.shape[0], max(y0, y1) + pad + 1)
        crops = [Image.fromarray(a[top:bottom, left:right]) for a in (source, base, corrected)]
        geometry = Image.fromarray(base[top:bottom, left:right])
        draw = ImageDraw.Draw(geometry)
        draw.line((x0-left, y0-top, x1-left, y1-top), fill=(255, 0, 0), width=stroke["width"])
        crops.append(geometry)
        cw = max(c.width for c in crops)
        ch = max(c.height for c in crops) + 24
        composite = Image.new("RGB", (cw * 4, ch), "white")
        d = ImageDraw.Draw(composite)
        for i, (crop, label) in enumerate(zip(crops, ["Source crop", "Base crop", "Stroke corrected", "Encoded line geometry"])):
            composite.paste(crop, (i*cw, 24))
            d.text((i*cw+4, 4), label, fill="black")
        composite.save(VIS / f"{stem}_stroke_crop.png")


def main():
    VIS.mkdir(parents=True, exist_ok=True)
    baseline = json.loads((BASE / "baseline_metrics.json").read_text())
    candidate_rows = []
    profiles = []
    for name in IMAGES:
        input_path = ROOT / "hpt_data" / name
        expected_hash = ASSETS["inputs"][f"hpt_data/{name}"]["sha256"]
        if sha(input_path.read_bytes()) != expected_hash:
            raise RuntimeError(f"source hash mismatch: {name}")
        with Image.open(input_path) as im:
            source = np.asarray(ImageOps.exif_transpose(im).convert("RGB"), dtype=np.uint8)
        base_path = BASE / f"{pathlib.Path(name).stem}_pbc_compression.png"
        with Image.open(base_path) as im:
            base = np.asarray(im.convert("RGB"), dtype=np.uint8)
        base_sse = mse_sse(source, base)
        t0 = time.perf_counter()
        lines, proposal_stats = find_lines(source, base)
        proposal_seconds = time.perf_counter() - t0
        fitted = []
        fit_start = time.perf_counter()
        for line_id, line in enumerate(lines):
            variants = [fit_stroke(source, base, line, width) for width in (1, 2, 4)]
            variants = [v for v in variants if v is not None]
            if not variants:
                continue
            best = max(variants, key=lambda v: (v["gain"] / max(len(stroke_packet([v])[1]), 1), v["gain"], -v["width"]))
            best["line_id"] = line_id
            fitted.append(best)
        fit_seconds = time.perf_counter() - fit_start
        fitted.sort(key=lambda v: (v["gain"] / max(len(stroke_packet([v])[1]), 1), v["gain"]), reverse=True)
        selected = []
        for candidate in fitted:
            # Avoid selecting a duplicate detector response for the same physical line.
            if any(math.hypot(candidate["x0"]-old["x0"], candidate["y0"]-old["y0"]) < 20 and math.hypot(candidate["x1"]-old["x1"], candidate["y1"]-old["y1"]) < 20 for old in selected):
                continue
            selected.append(candidate)
            if len(selected) == 8:
                break
        if not selected:
            make_diagnostic_visuals(name, source, base, base)
            candidate_rows.append({"image": name, "source_sha256": expected_hash, "source_rgb_sha256": sha(source.tobytes()), "base_pixel_sha256": sha(base.tobytes()), "base_sse": base_sse, "base_rgb_mse": base_sse/(source.shape[0]*source.shape[1]*3), "proposal_stats": proposal_stats, "proposal_seconds": proposal_seconds, "fit_seconds": fit_seconds, "native_comparison_seconds": [], "candidate_results": [], "selected_stroke_count": 0, "selected_group_raw_bytes": 8, "selected_group_xz_bytes": None, "selected_group_allowance_bytes_at_5pct": None, "selected_group_within_5pct_packet_allowance": None, "selected_group_gain_sse_recomputed": 0, "selected_group_mse": base_sse/(source.shape[0]*source.shape[1]*3), "selected_group_gain_fraction_of_base_sse": 0, "selected_group_decoded_pixel_sha256": sha(base.tobytes()), "top1_corrected_mse": None, "top1_corrected_pixel_sha256": sha(base.tobytes()), "top1_corrected_ssimulacra2": None})
            profiles.append({"image": name, "source_sha256": expected_hash, "base_sse": base_sse, "proposal": proposal_stats, "proposal_seconds": proposal_seconds, "fit_seconds": fit_seconds, "selected_candidate_count": 0})
            continue

        candidate_native = []
        native_seconds = []
        for idx, stroke in enumerate(selected):
            controls = []
            for cell, bits in ((1, 2), (2, 3)):
                t0 = time.perf_counter()
                control = native_control(source, base, [stroke], cell, bits)
                native_seconds.append(time.perf_counter() - t0)
                controls.append(control)
            raw, compressed = stroke_packet([stroke])
            comparable = [c for c in controls if c["gain_sse"] >= 0.8 * max(stroke["gain"], 0)]
            native_best = max(comparable, key=lambda c: c["gain_sse"] / max(c["compressed_packet_bytes"], 1)) if comparable else None
            candidate_native.append({
                "rank": idx + 1,
                "line_id": stroke["line_id"],
                "geometry": {k: stroke[k] for k in ("x0", "y0", "x1", "y1", "width", "amplitudes", "length", "support", "support_pixels", "hough_weight")},
                "stroke_raw_bytes": len(raw),
                "stroke_xz_bytes": len(compressed),
                "stroke_gain_sse": stroke["gain"],
                "stroke_gain_fraction_of_full_base_sse": stroke["gain"] / max(base_sse, 1),
                "native_controls": [{k: v for k, v in c.items() if k != "decoded"} for c in controls],
                "best_comparable_native": ({k: v for k, v in native_best.items() if k != "decoded"} if native_best else None),
                "gain_per_xz_byte_ratio_vs_best_comparable_native": (stroke["gain"] / max(len(compressed), 1)) / max(native_best["gain_sse"] / max(native_best["compressed_packet_bytes"], 1), 1e-30) if native_best else None,
            })

        chosen = []
        for stroke in selected:
            out = apply_stroke(base if not chosen else corrected, stroke)
            chosen.append(stroke)
            corrected = out
        stroke_raw, stroke_group = stroke_packet(chosen)
        group_gain = base_sse - mse_sse(source, corrected)
        target_bytes = next(r["completed_bytes"] for r in baseline["rows"] if r["image"] == name and r["codec"] == "canonical_pbc31" and r["preset"] == "compression")
        group_allowance = max(1, int(math.floor(0.05 * target_bytes)))
        group_sse_after = mse_sse(source, corrected)
        group_mse = group_sse_after / (source.shape[0] * source.shape[1] * 3)
        best_image = candidate_native[0]
        top_stroke = selected[0]
        corrected_top = apply_stroke(base, top_stroke)
        make_diagnostic_visuals(name, source, base, corrected_top, top_stroke)
        candidate_rows.append({"image": name, "source_sha256": expected_hash, "source_rgb_sha256": sha(source.tobytes()), "base_pixel_sha256": sha(base.tobytes()), "base_sse": base_sse, "base_rgb_mse": base_sse/(source.shape[0]*source.shape[1]*3), "proposal_stats": proposal_stats, "proposal_seconds": proposal_seconds, "fit_seconds": fit_seconds, "native_comparison_seconds": native_seconds, "candidate_results": candidate_native, "selected_stroke_count": len(selected), "selected_group_raw_bytes": len(stroke_raw), "selected_group_xz_bytes": len(stroke_group), "selected_group_allowance_bytes_at_5pct": group_allowance, "selected_group_within_5pct_packet_allowance": len(stroke_group) <= group_allowance, "selected_group_gain_sse_recomputed": group_gain, "selected_group_mse": group_mse, "selected_group_gain_fraction_of_base_sse": group_gain/max(base_sse,1), "selected_group_decoded_pixel_sha256": sha(corrected.tobytes()), "top1_corrected_mse": mse_sse(source, corrected_top)/(source.shape[0]*source.shape[1]*3), "top1_corrected_pixel_sha256": sha(corrected_top.tobytes()), "top1_corrected_ssimulacra2": float(subprocess.run([str(SSIMULACRA2), str(BASE / f"{pathlib.Path(name).stem}.png"), str(VIS / f"{pathlib.Path(name).stem}_top1_tmp.png")], capture_output=True, text=True).stdout.strip() or "nan") if False else None})
        corrected_path = BASE / f"{pathlib.Path(name).stem}_stroke_top1.png"
        Image.fromarray(corrected_top).save(corrected_path)
        metric = subprocess.run([str(SSIMULACRA2), str(BASE / f"{pathlib.Path(name).stem}.png"), str(corrected_path)], capture_output=True, text=True, check=True)
        candidate_rows[-1]["top1_corrected_ssimulacra2"] = float(metric.stdout.strip().splitlines()[-1])
        profiles.append({"image": name, "source_sha256": expected_hash, "base_sse": base_sse, "proposal": proposal_stats, "proposal_seconds": proposal_seconds, "fit_seconds": fit_seconds, "selected_candidate_count": len(selected), "stroke_group_bytes": len(stroke_group), "native_comparison_seconds": native_seconds})

    result = {
        "pass": 46,
        "frozen_design": str((PASS / "EXPERIMENT_DESIGN.md").relative_to(ROOT)),
        "coding_mode": "one straight source-resolution residual stroke with 16-bit endpoints, 8-bit width and signed RGB int8 amplitudes; no mask",
        "fixed_limits": {"max_edge_points": 5000, "hough_angles": 30, "angle_step_degrees": 6, "rho_step_pixels": 4, "max_proposed_lines": 64, "tested_widths": [1, 2, 4], "max_strokes_per_image": 8},
        "packet_format": "8-byte P46S header/count plus 12-byte records, shared XZ preset 6; packet measurements only, not complete-file RD",
        "native_controls": "released PBC make_patch, 3 RGB channels, cell1/max2bits and cell2/max3bits, chunked to its 400-pixel rectangle bound, one shared XZ packet",
        "runtime": {"python": sys.version, "numpy": np.__version__, "threads": 1},
        "images": candidate_rows,
        "profiles": profiles,
        "gate": {
            "useful_correction_rule": "at least 0.5% of full decoded-base RGB SSE; visual inspection also required",
            "comparable_native_rule": "native local gain at least 80% of stroke gain",
            "efficiency_ratio_threshold": 1.2,
            "required_structural_wins": 3,
            "required_content_classes": 2,
            "complete_codec": "not attempted in this script",
        },
    }
    (OUT / "feasibility_results.json").write_text(json.dumps(result, indent=2) + "\n")
    compact = [{"image": r["image"], "lines": r["proposal_stats"]["hough_peaks"], "selected": len(r["candidate_results"]), "group_bytes": r["selected_group_xz_bytes"], "packet_allowance": r["selected_group_allowance_bytes_at_5pct"], "group_mse_gain": r["selected_group_gain_fraction_of_base_sse"], "best_efficiency_ratio": max([v["gain_per_xz_byte_ratio_vs_best_comparable_native"] or 0 for v in r["candidate_results"]], default=0), "top1_ssimulacra2": r["top1_corrected_ssimulacra2"]} for r in candidate_rows]
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
