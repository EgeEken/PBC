import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np
import psutil
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[2]
DEV = ROOT / "hpt_data"
CASES = [
    ("googlewiki.png", 5, 6), ("googlewiki.png", 20, 6),
    ("mario64.png", 50, 6), ("flower.jpg", 80, 6),
    ("cat.jpg", 35, 6), ("cat.jpg", 50, 6), ("coast.jpg", 80, 6),
    ("googlewiki.png", 5, 8), ("cat.jpg", 35, 8),
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def encode(exe, image, q, speed, out, logbase, measure_memory=False):
    args = [str(exe), "--codec", "aom", "--qcolor", str(q), "--qalpha", "0",
            "--speed", str(speed), "--jobs", "1", "--depth", "8", "--yuv", "420",
            "--range", "full", "--cicp", "1/13/1", "--ignore-exif",
            "--ignore-xmp", "--ignore-profile", str(image), str(out)]
    env = os.environ.copy()
    env["PATH"] = str(exe.parent) + os.pathsep + r"C:\Strawberry\c\bin;C:\Windows\System32;C:\Windows" + os.pathsep + env.get("PATH", "")
    with open(str(logbase) + ".stdout", "wb") as stdout, open(str(logbase) + ".stderr", "wb") as stderr:
        start = time.perf_counter()
        proc = subprocess.Popen(args, stdout=stdout, stderr=stderr, env=env)
        peak = 0
        if measure_memory:
            p = psutil.Process(proc.pid)
            while proc.poll() is None:
                try:
                    peak = max(peak, p.memory_info().rss)
                except psutil.Error:
                    pass
                time.sleep(0.002)
            code = proc.returncode
            try:
                peak = max(peak, p.memory_info().rss)
            except psutil.Error:
                pass
        else:
            code = proc.wait()
        elapsed = time.perf_counter() - start
    if code != 0 or not out.is_file():
        raise RuntimeError(f"encode failed: {exe.name}, {image.name}, q{q}, s{speed}, code={code}")
    return {"seconds": elapsed, "peak_rss_bytes": peak if measure_memory else None, "exit_code": code,
            "bytes": out.stat().st_size, "avif_sha256": sha256(out), "path": str(out)}


def decode(decoder, avif, out, logbase):
    env = os.environ.copy()
    env["PATH"] = str(decoder.parent) + os.pathsep + r"C:\Strawberry\c\bin;C:\Windows\System32;C:\Windows" + os.pathsep + env.get("PATH", "")
    with open(str(logbase) + ".decode.stdout", "wb") as stdout, open(str(logbase) + ".decode.stderr", "wb") as stderr:
        p = subprocess.run([str(decoder), "--jobs", "1", "--ignore-icc", str(avif), str(out)], stdout=stdout, stderr=stderr, env=env)
    if p.returncode != 0 or not out.is_file():
        raise RuntimeError(f"decode failed: {avif}")
    return hashlib.sha256(rgb(out).tobytes()).hexdigest()


def rgb(path):
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        return np.asarray(im, dtype=np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--ctrl", type=Path, required=True)
    ap.add_argument("--exp", type=Path, required=True)
    ap.add_argument("--decoder", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--repetitions", type=int, default=5)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    binaries = {"REF": args.ref.resolve(), "CTRL": args.ctrl.resolve(), "EXP": args.exp.resolve()}
    run_rows = []
    sources = {}
    all_avifs = {}
    for image_name, q, speed in CASES:
        case_id = f"{Path(image_name).stem}_q{q}_s{speed}"
        src = DEV / image_name
        source = rgb(src)
        sources[case_id] = {"path": str(src), "sha256": sha256(src), "width": int(source.shape[1]), "height": int(source.shape[0]), "rgb": source}
        case_dir = args.out / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        # One uninstrumented REF/CTRL encode establishes the research-off control.
        control = {}
        for label in ("REF", "CTRL"):
            path = case_dir / f"{label.lower()}_control.avif"
            control[label] = encode(binaries[label], src, q, speed, path, case_dir / f"{label.lower()}_control")
            all_avifs[control[label]["avif_sha256"]] = path
        if control["REF"]["avif_sha256"] != control["CTRL"]["avif_sha256"]:
            raise RuntimeError(f"CTRL differs from REF: {case_id}")
        warm = {}
        for label in ("REF", "EXP"):
            path = case_dir / f"{label.lower()}_warm.avif"
            warm[label] = encode(binaries[label], src, q, speed, path, case_dir / f"{label.lower()}_warm")
            all_avifs[warm[label]["avif_sha256"]] = path
        timings = {"REF": [], "EXP": []}
        for rep in range(args.repetitions):
            order = ("REF", "EXP") if rep % 2 == 0 else ("EXP", "REF")
            for label in order:
                path = case_dir / f"{label.lower()}_r{rep + 1}.avif"
                row = encode(binaries[label], src, q, speed, path, case_dir / f"{label.lower()}_r{rep + 1}")
                row.update({"case": case_id, "config": label, "repetition": rep + 1})
                timings[label].append(row)
                run_rows.append(row)
                all_avifs[row["avif_sha256"]] = path
        # EXP must remain deterministic across all five timed outputs.
        exp_hashes = {r["avif_sha256"] for r in timings["EXP"]}
        if len(exp_hashes) != 1:
            raise RuntimeError(f"EXP output is nondeterministic: {case_id}")

    decoded = {}
    for digest, avif in all_avifs.items():
        png = args.out / "decoded" / f"{digest}.png"
        png.parent.mkdir(parents=True, exist_ok=True)
        decoded[digest] = decode(args.decoder, avif, png, png.with_suffix(""))
    memory_probes = {}
    for image_name, q, speed in CASES:
        case_id = f"{Path(image_name).stem}_q{q}_s{speed}"
        src = DEV / image_name
        source = sources[case_id]["rgb"]
        memory_probes[case_id] = {}
        for label in ("REF", "EXP"):
            path = args.out / "memory_probes" / case_id / f"{label.lower()}.avif"
            path.parent.mkdir(parents=True, exist_ok=True)
            probe = encode(binaries[label], src, q, speed, path,
                           path.with_suffix(""), measure_memory=True)
            expected = next(r for r in run_rows if r["case"] == case_id and r["config"] == label)
            if probe["avif_sha256"] != expected["avif_sha256"]:
                raise RuntimeError(f"memory probe output differs: {case_id} {label}")
            memory_probes[case_id][label] = {"peak_sampled_rss_bytes": probe["peak_rss_bytes"],
                                              "avif_sha256": probe["avif_sha256"]}
    rows = []
    for image_name, q, speed in CASES:
        case_id = f"{Path(image_name).stem}_q{q}_s{speed}"
        source = sources[case_id]["rgb"]
        case_dir = args.out / case_id
        metrics = {}
        for label in ("REF", "CTRL", "EXP"):
            sample = next((r for r in run_rows if r["case"] == case_id and r["config"] == label), None)
            path = Path(sample["path"]) if sample else case_dir / f"{label.lower()}_control.avif"
            digest = sha256(path)
            decoded_png = args.out / "decoded" / f"{digest}.png"
            recon = rgb(decoded_png)
            if recon.shape != source.shape:
                raise RuntimeError(f"dimension mismatch {case_id} {label}: {recon.shape} != {source.shape}")
            mse = float(np.mean((source.astype(np.float64) - recon.astype(np.float64)) ** 2))
            samples = [r for r in run_rows if r["case"] == case_id and r["config"] == label]
            if label == "CTRL":
                samples = []
            if label in ("REF", "EXP"):
                values = [r["seconds"] for r in samples]
                peak = memory_probes[case_id][label]["peak_sampled_rss_bytes"]
            else:
                values = []
                peak = 0
            metrics[label] = {"bytes": path.stat().st_size, "bpp": path.stat().st_size * 8 / (source.shape[0] * source.shape[1]),
                              "avif_sha256": digest, "decoded_rgb_sha256": decoded[digest], "rgb_mse": mse,
                              "timed_seconds": values, "median_seconds": statistics.median(values) if values else None,
                              "mean_seconds": statistics.mean(values) if values else None,
                              "stdev_seconds": statistics.stdev(values) if len(values) > 1 else None,
                              "peak_rss_bytes": peak}
        ref, exp = metrics["REF"], metrics["EXP"]
        rows.append({"image": image_name, "qcolor": q, "speed": speed,
                     "source_sha256": sources[case_id]["sha256"], "width": source.shape[1], "height": source.shape[0],
                     "ref_ctrl_byte_identical": metrics["REF"]["avif_sha256"] == metrics["CTRL"]["avif_sha256"],
                     "exp_deterministic": len({r["avif_sha256"] for r in run_rows if r["case"] == case_id and r["config"] == "EXP"}) == 1,
                     "time_reduction_percent": (ref["median_seconds"] - exp["median_seconds"]) / ref["median_seconds"] * 100,
                     "speedup": ref["median_seconds"] / exp["median_seconds"],
                     "byte_change_percent": (exp["bytes"] - ref["bytes"]) / ref["bytes"] * 100,
                     "mse_change_percent": (exp["rgb_mse"] - ref["rgb_mse"]) / ref["rgb_mse"] * 100 if ref["rgb_mse"] else None,
                     "configs": metrics})
    summary = {"source": {"libaom": "10aece4157eb79315da205f39e19bf6ab3ee30d0", "libavif": "1aadfad932c98c069a1204261b1856f81f3bc199"},
               "timing_boundary": "full avifenc process wall time; includes CLI startup, frontend, encoding, and file write",
               "repetitions": args.repetitions, "memory_method": "separate encode probes with 2 ms psutil RSS polling; excluded from timed runs", "memory_probes": memory_probes, "runs": run_rows, "rows": rows}
    (args.out / "benchmark.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps([{k: r[k] for k in ("image", "qcolor", "speed", "speedup", "time_reduction_percent", "byte_change_percent", "mse_change_percent")} for r in rows], indent=2))

if __name__ == "__main__":
    main()

