import hashlib
import json
import os
import pathlib
import random
import statistics
import subprocess
import time

import numpy as np
from PIL import Image, ImageOps

ROOT = pathlib.Path(__file__).resolve().parents[2]
PASS35 = ROOT / "research_pass35"
OUT = ROOT / "research_pass39" / "outputs" / "benchmark"
BIN = PASS35 / "outputs" / "binaries" / "reference"
SSIMULACRA = pathlib.Path(os.environ["LOCALAPPDATA"]) / "Microsoft" / "WinGet" / "Packages" / "libjxl.libjxl_Microsoft.Winget.Source_8wekyb3d8bbwe" / "bin" / "ssimulacra2.exe"
MANIFEST = json.loads((PASS35 / "SOURCE_MANIFEST.json").read_text())
IMAGES = {
    "googlewiki.png": [5, 20],
    "mario64.png": [50, 65],
    "flower.jpg": [80, 90],
    "cat.jpg": [35, 50],
    "coast.jpg": [80, 90],
}
CONFIGS = {
    "REF": [],
    "A": ["--advanced", "aq-mode=0", "--advanced", "tune=psnr"],
    "B": ["--advanced", "aq-mode=1", "--advanced", "tune=psnr"],
    "C": ["--advanced", "aq-mode=0", "--advanced", "tune=iq"],
    "D": ["--advanced", "aq-mode=1", "--advanced", "tune=iq"],
}
ENV = dict(os.environ, PATH=str(BIN) + ";C:\\Strawberry\\c\\bin;C:\\Windows\\System32;C:\\Windows")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(cmd, sample_memory=False):
    start = time.perf_counter()
    if not sample_memory:
        proc = subprocess.run(cmd, capture_output=True, text=True, env=ENV)
        peak_rss = None
    else:
        import psutil
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=ENV)
        process = psutil.Process(proc.pid)
        peak_rss = 0
        while proc.poll() is None:
            try:
                peak_rss = max(peak_rss, process.memory_info().rss)
            except psutil.Error:
                pass
            time.sleep(0.01)
        stdout, stderr = proc.communicate()
        proc.stdout, proc.stderr = stdout, stderr
        proc.returncode = proc.returncode
    elapsed = time.perf_counter() - start
    if proc.returncode:
        raise RuntimeError(f"{cmd[0]} failed ({proc.returncode}):\n{proc.stderr}\n{proc.stdout}")
    return elapsed, proc.stdout, proc.stderr, peak_rss


def encode(src, dst, q, config, speed=6):
    cmd = [str(BIN / "avifenc.exe"), "--codec", "aom", "--qcolor", str(q), "--qalpha", "0",
           "--speed", str(speed), "--jobs", "1", "--depth", "8", "--yuv", "420", "--range", "full",
           "--cicp", "1/13/1", "--ignore-exif", "--ignore-xmp", "--ignore-profile"]
    cmd += CONFIGS[config] + [str(src), str(dst)]
    return run(cmd)


def decode(avif, png):
    return run([str(BIN / "avifdec.exe"), "--jobs", "1", "--ignore-icc", str(avif), str(png)])[0]


def images(src, recon):
    with Image.open(src) as im:
        original = np.asarray(ImageOps.exif_transpose(im).convert("RGB"), dtype=np.float64)
    with Image.open(recon) as im:
        decoded = np.asarray(im.convert("RGB"), dtype=np.float64)
    if original.shape != decoded.shape:
        raise RuntimeError(f"shape mismatch {original.shape} vs {decoded.shape}")
    mse = float(np.mean((original - decoded) ** 2))
    return original.astype(np.uint8), decoded.astype(np.uint8), mse


def one(src, name, q, config, speed, rep, phase):
    key = f"s{speed}_{pathlib.Path(name).stem}_q{q}_{config}_{phase}{rep}"
    avif = OUT / f"{key}.avif"
    png = OUT / f"{key}.png"
    enc_cmd = [str(BIN / "avifenc.exe"), "--codec", "aom", "--qcolor", str(q), "--qalpha", "0",
               "--speed", str(speed), "--jobs", "1", "--depth", "8", "--yuv", "420", "--range", "full",
               "--cicp", "1/13/1", "--ignore-exif", "--ignore-xmp", "--ignore-profile"] + CONFIGS[config] + [str(src), str(avif)]
    enc_time, enc_log, enc_err, peak_rss = run(enc_cmd, sample_memory=(phase == "memory"))
    if enc_err:
        raise RuntimeError(f"avifenc warnings/errors: {enc_err}")
    dec_time = decode(avif, png)
    original, decoded, mse = images(src, png)
    original_png = OUT / f"{pathlib.Path(name).stem}_source_rgb.png"
    if not original_png.exists():
        with Image.open(src) as im:
            ImageOps.exif_transpose(im).convert("RGB").save(original_png)
    score_out = run([str(SSIMULACRA), str(original_png), str(png)])[1].strip()
    try:
        score = float(score_out.splitlines()[-1])
    except (ValueError, IndexError) as exc:
        raise RuntimeError(f"could not parse SSIMULACRA 2 output: {score_out!r}") from exc
    row = {
        "image": name, "qcolor": q, "config": config, "speed": speed,
        "phase": phase, "repeat": rep, "bytes": avif.stat().st_size,
        "bpp": avif.stat().st_size * 8 / (original.shape[0] * original.shape[1]),
        "rgb_mse": mse,
        "rgb_psnr": float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse),
        "ssimulacra2": score,
        "encode_seconds": enc_time, "decode_seconds": dec_time,
        "avif_sha256": digest(avif), "decoded_rgb_sha256": hashlib.sha256(decoded.tobytes()).hexdigest(),
        "decoder_ok": True,
    }
    if peak_rss is not None:
        row["peak_working_set_bytes"] = peak_rss
    return row, original, decoded


def verify_assets():
    expected = {item["name"]: item["sha256"] for item in MANIFEST["assets"]}
    for name in IMAGES:
        path = ROOT / "hpt_data" / name
        if digest(path) != expected[name]:
            raise RuntimeError(f"source hash mismatch: {name}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    verify_assets()
    if not SSIMULACRA.exists():
        raise RuntimeError(f"SSIMULACRA 2 binary unavailable: {SSIMULACRA}")
    if float(np.mean((np.array([0.0]) - np.array([255.0])) ** 2)) != 65025.0:
        raise RuntimeError("float64 MSE fixture failed")
    rows = []
    for name, qvalues in IMAGES.items():
        src = ROOT / "hpt_data" / name
        for point, q in enumerate(qvalues, 1):
            configs = list(CONFIGS)
            random.Random(f"{name}-{q}").shuffle(configs)
            originals = {}
            for config in configs:
                row, original, _ = one(src, name, q, config, 6, 0, "warm")
                rows.append(row)
                originals[config] = original
            if name == "coast.jpg" and q == 90:
                for config in configs:
                    row, _, _ = one(src, name, q, config, 6, 0, "memory")
                    rows.append(row)
            for rep in range(1, 4):
                order = configs[rep % len(configs):] + configs[:rep % len(configs)]
                for config in order:
                    row, _, _ = one(src, name, q, config, 6, rep, "timed")
                    row["point"] = point
                    row["source_sha256"] = digest(src)
                    rows.append(row)
                    print(f"s6 {name:14} q{q:2} {config:3} r{rep} {row['bytes']:8}B {row['bpp']:.4f}bpp MSE {row['rgb_mse']:.4f} enc {row['encode_seconds']:.3f}s", flush=True)
    for name, q, speed in (("googlewiki.png", 20, 8), ("cat.jpg", 50, 8)):
        src = ROOT / "hpt_data" / name
        configs = ["REF", "A", "B", "C"]
        for config in configs:
            row, _, _ = one(src, name, q, config, speed, 0, "warm")
            rows.append(row)
        for rep in range(1, 4):
            order = configs[rep % len(configs):] + configs[:rep % len(configs)]
            for config in order:
                row, _, _ = one(src, name, q, config, speed, rep, "timed")
                row["source_sha256"] = digest(src)
                rows.append(row)
                print(f"s8 {name:14} q{q:2} {config:3} r{rep} {row['bytes']:8}B {row['bpp']:.4f}bpp MSE {row['rgb_mse']:.4f} enc {row['encode_seconds']:.3f}s", flush=True)
    result = {
        "build": {"libaom": "3.12.1", "libavif": "1.3.0", "libaom_commit": "10aece4157eb79315da205f39e19bf6ab3ee30d0", "libavif_commit": "1aadfad932c98c069a1204261b1856f81f3bc199"},
        "settings": {"depth": 8, "yuv": "420", "range": "full", "jobs": 1, "cicp": "1/13/1", "metadata": "ignored", "timed_repetitions": 3, "warmups": 1, "mse": "RGB arrays promoted to float64 before subtraction and squaring"},
        "configurations": CONFIGS,
        "rows": rows,
    }
    (OUT / "results.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
