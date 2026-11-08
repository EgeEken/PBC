import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "hpt_data"
CASES = [
    ("googlewiki.png", 5, 6),
    ("mario64.png", 50, 6),
    ("cat.jpg", 35, 6),
    ("googlewiki.png", 5, 8),
    ("cat.jpg", 35, 8),
]

def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

def run(exe, image, q, speed, out, stem):
    args = [str(exe), "--codec", "aom", "--qcolor", str(q), "--qalpha", "0",
            "--speed", str(speed), "--jobs", "1", "--depth", "8", "--yuv", "420",
            "--range", "full", "--cicp", "1/13/1", "--ignore-exif",
            "--ignore-xmp", "--ignore-profile", str(image), str(out)]
    env = os.environ.copy()
    env["PATH"] = str(exe.parent) + os.pathsep + r"C:\Strawberry\c\bin;C:\Windows\System32;C:\Windows"
    with open(str(stem) + ".stdout", "wb") as stdout, open(str(stem) + ".stderr", "wb") as stderr:
        proc = subprocess.run(args, stdout=stdout, stderr=stderr, env=env)
    if proc.returncode or not out.is_file():
        raise RuntimeError(f"encode failed: {exe}, {image}, q{q}, s{speed}")
    return {"bytes": out.stat().st_size, "sha256": sha(out)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--diagnostic", type=Path, required=True)
    ap.add_argument("--decoder", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, q, speed in CASES:
        image = DATA / name
        point = f"{Path(name).stem}_q{q}_s{speed}"
        case = args.out / point
        case.mkdir(exist_ok=True)
        ref = case / "ref.avif"
        diag = case / "diag.avif"
        ref_row = run(args.reference, image, q, speed, ref, case / "ref")
        diag_row = run(args.diagnostic, image, q, speed, diag, case / "diag")
        if ref.read_bytes() != diag.read_bytes():
            raise RuntimeError(f"REF/DIAG byte mismatch: {point}")
        decoded = case / "decoded.png"
        env = os.environ.copy()
        env["PATH"] = str(args.decoder.parent) + os.pathsep + r"C:\Strawberry\c\bin;C:\Windows\System32;C:\Windows"
        with open(case / "decode.stdout", "wb") as stdout, open(case / "decode.stderr", "wb") as stderr:
            p = subprocess.run([str(args.decoder), "--jobs", "1", "--ignore-icc", str(diag), str(decoded)],
                               stdout=stdout, stderr=stderr, env=env)
        if p.returncode or not decoded.is_file():
            raise RuntimeError(f"decode failed: {point}")
        with Image.open(image) as im:
            src = np.asarray(ImageOps.exif_transpose(im).convert("RGB"), dtype=np.float64)
        with Image.open(decoded) as im:
            rec = np.asarray(ImageOps.exif_transpose(im).convert("RGB"), dtype=np.float64)
        if src.shape != rec.shape:
            raise RuntimeError(f"decoded dimensions mismatch: {point}")
        mse = float(np.mean((src - rec) ** 2))
        diag_log = (case / "diag.stderr").read_text(errors="replace")
        rows.append({
            "image": name, "qcolor": q, "speed": speed,
            "input_sha256": sha(image), "bytes": ref_row["bytes"],
            "avif_sha256": ref_row["sha256"], "identical_ref_diag": True,
            "decoded": True, "rgb_mse": mse,
            "summary": next((x for x in diag_log.splitlines() if x.startswith("PBC38_SUMMARY,")), None),
            "event_count": sum(x.startswith("PBC38_EVENT,") for x in diag_log.splitlines()),
            "event_log": str(case / "diag.stderr"),
        })
    result = {
        "source": {"libaom": "10aece4157eb79315da205f39e19bf6ab3ee30d0",
                   "libavif": "1aadfad932c98c069a1204261b1856f81f3bc199"},
        "sampling_modulus": 16, "sample_cap_per_process": 128,
        "cases": rows,
    }
    (args.out / "diagnostic.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))

if __name__ == "__main__":
    main()

