import hashlib
import io
import json
import lzma
import os
import pathlib
import statistics
import shutil
import subprocess
import sys
import time

os.environ["NUMBA_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
from PIL import Image, ImageOps, __version__ as PIL_VERSION

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = pathlib.Path(__file__).resolve().parents[1] / "outputs" / "baseline"
PUBLISHED = ROOT / "research_pass45" / "outputs" / "published_release_python"
ASSETS = json.loads((ROOT / "research" / "assets.json").read_text())
SSIMULACRA2 = pathlib.Path(os.environ["SSIMULACRA2"]) if os.environ.get("SSIMULACRA2") else shutil.which("ssimulacra2")
if SSIMULACRA2 is None:
    raise SystemExit("Set SSIMULACRA2 or add ssimulacra2 to PATH")
AVIFENC = ROOT / "research_pass35" / "outputs" / "binaries" / "reference" / "avifenc.exe"
AVIFDEC = ROOT / "research_pass35" / "outputs" / "binaries" / "reference" / "avifdec.exe"
ENV = dict(os.environ, PATH=str(AVIFENC.parent) + ";C:\\Strawberry\\c\\bin;C:\\Windows\\System32;C:\\Windows", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
IMAGES = ("coast.jpg", "googlewiki.png", "mario64.png")
PRESETS = ("compression", "balanced")

sys.path.insert(0, str(PUBLISHED))
sys.path.insert(0, str(PUBLISHED / "pbc3"))
from PBC3 import PBC3
from pbc3_types import PBC3Config


def sha(data):
    return hashlib.sha256(data).hexdigest()


def rgb_mse(a, b):
    delta = a.astype(np.int32) - b.astype(np.int32)
    return float(np.mean(delta * delta, dtype=np.float64))


def ssim_score(source_path, decoded_path):
    result = subprocess.run([str(SSIMULACRA2), str(source_path), str(decoded_path)], capture_output=True, text=True, check=True)
    return float(result.stdout.strip().splitlines()[-1])


def run_cli(args):
    start = time.perf_counter()
    subprocess.run(list(map(str, args)), capture_output=True, text=True, env=ENV, check=True)
    return time.perf_counter() - start


def baseline_row(image_name, codec, preset, source, reconstructed, stream_path, decoded_path, encode_times, decode_times, extra):
    width, height = source.shape[1], source.shape[0]
    return {
        "image": image_name,
        "codec": codec,
        "preset": preset,
        "dimensions": [width, height],
        "completed_bytes": stream_path.stat().st_size,
        "bpp": stream_path.stat().st_size * 8 / (width * height),
        "rgb_mse": rgb_mse(source, reconstructed),
        "rgb_psnr": 10 * np.log10(255**2 / max(rgb_mse(source, reconstructed), 1e-30)),
        "ssimulacra2": ssim_score(OUT / image_name.replace(pathlib.Path(image_name).suffix, ".png"), decoded_path),
        "stream_sha256": sha(stream_path.read_bytes()),
        "decoded_rgb_sha256": sha(reconstructed.tobytes()),
        "encode_samples_seconds": encode_times,
        "decode_samples_seconds": decode_times,
        "encode_median_seconds": statistics.median(encode_times) if encode_times else None,
        "decode_median_seconds": statistics.median(decode_times) if decode_times else None,
        **extra,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    all_hashes = {}
    for name in IMAGES:
        source_path = ROOT / "hpt_data" / name
        expected = ASSETS["inputs"][f"hpt_data/{name}"]["sha256"]
        source_bytes = source_path.read_bytes()
        if sha(source_bytes) != expected:
            raise RuntimeError(f"source file hash mismatch: {name}")
        with Image.open(source_path) as image:
            original = ImageOps.exif_transpose(image).convert("RGB")
        src = np.asarray(original, dtype=np.uint8)
        source_png = OUT / f"{pathlib.Path(name).stem}.png"
        original.save(source_png, format="PNG")
        source_pixels_sha = sha(src.tobytes())
        all_hashes[name] = {"source_file_sha256": expected, "source_rgb_sha256": source_pixels_sha, "dimensions": list(original.size)}

        for preset in PRESETS:
            config = getattr(PBC3Config, preset)()
            if not pathlib.Path(config.learned_filler_model_path).is_file():
                config.learned_filler_model_path = str(PUBLISHED / "patch_policy.npz")
            PBC3.compress(original, config=config, rng_mode="release")
            enc_times = []
            results = []
            for _ in range(3):
                start = time.perf_counter()
                result = PBC3.compress(original, config=config, rng_mode="release")
                enc_times.append(time.perf_counter() - start)
                results.append(result)
            stream = results[-1].data
            if any(result.data != stream for result in results):
                raise RuntimeError(f"nondeterministic canonical stream: {name}/{preset}")
            stream_path = OUT / f"{pathlib.Path(name).stem}_pbc_{preset}.pbc"
            stream_path.write_bytes(stream)
            PBC3.decompress(stream)
            dec_times = []
            decoded = None
            for _ in range(3):
                start = time.perf_counter()
                decoded = PBC3.decompress(stream).image.convert("RGB")
                dec_times.append(time.perf_counter() - start)
            rec = np.asarray(decoded, dtype=np.uint8)
            decoded_path = OUT / f"{pathlib.Path(name).stem}_pbc_{preset}.png"
            decoded.save(decoded_path, format="PNG")
            expected_point = next((r for r in json.loads((ROOT / "research_pass44" / "outputs" / "competitive_baseline.json").read_text())["v31_points"] if r["image"] == name and r["preset"] == preset), None)
            extra = {"source_file_sha256": expected, "source_rgb_sha256": source_pixels_sha, "canonical_stream_repeated": True}
            if expected_point:
                if sha(stream) != expected_point["file_sha256"] or sha(rec.tobytes()) != expected_point["pixel_sha256"]:
                    raise RuntimeError(f"canonical stream differs from frozen row: {name}/{preset}")
                extra["frozen_pass44_exact_match"] = True
            rows.append(baseline_row(name, "canonical_pbc31", preset, src, rec, stream_path, decoded_path, enc_times, dec_times, extra))

        # HF-compatible Pillow/libavif frontend, same quality-0 call as Pass 45.
        pillow_times = []
        pillow_bytes = None
        for _ in range(3):
            encoded = io.BytesIO()
            start = time.perf_counter()
            original.save(encoded, format="AVIF", quality=0)
            pillow_times.append(time.perf_counter() - start)
            pillow_bytes = encoded.getvalue()
        pillow_path = OUT / f"{pathlib.Path(name).stem}_avif_pillow_q0.avif"
        pillow_path.write_bytes(pillow_bytes)
        pillow_dec_times = []
        pillow_rec = None
        for _ in range(3):
            start = time.perf_counter()
            with Image.open(io.BytesIO(pillow_bytes)) as decoded:
                pillow_rec = np.asarray(decoded.convert("RGB"), dtype=np.uint8)
            pillow_dec_times.append(time.perf_counter() - start)
        pillow_decoded_path = OUT / f"{pathlib.Path(name).stem}_avif_pillow_q0.png"
        Image.fromarray(pillow_rec, "RGB").save(pillow_decoded_path, format="PNG")
        rows.append(baseline_row(name, "avif_pillow_11_3_quality0", "quality=0", src, pillow_rec, pillow_path, pillow_decoded_path, pillow_times, pillow_dec_times, {"source_file_sha256": expected, "source_rgb_sha256": source_pixels_sha, "settings": {"pillow": "11.3.0", "quality": 0, "frontend_defaults": "Pillow plugin defaults"}}))

        # Pinned libavif/libaom profile. CLI wall includes launch and file I/O.
        pinned_path = OUT / f"{pathlib.Path(name).stem}_avif_pinned_q0.avif"
        pinned_enc_times = []
        for _ in range(3):
            pinned_enc_times.append(run_cli([AVIFENC, "--codec", "aom", "--qcolor", "0", "--qalpha", "0", "--speed", "8", "--jobs", "1", "--depth", "8", "--yuv", "420", "--range", "full", "--cicp", "1/13/1", "--ignore-exif", "--ignore-xmp", "--ignore-profile", source_path, pinned_path]))
        pinned_dec_times = []
        pinned_rec = None
        pinned_decoded_path = OUT / f"{pathlib.Path(name).stem}_avif_pinned_q0.png"
        for _ in range(3):
            pinned_dec_times.append(run_cli([AVIFDEC, "--jobs", "1", "--ignore-icc", pinned_path, pinned_decoded_path]))
        with Image.open(pinned_decoded_path) as decoded:
            pinned_rec = np.asarray(decoded.convert("RGB"), dtype=np.uint8)
        rows.append(baseline_row(name, "avif_pinned_libaom_q0", "qcolor=0", src, pinned_rec, pinned_path, pinned_decoded_path, pinned_enc_times, pinned_dec_times, {"source_file_sha256": expected, "source_rgb_sha256": source_pixels_sha, "settings": {"libavif": "1.3.0", "libaom": "3.12.1", "speed": 8, "threads": 1, "depth": 8, "subsampling": "420", "range": "full", "cicp": "1/13/1", "timing_boundary": "CLI launch, input/output I/O, and PNG output included; not compared as codec-call timing"}}))

    ssim_hash = sha(SSIMULACRA2.read_bytes())
    result = {
        "pass": 46,
        "source_revision": "3723dc14de88c6c3dcbd98dd80a097e019a92a9f",
        "canonical_python_dir": str(PUBLISHED.relative_to(ROOT)),
        "canonical_model_sha256": ASSETS["model"]["sha256"],
        "ssimulacra2": {"path": str(SSIMULACRA2), "sha256": ssim_hash, "paired_cjxl_version": "v0.11.2", "source_input": "EXIF-transposed RGB PNG", "score_semantics": "higher is better, 100 is identical"},
        "runtime": {"python": sys.version, "numpy": np.__version__, "pillow": PIL_VERSION, "threads": 1},
        "input_hashes": all_hashes,
        "rows": rows,
    }
    (OUT / "baseline_metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"rows": [{"image": r["image"], "codec": r["codec"], "bytes": r["completed_bytes"], "mse": r["rgb_mse"], "ssimulacra2": r["ssimulacra2"], "enc_median": r["encode_median_seconds"], "dec_median": r["decode_median_seconds"]} for r in rows]}, indent=2))


if __name__ == "__main__":
    main()
