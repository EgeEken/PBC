import hashlib
import json
import pathlib
import statistics

import numpy as np
from PIL import Image

from run_baseline import AVIFDEC, AVIFENC, OUT as BASE, ROOT, run_cli, sha, ssim_score


def main():
    metrics_path = BASE / "baseline_metrics.json"
    metrics = json.loads(metrics_path.read_text())
    reference = json.loads((ROOT / "research_pass44" / "outputs" / "competitive_baseline.json").read_text())
    selected = {("coast.jpg", 7), ("googlewiki.png", 12), ("googlewiki.png", 28)}
    rows = []
    for source_name, q in sorted(selected):
        source_path = ROOT / "hpt_data" / source_name
        stem = pathlib.Path(source_name).stem
        source = np.asarray(Image.open(BASE / f"{stem}.png").convert("RGB"), dtype=np.uint8)
        expected_source_sha = metrics["input_hashes"][source_name]["source_file_sha256"]
        output = BASE / f"{stem}_avif_pinned_q{q}.avif"
        decoded_path = BASE / f"{stem}_avif_pinned_q{q}.png"
        enc_times, dec_times = [], []
        for _ in range(3):
            enc_times.append(run_cli([AVIFENC, "--codec", "aom", "--qcolor", str(q), "--qalpha", "0", "--speed", "8", "--jobs", "1", "--depth", "8", "--yuv", "420", "--range", "full", "--cicp", "1/13/1", "--ignore-exif", "--ignore-xmp", "--ignore-profile", source_path, output]))
            dec_times.append(run_cli([AVIFDEC, "--jobs", "1", "--ignore-icc", output, decoded_path]))
        decoded = np.asarray(Image.open(decoded_path).convert("RGB"), dtype=np.uint8)
        delta = source.astype(np.int32) - decoded.astype(np.int32)
        mse = float(np.mean(delta * delta, dtype=np.float64))
        archived = next(r for r in reference["avif_points"] if r["image"] == source_name and r["qcolor"] == q)
        digest = sha(output.read_bytes())
        archived_digest = archived.get("avif_sha256", archived.get("sha256"))
        if output.stat().st_size != archived["bytes"] or digest != archived_digest:
            raise RuntimeError(f"pinned AVIF differs from Pass 44 archived output: {source_name} q{q}")
        rows.append({
            "image": source_name,
            "codec": "avif_pinned_libaom_qcolor",
            "preset": f"qcolor={q}",
            "dimensions": [source.shape[1], source.shape[0]],
            "completed_bytes": output.stat().st_size,
            "bpp": output.stat().st_size * 8 / (source.shape[0] * source.shape[1]),
            "rgb_mse": mse,
            "rgb_psnr": 10 * np.log10(255**2 / max(mse, 1e-30)),
            "ssimulacra2": ssim_score(BASE / f"{stem}.png", decoded_path),
            "stream_sha256": digest,
            "decoded_rgb_sha256": sha(decoded.tobytes()),
            "encode_samples_seconds": enc_times,
            "decode_samples_seconds": dec_times,
            "encode_median_seconds": statistics.median(enc_times),
            "decode_median_seconds": statistics.median(dec_times),
            "source_file_sha256": expected_source_sha,
            "source_rgb_sha256": metrics["input_hashes"][source_name]["source_rgb_sha256"],
            "settings": {"libavif": "1.3.0", "libaom": "3.12.1", "qcolor": q, "speed": 8, "threads": 1, "depth": 8, "subsampling": "420", "range": "full", "cicp": "1/13/1", "timing_boundary": "CLI launch and input/output I/O included"},
        })
    metrics["rows"].extend(rows)
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps([{k: r[k] for k in ("image", "preset", "completed_bytes", "rgb_mse", "ssimulacra2")} for r in rows], indent=2))


if __name__ == "__main__":
    main()
