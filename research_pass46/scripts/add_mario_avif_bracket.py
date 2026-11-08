import json
import pathlib
import statistics

import numpy as np
from PIL import Image

from run_baseline import AVIFDEC, AVIFENC, OUT as BASE, ROOT, run_cli, sha, ssim_score


def main():
    metrics_path = BASE / "baseline_metrics.json"
    metrics = json.loads(metrics_path.read_text())
    source_path = ROOT / "hpt_data" / "mario64.png"
    source = np.asarray(Image.open(BASE / "mario64.png").convert("RGB"), dtype=np.uint8)
    source_sha = metrics["input_hashes"]["mario64.png"]["source_file_sha256"]
    pixel_sha = metrics["input_hashes"]["mario64.png"]["source_rgb_sha256"]
    rows = []
    for q in (5, 12, 20, 35, 50):
        stream = BASE / f"mario64_avif_pinned_q{q}.avif"
        decoded_path = BASE / f"mario64_avif_pinned_q{q}.png"
        encode_times = []
        decode_times = []
        for _ in range(3):
            encode_times.append(run_cli([AVIFENC, "--codec", "aom", "--qcolor", str(q), "--qalpha", "0", "--speed", "8", "--jobs", "1", "--depth", "8", "--yuv", "420", "--range", "full", "--cicp", "1/13/1", "--ignore-exif", "--ignore-xmp", "--ignore-profile", source_path, stream]))
            decode_times.append(run_cli([AVIFDEC, "--jobs", "1", "--ignore-icc", stream, decoded_path]))
        decoded = np.asarray(Image.open(decoded_path).convert("RGB"), dtype=np.uint8)
        delta = source.astype(np.int32) - decoded.astype(np.int32)
        mse = float(np.mean(delta * delta, dtype=np.float64))
        rows.append({
            "image": "mario64.png",
            "codec": "avif_pinned_libaom_qcolor",
            "preset": f"qcolor={q}",
            "dimensions": [source.shape[1], source.shape[0]],
            "completed_bytes": stream.stat().st_size,
            "bpp": stream.stat().st_size * 8 / (source.shape[0] * source.shape[1]),
            "rgb_mse": mse,
            "rgb_psnr": 10 * np.log10(255**2 / max(mse, 1e-30)),
            "ssimulacra2": ssim_score(BASE / "mario64.png", decoded_path),
            "stream_sha256": sha(stream.read_bytes()),
            "decoded_rgb_sha256": sha(decoded.tobytes()),
            "encode_samples_seconds": encode_times,
            "decode_samples_seconds": decode_times,
            "encode_median_seconds": statistics.median(encode_times),
            "decode_median_seconds": statistics.median(decode_times),
            "source_file_sha256": source_sha,
            "source_rgb_sha256": pixel_sha,
            "settings": {"libavif": "1.3.0", "libaom": "3.12.1", "qcolor": q, "speed": 8, "threads": 1, "depth": 8, "subsampling": "420", "range": "full", "cicp": "1/13/1", "timing_boundary": "CLI launch and input/output I/O included"},
        })
    metrics["rows"].extend(rows)
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps([{k: r[k] for k in ("preset", "completed_bytes", "rgb_mse", "ssimulacra2", "encode_median_seconds", "decode_median_seconds")} for r in rows], indent=2))


if __name__ == "__main__":
    main()
