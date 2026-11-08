import json
import pathlib
import statistics
import sys
import time

PASS = pathlib.Path(__file__).resolve().parents[1]
ROOT = PASS.parent
sys.path.insert(0, str(PASS / "codec"))
from PBC3 import PBC3
from run_oracle import BASELINE, OUT, sha


def main():
    path = OUT / "oracle_results.json"
    result = json.loads(path.read_text())
    for row in result["cases"]:
        stream_path = BASELINE / f"{pathlib.Path(row['image']).stem}_{row['preset']}.pbc"
        stream = stream_path.read_bytes()
        PBC3.decompress(stream)
        samples = []
        decoded = None
        for _ in range(3):
            start = time.perf_counter()
            decoded = PBC3.decompress(stream).image.convert("RGB")
            samples.append(time.perf_counter() - start)
        if sha(decoded.tobytes()) != row["decoded_rgb_sha256"]:
            raise RuntimeError(f"control decode mismatch: {row['image']}/{row['preset']}")
        row["decode_samples_seconds"] = samples
        row["decode_median_seconds"] = statistics.median(samples)
        # The stream header is authoritative for serialized patch count.
        _, body = PBC3._open_body(stream)
        from pbc3_types import BitReader
        header = PBC3._read_header(BitReader(body))
        row["serialized_patch_count"] = int(header[10])
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps([{"image": r["image"], "preset": r["preset"], "patches": r["serialized_patch_count"], "encode_median": r["encode_median_seconds"], "decode_median": r["decode_median_seconds"]} for r in result["cases"]], indent=2))


if __name__ == "__main__":
    main()
