import hashlib
import json
import os
import pathlib
import statistics
import sys
import time

os.environ["NUMBA_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np

source = pathlib.Path(sys.argv[1]).resolve()
stream_path = pathlib.Path(sys.argv[2]).resolve()
sys.path.insert(0, str(source))
sys.path.insert(0, str(source / "pbc3"))
from PBC3 import PBC3

data = stream_path.read_bytes()
PBC3.decompress(data)
samples = []
decoded = None
for _ in range(5):
    start = time.perf_counter()
    decoded = PBC3.decompress(data).image
    samples.append(time.perf_counter() - start)
pixels = np.asarray(decoded.convert("RGB"), dtype=np.uint8)
print(json.dumps({
    "source": str(source),
    "stream_sha256": hashlib.sha256(data).hexdigest(),
    "decoded_rgb_sha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
    "samples_seconds": samples,
    "median_seconds": statistics.median(samples),
}))
