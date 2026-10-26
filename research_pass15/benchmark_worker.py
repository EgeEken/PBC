import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.update({key: "1" for key in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
)})

import numba
import numpy as np
import psutil
from PIL import Image, ImageOps

numba.set_num_threads(1)
psutil.Process().cpu_affinity([int(os.environ["PBC_CPU"])])
SOURCE = Path(os.environ["PBC_SOURCE"]).resolve()
sys.path.insert(0, str(SOURCE))
from PBC3 import PBC3
from pbc3_types import BitReader, PBC3Config
if os.environ.get("PBC_INTEGRAL_CANDIDATE") == "1":
    import pbc3_ops
    from integral_candidate import integral
    pbc3_ops.integral = integral

IMAGE_PATHS = [Path(path) for path in json.loads(os.environ["PBC_IMAGES"])]
PRESETS = ("compression", "balanced", "quality", "high_quality")
IMAGES = []
PIXELS = []
for path in IMAGE_PATHS:
    with Image.open(path) as image:
        normalized = ImageOps.exif_transpose(image).convert("RGB")
    IMAGES.append(normalized)
    PIXELS.append(np.asarray(normalized, dtype=np.uint8))


def evaluate(image_index, preset):
    config = getattr(PBC3Config, preset)()
    if config.random_seed != 2003 or not config.compute_final_mse:
        raise AssertionError("preset seed or internal MSE differs from the frozen defaults")
    start = time.perf_counter()
    encoded = PBC3.compress(IMAGES[image_index], config=config)
    encode_seconds = time.perf_counter() - start
    start = time.perf_counter()
    decoded = PBC3.decompress(encoded.data)
    decode_seconds = time.perf_counter() - start

    output = np.asarray(decoded.image.convert("RGB"), dtype=np.uint8)
    source = PIXELS[image_index]
    mse = float(np.mean((source.astype(np.float32) - output.astype(np.float32)) ** 2, dtype=np.float64))
    body = PBC3._open_body(encoded.data)[1]
    patch_count = int(PBC3._read_header(BitReader(body))[10])
    return {
        "image": IMAGE_PATHS[image_index].name,
        "preset": preset,
        "encode_seconds": encode_seconds,
        "decode_seconds": decode_seconds,
        "stream_sha256": hashlib.sha256(encoded.data).hexdigest(),
        "decoded_rgb_sha256": hashlib.sha256(output.tobytes()).hexdigest(),
        "bytes": len(encoded.data),
        "bpp": 8 * len(encoded.data) / (source.shape[0] * source.shape[1]),
        "patch_count": patch_count,
        "internal_mse": float(encoded.mse),
        "rgb_mse": mse,
        "size": [int(source.shape[1]), int(source.shape[0])],
    }


for line in sys.stdin:
    request = json.loads(line)
    if request["op"] == "warm":
        for preset in PRESETS:
            evaluate(0, preset)
        print(json.dumps({"ready": True, "source": str(SOURCE), "warm_presets": list(PRESETS)}), flush=True)
    elif request["op"] == "measure":
        print(json.dumps(evaluate(request["image_index"], request["preset"])), flush=True)
    elif request["op"] == "stop":
        break
