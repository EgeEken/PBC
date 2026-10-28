import base64
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
import learned_filler
import pbc3_features
import pbc3_ops
REFERENCE_FEATURES = pbc3_features.extract_features
if os.environ.get("PBC_FEATURE_CANDIDATE") == "1":
    from check_features import candidate_function
    learned_filler.extract_features = candidate_function()

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
        "stream_base64": base64.b64encode(encoded.data).decode(),
        "ordered_body_sha256": hashlib.sha256(body).hexdigest(),
        "decoded_rgb_sha256": hashlib.sha256(output.tobytes()).hexdigest(),
        "bytes": len(encoded.data),
        "bpp": 8 * len(encoded.data) / (source.shape[0] * source.shape[1]),
        "patch_count": patch_count,
        "internal_mse": float(encoded.mse),
        "rgb_mse": mse,
        "size": [int(source.shape[1]), int(source.shape[0])],
    }


def diagnose(image_index, preset):
    feature_digest, proposal_digest, policy_digest = (hashlib.sha256() for _ in range(3))
    counts = {"features": 0, "proposals": 0, "policies": 0}
    feature_fn, sampler, forward = learned_filler.extract_features, pbc3_ops.sample_box, learned_filler.LearnedFiller._forward
    def features(*args, **kwargs):
        result = feature_fn(*args, **kwargs)
        reference = REFERENCE_FEATURES(*args, **kwargs)
        assert result.tobytes() == reference.tobytes(), "feature drift during encode"
        feature_digest.update(result.tobytes())
        counts["features"] += 1
        return result
    def sample(rng, *args, **kwargs):
        result = sampler(rng, *args, **kwargs)
        proposal_digest.update(json.dumps([result, rng.bit_generator.state], sort_keys=True).encode())
        counts["proposals"] += 1
        return result
    def policy(self, *args, **kwargs):
        logits, values = forward(self, *args, **kwargs)
        policy_digest.update(logits.tobytes()); policy_digest.update(values.tobytes())
        counts["policies"] += 1
        return logits, values
    learned_filler.extract_features, pbc3_ops.sample_box, learned_filler.LearnedFiller._forward = features, sample, policy
    try:
        row = evaluate(image_index, preset)
    finally:
        learned_filler.extract_features, pbc3_ops.sample_box, learned_filler.LearnedFiller._forward = feature_fn, sampler, forward
    row.update({"feature_sha256": feature_digest.hexdigest(), "proposal_rng_sha256": proposal_digest.hexdigest(), "policy_sha256": policy_digest.hexdigest(), "counts": counts})
    return row


for line in sys.stdin:
    request = json.loads(line)
    if request["op"] == "warm":
        for preset in PRESETS:
            evaluate(0, preset)
        print(json.dumps({"ready": True, "source": str(SOURCE), "warm_presets": list(PRESETS)}), flush=True)
    elif request["op"] == "measure":
        print(json.dumps(evaluate(request["image_index"], request["preset"])), flush=True)
    elif request["op"] == "diagnose":
        print(json.dumps(diagnose(request["image_index"], request["preset"])), flush=True)
    elif request["op"] == "stop":
        break
