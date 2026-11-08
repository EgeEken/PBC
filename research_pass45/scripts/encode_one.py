import hashlib
import json
import pathlib
import sys
import time

import numpy as np
from PIL import Image, ImageOps

source_dir = pathlib.Path(sys.argv[1]).resolve()
image_path = pathlib.Path(sys.argv[2]).resolve()
preset = sys.argv[3]
output_path = pathlib.Path(sys.argv[4]).resolve()
sys.path.insert(0, str(source_dir))
sys.path.insert(0, str(source_dir / "pbc3"))
import os
from PBC3 import PBC3
from pbc3_types import PBC3Config

config = getattr(PBC3Config, preset)()
if not pathlib.Path(config.learned_filler_model_path).is_file():
    candidates = [source_dir / "pbc3" / "patch_policy.npz", source_dir / "patch_policy.npz"]
    config.learned_filler_model_path = str(next(path for path in candidates if path.is_file()))
config_values = {**config.__dict__, "learned_filler_model_path": "<model>"}
config_json = json.dumps(config_values, sort_keys=True, default=str, separators=(",", ":"))
with Image.open(image_path) as image:
    original = ImageOps.exif_transpose(image).convert("RGB")
original_array = np.asarray(original, dtype=np.int32)
start = time.perf_counter()
result = PBC3.compress(original, config=config, rng_mode="release")
encode_seconds = time.perf_counter() - start
output_path.write_bytes(result.data)
start = time.perf_counter()
decoded = PBC3.decompress(result.data).image.convert("RGB")
decode_seconds = time.perf_counter() - start
decoded_array = np.asarray(decoded, dtype=np.uint8)
diff = original_array - decoded_array.astype(np.int32)
print(json.dumps({
    "source_dir": str(source_dir),
    "preset": preset,
    "bytes": len(result.data),
    "file_sha256": hashlib.sha256(result.data).hexdigest(),
    "decoded_rgb_sha256": hashlib.sha256(decoded_array.tobytes()).hexdigest(),
    "rgb_mse": float(np.mean(diff * diff, dtype=np.float64)),
    "internal_mse": result.mse,
    "encode_seconds": encode_seconds,
    "decode_seconds": decode_seconds,
    "model_sha256": hashlib.sha256(pathlib.Path(config.learned_filler_model_path).read_bytes()).hexdigest(),
    "config_sha256": hashlib.sha256(config_json.encode()).hexdigest(),
}))
