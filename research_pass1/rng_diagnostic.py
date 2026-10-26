import hashlib
import json
import os
from pathlib import Path
import sys

from benchmark import ROOT, OUT, THREAD_ENV

os.environ.update(THREAD_ENV)
sys.path.insert(0, str(ROOT))
import numpy as np
import psutil
from PIL import Image, ImageOps
from PBC3 import PBC3
from pbc3_types import PBC3Config
import pbc3_ops as ops


if __name__ == '__main__':
    process = psutil.Process()
    process.cpu_affinity([process.cpu_affinity()[0]])
    rows = []
    production_rng = ops.PBC3Rng
    for name in ('flower.jpg', 'googlewiki.png', 'monet.jpg'):
        image = ImageOps.exif_transpose(Image.open(ROOT / 'hpt_data' / name)).convert('RGB')
        for preset in ('compression', 'balanced', 'quality', 'high_quality'):
            config = getattr(PBC3Config, preset)(learned_filler_model_path=str(ROOT / 'patch_policy.npz'))
            ops.PBC3Rng = np.random.default_rng
            try:
                result = PBC3.compress(image, config=config)
            finally:
                ops.PBC3Rng = production_rng
            reference = (OUT / 'streams' / 'release' / 'hpt_data' / (name + '.' + preset + '.pbc3')).read_bytes()
            rows.append({'image': name, 'preset': preset, 'current_with_numpy_rng_equals_release': result.data == reference,
                         'stream_sha256': hashlib.sha256(result.data).hexdigest()})
        print('RNG attribution', name, flush=True)
    (OUT / 'rng_attribution.json').write_text(json.dumps(rows, indent=2))
