import hashlib
import json
import pathlib
import sys
import time

import numpy as np
import threadpoolctl
from PIL import Image

HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'research_pass42'))
import pbc4_codec
import pbc4_decode_opt

path = pathlib.Path(sys.argv[1])
output_dir = pathlib.Path(sys.argv[2])
data = path.read_bytes()
results = {}
with threadpoolctl.threadpool_limits(limits=1, user_api='blas'):
    for label, decoder in (('ref', pbc4_codec.decode), ('opt', pbc4_decode_opt.decode)):
        start = time.perf_counter()
        pixels = decoder(data)
        elapsed = time.perf_counter() - start
        target = output_dir / f'{path.stem}_{label}_fresh.png'
        Image.fromarray(pixels).save(target)
        results[label] = {'seconds': elapsed, 'sha256': hashlib.sha256(pixels.tobytes()).hexdigest(),
                          'dimensions': [int(pixels.shape[1]), int(pixels.shape[0])],
                          'png_path': str(target)}
print(json.dumps(results, separators=(',', ':')))
