import hashlib
import json
import pathlib
import sys
import time

import threadpoolctl

HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'research_pass42'))
import pbc4_codec_v2

path = pathlib.Path(sys.argv[1])
data = path.read_bytes()
with threadpoolctl.threadpool_limits(limits=1, user_api='blas'):
    start = time.perf_counter()
    pixels, info = pbc4_codec_v2.decode(data, return_info=True)
    elapsed = time.perf_counter() - start
print(json.dumps({'seconds': elapsed, 'pixel_sha256': hashlib.sha256(pixels.tobytes()).hexdigest(),
                  'dimensions': [pixels.shape[1], pixels.shape[0]], 'skip_exact': info['skip_exact']}))
