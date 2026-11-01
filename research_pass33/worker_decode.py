from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import psutil

from pbc33_codec import decode, sha


def main(path: str) -> None:
    manifest = json.loads(Path(path).read_text())
    rows = []
    for item in manifest['streams']:
        base = np.load(item['base_npy'], allow_pickle=False)
        profile = {}
        pixels, modes, params = decode(Path(item['stream']).read_bytes(), base, profile)
        rows.append({'key': item['key'], 'pixels_sha256': sha(pixels.tobytes()),
                     'modes_sha256': sha(modes.tobytes()), 'params_sha256': sha(params),
                     'profile': profile})
    print(json.dumps({'rows': rows, 'process_peak_working_set_bytes': psutil.Process().memory_info().peak_wset}))


if __name__ == '__main__':
    main(sys.argv[1])
