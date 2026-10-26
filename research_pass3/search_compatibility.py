import os
import sys

from support import ROOT, THREAD_ENV, save, snapshot, sha

os.environ.update(THREAD_ENV)
sys.path.insert(0, str(ROOT))
import numpy as np
import psutil
from PIL import Image
from PBC3 import PBC3
from pbc3_types import PBC3Config, BitReader
import pbc3_ops as ops
from search_reuse import SearchReuse


def check():
    before = snapshot()
    psutil.Process().cpu_affinity([0])
    rng = np.random.default_rng(2003)
    state = SearchReuse(True, verify=True)
    target = rng.integers(0, 256, (17, 19, 4), dtype=np.int32)
    canvas = rng.integers(-32768, 32768, target.shape, dtype=np.int16)
    with state.enabled():
        for channel in range(4):
            state.score(target, canvas, channel)
        for _ in range(100):
            channel = int(rng.integers(0, 4))
            x, y = int(rng.integers(0, 19)), int(rng.integers(0, 17))
            w, h = int(rng.integers(1, 20 - x)), int(rng.integers(1, 18 - y))
            delta = rng.integers(-300, 301, (h, w), dtype=np.int16)
            ops.apply_delta(canvas[:, :, channel], x, y, w, h, delta)
            state.score(target, canvas, channel)
            state.error(target, canvas, channel)
    rows = []
    for channels in (3, 4):
        image = Image.fromarray(rng.integers(0, 256, (97, 131, channels), dtype=np.uint8))
        for reuse_delta in (False, True):
            for warmup in (-1, .5):
                config = PBC3Config.quality(patch_count=6, auto_downsample_max_pixels=4096,
                    warm_downsample_max_pixels=12000, warmup_ratio=warmup, reuse_selected_delta=reuse_delta,
                    learned_filler_model_path=str(ROOT / 'patch_policy.npz'))
                baseline = PBC3.compress(image, config=config)
                pixels = np.asarray(PBC3.decompress(baseline.data).image)
                for incremental in (False, True):
                    reuse = SearchReuse(incremental, verify=True)
                    with reuse.enabled():
                        for repeat in range(2):
                            candidate = PBC3.compress(image, config=config)
                            assert candidate.data == baseline.data
                            assert np.array_equal(np.asarray(PBC3.decompress(candidate.data).image), pixels)
                    _, raw = PBC3._open_body(candidate.data)
                    header = PBC3._read_header(BitReader(raw))
                    rows.append(dict(channels=channels, reuse_delta=reuse_delta, warmup=warmup,
                        emitted_warmup=header[12], incremental=incremental, repeats=2, exact=True,
                        stream_sha256=sha(candidate.data), counters=reuse.counters))
    save('search_compatibility', dict(signed_int16_update_counters=state.counters, image_checks=rows))
    save('search_compatibility_provenance', dict(before=before, after=snapshot()))


if __name__ == '__main__':
    check()
