import argparse
import sys

from support import ROOT, OUT, PRESETS, SCREEN, images, save, snapshot


def plan(stage):
    paths = [ROOT / 'hpt_data' / n for n in SCREEN] if stage == 'screen' else images()
    variants = [('baseline', 'none', True), ('grid_skip', 'grid', False),
                ('search', 'search', True), ('all', 'all', False)]
    if stage == 'exactness':
        variants = [('baseline', 'none', True), ('all_verify', 'all_verify', False)]
    cases = [dict(path=str(p), preset=preset, label=label, optimization=mode,
        overrides={'compute_final_mse': metric}, retain_stream=stage != 'full')
        for p in paths for preset in PRESETS for label, mode, metric in variants]
    save('runtime_' + stage + '_plan', dict(decision=snapshot(), cases=cases, repeats=1 if stage == 'exactness' else 3))


def edges():
    sys.path.insert(0, str(ROOT))
    import numpy as np
    import psutil
    from PIL import Image
    from PBC3 import PBC3
    from pbc3_types import PBC3Config
    from variants import enabled, stats
    psutil.Process().cpu_affinity([psutil.Process().cpu_affinity()[0]])
    before = snapshot()
    rng = np.random.default_rng(2003)
    rows = []
    for mode in ('RGB', 'RGBA'):
        image = Image.fromarray(rng.integers(0, 256, (73, 97, len(mode)), dtype=np.uint8), mode)
        for preset in PRESETS:
            for reuse in (False, True):
                cfg = getattr(PBC3Config, preset)(patch_count=5, auto_downsample_max_pixels=4096,
                    reuse_selected_delta=reuse, learned_filler_model_path=str(ROOT/'patch_policy.npz'))
                control = PBC3.compress(image, config=cfg)
                expected = np.asarray(PBC3.decompress(control.data).image)
                with enabled('all_verify'):
                    candidate = PBC3.compress(image, config=cfg, compute_final_mse=False)
                    decoded = PBC3.decompress(candidate.data)
                counters = stats()
                assert control.data == candidate.data
                assert np.array_equal(expected, np.asarray(decoded.image))
                assert candidate.mse is None and control.mse is not None
                rows.append(dict(mode=mode, preset=preset, reuse_selected_delta=reuse,
                    byte_equal=True, pixel_equal=True, counters=counters))
    save('runtime_edges', dict(before=before, after=snapshot(), rows=rows))


if __name__ == '__main__':
    mode = argparse.ArgumentParser()
    mode.add_argument('stage', choices=('exactness', 'screen', 'full', 'edges'))
    args = mode.parse_args()
    edges() if args.stage == 'edges' else plan(args.stage)
