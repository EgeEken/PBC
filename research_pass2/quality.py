import json
import os
import sys
import time

from common import ROOT, OUT, PASS1, SCREEN, PRESETS, THREAD_ENV, save, snapshot, run_plan, images


def ablations():
    variants = {
        'hq_budget50': ('high_quality', {'patch_count': 50}),
        'quality_q095': ('quality', {'learned_filler_q': 0.95}),
        'hq_search05': ('high_quality', {'search_q_start': 0.5}),
    }
    save('allocation_screen_plan', {'fixed_before_measurement': snapshot(), 'variants': variants,
        'hypotheses': ['larger budget', 'representation quality input', 'initial proposal weighting']})
    cases = [dict(path=str(ROOT / 'hpt_data' / name), preset=preset, source='current', label=label, overrides=overrides)
        for name in SCREEN for label, (preset, overrides) in variants.items()]
    run_plan('allocation_screen', cases, 2)


def planned(name):
    plan = json.loads((OUT / (name + '_plan.json')).read_text())
    run_plan(name, plan['cases'], plan.get('repeats', 3))


def trajectories():
    os.environ.update(THREAD_ENV)
    sys.path.insert(0, str(ROOT))
    import numpy as np
    import psutil
    from PIL import Image, ImageOps
    from PBC3 import PBC3
    import pbc3_ops as ops
    from stream_tools import parse, serialize
    from optimizations import enabled

    before = snapshot()
    psutil.Process().cpu_affinity([0])
    rows, patches = [], []
    controls = [r for r in json.loads((OUT / 'resolution_screen.json').read_text()) if r['trial'] == 0 and r['config']['auto_downsample_max_pixels'] == 250000]
    # Trajectory diagnostics serialize frozen prefixes; they are not fresh-budget encodes.
    for row in controls:
        data = (OUT / row['stream_path']).read_bytes()
        h, ps, accounting = parse(data)
        with Image.open(ROOT / row['dataset'] / row['image']) as opened:
            image = ImageOps.exif_transpose(opened).convert('RGB')
        reference = np.asarray(image)
        previous = None
        for count in range(h[6], len(ps) + 1):
            with enabled('grid_numpy'):
                prefix = serialize(h, ps[:count])
                decoded = PBC3.decompress(prefix).image
            error = np.subtract(reference, np.asarray(decoded), dtype=np.float32)
            mse = float(np.mean(error ** 2, dtype=np.float64))
            point = dict(source=row['source'], image=row['image'], preset=row['preset'], prefix_iterative=count - h[6],
                mse=mse, final_bytes=len(prefix), bpp=8 * len(prefix) / (image.width * image.height),
                raw_bits=accounting['header_bits'] + sum(p['total_bits'] for p in accounting['patches'][:count]) + 48,
                parent_stream_sha256=row['stream_sha256'])
            if previous:
                point['marginal_sse_reduction'] = (previous['mse'] - mse) * image.width * image.height * 3
                point['marginal_file_bits'] = 8 * (len(prefix) - previous['final_bytes'])
                point['sse_reduction_per_added_file_bit'] = point['marginal_sse_reduction'] / point['marginal_file_bits'] if point['marginal_file_bits'] > 0 else None
            rows.append(point)
            previous = point
        coverages = np.zeros((h[4], h[3], h[6]), dtype=np.uint16)
        for p, ledger in zip(ps, accounting['patches']):
            region = coverages[p['y']:p['y'] + p['h'], p['x']:p['x'] + p['w'], p['channel']]
            patches.append(dict(source=row['source'], image=row['image'], preset=row['preset'], **ledger,
                max_bitcount=p['max_bitcount'], palette_mask=p['mask'], negative_bound=p['neg'], positive_bound=p['pos'],
                previous_iterative_coverage_fraction=float(np.mean(region > 0)), area=p['w'] * p['h']))
            if not ledger['initialization']:
                region += 1
        print('trajectory', row['source'], row['image'], row['preset'], flush=True)
    save('quality_trajectories', rows)
    save('quality_patch_ledgers', patches)
    save('quality_trajectory_provenance', dict(before=before, after=snapshot(), prefix_cost='full original encode plus extraction; no encoder speed claim'))


if __name__ == '__main__':
    if sys.argv[1] == 'planned':
        planned(sys.argv[2])
    else:
        globals()[sys.argv[1]]()
