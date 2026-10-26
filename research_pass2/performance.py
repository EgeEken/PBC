import json
import os
import sys
import time

from common import ROOT, OUT, SCREEN, PRESETS, THREAD_ENV, run_plan, images, save, snapshot


def correctness():
    before = snapshot()
    os.environ.update(THREAD_ENV)
    sys.path.insert(0, str(ROOT))
    import numpy as np
    from pbc3_types import BitReader, BitWriter
    import pbc3_stream as stream
    from optimizations import read_grid, write_grid, channel_sum

    rng = np.random.default_rng(123)
    checks = []
    for width in range(10):
        for offset in range(8):
            for count in (0, 1, 7, 8, 9, 31, 65, 1024):
                values = rng.integers(0, 1 << width, count, dtype=np.uint16)
                a, b = BitWriter(), BitWriter()
                prefix = (1 << offset) - 1
                a.write(prefix, offset)
                b.write(prefix, offset)
                stream.write_grid(a, values, width)
                write_grid(b, values, width)
                a.write(37, 6)
                b.write(37, 6)
                original, fast = a.finish(), b.finish()
                assert original == fast
                a, b = BitReader(original), BitReader(original)
                assert a.read(offset) == b.read(offset)
                assert np.array_equal(stream.read_grid(a, count, width), read_grid(b, count, width))
                assert (a.i, a.nbits, a.acc) == (b.i, b.nbits, b.acc)
                assert a.read(6) == b.read(6) == 37
                checks.append({'width': width, 'offset': offset, 'count': count, 'equal': True})
    total_checks = []
    for shape in ((1, 1, 3), (17, 31, 4), (501, 499, 3)):
        target = rng.integers(0, 256, shape, dtype=np.int32)
        canvas = rng.integers(-32768, 32768, shape, dtype=np.int16)
        for c in range(shape[2]):
            exact = float(np.sum(np.abs(target[:, :, c] - np.clip(canvas[:, :, c], 0, 255))))
            assert channel_sum(target, canvas, c) == exact
            total_checks.append({'shape': shape, 'channel': c, 'equal': True})
    save('optimization_unit_checks', {'grid_io': checks, 'channel_sum': total_checks,
        'scope': 'bits 0..9, every bit alignment, tails, empty grid, suffix-state preservation; signed int16 extremes and RGBA'})
    save('optimization_unit_provenance', dict(before=before, after=snapshot()))


def screen():
    cases = []
    for name in SCREEN:
        for preset in PRESETS:
            for method in ('none', 'channel_sum', 'grid_numpy', 'skip_final_mse'):
                cases.append(dict(path=str(ROOT / 'hpt_data' / name), preset=preset, source='current', label=method,
                    optimization=method if method in ('channel_sum', 'grid_numpy') else 'none',
                    overrides={'compute_final_mse': False} if method == 'skip_final_mse' else {}))
    run_plan('performance_screen', cases, 3)


def confirm():
    decision = json.loads((OUT / 'performance_promotion.json').read_text())
    cases = []
    for path in images():
        for preset in PRESETS:
            for method in ['none'] + decision['variants']:
                cases.append(dict(path=str(path), preset=preset, source='current', label=method,
                    optimization=method if method in ('channel_sum', 'grid_numpy') else 'none',
                    overrides={'compute_final_mse': False} if method == 'skip_final_mse' else {}))
    run_plan('performance_confirmation', cases, 3)


def components():
    cases = []
    for name in ('flower.jpg', 'monet.jpg', 'googlewiki.png'):
        for method in ('none', 'channel_sum', 'grid_numpy'):
            cases.append(dict(path=str(ROOT / 'hpt_data' / name), preset='balanced', source='current', label=method,
                optimization=method, profile=True))
    run_plan('performance_components', cases, 2)


def decoder_components():
    os.environ.update(THREAD_ENV)
    sys.path.insert(0, str(ROOT))
    import numpy as np
    import psutil
    from PBC3 import PBC3
    from optimizations import enabled
    from contextlib import nullcontext
    before = snapshot()
    psutil.Process().cpu_affinity([0])
    rows = []
    for name in ('flower.jpg', 'monet.jpg', 'googlewiki.png'):
        for preset in PRESETS:
            data = (ROOT / 'research_pass1' / 'streams' / 'current' / 'hpt_data' / (name + '.' + preset + '.pbc3')).read_bytes()
            expected = np.asarray(PBC3.decompress(data).image)
            for method in ('none', 'grid_numpy'):
                with enabled(method) if method != 'none' else nullcontext():
                    PBC3.decompress(data, trace=True)
            for trial in range(2):
                methods = ('none', 'grid_numpy') if trial == 0 else ('grid_numpy', 'none')
                for method in methods:
                    with enabled(method) if method != 'none' else nullcontext():
                        decoded = PBC3.decompress(data, trace=True)
                    assert np.array_equal(np.asarray(decoded.image), expected)
                    for phase, stats in decoded.timings['summary'].items():
                        rows.append(dict(image=name, preset=preset, trial=trial, method=method, phase=phase,
                            inclusive_seconds=stats['total_ns'] / 1e9, exclusive_seconds=stats['exclusive_ns'] / 1e9, count=stats['count']))
    save('decoder_components', rows)
    save('decoder_components_provenance', dict(before=before, after=snapshot(), scope='Traced decode phases; primary untraced full latency remains separate'))


if __name__ == '__main__':
    globals()[sys.argv[1]]()
