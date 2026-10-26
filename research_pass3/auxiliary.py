import argparse
import contextlib
import json
import os
import subprocess
import sys
import time
import tracemalloc

from support import ROOT, OUT, PASS2, THREAD_ENV, save, snapshot, sha


def memory():
    import numpy as np
    import psutil
    sys.path.insert(0, str(ROOT))
    from pbc3_types import BitReader, BitWriter
    import pbc3_stream as scalar
    import grid_io as vector
    before = snapshot()
    psutil.Process().cpu_affinity([0])
    rng = np.random.default_rng(2003)
    rows = []
    for count in (16, 65536, 500000):
        for width in (1, 5, 9):
            values = rng.integers(0, 1 << width, count, dtype=np.uint16)
            bw = BitWriter()
            bw.write(3, 3)
            scalar.write_grid(bw, values, width)
            raw = bw.finish()
            for label, module in (('scalar', scalar), ('numpy', vector)):
                for operation in ('write', 'read'):
                    tracemalloc.start()
                    if operation == 'write':
                        bw = BitWriter()
                        bw.write(3, 3)
                        module.write_grid(bw, values, width)
                        encoded = bw.finish()
                    else:
                        br = BitReader(raw)
                        br.read(3)
                        decoded = module.read_grid(br, count, width)
                    current, peak = tracemalloc.get_traced_memory()
                    tracemalloc.stop()
                    assert encoded == raw if operation == 'write' else np.array_equal(decoded, values)
                    rows.append(dict(implementation=label, operation=operation, count=count, width=width,
                        python_tracked_current_bytes=current, python_tracked_peak_bytes=peak, exact=True))
    save('grid_allocation', rows)
    save('grid_allocation_provenance', dict(before=before, after=snapshot(),
        scope='Separate allocation probe, no speed claim; Python/NumPy tracked allocations, not a total native RSS bound'))


def startup_child(mode):
    import psutil
    process = psutil.Process()
    process.cpu_affinity([process.cpu_affinity()[0]])
    sys.path.insert(0, str(ROOT))
    started = time.perf_counter()
    from PBC3 import PBC3
    imported = time.perf_counter()
    if mode == 'grid':
        import grid_io
    extra = time.perf_counter()
    print(json.dumps(dict(mode=mode, codec_import_seconds=imported - started, extra_grid_import_seconds=extra - imported)))


def startup():
    before = snapshot()
    rows = []
    for trial in range(3):
        for mode in ('baseline', 'grid') if trial % 2 == 0 else ('grid', 'baseline'):
            started = time.perf_counter()
            output = subprocess.check_output([sys.executable, str(OUT / 'auxiliary.py'), 'startup_child', '--variant', mode],
                env={**os.environ, **THREAD_ENV}, text=True, cwd=ROOT)
            row = json.loads(output)
            row.update(trial=trial, full_child_wall_seconds=time.perf_counter() - started)
            rows.append(row)
    save('startup', rows)
    save('startup_provenance', dict(before=before, after=snapshot(), scope='Fresh processes, warm OS file cache; no pristine machine cold-start or first-encode claim'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('memory', 'startup', 'startup_child'))
    parser.add_argument('--variant', choices=('baseline', 'grid'))
    args = parser.parse_args()
    if args.mode == 'startup_child':
        startup_child(args.variant)
    else:
        globals()[args.mode]()
