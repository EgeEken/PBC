import difflib

from support import ROOT, OUT, save, sha, snapshot


def prepare():
    before = snapshot()
    original = (ROOT / 'pbc3_stream.py').read_text()
    proposed = original.replace('import numpy as np', 'import operator\n\nimport numpy as np', 1)
    proposed = proposed.replace('def write_grid(', 'def _write_grid_scalar(', 1).replace('def read_grid(', 'def _read_grid_scalar(', 1)
    prototype = (OUT / 'grid_io.py').read_text()
    prototype = prototype[prototype.index('def read_grid('):]
    prototype = prototype.replace('def read_grid(', 'def _read_grid_numpy(', 1).replace('def write_grid(', 'def _write_grid_numpy(', 1)
    proposed += '\n\n' + prototype + '''

GRID_IO_MODE = "scalar"
_GRID_READERS = {"scalar": _read_grid_scalar, "numpy": _read_grid_numpy}
_GRID_WRITERS = {"scalar": _write_grid_scalar, "numpy": _write_grid_numpy}


def read_grid(br, count, bitcount):
    return _GRID_READERS[GRID_IO_MODE](br, count, bitcount)


def write_grid(bw, values, bitcount):
    return _GRID_WRITERS[GRID_IO_MODE](bw, values, bitcount)
'''
    candidate = OUT / 'integration_candidate'
    candidate.mkdir(exist_ok=True)
    (candidate / 'pbc3_stream.py').write_text(proposed)
    patch = ''.join(difflib.unified_diff(original.splitlines(keepends=True), proposed.splitlines(keepends=True),
        fromfile='a/pbc3_stream.py', tofile='b/pbc3_stream.py'))
    (OUT / 'proposed_grid_io.patch').write_text(patch)
    save('integration_patch_provenance', dict(before=before, after=snapshot(),
        production_source_sha256=sha((ROOT / 'pbc3_stream.py').read_bytes()),
        candidate_sha256=sha((candidate / 'pbc3_stream.py').read_bytes()), applied=False,
        default='scalar', selection='Dedicated serial benchmark worker only; no live server mode toggling'))


if __name__ == '__main__':
    prepare()
