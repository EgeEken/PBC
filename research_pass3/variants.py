from contextlib import contextmanager

_stats = {}


def stats():
    global _stats
    result = dict(_stats)
    _stats = {}
    return result


@contextmanager
def enabled(name):
    global _stats
    _stats = {}
    from PBC3 import PBC3
    if name == 'grid_safe':
        import pbc3_stream as stream
        import sys
        from support import load, OUT
        candidate = sys.modules.get('pass3_integration_stream')
        if candidate is None:
            candidate = load('pass3_integration_stream', OUT / 'integration_candidate' / 'pbc3_stream.py')
        candidate.GRID_IO_MODE = 'numpy'
        read_grid, write_grid = candidate.read_grid, candidate.write_grid
        old_read, old_write, old_class_read = stream.read_grid, stream.write_grid, PBC3._read_grid
        try:
            stream.read_grid, stream.write_grid = read_grid, write_grid
            PBC3._read_grid = staticmethod(read_grid)
            yield
        finally:
            stream.read_grid, stream.write_grid = old_read, old_write
            PBC3._read_grid = staticmethod(old_class_read)
    elif name.startswith('fixed_'):
        import time
        from representation import replay
        descriptor = PBC3.__dict__['compress']
        ordinary = PBC3.compress
        identity = name == 'fixed_identity'
        weight = 0.0 if identity else float(name.split('_t')[1])
        def compress(image, config=None, **kwargs):
            started = time.perf_counter()
            result = ordinary(image, config=config, **kwargs)
            baseline_seconds = time.perf_counter() - started
            result = replay(image, result, weight, identity, _stats)
            result.encode_seconds = time.perf_counter() - started
            _stats['original_encode_seconds'] = baseline_seconds
            _stats['replay_and_finalize_seconds'] = result.encode_seconds - baseline_seconds
            return result
        try:
            PBC3.compress = staticmethod(compress)
            yield
        finally:
            PBC3.compress = descriptor
    elif name.startswith('integrated_t'):
        import learned_filler as filler
        from representation import select
        ordinary = filler.build_action
        weight = float(name.split('_t')[1])
        def build(target, canvas, box, config, channel_bits, action, trace=None):
            reference = ordinary(target, canvas, box, config, channel_bits, action, trace=trace)
            return select(target, canvas, box, config, channel_bits, reference, weight, _stats)
        try:
            filler.build_action = build
            yield
        finally:
            filler.build_action = ordinary
    elif name.startswith('search_'):
        import time
        from search_reuse import SearchReuse
        descriptor = PBC3.__dict__['compress']
        ordinary = PBC3.compress
        def compress(image, config=None, **kwargs):
            started = time.perf_counter()
            state = SearchReuse(incremental='rectangles' in name, verify=name.endswith('_verify'))
            with state.enabled():
                installed = time.perf_counter()
                result = ordinary(image, config=config, **kwargs)
                compressed = time.perf_counter()
            finished = time.perf_counter()
            _stats.update(state.counters)
            _stats.update(prototype_install_seconds=installed - started,
                prototype_cleanup_seconds=finished - compressed, core_compress_seconds=compressed - installed)
            result.encode_seconds = finished - started
            return result
        try:
            PBC3.compress = staticmethod(compress)
            yield
        finally:
            PBC3.compress = descriptor
    else:
        raise ValueError(name)
