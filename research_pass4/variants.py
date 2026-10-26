from contextlib import contextmanager, ExitStack
import time

from support import PASS3, load

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
    import learned_filler as filler
    import pbc3_stream as stream
    all_runtime = name == 'all' or name.startswith('all_')
    with ExitStack() as stack:
        if 'grid' in name or all_runtime:
            candidate = load('pass4_grid_candidate', PASS3 / 'integration_candidate' / 'pbc3_stream.py')
            candidate.GRID_IO_MODE = 'numpy'
            old_read, old_write, old_class_read = stream.read_grid, stream.write_grid, PBC3.__dict__['_read_grid']
            stack.callback(setattr, stream, 'read_grid', old_read)
            stack.callback(setattr, stream, 'write_grid', old_write)
            stack.callback(setattr, PBC3, '_read_grid', old_class_read)
            stream.read_grid, stream.write_grid = candidate.read_grid, candidate.write_grid
            PBC3._read_grid = staticmethod(candidate.read_grid)
        if 'alloc' in name:
            original_build = filler.build_action
            stack.callback(setattr, filler, 'build_action', original_build)
            select = load('pass4_frozen_representation', PASS3 / 'representation.py').select
            cache = None
            if 'opt' in name:
                from allocator_cost import action_cache, prepare_score
                prepare_score()
                cache = name.split('opt_')[1]
            def build(target, canvas, box, config, channel_bits, action, trace=None):
                if cache is None:
                    reference = original_build(target, canvas, box, config, channel_bits, action, trace=trace)
                    return select(target, canvas, box, config, channel_bits, reference, 0.4, _stats)
                with action_cache(cache, _stats):
                    reference = original_build(target, canvas, box, config, channel_bits, action, trace=trace)
                    return select(target, canvas, box, config, channel_bits, reference, 0.4, _stats)
            filler.build_action = build
        if 'search' in name or all_runtime:
            SearchReuse = load('pass4_search_reuse', PASS3 / 'search_reuse.py').SearchReuse
            descriptor, ordinary = PBC3.__dict__['compress'], PBC3.compress
            stack.callback(setattr, PBC3, 'compress', descriptor)
            def compress(image, config=None, **kwargs):
                started = time.perf_counter()
                state = SearchReuse(incremental=True, verify='verify' in name)
                with state.enabled():
                    installed = time.perf_counter()
                    result = ordinary(image, config=config, **kwargs)
                    compressed = time.perf_counter()
                finished = time.perf_counter()
                _stats.update(state.counters)
                _stats.update(prototype_install_seconds=installed-started,
                    prototype_cleanup_seconds=finished-compressed)
                result.encode_seconds = finished-started
                return result
            PBC3.compress = staticmethod(compress)
        yield
