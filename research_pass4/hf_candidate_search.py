from contextlib import contextmanager
import copy
import inspect

import numpy as np


class SearchReuse:
    def __init__(self, incremental, verify=False):
        self.incremental = incremental
        self.verify = verify
        self.target = self.canvas = None
        self.errors = {}
        self.totals = {}
        self.counters = {}

    def count(self, key, value=1):
        self.counters[key] = self.counters.get(key, 0) + value

    def owner(self, target, canvas):
        if target is not self.target or canvas is not self.canvas:
            self.errors.clear()
            self.totals.clear()
            self.target, self.canvas = target, canvas
            self.count('state_initializations')

    def score(self, target, canvas, channel):
        self.owner(target, canvas)
        if not self.incremental or channel not in self.errors:
            error = np.abs(target[:, :, channel] - np.clip(canvas[:, :, channel], 0, 255))
            self.errors[channel] = error
            self.totals[channel] = int(np.sum(error))
            self.count('full_map_refreshes')
        if self.verify:
            actual = np.abs(target[:, :, channel] - np.clip(canvas[:, :, channel], 0, 255))
            assert np.array_equal(actual, self.errors[channel])
            assert int(np.sum(actual)) == self.totals[channel]
            self.count('verified_totals')
        return float(self.totals[channel])

    def error(self, target, canvas, channel):
        self.owner(target, canvas)
        if channel not in self.errors:
            self.score(target, canvas, channel)
        error = self.errors[channel]
        if self.verify:
            expected = np.abs(target[:, :, channel] - np.clip(canvas[:, :, channel], 0, 255).astype(np.int32))
            assert np.array_equal(error, expected)
            self.count('verified_search_maps')
        self.count('reused_search_maps')
        return error

    def update(self, layer, x, y, w, h):
        if self.canvas is None:
            return
        offset = layer.ctypes.data - self.canvas.ctypes.data
        channel, remainder = divmod(offset, self.canvas.itemsize)
        if remainder or channel not in range(self.canvas.shape[2]) or layer.shape != self.canvas.shape[:2] or layer.strides != self.canvas[:, :, 0].strides:
            self.target = self.canvas = None
            self.errors.clear()
            self.totals.clear()
            self.count('owner_invalidations')
            return
        if channel not in self.errors:
            return
        region = self.errors[channel][y:y + h, x:x + w]
        old_sum = int(np.sum(region))
        error = np.abs(self.target[y:y + h, x:x + w, channel] - np.clip(layer[y:y + h, x:x + w], 0, 255))
        region[:] = error
        self.totals[channel] += int(np.sum(error)) - old_sum
        self.count('rectangle_updates')
        self.count('updated_pixels', w * h)
        self.count('full_pixels_avoided', self.canvas.shape[0] * self.canvas.shape[1] - w * h)

    @contextmanager
    def enabled(self):
        from PBC3 import PBC3
        import pbc3_ops as ops
        old_search, old_delta, old_grid = ops.search_boxes, ops.apply_delta, ops.apply_grid
        score_descriptor = PBC3.__dict__['_channel_sum_error']
        code = inspect.getsource(old_search)
        needle = 'visible = np.clip(canvas[:, :, channel], 0, 255).astype(np.int32)\n        abs_error = np.abs(target[:, :, channel] - visible)'
        assert code.count(needle) == 1
        code = code.replace(needle, 'abs_error = cached_error(target, canvas, channel)')
        namespace = dict(vars(ops), cached_error=self.error)
        exec(compile(code, __file__, 'exec'), namespace)
        reused_search = namespace['search_boxes']
        def search(target, canvas, config, rng, channel, depth, search_q, ordered=False, trace=None):
            if self.verify:
                cloned = copy.deepcopy(rng)
                expected = old_search(target, canvas, config, cloned, channel, depth, search_q, ordered=ordered)
            result = reused_search(target, canvas, config, rng, channel, depth, search_q, ordered=ordered, trace=trace)
            if self.verify:
                assert result == expected and rng.state == cloned.state
                self.count('verified_proposal_lists_and_rng')
            return result
        def apply_delta(layer, x, y, w, h, delta):
            old_delta(layer, x, y, w, h, delta)
            self.update(layer, x, y, w, h)
        def apply_grid(layer, x, y, w, h, cell, values, trace=None):
            old_grid(layer, x, y, w, h, cell, values, trace=trace)
            self.update(layer, x, y, w, h)
        try:
            PBC3._channel_sum_error = staticmethod(self.score)
            ops.search_boxes = search
            if self.incremental:
                ops.apply_delta, ops.apply_grid = apply_delta, apply_grid
            yield self.counters
        finally:
            PBC3._channel_sum_error = score_descriptor
            ops.search_boxes, ops.apply_delta, ops.apply_grid = old_search, old_delta, old_grid
