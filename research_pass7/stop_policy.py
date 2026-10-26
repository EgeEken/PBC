from contextlib import contextmanager
import time

import numpy as np


@contextmanager
def enabled(mode, stats):
    import learned_filler as lf
    started = time.perf_counter()
    original_select = lf.LearnedFiller.select_patch
    original_propose = lf.propose_boxes
    original_build = lf.build_action
    observed = {}
    stats.update(mode=mode, policy='legacy single-action/value; no stop head', accepted=0,
        stop_head=0, nonpositive=0, no_proposals=0, alternatives=0, attempts=0, rescued=0,
        second_nonpositive=0, extra_inference_seconds=0., extra_construction_seconds=0.,
        extra_total_seconds=0., events=[])

    def propose(*args, **kwargs):
        boxes = original_propose(*args, **kwargs)
        observed['boxes'] = boxes
        return boxes

    def build(*args, **kwargs):
        result = original_build(*args, **kwargs)
        observed['reduction'] = result[3]
        return result

    def select(self, target, canvas, config, rng, channel_bits, step, current_channel, trace=None, abs_error=None):
        assert self.candidates == 1 and self.top_k == 1
        observed.clear()
        result = original_select(self, target, canvas, config, rng, channel_bits, step, current_channel,
            trace=trace, abs_error=abs_error)
        stats['requested_budget'] = config.patch_count
        if result[0] is not None:
            stats['accepted'] += 1
            return result
        boxes = observed.get('boxes', [])
        reason = 'no_proposals' if not boxes else 'nonpositive'
        stats[reason] += 1
        event = dict(step=step, remaining=config.patch_count-step+1, reason=reason, proposals=len(boxes),
            top_reduction=observed.get('reduction'), rescued=False)
        stats['events'].append(event)
        if len(boxes) > 1:
            stats['alternatives'] += 1
        if mode != 'S2' or reason != 'nonpositive' or len(boxes) < 2:
            return result
        extra_started = time.perf_counter()
        stats['attempts'] += 1
        h, w, channels = target.shape
        box = boxes[1]
        infer_started = time.perf_counter()
        q = self.q_override if self.q_override >= 0 else lf.ops.interp(config.q_start, config.q_end, step, config.patch_count)
        features = self._featurize(target, canvas, [box], step, q, w, h, config.patch_count, channels).astype(np.float32)
        logits, value = self._forward((features-self.feat_mean)/self.feat_std, trace=trace)
        action = int(logits.argmax(1)[0])
        # The legacy value head ranks a singleton; it has no STOP output or threshold.
        assert value.shape[0] == 1
        stats['extra_inference_seconds'] += time.perf_counter()-infer_started
        build_started = time.perf_counter()
        patch, values, delta, reduction, bits = original_build(target, canvas, box, config,
            channel_bits, action, trace=trace)
        stats['extra_construction_seconds'] += time.perf_counter()-build_started
        event['second_reduction'] = reduction
        if reduction > 0:
            stats['accepted'] += 1
            stats['rescued'] += 1
            event['rescued'] = True
            result = patch, values, delta
        else:
            stats['second_nonpositive'] += 1
        stats['extra_total_seconds'] += time.perf_counter()-extra_started
        return result

    try:
        lf.propose_boxes = propose
        lf.build_action = build
        lf.LearnedFiller.select_patch = select
        yield
    finally:
        lf.LearnedFiller.select_patch = original_select
        lf.propose_boxes = original_propose
        lf.build_action = original_build
        stats['budget_exhausted'] = stats.get('accepted') == stats.get('requested_budget')
        stats['encode_scope_seconds'] = time.perf_counter()-started
