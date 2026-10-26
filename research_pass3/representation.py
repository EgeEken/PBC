import time

import numpy as np


def choices(cell, bits, width, height, config, finer=False):
    candidates = [('coarser_grid', min(cell * 2, config.max_cell_size, width, height), bits),
                  ('lower_precision', cell, max(1, bits - 1))]
    if finer:
        candidates.append(('finer_grid', max(1, cell // 2), bits))
    seen = {(cell, bits)}
    result = []
    for name, new_cell, new_bits in candidates:
        if (new_cell, new_bits) not in seen:
            result.append((name, new_cell, new_bits))
            seen.add((new_cell, new_bits))
    return result


def select(target, canvas, box, config, channel_bits, original, weight, counters):
    import learned_filler as filler
    patch, values, delta, gain, bits = original
    counters['policy_actions'] = counters.get('policy_actions', 0) + 1
    if gain <= 0:
        counters['policy_stops_preserved'] = counters.get('policy_stops_preserved', 0) + 1
        return original
    rate_lambda = weight * gain / max(1.0, bits)
    best, best_score = original, gain - rate_lambda * bits
    started = time.perf_counter()
    for name, cell, precision in choices(patch['cell_size'], patch['max_bitcount'], box[3], box[4], config):
        candidate = filler._build_action(target, canvas, box, config, channel_bits, cell, precision)
        counters['extra_representations'] = counters.get('extra_representations', 0) + 1
        score = candidate[3] - rate_lambda * candidate[4]
        if candidate[3] > 0 and score > best_score:
            best, best_score = candidate, score
    counters['extra_selection_seconds'] = counters.get('extra_selection_seconds', 0) + time.perf_counter() - started
    counters['original_raw_bits'] = counters.get('original_raw_bits', 0) + bits
    counters['selected_raw_bits'] = counters.get('selected_raw_bits', 0) + best[4]
    counters['original_local_sse_gain'] = counters.get('original_local_sse_gain', 0) + gain
    counters['selected_local_sse_gain'] = counters.get('selected_local_sse_gain', 0) + best[3]
    counters['representations_changed'] = counters.get('representations_changed', 0) + int(best is not original)
    return best


def replay(image, result, weight, identity, counters):
    from dataclasses import replace
    from PBC3 import PBC3
    import pbc3_ops as ops
    import learned_filler as filler
    from stream_tools import parse, serialize

    header, original, _ = parse(result.data)
    config = result.config
    target = PBC3.prepare(image, config)['target']
    canvas = PBC3._decode_to_canvas(result.data, max_patches=0)[0]
    selected = []
    for index, patch in enumerate(original):
        box = tuple(patch[k] for k in ('channel', 'x', 'y', 'w', 'h'))
        c, x, y, w, h = box
        if index < header[6]:
            palette = ops.palette_generator(patch['mask'], patch['max_bitcount'], patch['neg'], patch['pos'], header[8])
            delta = ops.signed_resample(palette[patch['indices']], h, w)
            chosen = patch
        else:
            reference = filler._build_action(target, canvas, box, config, header[7], patch['cell_size'], patch['max_bitcount'])
            if identity:
                assert all(np.array_equal(reference[0][k], patch[k]) for k in patch)
                chosen, _, delta, _, _ = reference
            else:
                chosen, _, delta, _, _ = select(target, canvas, box, config, header[7], reference, weight, counters)
        selected.append(chosen)
        ops.apply_delta(canvas[:, :, c], x, y, w, h, delta)
    data = serialize(header, selected)
    if identity:
        assert data == result.data
    out = PBC3._canvas_to_image(canvas, header[5], header[9])
    if out.size != image.size:
        out = out.resize(image.size, PBC3.RESAMPLE_FILTER, reducing_gap=PBC3.RESAMPLE_REDUCING_GAP)
    counters['fixed_geometry_patches'] = len(selected) - header[6]
    return replace(result, image=out, data=data, mse=ops.final_mse(image, out) if config.compute_final_mse else None,
        total_bits=len(data) * 8, timings=None)
