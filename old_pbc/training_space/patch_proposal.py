"""Shared patch-proposal front-end for PBC3.

The encoder (`PBC3._select_patch`), the learned filler, and the RL trainers all
build candidate boxes the same way: top-k error anchors -> sampled boxes ->
abs-error/area prescore -> keep the top `proposal_depth`. This is the single
source of truth for that stage.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PBC3 import PBC3  # noqa: E402


def propose_boxes(target, canvas, config, rng, channel, step):
    """Return candidate boxes `(c, x, y, w, h)` for `channel`, sorted by prescore
    (descending). Consumes `rng` identically to the original inline loops."""
    h_img, w_img, _ = target.shape
    search_q = PBC3._interp(config.search_q_start, config.search_q_end, step, config.patch_count)
    visible = np.clip(canvas[:, :, channel], 0, 255).astype(np.int32)
    abs_error = np.abs(target[:, :, channel] - visible).astype(np.int64)
    integral_abs = PBC3._integral(abs_error)
    anchors = PBC3._top_anchors(abs_error.astype(np.float32), config.top_k, config.anchor_block_size, channel)
    if not anchors:
        return []
    specs, sums, areas = [], [], []
    for i in range(max(1, int(config.search_depth))):
        c, x, y, bw, bh, _, _ = PBC3._sample_box(rng, anchors[i % len(anchors)], w_img, h_img, config)
        s = integral_abs[y + bh, x + bw] - integral_abs[y, x + bw] - integral_abs[y + bh, x] + integral_abs[y, x]
        if s <= 0:
            continue
        sums.append(float(s)); areas.append(float(bw * bh)); specs.append((c, x, y, bw, bh))
    if not specs:
        return []
    pre = search_q * PBC3._norm(sums) - (1.0 - search_q) * PBC3._norm(areas)
    keep = sorted(PBC3._select_top_indices(pre, config.proposal_depth), key=lambda i: pre[i], reverse=True)
    return [specs[i] for i in keep]