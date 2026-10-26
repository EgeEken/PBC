import dataclasses
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PBC3 import PBC3  # noqa: E402

CELL_SIZES = [1, 2, 4, 8, 16, 32]
BITCOUNTS = [1, 2, 3, 4, 5]
ACTIONS = [(cs, bc) for cs in CELL_SIZES for bc in BITCOUNTS]
NUM_ACTIONS = len(ACTIONS)

# Factored (multi-discrete) action axes for the RL filler. mask_size is a new
# degree of freedom (previously fixed by config) that strongly shapes the palette.
MASK_SIZES = [2, 4, 6, 8, 16]
FACTORED_AXES = (CELL_SIZES, BITCOUNTS, MASK_SIZES)
FACTORED_HEAD_SIZES = tuple(len(a) for a in FACTORED_AXES)

# Place-or-stop axis appended as the LAST policy head: the rate lever. The policy
# decides each step whether to place the proposed patch (0) or end the encode (1),
# so it can spend fewer bits on demand (low bpp) instead of always filling. Kept
# separate from FACTORED_HEAD_SIZES so build_action_factored indexing is unchanged.
STOP_PLACE, STOP_STOP = 0, 1
STOP_AXIS = len(FACTORED_HEAD_SIZES)
POLICY_HEAD_SIZES = FACTORED_HEAD_SIZES + (2,)


def action_params(idx):
    cs, bc = ACTIONS[idx]
    return {"cell_size": cs, "patch_palette_bitcount": bc, "palette_mode": "generated"}


def lambda_for_q(q, lmin=0.05, lmax=5.0):
    q = float(np.clip(q, 0.0, 1.0))
    return lmin * (lmax / lmin) ** (1.0 - q)


def action_rewards(reductions, bits, q, lmin=0.05, lmax=5.0):
    """Marginal RD value per action in absolute (total) units, so values are
    comparable across boxes of different sizes."""
    lam = lambda_for_q(q, lmin, lmax)
    return np.asarray(reductions, np.float64) - lam * np.asarray(bits, np.float64)


def _score_patch(target, canvas, box, channel_bits, patch, values):
    c, x, y, bw, bh = box
    canvas_box = canvas[y:y + bh, x:x + bw, c]
    target_box = target[y:y + bh, x:x + bw, c]
    delta = PBC3.signed_resample(values, bh, bw).astype(np.int32)
    before = target_box - np.clip(canvas_box, 0, 255)
    after = target_box - np.clip(canvas_box + delta, 0, 255)
    before_sse = float(np.sum(before.astype(np.int64) ** 2))
    reduction = before_sse - float(np.sum(after.astype(np.int64) ** 2))
    bits = float(PBC3._patch_bits_for(patch, channel_bits))
    return reduction, bits


def build_action(target, canvas, box, config, channel_bits, idx):
    c, x, y, bw, bh = box
    cs, bc = ACTIONS[idx]
    cell = max(1, min(cs, bw, bh))
    residual = target[y:y + bh, x:x + bw, c] - canvas[y:y + bh, x:x + bw, c]
    patch, values = PBC3._make_patch(c, x, y, bw, bh, cell, residual, config, bc, PBC3.PALETTE_GENERATED, 0)
    reduction, bits = _score_patch(target, canvas, box, channel_bits, patch, values)
    return patch, values, reduction, bits


def build_action_factored(target, canvas, box, config, channel_bits, cs_idx, bc_idx, ms_idx):
    """Build a patch from factored action indices (cell_size, bitcount, mask_size)."""
    c, x, y, bw, bh = box
    cs, bc, ms = CELL_SIZES[cs_idx], BITCOUNTS[bc_idx], MASK_SIZES[ms_idx]
    cell = max(1, min(cs, bw, bh))
    cfg = dataclasses.replace(config, mask_size=ms)
    residual = target[y:y + bh, x:x + bw, c] - canvas[y:y + bh, x:x + bw, c]
    patch, values = PBC3._make_patch(c, x, y, bw, bh, cell, residual, cfg, bc, PBC3.PALETTE_GENERATED, 0)
    reduction, bits = _score_patch(target, canvas, box, channel_bits, patch, values)
    return patch, values, reduction, bits


def evaluate_box_actions(target, canvas, box, config, channel_bits):
    reductions = np.empty(NUM_ACTIONS, np.float64)
    bits = np.empty(NUM_ACTIONS, np.float64)
    for i in range(NUM_ACTIONS):
        _, _, reductions[i], bits[i] = build_action(target, canvas, box, config, channel_bits, i)
    return reductions, bits