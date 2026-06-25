import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_space"))
from PBC3 import PBC3  # noqa: E402

# ========= patch_actions.py START ================
import dataclasses
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

def lambda_for_q(q, lmin=0.05, lmax=5.0):
    q = float(np.clip(q, 0.0, 1.0))
    return lmin * (lmax / lmin) ** (1.0 - q)

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

# ========= patch_actions.py  END ================

from pbc3_features import extract_cheap, extract_features, FEATURE_NAMES, SORT_FEATURES  # noqa: E402

# ========= rl_state.py START ================

BOX_FEATURES = [n for n in FEATURE_NAMES if n not in SORT_FEATURES and n != "q"]
PROGRESS_FEATURES = ["lambda_log", "bpp_running", "patches_placed_frac", "patches_remaining_frac"]
STATE_NAMES = BOX_FEATURES + PROGRESS_FEATURES
STATE_DIM = len(STATE_NAMES)
import math

def extract_state(target, canvas, box, step, image_w, image_h, patch_count,
                  channels, lam, bits_spent, pixels):
    box_feats = extract_cheap(BOX_FEATURES, target, canvas, box, step, 0.0,
                              image_w, image_h, patch_count, channels)
    prog = np.array([
        math.log(max(float(lam), 1e-6)),
        bits_spent / max(1, pixels),
        step / max(1, patch_count),
        (patch_count - step) / max(1, patch_count),
    ], dtype=np.float32)
    return np.concatenate([box_feats, prog]).astype(np.float32)  # noqa: E402


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

# ========= rl_state.py  END ================

_CACHE = {}


def _silu(x):
    return x / (1.0 + np.exp(-x))


class LearnedFiller:
    """Torch-free numpy filler. Replaces PBC3._select_patch's mid+fill stages:
    keeps the cheap prescore front-end, the policy (a small MLP exported to .npz)
    picks an action per proposal box and ranks by its value head, then the top-k
    boxes are exact-scored before picking. Use candidates=1, top_k=1 for the
    fastest path. `cheap` models featurize only their non-sort feature subset."""

    def __init__(self, npz, top_k=1, q_override=-1.0, candidates=1):
        self.hidden = int(npz["hidden"])
        self.Wa, self.ba = npz["Wa"], npz["ba"]
        self.Wv, self.bv = npz["Wv"], npz["bv"]
        if self.hidden > 0:
            self.W0, self.b0 = npz["W0"], npz["b0"]
            self.W2, self.b2 = npz["W2"], npz["b2"]
        self.feat_mean, self.feat_std = npz["feat_mean"], npz["feat_std"]
        self.lmin, self.lmax = float(npz["lmin"]), float(npz["lmax"])
        self.cheap = bool(npz["cheap"])
        self.feature_names = [str(s) for s in npz["feature_names"]]
        self.top_k = int(top_k); self.q_override = float(q_override); self.candidates = int(candidates)

    @classmethod
    def load(cls, path, top_k=1, q_override=-1.0, candidates=1, device=None):
        key = os.path.abspath(path)
        inst = _CACHE.get(key)
        if inst is None:
            npz = np.load(path, allow_pickle=True)
            inst = FactoredFiller(npz, q_override) if "head_sizes" in npz.files \
                else cls(npz, top_k, q_override, candidates)
            _CACHE[key] = inst
        if isinstance(inst, FactoredFiller):
            inst.set_lambda(q_override)
        else:
            inst.top_k = int(top_k); inst.q_override = float(q_override); inst.candidates = int(candidates)
        return inst

    def _forward(self, x):
        h = x
        if self.hidden > 0:
            h = _silu(h @ self.W0.T + self.b0)
            h = _silu(h @ self.W2.T + self.b2)
        return h @ self.Wa.T + self.ba, h @ self.Wv.T + self.bv[None, :]

    def _featurize(self, target, canvas, boxes, step, q, w_img, h_img, patch_count, channels):
        fn = (lambda b: extract_cheap(self.feature_names, target, canvas, b, step, q, w_img, h_img, patch_count, channels)) \
            if self.cheap else \
            (lambda b: extract_features(target, canvas, b, step, q, w_img, h_img, patch_count, channels))
        return np.stack([fn(b) for b in boxes])

    def select_patch(self, target, canvas, config, rng, channel_bits, step, current_channel):
        h_img, w_img, channels = target.shape
        boxes = propose_boxes(target, canvas, config, rng, current_channel, step)[:max(1, self.candidates)]
        if not boxes:
            return None, None
        q = self.q_override if self.q_override >= 0 else PBC3._interp(config.q_start, config.q_end, step, config.patch_count)

        feats = self._featurize(target, canvas, boxes, step, q, w_img, h_img, config.patch_count, channels).astype(np.float32)
        feats = (feats - self.feat_mean) / self.feat_std
        logits, value = self._forward(feats)
        best_action = logits.argmax(1)
        order = value[:, 0].argsort()[::-1]

        lam = lambda_for_q(q, self.lmin, self.lmax)
        best, best_score = None, -np.inf
        for bi in order[:max(1, self.top_k)]:
            box = boxes[int(bi)]
            patch, values, reduction, bits = build_action(target, canvas, box, config, channel_bits, int(best_action[bi]))
            if reduction <= 0:
                continue
            score = reduction - lam * bits
            if score > best_score:
                best_score, best = score, (patch, values)
        return best if best is not None else (None, None)


class FactoredFiller:
    """Torch-free numpy inference for the factored (multi-discrete) RL filler.

    Frozen heuristic selector (top-prescore box); the policy greedily picks
    cell_size / bitcount / mask_size per step AND a place-or-stop decision (the
    rate lever): when the stop head fires it ends the encode, so high lambda
    spends fewer patches (low bpp) and low lambda fills more (high bpp). lambda is
    the RD tradeoff weight (reward = quality - lambda*bpp at train time), from
    `config.learned_filler_lambda` if present else the quality slider
    (`learned_filler_q`, passed as `q_override`). `begin_encode` seeds the exact
    pixel count + init/header bits so the running-bpp state feature is accurate.
    """

    DEFAULT_LAMBDA = 15.0

    def __init__(self, npz, q_override=-1.0):
        self.hidden = int(npz["hidden"])
        self.W0, self.b0 = npz["W0"], npz["b0"]
        self.W2, self.b2 = npz["W2"], npz["b2"]
        self.head_sizes = [int(n) for n in npz["head_sizes"]]
        self.heads = [(npz[f"hW{i}"], npz[f"hb{i}"]) for i in range(len(self.head_sizes))]
        self.feat_mean, self.feat_std = npz["feat_mean"], npz["feat_std"]
        self.set_lambda(q_override)
        self._reset(None)

    def set_lambda(self, value):
        v = float(value)
        self.lam = v if v > 0 else self.DEFAULT_LAMBDA

    def _reset(self, pixels):
        self._pixels = pixels
        self._bits = 0.0

    def _lambda(self, config):
        v = getattr(config, "learned_filler_lambda", None)
        return float(v) if v is not None and v > 0 else self.lam

    def begin_encode(self, config, channels, channel_bits, w, h, original_w, original_h, init_patches):
        init_bits = sum(PBC3._patch_bits_for(p, channel_bits) for p in init_patches)
        init_bits += PBC3._patch_header_bits(channel_bits, config.mask_size)
        self._reset(max(1, int(original_w) * int(original_h)))
        self._bits = float(init_bits)

    def _forward(self, x):
        h = _silu(x @ self.W0.T + self.b0)
        h = _silu(h @ self.W2.T + self.b2)
        return [h @ W.T + b for (W, b) in self.heads]

    def select_patch(self, target, canvas, config, rng, channel_bits, step, current_channel):
        h_img, w_img, channels = target.shape
        if step == 1 and self._pixels is None:
            self._reset(max(1, h_img * w_img))
        pixels = self._pixels or max(1, h_img * w_img)
        boxes = propose_boxes(target, canvas, config, rng, current_channel, step)
        if not boxes:
            return None, None
        box = boxes[0]
        st = extract_state(target, canvas, box, step, w_img, h_img, config.patch_count,
                           channels, self._lambda(config), self._bits, pixels).astype(np.float32)
        st = (st - self.feat_mean) / self.feat_std
        logits = self._forward(st)
        idxs = [int(np.argmax(l)) for l in logits]
        if len(idxs) > STOP_AXIS and idxs[STOP_AXIS] == STOP_STOP:
            return None, None                               # policy chose to stop the encode
        patch, values, reduction, bits = build_action_factored(
            target, canvas, box, config, channel_bits, idxs[0], idxs[1], idxs[2])
        if reduction <= 0:
            return None, None
        self._bits += bits
        return patch, values