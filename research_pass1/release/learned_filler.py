import dataclasses
import math
import os

import numpy as np

import pbc3_ops as ops
from pbc3_features import FEATURE_NAMES, SORT_FEATURES, extract_cheap, extract_features

CELL_SIZES = [1, 2, 4, 8, 16, 32]
BITCOUNTS = [1, 2, 3, 4, 5]
ACTIONS = [(cs, bc) for cs in CELL_SIZES for bc in BITCOUNTS]
NUM_ACTIONS = len(ACTIONS)

MASK_SIZES = [2, 4, 6, 8, 16]
FACTORED_AXES = (CELL_SIZES, BITCOUNTS, MASK_SIZES)
FACTORED_HEAD_SIZES = tuple(len(a) for a in FACTORED_AXES)

STOP_PLACE, STOP_STOP = 0, 1
STOP_AXIS = len(FACTORED_HEAD_SIZES)
POLICY_HEAD_SIZES = FACTORED_HEAD_SIZES + (2,)

BOX_FEATURES = [n for n in FEATURE_NAMES if n not in SORT_FEATURES and n != "q"]
PROGRESS_FEATURES = ["lambda_log", "bpp_running", "patches_placed_frac", "patches_remaining_frac"]
STATE_NAMES = BOX_FEATURES + PROGRESS_FEATURES
STATE_DIM = len(STATE_NAMES)
_MODEL_CACHE = {}


def lambda_for_q(q: float, lmin: float = 0.05, lmax: float = 5.0) -> float:
    """## Converts a quality slider into a simple RD lambda"""
    q = float(np.clip(q, 0.0, 1.0))
    return lmin * (lmax / lmin) ** (1.0 - q)


def _score_patch(target, canvas, box, channel_bits: int, patch, values) -> tuple[float, float]:
    """## Returns exact SSE reduction and bit cost for a patch"""
    c, x, y, bw, bh = box
    canvas_box = canvas[y:y + bh, x:x + bw, c]
    target_box = target[y:y + bh, x:x + bw, c]
    delta = ops.signed_resample(values, bh, bw).astype(np.int32)
    before = target_box - np.clip(canvas_box, 0, 255)
    after = target_box - np.clip(canvas_box + delta, 0, 255)
    before_sse = float(np.sum(before.astype(np.int64) ** 2))
    reduction = before_sse - float(np.sum(after.astype(np.int64) ** 2))
    bits = float(ops.patch_bits_for(patch, channel_bits))
    return reduction, bits


def build_action(target, canvas, box, config, channel_bits: int, idx: int):
    """## Builds and scores one legacy single-index action"""
    c, x, y, bw, bh = box
    cs, bc = ACTIONS[idx]
    cell = max(1, min(cs, bw, bh))
    residual = target[y:y + bh, x:x + bw, c] - canvas[y:y + bh, x:x + bw, c]
    patch, values = ops.make_patch(c, x, y, bw, bh, cell, residual, config, bc)
    reduction, bits = _score_patch(target, canvas, box, channel_bits, patch, values)
    return patch, values, reduction, bits


def build_action_factored(target, canvas, box, config, channel_bits: int, cs_idx: int, bc_idx: int, ms_idx: int):
    """## Builds and scores one factored action: cell size, bitcount, and mask size"""
    c, x, y, bw, bh = box
    cs, bc, ms = CELL_SIZES[cs_idx], BITCOUNTS[bc_idx], MASK_SIZES[ms_idx]
    cell = max(1, min(cs, bw, bh))
    cfg = dataclasses.replace(config, mask_size=ms)
    residual = target[y:y + bh, x:x + bw, c] - canvas[y:y + bh, x:x + bw, c]
    patch, values = ops.make_patch(c, x, y, bw, bh, cell, residual, cfg, bc)
    reduction, bits = _score_patch(target, canvas, box, channel_bits, patch, values)
    return patch, values, reduction, bits


def extract_state(target, canvas, box, step: int, image_w: int, image_h: int, patch_count: int, channels: int, lam: float, bits_spent: float, pixels: int) -> np.ndarray:
    """## Returns the factored policy state for one candidate box"""
    box_feats = extract_cheap(BOX_FEATURES, target, canvas, box, step, 0.0, image_w, image_h, patch_count, channels)
    prog = np.array([
        math.log(max(float(lam), 1e-6)),
        bits_spent / max(1, pixels),
        step / max(1, patch_count),
        (patch_count - step) / max(1, patch_count),
    ], dtype=np.float32)
    return np.concatenate([box_feats, prog]).astype(np.float32)


def propose_boxes(target, canvas, config, rng, channel: int, step: int) -> list[tuple[int, int, int, int, int]]:
    """## Returns candidate boxes sorted by the learned filler's prescore front-end"""
    h_img, w_img, _ = target.shape
    search_q = ops.interp(config.search_q_start, config.search_q_end, step, config.patch_count)
    visible = np.clip(canvas[:, :, channel], 0, 255).astype(np.int32)
    abs_error = np.abs(target[:, :, channel] - visible).astype(np.int64)
    integral_abs = ops.integral(abs_error)
    anchors = ops.top_anchors(abs_error.astype(np.float32), config.top_k, config.anchor_block_size, channel)
    if not anchors:
        return []

    specs, sums, areas = [], [], []
    for i in range(max(1, int(config.search_depth))):
        c, x, y, bw, bh, _, _ = ops.sample_box(rng, anchors[i % len(anchors)], w_img, h_img, config)
        s = integral_abs[y + bh, x + bw] - integral_abs[y, x + bw] - integral_abs[y + bh, x] + integral_abs[y, x]
        if s <= 0:
            continue
        sums.append(float(s))
        areas.append(float(bw * bh))
        specs.append((c, x, y, bw, bh))
    if not specs:
        return []

    pre = search_q * ops.norm(sums) - (1.0 - search_q) * ops.norm(areas)
    keep = sorted(ops.select_top_indices(pre, config.proposal_depth), key=lambda i: pre[i], reverse=True)
    return [specs[i] for i in keep]


def _load_npz_cached(path: str) -> dict:
    """## Loads and caches one exported numpy policy file"""
    key = os.path.abspath(path)
    data = _MODEL_CACHE.get(key)
    if data is None:
        with np.load(path, allow_pickle=True) as npz:
            data = {name: npz[name] for name in npz.files}
        _MODEL_CACHE[key] = data
    return data


def _silu(x):
    """## SiLU activation used by the tiny exported MLP"""
    return x / (1.0 + np.exp(-x))


class LearnedFiller:
    """## Torch-free numpy inference for the legacy single-action learned filler"""

    def __init__(self, npz, top_k: int = 1, q_override: float = -1.0, candidates: int = 1) -> None:
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
        self.top_k = int(top_k)
        self.q_override = float(q_override)
        self.candidates = int(candidates)

    @classmethod
    def load(cls, path: str, top_k: int = 1, q_override: float = -1.0, candidates: int = 1, device=None):
        """## Loads either the legacy or factored exported filler"""
        npz = _load_npz_cached(path)
        if "head_sizes" in npz:
            return FactoredFiller(npz, q_override)
        return cls(npz, top_k, q_override, candidates)

    def _forward(self, x) -> tuple[np.ndarray, np.ndarray]:
        """## Runs the legacy action and value heads"""
        h = x
        if self.hidden > 0:
            h = _silu(h @ self.W0.T + self.b0)
            h = _silu(h @ self.W2.T + self.b2)
        return h @ self.Wa.T + self.ba, h @ self.Wv.T + self.bv[None, :]

    def _featurize(self, target, canvas, boxes, step: int, q: float, w_img: int, h_img: int, patch_count: int, channels: int) -> np.ndarray:
        """## Builds the model feature matrix for candidate boxes"""
        if self.cheap:
            fn = lambda b: extract_cheap(self.feature_names, target, canvas, b, step, q, w_img, h_img, patch_count, channels)
        else:
            fn = lambda b: extract_features(target, canvas, b, step, q, w_img, h_img, patch_count, channels)
        return np.stack([fn(b) for b in boxes])

    def select_patch(self, target, canvas, config, rng, channel_bits: int, step: int, current_channel: int):
        """## Picks the best learned legacy patch for the current channel"""
        h_img, w_img, channels = target.shape
        boxes = propose_boxes(target, canvas, config, rng, current_channel, step)[:max(1, self.candidates)]
        if not boxes:
            return None, None
        q = self.q_override if self.q_override >= 0 else ops.interp(config.q_start, config.q_end, step, config.patch_count)

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
    """## Torch-free numpy inference for the factored RL filler"""

    DEFAULT_LAMBDA = 15.0

    def __init__(self, npz, q_override: float = -1.0) -> None:
        self.hidden = int(npz["hidden"])
        self.W0, self.b0 = npz["W0"], npz["b0"]
        self.W2, self.b2 = npz["W2"], npz["b2"]
        self.head_sizes = [int(n) for n in npz["head_sizes"]]
        self.heads = [(npz[f"hW{i}"], npz[f"hb{i}"]) for i in range(len(self.head_sizes))]
        self.feat_mean, self.feat_std = npz["feat_mean"], npz["feat_std"]
        self.set_lambda(q_override)
        self._reset(None)

    def set_lambda(self, value: float) -> None:
        """## Sets the active RD lambda for this encode"""
        v = float(value)
        self.lam = v if v > 0 else self.DEFAULT_LAMBDA

    def _reset(self, pixels) -> None:
        """## Resets per-encode accounting"""
        self._pixels = pixels
        self._bits = 0.0

    def _lambda(self, config) -> float:
        """## Returns config override lambda or the model's current lambda"""
        v = getattr(config, "learned_filler_lambda", None)
        return float(v) if v is not None and v > 0 else self.lam

    def begin_encode(self, config, channels: int, channel_bits: int, w: int, h: int, original_w: int, original_h: int, init_patches) -> None:
        """## Seeds running bpp state from init patches and stream header cost"""
        init_bits = sum(ops.patch_bits_for(p, channel_bits) for p in init_patches)
        init_bits += ops.patch_header_bits(channel_bits, config.mask_size)
        self._reset(max(1, int(original_w) * int(original_h)))
        self._bits = float(init_bits)

    def _forward(self, x) -> list[np.ndarray]:
        """## Runs the factored policy heads"""
        h = _silu(x @ self.W0.T + self.b0)
        h = _silu(h @ self.W2.T + self.b2)
        return [h @ W.T + b for W, b in self.heads]

    def select_patch(self, target, canvas, config, rng, channel_bits: int, step: int, current_channel: int):
        """## Picks one factored-policy patch, or stops the encode"""
        h_img, w_img, channels = target.shape
        if step == 1 and self._pixels is None:
            self._reset(max(1, h_img * w_img))
        pixels = self._pixels or max(1, h_img * w_img)
        boxes = propose_boxes(target, canvas, config, rng, current_channel, step)
        if not boxes:
            return None, None

        box = boxes[0]
        st = extract_state(
            target, canvas, box, step, w_img, h_img, config.patch_count,
            channels, self._lambda(config), self._bits, pixels,
        ).astype(np.float32)
        st = (st - self.feat_mean) / self.feat_std
        logits = self._forward(st)
        idxs = [int(np.argmax(l)) for l in logits]
        if len(idxs) > STOP_AXIS and idxs[STOP_AXIS] == STOP_STOP:
            return None, None

        patch, values, reduction, bits = build_action_factored(target, canvas, box, config, channel_bits, idxs[0], idxs[1], idxs[2])
        if reduction <= 0:
            return None, None
        self._bits += bits
        return patch, values