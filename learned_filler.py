import dataclasses
import math
import os

import numpy as np

import pbc3_ops as ops
from pbc3_features import FEATURE_NAMES, SORT_FEATURES, extract_cheap, extract_features
from pbc3_trace import timed

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


def _score_patch(target, canvas, box, channel_bits: int, patch, values, trace=None) -> tuple[float, float, np.ndarray]:
    """## Returns exact SSE reduction and bit cost for a patch"""
    c, x, y, bw, bh = box
    canvas_box = canvas[y:y + bh, x:x + bw, c]
    target_box = target[y:y + bh, x:x + bw, c]
    with timed(trace, "action.filler.score.resample"):
        delta = ops.signed_resample(values, bh, bw, trace=trace, purpose="candidate_reconstruction")
    with timed(trace, "action.filler.score.reduction"):
        before = target_box - np.clip(canvas_box, 0, 255)
        after = target_box - np.clip(canvas_box + delta, 0, 255)
        before_sse = float(np.sum(before.astype(np.int64) ** 2))
        reduction = before_sse - float(np.sum(after.astype(np.int64) ** 2))
    with timed(trace, "action.filler.score.bit_cost"):
        bits = float(ops.patch_bits_for(patch, channel_bits))
    if trace is not None:
        trace.count("candidates_exact_scored")
    return reduction, bits, delta


def _build_action(target, canvas, box, config, channel_bits, cell_size, bitcount, trace=None):
    c, x, y, bw, bh = box
    cell = max(1, min(cell_size, bw, bh))
    with timed(trace, "action.filler.build.residual"):
        residual = target[y:y + bh, x:x + bw, c] - canvas[y:y + bh, x:x + bw, c]
    with timed(trace, "action.filler.build.make_patch", cell_size=cell, bitcount=bitcount):
        patch, values = ops.make_patch(c, x, y, bw, bh, cell, residual, config, bitcount, trace=trace)
    with timed(trace, "action.filler.build.score_patch"):
        reduction, bits, delta = _score_patch(target, canvas, box, channel_bits, patch, values, trace=trace)
    return patch, values, delta, reduction, bits


def build_action(target, canvas, box, config, channel_bits: int, idx: int, trace=None):
    """## Builds and scores one legacy single-index action"""
    cell_size, bitcount = ACTIONS[idx]
    return _build_action(target, canvas, box, config, channel_bits, cell_size, bitcount, trace=trace)


def build_action_factored(target, canvas, box, config, channel_bits: int, cs_idx: int, bc_idx: int, ms_idx: int, trace=None):
    """## Builds and scores one factored action: cell size, bitcount, and mask size"""
    cfg = dataclasses.replace(config, mask_size=MASK_SIZES[ms_idx])
    return _build_action(target, canvas, box, cfg, channel_bits, CELL_SIZES[cs_idx], BITCOUNTS[bc_idx], trace=trace)


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


def propose_boxes(target, canvas, config, rng, channel: int, step: int, trace=None) -> list[tuple[int, int, int, int, int]]:
    """## Returns candidate boxes sorted by the learned filler's prescore front-end"""
    search_q = ops.interp(config.search_q_start, config.search_q_end, step, config.patch_count)
    return ops.search_boxes(target, canvas, config, rng, channel, config.search_depth, search_q, ordered=True, trace=trace)


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


def _mlp(x, weights):
    """## Runs the exported SiLU trunk"""
    for W, b in weights:
        x = _silu(x @ W.T + b)
    return x


class LearnedFiller:
    """## Torch-free numpy inference for the legacy single-action learned filler"""

    def __init__(self, npz, top_k: int = 1, q_override: float = -1.0, candidates: int = 1) -> None:
        self.hidden = int(npz["hidden"])
        self.Wa, self.ba = npz["Wa"], npz["ba"]
        self.Wv, self.bv = npz["Wv"], npz["bv"]
        if self.hidden > 0:
            if "layer_count" in npz:
                self.layers = [(npz[f"W{i}"], npz[f"b{i}"]) for i in range(int(npz["layer_count"]))]
            else:
                self.layers = [(npz["W0"], npz["b0"]), (npz["W2"], npz["b2"])]
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

    def _forward(self, x, trace=None) -> tuple[np.ndarray, np.ndarray]:
        """## Runs the legacy action and value heads"""
        with timed(trace, "action.filler.policy.trunk_forward"):
            h = _mlp(x, self.layers) if self.hidden > 0 else x
        with timed(trace, "action.filler.policy.action_head_forward"):
            action_logits = h @ self.Wa.T + self.ba
        with timed(trace, "action.filler.policy.value_head_forward"):
            value = h @ self.Wv.T + self.bv[None, :]
        return action_logits, value

    def _featurize(self, target, canvas, boxes, step: int, q: float, w_img: int, h_img: int, patch_count: int, channels: int) -> np.ndarray:
        """## Builds the model feature matrix for candidate boxes"""
        if self.cheap:
            fn = lambda b: extract_cheap(self.feature_names, target, canvas, b, step, q, w_img, h_img, patch_count, channels)
        else:
            fn = lambda b: extract_features(target, canvas, b, step, q, w_img, h_img, patch_count, channels)
        return np.stack([fn(b) for b in boxes])

    def select_patch(self, target, canvas, config, rng, channel_bits: int, step: int, current_channel: int, trace=None):
        """## Picks the best learned legacy patch for the current channel"""
        h_img, w_img, channels = target.shape
        with timed(trace, "action.filler.policy.propose_boxes"):
            boxes = propose_boxes(target, canvas, config, rng, current_channel, step, trace=trace)[:max(1, self.candidates)]
        if not boxes:
            return None, None, None
        q = self.q_override if self.q_override >= 0 else ops.interp(config.q_start, config.q_end, step, config.patch_count)

        with timed(trace, "action.filler.policy.feature_extract"):
            feats = self._featurize(target, canvas, boxes, step, q, w_img, h_img, config.patch_count, channels).astype(np.float32)
        with timed(trace, "action.filler.policy.feature_normalize"):
            feats = (feats - self.feat_mean) / self.feat_std
        with timed(trace, "action.filler.policy.forward"):
            logits, value = self._forward(feats, trace=trace)
        with timed(trace, "action.filler.policy.select_action_head"):
            best_action = logits.argmax(1)
        with timed(trace, "action.filler.policy.select_value_order"):
            order = value[:, 0].argsort()[::-1]

        lam = lambda_for_q(q, self.lmin, self.lmax)
        best, best_score = None, -np.inf
        for bi in order[:max(1, self.top_k)]:
            box = boxes[int(bi)]
            action_index = int(best_action[bi])
            with timed(trace, "action.filler.policy.evaluate_action", candidate=int(bi), action_index=action_index, cell_size=ACTIONS[action_index][0], bitcount=ACTIONS[action_index][1]):
                patch, values, delta, reduction, bits = build_action(target, canvas, box, config, channel_bits, action_index, trace=trace)
            if reduction <= 0:
                continue
            with timed(trace, "action.filler.policy.rd_score", candidate=int(bi)):
                score = reduction - lam * bits
            if score > best_score:
                best_score, best = score, (patch, values, delta)
        return best if best is not None else (None, None, None)


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

    def _forward(self, x, trace=None) -> list[np.ndarray]:
        """## Runs the factored policy heads"""
        with timed(trace, "action.filler.policy.trunk_forward"):
            h = _mlp(x, [(self.W0, self.b0), (self.W2, self.b2)])
        names = ["cell_size", "bitcount", "mask_size", "stop"]
        outputs = []
        for name, (W, b) in zip(names, self.heads):
            with timed(trace, f"action.filler.policy.{name}_head_forward"):
                outputs.append(h @ W.T + b)
        return outputs

    def select_patch(self, target, canvas, config, rng, channel_bits: int, step: int, current_channel: int, trace=None):
        """## Picks one factored-policy patch, or stops the encode"""
        h_img, w_img, channels = target.shape
        if step == 1 and self._pixels is None:
            self._reset(max(1, h_img * w_img))
        pixels = self._pixels or max(1, h_img * w_img)
        with timed(trace, "action.filler.policy.propose_boxes"):
            boxes = propose_boxes(target, canvas, config, rng, current_channel, step, trace=trace)
        if not boxes:
            return None, None, None

        box = boxes[0]
        with timed(trace, "action.filler.policy.state_extract"):
            st = extract_state(
                target, canvas, box, step, w_img, h_img, config.patch_count,
                channels, self._lambda(config), self._bits, pixels,
            ).astype(np.float32)
        with timed(trace, "action.filler.policy.state_normalize"):
            st = (st - self.feat_mean) / self.feat_std
        with timed(trace, "action.filler.policy.forward"):
            logits = self._forward(st, trace=trace)
        head_names = ["cell_size", "bitcount", "mask_size", "stop"]
        idxs = []
        for head_index, (head_name, values) in enumerate(zip(head_names, logits)):
            with timed(trace, f"action.filler.policy.head.{head_name}", head_index=head_index):
                idxs.append(int(np.argmax(values)))
        with timed(trace, "action.filler.policy.stop_check"):
            should_stop = len(idxs) > STOP_AXIS and idxs[STOP_AXIS] == STOP_STOP
        if should_stop:
            return None, None, None

        with timed(trace, "action.filler.policy.evaluate_factored_action", cell_size=CELL_SIZES[idxs[0]], bitcount=BITCOUNTS[idxs[1]], mask_size=MASK_SIZES[idxs[2]]):
            patch, values, delta, reduction, bits = build_action_factored(target, canvas, box, config, channel_bits, idxs[0], idxs[1], idxs[2], trace=trace)
        if reduction <= 0:
            return None, None, None
        self._bits += bits
        return patch, values, delta
