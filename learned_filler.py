import os
import numpy as np

from pbc3_features import extract_cheap, extract_features

_CACHE = {}
CELL_SIZES = (1, 2, 4, 8, 16, 32)
BITCOUNTS = (1, 2, 3, 4, 5)
ACTIONS = tuple((cell, bitcount) for cell in CELL_SIZES for bitcount in BITCOUNTS)


def _silu(x):
    return x / (1.0 + np.exp(-x))


def lambda_for_q(q, lmin, lmax):
    q = max(0.0, min(1.0, float(q)))
    lmin = max(float(lmin), 1e-12)
    lmax = max(float(lmax), lmin)
    return float(np.exp(np.log(lmax) * (1.0 - q) + np.log(lmin) * q))


def build_action(target, canvas, box, config, channel_bits, action_index):
    from PBC3 import PBC3

    c, x, y, bw, bh = box
    cell_size, bitcount = ACTIONS[int(action_index) % len(ACTIONS)]
    cell_size = max(1, min(int(cell_size), bw, bh))
    bitcount = int(bitcount)
    residual = target[y:y + bh, x:x + bw, c] - canvas[y:y + bh, x:x + bw, c]
    before = target[y:y + bh, x:x + bw, c] - np.clip(canvas[y:y + bh, x:x + bw, c], 0, 255)
    before_sse = float(np.sum(before.astype(np.int64) ** 2))
    patch, values = PBC3._make_patch(c, x, y, bw, bh, cell_size, residual, config, bitcount, PBC3.PALETTE_GENERATED, 0)
    delta = PBC3.signed_resample(values, bh, bw).astype(np.int32)
    after = target[y:y + bh, x:x + bw, c] - np.clip(canvas[y:y + bh, x:x + bw, c] + delta, 0, 255)
    reduction = before_sse - float(np.sum(after.astype(np.int64) ** 2))
    return patch, values, reduction, PBC3._patch_bits_for(patch, channel_bits)


class LearnedFiller:
    def __init__(self, npz, top_k=1, q_override=-1.0, candidates=1):
        if "head_sizes" in npz.files:
            raise ValueError("Factored RL filler checkpoints are a PBC3.1 training artifact and are not supported by the PBC3.0 runtime.")
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
    def load(cls, path, top_k=1, q_override=-1.0, candidates=1, device=None):
        key = os.path.abspath(path)
        inst = _CACHE.get(key)
        if inst is None:
            inst = cls(np.load(path, allow_pickle=True), top_k, q_override, candidates)
            _CACHE[key] = inst
        inst.top_k = int(top_k)
        inst.q_override = float(q_override)
        inst.candidates = int(candidates)
        return inst

    def _forward(self, x):
        h = x
        if self.hidden > 0:
            h = _silu(h @ self.W0.T + self.b0)
            h = _silu(h @ self.W2.T + self.b2)
        return h @ self.Wa.T + self.ba, h @ self.Wv.T + self.bv[None, :]

    def _featurize(self, target, canvas, boxes, step, q, w_img, h_img, patch_count, channels):
        if self.cheap:
            return np.stack([extract_cheap(self.feature_names, target, canvas, b, step, q, w_img, h_img, patch_count, channels) for b in boxes])
        return np.stack([extract_features(target, canvas, b, step, q, w_img, h_img, patch_count, channels) for b in boxes])

    def select_patch(self, target, canvas, config, rng, channel_bits, step, current_channel):
        from PBC3 import PBC3
        from pbc3_heads import SearchHead

        h_img, w_img, channels = target.shape
        boxes = SearchHead(PBC3).propose(target, canvas, config, rng, step, current_channel, {})[:max(1, self.candidates)]
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
            patch, values, reduction, bits = build_action(target, canvas, boxes[int(bi)], config, channel_bits, int(best_action[bi]))
            if reduction <= 0:
                continue
            score = reduction - lam * bits
            if score > best_score:
                best_score, best = score, (patch, values)
        return best if best is not None else (None, None)