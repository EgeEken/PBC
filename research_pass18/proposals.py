import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research_pass18.diagnose import np
from scipy.ndimage import label, find_objects


def content_boxes(target, canvas, channel, config):
    size = 32
    h, w = target.shape[:2]
    hh, ww = h // size * size, w // size * size
    if min(hh, ww) == 0:
        return [], []
    residual = (target[:hh, :ww, channel] - canvas[:hh, :ww, channel]).astype(np.float32)
    tiles = residual.reshape(hh // size, size, ww // size, size).transpose(0, 2, 1, 3)
    mean = tiles.mean(axis=(2, 3), keepdims=True)
    upper = tiles > mean
    n = upper.sum(axis=(2, 3))
    sums = np.sum(tiles, axis=(2, 3))
    high = np.sum(tiles * upper, axis=(2, 3))
    low = sums - high
    m1 = high / np.maximum(n, 1)
    m0 = low / np.maximum(size * size - n, 1)
    total = np.sum(tiles * tiles, axis=(2, 3))
    within = total - high * m1 - low * m0
    variance = total - sums * sums / (size * size)
    fit = 1 - within / np.maximum(variance, 1)
    active = (fit >= .8) & (m1 - m0 >= 24) & (n >= size * size * .05) & (n <= size * size * .95)
    labels, _ = label(active)
    candidates = []
    for index, slices in enumerate(find_objects(labels), 1):
        if slices is None:
            continue
        ys, xs = slices
        bw, bh = min(config.max_patch_size, (xs.stop - xs.start) * size), min(config.max_patch_size, (ys.stop - ys.start) * size)
        x = min(w - bw, max(0, (xs.start + xs.stop) * size // 2 - bw // 2))
        y = min(h - bh, max(0, (ys.start + ys.stop) * size // 2 - bh // 2))
        selected = labels[ys, xs] == index
        score = float(np.sum((total - within)[ys, xs][selected]) / (bw * bh / 4 + 128))
        candidates.append((score, (channel, int(x), int(y), int(bw), int(bh))))
    candidates.sort(key=lambda pair: (-pair[0], pair[1]))
    return [box for score, box in candidates[:3]], [score for score, box in candidates[:3]]


def learned_action(model, target, canvas, box, config, step):
    h, w, channels = target.shape
    q = model.q_override
    features = model._featurize(target, canvas, [box], step, q, w, h, config.patch_count, channels).astype(np.float32)
    logits, _ = model._forward((features - model.feat_mean) / model.feat_std)
    return int(logits[0].argmax())
