"""State features for the RL filler.

State = the cheap (non-sort) box residual features + the rate-distortion knob and
spend/progress features. The policy is conditioned on lambda (the RD tradeoff
weight, reward = quality - lambda*bpp): it sees log(lambda) plus how many bits per
pixel it has already spent and how far through the patch budget it is, and learns
to allocate bits adaptively per image at the requested operating point.
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from patch_features import FEATURE_NAMES, SORT_FEATURES, extract_cheap  # noqa: E402

BOX_FEATURES = [n for n in FEATURE_NAMES if n not in SORT_FEATURES and n != "q"]
PROGRESS_FEATURES = ["lambda_log", "bpp_running", "patches_placed_frac", "patches_remaining_frac"]
STATE_NAMES = BOX_FEATURES + PROGRESS_FEATURES
STATE_DIM = len(STATE_NAMES)


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
    return np.concatenate([box_feats, prog]).astype(np.float32)