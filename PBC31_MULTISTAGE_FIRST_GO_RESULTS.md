# PBC 3.1 Multistage First-Go Results

## Repository state

- Branch: `multistage_pbc31`
- Starting commit: `7faa6365186d201f44b842f100645b60273ffd4c`
- Reference: isolated `old_pbc/` PBC 3.0 subprocess
- Production defaults and `server.py`: unchanged
- Model: unchanged `patch_policy.npz`
- Main dataset: all 20 images in `hpt_data`
- Diagnostic subset: `flower.jpg`, `monet.jpg`, `googlewiki.png`, `cat.jpg`, `panorama.png`

## Implementation

`pbc31_multistage.py` is an isolated experimental codec subclass. It provides:

- arbitrary monotonic stage plans;
- independently prepared stage targets;
- one persistent reconstructed canvas;
- one stage-zero downsample initialization;
- the existing search, learned/heuristic filler, patch construction, and LZMA paths;
- fixed total iterative patch count;
- global patch-step semantics;
- coarse `None`/STOP interpreted as stage advance;
- a version-1 stage table with unchanged patch payloads;
- v0 decoding compatibility;
- deterministic decoder resize transitions;
- per-stage timing, accepted-patch, bit-cost, and stop diagnostics.

One-stage v1 reconstructs exactly the same pixels as the untouched v0 path. Its bytes differ because the experimental version and stage header differ.

## Correctness

`python -m unittest -v test_pbc31_multistage.py`

Five tests pass:

- v0 compatibility and one-stage pixel identity;
- two- and three-stage exact round trips;
- deterministic output;
- odd-size RGBA without LZMA;
- partial-decode transition behavior.

All 240 full-main benchmark rows passed exact encoder/decoder image equality.

## Phase 0

The existing warmup path showed rate/time signal but no joint improvement. Aggregate diagnostic-subset changes versus actual PBC 3.0:

| Variant | MSE | bpp | Encode | Decode |
|---|---:|---:|---:|---:|
| Current single-stage | +0.95% | -0.16% | -26.47% | -25.68% |
| W1: 0.5x, 10% early | +6.86% | -12.31% | -25.10% | -27.65% |
| W2: 0.5x, 25% early | +24.38% | -28.55% | -37.21% | -39.13% |
| W3: 0.5x, 50% early | +52.01% | -64.46% | -55.42% | -52.73% |

More aggressive variants were worse. Several settings stopped before the legacy resize, motivating the generic `STOP_STAGE` path.

## Generic-stage diagnostics

The generic path fixed premature coarse termination, but MSE still worsened monotonically as more budget moved coarse. The best initial plan, `0.5x -> 1.0x` with `10/90`, changed the diagnostic aggregate versus PBC 3.0 by:

- MSE: +6.64%
- bpp: -11.01%
- encode: -20.74%
- decode: -27.65%

A less aggressive 0.75x stage preserved more quality but removed the bitrate benefit. A `0.5x -> 1.0x` plan with only `5/95` was selected for full evaluation.

## Full 20-image result

Mean absolute results:

| Source | MSE | bpp | Encode s | Decode s | Raw bytes | Final bytes |
|---|---:|---:|---:|---:|---:|---:|
| PBC 3.0 `old_pbc` | 193.838 | 0.273556 | 0.5564 | 0.1017 | 102,267 | 41,939 |
| Current v0 single-stage | 193.463 | 0.279276 | 0.4262 | 0.0791 | 100,127 | 41,842 |
| Multistage `0.5x`, `5/95` | 202.181 | 0.263780 | 0.4518 | 0.0793 | 94,828 | 40,875 |

Candidate changes:

| Comparison | MSE | bpp | Encode | Decode | Raw bytes | Final bytes |
|---|---:|---:|---:|---:|---:|---:|
| vs actual PBC 3.0 | +4.30% | -3.57% | -18.80% | -21.97% | -7.27% | -2.54% |
| vs current single-stage | +4.51% | -5.55% | +6.01% | +0.29% | -5.29% | -2.31% |

The speed improvement versus PBC 3.0 is not caused by multistage. The current single-stage codec is already substantially faster; multistage regresses encode time by 6.01% against that attributable control.

Per-preset changes versus actual PBC 3.0:

| Preset | MSE | bpp | Encode | Decode |
|---|---:|---:|---:|---:|
| Compression | +0.29% | -10.98% | -17.32% | -17.11% |
| Balanced | +9.29% | -7.53% | -19.63% | -18.49% |
| Quality | +6.45% | -1.26% | -14.82% | -21.86% |
| High quality | +2.95% | -2.48% | -22.85% | -25.82% |

Win counts versus PBC 3.0 across 80 image-preset pairs:

| Metric | Wins | Losses |
|---|---:|---:|
| MSE | 15 | 65 |
| bpp | 53 | 27 |
| Encode time | 68 | 12 |
| Decode time | 78 | 2 |

Worst MSE outliers included:

- `eiffel.png`, high quality: +71.3% MSE;
- `monkey.png`, compression: +62.0% MSE;
- `monkey.png`, balanced: +56.1% MSE;
- `ratedistortion.png`, quality: +51.2% MSE.

## Stage diagnostics

Across the 80 candidate encodes:

- stage 0 accepted all 200 allocated patches;
- stage 1 accepted 2,347 of 3,200 allocated patches;
- stage 1 stopped through filler `None` 44 times;
- stage 0 search/fill time totaled 0.62 seconds;
- stage 1 search/fill time totaled 15.03 seconds;
- transition resizes totaled 0.35 seconds.

The coarse stage is cheap, but its initialization and patches alter the greedy trajectory enough to lose final detail. Later patches do not recover that loss within the fixed budget.

## Heuristic-filler ablation

On the diagnostic subset, heuristic multistage versus heuristic single-stage changed:

- MSE: +7.92%
- bpp: -23.48%
- encode: -6.55%
- decode: -9.24%

The failure is therefore not mainly learned-filler distribution shift.

## Interpretation

Reject the tested fixed coarse-to-fine architecture as a PBC 3.1 candidate. It is a rate-quality trade-off and does not meet the release bar.

The evidence does not support adding more stages, fixed coarse budget, retraining, or RL yet. If continuing this direction, the most defensible next experiment is not another fixed schedule: use the coarse level only to seed parent regions for full-resolution search, while keeping initialization and all serialized patches at the normal final working resolution. That could retain search acceleration without permanently coding a smoothed coarse reconstruction.
