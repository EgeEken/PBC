# Pass 23 working-target feasibility

Ordinary downsampling is suboptimal for the tested decoder-equivalent reconstruction, but this does not imply a completed-codec improvement. No image encoding preceded this gate. The control remains pbc31_feature_histogram, source 1c9c4016fdb213d7d16663542d268c7c3df4ed9b; research starts at 0ca503759f6fcc507489dd999a5fe4c848ae2eb7.

Method: preserve production RGB-to-YCbCr conversion before PIL bicubic downsampling at the 250k cap. Start with that integer target. For each fixed update, reconstruct with production _canvas_to_image (clip, uint8 cast, actual YCbCr-to-RGB), then production bicubic RGB resize with reducing_gap=None. Compute signed RGB source-minus-prediction errors, resize each signed float32 error plane in PIL mode F, and multiply the three low-resolution error planes by a fixed linear RGB-to-YCbCr difference matrix with no offsets. Add the scaled correction, bound total deviation from the ordinary target, round and clip to integer 0..255. This approximate projection is not an established adjoint or exact gradient. The real quantized/color-converted decoder-equivalent output is scored, not a floating-point reconstruction surrogate.

Fixed before results: conservative step 0.5, one iteration, total per-component deviation cap 8; stronger step 1.0, two iterations, cap 16. No other strengths, sizes or quantizers. Non-downsampled inputs are identity. Alpha/warmup configurations are explicitly outside the research implementation.

| Image | Working size | Ordinary RGB MSE | Conservative MSE / change | Stronger MSE / change |
| --- | ---: | ---: | ---: | ---: |
| googlewiki.png | 501x499 | 684.3108 | 650.4107 / -4.95% | 631.5506 / -7.71% |
| mario64.png | 577x433 | 89.1794 | 82.2845 / -7.73% | 77.3573 / -13.26% |
| cat.jpg | 612x408 | 280.5748 | 272.0455 / -3.04% | 263.8167 / -5.97% |
| flower.jpg | 612x408 | 22.1207 | 19.6304 / -11.26% | 17.8314 / -19.39% |

Gate: at least two images improve by 5% and none worsens by more than 1%, using one fixed setting. Both settings pass; the predeclared preference selects conservative (Mario64 and Flower exceed 5%, Googlewiki is just under 5%). Only conservative proceeds to encoding. Stronger is retained as a target-only result, not retried as an RD rescue.

| Image | Setting | Mean / max target change | Target variation change | Edge MSE | Nonedge MSE | Precondition ms | Traced peak MB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| googlewiki.png | ordinary | 0.000 / 0 | +0.00% | 3099.423 | 415.928 | 0.69 | 3.00 |
| googlewiki.png | conservative | 0.833 / 8 | +13.17% | 3051.475 | 383.589 | 80.25 | 36.68 |
| googlewiki.png | stronger | 2.316 / 16 | +37.19% | 3071.116 | 360.450 | 151.98 | 42.68 |
| mario64.png | ordinary | 0.000 / 0 | +0.00% | 769.895 | 13.544 | 0.78 | 3.00 |
| mario64.png | conservative | 0.337 / 8 | +5.19% | 710.695 | 12.461 | 90.32 | 40.00 |
| mario64.png | stronger | 1.147 / 16 | +19.47% | 662.636 | 12.326 | 171.57 | 46.00 |
| cat.jpg | ordinary | 0.000 / 0 | +0.00% | 885.927 | 213.313 | 0.88 | 3.00 |
| cat.jpg | conservative | 0.394 / 8 | +12.76% | 856.425 | 207.114 | 504.49 | 196.55 |
| cat.jpg | stronger | 1.328 / 16 | +42.93% | 817.080 | 202.343 | 992.35 | 205.53 |
| flower.jpg | ordinary | 0.000 / 0 | +0.00% | 171.419 | 5.531 | 0.71 | 3.00 |
| flower.jpg | conservative | 0.206 / 8 | +3.38% | 157.015 | 4.364 | 436.86 | 196.55 |
| flower.jpg | stronger | 0.704 / 16 | +13.31% | 143.935 | 3.819 | 818.29 | 205.53 |

Variation is the sum of mean absolute horizontal/vertical differences of the integer working components, a complexity proxy, not entropy or final rate. Edge pixels are the source luminance Sobel magnitude at/above its 90th percentile; exact edge fractions, source-region 4x4 MSE maps and target/output hashes are in feasibility.json. The source masks/envelopes are fixed for all settings.

Clipping/overshoot diagnostics: outside-range counts refer to proposed bounded working components before color-range clipping, not invalid stored target values. All final targets are bounded uint8-compatible integers. Local-envelope overshoot compares final RGB components on edge pixels to the corresponding original RGB 3x3 neighborhood range; it is an artifact proxy, not a visual ringing verdict.

| Image | Setting | Outside-range working components by iteration | Deviation cap hits by iteration | Edge envelope mean / max overshoot | Edge overshoot component fraction | Saturated output components |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| googlewiki.png | ordinary | 0 | 0 | 0.0014 / 10 | 0.055% | 51.503% |
| googlewiki.png | conservative | 36815 | 23681 | 0.0040 / 18 | 0.104% | 50.582% |
| googlewiki.png | stronger | 41829/51210 | 23681/39609 | 0.0240 / 25 | 0.468% | 46.683% |
| mario64.png | ordinary | 0 | 0 | 0.0759 / 34 | 2.025% | 31.620% |
| mario64.png | conservative | 2830 | 1763 | 0.0819 / 32 | 2.082% | 32.460% |
| mario64.png | stronger | 5545/5926 | 1763/6317 | 0.2177 / 40 | 5.153% | 39.844% |
| cat.jpg | ordinary | 0 | 0 | 0.2074 / 64 | 3.177% | 0.225% |
| cat.jpg | conservative | 542 | 14 | 0.1850 / 62 | 2.854% | 0.241% |
| cat.jpg | stronger | 668/747 | 14/624 | 0.1668 / 60 | 2.592% | 0.243% |
| flower.jpg | ordinary | 0 | 0 | 0.6535 / 92 | 9.009% | 1.388% |
| flower.jpg | conservative | 207 | 156 | 0.5305 / 90 | 6.681% | 1.286% |
| flower.jpg | stronger | 384/498 | 156/827 | 0.4679 / 92 | 5.696% | 1.345% |

Conservative preconditioning increases working variation on all four images; stronger increases it further and raises the text/graphics overshoot proxy. The photo edge proxy decreases. Clipping is accounted for in every reported RGB MSE. No claim of recovered arbitrary lost high frequencies or absence of visible artifacts.

Target timings: one warmed untraced operation per setting/image, excluding source load, ordinary prepare and external diagnostics. The update timer includes decoder-equivalent prediction and all signed projections. A separate repeated call enables tracemalloc; these peaks include NumPy/Python registered allocations, not all native PIL storage or total process working memory. At Cat/Flower source size, one float32 source error plane is 34.10 MB and the full uint8 RGB source/output is 25.58 MB; the target is about 3.00 MB int32. Traced conservative temporary peaks are about 196.5 MB, stronger 205.5 MB. They are measured temporary costs, not a low-memory implementation claim. No optimization rescue was attempted.

The target-only gate establishes useful representational headroom for this bounded update rule. Its 0.44-0.50 s photographic cost is already substantial; completed-file encodes measure whether any RD gain justifies it.
