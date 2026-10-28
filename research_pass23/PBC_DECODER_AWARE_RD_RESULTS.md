# Pass 23 decoder-aware completed-file results

The conservative target adjustment fails the frozen progression gate. It yields one same-preset strict RD win (Cat Quality), several HQ quality/rate trades, and two Quality points dominated by their unchanged controls. No Monet/Eiffel extension, three-repeat timing campaign, classifier, dual encode or parameter rescue followed.

Implementation: research_pass23.encoder.DecoderAwarePBC overrides only prepare. It calls the original preparation, adjusts the target once, then updates arr(uint8) and target(int32) together. Initialization means, downsample initialization, learned features, patch search/channel scores and gain calculations therefore read the adjusted data. Original RGB orig_compare, dimensions, warmup metadata, model, release RNG, patch syntax, LZMA and decoder remain unchanged. Internal MSE is enabled and independently matches the true source RGB score. There are no global production monkeypatches.

One deterministic completed encode/decode per image/preset/variant after tiny Q/HQ warmups; alternating pair order. Intel i9-12900H logical P-core 12, one numerical thread, Python 3.11.16/NumPy 2.4.6/Numba 0.68.0/Pillow 11.3.0. Encode includes preparation, preconditioning, patches, entropy and internal original-source MSE. Full ordinary v0 decode is timed independently. Input loading, extra diagnostic prepare, hashing, file I/O, parsing and external scoring are outside timers. Sixteen measured full-image encodes, no 100-case matrix. Timings are one-pass feasibility observations, not repeated speed claims or HF timings.

| Image | Preset | Baseline bytes | Candidate bytes | Baseline RGB MSE | Candidate RGB MSE | Encode change |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| googlewiki.png | quality | 82,440 | 92,666 | 781.0294 | 788.3215 | +7.09% |
| googlewiki.png | high_quality | 176,905 | 182,805 | 714.4318 | 670.9446 | +13.51% |
| mario64.png | quality | 64,809 | 71,219 | 135.4717 | 137.9232 | +15.65% |
| mario64.png | high_quality | 89,460 | 90,779 | 122.0256 | 113.9095 | +9.59% |
| cat.jpg | quality | 63,297 | 57,391 | 299.1920 | 291.9883 | +54.15% |
| cat.jpg | high_quality | 78,898 | 93,838 | 293.1337 | 280.7868 | +58.91% |
| flower.jpg | quality | 28,979 | 40,581 | 43.4348 | 38.7946 | +58.15% |
| flower.jpg | high_quality | 46,094 | 46,997 | 34.8409 | 32.6110 | +53.78% |

| Image / preset | Final bpp ordinary -> candidate | Rate change | RGB MSE change | Encode s ordinary -> candidate | Decode ms ordinary -> candidate |
| --- | ---: | ---: | ---: | ---: | ---: |
| googlewiki.png / quality | 0.99781 -> 1.12158 | +12.40% | +0.93% | 0.5535 -> 0.5928 | 39.82 -> 37.36 |
| googlewiki.png / high_quality | 2.14116 -> 2.21257 | +3.34% | -6.09% | 0.6300 -> 0.7151 | 57.51 -> 62.15 |
| mario64.png / quality | 0.52091 -> 0.57243 | +9.89% | +1.81% | 0.5622 -> 0.6502 | 48.18 -> 48.74 |
| mario64.png / high_quality | 0.71904 -> 0.72964 | +1.47% | -6.65% | 0.5243 -> 0.5745 | 53.69 -> 50.88 |
| cat.jpg / quality | 0.05940 -> 0.05386 | -9.33% | -2.41% | 0.6480 -> 0.9989 | 120.93 -> 117.68 |
| cat.jpg / high_quality | 0.07404 -> 0.08806 | +18.94% | -4.21% | 0.7216 -> 1.1467 | 120.71 -> 127.22 |
| flower.jpg / quality | 0.02719 -> 0.03808 | +40.04% | -10.68% | 0.6947 -> 1.0987 | 116.15 -> 118.57 |
| flower.jpg / high_quality | 0.04325 -> 0.04410 | +1.96% | -6.40% | 0.7093 -> 1.0908 | 125.17 -> 128.06 |

Bpp is complete post-LZMA bytes divided by original pixel count. Raw patch savings do not imply final file savings: Mario64 Quality raw bytes fall 172,902 ->160,486 while final bytes rise 64,809 ->71,219. Both raw and wrapped bytes are retained below and in screen.json.

| Image / preset | Raw bytes ordinary -> candidate | Total patches ordinary -> candidate | Working target MSE ordinary -> candidate | Ordinary-target MSE of candidate canvas |
| --- | ---: | ---: | ---: | ---: |
| googlewiki.png / quality | 196,754 -> 204,360 | 53 -> 53 | 31.522 -> 51.241 | 44.319 |
| googlewiki.png / high_quality | 484,447 -> 489,924 | 23 -> 23 | 9.311 -> 7.619 | 10.155 |
| mario64.png / quality | 172,902 -> 160,486 | 53 -> 53 | 19.837 -> 25.300 | 23.324 |
| mario64.png / high_quality | 284,041 -> 281,006 | 23 -> 23 | 15.032 -> 16.431 | 15.715 |
| cat.jpg / quality | 134,104 -> 121,877 | 21 -> 18 | 6.950 -> 7.364 | 7.776 |
| cat.jpg / high_quality | 205,668 -> 281,841 | 23 -> 13 | 4.602 -> 3.259 | 3.833 |
| flower.jpg / quality | 51,855 -> 100,118 | 38 -> 32 | 8.559 -> 8.016 | 8.072 |
| flower.jpg / high_quality | 120,118 -> 121,444 | 23 -> 14 | 5.365 -> 5.386 | 5.436 |

Working MSE is mean squared YCbCr component error against the clipped decoded working canvas. Candidate working-target MSE uses its adjusted target; it is not original-resolution RGB MSE and the target references differ. The last column instead uses the same ordinary target. Unclipped canvas error, initialization/iterative counts and grid-cell counts by bit depth are retained in screen.json. Total patch counts include 3 initialization patches.

Actual effective palette-bit distribution, all serialized patches including initialization (bitcount:patch count):

| Image / preset | Ordinary | Candidate |
| --- | --- | --- |
| googlewiki.png / quality | 2:40, 3:3, 4:6, 5:3, 6:1 | 2:34, 3:11, 4:3, 5:3, 6:2 |
| googlewiki.png / high_quality | 3:7, 4:4, 5:11, 6:1 | 2:1, 3:2, 4:4, 5:14, 6:2 |
| mario64.png / quality | 2:2, 3:9, 4:23, 5:16, 6:3 | 2:1, 3:8, 4:20, 5:21, 6:3 |
| mario64.png / high_quality | 4:3, 5:17, 6:3 | 4:3, 5:17, 6:3 |
| cat.jpg / quality | 1:4, 2:3, 3:4, 4:4, 5:4, 6:2 | 1:1, 2:5, 3:2, 4:5, 5:3, 6:2 |
| cat.jpg / high_quality | 2:7, 3:1, 4:7, 5:6, 6:2 | 2:2, 3:1, 5:8, 6:2 |
| flower.jpg / quality | 1:1, 2:11, 3:3, 4:17, 5:5, 6:1 | 1:1, 2:7, 3:6, 4:10, 5:7, 6:1 |
| flower.jpg / high_quality | 2:6, 4:10, 5:6, 6:1 | 2:2, 4:5, 5:6, 6:1 |

Stronger ordinary comparisons: all saved Q/HQ, HQ p10, Quality p10 bases, existing targeted budgets, and Googlewiki HQ500k were independently decoded, hash/MSE checked, and reused without new control encodes. The candidate always keeps 250k. Current ordinary screen streams exactly reproduce Pass17 hashes. The verified controls and provenance are in comparisons.json.

| Image / preset | Best actual ordinary within 5% bytes | Signed rate gap | Candidate MSE change |
| --- | --- | ---: | ---: |
| googlewiki.png / quality | none | - | - |
| googlewiki.png / high_quality | high_quality | +3.34% | -6.09% |
| mario64.png / quality | high_quality_p10 | -3.81% | -26.55% |
| mario64.png / high_quality | high_quality | +1.47% | -6.65% |
| cat.jpg / quality | none | - | - |
| cat.jpg / high_quality | none | - | - |
| flower.jpg / quality | high_quality_p10 | -1.81% | +1.02% |
| flower.jpg / high_quality | high_quality | +1.96% | -6.40% |

Nearby does not mean equal rate. Mario64 Quality looks favorable against inefficient HQ p10 but is actually dominated by ordinary default Quality (fewer bytes, lower MSE); Googlewiki Quality is also dominated by its default Quality control. Flower Quality saves 1.81% rate versus HQ p10 but worsens MSE 1.02%. The three HQ candidates improve MSE roughly 6.1-6.7% versus default HQ at 1.5-3.3% extra rate; that is useful narrow evidence, not exact matched-rate proof.

Conservative matched-rate estimates use the lowest interpolated ordinary MSE among all valid measured bracketing chords, never extrapolate. These are estimates across sparse operating points, not achievable equal-rate files. Googlewiki includes the stronger 500k point; its broad bracket is especially uncertain.

| Image / preset | Ordinary bracket | Estimated ordinary MSE | Candidate MSE | Estimated MSE change |
| --- | --- | ---: | ---: | ---: |
| googlewiki.png / quality | quality -> high_quality_500k | 754.6953 | 788.3215 | +4.46% |
| googlewiki.png / high_quality | quality -> high_quality_500k | 522.5685 | 670.9446 | +28.39% |
| mario64.png / quality | quality -> high_quality | 131.9753 | 137.9232 | +4.51% |
| mario64.png / high_quality | no measured bracket | - | 113.9095 | - |
| cat.jpg / quality | quality_p10_base -> quality | 306.2343 | 291.9883 | -4.65% |
| cat.jpg / high_quality | no measured bracket | - | 280.7868 | - |
| flower.jpg / quality | quality -> high_quality | 37.6091 | 38.7946 | +3.15% |
| flower.jpg / high_quality | no measured bracket | - | 32.6110 | - |

Only Cat Quality has a favorable bracket estimate (-4.65%). Googlewiki HQ improves the same default HQ point, yet does not beat the conservative quality-to-500k ordinary chord (+28.39% estimated MSE); neither an actual domination claim nor a reliable continuous frontier follows from that interpolation. Mario64/Cat/Flower HQ rates exceed the measured ordinary maxima, so no equal-rate estimate is invented.

Screen gate: 3 useful points across 3 images (at least 3% MSE reduction at no more than 5% extra rate), but only 1 same-preset strict RD win, and mean per-point final-rate increase 9.84%. Required at least 2 strict wins and mean rate increase at most 5%; gate fails. Maximum MSE worsening is 1.81%, so catastrophic quality regression is not the failure reason. Substantial rate and encoder costs prevent general adoption.

Interpretation: ideal preconditioning helps all four images, but the adjusted working images have higher spatial variation and changed patch trajectories. Googlewiki Quality fitting error rises 31.52 ->51.24 against its own target, and 31.52 ->44.32 against the original target; Mario64 Quality likewise fits worse. Flower Quality uses fewer patches yet 40.0% more final bytes. Cat Quality is a real isolated RD success, while Cat HQ spends 18.9% more final bytes for 4.2% MSE improvement. These observations support complexity/representation and trajectory limitations; this pass does not separately identify model-policy mismatch versus quantization/search as the causal factor.

The 0.44-0.50 s conservative photographic preprocessing appears in complete encode overhead of roughly 54-59%, exceeding the 10% preference. Graphics overhead is about 7-16% in one-pass observations. Actual decode changes span roughly -6.2% to +8.1%, reflecting different payload/patch paths plus timing noise. There is no added decoder operation, but exact equal decode times are not claimed.

Validation: all 16 streams pass unchanged ordinary v0 decoding, file/pixel hashes, repeat decoding, original RGB dimensions/MSE and enabled internal MSE agreement. Twelve deterministic preparation/reconstruction fixtures cover RGB/YCbCr, identity/downsampled,1x1 and odd dimensions, consistent arr/target and bounded corrections. Alpha/warmup unsupported configurations reject. Four fresh subprocesses decode candidate text/photo files with policy loading forbidden, through the practical and immutable PBC3.0 release decoders. No production source, public default, model, decoder or enhancement container changes.
