# Single-stage quality scaling

The experiment changes only opt-in single-stage configurations under research_pass2. The codec's v0 representation, decoder, source inputs, seed, RGB benchmark convention and model stay fixed. No serialized multistage, coarse initialization, pruning, refitting, alternate search architecture or training is reintroduced.

## Fixed screen and measured points

The six images were selected before measurement in PLAN.md: cat (large photograph/fine fur), Monet (texture/art), flower (large smooth/bokeh photograph), cathedral (architecture/edges), googlewiki (text/synthetic), stockpoint (relatively low 250k round-trip loss). Both release and current root were measured at 250k/500k/1M, Quality/High Quality, two interleaved warmed repeats. Times are means of per-image medians, not HF timings.

| Source/limit | Preset | RGB MSE | bpp | Raw bytes | Final bytes | Iterative patches | Encode s | Decode s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| release_250000 | quality | 271.3716 | 0.2944 | 106029.0000 | 55632.6667 | 33.0000 | 0.6144 | 0.1164 |
| release_500000 | quality | 196.5607 | 0.5219 | 206990.1667 | 96929.0000 | 43.0000 | 1.2902 | 0.2028 |
| release_1000000 | quality | 152.4661 | 0.7019 | 280951.5000 | 149778.6667 | 50.0000 | 2.2187 | 0.2580 |
| current_250000 | quality | 273.8180 | 0.3024 | 110728.3333 | 57618.6667 | 29.0000 | 0.5034 | 0.0998 |
| current_500000 | quality | 195.1362 | 0.5219 | 195236.5000 | 97590.3333 | 40.5000 | 0.9721 | 0.1470 |
| current_1000000 | quality | 136.2047 | 0.7223 | 290942.0000 | 149362.0000 | 50.0000 | 1.7521 | 0.2051 |
| release_250000 | high_quality | 246.7386 | 0.6342 | 246828.6667 | 98997.6667 | 20.0000 | 0.6949 | 0.2068 |
| release_500000 | high_quality | 171.3300 | 0.8872 | 348647.6667 | 153783.3333 | 19.6667 | 1.0112 | 0.2618 |
| release_1000000 | high_quality | 131.9983 | 1.0671 | 417479.6667 | 211117.5000 | 20.0000 | 1.4106 | 0.3099 |
| current_250000 | high_quality | 246.0700 | 0.6323 | 241033.3333 | 98008.3333 | 17.3333 | 0.5381 | 0.1378 |
| current_500000 | high_quality | 174.1088 | 0.8846 | 364772.5000 | 152807.5000 | 20.0000 | 0.8375 | 0.1958 |
| current_1000000 | high_quality | 130.9284 | 1.1122 | 472848.3333 | 221046.0000 | 20.0000 | 1.2144 | 0.2593 |

These six-image means are descriptive operating points, not matched-rate evidence. Source dimensions, actual working dimensions, expanded configs and per-image results are retained in resolution_screen.jsonl. Image limits are caps: a small source can stop being resized before the 1M limit.

## Preset diagnosis

Source inspection confirms only three preset differences between Quality and High Quality: requested iterative patches 50 versus 20, search_q_start 0.5 versus 0.7, learned_filler_q 0.8 versus 0.95. Both have 250k default working pixels, identical automatic initialization, cell-size/action support, allowed palette limits, mask size and decoder syntax. The learned model chooses actual cell size and bitcount; the heuristic patch_palette_bitcount is not a cap on that learned action path. Raising learned q changes features/actions, not merely a quantizer knob. Early stopping occurs when the selected learned proposal has no positive exact reduction; it is not an exhaustive search saturation proof.

On the 20-image immutable-release controls, Quality to HQ changes mean RGB MSE 160.3306 -> 143.1334 (-10.73%) while bpp rises 0.2996 -> 0.5425 (about +81%). Pass 1's 250k round-trip mean is 128.3761, close to the HQ MSE but not a mathematical floor or an additive error decomposition.

Per-iterative-patch representation on the six default-resolution trajectories:

| Source | Preset | Patches | Mean cells | Mean bitcount | Mean cell size | Raw bytes/patch | Previous iterative overlap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| release | quality | 198 | 5896.9596 | 2.8434 | 2.9141 | 2909.5764 | 0.4482 |
| release | high_quality | 120 | 20441.6333 | 3.6583 | 2.3917 | 11840.7729 | 0.5354 |
| current | quality | 174 | 6955.3563 | 2.8563 | 2.8506 | 3472.9461 | 0.4024 |
| current | high_quality | 104 | 23098.2500 | 3.8558 | 2.7500 | 13328.1010 | 0.4987 |

These are patch-weighted descriptive means, not independently causal allocations. Initialization is excluded here and kept in all final file sizes. Palette masks/bounds, grid cost, area, coverage and per-prefix final bytes are in quality_patch_ledgers.json and quality_trajectories.json. Prefixes are replayed from frozen completed streams; increasing a fresh budget changes search interpolation and the model's step fraction, so a prefix is not a smaller-budget encoding policy. Marginal RGB SSE reduction is measured after actual serialization/LZMA/decode. Added final bits are the difference between complete prefix files; they can be zero/negative and are context-dependent, unlike per-patch raw bits.

| Source | Preset | Measured increments | RGB MSE increases | Nonpositive added final bits |
| --- | --- | --- | --- | --- |
| release | quality | 198 | 15 | 15 |
| release | high_quality | 120 | 9 | 1 |
| current | quality | 174 | 8 | 22 |
| current | high_quality | 104 | 7 | 1 |

Three isolated targeted hypotheses were screened: High Quality budget 50; Quality learned q 0.95; High Quality search_q_start 0.5. These are the complete allocation ablations, rather than a Cartesian parameter sweep.

| Variant | MSE | bpp | Patches | Encode s | Decode s |
| --- | --- | --- | --- | --- | --- |
| hq_budget50 | 243.5798 | 0.6740 | 25.6667 | 0.7699 | 0.1798 |
| quality_q095 | 262.6652 | 0.5016 | 47.1667 | 0.6550 | 0.1124 |
| hq_search05 | 298.9687 | 0.3225 | 20.0000 | 0.4022 | 0.0867 |

Four strategically selected rate configurations followed: 500k Quality budgets 20/35, 1M Quality budget 20, and 250k Quality learned q 0.9, with fresh root Q/HQ controls. Two repeats and all six images. Their measured points and expanded configurations are in rate_screen_plan.json and rate_screen.jsonl. No interpolation is performed outside the observed range.

## Matched-rate comparisons and confirmation

The two fixed 500k Quality budgets (20 and 35) were selected before new holdout candidate runs. They test a class-dependent lead: p35 improves both rate and quality versus release HQ on flower/googlewiki in the screen, while textured Monet and low-round-trip-loss stockpoint lose efficiency. q1000_p20 and the preset-only ablations were not promoted. These choices are documented in quality_confirmation_plan.json; they are research confirmation, not production acceptance.

Full main/holdout absolute results, three repeats, same fresh release/root Q/HQ controls:

| Set | Variant | Preset | Cases | MSE | bpp | Patches | Encode s | Decode s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hpt_data | release_250000 | quality | 20 | 160.3306 | 0.2996 | 33 | 0.6081 | 0.1162 |
| hpt_data | release_hq250 | high_quality | 20 | 143.1334 | 0.5425 | 19.1500 | 0.6540 | 0.1847 |
| hpt_data | current_250000 | quality | 20 | 160.0960 | 0.3301 | 30.2000 | 0.4880 | 0.0911 |
| hpt_data | current_hq250 | high_quality | 20 | 143.9347 | 0.5351 | 17.7500 | 0.5585 | 0.1368 |
| hpt_data | q500_p20 | quality | 20 | 148.5846 | 0.3680 | 20 | 0.6078 | 0.1086 |
| hpt_data | q500_p35 | quality | 20 | 128.6912 | 0.4033 | 32.4000 | 0.7904 | 0.1222 |
| hpt_data_val | current_250000 | quality | 5 | 55.5447 | 0.9826 | 28.4000 | 0.4390 | 0.0779 |
| hpt_data_val | current_hq250 | high_quality | 5 | 41.5028 | 1.2248 | 14 | 0.4802 | 0.1213 |
| hpt_data_val | q500_p20 | quality | 5 | 55.8609 | 0.9491 | 17.4000 | 0.4753 | 0.0856 |
| hpt_data_val | q500_p35 | quality | 5 | 43.9331 | 1.0197 | 28.8000 | 0.6610 | 0.0947 |
| hpt_data_val | release_250000 | quality | 5 | 50.6763 | 1.0164 | 29.4000 | 0.5018 | 0.0983 |
| hpt_data_val | release_hq250 | high_quality | 5 | 40.2472 | 1.1746 | 17.6000 | 0.5647 | 0.1600 |

Candidate-point interpolation against each image's nondominated release default-preset controls:

| Set | Variant | Cases | Rate overlap | Mean matched-rate MSE % | Rate wins | MSE overlap | Mean matched-MSE bpp % | Observed dominated |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hpt_data | q500_p20 | 20 | 18 | 6.8190 | 10 | 11 | 30.9185 | 5 |
| hpt_data | q500_p35 | 20 | 14 | -5.9425 | 9 | 10 | 9.8136 | 1 |
| hpt_data_val | q500_p20 | 5 | 5 | 10.4003 | 2 | 5 | 18.9963 | 1 |
| hpt_data_val | q500_p35 | 5 | 5 | -5.9085 | 4 | 4 | 5.6524 | 1 |

The mean percentage is over valid overlapping images, not over the entire dataset. A missing overlap is not a win or loss. Piecewise interpolation uses actual neighboring measured nondominated points in arithmetic bpp/MSE coordinates; it is an estimate, not another achievable single-file operating point. Sparse default controls can span wide rate intervals. Nearest actually measured rate distances, exact brackets/weights and per-image outliers are retained in rd_matched_comparisons.json. A close-rate flag means within 5%, not exact equality. No BD-rate/extrapolation or continuous dominance claim is made.

Restricted to intervals whose measured neighbors are specifically release Quality and High Quality (rather than Compression/Balanced):

| Set | Variant | Q–HQ overlapping images | Matched-rate MSE % | Wins |
| --- | --- | --- | --- | --- |
| hpt_data | q500_p20 | 16 | 3.8835 | 10 |
| hpt_data | q500_p35 | 13 | -7.2164 | 9 |
| hpt_data_val | q500_p20 | 2 | -3.4176 | 1 |
| hpt_data_val | q500_p35 | 4 | -0.5354 | 3 |

Largest matched-rate increases versus the observed release frontier:

| Set | Image | Variant | MSE % | Nearest actual rate distance % |
| --- | --- | --- | --- | --- |
| hpt_data | mario64.png | q500_p20 | 84.3090 | 12.9118 |
| hpt_data_val | pigeon.JPG | q500_p20 | 61.7988 | 4.7564 |
| hpt_data_val | pigeon.JPG | q500_p35 | 35.4652 | 13.7602 |
| hpt_data | flag.jpg | q500_p20 | 33.0972 | 12.4036 |
| hpt_data | monkey.png | q500_p20 | 27.5096 | 17.8391 |
| hpt_data_val | 15k_kills.png | q500_p20 | 25.6812 | 6.2639 |
| hpt_data | monet.jpg | q500_p20 | 22.3865 | 36.0156 |
| hpt_data | ratedistortion.png | q500_p20 | 20.2288 | 10.9349 |

Matched-rate and matched-MSE tables can use different image subsets/ranges; their aggregate signs are not interchangeable. No joint dominance is inferred by combining those two means.

Largest pointwise MSE increases versus fresh release HQ (these are operating-point comparisons, not matched-rate regressions):

| Set | Image | Variant | MSE % | bpp % | Encode % | Decode % |
| --- | --- | --- | --- | --- | --- | --- |
| hpt_data | ratedistortion.png | q500_p20 | 224.8383 | -35.6669 | -43.6580 | -66.9657 |
| hpt_data | ratedistortion.png | q500_p35 | 173.2327 | -33.7856 | -21.3820 | -61.3543 |
| hpt_data_val | pigeon.JPG | q500_p20 | 147.6047 | -73.6028 | -22.5135 | -59.6942 |
| hpt_data_val | 15k_kills.png | q500_p20 | 128.6340 | -14.9978 | -43.8462 | -56.6627 |
| hpt_data | monkey.png | q500_p20 | 116.1778 | -36.1511 | -26.9094 | -57.3216 |
| hpt_data | flag.jpg | q500_p20 | 96.0313 | -51.0634 | 10.1273 | -33.6253 |
| hpt_data_val | old_man_standing.jpg | q500_p20 | 94.7450 | -24.2895 | -45.8709 | -60.4276 |
| hpt_data_val | old_man_standing.jpg | q500_p35 | 94.6079 | -23.8219 | -37.6019 | -60.8265 |

quality_confirmation_paired.json retains every pointwise comparison, strict RD image wins and actual runtime changes. budget_trajectory_comparisons.json explicitly checks initialization and the first changed iterative patch between fresh p20/p35 encodes, rather than treating the two budgets as nested trajectories.

For the six screening images, the 500k family interpolated at release HQ rate:

| Image | Release HQ bpp | Release HQ MSE | MSE at HQ rate % | Rate at HQ MSE % |
| --- | --- | --- | --- | --- |
| cat.jpg | 0.0740 | 293.1337 | — | — |
| cathedral.png | 0.0865 | 367.2938 | -9.3576 | — |
| flower.jpg | 0.0433 | 34.8409 | -11.8038 | — |
| googlewiki.png | 2.1412 | 714.4318 | -31.6953 | — |
| monet.jpg | 1.2321 | 53.9761 | 28.3248 | 40.9955 |
| stockpoint.jpg | 0.2282 | 16.7551 | 47.1329 | — |

observed_rd_frontiers.json retains the best measured operating points across available configurations, per image. This adaptive observed envelope is not a learned selector or a fixed general-purpose preset. curves_at_control_rates.json compares both release and root Q/HQ targets with measured 500k/1M families. quality_rd_screen.png/.svg and plot_rd.py show raw observed points and connecting guides; the numerical interpolation/brackets in JSON are authoritative. Main and holdout remain separately summarized.

## Interpretation limits

The round-trip diagnostics and actual larger-resolution MSE gains establish that the 250k canvas is a material spatial-detail limitation on several large/detail-rich images. They do not establish a universal distortion bound. Extra pixels alone do not guarantee a better RD frontier: texture and already smooth/low-loss inputs can consume more grid bits and search time without a matched-rate benefit. At 1M, the fixed-size local boxes also cover a smaller fraction of the working canvas, and requested budgets can fill completely. This is consistent with coverage/representation pressure; it does not by itself isolate the model, geometry, or search as the sole cause.

High Quality's inefficiency is a measured combination of higher cost per chosen representation, fewer requested corrections, and spatial-detail limitations. The one-proposal learned stop rule and greedy trajectory remain unresolved bottlenecks. More patches or a higher learned q alone did not establish a general curve improvement in the screen. Positive 500k image-specific RD results show that the additive single-stage architecture is not proven fundamentally saturated. A distinct next architecture test would need cost-aware spatial support/action selection or an explicit high-frequency residual representation, with matched-rate controls, rather than another fixed coarse-to-fine stream schedule or more RL.

All raw points, trajectories, per-image comparisons and provenance are retained. Reproduce resolution via runner.py resolution; allocation via quality.py ablations; rate/confirmation via quality.py planned rate_screen / quality_confirmation; ledgers via quality.py trajectories; then analyze.py and plot_rd.py. These commands overwrite only their own Pass 2 output names; they never alter Pass 1.
