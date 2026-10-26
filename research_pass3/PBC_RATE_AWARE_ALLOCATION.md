# Rate-aware patch representation and allocation

The weight-0.4 integrated selector improves the observed/estimated rate-distortion frontier on most tested images, beyond a better local proxy score: final files are serialized, decoded and measured on all 25 images. Relative to the strongest previous measured release/current envelope, mean interpolated matched-rate MSE changes -5.16% on main (17/20 wins) and -9.26% on holdout (4/5 wins). This merits continued research, with explicit latency and sparse-curve limits. No universal default or PBC 3.1 release is accepted.

## Frozen-trajectory diagnostics

Thirty existing streams cover six predetermined main images (cat, Monet, flower, cathedral, googlewiki and stockpoint): release/current Quality and HQ at 250k, plus current 500k/35. No search is rerun for this study. patch_ledger.json records every iterative geometry, area, grid shape/cells, palette bounds/precision/mask, exact serialized raw bits, exact clipped working-space SSE change and cumulative full-canvas SSE. Summed rectangle deltas are checked against full working-canvas recomputation at sampled positions. All signed quantized grids are retained in 30 compact NPZ files, linked by key/hash from each row; original indices remain in the protected source streams.

| Control | Patches | Mean raw bytes/patch | Mean cells | Mean actual bits/index | Working SSE gain/raw bit |
| --- | --- | --- | --- | --- | --- |
| current_250000 | 174 | 3472.9461 | 6955.3563 | 2.8563 | 95.0807 |
| current_hq250 | 104 | 13328.1010 | 23098.2500 | 3.8558 | 45.2091 |
| q500_p35 | 178 | 5306.8237 | 10147.4944 | 3.5000 | 122.1344 |
| release_250000 | 198 | 2909.5764 | 5896.9596 | 2.8434 | 100.5120 |
| release_hq250 | 120 | 11840.7729 | 20441.6333 | 3.6583 | 44.0563 |

The first, one-third, two-thirds, last and largest-raw-cost patches (deduplicated) also have actual original-size RGB prefix changes and marginal complete-file bits. These use actual serialized/LZMA-compressed/decoded prefixes. Negative RGB gain can occur even when working-channel SSE decreases. Prefix bit differences include entropy context and header-count changes; raw patch bytes are not additive post-LZMA ownership. No LZMA is called inside either encoder selector.

## Fixed-state representation study

For at most five patches per stream, hold pre-patch canvas and geometry fixed. Test original, at most twice the cell size, one lower maximum palette bitcount, and an optional half-cell finer-grid diagnostic. These all use v0 patch syntax. Each alternative is actually projected, quantized, resampled and exact-scored; not estimated from unquantized residual energy. An untimed discarded action build warms relevant kernel paths before timings.

| Control | Alternative | Patches | Mean retained local gain | Mean raw-bit fraction | 98% gain / 75% bits cases | Evaluation s |
| --- | --- | --- | --- | --- | --- | --- |
| release_250000 | coarser_grid | 26 | 0.5018 | 0.3767 | 0 | 0.0008 |
| release_250000 | lower_precision | 26 | 0.7504 | 0.7387 | 0 | 0.0008 |
| release_250000 | finer_grid | 14 | 2.9433 | 3.4949 | 0 | 0.0006 |
| release_hq250 | coarser_grid | 27 | 0.5115 | 0.2817 | 0 | 0.0019 |
| release_hq250 | lower_precision | 27 | 0.7525 | 0.7111 | 0 | 0.0019 |
| release_hq250 | finer_grid | 8 | 1.5954 | 3.7785 | 0 | 0.0019 |
| current_250000 | coarser_grid | 26 | 0.4697 | 0.3560 | 0 | 0.0009 |
| current_250000 | lower_precision | 26 | 0.7537 | 0.7474 | 0 | 0.0008 |
| current_250000 | finer_grid | 14 | 6.8060 | 3.6310 | 0 | 0.0007 |
| current_hq250 | coarser_grid | 25 | 0.5332 | 0.3057 | 0 | 0.0018 |
| current_hq250 | lower_precision | 25 | 0.9060 | 0.7917 | 0 | 0.0018 |
| current_hq250 | finer_grid | 10 | 2.5709 | 3.9980 | 0 | 0.0016 |
| q500_p35 | coarser_grid | 27 | 0.5581 | 0.3349 | 0 | 0.0010 |
| q500_p35 | lower_precision | 27 | 0.8294 | 0.7377 | 0 | 0.0008 |
| q500_p35 | finer_grid | 10 | 1.3703 | 3.2838 | 0 | 0.0007 |

No cheaper alternative locally dominates the original in the sampled set, and none meets the strict >=98% gain / <=75% bits shortlist. Thus no fixed-suffix full-file shortlist was manufactured. However, 44 of 262 cheap alternatives retain >=95% local gain at <=85% raw bits, and 184 improve local gain/raw-bit ratio. This is sufficient to test a small rate tradeoff, not evidence of a global codec gain. Precision reductions are generally gentler than density reductions; coarsening often sacrifices about half the local gain. The original selected high-detail corrections are not simply redundant or free to remove.

## Selector and independent ablations

Use three representations maximum: original plus coarser grid and lower precision, deduplicated. Let G0/B0 be the ordinary representation's exact clipped SSE gain/raw bits. Select by `gain - t*(G0/B0)*raw_bits`, t in {0.1, 0.4, 0.8}. Original wins exact utility ties. Retain the ordinary policy's positive-gain stop decision in the integrated encoder; no cheaper action rescues an ordinary nonpositive prediction. All tested t<1 preserve a positive original utility. There are no new models, proposals, random draws, geometry beams or entropy evaluations in selection. The rate is an inexpensive raw-bit proxy, not a fitted final-size predictor.

B3a performs one forward reconstruction with original channel/geometry/order/count fixed; representations are fitted against the evolving replay canvas using the original action as reference. It first performs the complete ordinary encode, then parsing/replay and final serialization, so the full duplicate work is charged. Identity replay reproduces all 18 screen controls byte-for-byte. It is a diagnostic, not an efficient shipping design, and is distinct from earlier backward refitting/pruning with repeated full decoding. If a frozen-geometry reference becomes nonpositive, count/order are still retained for this ablation; its nonpositive-reference counters are not a claim of an ordinary encoder stop.

B3b hooks only the representation decision after the ordinary chosen geometry/action, evaluates the same bounded alternatives and lets later channels/residuals/proposals/model actions evolve naturally. Initialization, RNG algorithm/seed, resolution, requested budget, proposal count and model calls per step remain the matched ordinary controls. Actual emitted counts may change through the ordinary stop path. Geometry/payload changes are recorded in allocation_trajectories.json.

Both screens used six images, three existing reference configurations (Q250/HQ250/500k35), three weights and two repeats. Full-stream screening selected only 500k/35 weights 0.4 and 0.8, frozen before new holdout results. Matched fixed-geometry counterparts are retained as full ablations. HQ250 failed the stronger previous-operating-point control despite some release-default wins; Q250 mostly moved below the target Q–HQ rate range. No 1M, classifier, new action model or broad candidate search was added.

## Full main/holdout measurements

All 20 main and five holdout images, three interleaved warmed repeats, nine configurations: release/current Q/HQ, current 500k35, the two frozen integrated candidates and their two fixed-geometry counterparts. Every artifact is fully serialized and independently decoded/scored. All extra representation work, changed stopping counts and fixed replay cost are included in encode wall time.

| Set | Configuration | MSE | bpp | Raw bytes | Final bytes | Iterative patches | Encode s | Decode s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hpt_data | release_q250_none | 160.3306 | 0.2996 | 99313.7500 | 47476 | 33 | 0.5889 | 0.1133 |
| hpt_data | release_hq250_none | 143.1334 | 0.5425 | 239563.8500 | 84790.8500 | 19.1500 | 0.6410 | 0.1861 |
| hpt_data | q250_none | 160.0960 | 0.3301 | 99973.1500 | 49298.2500 | 30.2000 | 0.4770 | 0.0901 |
| hpt_data | hq250_none | 143.9347 | 0.5351 | 232539.3000 | 82941.4500 | 17.7500 | 0.5379 | 0.1354 |
| hpt_data | q500p35_none | 128.6912 | 0.4033 | 148907 | 74232.5500 | 32.4000 | 0.7605 | 0.1174 |
| hpt_data | q500p35_fixed_t0.4 | 146.3408 | 0.2651 | 98757.6500 | 49725.4000 | 32.4000 | 1.1048 | 0.1028 |
| hpt_data | q500p35_fixed_t0.8 | 174.4525 | 0.2178 | 66313.1000 | 36559.0500 | 32.4000 | 1.0881 | 0.0893 |
| hpt_data | q500p35_integrated_t0.4 | 138.4631 | 0.3211 | 117060.8500 | 56639 | 34.2500 | 0.8055 | 0.1125 |
| hpt_data | q500p35_integrated_t0.8 | 153.4165 | 0.2692 | 89977.6500 | 47083.6000 | 34.5500 | 0.7859 | 0.1021 |
| hpt_data_val | q250_none | 55.5447 | 0.9826 | 79964.6000 | 41825.8000 | 28.4000 | 0.4156 | 0.0765 |
| hpt_data_val | hq250_none | 41.5028 | 1.2248 | 189679.6000 | 71999 | 14 | 0.4449 | 0.1134 |
| hpt_data_val | q500p35_none | 43.9331 | 1.0197 | 110766.4000 | 56876.2000 | 28.8000 | 0.6077 | 0.0929 |
| hpt_data_val | q500p35_fixed_t0.4 | 55.8793 | 0.6579 | 77273.6000 | 37145.2000 | 28.8000 | 0.8877 | 0.0840 |
| hpt_data_val | q500p35_fixed_t0.8 | 128.8541 | 0.4685 | 40916.2000 | 26488.6000 | 28.8000 | 0.8769 | 0.0685 |
| hpt_data_val | q500p35_integrated_t0.4 | 59.0152 | 0.6975 | 79048 | 37077 | 29.6000 | 0.6047 | 0.0854 |
| hpt_data_val | q500p35_integrated_t0.8 | 72.9044 | 0.7059 | 57724 | 32524.4000 | 30.2000 | 0.6101 | 0.0787 |
| hpt_data_val | release_q250_none | 50.6763 | 1.0164 | 83924.6000 | 42033.2000 | 29.4000 | 0.4956 | 0.0957 |
| hpt_data_val | release_hq250_none | 40.2472 | 1.1746 | 199269.4000 | 75094.2000 | 17.6000 | 0.5351 | 0.1608 |

Paired against ordinary current 500k35 (same resolution, model input, budget and RNG):

| Set | Candidate | MSE % | bpp % | Raw bytes % | Encode % | Decode % | Strict RD image wins |
| --- | --- | --- | --- | --- | --- | --- | --- |
| hpt_data | q500p35_fixed_t0.4 | 13.7147 | -34.2774 | -33.6783 | 45.2697 | -12.4824 | 0 |
| hpt_data_val | q500p35_fixed_t0.4 | 27.1920 | -35.4823 | -30.2373 | 46.0831 | -9.5300 | 0 |
| hpt_data | q500p35_fixed_t0.8 | 35.5589 | -45.9844 | -55.4668 | 43.0849 | -23.9370 | 0 |
| hpt_data_val | q500p35_fixed_t0.8 | 193.2962 | -54.0549 | -63.0608 | 44.3131 | -26.2361 | 0 |
| hpt_data | q500p35_integrated_t0.4 | 7.5932 | -20.3871 | -21.3866 | 5.9179 | -4.1755 | 2 |
| hpt_data_val | q500p35_integrated_t0.4 | 34.3298 | -31.5992 | -28.6354 | -0.4926 | -8.0740 | 0 |
| hpt_data | q500p35_integrated_t0.8 | 19.2128 | -33.2372 | -39.5746 | 3.3426 | -13.0821 | 0 |
| hpt_data_val | q500p35_integrated_t0.8 | 65.9443 | -30.7802 | -47.8867 | 0.3969 | -15.2412 | 0 |

Integrated points versus the measured fixed-geometry family (ordinary 500k35 plus fixed weights 0.4/0.8), at overlapping rates:

| Set | Integrated setting | Overlap | Matched MSE % | Wins |
| --- | --- | --- | --- | --- |
| hpt_data | q500p35_integrated_t0.4 | 17 | -5.0566 | 14 |
| hpt_data | q500p35_integrated_t0.8 | 17 | 1.2353 | 9 |
| hpt_data_val | q500p35_integrated_t0.4 | 5 | -16.6084 | 5 |
| hpt_data_val | q500p35_integrated_t0.8 | 5 | 27.6933 | 3 |

geometry_ablation_matched.json retains these exact brackets. This separates removed replay overhead from any RD effect of allowing future geometry to evolve; neither a lower absolute MSE nor the local utility alone proves an improved curve.

Counterfactual original/selected bits/gains in experimental_stats refer to representations on that candidate's own visited states; they are not the baseline trajectory's total cost. Per-image actual final files determine the comparisons.

## Matched-rate and measured-frontier evaluation

Default release frontier comparison:

| Set | Integrated candidate | All-rate overlap | Mean MSE % | Wins | Q–HQ overlap | Q–HQ MSE % | Q–HQ wins |
| --- | --- | --- | --- | --- | --- | --- | --- |
| hpt_data | q500p35_integrated_t0.4 | 20 | -8.1173 | 17 | 15 | -11.2574 | 13 |
| hpt_data | q500p35_integrated_t0.8 | 20 | -3.4682 | 12 | 11 | -7.7061 | 9 |
| hpt_data_val | q500p35_integrated_t0.4 | 5 | -12.6898 | 4 | 2 | -10.3426 | 2 |
| hpt_data_val | q500p35_integrated_t0.8 | 5 | 0.6050 | 3 | 1 | -13.5312 | 1 |

Stronger comparison with the union of all previously measured Pass 2 release/current operating points, including available resolution/budget controls:

| Set | Candidate | Rate overlap | Mean MSE % | Median MSE % | Wins | Within 5% rate | Near-rate MSE wins | MSE overlap | Matched-MSE bpp % | Measured dominated |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hpt_data | q500p35_integrated_t0.4 | 20 | -5.1601 | -6.3272 | 17 | 7 | 6 | 19 | -8.4129 | 2 |
| hpt_data | q500p35_integrated_t0.8 | 20 | -1.8897 | -2.6882 | 13 | 4 | 3 | 20 | -1.9691 | 4 |
| hpt_data_val | q500p35_integrated_t0.4 | 5 | -9.2590 | -8.3344 | 4 | 1 | 1 | 5 | -5.0251 | 1 |
| hpt_data_val | q500p35_integrated_t0.8 | 5 | -0.0535 | -2.5713 | 3 | 0 | 0 | 5 | 3.1366 | 1 |

All matching is per image, with exact measured neighboring nondominated points, arithmetic bpp/MSE interpolation and no extrapolation. Raw brackets/weights are in allocation_matched.json. Q–HQ denotes brackets whose measured neighbors are specifically Quality/HQ defaults; broader controls may include lower-rate regions. Missing overlaps are neither wins nor losses. Matched-rate and matched-distortion populations differ, so aggregate signs must not be combined into joint dominance. Previous screening images have denser prior curves than other images; every comparison retains that coverage. Sparse interpolation is an estimate rather than an achieved extra file or continuous frontier proof.

Near-rate comparisons retain each actual nearest control, its bpp distance and measured MSE difference. A 5% rate tolerance is not exact matching, and most holdout comparisons still require interpolation. The interval between measured points can hide a nonlinear baseline curve; denser matched-rate controls remain a necessary follow-up before a release claim.

Largest matched-rate regressions against the stronger existing envelope:

| Set | Image | Candidate | MSE % | Candidate MSE | Candidate bpp |
| --- | --- | --- | --- | --- | --- |
| hpt_data_val | pigeon.JPG | q500p35_integrated_t0.8 | 25.7149 | 61.4319 | 0.0104 |
| hpt_data | mario64.png | q500p35_integrated_t0.4 | 25.2383 | 217.2607 | 0.3634 |
| hpt_data | mario64.png | q500p35_integrated_t0.8 | 23.0830 | 208.1040 | 0.3816 |
| hpt_data_val | pigeon.JPG | q500p35_integrated_t0.4 | 18.8823 | 53.6133 | 0.0123 |
| hpt_data | panorama.png | q500p35_integrated_t0.8 | 16.1617 | 120.3264 | 0.2191 |
| hpt_data_val | 15k_kills.png | q500p35_integrated_t0.8 | 14.9749 | 70.3795 | 2.1230 |
| hpt_data | monet.jpg | q500p35_integrated_t0.4 | 12.0163 | 88.3818 | 0.8239 |
| hpt_data | eiffel.png | q500p35_integrated_t0.4 | 10.2461 | 27.4426 | 1.8878 |
| hpt_data | seagull.jpg | q500p35_integrated_t0.8 | 8.4652 | 85.5425 | 0.0869 |
| hpt_data | cathedral.png | q500p35_integrated_t0.8 | 5.5947 | 423.1711 | 0.0423 |

allocation_paired.json additionally retains strict observed-point wins, actual operating-point tradeoffs and all latency changes against five fresh controls. RD plots (rd_screen, rd_hpt_data, rd_hpt_data_val PNG/SVG) are backed by allocation_points.json/.csv. Connecting lines are guides; JSON brackets are authoritative. Improvements on a subset are content-dependent. No Compression/Balanced default is changed or proposed: selection is confined to Q/HQ-derived opt-in experiments, so no universal low-bpp claim is made. Content routing is skipped; 25 images and the observed heterogeneity do not support tuning another automatic policy on this holdout.

## Interpretation

Main weight-0.4 output is 138.4631 MSE at 0.321073 bpp, 0.8055 s encode and 0.1125 s decode. Ordinary 500k/35 is 128.6912 MSE at 0.403293 bpp, 0.7605/0.1174 s: this is an operating-point quality/rate tradeoff with 5.92% additional encoding cost. Released HQ is 143.1334 MSE at 0.542457 bpp, 0.6410/0.1861 s, so the candidate has lower main mean MSE and rate but is 25.65% slower to encode. Holdout weight 0.4 has 59.0152 MSE at 0.697512 bpp, compared with ordinary 500k/35's 43.9331 at 1.019742: absolute quality is worse at much lower rate, despite the matched-rate estimated improvement. Working resolution, precision, patch counts and evolving geometry interact; excessive grids are part of the inefficiency, not free redundant information. The fixed/integrated comparison supports a real sequential effect at weight 0.4, while weight 0.8 demonstrates that local efficiency and stronger rate pressure can fail globally.

An observed proxy failure is 15k_kills.png: increasing weight from 0.4 to 0.8 reduces raw bytes 128,002 -> 109,669 but raises complete bytes 37,115 -> 46,012 and RGB MSE 62.41 -> 70.38. Actual accepted patches change 33 -> 35. Thus neither raw cost nor a larger rate penalty guarantees a smaller final file after the sequential trajectory and LZMA wrapper. This is a measured reversal; attributing it to any one symbol statistic would require another experiment.

The local proxy identifies real tradeoffs, but its success alone is not evidence of the final encoder. Search geometry, clipping, RGB conversion/resizing, later patch interference, ordinary stop decisions and LZMA jointly determine outcomes. This pass separates those through the fixed/integrated ablations and actual final-file comparisons. No strict production release is accepted and no performance optimization is combined with the allocation candidates.
