# Pass 24 final-patch redundancy

Stop at the completed-stream omission feasibility gate. Three strict RD improvements exist, but their final-file savings are only 14, 74 and 97 bytes. No sampled omission reaches 5% savings with at most 1% RGB MSE growth. This does not justify a pruning algorithm, cheap working-SSE predictor or integrated encoder experiment. Only this feasibility report and the summary are created; no postencode-pruning report.

Control: unchanged pbc31_feature_histogram at 1c9c4016fdb213d7d16663542d268c7c3df4ed9b; research start 5d1b3a1f0db036dfef0dd9d51c6898802f7d48d8. Four existing ordinary images, Quality/HQ, default 250k working cap, seed 2003/release RNG/model/configs preserved. Saved Pass17 streams are checked by exact stream/pixel SHA256 and original-size RGB MSE. Existing parser/serializer reproduces every original byte on all 8 files. No full image encode ran in this pass.

Sampling: keep all 3 initialization patches. Select 10 iterative patches per file from deterministic first/second, quartile/median/three-quarter, last-two positions plus largest raw-cost and most-overlapped candidates, deduplicated/fill in ascending order. This yields 80 tests from 233 iterative patches. All sampled indices and every original patch geometry/channel/grid size/effective bit depth/raw serialized cost/later overlap are retained in omissions.json. Positions are zero-based original serialized indices; first iterative index is 3. These are middle-of-completed-stream omissions, not prefix truncations or fresh lower-budget encodes.

Every test removes exactly one iterative patch, updates the header count through the existing serializer, uses production LZMA/framing and independently invokes the ordinary decoder. True original-resolution RGB MSE includes signed int16 canvas reconstruction, clipping, YCbCr conversion and final PIL bicubic resizing. No inference from historical local patch gain or raw-bit ownership. All survivors retain their original order/values.

| Image | Preset | Sampled / iterative | RGB-MSE-reducing omissions | Strict RD improvements |
| --- | --- | ---: | ---: | ---: |
| googlewiki.png | quality | 10 / 50 | 1 | 0 |
| googlewiki.png | high_quality | 10 / 20 | 1 | 1 |
| mario64.png | quality | 10 / 50 | 0 | 0 |
| mario64.png | high_quality | 10 / 20 | 0 | 0 |
| cat.jpg | quality | 10 / 18 | 3 | 0 |
| cat.jpg | high_quality | 10 / 20 | 0 | 0 |
| flower.jpg | quality | 10 / 35 | 1 | 1 |
| flower.jpg | high_quality | 10 / 20 | 1 | 1 |

RGB MSE decreases in 7/80 sampled omissions (8.75%); only 3/80 (3.75%) also reduce final bytes. No exact MSE ties. The other 4 MSE-reducing omissions grow the file. Categories: 3 strict, 28 low-loss rate-saving trades, 3 negligible effects, 46 outside the low-loss screen. Outside-screen is a screening label, not a theorem that every larger distortion trade is dominated.

Exact strict improvements:

| Image / preset | Omitted index | Original -> omitted bytes | Bytes saved | Original -> omitted RGB MSE | Rate change | MSE change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| googlewiki.png / high_quality | 10 | 176,905 -> 176,808 | 97 | 714.431799 -> 714.422625 | -0.0548% | -0.0013% |
| flower.jpg / quality | 36 | 28,979 -> 28,965 | 14 | 43.434844 -> 43.431060 | -0.0483% | -0.0087% |
| flower.jpg / high_quality | 13 | 46,094 -> 46,020 | 74 | 34.840878 -> 34.828414 | -0.1605% | -0.0358% |

These are valid strict pointwise wins across Googlewiki/Flower, but 0.048-0.161% final-file savings are too small for the requested practical direction. The frozen magnitude check requires at least 3 strict removals saving 0.5% each across 2 images, or 5% savings within 1% MSE growth across 2 images. Neither passes. Counts alone would conceal the tiny benefits.

Best actual byte-saving omission within 1% MSE growth in each stream, with the original completed file as control:

| Image / preset | Index | Final rate change | RGB MSE change |
| --- | ---: | ---: | ---: |
| googlewiki.png / quality | 15 | -3.2132% | +0.6878% |
| googlewiki.png / high_quality | 13 | -1.1758% | +0.2614% |
| mario64.png / quality | 27 | -0.1111% | +0.5854% |
| mario64.png / high_quality | 21 | -0.3029% | +0.3407% |
| cat.jpg / quality | 16 | -0.4692% | +0.2953% |
| cat.jpg / high_quality | 7 | -0.4069% | +0.0197% |
| flower.jpg / quality | 20 | -0.2692% | +0.3854% |
| flower.jpg / high_quality | 13 | -0.1605% | -0.0358% |

The largest low-loss saving is Googlewiki Quality 3.213% at 0.688% more MSE. Every image other than Googlewiki stays below 0.5% saving in this tolerance. A near-threshold Cat HQ omission (index 8) saves 10.492% but increases MSE 1.172%; it is explicitly retained, not silently dropped at the 1% boundary. No combination of these independent points is used to manufacture a joint gain.

Patch position, overlap and precision:

| Group | Sample count | MSE decreases | Strict RD | Low-loss byte savings | Median MSE change | Median final rate change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| early | 47 | 1 | 0 | 3 | +15.1617% | -9.5943% |
| middle | 9 | 3 | 2 | 7 | +0.0423% | -0.1605% |
| late | 24 | 3 | 1 | 21 | +0.0345% | -0.0798% |
| overlap_at_least_half | 21 | 3 | 1 | 4 | +22.0606% | -10.6797% |
| overlap_below_half | 59 | 4 | 2 | 27 | +1.2802% | -1.1457% |
| effective_bits_1to3 | 22 | 7 | 3 | 18 | +0.0158% | -0.0776% |
| effective_bits_4plus | 58 | 0 | 0 | 13 | +8.6405% | -4.7868% |
| quality | 40 | 5 | 1 | 15 | +2.3792% | -1.3348% |
| high_quality | 40 | 2 | 2 | 16 | +3.1835% | -2.1775% |

Early/middle/late divide accepted iterative positions into thirds. Later-patch omission generally causes less damage and saves fewer bits; late patches are not more substantial redundant opportunities. All 3 strict omissions have effective depth 2 or 3. None of 58 sampled depth 4+ patches improves MSE when removed. Earlier costly/high-depth corrections remain important despite later additive overlap.

Overlap is the fraction of the patch rectangle covered by the union of later same-component patch support. It is additive overlap, not literal overwrite. At overlap at least 50%, median omission MSE damage is 22.06%, versus 1.28% below 50%. This pooled association is confounded by patch size/position/content and does not show that overlap causes usefulness. Strict wins occur at overlaps 62.02%, 0% and 1.83%, so overlap alone does not identify them.

Descriptive pooled Spearman correlations with signed MSE increase: position fraction -0.827, later overlap +0.467, effective depth +0.740, raw patch bits +0.831. No predictor was fitted or validated. These deliberately selected structural samples do not estimate the population frequency of redundant patches; group medians do not describe a deployable selection policy.

Two-stage rate behavior matters. Googlewiki Quality index 6 removes 689 raw patch bits and reduces MSE 0.0130%, yet its recompressed file grows 43 bytes. Cat has three tiny MSE-reducing omissions that also grow the file. Raw patch cost is not marginal post-LZMA byte ownership. All 80 results use actual completed-file bytes/bpp, and all signed byte/MSE deltas are preserved.

Ordinary RD controls: reuse the verified saved Q/HQ, HQ-p10, Quality-p10 and prior targeted controls, including Googlewiki HQ 500k, from the unchanged Pass23 control manifest. No new controls or larger-resolution candidates are encoded. Exact parent Q/HQ controls were independently revalidated in this pass. Selected strict/best-low-loss/near-threshold comparisons follow. Near-rate means within 5%, with signed gaps; no equal-rate claim.

| Image / preset / index | Best actual control within 5% | Signed rate gap | MSE change vs actual control | Ordinary interpolated MSE change |
| --- | --- | ---: | ---: | ---: |
| googlewiki.png / quality / 15 | quality | -3.2132% | +0.6878% | -0.8376% |
| googlewiki.png / high_quality / 10 | high_quality | -0.0548% | -0.0013% | +32.7893% |
| googlewiki.png / high_quality / 13 | high_quality | -1.1758% | +0.2614% | +31.8864% |
| mario64.png / quality / 27 | quality | -0.1111% | +0.5854% | +0.0465% |
| mario64.png / high_quality / 21 | high_quality | -0.3029% | +0.3407% | +0.2192% |
| cat.jpg / quality / 16 | quality | -0.4692% | +0.2953% | +0.1767% |
| cat.jpg / high_quality / 7 | high_quality | -0.4069% | +0.0197% | -0.0228% |
| cat.jpg / high_quality / 8 | high_quality_p10 | +0.1361% | -0.6190% | +0.0746% |
| flower.jpg / quality / 36 | quality | -0.0483% | -0.0087% | -0.0896% |
| flower.jpg / quality / 20 | quality | -0.2692% | +0.3854% | -0.0651% |
| flower.jpg / high_quality / 13 | high_quality | -0.1605% | -0.0358% | -0.1423% |

Interpolation is the minimum MSE across valid measured bracketing chords, never extrapolated; endpoints/weights remain in analysis.json. It is a conservative sparse estimate, not an achievable equal-rate file. The broad Googlewiki quality-to500k bracket produces a much stronger estimate than default HQ and cannot be presented as actual dominance. The three strict omissions improve their actual same-preset points only slightly; this does not establish meaningful broad-frontier improvement. The Googlewiki Quality rate trade gives an estimated 0.84% matched-rate MSE gain, still small and isolated.

The near-threshold Cat HQ index 8 file is 70,620 bytes/MSE 296.569203, compared with saved ordinary HQ-p10 at 70,524 bytes/MSE 298.416534: +0.1361% rate and -0.6190% MSE. The full measured ordinary bracket instead estimates a slight 0.0746% MSE regression. It is a small near-rate trade rather than evidence of a new large pruning opportunity.

Offline evaluation cost, one warmed call per omission on i9-12900H logical P-core 12, numerical threads 1, Python 3.11.16/NumPy 2.4.6/Numba 0.68.0/Pillow 11.3.0. Actual serializer/decoder warmed on each original file first. Timers include serialization/LZMA, complete decoder and external true-source scoring; exclude file I/O, hashes and structural selection. Medians of distinct omissions are descriptive costs, not repeated performance estimates for one candidate. Component medians need not sum to the median full evaluation.

| Image / preset | Median repack ms | Median decode ms | Median score ms | Median complete evaluation ms | Ten-test sum s |
| --- | ---: | ---: | ---: | ---: | ---: |
| googlewiki.png / quality | 96.38 | 44.77 | 8.98 | 150.37 | 1.455 |
| googlewiki.png / high_quality | 259.59 | 53.59 | 8.84 | 322.22 | 3.090 |
| mario64.png / quality | 74.05 | 47.27 | 13.54 | 136.43 | 1.319 |
| mario64.png / high_quality | 155.15 | 50.55 | 13.44 | 218.89 | 2.097 |
| cat.jpg / quality | 47.34 | 146.31 | 113.91 | 308.66 | 3.101 |
| cat.jpg / high_quality | 88.51 | 151.24 | 114.94 | 354.47 | 3.502 |
| flower.jpg / quality | 18.69 | 144.00 | 113.48 | 276.35 | 2.753 |
| flower.jpg / high_quality | 53.45 | 152.21 | 114.92 | 320.43 | 3.171 |

The 80 exact evaluations consume 20.487 s in these measured spans, before baseline roundtrip/warmup/selection/I/O. These are expensive offline diagnostics, not a shipping encoder mechanism. Complete encode-plus-pruning overhead and decode speedup are unmeasured because no integrated/pruned multi-removal candidate was justified. No rejected candidate receives repeated timing.

Disposition: no C/D/E. Combined deletions, signed-canvas subtraction proxies, prediction cost, integrated encode overhead and additional-image generalization remain untested. It would be incorrect to add individual MSE/byte deltas to predict a joint result. There is insufficient useful signal to justify those conditional steps, or to spend more implementation effort classifying 14-97-byte wins. Existing geometric/bit-depth measurements do not establish a cheap practical predictor.

Validation: 8 exact original parse/serialize/pixel/MSE checks; 80 actual omitted files independently decoded/scored; 80 saved-file header/count/initialization/survivor-order/hash checks; three strict files rescored; six fresh model-and-encoder-forbidden decoder processes (practical and immutable release v0 for each strict file). No new representation/decoder/model or online allocator, no full image encoding.

The negative result is bounded: these completed streams contain very little exploitable sampled single-iterative-patch redundancy under preserved initialization. It does not prove global rate-allocation optimality, absence of unsampled/joint redundancy or a fundamental representation ceiling. Preserve the tiny strict wins and low-loss/near-threshold trades as measured evidence, and stop this postencode-removal direction.
