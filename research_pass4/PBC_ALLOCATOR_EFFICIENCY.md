# Allocator efficiency

No substantial allocator cost reduction was established. On the six-image uninstrumented screen, ordinary/frozen/composite encode means of case medians are 0.76551/0.82826/0.82083 s. Frozen allocation costs8.20% more than ordinary; the exact composite saves only 0.90% of allocator time (five/six median wins), below the predetermined2% promotion gate and comparable to repeat variation. Projection alone is +1.04%, before-SSE alone -0.41%. No full-dataset cache timing or optional fast-runtime allocation combination was run.

## Attribution and corrected profiling

All C comparisons use the same generic runtime flags for ordinary/frozen/optimized methods. Ordinary/frozen requested config is Quality 500k/35, unchanged model/RNG/proposal semantics and internal MSE enabled. Additional representations are a selector cost; additional accepted/attempted steps are a different, measured consequence of the evolving trajectory. A reduction in encode time is never counted as an RD improvement.

An initial harness prefix test used startswith('all'), which also matched 'alloc'. This accidentally applied grid/search to the first allocator timing/control profile. It did not affect files or RD, because the runtime operations are exact, and it did not affect A's four mode inputs. Those timing rows are excluded and retained under invalid_runtime_label, with the cause/fix in routing_correction.json. matched_confirmation and efficiency_profile were fully rerun after replacing the condition with exact all or all_ matching. The final gates reject generic search counters in ordinary-runtime allocation rows. No old invalid comparison is presented as a cost improvement.

Corrected profile: six main images, ordinary and frozen allocator, two warmed instrumented repetitions. Function cumulative time nests other work and must not be summed; self time and calls are retained. cProfile wall times contain instrumentation overhead and are not primary latency claims. There are no cold compilation calls in the corrected profile. The production legacy selector does not add a model inference for each representation, but its changed trajectory can run more model steps overall.

| Method | Function | Calls/encode | Self s | Cumulative s |
| --- | --- | --- | --- | --- |
| ordinary | mask_from_values | 39.3333 | 0.0029 | 0.0042 |
| ordinary | _score_patch | 30.3333 | 0.0034 | 0.0088 |
| ordinary | palette_generator | 39.3333 | 0.0010 | 0.0037 |
| ordinary | make_patch | 39.3333 | 0.0022 | 0.0427 |
| ordinary | quantize_signed | 39.3333 | 0.0048 | 0.0050 |
| ordinary | project_residual_to_grid | 39.3333 | 0.0001 | 0.0257 |
| ordinary | palette_bounds | 39.3333 | 0.0003 | 0.0003 |
| ordinary | patch_bits_for | 39.3333 | 0.0001 | 0.0002 |
| ordinary | search_boxes | 30.3333 | 0.0574 | 0.3565 |
| ordinary | extract_features | 30.3333 | 0.0075 | 0.0362 |
| ordinary | _build_action | 30.3333 | 0.0021 | 0.0270 |
| allocator | mask_from_values | 113.8333 | 0.0068 | 0.0095 |
| allocator | _score_patch | 104.8333 | 0.0097 | 0.0291 |
| allocator | palette_generator | 113.8333 | 0.0023 | 0.0089 |
| allocator | make_patch | 113.8333 | 0.0047 | 0.0737 |
| allocator | quantize_signed | 113.8333 | 0.0092 | 0.0097 |
| allocator | project_residual_to_grid | 113.8333 | 0.0002 | 0.0358 |
| allocator | palette_bounds | 113.8333 | 0.0005 | 0.0006 |
| allocator | patch_bits_for | 113.8333 | 0.0002 | 0.0004 |
| allocator | search_boxes | 35 | 0.0684 | 0.4222 |
| allocator | extract_features | 35 | 0.0090 | 0.0423 |
| allocator | _build_action | 104.8333 | 0.0055 | 0.0802 |

The screened trajectory changes search/model calls from 30.33 to 35 per encode; representation builds/exact scores from 30.33 to 104.83, and total patch grid/palette constructions including init from 39.33 to 113.83. Native resampling, quantization, masks/palettes and repeated before-SSE allocations contribute to the extra action work. Raw patch-bit calculation is small (roughly 0.0004 s/encode in the instrumented allocator); LZMA is not added to selection. Extra selection time is separately retained in experimental_stats, and cannot explain the whole ordinary-versus-allocator wall difference by itself.

## Two exact mechanisms

1. Projection reuse: within one unchanged box/canvas/action, cache project_residual_to_grid by cell/mode. The original and lower-precision alternatives need the same unquantized projected grid; coarser cells get a different entry. Generated palettes/quantization, delta reconstruction, exact gain, raw bit cost, alternative order and the frozen utility selector remain ordinary. No values or prepared state survive a patch mutation.
2. Before-SSE reuse: compute the identical clipped working-space pre-patch SSE once per action, reduced in the same int64 order and converted to the same float. Each representation still reconstructs its full delta, clips the after-canvas exactly as before and calculates its own after-SSE/raw bits. No gain approximation or candidate rejection is introduced. Signed canvas/delta overflow semantics are unchanged.

Fresh caches wrap the original action build and the frozen Pass 3 select call, then restore functions before applying any patch. Projection arrays are read-only inputs to subsequent rint/copy/quantization operations. Per-action state/closures/dictionaries/namespace creation, reductions, allocations and restoration are included in compress. Only one-time experimental function installation/code preparation is mode setup outside the compress timer, as in the other paired variants. The global hooks are private-process research mechanisms, not a concurrent-production design. A future implementation would pass owned action state explicitly.

No extra model/feature call, proposal, random draw, representation, rate weight, LZMA evaluation or stream syntax was added. Only these two methods and their composition were screened. No third optimization was pursued. Decision basis, exactness requirements and the pre-timing promotion gate are recorded in cost_decision.json.

## Exactness and uninstrumented timing

Twenty-four six-image correctness cases (four allocator variants) first matched the frozen Pass 3 files/decoded pixels/patch signatures/MSE/bpp and selection counters. Primary screening is six images/five methods/three interleaved warmed repetitions, 90 rows. Claimed exact variants must preserve all original/selected raw-bit and local-gain totals, changed-representation counts, stop counts and model-action counts, not merely final aggregate quality.

| Experiment | Set | Candidate | Baseline | Cases | Encode base s | Encode s | Encode % | Encode wins | Byte-equal | Pixel-equal |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| efficiency_screen | hpt_data | allocator | ordinary | 6 | 0.7655 | 0.8283 | 8.1972 | 2 | 0 | 0 |
| efficiency_screen | hpt_data | projection | allocator | 6 | 0.8283 | 0.8369 | 1.0403 | 2 | 6 | 6 |
| efficiency_screen | hpt_data | before_sse | allocator | 6 | 0.8283 | 0.8249 | -0.4054 | 3 | 6 | 6 |
| efficiency_screen | hpt_data | optimized | allocator | 6 | 0.8283 | 0.8208 | -0.8977 | 5 | 6 | 6 |

Full operating points and repeat variability where measured:

| Experiment | Set | Method | MSE | bpp | Patches | Encode s | Encode CV | Decode s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| efficiency_screen | hpt_data | ordinary | 205.1380 | 0.4799 | 29.6667 | 0.7655 | 0.0384 | 0.1269 |
| efficiency_screen | hpt_data | allocator | 215.3806 | 0.3825 | 35 | 0.8283 | 0.0158 | 0.1160 |
| efficiency_screen | hpt_data | projection | 215.3806 | 0.3825 | 35 | 0.8369 | 0.0375 | 0.1172 |
| efficiency_screen | hpt_data | before_sse | 215.3806 | 0.3825 | 35 | 0.8249 | 0.0492 | 0.1167 |
| efficiency_screen | hpt_data | optimized | 215.3806 | 0.3825 | 35 | 0.8208 | 0.0226 | 0.1188 |

The composite saves about7.44 ms from the frozen allocator's828.26 ms mean; its net overhead above ordinary remains 55.31 ms (7.23%), versus 62.75 ms (8.20%) originally on this six-image set. It nominally removes about11.85% of the incremental difference but only 0.90% of the complete encoder wall, so this is not a substantial full-codec result. Average within-case encode CV is 1.58% frozen and 2.26% composite (projection3.75%, before-SSE4.92%). Projection avoids35 repeated projections and before-SSE70 repeated reductions on cat, but function savings are too small relative to the remaining search/model/resampling/quantization work and bookkeeping. All tested caches preserve exact files, patch order, policy counts, raw-bit totals and gain totals. The first unequal-path profile/timing runs were corrected and excluded, not used to support these values.

The primary overhead is the paired same-stage difference allocator_seconds-ordinary_seconds, with its relative percentage. It includes real changed trajectories, not just local microbenchmarks. Optimized-versus-frozen isolates the exact cache benefit because all files and choices remain identical. A negative/near-zero net difference on a small image is not evidence that the extra representations are free. Per-image wins/losses/CV/extrema, cache-hit counts and original extra_selection_seconds remain in raw rows and paired summaries. Micro/function savings alone are not a substantial whole-codec improvement.

Combined fast-runtime allocator cost was not measured: the cache composite failed its predeclared substantial-saving gate, so the optional broad composition was stopped. The independently confirmed fast runtime remains valid, but its percentages are not added to allocation timings or used to infer an unmeasured release/HF cost. A already measures the combined runtime without allocation; C's only allocator cost result is the same-underlying-runtime six-image screen.

Projection reuse holds at most the original/coarser projected arrays for the current action; it adds short-lived memory rather than a whole-image cache. Before-SSE reuse retains a scalar. It avoids repeated full-box clipping/subtraction/int64 square/reduction temporaries on cache hits, while after-SSE still allocates normally. Existing full-image search maps and NumPy bit-grid temporaries belong to generic runtime work and are not attributed to allocator cost reduction. No whole-process peak-RSS saving is claimed from avoided operation counts.

## RD and release implications

Stop this cost branch. The methods are exact operation-reuse demonstrations, but not ready for production on their measured benefit/complexity. No third optimization, broad cache confirmation, new selector, proxy or weight was tried. Retaining files exactly retains RD wins and failures; it does not repair the allocation gate or establish an overall faster release.

Exact output preservation retains both the allocator's measured wins and its content regressions. Runtime work cannot fix mario64/pigeon dominance, eiffel's rate/quality loss or limited high-quality holdout coverage. No new fixed weight, training, expanded action search, model or representation was tried. Production defaults, stream version, decoder and public UI remain unchanged.
