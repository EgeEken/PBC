# Exact search-state reuse

Two local approaches were tested without changing any proposal/RNG/patch semantics. Both retain the original NumPy integral construction; there is no arbitrary-rectangle integral update, integral-fusion kernel, native rewrite or approximate search.

## Valid state and invalidation

Constant: prepared target values, dimensions/config/model weights within a stage. Evolving: canvas, clipped visible errors, channel totals, search integrals/anchors, sampled geometry and all residual-dependent features. Search-q and step-fraction features change each step. Target/canvas owners change on a new encode or warmup resize and invalidate cached maps. A final integral cannot simply survive an accepted patch; it is always rebuilt with the original operation/order.

Approach 1 (`scores`) retains the absolute-error array already computed for each channel score and reuses it in the next search for that unchanged channel. Changed-channel scoring still fully refreshes that map. It removes the duplicate clip/subtract/abs calculation without changing reduction order.

Approach 2 (`rectangles`) stores each channel's int32 absolute-error map and exact integer total. After the real accepted grid/delta is applied to the signed-int16 canvas, recompute only its affected rectangle from the actual clipped post-update canvas and adjust total by exact old/new rectangle sums. Unchanged pixels/channels remain valid. This handles hidden overshoot, clipping and even signed-int16 wrap consistently with existing behavior. It adds two rectangle reductions and bookkeeping, so it is not free. Integrals/anchors are still computed normally, from exactly equal full maps. No target mean/std or percentile result is cached for arbitrary new boxes.

Before timing, both approaches matched all 48 six-image/four-preset checks. This includes 1178 complete proposal-list and RNG-state comparisons, full error maps, exact channel totals, complete streams, pixels and patch trajectories. Synthetic signed-int16 clipping/wrap tests validate 100 rectangle mutations. Additional RGB/RGBA, both selected-delta and apply-grid paths, warmup/new owners and repeated encodes in one context are in search_compatibility.json (16 cases, two repetitions each). Validation mode's full recomputations and duplicate searches are excluded from speed tests.

## Timing and bookkeeping

The initial screen excluded one-time experimental scope installation and is retained separately where applicable. The primary confirmation charges cache initialization and any per-call prototype installation/cleanup as documented in its plan. All map updates, rectangle sums, array allocations, ordinary integral reconstruction and other codec work are inside full compress latency. Only one strategy is promoted. The source patching mechanism is experimental; a future production implementation would own per-encode state explicitly rather than mutate global functions in an HTTP process.

Screen:

| Variant | Cases | Encode baseline s | Encode candidate s | Encode % | Encode wins |
| --- | --- | --- | --- | --- | --- |
| rectangles | 24 | 0.4606 | 0.4217 | -8.4470 | 22 |
| scores | 24 | 0.4606 | 0.4424 | -3.9334 | 20 |

Frozen promoted-strategy reconfirmation after charging prototype installation:

| Variant | Cases | Encode baseline s | Encode candidate s | Encode % | Encode wins |
| --- | --- | --- | --- | --- | --- |
| rectangles | 24 | 0.4510 | 0.4176 | -7.3930 | 23 |

Mean per-call prototype state/namespace installation in the full run is 0.001100 s, included in the encode walls below along with cleanup. Runtime mode selection itself remains benchmark setup, as for other paired modes.

Full fixed-candidate main/holdout confirmation:

| Set | Variant | Cases | MSE | bpp | Encode baseline s | Encode s | Encode % | Decode % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hpt_data | rectangles | 80 | 193.4631 | 0.2793 | 0.4798 | 0.4472 | -6.7971 | -0.5558 |
| hpt_data_val | rectangles | 20 | 91.1562 | 0.7189 | 0.3788 | 0.3561 | -5.9865 | 0.0772 |

The decoder is untouched by reuse; small decoder timing differences are variation. Every candidate stream/pixel/count matches the exact same current implementation with reuse disabled. Per-image wins and repeat variability are retained in search_full_paired.json and search_full_summary.json. No release-RNG change or speed optimization is combined with this experiment.

Traced target phases, separately from uninstrumented primary timing:

| Variant | Phase | Inclusive s | Exclusive s | Count |
| --- | --- | --- | --- | --- |
| baseline | action.search.integral_error | 0.1052 | 0.1052 | 39.3333 |
| baseline | action.search.visible_error | 0.0240 | 0.0240 | 39.3333 |
| baseline | encode.patch.apply | 0.0010 | 0.0010 | 38.3333 |
| baseline | encode.patch.score | 0.0298 | 0.0298 | 38.3333 |
| scores | action.search.integral_error | 0.1045 | 0.1045 | 39.3333 |
| scores | action.search.visible_error | 0.0001 | 0.0001 | 39.3333 |
| scores | encode.patch.apply | 0.0008 | 0.0008 | 38.3333 |
| scores | encode.patch.score | 0.0284 | 0.0284 | 38.3333 |
| rectangles | action.search.integral_error | 0.1025 | 0.1025 | 39.3333 |
| rectangles | action.search.visible_error | 0.0001 | 0.0001 | 39.3333 |
| rectangles | encode.patch.apply | 0.0040 | 0.0040 | 38.3333 |
| rectangles | encode.patch.score | 0.0001 | 0.0001 | 38.3333 |

Read apply+score+visible-error together: bookkeeping shifts work into patch application. Claiming only the reduction in score time would hide that cost. Integral time is retained; unchanged integral inputs do not imply a cached integral. Experimental counters report refreshed maps, updated pixels, full pixels avoided and verification counts. RGB maps occupy roughly 3 × working_pixels × 4 bytes (about 3 MB at 250k, 6 MB at 500k) plus totals/objects; they persist for an encode. Views/copy/transient allocation and dtype assumptions are explicit in search_reuse.py.

## Decision and transfer limits

Rectangle reuse preserves all 100 default image/preset streams and decoded pixels on all three full repeats. Full encode latency, including per-call prototype state/installation/cleanup, changes -6.80% on main (72/80 case wins) and -5.99% on holdout (17/20). The average prototype installation is about 1.10 ms, included. This is a meaningful local runtime-only improvement; decoder differences are noise. It remains unvalidated on HF and requires explicit per-encode state ownership before concurrent production integration.

Integer error-map equality and unchanged ordering are compatible with a single-thread CPU runtime, but local timing does not establish HF benefit. Memory bandwidth, rectangle area distribution, NumPy/Numba versions, CPU dispatch, service contention and installation design still need a future authorized same-worker HF comparison. Existing cross-runtime encoder ties are not fixed by caching; the required gate is equality to the same-runtime baseline. No cache code, production default or deployment was changed.
