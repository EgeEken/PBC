# Pass 18 deterministic spatial proposals

The tested contrast/binary-fit proposal generator did not identify a useful replacement for the ordinary chosen patch. It is rejected at frozen feasibility; this is a negative result for this one method, not proof that the stochastic search is optimal.

## Frozen method

Same 47 states and current channel as the low-bit analysis. The ordinary comparison is the actual ranked stochastic pool: search_depth 200, proposal_depth 50, followed by one model action/exact evaluation on the first retained box. RNG replay exactly matches the original ordered pool. Only retained 50-box equivalence is assessed; there is no claim about all unretained rectangles or all possible actions.

The deterministic method tiles the working signed residual into complete 32x32 tiles. For each tile, split at its mean and calculate group means and the fraction of centered variance explained by that split. Require explained fraction >=0.8, group-mean contrast >=24 and each group >=5% of pixels. Thresholds and geometry were fixed for the first run; no threshold/cell-count rescue sweep followed. The analyzed target enters through target-minus-current-canvas, rather than classifying source RGB colors as residual codes.

Four-connected eligible tiles produce component bounding boxes, capped at the existing 400-pixel maximum. Rank by explained uncentered residual SSE divided by estimated cell-2 grid cells plus metadata; retain at most three boxes. No neural training, exhaustive rectangles, source-name switch, second encode or new production field. Existing SearchState residuals differ from this signed residual map, so this prototype charges its own map/statistics; it does not claim reuse of a binary map already computed by production. Partial edge tiles under 32 pixels are excluded, an explicit limitation.

Each returned box receives exactly two production evaluations: its ordinary learned action and a fixed cell-2 1-bit action. The chosen baseline gets its actual single ordinary evaluation; the original 50 retained boxes are not all exact-scored or relabeled an exhaustive oracle. This compares the evaluated shortlists, not their unknown best possible 30-action patch.

## Results

Only three rectangles, across two of 47 states: two Cat HQ step-3 tiles and one Mario64 HQ step-3 tile. All other states, including Googlewiki, return none. Six exact content evaluations total. Three binary patches have positive gain, but **zero** locally dominate, retain >=98% gain at <=75% bits, or retain >=90% gain at <=60% bits relative to the ordinary chosen patch.

| State / candidate | Gain / ordinary gain | Raw cost / ordinary cost | Gain per raw bit |
| --- | ---: | ---: | ---: |
| Cat HQ step 3 / tile (224,96), model | 10.060% | 4.467% | 281.369 |
| Cat HQ step 3 / tile (224,96), binary | 2.696% | 1.193% | 282.445 |
| Cat HQ step 3 / tile (320,128), model | 8.004% | 4.467% | 223.881 |
| Cat HQ step 3 / tile (320,128), binary | 3.475% | 1.193% | 363.997 |
| Mario64 HQ step 3 / tile (352,192), model | 1.341% | 1.241% | 126.898 |
| Mario64 HQ step 3 / tile (352,192), binary | 0.277% | 0.331% | 98.257 |

All three rectangles are 32x32. Their maximum IoU against the 50 retained stochastic proposals is 0.0878/0.0972/0.0357. Thus the exact tiles are different regions, but their absence does not imply a useful omission: they make very small total corrections per patch. Some lie on a local gain/cost frontier and have higher gain per bit; none wins the actual gain-minus-lambda*bits utility. The remaining iteration budget makes replacing one large useful patch with one tiny efficient patch a nontrivial allocation tradeoff. All evaluated local frontiers/utilities are in summary.json; no single gain-per-bit ratio is called completed-codec improvement.

Warmed median proposal time: ordinary **6.356 ms**, deterministic **2.785 ms**. Ordinary retains 50 candidates and exact-evaluates one; deterministic returns at most three and exact-evaluates two per returned box. Deterministic time includes residual construction/statistics/components/ranking and occurs even on the 45 states returning none. These one-pass component timings are not full-codec speedups or an integrated overhead measurement. A warmup explicitly replays the first ordinary/content proposal paths outside these measurements. Initial unwarmed component evidence remains ignored and excluded. No three-repeat full-call timing was run because no practical candidate passed.

## Googlewiki and working resolution

Googlewiki source luminance has 225 eligible binary tiles of 625 complete 32x32 tiles. The 250k working luminance has 32/225; 500k has 79/484. These are different physical tile extents, not paired source-region statistics or a causal attribution of lost error. The default full preparation roundtrip has substantial loss (Pass 17 source-roundtrip RGB MSE 684.311), so an existing-resolution region selector cannot restore all source edges.

One permitted targeted diagnostic encoded **Googlewiki HQ at 500k**, with all other defaults unchanged: 214,902 final bytes, independent original RGB MSE 439.912, versus ordinary 250k HQ 176,905 / 714.432. This is about +21.48% bytes and -38.43% MSE, an operating-point tradeoff, not a matched-rate win. It is the only added resolution point, not a sweep.

At higher-resolution steps 3/8/13/18, binary residual proposals still number zero. Maximum mean-split explained variance among contrast-eligible residual tiles is approximately 0.654/0.667/0.694/0.694, below the frozen 0.8 threshold. Source/target bimodality does not imply bimodal residuals after initialization. These observations support both downsampling loss and a residual-representation mismatch; they do not establish that loosening the threshold or raising resolution would be efficient.

## Decision and next hypothesis

Do not integrate this spatial rule or force 1-bit actions. The strongest next hypothesis is a narrowly gated signed hard-edge correction that distinguishes source structure from the initialized residual, with two explicitly fitted signed levels and geometry chosen jointly with cell scale. Present evidence argues against a bare free-two-level palette replacement; any architecture needs to explain how it handles multilevel initialized residuals and stays cheap. The non-interpolated mask itself was **not** implemented or encoded in this pass. Another full-resolution dense layer would repeat Pass 17's rate/runtime risks.

A better geometry/precision policy remains possible: the isolated Cat finer-grid dominance is an example. It does not justify a global fallback, broader benchmark or production promotion. No complete-file candidate, new serializer/decoder, repeated timing, holdout, release rebuild or deployment ran. Production v0 decoding is unchanged.
