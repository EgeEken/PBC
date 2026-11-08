# Transform-search mechanism

## Source and call path

Pinned source is libaom 3.12.1 (`10aece4157eb79315da205f39e19bf6ab3ee30d0`). The central routine is `search_tx_type()` in `av1/encoder/tx_search.c`. Intra transform blocks reach it through `av1_txfm_rd_in_plane()` and `try_tx_block_no_split()` in the same file; inter blocks also use `tx_type_rd()`. The AVIF still-image points use all-intra AV1 frames. Speed 8's hybrid intra path still routes some blocks through RD intra, as Pass 35 counters show.

## Eligibility and ordering

`get_tx_mask()` forms `allowed_tx_mask` from the transform set legal for transform size, coding mode, prediction direction, reduced-set flags, lossless state, and encoder restrictions. A forced transform can reduce the mask to one entry. The search loop visits `txk_map` in order and ignores invalid/disallowed entries. Unless `prune_tx_2D()` runs, `txk_map` starts in enum order. `prune_tx_2D()` uses horizontal/vertical residual features and a neural probability model to remove transforms and order survivors, but its call in `get_tx_mask()` is restricted to inter blocks. `prune_tx_type_est_rd` invokes proxy-RD pruning and is disabled at all-intra speed 6 by the speed-6 override. Speed 6 does enable `fast_intra_tx_type_search` for MODE_EVAL; that forces a default intra transform there. Winner-mode evaluation resets that single-transform setting and performs refinement over the legal set.

At speed 6, `adaptive_txb_search_level` is set to 2 for all-intra frames. Speed 8 inherits the level and fast-intra mode setting; its all-intra speed setup also enables RD winner processing. Thus the adaptive exit is active in the RD transform path at both speeds when its caller-bound condition fires. The winner path's exact frequency is measured in this pass; feature values alone do not prove it fires on a particular image.

## Candidate RD

For each permitted candidate, `search_tx_type()` transforms and quantizes the residual, gets coefficient rate cost from `av1_optimize_b()` or `cost_coeffs()`, computes pixel- or transform-domain distortion as selected by `TxfmSearchParams`, and calculates `RDCOST(x->rdmult, rate, distortion)`. A strictly lower RD becomes the incumbent; its coefficient buffer, entropy context, EOB and transform type are retained. The chosen transform is reconstructed after search.

The rate-only condition is:

`RDCOST(x->rdmult, rate_cost, 0) > best_rd`

It skips only that candidate's distortion evaluation (`continue`). The candidate rate is the same value later used by this local RD calculation. In the pinned source, both coefficient-cost paths include the transform-type signaling cost when EOB is nonzero; the zero-EOB path uses the skip cost. With nonnegative RD multiplier, nonnegative distortion, and the same rate units and coding context, setting distortion to zero yields a lower bound. It is safe to skip a candidate whose lower-bound cost already exceeds the incumbent. It is not the missed-opportunity target.

## Heuristic early exit

After each complete candidate, the adaptive condition is:

`best_rd - (best_rd >> adaptive_txb_search_level) > ref_best_rd`

If true, the loop breaks, leaving later permitted candidates unevaluated. `ref_best_rd` comes from a competing mode/partition RD bound. The fractional reduction of the best candidate is a heuristic estimate of whether further transform choices could matter; it is not a lower bound on unseen candidates. A later transform can have lower coefficient rate and distortion, so this rule can miss a local winner. It is distinct from the safe rate-only skip.

A separate `skip_tx_search && !best_eob` break is also present. Diagnostic rows record when that condition limits completion after the adaptive cutoff. It is not disabled.

## State equivalence

The diagnostic is a same-call shadow continuation. At a sampled adaptive cutoff, it records the current winner RD and continues subsequent legal candidates without allowing them to update the official winner buffers, EOB, entropy context, or transform type. The existing safe rate-only check remains enabled. It then follows the ordinary end-of-function restoration and reconstruction using the pre-cutoff winner. Thus the selected transform and encoder state remain the REF result, while each shadow candidate is evaluated from the same residual, quantization state, coefficient context, RD multiplier, and transform block. DIAG and REF should be byte-identical. Only the bounded extra transform/quantization/RD work differs.

