# Experiment design

## Frozen hypothesis and rule

Pass 37 tests whether a cheap seed order makes the existing modeled-RD pruning in av1_rd_pick_intra_sby_mode() reject more candidates before transform/RD search.

For the first 13 base intra modes, sort the existing intra_rd_search_mode_order indices by ascending bmode_costs[mode]. Use a stable insertion sort; equal costs retain their original Pass 35 order. Process all directional delta-angle candidates afterward in the existing order. Eligibility checks and pruning thresholds remain unchanged. The rule adds no pixel features, no model calls, and no candidate state cache. Its only new work is sorting 13 integers per eligible block. The experimental ordering is compile-time gated by PBC37_EXPERIMENT; without the definition the research source follows its original loop order (CTRL).

This signal is the syntax cost already used in actual RD cost. It is available before expensive transform evaluation, but it does not estimate prediction distortion directly. The expected benefit is earlier establishment of useful best_model_rd and top-model bounds, allowing the existing model-RD prune to reject later base-mode searches. The failure risk is that syntax cost is poorly correlated with model RD or that base-mode order has little influence on total transform work. Since candidate evaluation uses mutable encoder scratch, byte parity and repeated deterministic output are required.

## Frozen operating points

Use Pass 35's selected speed-8 quality points at speed 6: Googlewiki q5 and q20; Mario64 q50; Flower q80; Cat q35 and q50; Coast q80. Speed-8 controls are Googlewiki q5 and Cat q35. All runs use the pinned libaom 3.12.1/libavif 1.3.0 AVIF frontend, 8-bit, 4:2:0, full range, CICP 1/13/1, one worker, qalpha 0, metadata omitted, and the same source bytes.

Primary correctness and speed comparisons are REF (clean pinned archive), CTRL (the same research source with the switch off), and EXP (switch on). Quality points are fixed before EXP results; no image-specific tuning is allowed. Benchmark repetitions and decoder checks are specified in the Pass 37 brief and recorded in the scripts/results.

## Resource/state assumptions

Only mode visitation changes. No candidate result, reconstruction, entropy context, or transform state is reused. The original per-candidate initialization and winner-state copies remain in place. Stack storage is a 13-element integer order array; no heap allocations. The order is deterministic and local to one block. The patch is research-only and leaves AV1 syntax and decoder code untouched.
