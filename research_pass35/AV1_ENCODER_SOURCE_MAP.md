# AV1 Encoder Source Map

References are pinned libaom `10aece4157eb79315da205f39e19bf6ab3ee30d0` (v3.12.1). Active branches were verified by the profile report.

## Partition and call paths

- `av1/encoder/encodeframe.c:encode_nonrd_sb` chooses fixed or variance-based partitioning (`av1_choose_var_based_partitioning`) and applies it through `av1_nonrd_use_partition`. This was active at speed 8.
- `av1/encoder/partition_search.c:av1_nonrd_use_partition` reads precomputed partition types, may call `try_split_partition`, then dispatches `pick_sb_modes_nonrd` and recurses through the tree. `hybrid_intra_mode_search` selects RD or non-RD intra per block.
- `partition_search.c:av1_nonrd_pick_partition` performs ML-assisted NONE/SPLIT evaluation, but is conditional on ML partition configuration and was inactive in these runs.
- `encodeframe.c` calls `av1_rd_pick_partition` at speed 6. `partition_search.c:av1_rd_pick_partition` recursively evaluates candidates under RD bounds and pruning (`av1_prune_partitions`, best-RD limits, speed features). Speed-6 partition CSVs were nonzero; this path was inactive at speed 8.
- `av1/encoder/speed_features.c` configures frame-size, independent and qindex-dependent speed features; speed level alone does not define the active branch.

## Intra modes

- `partition_search.c:hybrid_intra_mode_search` calls `nonrd_pickmode.c:av1_nonrd_pick_intra_mode` or `rdopt.c:av1_rd_pick_intra_mode_sb`.
- Non-RD intra tests `intra_mode_list` up to `RTC_INTRA_MODES`; it estimates block modes with `av1_estimate_block_intra` and scores rate/distortion. Existing pruning includes flat top-left blocks, neighbor modes, best-SAD/H-pred rules and optional palette search.
- RD intra calls `intra_mode_search.c:av1_rd_pick_intra_sby_mode` for luma, then chroma search. It considers directional and non-directional modes and angle deltas. Existing pruning includes HOG masks, mode masks, odd-angle RD checks, model-RD/top-mode selection, then full quantized RD evaluation. Palette, filter-intra and winner-mode processing are later paths.
- Candidate costs use `RDCOST` from `av1/encoder/rd.h`, combining estimated rate and distortion under the current RD multiplier.

## Transform, quantization and pruning

- RD intra calls `av1_pick_uniform_tx_size_type_yrd` in `av1/encoder/tx_search.c`; block size and speed features constrain transform evaluation.
- Quantization dispatches through `av1/encoder/av1_quantize.c` facades such as `av1_quantize_fp_facade` and `av1_quantize_b_facade`, selecting SIMD implementations at runtime.
- The search is mixed-cost: model-RD and rate thresholds reject candidates before full reconstruction; RD partition search uses bounds; non-RD mode uses precomputed variance partitions and estimated costs.
- AV1 already provides adaptive quantization, segmentation/delta-q, skip decisions, transform choices and coefficient quantization. Pass 35 did not establish that PBC's earlier mechanisms are absent from libaom.
