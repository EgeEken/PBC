# RD search profile

## Coverage

This is one optimized, symbol-enabled WPR sample profile: 25 fresh-process encodes of Googlewiki (`qcolor=5`, speed 6, one encoder thread). The symbol build passed AVIF and decoded-RGB parity before its symbols were used. It is not the requested five-case/two-capture matrix. Percentages below use 7,011 encoder-process samples at 1 ms each. Inclusive samples overlap through the call tree; exclusive samples count only the sampled instruction's function.

## Hotspots

| Function | Source | Inclusive CPU | Exclusive CPU | Interpretation |
|---|---|---:|---:|---|
| `split_partition_search` | `av1/encoder/partition_search.c:4546` | 80.02% (5,610 ms) | 0.20% (14 ms) | Recursive split evaluation; inclusive cost includes child partition and mode work. |
| `block_rd_txfm` | `av1/encoder/tx_search.c:3056` | 33.78% (2,368 ms) | 0.77% (54 ms) | Transform block RD path and descendants. |
| `av1_rd_pick_intra_sby_mode` | `av1/encoder/intra_mode_search.c:1718` | 27.66% (1,939 ms) | 2.21% (155 ms) | Luma intra mode search, including transform evaluation. |
| `search_tx_type` | `av1/encoder/tx_search.c:2165` | 24.30% (1,704 ms) | 1.85% (130 ms) | Transform type candidate search. |
| `av1_rd_pick_partition` | `av1/encoder/partition_search.c:5670` | 18.78% (1,317 ms) | 0.27% (19 ms) | Recursive RD partition search entry and pruning. |
| `av1_pick_uniform_tx_size_type_yrd` | `av1/encoder/tx_search.c:3651` | 18.77% (1,316 ms) | 0.07% (5 ms) | Transform-size/type RD evaluation. |
| `av1_optimize_txb` | `av1/encoder/encodemb.c:101` | 19.17% (1,344 ms) | 0.37% (26 ms) | Coefficient optimization. |
| `log_sub_block_var` | `av1/encoder/partition_search.c:5500` | 0.74% (52 ms) | 0.17% (12 ms) | All-intra 4×4 source-variance summaries. |

The top sampled leaf functions were `update_coeff_eob` (3.48%), `aom_memalign` (3.39%), `build_non_directional_intra_predictors` (2.34%), `update_coeff_simple` (2.28%), `av1_rd_pick_intra_sby_mode` (2.21%), `av1_optimize_txb` (2.05%), `av1_predict_intra_block` (2.00%), `get_nz_mag` (1.93%), `search_tx_type` (1.85%), and `av1_free_pmc` (1.57%). Allocation samples are attributed to the allocator function, not to a specific caller; they do not establish that `av1_alloc_pmc` alone accounts for that share.

## Call path and existing work reduction

The sampled chain entered `aom_codec_encode` and `encoder_encode`, then `av1_get_compressed_data` → `encode_frame_to_data_rate` → `av1_encode_frame` → `encode_tiles` → `encode_rd_sb` → `split_partition_search` / `av1_rd_pick_partition`. Partition recursion calls `pick_sb_modes`, intra prediction/mode search, and transform/quantization routines. The large inclusive partition values are expected for recursive callers and are not independent costs.

The pinned source already prunes partition types before search, applies best-RD bounds and post-NONE/SPLIT pruning, reuses ready RD contexts for eligible AB partition shapes, and prunes intra modes using HOG masks, mode masks, angle-RD checks, and model-RD/top-mode checks. `av1_pick_uniform_tx_size_type_yrd` has an MB residual-hash cache for inter blocks; its lookup is guarded by `is_inter`, so it does not cover these all-intra AVIF runs.

Pass 35's separate, instrumented counters confirm speed 6 enters RD partition search and speed 8 uses the variance-based non-RD partition path. No sampled speed-8 profile was obtained here, so there is no speed-6 versus speed-8 function-share comparison.

## Attribution limits

The symbol-enabled build kept the same `.text` bytes as the ordinary executable. Sample stacks resolved to functions for 92.4% of all sampled encoder events. Inlining and the missing 7.6% of symbolized samples limit function-level attribution. This is one image/quality point, one independent capture, and a WPR profile with measured overhead around 13.6%; treat it as directional evidence only.
