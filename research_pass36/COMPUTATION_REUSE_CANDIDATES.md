# Computation reuse candidates

## 1. Reuse the repeated all-intra source-variance summary

- **Location:** `av1/encoder/partition_search.c`, `av1_rd_pick_partition`, around the two `log_sub_block_var` calls at lines 5623 and 5708; helper at line 5472.
- **Evidence:** For a 64×64 all-intra block, the function computes min/max normalized variance over 4×4 source cells before the partition pre-prune, then calls the same helper again before NONE/SPLIT search. The one sampled case attributed 52 inclusive samples (0.74%) and 12 exclusive samples (0.17%) to `log_sub_block_var` across the whole encode.
- **Invariant:** `log_sub_block_var` reads only the current luma source plane, 4×4 variance kernel, block dimensions/edge overflow, stride, and bit depth. `av1_set_offsets` binds the current block before the first call. The intervening all-intra pruning reads that source and does not rebind the source-plane pointer. A local pair of doubles could carry the already computed values into the second branch with no cache or added memory.
- **Invalidation risks:** Must preserve the first call's exact `double` results and avoid sharing them across another block, frame, or bit depth. The second call is reached only when the all-intra block-size condition holds.
- **Existing mechanism:** No memoization for this local helper. The partition and intra searches have other, context-aware pruning/reuse mechanisms.
- **Decision:** Not implemented. The measured share is below 1% inclusive in the only sampled case, so the potential full-encode gain is too small to justify a change without broader evidence.

## 2. Reuse transform/RD results across intra candidates

- **Location:** `av1/encoder/intra_mode_search.c:av1_rd_pick_intra_sby_mode` and `av1/encoder/tx_search.c:av1_pick_uniform_tx_size_type_yrd` / `av1_txfm_rd_in_plane`.
- **Evidence:** These paths account for substantial inclusive samples in the one speed-6 case. The intra search fully evaluates the modes surviving HOG, mode-mask, angle, and model-RD pruning.
- **Invariant needed:** Prediction pixels, above/left reconstruction, transform mode/size, coefficient context, quantizer, RD multiplier, entropy state, and best-RD bound all need to match. The same spatial block does not establish these conditions.
- **Existing mechanism:** `reuse_prev_rd_results_for_part_ab` copies prior results only when the partition contexts are marked ready and match. The transform residual-hash result cache is guarded by `is_inter` and is not active for all-intra candidates.
- **Decision:** Reject for exact reuse across different intra candidates. The inputs and search bounds differ.

## 3. Reuse partition-candidate context allocations

- **Location:** `av1/encoder/partition_search.c` candidate construction via `av1_alloc_pmc`, `av1/encoder/context_tree.c`, and the shared coefficient buffer.
- **Evidence:** `aom_memalign` has 3.39% exclusive samples and `av1_free_pmc` 1.57% in the one case. The sampled allocator leaf does not identify which caller owns that cost.
- **Invariant needed:** Candidate contexts contain coefficients, quantized coefficients, entropy contexts, mode info, and reconstruction-dependent state. Reuse requires resetting all fields and respecting PC_TREE winner lifetimes and `shared_coeff_buf` ownership.
- **Existing mechanism:** Libaom already uses `shared_coeff_buf`, PC_TREE context lifetimes, and explicit context copying for eligible partition reuse.
- **Decision:** Not selected. The current trace does not establish the allocator's caller share, and safe reuse risks duplicating existing ownership/lifetime machinery.
