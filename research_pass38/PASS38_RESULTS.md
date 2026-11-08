# Pass 38 results

## Objective and provenance

This pass tested whether libaom 3.12.1's adaptive transform-type RD exit skips useful candidates and whether fully expanding that search improves completed AVIF output. The pinned revisions were libaom 10aece4157eb79315da205f39e19bf6ab3ee30d0 and libavif 1aadfad932c98c069a1204261b1856f81f3bc199. The branch is research/experiments, starting HEAD 3723dc14de88c6c3dcbd98dd80a097e019a92a9f. Builds used Windows 11 x64, GCC/G++ 13.2, CMake 3.29.2, Ninja 1.12, Release -O3 -DNDEBUG, NASM, and one encoder thread.

The bounded same-state diagnostic patch is adaptive-tx-shadow-diagnostic.patch. The full-file expanded-search patch is adaptive-tx-expanded-search.patch. Both patches change only av1/encoder/tx_search.c in isolated ignored pinned-source exports. Production PBC files, Pass 35–37, models, and decoder code were not changed.

## Transform search findings

search_tx_type() is in av1/encoder/tx_search.c. Intra RD callers include av1_txfm_rd_in_plane() and try_tx_block_no_split(); inter search also reaches the function through tx_type_rd(). get_tx_mask() derives legally permitted candidates from transform size, intra/inter mode, reduced transform-set flags, prediction direction, and config restrictions. txk_map controls visit order; absent the inter 2D-pruning helper, the initial order is transform enum order.

Speed 6 enables fast_intra_tx_type_search for ordinary mode evaluation, where one default intra transform can be used before winner-mode refinement. It disables prune_tx_type_est_rd at the all-intra speed-6 override. prune_tx_2D() uses residual-energy projections and a neural probability model, but the call from get_tx_mask() is restricted to inter blocks. Winner-mode intra search therefore has a legal multi-transform set, and the adaptive early exit can fire there. The diagnostic observed 2,353, 1,303, and 13,631 such cutoffs on the three speed-6 images. Speed 8 also entered the same path.

For each candidate, the routine transforms and quantizes coefficients, obtains coefficient rate from av1_optimize_b() or cost_coeffs(), computes its configured distortion, and calculates RDCOST(x->rdmult, rate, distortion). The incumbent's coefficient buffer, EOB, entropy context, statistics, and transform are retained and reconstructed after search.

The rate-only check RDCOST(x->rdmult, rate_cost, 0) > best_rd skips only that candidate's distortion calculation. Both pinned coefficient-cost paths include the transform-type signaling cost when EOB is nonzero; the zero-EOB path uses the skip cost. With nonnegative rate multiplier and distortion and the same coding context, zero distortion gives a lower bound under this local objective. This is safe branch-and-bound pruning and was left enabled.

The adaptive break best_rd - (best_rd >> level) > ref_best_rd uses a fraction of the current best RD against an external competing-mode/partition bound. No guaranteed bound exists on the improvement an unseen transform might provide, so this is heuristic pruning. A separate skip_tx_search && !best_eob break remains active and is reported as an incomplete shadow event when it prevents expansion.

## Local pruning-accuracy diagnostic

The speed-6 sample totaled 335 events. Of those, 217 (64.8%) had a later legal transform with a lower local RD; all 217 wins occurred in fully completed shadow searches. Median improvements among those wins were 18.13%, 21.81%, and 7.95% of the cutoff RD for Googlewiki, Mario64, and Cat. The shadow added 1,122 candidate attempts, 1,068 of which reached full RD calculation.

The shadow evaluation used the same pre-search pixels, coefficient context, quantization state, RD multiplier, and candidate objective. It preserved the cutoff winner for reconstruction. All REF/DIAG byte hashes matched, and outputs decoded. Event-level data and aggregates are in PRUNING_DIAGNOSTIC_RESULTS.md and ignored outputs/shadow_mod16/.

## Expanded-search completed-file comparison

The REF and EXP builds used ordinary AVIF. EXP disabled only the adaptive transform exit; the safe coefficient-rate check, legal candidate set, zero-EOB stop, quantization, prediction, and syntax were retained. Same-qcolor rows below are not matched-rate comparisons. Each row has one exploratory encode-time sample; RSS was sampled every 5 ms. The diagnostic is not a speed claim.

| Image / q / speed | Bytes REF → EXP | Byte change | RGB MSE REF → EXP | MSE change | Encode seconds REF → EXP | Decode seconds REF → EXP |
|---|---:|---:|---:|---:|---:|---:|
| Googlewiki / 5 / 6 | 20,613 → 20,641 | +0.136% | 619.155 → 619.217 | +0.010% | 0.270 → 0.263 | 0.083 → 0.079 |
| Mario64 / 50 / 6 | 39,332 → 39,214 | −0.300% | 57.886 → 57.858 | −0.047% | 0.267 → 0.262 | 0.214 → 0.210 |
| Cat / 35 / 6 | 262,385 → 262,490 | +0.040% | 46.502 → 46.482 | −0.043% | 1.809 → 1.887 | 1.116 → 1.130 |

Peak sampled RSS was about 27 MB on Googlewiki, 34 MB on Mario64, and 129 MB on Cat, with REF/EXP essentially equal. Repeat EXP encodes produced identical AVIF SHA-256 on all three images. All REF/EXP files decoded with the unchanged pinned avifdec; all EXP files also decoded with FFmpeg. RGB MSE promoted both arrays to float64 before subtraction and squaring; a 0-to-255 channel difference checked to 65,025.

Gate C fails: the local RD wins did not produce a material or consistent completed-file improvement. At fixed qcolor, rate changes ranged from −0.30% to +0.14%, RGB MSE changes stayed within ±0.05%, and the one-run time samples were below a reliable speed claim. The tiny Mario64 improvement does not approach the pass's RD threshold and does not generalize across these development images.

## Gates and decision

| Gate | Result | Evidence |
|---|---|---|
| A — Mechanism and feasibility | Pass | Active speed-6/8 path; safe rate bound separated from adaptive heuristic; same-state shadow is possible. |
| B — Missed local RD | Pass | 217/335 sampled speed-6 events found a lower-RD legal transform. |
| C — Completed-file relevance | Fail | Expanded search had mixed, sub-0.3% rate changes and sub-0.05% MSE changes. |
| D — Selective modification | Not attempted | Gate C failed; no selective heuristic was justified. |
| E — Held-out generalization | Not attempted | Development files did not justify a frozen EXP candidate. |

No algorithmic improvement is proposed. The three held-out images were not encoded.

## Next direction

Stop pursuing transform ordering and pruning for now. A bounded next experiment should test libaom's existing variance adaptive quantization against aq-mode=0 using aq-mode=1 at speed 6 on fixed image/quality points. In aq_variance.c, av1_vaq_frame_setup() defines segment qindex deltas from fixed rate ratios; partition_search.c::setup_block_rdmult() assigns an energy segment per block from mb_energy or log block variance. avifenc exposes the existing control as --advanced aq-mode=1.

Record per-segment area and block energy, then compare complete-file bytes and float64 RGB MSE. Add at most two measured bracketing q points per mode for matched-rate comparisons. The initial question is whether current energy-based allocation improves rate-distortion across smooth, edge, and textured images; do not alter AQ in this pass.

## Worktree

No files were staged, committed, pushed, reset, or cleaned. Passes 35–37 remain preserved. See the final status in the handoff response.

