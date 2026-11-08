# Pruning diagnostic results

## Method

The selected heuristic was the adaptive early exit in av1/encoder/tx_search.c::search_tx_type(). DIAG shadow-continued from the exact cutoff state on a deterministic coordinate hash (modulus 16), with a cap of 128 sampled events per image. It kept the coefficient-rate lower-bound check and zero-EOB stop. The incumbent's transform, coefficient buffer, EOB, entropy context, and reconstruction remained authoritative. REF and DIAG AVIF bytes matched exactly on every checked point.

A sampled event is eligible only when the adaptive condition actually fires on luma intra search and more than one transform is legal. “Complete” means every remaining legal candidate passed through the shadow loop; 18 sampled events stopped at the existing zero-EOB break. No observed lower-RD event was incomplete.

## Speed-6 results

| Image / q / speed | Eligible search calls | Adaptive cutoffs | Hash-selected | Sampled | Complete | Lower-RD alternatives | Median RD improvement among wins | Median improvement / cutoff RD | Extra candidate attempts / full RD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Googlewiki / 5 / 6 | 31,682 | 2,353 | 154 | 128 | 87 | 80 (62.5% of sample) | 27,267,545 | 18.13% | 348 / 347 |
| Mario64 / 50 / 6 | 17,871 | 1,303 | 79 | 79 | 71 | 50 (63.3%) | 684,878 | 21.81% | 284 / 241 |
| Cat / 35 / 6 | 157,430 | 13,631 | 862 | 128 | 123 | 87 (68.0%) | 1,558,891 | 7.95% | 490 / 480 |
| Total | 206,983 | 17,287 | 1,095 | 335 | 281 | 217 (64.8%) | — | — | 1,122 / 1,068 |

Among complete events, 217/281 (77.2%) had a lower-RD alternative. The median/mean relative improvement among winning alternatives was 18.13%/21.19% for Googlewiki, 21.81%/21.47% for Mario64, and 7.95%/9.44% for Cat. The diagnostic evaluated about 3.2–3.8 remaining candidates per sampled cutoff. These are local libaom RD values under the exact state and objective used by that call; they are not file-size or RGB-quality gains.

## Block size and transform pattern

Observed wins by transform size were concentrated in 4x4 and 8x8 blocks:

| Image | 4x4 | 8x8 | 16x16 | Other |
|---|---:|---:|---:|---:|
| Googlewiki | 41 / 79 sampled | 39 / 49 | 0 | 0 |
| Mario64 | 13 / 18 | 33 / 54 | 3 / 6 | 1 / 1 at 16x8 |
| Cat | 2 / 4 | 79 / 110 | 6 / 14 | 0 |

The ordinary incumbent was DCT_DCT in 305/335 events. Winning alternatives most often used ADST_DCT, DCT_ADST, ADST_ADST, or a 1D DCT. For Googlewiki, H_DCT was the most frequent winner (39/80); Mario64's winners were spread across five transforms; Cat's were mainly the three ADST combinations. This suggests the misses vary with block size and transform orientation. No residual-energy correlation was measured, so this pass does not claim a causal feature relationship.

## Speed-8 path check

The same adaptive condition was active at speed 8: Googlewiki q5 had 22 sampled / 385 pruning events and 8 observed lower-RD alternatives; Cat q35 had 79 sampled / 1,232 pruning events and 55 observed lower-RD alternatives. These were path checks only; no speed-8 expanded-file comparison was run after Gate C failed at speed 6.

## Interpretation

Gate B passes. The heuristic sometimes stops before locally better legal transforms, and the wins were not limited to a single image or transform size. However, many sampled events had no later candidate that beat the incumbent, and the local wins must be checked against completed files. Gate C results are in PASS38_RESULTS.md.
