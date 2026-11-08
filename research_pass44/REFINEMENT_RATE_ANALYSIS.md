# Refinement rate analysis

## Method

Pass 44 did not alter V4 mode syntax or decoder behavior. It reused Pass 43 V4-B candidates at three coarse budgets per image: 3, 6, and the complete V3.1 Compression prefix (12 P3 patches for Googlewiki, 19 for Cat, 50 for Coast). The first Cat/Coast pass used a truncated 13-patch candidate; those rows are excluded from the corrected results. For each completed V4 candidate, the script decoded its coarse base and full reconstruction, measured each selected region’s RGB SSE reduction against that base, and ranked regions by reduction per raw serialized record byte. Existing procedural and transform records were copied without changing their parameters or reconstruction semantics.

The initial screening tested five subset counts (0, 8, 32, 128, 512) for every coarse choice, operating point and image: 90 complete V4 files. Each file used the same v2 header, region map, shared XZ stream and decoder. Screening compared actual completed bytes and decoded RGB MSE. It retained the all-SKIP coarse-only file for every base.

A second bounded search moved the top-ranked subset count from 0 through 512 with at most 10 completed-file evaluations per coarse choice. The corrected search evaluated 186 complete V4 files. The search chose the best measured feasible file by decoded MSE, with a hard cap of 105% of the V3.1 Compression target at low rate and 105% of the V3.1 Quality target at the intermediate point. No per-region XZ measurements were run. No source encoder, bitstream, decoder or coding mode was changed.

This is a bounded offline allocator experiment: Pass 43 had already generated the V4 native-region candidates. It tests whether a sparse subset of those real records can fit and improve the RD point; it does not measure an integrated source-to-file sparse encoder.

## Costs and results

| Image / target | Coarse-only A (B / MSE) | Frozen-base selective B (B / MSE) | Joint coarse/detail C (B / MSE) | V3.1 target MSE |
|---|---:|---:|---:|---:|
| Googlewiki / Compression | 29,180 / 1,005.02 | 30,400 / 994.95 | 30,400 / 994.95 | 1,005.02 |
| Cat / Compression | 7,184 / 433.81 | 7,384 / 433.04 | 7,384 / 433.04 | 433.81 |
| Coast / Compression | 6,616 / 73.13 | 6,780 / 72.40 | 6,780 / 72.40 | 73.13 |
| Googlewiki / Quality | 29,180 / 1,005.02 | 86,352 / 828.11 | 86,352 / 828.11 | 781.03 |
| Cat / Quality | 7,184 / 433.81 | 12,792 / 427.01 | 12,792 / 427.01 | 299.19 |
| Coast / Quality | 6,616 / 73.13 | 12,808 / 66.90 | 12,808 / 66.90 | 40.88 |

All selective files fit the hard cap. At low rate, the corrected winners use 97 regions for Googlewiki, 14 for Cat and 10 for Coast. The intermediate winners use 180, 512 and 512 regions. Complete V3.1 prefixes restore the V3.1 base MSE on Cat and Coast; native detail then reduces MSE by only 0.18% and 1.01%, respectively. Full mode distributions, file/pixel hashes, exact rate gaps and per-prefix search points are in `outputs/final_joint_allocation.json`, `outputs/full_prefix_budget_search.json` and `PASS44_SUMMARY.json`.

At Compression rate, the largest MSE decrease against the corresponding coarse-only point is 1.01% on Coast, using 2.5% more bytes than its coarse-only file. Relative to V3.1 Compression at approximately the same total rate, MSE is 1.00% lower on Googlewiki, 0.18% lower on Cat and 1.01% lower on Coast. These changes are small, do not meet the 10% screen, and do not demonstrate a meaningful practical improvement over V3.1. The result does establish that the missing complete coarse prefix explains much of the earlier Cat/Coast base-quality gap.

At the Quality target, the bounded candidate menu does not approach the available V3.1 Quality rate budget on Cat and Coast; the best tested files are still worse than V3.1 Quality MSE. The full-prefix search only tested up to 512 regions, so it does not characterize the rest of those large budgets. Within the tested menu, joint selection chose the full V3.1 prefix. No extrapolation beyond the measured files is made.

## Rate accounting

The raw candidate rank is only a cheap estimate: gain divided by the selected region record’s uncompressed bytes. The mode map costs a fixed 2 bits per 16×16 region, and the coarse prefix is part of the same V4 payload. The final V4 file is then XZ-compressed once for each tested subset. The actual compressed size is not the sum of standalone record costs. Complete-base Cat and Coast files are 7,184 B and 6,616 B; their low-rate winners are 7,384 B and 6,780 B. That measured interaction is why only completed-file size determines feasibility.

## Decision

The complete V3.1 prefixes recover the base reconstruction at near-V3.1 rates. Sparse selection then adds only a small gain within 5% of the V3.1 Compression file sizes. At intermediate rate, the bounded subset menu uses only part of the available budget and remains far from V3.1 Quality. The current 16×16 procedural/transform representation has not demonstrated a competitive general RD frontier. Stop allocator tuning here; improving the representation itself would be required for a larger gain.
