# Strict-budget results

## Allocation outcomes

Target budgets are actual frozen V3.1 files; the hard limit is 105%. A is the best coarse-only point inside the cap. B fixes that coarse base and adds the best feasible sparse subset. C searches all three existing coarse prefixes and reallocates rate jointly. All reported bytes are completed V4 files using the shared XZ stream.

| Image / target | V3.1 target | A coarse-only bytes / MSE | B frozen base + detail bytes / MSE | C joint bytes / MSE | C rate gap |
|---|---:|---:|---:|---:|---:|
| Googlewiki / Compression | 28,974 | 29,180 / 1,005.02 | 30,400 / 994.95 | 30,400 / 994.95 | +4.92% |
| Cat / Compression | 7,036 | 7,184 / 433.81 | 7,384 / 433.04 | 7,384 / 433.04 | +4.95% |
| Coast / Compression | 6,461 | 6,616 / 73.13 | 6,780 / 72.40 | 6,780 / 72.40 | +4.94% |
| Googlewiki / Quality | 82,440 | 29,180 / 1,005.02 | 86,352 / 828.11 | 86,352 / 828.11 | +4.75% |
| Cat / Quality | 63,297 | 7,184 / 433.81 | 12,792 / 427.01 | 12,792 / 427.01 | −79.79% |
| Coast / Quality | 18,717 | 6,616 / 73.13 | 12,808 / 66.90 | 12,808 / 66.90 | −31.57% |

B and C select the same complete V3.1 Compression prefix on all six points. The joint search provides no improvement over the simpler fixed-base path. At Compression rate, selective V4 is 1.00%, 0.18% and 1.01% lower MSE than V3.1 on Googlewiki, Cat and Coast, respectively. At the Quality target, all three V4 selective points have higher MSE than V3.1 Quality. No result meets the 10% multi-image RD screen. The intermediate Cat/Coast search tested at most 512 regions and leaves much of the rate budget unused; it is a bounded candidate result, not an optimum claim.

## Four-axis operating points

The exact V3.1 Compression, AVIF near-rate/q0, V4-B D-low and V4 selective low points are in `COMPETITIVE_BASELINE.md`. The table below adds the actual selected-stream decode and its encode-timing status.

| Image / rate target | Codec | Bytes | bpp | RGB MSE | Encode (s) | Decode (s) |
|---|---|---:|---:|---:|---:|---:|
| Googlewiki / Compression | V3.1 Compression | 28,974 | 0.3507 | 1,005.02 | 0.178 | 0.0175 |
|  | AVIF q12 | 29,727 | 0.3598 | 498.36 | 0.087† | 0.090† |
|  | V4-B full D-low p3003 | 423,996 | 5.1318 | 70.16 | 3.263 | 1.428 |
|  | V4 Pass44 selective C | 30,400 | 0.3679 | 994.95 | not measured end-to-end | 0.038 |
| Cat / Compression | V3.1 Compression | 7,036 | 0.00660 | 433.81 | 0.468 | 0.0664 |
|  | AVIF q0 (unmatched, +33.7%) | 9,406 | 0.00883 | 241.48 | 0.294† | 0.415† |
|  | V4-B full D-low p3003 | 978,704 | 0.9184 | 143.80 | 14.847 | 5.960 |
|  | V4 Pass44 selective C | 7,384 | 0.00693 | 433.04 | not measured end-to-end | 0.262 |
| Coast / Compression | V3.1 Compression | 6,461 | 0.00430 | 73.13 | 0.848 | 0.116 |
|  | AVIF q0 | 6,747 | 0.00450 | 88.54 | 0.363† | 0.506† |
|  | V4-B full D-low p3003 | 182,676 | 0.1217 | 12.23 | 9.086 | 1.626 |
|  | V4 Pass44 selective C | 6,780 | 0.00452 | 72.40 | not measured end-to-end | 0.523 |

`†` AVIF timings include CLI launch, file I/O and PNG output. They are not directly comparable to in-process PBC timings; no AVIF speed ratio is claimed. The selective streams were assembled offline from already-generated V4 regional records, so their complete source-to-file encode time is unavailable. Do not infer it from the full V4-B encode time.

In matched in-process single-thread measurements, V4-B full D-low is 18.4×, 31.7× and 10.7× slower to encode than V3.1 on Googlewiki, Cat and Coast. Its decode is 81.6×, 89.8× and 14.0× slower, respectively. Corrected selective outputs decode in 0.039 s, 0.262 s and 0.523 s (five warmed samples), still about 2.1×, 3.9× and 4.5× the V3.1 decode time. AVIF speed remains unranked due to the timing boundary.

## Correctness and visuals

All six corrected final C files were decoded in fresh processes. Each pixel hash matched the encoder-side decoded result, and `skip_exact=true` for every file. Pass 44 uses the unchanged v2 decoder and stream syntax. The Pass 43 odd-dimension case (73×59) was not repeated in this pass. No neural network is required.

Visuals are in `visuals/`. The Googlewiki crop shows that the selected detail does not recover the sharp text lost in the coarse-only reconstruction. The Coast smooth-sky crop shows visible broad banding in V4-A; the corrected full-prefix Pass 44 crop remains visually close to the coarse-only reconstruction and introduces no obvious new seam in that smooth region. The map is sparse: 10 selected regions out of the 46,905-region Coast grid at the low target. This does not establish edge behavior across all images.

## Decision

Gate B passes: valid completed V4 files fit under all six caps. Gates C and D do not pass: complete prefixes recover the V3.1 base reconstruction, but the sparse details add only 0.18% to 1.01% MSE reduction at Compression rates. The 10% threshold is unmet. Gate E does not pass: existing V4-B is much slower than V3.1 and the selective encoder was not integrated or timed. The allocator should not be integrated into production or followed by a larger threshold sweep.


