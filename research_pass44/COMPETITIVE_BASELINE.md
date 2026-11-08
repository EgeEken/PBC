# Competitive baseline — Phase A

## Competitive answer

**No competitive V4 result demonstrated so far.** The Pass 43 V4-B native-refinement point has much lower MSE than V3.1 Compression, but costs 14.6× to 139× the completed-file rate across these samples; it also encodes 10.7× to 31.7× slower and decodes 14.0× to 89.8× slower than the actual released V3.1 under matched in-process single-thread timings. The Pass 44 sparse allocator produces strict-budget files, but at those rates its MSE is about 1% lower on Googlewiki and 3–14% higher on Cat and Coast than V3.1. The selective files were assembled from existing records, so a complete source-to-file encode timing for that allocator was not measured.

AVIF has actual ±5% rate matches for 11 of 12 V3.1 preset targets. At Compression rate, AVIF q12 is near-rate and much lower MSE than V3.1 on Googlewiki; Cat q0 is 33.7% above the V3.1 target and Coast q0 is within 4.4%. Corrected Pass 44 Coast V4 is 0.49% larger than AVIF q0 and has lower MSE (72.396 vs 88.537), but V4 is much slower than the available startup-inclusive AVIF CLI observation and no direct speed ratio is valid. The four-axis AVIF timing gate remains incomplete.

## Direct four-axis comparison

Times use complete in-process codec calls except AVIF, whose timing includes CLI startup and file/PNG I/O. AVIF times are therefore shown with `†` and are not used for cross-codec speed ratios. V4-B D-low timing is the full p3003 stream, not the post-hoc selective stream.

| Image / target | Codec and point | Bytes | bpp | RGB MSE | Encode (s) | Decode (s) |
|---|---|---:|---:|---:|---:|---:|
| Googlewiki / V3 Compression | V3.1 Compression | 28,974 | 0.3507 | 1,005.02 | 0.178 | 0.0175 |
|  | AVIF q12 (+2.6%) | 29,727 | 0.3598 | 498.36 | 0.087† | 0.090† |
|  | V4-B D-low p3003 (+1,363.5%) | 423,996 | 5.1318 | 70.16 | 3.263 | 1.428 |
|  | V4 Pass44 selective (+4.9%) | 30,400 | 0.3679 | 994.95 | not measured end-to-end | 0.039 |
| Cat / V3 Compression | V3.1 Compression | 7,036 | 0.00660 | 433.81 | 0.468 | 0.0664 |
|  | AVIF q0 (+33.7%; no near-rate AVIF point) | 9,406 | 0.00883 | 241.48 | 0.294† | 0.415† |
|  | V4-B D-low p3003 (+13,806%) | 978,704 | 0.9184 | 143.80 | 14.847 | 5.960 |
|  | V4 Pass44 selective (+4.9%) | 7,384 | 0.00693 | 433.04 | not measured end-to-end | 0.262 |
| Coast / V3 Compression | V3.1 Compression | 6,461 | 0.00430 | 73.13 | 0.848 | 0.116 |
|  | AVIF q0 (+4.4%) | 6,747 | 0.00450 | 88.54 | 0.363† | 0.506† |
|  | V4-B D-low p3003 (+2,726%) | 182,676 | 0.1217 | 12.23 | 9.086 | 1.626 |
|  | V4 Pass44 selective (+4.9%) | 6,780 | 0.00452 | 72.40 | not measured end-to-end | 0.523 |

The selective V4 decode times are in-process measurements on the exact final streams. Its encode timing is not listed as a full codec time: Pass 44 reused already-generated regional records and did not integrate or time the selection stage with source preparation and candidate generation. The V4-B D-low and V3.1 timings above are exact-stream-verified complete encodes with one warmup and three single-thread repetitions; decode samples use one warmup and five repetitions.

The selected AVIF points were encoded with the pinned reference frontend. qcolor0 was explicitly measured at Googlewiki 5,218 B / MSE 1,390.61, Cat 9,406 B / MSE 241.48, and Coast 6,747 B / MSE 88.54. AVIF qcolor 0 is not a fixed bitrate. A bounded qcolor grid plus targeted points matched 11/12 V3.1 preset targets within ±5%; only Cat Compression had no match.

## Phase A diagnosis

V4-A’s low-rate regression is mainly a representation and allocation problem. Its source-resolution palette/index initialization costs much more than V3.1’s generated-grid initialization. Native regional corrections then add large transform payloads. Pass 42’s raw accounting shows the mode map is only 651 bytes and the header 32 bytes; transform records dominate raw payload size. The rate regression is not mainly mode-map signaling or the LZMA wrapper.

V3.1 earns its low-rate efficiency through both its downsampled initialization and early coarse procedural residual patches. The corrected Pass 44 experiment compares complete V3.1 Compression prefixes: 12 P3 patches for Googlewiki, 19 for Cat and 50 for Coast. Their V4 all-SKIP wrappers reconstruct the V3.1 Compression pixels exactly at 29,180 B / MSE 1,005.02, 7,184 B / MSE 433.81 and 6,616 B / MSE 73.13, respectively. This isolates the missing coarse stage as a major part of the low-rate loss. Adding sparse native detail within the 105% cap improves MSE only to 994.95, 433.04 and 72.40. The full-prefix search was limited to K≤512 and ten completed files per target; the intermediate-rate results therefore describe only this bounded candidate menu.

## Timing comparability

The controlled V3.1 and V4-B timings use the same single-thread in-process boundary. AVIF’s pinned output bytes and MSE are valid, but Pillow 11.3’s in-process AVIF encoder cannot set the required matrix coefficient and does not match the pinned reference bytes. A native C API probe crashed before `main`. AVIF timing consequently includes process startup and file I/O; no direct AVIF speed ratio is claimed. See `BENCHMARK_CONTRACT.md` and raw timing samples in `outputs/`.

The complete machine-readable rows, file hashes, per-preset rate matches, and stage accounting are in `outputs/competitive_baseline.json` and `PASS44_SUMMARY.json`.

