# Pass 44 results

### 1. Competitive result — primary conclusion

No competitive V4 result demonstrated so far. The existing V4-B native-refinement files are far above V3.1 Compression rates and encode 10.7×–31.7× slower while decoding 14.0×–89.8× slower under controlled single-thread in-process timing. The corrected strict-budget experiment uses complete V3.1 Compression prefixes (12/19/50 patches for Googlewiki/Cat/Coast), then selects sparse native detail. Its final MSE is 0.18%–1.01% lower than V3.1 Compression at 4.9%–4.95% larger files, but this is a small offline result, not a measured integrated encoder improvement. Complete source-to-file encode time is unavailable because the allocator reused Pass 43 candidate records.

AVIF rate/quality points are established for all three images, including qcolor 0 and 11 of 12 V3.1 preset targets within ±5%. The AVIF timing path includes CLI startup and file/PNG I/O, so a direct AVIF speed ratio is unavailable. Corrected Coast selective V4 is 0.49% larger than AVIF q0 and has 18.2% lower MSE, but this is an RD signal, not a four-axis win.

### 2. Direct four-axis comparison table

AVIF times marked `†` include process startup and file/PNG I/O; do not compare them as direct codec-call timings. V4 Pass44 selective encode time is unavailable because the test constructed files offline from existing regional records.

| Image / rate target | Codec and point | Bytes | bpp | RGB MSE | Encode (s) | Decode (s) |
|---|---|---:|---:|---:|---:|---:|
| Googlewiki / Compression | V3.1 Compression | 28,974 | 0.3507 | 1,005.02 | 0.178 | 0.0175 |
|  | AVIF q12 (+2.6%) | 29,727 | 0.3598 | 498.36 | 0.087† | 0.090† |
|  | V4-B full D-low p3003 (+1,363.5%) | 423,996 | 5.1318 | 70.16 | 3.263 | 1.428 |
|  | V4 Pass44 selective (+4.9%) | 30,400 | 0.3679 | 994.95 | not measured end-to-end | 0.038 |
| Cat / Compression | V3.1 Compression | 7,036 | 0.00660 | 433.81 | 0.468 | 0.0664 |
|  | AVIF q0 (unmatched, +33.7%) | 9,406 | 0.00883 | 241.48 | 0.294† | 0.415† |
|  | V4-B full D-low p3003 (+13,806%) | 978,704 | 0.9184 | 143.80 | 14.847 | 5.960 |
|  | V4 Pass44 selective (+4.95%) | 7,384 | 0.00693 | 433.04 | not measured end-to-end | 0.262 |
| Coast / Compression | V3.1 Compression | 6,461 | 0.00430 | 73.13 | 0.848 | 0.116 |
|  | AVIF q0 (+4.4%) | 6,747 | 0.00450 | 88.54 | 0.363† | 0.506† |
|  | V4-B full D-low p3003 (+2,726%) | 182,676 | 0.1217 | 12.23 | 9.086 | 1.626 |
|  | V4 Pass44 selective (+4.94%) | 6,780 | 0.00452 | 72.40 | not measured end-to-end | 0.523 |

Exact V3.1 preset points, AVIF q grid, all file hashes, and unmatched-rate gaps are in `PASS44_SUMMARY.json`. No AVIF speed ratio is claimed.

### 3. Pass 44 intervention result

The sparse-region allocator generated valid completed files under every Compression and Quality cap. After replacing truncated Cat/Coast prefixes with complete V3.1 Compression prefixes, the best joint selections were Googlewiki 30,400 B / MSE 994.95, Cat 7,384 B / 433.04 and Coast 6,780 B / 72.40. Relative to V3.1 Compression, these are 1.00%, 0.18% and 1.01% lower MSE, respectively. Each uses a complete coarse base plus 97, 14 and 10 selected native regions. All results fit under the 105% cap. The intermediate Cat/Coast menu was bounded to 512 regions and leaves much of the matched V3.1 Quality budget unused.

The complete V3.1 prefixes recover the V3.1 Compression base MSE on Cat and Coast at 7,184 B and 6,616 B, respectively; this shows that missing coarse procedural refinement was a principal source of the earlier low-rate regression. Sparse native detail adds only small MSE gains at the strict rate cap. Joint selection matched frozen-base selection for all six measured targets. The full-prefix search was bounded to ten completed-file evaluations per target and 512 detail regions. Stop allocator tuning; the remaining gap is unlikely to be solved by selecting among the same current native records.

### 4. Quality-scaling result

Googlewiki keeps improving as V4-B adds original-resolution corrections, but at substantially higher rates than JPEG, AVIF and V3.1:

| Codec point | Bytes | bpp | RGB MSE | Encode (s) | Decode (s) |
|---|---:|---:|---:|---:|---:|
| V3.1 High Quality | 176,905 | 2.139 | 714.43 | 0.528 | 0.0378 |
| JPEG q90 | 246,700 | 2.986 | 204.67 | 0.00644 | 0.00600 |
| AVIF q95 | 247,851 | 2.9999 | 193.92 | 0.178† | 0.111† |
| V4-B D-low p3003 | 423,996 | 5.132 | 70.16 | 3.263 | 1.428 |
| V4-B D-intermediate p3003 | 737,264 | 8.923 | 2.55 | 4.298 | 2.001 |
| V4-B D-high p3012 | 1,225,860 | 14.837 | 0.0384 | 5.327 | 2.637 |

High-quality encoder/decoder timing uses the complete V3-prefix-plus-V4 encode wrapper and the in-process v2 decoder. AVIF timing remains startup-inclusive. V4 retains its finer reconstruction ceiling and gives useful within-V4 MSE scaling, but this does not establish competitive RD: the high point is about five times the AVIF/JPEG bytes and much slower than V3.1.

### 5. Computational cost

The one-thread complete V4-B low encoder medians are 3.263 s (Googlewiki), 14.847 s (Cat), and 9.086 s (Coast), including coarse-prefix generation. Relative to V3.1 Compression these are 18.4×, 31.7× and 10.7× slower. Full V4-B low decode medians are 1.428 s, 5.960 s and 1.626 s, or 81.6×, 89.8× and 14.0× slower than V3.1. Corrected selective-stream decode medians are 0.039 s, 0.262 s and 0.523 s, still 2.1×–4.5× V3.1 decode.

Pass 43’s optimized Coast decoder profile recorded 116 MB process peak RSS and a 73 MiB observed increase for that decode run. The approximately 746 MiB Pass 42 high-water measurement was from an encode-only run. Pass 44 did not measure new peak memory. The corrected bounded search evaluated 186 completed files; its full source-to-file allocator time was not instrumented, so no runtime projection is made.

### 6. Failure analysis

The earlier truncated-prefix comparison made the Cat/Coast coarse base look worse than it was. With complete V3.1 prefixes, all-SKIP V4 files reproduce the V3.1 Compression pixels exactly at near-matched sizes. This shows that coarse procedural refinement was a principal missing low-rate representation stage. Current native detail adds only small MSE reductions under the 5% byte allowance. The mode map and header are small relative to transform payloads, so signaling is not the main cause.

The mode selection test was offline and does not prove a practical encoder path. Full V4-B decoder time is also dominated by processing large region maps and procedural/transform records; despite Pass 43’s batched interpolation, decode remains much slower than V3.1. The AVIF timing frontend remains the only four-axis baseline gap.

Visual checks show the Googlewiki selective reconstruction remains close to the coarse image, with text still soft. The Coast smooth-sky crop has broad banding in V4-A; no obvious new seam appears in the corrected sparse full-prefix crop. Visuals: `visuals/googlewiki_text_crop.png`, `visuals/coast_crop_comparison.png`, and `visuals/coast_mode_map_crop.png`.

### 7. Gate decisions

| Gate | Decision | Evidence |
|---|---|---|
| A — Competitive baseline | Incomplete | AVIF rate/MSE are measured, but AVIF timings include CLI startup and file I/O. Direct four-axis speed comparison is unavailable. |
| B — Strict-budget feasibility | Pass | All six selected outputs are valid and within 105% of their V3.1 target. |
| C — Useful original-resolution refinement | Fail | Restoring the complete coarse base improves RD, but native detail gives only 0.18%–1.01% MSE reduction at Compression rates. |
| D — Improvement over V3.1 | Fail | No roughly 10% lower MSE at matched rate on multiple images; measured improvements are only 0.18%–1.01%. |
| E — Runtime feasibility | Fail | Existing V4-B is 10.7×–31.7× slower to encode and 14.0×–89.8× slower to decode than V3.1. Sparse encode time is not integrated/measured. |
| F — Quality scaling | Pass with rate caveat | Original-resolution error falls from MSE 70.16 to 2.55 to 0.0384 as file size grows from 424 KB to 1.23 MB. This remains a weak overall RD curve. |
| G — AVIF niche | Not passed; exploratory signal | Coast selective V4 is +0.49% in bytes and 18.2% lower MSE than AVIF q0, but AVIF timings are not comparable and V4 encode is much slower. |

Fresh-process decoding matched the final pixel hashes for all six selected streams and preserved SKIP regions. No decoder or stream syntax change was made. The Pass 43 odd-dimension case was not repeated in Pass 44.

### 8. Next research recommendation

Stop tuning the current V4 allocator. Retain complete V3.1-style coarse refinement and investigate a cheaper native-resolution residual representation, gated first on an integrated complete encoder that stays within 105% of V3.1 Compression on Coast. The current offline selector is not an encoder result. Do not start Pass 45 automatically.


