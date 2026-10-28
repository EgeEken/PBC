# Pass 20 multiscale completed-file results

Stop after the fixed three-image screen. The prototype retains some ordinary-PBC RD wins but fails to improve the frozen Pass 17 frontier. Nine of twelve files are strictly dominated by frozen full-DCT files; the remaining three are dominated by the actual A-band repacks. No six-image extension, repeated candidate timing or activation rule follows.

## Frozen representation

P20M is a research container, not a version or production default. A 40-byte header stores original width/height, base/payload lengths, coarse width/height, two float32 quantizer steps, block size 8, scale 4, fine band floor 4 and filter ID 1. Fixed color matrix and orthonormal DCT are inherited from Pass 17.

The original RGB residual is computed against an independently decoded ordinary Quality base. Analyze it channelwise using signed float32 PIL mode-F bicubic at ceil(width/4), ceil(height/4), reducing_gap=None. DCT/color-transform and quantize the reduced residual. Reconstruct those stored quantized coefficients, inverse-color-transform, then upsample mode-F bicubic to original dimensions. This exact decoder correction is subtracted from the original residual before the fine DCT. There is no double-counting of the same coarse correction.

Fine coding retains u+v>=4 (54/64 coefficients) at original-size 8x8 blocks. Coarse and retained fine coefficients are little-endian int16, frequency-plane ordered, concatenated into **one shared preset-0 XZ stream**. Dimensions determine the raw split; no tile frames or source image/model are needed at decode. Add decoded base + coarse + fine in float32, then np.rint, clip [0,255], uint8.

Two predeclared settings only: lean qc32/qf128, rich qc16/qf64. Two actual Quality base budgets: p10 and p5; p10 bytes match the frozen Pass 17 base, and p5 is verified identical across quantizer settings. Every timed full encode includes fresh base compression with internal MSE enabled. No patch budget is treated as a nested prefix. PLAN.md was fixed before measurements.

## Actual complete files

Every file includes its full base, shared coefficient stream and 40-byte header. Coarse XZ is a diagnostic compression of the coarse prefix alone; fine delta is final shared size minus that prefix size. They sum to actual enhancement bytes but **are not unique compressed-byte ownership**. They are not two separately stored entropy frames. Actual raw coarse/fine bytes and all hashes are in files.csv/screen.json.

| Image | Config | Final bytes | RGB MSE | Base bytes | Coarse XZ prefix | Fine XZ delta | Full encode / decode s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| googlewiki.png | p10 lean | 152,400 | 311.886 | 57,284 | 4,992 | 90,084 | 0.373 / 0.094 |
| googlewiki.png | p10 rich | 259,872 | 180.562 | 57,284 | 12,452 | 190,096 | 0.406 / 0.127 |
| googlewiki.png | p5 lean | 151,873 | 316.667 | 56,537 | 5,180 | 90,116 | 0.401 / 0.116 |
| googlewiki.png | p5 rich | 259,289 | 185.110 | 56,537 | 12,608 | 190,104 | 0.427 / 0.124 |
| mario64.png | p10 lean | 81,597 | 102.077 | 39,797 | 18,860 | 22,900 | 0.430 / 0.149 |
| mario64.png | p10 rich | 128,977 | 73.744 | 39,797 | 31,868 | 57,272 | 0.427 / 0.143 |
| mario64.png | p5 lean | 74,484 | 119.563 | 28,668 | 21,780 | 23,996 | 0.371 / 0.151 |
| mario64.png | p5 rich | 122,332 | 90.156 | 28,668 | 34,992 | 58,632 | 0.397 / 0.151 |
| cat.jpg | p10 lean | 214,043 | 185.264 | 46,135 | 58,960 | 108,908 | 2.207 / 1.190 |
| cat.jpg | p10 rich | 547,135 | 153.610 | 46,135 | 133,676 | 367,284 | 2.368 / 1.170 |
| cat.jpg | p5 lean | 211,142 | 185.732 | 42,510 | 59,648 | 108,944 | 2.134 / 1.197 |
| cat.jpg | p5 rich | 543,806 | 154.124 | 42,510 | 133,972 | 367,284 | 2.265 / 1.206 |

The p10 lean files strictly dominate default ordinary HQ on Googlewiki (-13.85% bytes/-56.34% MSE) and Mario64 (-8.79%/-16.35%). Cat is a tradeoff: +171.29% bytes/-36.80% MSE. This is not sufficient: both positive p10 lean files are themselves dominated by frozen full-DCT q128 (Googlewiki 136,602/222.834; Mario64 78,167/64.788).

Against stronger actually measured 5%-near ordinary controls:

| Image | Ordinary control | Ordinary bytes / MSE | p10 lean bytes / MSE | Signed rate gap | MSE change |
| --- | --- | ---: | ---: | ---: | ---: |
| googlewiki.png | high_quality_p13 | 147,416 / 727.003 | 152,400 / 311.886 | +3.38% | -57.10% |
| mario64.png | high_quality_p13 | 81,086 / 147.774 | 81,597 / 102.077 | +0.63% | -30.92% |

Googlewiki p10 lean also strictly dominates the available same-runtime ordinary 500k HQ control (214,902 bytes/MSE 439.912). Other same-runtime 500k controls were not available; no new dense ordinary frontier was generated. Full controls/config provenance, strongest close controls, actual dominance and conservative lowest-bracketing-chord estimates are in comparisons.json. No extrapolation or all-image BD-rate claim.

The smaller p5 base does not dominate p10 in either setting on any image. It saves little on Googlewiki and Cat, with worse quality. Mario64 lean falls to 74,484 bytes/MSE 119.563, near the ordinary-HQ quality, but its rate is 4.71% below frozen q128 with 84.55% higher MSE; this signed near-rate comparison favors the old quality point and is not called equal-rate. The low-only q128 diagnostic is actually smaller and better (62,519/110.787). Cat's p10/p5 lean files, 211-214 KB/MSE ~185, are dominated by the low-only diagnostic 168,749/116.045. All rich files have large rate penalties. All twelve are dominated by at least one actual full-or-band Pass 17 control, with the categories kept separate.

No signal justifies extending to Monet/Eiffel/Flower. Flower is untested in this pass, not assumed fixed. Cat remains the explicit photographic failure. This is not a universal encoder policy or an oracle-selected codec.

## Coarse/fine failure diagnosis

Coarse-only rounded/clipped RGB MSE for p10 lean is 860.308 Googlewiki, 151.006 Mario64 and 209.297 Cat. Adding fine data improves these to 311.886/102.077/185.264, but costs XZ deltas 90,084/22,900/108,908 bytes. Cat's coarse prefix alone is 58,960 bytes, before its 46,135-byte PBC base and header. Total fine cost largely cancels the coarse coding savings.

The coarse analysis/synthesis filter is not complementary to a fixed original 8x8 frequency cutoff. After subtracting the exact quantized coarse reconstruction, substantial **low 8x8-band information remains**:

| Image | Remaining energy in omitted bands | Coded p10 lean RGB MSE | Unencoded all-band reconstruction MSE |
| --- | ---: | ---: | ---: |
| googlewiki.png | 16.70% | 311.886 | 218.956 |
| mario64.png | 48.60% | 102.077 | 59.494 |
| cat.jpg | 66.44% | 185.264 | 84.932 |

The last column restores the omitted quantized bands as an explanatory reconstruction only. No all-band P20 file was encoded, no rate is assigned, and it is not a third screened setting or a demonstrated RD repair. This isolates a significant frequency-selection loss; merely counting low-band energy as already represented by the coarse layer was wrong. Small/high-frequency text coefficients also matter, as the separate A-band repacks show.

The new format adds only ten header bytes over Pass 17, so metadata is not the meaningful cause of the rate failure. Normal-0 entropy worsens coefficient compression, while fixed-band omission loses quality and dense fine coefficients retain most storage/traversal cost. Scale 4 adds only roughly 1/16 coarse samples but still stores 54/64 full-resolution frequencies, so raw coefficient count falls only about 9% before boundary padding. The fine DCT still computes every frequency before selection. This design is not a substantially sparse transform implementation.

## Practical runtime and memory

One warmed complete call per file, with loading/I/O/hashing/external scoring outside the timers. New HQ timing observations reproduce existing control file hashes; no new RD controls. These batches are **not three-repeat interleaved timing evidence**. No fine-grained speed percentages, universal overhead or release claims follow. Repetition was conditional on the RD gate, which failed.

| Image | Ordinary HQ encode / decode s | p10 lean encode / decode s | Observed encode ratio | Observed decode ratio |
| --- | ---: | ---: | ---: | ---: |
| googlewiki.png | 0.534 / 0.043 | 0.373 / 0.094 | 0.70x | 2.17x |
| mario64.png | 0.428 / 0.041 | 0.430 / 0.149 | 1.00x | 3.64x |
| cat.jpg | 0.642 / 0.134 | 2.207 / 1.190 | 3.44x | 8.88x |

Fast entropy makes encoding useful on the two small positive images in this screen, but Cat still exceeds 2x ordinary encode time and all decodes exceed the 10-15% overhead target. Coarse inverse DCT + original-size float upsample + fine inverse DCT/copies are additional decoder work. Cat's process high-water working set reaches roughly 1.47 GiB by the rich cases; this is cumulative within the screening process, not an independent per-file peak or fair cross-process memory comparison. There is no evidence that optimizing only Python bookkeeping would remove this architectural decoder work.

## Correctness and disposition

Twelve completed P20 files and nine A entropy/band P17 repacks pass independent size/SHA/pixel/MSE checks. Raw layer readback, signed int16 range +/-32,000, coarse-before-fine reconstruction, final round/clip, four repeated non-block-aligned fixtures, 16 malformed/magic rejection cases and two fresh file-only subprocess decodes with policy loading forbidden pass. Ordinary v0 and frozen P17 decoders reject P20 magic; production source/model remains byte-identical.

Do not promote or extend this prototype. No content activation is investigated because it failed the multi-image RD gate; an inexpensive routing policy is not established, and an offline best-of-files oracle is not reported as deployable. The ordinary practical codec and staged ZIP are unchanged. Strongest follow-up: preserve useful single-scale residual information while reducing dense/sparse coefficient entropy cost, and make any scale partition match its analysis/synthesis filters. The preset test shows large removable encode cost, but normal-0's 33-42% payload penalty is not the solution. No additional preset, scale or coding experiment was launched.
