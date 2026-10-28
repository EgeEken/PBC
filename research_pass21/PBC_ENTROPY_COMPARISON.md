# Pass 21 entropy comparison

Zero-run/zigzag + conventional XZ0 approaches the extreme-XZ size while making coefficient packing/compression 39.5-128.7x faster on these three q128 cases. Complete hybrid speedups are smaller and reported separately.

## Controlled coefficient measurement

Same saved coefficient values for all methods. Initial inputs are Googlewiki/Mario64/Cat q128; Pass20 saved NumPy arrays are verified against the corresponding frozen Pass17 enhancement before measurement. Source image/model/runtime identities, input file hashes, raw representation digests, exact coefficient counts and experiment-source hashes accompany the rows.

Python 3.11.16, NumPy 2.4.6, Numba 0.68.0, Pillow 11.3.0, SciPy 1.17.1; local i9-12900H, pinned logical P-core12, numerical threads one. Kernels and each actual coefficient path are warmed before three rotated repetitions. Dense extreme is preset0+EXTREME; dense normal and sparse both use preset0. One XZ frame each. No entropy-level sweep or optional compressor comparison.

Every timed method starts from the same 5D int16 coefficients. Dense packing includes reorder/tobytes; sparse packing includes nonzero counting, direct-order run/value generation, prefix and output conversion. Decompression includes bounded sparse prefix/body framing. Unpack/scatter is separately timed. Every result is compared with the exact dense expected vector outside its timers; repeated payload hashes match. Cold JIT/cache startup is outside warmed timings.

Table medians are per-stage; median total times are recorded separately in coefficients.csv/JSON because a sum of stage medians need not equal the median of per-call totals. Packing/unpacking are not omitted from gate speed/ratio calculations.

| Image | Method | Raw bytes | Compressed bytes | Pack ms | Compress ms | Decompress ms | Unpack ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| googlewiki.png | dense_extreme | 3,995,136 | 79,288 | 2.19 | 1808.10 | 6.10 | 0.02 |
| googlewiki.png | dense_normal | 3,995,136 | 109,544 | 2.26 | 34.25 | 9.48 | 0.02 |
| googlewiki.png | zero_runs | 186,192 | 80,656 | 3.61 | 13.02 | 4.49 | 1.05 |
| mario64.png | dense_extreme | 5,971,968 | 38,340 | 4.79 | 1341.09 | 6.38 | 0.02 |
| mario64.png | dense_normal | 5,971,968 | 51,000 | 4.25 | 31.22 | 7.38 | 0.02 |
| mario64.png | zero_runs | 81,327 | 34,816 | 4.90 | 5.55 | 1.85 | 1.02 |
| cat.jpg | dense_extreme | 51,151,104 | 193,688 | 76.88 | 5230.22 | 41.06 | 0.02 |
| cat.jpg | dense_normal | 51,151,104 | 275,048 | 78.61 | 261.23 | 55.29 | 0.02 |
| cat.jpg | zero_runs | 593,435 | 202,032 | 98.63 | 33.90 | 11.95 | 8.93 |

Gate comparisons include pack+compress and decompress+unpack:

| Image | Sparse payload change vs extreme | Total coefficient encode speedup | Total coefficient decode ratio |
| --- | ---: | ---: | ---: |
| googlewiki.png | +1.73% | 107.8x | 0.898x |
| mario64.png | -9.19% | 128.7x | 0.449x |
| cat.jpg | +4.31% | 39.5x | 0.508x |

Frozen gate: at least two cases <=10% payload increase, none >15%; >=5x faster total coefficient encoding on all three; no >2x coefficient decode regression. All three pass. These size differences are far smaller than Pass20 normal-dense's 33-42% payload increase. Mario64 improves payload size as well as runtime. Exact reconstruction holds for all representations.

This is not a claim that whole-image encoding is 40-129x faster. The base compressor, original-resolution residual/color/DCT, inverse transform and image reconstruction still run. Whole files preserve their original RD points with changed byte sizes; whole-call repetitions establish the actual speed benefit.

## Interpretation

XZ0 sees a roughly 81-593 KB stream of meaningful gaps/values rather than 4-51 MB of mostly zero int16 bytes. Explicit zero runs retain the long-range sparsity without requiring expensive extreme-mode matching. There is some cost to varints, count fields and mixed gap/value symbols, but measured final bytes remain close to or below the frozen result. This is measured behavior on the specified cases, not a universal sparse-coding guarantee.

No second format, custom arithmetic coder or compressor dependency is justified by this passing primary test. Process peak-working-set observations are cumulative, with verification buffers outside timers; they are not proof of a particular whole-codec peak-memory reduction. The direct packer's compact scratch requirement is explicit in the implementation.
