# Pass 22 reconstruction results

One sparse gather/IDCT/scatter method and one buffer-reuse method. No fused Numba kernel was implemented: compatible in-place NumPy/SciPy operations were sufficient to test the allocation hypothesis without changing color arithmetic. Frozen Pass21 remains the regression decoder; research_pass22.decoder.decode reads the identical P21S container through the unchanged split/read_coefficients functions.

The selected float32 block buffer is scaled in place and passed to SciPy with overwrite_x=True. Measured buffer sharing confirms that IDCT reuses it. The base remains uint8. The color-correction buffer receives base addition, np.rint and np.clip in place, then is converted to final uint8. The same float32 color matrix/matmul and rounding are preserved. Dense reuse-only and sparse-only flags exist for diagnostic ablations, not automatic routing.

Actual allocations and views:

- Sparse recovery still allocates the complete frequency-ordered int16 coefficient buffer (2 bytes per coefficient). q reshape/transpose shares that memory.
- Expression-level tracing confirms original astype(float32)*step temporarily materializes two 4-byte-per-coefficient arrays. Non-overwriting IDCT produces another.
- Sparse selection q!=0 temporarily allocates one byte per coefficient, then a three-component block mask. Boolean gathering allocates occupied int16 blocks; conversion allocates occupied float32 blocks.
- Sparse reconstruction still allocates a dense 4-byte-per-coefficient spatial buffer. Its transpose is a view, but the subsequent image reshape materializes another full padded float array; np.shares_memory is false. The crop is a view.
- Fixed color matmul allocates a source-size float32 RGB correction. Original base-to-float and add/rint/clip results require more source-size float arrays. The optimized base is uint8 and add/rint/clip reuse the correction; final uint8 output remains.
- Allocation estimates below are logical buffer sizes, not sums of simultaneous live bytes. The profile retains buffers for inspection and must not be used as a peak-memory measurement.

| Image | Dense int16 MB | One padded float buffer MB | Base float -> uint8 MB | Occupied float buffer MB | Occupancy comparison temporary MB |
| --- | ---: | ---: | ---: | ---: | ---: |
| googlewiki.png | 3.995 | 7.990 | 7.932 -> 1.983 | 2.409 | 1.998 |
| eiffel.png | 1.429 | 2.858 | 2.846 -> 0.712 | 1.893 | 0.714 |
| cat.jpg | 51.151 | 102.302 | 102.302 -> 25.576 | 25.277 | 25.576 |

MB above is decimal; memory peaks below use MiB. For Cat, each dense float image is 102.30 MB; the occupied transform batch is 25.28 MB. This avoids dense transform input/output work but retains dense scatter, layout conversion and RGB reconstruction.

Final whole-decoder timing: saved streams; i9-12900H logical P-core 12; Python 3.11.16, NumPy 2.4.6, Numba 0.68.0, SciPy 1.17.1, Pillow 11.3.0; one numerical thread/FFT worker. One full warmup per actual stream/variant then three rotated interleaved repetitions. Includes parsing, base decode, sparse recovery, transforms and uint8 output. I/O, hashing, logging and external scoring excluded. Ordinary HQ is its saved authentic practical-codec operating point, not matched-rate output.

| Image | HQ ms | Frozen P21S ms | Optimized ms | Full-call reduction | Speedup | Optimized / HQ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| googlewiki.png | 59.62 | 100.14 | 78.02 | 22.09% | 1.28x | 1.31x |
| eiffel.png | 36.43 | 39.72 | 37.72 | 5.04% | 1.05x | 1.04x |
| cat.jpg | 135.82 | 835.24 | 530.12 | 36.53% | 1.58x | 3.90x |

| Image | Variant | Mean ms | Median ms | Min-max ms | CV |
| --- | --- | ---: | ---: | ---: | ---: |
| googlewiki.png | ordinary_hq | 58.62 | 59.62 | 53.43-62.80 | 6.63% |
| googlewiki.png | reference | 98.17 | 100.14 | 90.41-103.95 | 5.81% |
| googlewiki.png | optimized | 75.91 | 78.02 | 68.55-81.15 | 7.06% |
| eiffel.png | ordinary_hq | 36.31 | 36.43 | 35.34-37.15 | 2.05% |
| eiffel.png | reference | 39.90 | 39.72 | 37.90-42.08 | 4.28% |
| eiffel.png | optimized | 37.77 | 37.72 | 36.60-38.99 | 2.59% |
| cat.jpg | ordinary_hq | 133.46 | 135.82 | 126.52-138.04 | 3.74% |
| cat.jpg | reference | 834.45 | 835.24 | 831.86-836.24 | 0.22% |
| cat.jpg | optimized | 541.95 | 530.12 | 521.02-574.72 | 4.33% |

All three complete rounds favor optimized over frozen on each image. Eiffel improvement is small. Final timings differ from Pass21 and the preliminary screen; use contemporary interleaved ratios, not historical absolute numbers. Three repetitions give local feasibility evidence, not HF performance or broad statistical certainty.

Stage medians in ms, measured separately with three warmed alternating disjoint profiles. Instrumentation, cache state and retained arrays change runtime; these numbers do not replace full-call medians and their median sums are not observed complete calls.

| Image | Variant | Base | Recovery | Selection + gather | Dequant | IDCT | Zero + scatter | Reorder copy | Color | Final output |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| googlewiki.png | reference | 40.34 | 6.69 | 0.00 | 4.42 | 21.77 | 0.00 | 5.93 | 6.61 | 9.91 |
| googlewiki.png | optimized | 34.70 | 6.14 | 1.64 | 0.87 | 5.73 | 2.13 | 5.67 | 4.64 | 5.02 |
| eiffel.png | reference | 23.74 | 1.65 | 0.00 | 1.71 | 8.01 | 0.00 | 2.02 | 2.08 | 3.22 |
| eiffel.png | optimized | 23.77 | 1.64 | 1.12 | 0.84 | 4.99 | 0.82 | 2.07 | 2.03 | 2.59 |
| cat.jpg | reference | 202.64 | 28.14 | 0.00 | 72.51 | 319.65 | 0.00 | 73.12 | 67.90 | 121.09 |
| cat.jpg | optimized | 157.58 | 24.67 | 25.02 | 10.53 | 58.56 | 18.94 | 76.54 | 71.34 | 76.02 |

Sparse selection/scatter cost is counted, not hidden. For Cat the profiled IDCT falls from about 320 to 59 ms, while selection/gather/scatter consumes about 44 ms. Dense layout/color/final work remains around 224 ms. Lower base-stage time includes avoiding float conversion and measurement variation; it is not evidence of a changed PBC base algorithm.

Isolated memory: two fresh processes per image/variant; both paths warmed on the same tiny decoder-only fixture, then one actual target decode. Record pre-call RSS and native Windows peak_wset; read peak before output hashing/scoring. No baseline decoded target or profile arrays coexist. Imported-runtime/warmup memory is included, with no claim that cumulative high-water marks isolate individual allocations. Here pre-call peaks equal pre-call RSS, so target peaks exceed them.

| Image | Frozen process peak MiB | Optimized peak MiB | Change | Frozen / optimized increase over pre-call RSS MiB |
| --- | ---: | ---: | ---: | ---: |
| googlewiki.png | 146.58 | 143.08 | -2.39% | 35.31 / 31.97 |
| eiffel.png | 123.73 | 125.19 | +1.18% | 12.51 / 14.23 |
| cat.jpg | 551.73 | 502.86 | -8.86% | 440.48 / 391.91 |

The small Eiffel peak regression is real in these fresh processes (roughly 1.5 MiB, 1.2% total); denser occupancy requires a relatively large gathered batch plus dense scatter. Cat saves roughly 49 MiB but still peaks near 503 MiB. Raw per-process RSS/peak measurements and both repeats remain in memory.json. No general memory-improvement claim is made across untested images.

Correctness: all six saved streams retain their byte hashes and exact frozen uint8 RGB dimensions/pixels/SHA256/original-resolution MSE, with repeat determinism on sparse-only, reuse-only and combined paths. 32 decoder-only synthetic containers cover 1x1, 8x8, 13x9, 35x37, steps 32/128, all-zero, single occupied component, mixed sparse/dense and signed int16 extremes at image edges. Every candidate matches the frozen decoder. 31 malformed/resource cases reject in reference and all three candidate paths, including counts, malformed/canonical varints, overflow, truncation, XZ trailing/concatenated/over-expanding content, dimensions/pixel/file caps and mismatched base dimensions. Two fresh processes decode saved Googlewiki/Flower with model loading and image encoding forbidden. No input validation was bypassed and no new compiled indexing kernel was introduced.

Existing bounded P21S/coefficient handling is reused unchanged. All experiments in this pass use only saved image streams or manually framed zero-patch fixture bases, never full-image encoding.

Expression-level allocation check (tracemalloc after inputs already exist; tracks NumPy-registered buffers, not every native allocation):

| Image | Original dequant traced peak MB | Original final expression peak MB | Reused final expression peak MB |
| --- | ---: | ---: | ---: |
| googlewiki.png | 15.982 | 15.864 | 1.983 |
| eiffel.png | 5.717 | 5.693 | 0.712 |
| cat.jpg | 204.606 | 204.605 | 25.576 |

The intact dequantization expression peaks at about twice its output buffer. Cat final add/rint/clip/cast peaks at 204.61 MB of new traced allocations in the original expression versus 25.58 MB with reuse (the final uint8 output). Inputs and pre-existing correction are excluded. These expression peaks do not add to process peaks and are not timing evidence.
