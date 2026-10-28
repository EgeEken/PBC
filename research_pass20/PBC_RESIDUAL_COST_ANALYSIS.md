# Pass 20 residual cost analysis

Pass 17 encoding is dominated by its actual preset-0 EXTREME XZ compressor, not the 8x8 DCT. Removing EXTREME saves substantial CPU but incurs a substantial rate penalty. This is an independent tradeoff, not a free production improvement.

## Method and provenance

Start research/experiments at 0a2f1e80d7fd5aaed5acf683eea03aeb1a7b947e; unchanged pbc31_feature_histogram and frozen Pass 17. Googlewiki, Mario64 and Cat, q128/p10. Saved base/coefficient payloads are reused; rebuilt complete original files exactly match the saved Pass 17 SHA. No ordinary RD frontier is rerun for this analysis.

Python 3.11.16, NumPy 2.4.6, Numba 0.68.0, Pillow 11.3.0, SciPy 1.17.1, i9-12900H pinned logical P-core 12, numerical threads/FFT workers one. Full warmup precedes measurements. costs.json contains disjoint instrumented stages and separately measured uninstrumented complete calls; decode spans are not added to encode spans. All expanded controls, source/model/input identities and measured source hashes are retained. These profile/full calls are one-pass observations; only the entropy microcomparison has three alternating repetitions.

Stages separately measured: base encode, base decode inside encode, residual construction, color transform, DCT, quantization, coefficient reorder/copy, entropy and container; decoder base, entropy, coefficient view/dequantization, IDCT, color/crop and round/clip. All raw spans are in costs.json.

| Image | DCT ms | Instrumented XZ extreme s | Separate full encode / decode s | Zero q128 coefficients |
| --- | ---: | ---: | ---: | ---: |
| googlewiki.png | 17.8 | 1.977 | 2.471 / 0.086 | 95.41% |
| mario64.png | 31.6 | 1.553 | 1.907 / 0.120 | 98.68% |
| cat.jpg | 316.4 | 6.243 | 7.903 / 0.896 | 98.85% |

XZ is 81-87% of the sum of the disjoint encode spans. Base encode is about 0.227/0.238/0.544 s; base decode inside the encoder 0.031/0.039/0.192 s. Color/DCT/quantization/copy and final reconstruction remain real costs after entropy is accelerated. Do not add sampled span percentages to earlier speed measurements.

## Exactly two entropy settings

The frozen code uses `lzma.PRESET_EXTREME` alone, meaning **preset 0 plus the EXTREME flag**. Compare conventional **preset 0**, changing only that flag. Preset 6/default and other parameters were not tested; there is no parameter sweep. Identical signed int16 frequency-plane bytes, one XZ frame, three alternating compression/decompression measurements. Every compressed result expands to identical bytes. Table reports medians.

| Image | Extreme bytes / encode s | Normal-0 bytes / encode s | Enhancement size change |
| --- | ---: | ---: | ---: |
| googlewiki.png | 79,288 / 2.218 | 109,544 / 0.041 | +38.16% |
| mario64.png | 38,340 / 1.552 | 51,000 / 0.035 | +33.02% |
| cat.jpg | 193,688 / 5.942 | 275,048 / 0.272 | +42.01% |

Normal-0 is about 22-54x faster in compression but increases enhancement bytes by 33.02-42.01%. The total file increase is smaller because the base is unchanged. Entropy decompression medians change from 8.75/7.78/48.92 ms to 11.42/8.68/65.30 ms. It does not improve entropy decode time.

The same setting was also wrapped into actually decodable P17D files, with one warmed **complete** encode/decode call per case. These timings include fresh base encoding, base decoding inside the encoder, transforms, entropy and reconstruction; all independent RGB scoring and I/O are outside the timers. Output pixels exactly match frozen q128.

| Image | Normal-0 complete bytes | RGB MSE | Full encode / decode s |
| --- | ---: | ---: | ---: |
| googlewiki.png | 166,858 | 222.834 | 0.366 / 0.079 |
| mario64.png | 90,827 | 64.788 | 0.467 / 0.130 |
| cat.jpg | 321,213 | 91.126 | 1.708 / 0.800 |

These are full-call observations, not sums of microbenchmarks or repeated speed claims. Googlewiki still has a completed-file win against measured ordinary HQ and its 500k HQ control; Mario64 is 1.53% larger than ordinary HQ at lower MSE; Cat remains far too large. This entropy-only tradeoff is not a validated universal candidate. No production or frozen Pass 17 source was edited; a scoped hook is used only in a standalone research process and restored afterward.

## Coefficient usefulness and actual repacks

Frequency-plane zero counts do not assign compressed bytes to frequencies. Cat is nearly 99% zero yet still requires a dense roughly 51 MB coefficient byte array. Dense traversal, copying and entropy processing cost work even on zeros. Process peak-working-set observations are cumulative high-water marks, not isolated per-variant allocations or an allocator benchmark.

At p10/q128, low u+v<4 energy fractions are about 19.91% Googlewiki, 79.36% Mario64 and 77.64% Cat. Coefficient counts and original-band energy refer to this p10 residual, not the distinct HQ residual diagnostics from Pass 17.

Two frozen A-band diagnostics omit low or high bands, repack a complete ordinary P17D container with the retained coefficients and zeros elsewhere, and independently decode it. They preserve the full dense coefficient layout and extreme entropy setting. No omitted frequency is assumed free.

| Image | Full q128 bytes / MSE | Low-only bytes / MSE | High-only bytes / MSE |
| --- | ---: | ---: | ---: |
| googlewiki.png | 136,602 / 222.834 | 73,022 / 752.843 | 123,330 / 352.445 |
| mario64.png | 78,167 / 64.788 | 62,519 / 110.787 | 57,195 / 341.025 |
| cat.jpg | 239,853 / 91.126 | 168,749 / 116.045 | 125,629 / 294.405 |

Dropping high frequencies badly damages Googlewiki despite their differing energy/count distributions. Dropping low frequencies damages Mario64 and Cat. The low-only Mario64 and Cat repacks are useful additional diagnostic controls, but they use the old slow entropy setting and are not practical promotions. Counts, energy fractions and completed repack sizes answer different questions; their byte contributions are not additive.

The multiscale experiment uses normal-0 to test the latency path directly. Its rate penalty remains counted in every final file. The next report separates that entropy tradeoff from the loss caused by the coarse/fine frequency partition.
