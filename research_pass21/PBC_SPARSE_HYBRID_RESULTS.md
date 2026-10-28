# Pass 21 sparse hybrid completed files

Sparse storage retains all four successful Pass17 ordinary-HQ dominance points with identical RGB pixels and much faster encoding. It solves the extreme-XZ encode bottleneck on this bounded screen. Full decoding remains too expensive for a general-purpose replacement; Cat/Flower remain RD failures.

## Completed files and ordinary comparisons

P21S is a research container, not a production version. A 26-byte `<4sIIIIfBB>` header stores magic, original width/height, base/payload lengths, float32 quantization step, block size8 and sparse version1. The XZ-framed R21V coefficient stream stores total/nonzero counts and all gaps/signed values. Original image dimensions determine the full edge-padded coefficient geometry. The fixed color matrix, DCT norm, base p10 and final round/clip are unchanged from Pass17. P17D header was30 bytes; its tile/mode fields are not needed here.

Encoder freshly compresses the same ordinary Quality p10 base, independently decodes it, computes the original residual/color/DCT at the **original measured quantizer**, packs every coefficient and uses conventional XZ0. Decoder recovers the base and exact complete coefficient array, then calls the original inverse/reconstruction routine. No coefficient dropping, multiscale, base reallocation, model training or quantizer sweep.

Initial three complete files preserve the coefficient gate's size behavior and exact pixels, so the declared extension ran at Monet q64, Eiffel q32 and Flower q128. One complete deterministic encode/decode per case establishes RD; only the two promising repeated cases receive the timing protocol below. Frozen files/bases/control rows are reused and byte-checked; no expensive ordinary frontier is regenerated.

| Image | q step | Frozen P17 bytes | Dense fast XZ bytes | Sparse complete bytes | RGB MSE (all three identical) | Sparse enhancement bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| googlewiki.png | 128 | 136,602 | 166,858 | 137,966 | 222.834 | 80,656 |
| mario64.png | 128 | 78,167 | 90,827 | 74,639 | 64.788 | 34,816 |
| cat.jpg | 128 | 239,853 | 321,213 | 248,193 | 91.126 | 202,032 |
| monet.jpg | 64 | 96,146 | 117,102 | 95,402 | 50.169 | 52,444 |
| eiffel.png | 32 | 59,265 | 63,637 | 58,145 | 10.568 | 14,564 |
| flower.jpg | 128 | 80,752 | 98,584 | 72,804 | 22.832 | 51,080 |

All six sparse files match frozen quantized arrays and decoded pixel SHA exactly. The complete file includes base + compressed enhancement + 26-byte header; bpp uses original pixels. Raw sparse bytes, enhancement fraction, full call walls and cumulative memory observations are in files.csv. Four files shrink versus frozen P17; Googlewiki grows 1.00% total, Cat 3.48%. Flower shrinks 9.84% total but remains above ordinary rates.

Actual ordinary HQ comparisons, all prior matching-runtime controls:

| Image | Ordinary HQ bytes / MSE | Sparse bytes / MSE | Signed rate change | MSE change |
| --- | ---: | ---: | ---: | ---: |
| googlewiki.png | 176,905 / 714.432 | 137,966 / 222.834 | -22.01% | -68.81% |
| mario64.png | 89,460 / 122.026 | 74,639 / 64.788 | -16.57% | -46.91% |
| cat.jpg | 78,898 / 293.134 | 248,193 / 91.126 | +214.57% | -68.91% |
| monet.jpg | 130,159 / 53.976 | 95,402 / 50.169 | -26.70% | -7.05% |
| eiffel.png | 85,874 / 15.410 | 58,145 / 10.568 | -32.29% | -31.42% |
| flower.jpg | 46,094 / 34.841 | 72,804 / 22.832 | +57.95% | -34.47% |

Monet, Googlewiki, Mario64 and Eiffel strictly dominate ordinary HQ. Cat and Flower exchange far more bytes for quality and are not RD wins. Their sparse enhancement payload changes are +4.31%/-13.46% versus frozen, so the storage method does not create a large new photographic penalty; it also does not solve the architecture's existing photo rate problem.

Strongest actually measured ordinary points within5% of sparse file size:

| Image | Ordinary point | Ordinary bytes / MSE | Signed sparse rate gap | Sparse MSE change |
| --- | --- | ---: | ---: | ---: |
| mario64.png | high_quality_p10 | 74,041 / 187.768 | +0.81% | -65.50% |
| monet.jpg | high_quality_p8 | 94,410 / 82.063 | +1.05% | -38.87% |

Googlewiki and Eiffel have no such actual 5%-near ordinary control; do not call their HQ comparison equal-rate. Googlewiki also strictly dominates the available same-runtime 500k HQ point (214,902 bytes/MSE439.912). comparisons.json retains all ordinary controls, signed close matches and conservative bracket estimates separately; no extrapolation or continuous frontier claim. The original Pass17 controls at the same quantizer remain the cleanest test of the new storage method because quality is identical.

## Repeated full encode/decode calls

Three warmed rotated repetitions per Googlewiki q128 and Eiffel q32, one pinned CPU/numerical thread. Timers include fresh base encode, base decode inside encoder, full residual/DCT, sparse packing/entropy and final decoding. Image loading, hash checks, external scoring and disk I/O are outside timers. Dense-normal uses a scoped research hook restored afterward; frozen source is untouched. Streams and pixels match their recorded controls on every repetition. Ordinary HQ is a different actual RD operating point; frozen/fast/sparse share exactly the same p10 base and quantizer.

| Image | Ordinary HQ encode / decode s | Frozen P17 encode / decode s | Dense-fast encode / decode s | Sparse encode / decode s |
| --- | ---: | ---: | ---: | ---: |
| googlewiki.png | 0.6135 / 0.0485 | 2.5143 / 0.0850 | 0.3860 / 0.0901 | 0.3738 / 0.0829 |
| eiffel.png | 0.5851 / 0.0313 | 0.6068 / 0.0503 | 0.2405 / 0.0465 | 0.2206 / 0.0453 |

Sparse full encoding is 6.73x faster than frozen on Googlewiki and 2.75x faster on Eiffel. Against ordinary HQ it takes about 0.61x/0.38x the encode time at the named lower-rate/better-quality points. These do not combine with earlier PBC3.0 or Pass16 speed percentages.

Full sparse decode medians are about 1.71x/1.45x ordinary HQ and broadly similar to frozen P17. Do not claim the coefficient microdecode gains transferred proportionally. Googlewiki sparse decode CV is16.35%; frozen Eiffel decode CV is12.95%, so small full-decode differences are uncertain. Other timing CVs and all raw repeats remain in timings.json. No mean across datasets/protocols is used.

Other four points are **one-pass** full observations against fresh byte-identical ordinary-HQ controls, not repeated speed evidence:

| Image | Ordinary HQ encode / decode s | Sparse encode / decode s | Encode ratio | Decode ratio |
| --- | ---: | ---: | ---: | ---: |
| mario64.png | 0.428 / 0.040 | 0.321 / 0.096 | 0.75x | 2.42x |
| cat.jpg | 0.617 / 0.112 | 1.348 / 0.683 | 2.19x | 6.08x |
| monet.jpg | 0.543 / 0.057 | 0.292 / 0.086 | 0.54x | 1.52x |
| flower.jpg | 0.716 / 0.125 | 1.199 / 0.737 | 1.68x | 5.87x |

Cat/Flower remain expensive in original-resolution transform/reconstruction. The one-pass decode observations are around6x ordinary HQ. Their existing low-rate/photo failures are explicit, and no default routing rule is inferred from the six examples.

## Remaining decoder work

One warmed, disjoint stage profile per case; its sums are separate from complete-call medians and are not substituted for those medians. All actual stage walls are in decode_profile.json.

| Image | Base -> float ms | Decompress ms | Unpack ms | IDCT ms | Other reconstruction ms | Recovery share of profile |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| googlewiki.png | 31.65 | 4.35 | 0.93 | 15.51 | 17.02 | 7.6% |
| eiffel.png | 16.15 | 0.97 | 0.32 | 5.69 | 6.02 | 4.4% |
| cat.jpg | 148.41 | 11.02 | 8.49 | 214.02 | 226.73 | 3.2% |

Coefficient recovery is only about3-8% of the current profiled decoder. Base decode + dequant/copy + inverse DCT + color + final round/clip total about64/28/589 ms when coefficient recovery is excluded, for Googlewiki/Eiffel/Cat. These are observations of this unchanged implementation, not theoretical lower bounds for every possible decoder. Profile totals differ from full-call medians because measurement/cache/allocation conditions differ; no production latency is predicted by adding component medians.

Eliminating the remaining sparse recovery entirely cannot remove the several hundred milliseconds of large-photo reconstruction. Full original-size coefficient/float/output buffers still exist, even though packing scratch and compressed input are much smaller. Memory fields are cumulative process high-water marks with validation/control buffers and are not isolated per-variant peak comparisons. No major whole-codec memory improvement is claimed without isolation.

## Disposition

Preserve this passing sparse storage prototype for bounded R&D. It substantially reduces entropy overhead without the Pass20 33-42% payload penalty, needs no second sparse format and retains four actual RD successes. It is **not** a practical universal replacement: observed decode overhead misses the10-15% target, and large photos also miss encode/rate targets.

Activation was conditional on completed-file **and runtime** feasibility; the general-purpose runtime gate is not met, so no classifier/routing policy was tested. Source pixel count and downsampling ratio plausibly flag reconstruction cost, but they are not established RD predictors. No image-name switch, dual full-image encode or offline oracle is presented as deployable performance.

The strongest next question is whether nonzero-block-aware inverse reconstruction and fewer full-image float copies can reduce decoder cost while preserving these exact pixels. This is a proposed bounded decode feasibility study, not a new experiment launched here or a promise that entropy changes solve all residual coding. If it cannot approach ordinary latency, keep this as an opt-in research codec and change the general-purpose direction rather than tune more varint formats. Production/v0/model/defaults/staged ZIP are unchanged.
