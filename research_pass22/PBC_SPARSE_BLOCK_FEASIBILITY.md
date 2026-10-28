# Pass 22 block feasibility

Saved P21S q128 Googlewiki/Cat and q32 Eiffel; no encodes or reference regeneration. Start a0824ff1a319f6cadcfbc16d146f666ec9f91832. Full input/source identities are in provenance.json and occupancy.json. Components are the fixed orthonormal transform components, not RGB channels. Each component block contains 64 coefficients.

| Image | Spatial blocks | Component blocks | Empty component blocks | Empty spatial blocks | Pixels covered by any occupied component |
| --- | ---: | ---: | ---: | ---: | ---: |
| googlewiki.png | 10,404 | 31,212 | 69.84% | 5,107 (49.09%) | 50.78% |
| eiffel.png | 3,721 | 11,163 | 33.77% | 188 (5.05%) | 94.93% |
| cat.jpg | 133,206 | 399,618 | 75.29% | 50,383 (37.82%) | 62.18% |

Coverage clips edge blocks to the true source dimensions. Component emptiness estimates the share of64-point transforms that can be skipped; it does not predict whole-decoder savings. Spatial coverage remains high on Eiffel, and output storage/base decoding still covers every source pixel.

| Image | Empty | One nonzero | 2-4 nonzeros | More than 4 | Occupied components 0 / 1 / 2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| googlewiki.png | 21,800 | 928 | 1,971 | 6,513 | 5166 / 3762 / 484 |
| eiffel.png | 3,770 | 4,091 | 2,517 | 785 | 3076 / 2309 / 2008 |
| cat.jpg | 300,879 | 35,071 | 41,374 | 22,294 | 78888 / 16469 / 3382 |

The frozen gate required at least 20% empty spatial blocks on two of three cases. Googlewiki and Cat pass; Eiffel is the low-sparsity control. One implementation gathers nonzero component blocks, runs the same SciPy float32 orthonormal IDCT on the compact batch, and scatters into a zero-filled dense spatial buffer. The representation, quantizer and coefficients are unchanged. There is no per-image routing rule.

Spatial distribution: occupied fractions in a 4x4 grid of block-coordinate regions (equal block-count splits; exact fractions plus per-component maps are in occupancy.json).

googlewiki.png

| Region row | Col1 | Col2 | Col3 | Col4 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 42.3% | 24.3% | 27.1% | 38.6% |
| 2 | 62.0% | 67.0% | 45.7% | 55.7% |
| 3 | 59.5% | 63.7% | 45.9% | 53.8% |
| 4 | 62.6% | 68.6% | 39.4% | 58.6% |

eiffel.png

| Region row | Col1 | Col2 | Col3 | Col4 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 92.2% | 93.8% | 85.0% | 92.1% |
| 2 | 97.1% | 98.7% | 96.0% | 97.3% |
| 3 | 92.1% | 96.4% | 93.8% | 97.3% |
| 4 | 92.9% | 99.6% | 99.1% | 97.3% |

cat.jpg

| Region row | Col1 | Col2 | Col3 | Col4 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.7% | 13.8% | 5.8% | 0.1% |
| 2 | 59.6% | 85.9% | 75.9% | 50.6% |
| 3 | 94.3% | 89.8% | 73.8% | 88.6% |
| 4 | 91.8% | 92.2% | 89.0% | 85.4% |

Cat has near-empty top regions but substantial occupancy below. Googlewiki emptiness is spread across the image. Eiffel is occupied nearly everywhere. Coefficient sparsity alone would not have established any of these facts.

Three warmed rotated full-call diagnostic repetitions, before the separate final HQ comparison:

| Image | Frozen ms | Sparse only ms | Reuse only ms | Combined ms |
| --- | ---: | ---: | ---: | ---: |
| googlewiki.png | 82.91 | 71.66 | 75.52 | 64.65 |
| eiffel.png | 36.10 | 35.54 | 32.76 | 32.14 |
| cat.jpg | 895.55 | 583.44 | 857.79 | 578.15 |

The sparse path materially improves Googlewiki/Cat full calls, with little standalone benefit on Eiffel. Compact contiguous transforms also change memory locality; do not attribute all IDCT savings solely to the empty-block percentage. Gathering/scattering overhead and dense output remain. No second selection method or custom transform was attempted. Final comparative numbers use the separately interleaved benchmark in the reconstruction report.
