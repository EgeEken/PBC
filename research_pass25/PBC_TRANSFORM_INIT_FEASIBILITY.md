# Pass 25 transform initialization feasibility

Initialization-only feasibility passes on text and photography, but not consistently across all four inputs. This is a representation screen, not proof of a complete hybrid advantage. Source starts at 4b41f1794d005241444e825c2b6ce125ed135e7f; unchanged practical control pbc31_feature_histogram at 1c9c4016fdb213d7d16663542d268c7c3df4ed9b.

Ordinary preprocessing is unchanged: EXIF-transposed RGB, conversion to YCbCr before normal PIL bicubic downsampling at the 250k cap. Ordinary initialization begins at rounded component means and selects/applies the normal 3 full-image grids. Fresh initialization selection and production entropy exactly reproduce saved Quality and HQ initialization-only prefixes for all 4 images. No ordinary iterative corrections are included here.

| Image | Working size | Ordinary init bytes | Init / full Q bytes | Init / full HQ bytes | Init working YCbCr MSE | Init source RGB MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| googlewiki.png | 501x499 | 11,771 | 14.28% | 6.65% | 254.518 | 1559.369 |
| monet.jpg | 490x511 | 7,475 | 13.88% | 5.74% | 54.652 | 203.502 |
| cat.jpg | 612x408 | 3,968 | 6.27% | 5.03% | 93.970 | 581.717 |
| flower.jpg | 612x408 | 4,065 | 14.03% | 8.82% | 82.387 | 276.341 |

These ratios compare separately compressed initialization-only and full files. They are not additive ownership of bytes inside the full LZMA stream. Raw patch-bit counts are recorded separately; final bytes are the rate metric. Initial errors include actual signed reconstruction, clipping/color conversion and source-size bicubic resize.

One transform design: each native working component minus its rounded global mean; edge padding; float32 orthonormal 8x8 type-II SciPy DCT; nearest integer signed int16 coefficients at the fixed steps 64/128/256. Exact Pass21 R21V zero-gap/zigzag frequency-plane serialization and XZ0, with no entropy tuning. No original-resolution transform, back-projection, chroma subsampling or second color transform.

P25I uses a 24-byte little-endian <4sHHHHB3BfI> header: magic, working width/height, original width/height, color ID, 3 byte means, float32 step and entropy payload length. The XZ payload contains bounded R21V counts and coefficients. Decode reconstructs float32 IDCT at working resolution, adds means, rounds to signed int16 without color-range clipping, then normal PBC output conversion/resize. This exact stored-data canvas starts refinement in the later test. No encoder weights or source image are needed.

| Image | Representation / step | Final init bytes | Source RGB MSE | Working component MSE | Working RGB MSE |
| --- | --- | ---: | ---: | ---: | ---: |
| googlewiki.png | ordinary | 11,771 | 1559.369 | 254.518 | 689.719 |
| googlewiki.png | transform / 64 | 38,224 | 875.780 | 69.572 | 181.168 |
| googlewiki.png | transform / 128 | 14,132 | 1148.740 | 155.214 | 411.411 |
| googlewiki.png | transform / 256 | 3,676 | 1579.453 | 280.361 | 761.735 |
| googlewiki.png | jpeg / 1 | 4,196 | 1600.277 | 285.296 | 784.219 |
| monet.jpg | ordinary | 7,475 | 203.502 | 54.652 | 151.709 |
| monet.jpg | transform / 64 | 13,940 | 149.255 | 36.752 | 107.191 |
| monet.jpg | transform / 128 | 5,272 | 252.540 | 72.464 | 208.948 |
| monet.jpg | transform / 256 | 1,892 | 436.887 | 137.826 | 394.800 |
| monet.jpg | jpeg / 1 | 2,249 | 452.326 | 146.317 | 411.680 |
| cat.jpg | ordinary | 3,968 | 581.717 | 93.970 | 275.318 |
| cat.jpg | transform / 64 | 7,280 | 339.622 | 21.409 | 61.643 |
| cat.jpg | transform / 128 | 3,388 | 447.120 | 58.706 | 171.115 |
| cat.jpg | transform / 256 | 1,480 | 633.352 | 119.936 | 346.761 |
| cat.jpg | jpeg / 1 | 2,600 | 563.114 | 92.335 | 267.263 |
| flower.jpg | ordinary | 4,065 | 276.341 | 82.387 | 246.033 |
| flower.jpg | transform / 64 | 12,700 | 74.156 | 20.062 | 56.177 |
| flower.jpg | transform / 128 | 6,668 | 160.654 | 52.107 | 144.182 |
| flower.jpg | transform / 256 | 2,808 | 318.725 | 116.770 | 310.329 |
| flower.jpg | jpeg / 1 | 2,999 | 419.469 | 145.106 | 410.398 |

Working component error for ordinary/DCT is YCbCr error against the clipped decoded working canvas. JPEG component error re-expresses its decoded RGB in YCbCr; working RGB MSE provides the common display-domain comparison. Working canvas differences from ordinary initialization are in feasibility.json.

The single initial JPEG reference is quality 1, 4:2:0, optimize=True,progressive=False, chosen before transform results for the low 4-12k initialization budgets. It is a normal JPEG at working resolution plus a 14-byte standard COM segment carrying original dimensions. Independent decode reads those dimensions and uses the same final resize. Its actual rates are 26-70% below ordinary init and are not close-rate controls; no matched-rate JPEG claim follows from this point. A later separate fixed q75 reference belongs to the completed-codec stage, not an initial quality sweep.

| Image | Actual qualifying transform steps | Estimated MSE change at ordinary init bytes | Measured bracket |
| --- | --- | ---: | --- |
| googlewiki.png | 256 | -20.10% | 256 -> 128 |
| monet.jpg | none | +11.20% | 128 -> 64 |
| cat.jpg | 128 | -25.89% | 128 -> 64 |
| flower.jpg | none | -3.29% | 256 -> 128 |

Actual gate conditions are 10% lower MSE with at most 5% extra bytes, or 10% fewer bytes with at most 2% extra MSE. Text step 256 saves 68.77% bytes at 1.29% extra source MSE; Cat step 128 saves 14.62% bytes and reduces MSE 23.14%. Sparse interpolation at ordinary initialization rate estimates -20.10% Googlewiki, -25.89% Cat, +11.20% Monet and -3.29% Flower. Interpolation is between measured quantizers, not an achievable equal-rate file. Cat/Flower are one coarse photo class, so text plus Cat satisfy the two-class gate. Monet and Flower limitations remain explicit.

Runtime and reconstruction buffers, one warmup plus one timed creation/full RGB decode per fixed setting, common production prepare excluded:

| Image | Kind / step | Encode ms | Decode ms | Coefficient slots / nonzeros | Signed canvas MB | One float coeff buffer MB | Traced decode peak MB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| googlewiki.png | ordinary | 86.23 | 20.75 | - | 1.500 | - | 3.751 |
| googlewiki.png | transform / 64 | 21.02 | 27.57 | 762048 / 47754 | 1.500 | 3.048 | 13.756 |
| googlewiki.png | transform / 128 | 17.55 | 26.14 | 762048 / 17609 | 1.500 | 3.048 | 13.672 |
| googlewiki.png | transform / 256 | 15.20 | 28.62 | 762048 / 5833 | 1.500 | 3.048 | 13.638 |
| googlewiki.png | jpeg / 1 | 1.83 | 9.30 | - | 1.500 | - | 0.003 |
| monet.jpg | ordinary | 66.41 | 21.68 | - | 1.502 | - | 3.757 |
| monet.jpg | transform / 64 | 19.56 | 33.15 | 761856 / 18727 | 1.502 | 3.047 | 13.681 |
| monet.jpg | transform / 128 | 19.49 | 33.36 | 761856 / 8618 | 1.502 | 3.047 | 13.652 |
| monet.jpg | transform / 256 | 18.23 | 32.97 | 761856 / 3708 | 1.502 | 3.047 | 13.639 |
| monet.jpg | jpeg / 1 | 2.20 | 13.12 | - | 1.502 | - | 0.003 |
| cat.jpg | ordinary | 90.78 | 106.83 | - | 1.498 | - | 3.746 |
| cat.jpg | transform / 64 | 19.98 | 104.57 | 753984 / 18912 | 1.498 | 3.016 | 13.579 |
| cat.jpg | transform / 128 | 16.69 | 114.87 | 753984 / 9254 | 1.498 | 3.016 | 13.556 |
| cat.jpg | transform / 256 | 14.82 | 119.52 | 753984 / 5619 | 1.498 | 3.016 | 13.547 |
| cat.jpg | jpeg / 1 | 2.10 | 88.92 | - | 1.498 | - | 0.003 |
| flower.jpg | ordinary | 70.17 | 98.94 | - | 1.498 | - | 3.746 |
| flower.jpg | transform / 64 | 17.94 | 111.03 | 753984 / 16701 | 1.498 | 3.016 | 13.580 |
| flower.jpg | transform / 128 | 18.92 | 113.93 | 753984 / 9047 | 1.498 | 3.016 | 13.559 |
| flower.jpg | transform / 256 | 18.34 | 117.11 | 753984 / 4161 | 1.498 | 3.016 | 13.545 |
| flower.jpg | jpeg / 1 | 1.84 | 90.28 | - | 1.498 | - | 0.003 |

Float buffers are working-size (around 3 MB each), while full photographic RGB output is 25.58 MB. A separate traced decode measures Python/NumPy allocations, not all native PIL storage; isolated whole-process measurements are in the completed results. All candidate steps retain a plausible working-size decode path at this stage. Initializer timings exclude shared prepare and are not whole-codec timings.

The frozen integrated base is step 128 for every image and preset. It was declared before this screen, with no image-specific quantizer/budget table. A different step might change results, but the pass does not search for a rescue.
