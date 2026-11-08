# AVIF Baseline Results

## Inputs and operating points

The five development images were EXIF-transposed and converted to RGB. Hashes and dimensions for them and the three held-outs are in [SOURCE_MANIFEST.json](SOURCE_MANIFEST.json). Held-outs were not encoded or used for tuning.

The bounded unmodified qcolor grid was `5, 20, 35, 50, 65, 80, 90, 95` (qcolor 80/90/95 were added only for flower and coast). Point 1 was chosen nearest 0.27 bpp; point 2 nearest 0.50 bpp above point 1. Photo rates do not map to one universal q value: coast brackets the higher target at 0.105 and 0.648 bpp.

| Image | qcolor grid | File rates (bpp) |
|---|---|---|
| googlewiki.png | 5, 20, 35, 50, 65 | 0.193, 0.515, 0.794, 1.132, 1.546 |
| mario64.png | 5, 20, 35, 50, 65 | 0.054, 0.117, 0.199, 0.320, 0.490 |
| flower.jpg | 5, 20, 35, 50, 65, 80, 90, 95 | 0.014, 0.024, 0.039, 0.061, 0.101, 0.166, 0.550, 1.352 |
| cat.jpg | 5, 20, 35, 50, 65 | 0.017, 0.071, 0.236, 0.493, 0.870 |
| coast.jpg | 5, 20, 35, 50, 65, 80, 90, 95 | 0.007, 0.013, 0.022, 0.037, 0.064, 0.105, 0.648, 1.469 |

## Fixed settings

libaom 3.12.1 via libavif 1.3.0; single frame; AOM; 8-bit; YUV 4:2:0; full range; CICP `1/13/1`; speed 8; one thread; qalpha 0; EXIF/XMP/ICC omitted; SharpYUV off; automatic tiling. bpp uses complete AVIF bytes and original pixel count.

## Reference results

Each point had one warmup encode/decode pair and three fresh-process timed pairs. Times below are medians; raw samples and hashes are in `outputs/baseline/baseline_results.json`. Decode timing includes process startup and PNG output. RGB MSE is averaged over source pixels and channels after decoding at original dimensions.

| Image | Point | qcolor | AVIF bytes | bpp | RGB MSE | Encode median (s) | Decode median (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| googlewiki.png | 1 | 5 | 15,985 | 0.1935 | 747.650 | 0.081 | 0.094 |
| googlewiki.png | 2 | 20 | 42,573 | 0.5153 | 358.850 | 0.115 | 0.116 |
| mario64.png | 1 | 50 | 39,785 | 0.3198 | 62.629 | 0.117 | 0.236 |
| mario64.png | 2 | 65 | 60,978 | 0.4901 | 51.635 | 0.133 | 0.242 |
| flower.jpg | 1 | 80 | 176,669 | 0.1658 | 3.421 | 0.556 | 1.770 |
| flower.jpg | 2 | 90 | 585,814 | 0.5497 | 2.489 | 0.729 | 2.384 |
| cat.jpg | 1 | 35 | 251,357 | 0.2359 | 51.484 | 0.624 | 1.102 |
| cat.jpg | 2 | 50 | 525,046 | 0.4927 | 23.852 | 0.808 | 1.201 |
| coast.jpg | 1 | 80 | 157,638 | 0.1050 | 3.439 | 0.591 | 0.976 |
| coast.jpg | 2 | 90 | 973,289 | 0.6484 | 2.317 | 0.850 | 2.487 |

## Equivalence and compatibility

All ten speed-8 points were byte-identical between the unmodified reference and the instrumentation-off research build. The final instrumented build preserved output bytes on all five speed-8 profile cases and both speed-6 cases. Those seven files decoded in fresh `avifdec` processes and Pillow AVIF frontend processes; decoded RGB and MSE matched exactly between the two frontends. No private syntax or decoder change was added.

## Pass 34 historical sanity check

Pass 34 used the same coast source with Pillow 11.3.0/libavif 1.3.0/libaom 3.12.1 at speed 8 and reported q43 at 43,855 bytes/MSE 6.30 and q50 at 55,075 bytes/MSE 5.41. The new direct `avifenc` calibration's qcolor 50 point is 55,123 bytes/MSE 5.3985: 48 bytes larger (0.087%) and 0.0115 lower MSE. This close same-rate result is a historical sanity check. The frontend and quality parameter semantics differ, so AVIF bytes are not expected to match.
