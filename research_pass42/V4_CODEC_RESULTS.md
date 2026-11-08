# PBC V4 Pass 42 results

## Scope

This is a first isolated V4-A prototype, not a release candidate. It uses the fixed 16×16 regional grid, compact 64-color initialization, protected SKIP, procedural residual surfaces, full 8×8 RGB DCT refinement, and one shared XZ payload. Measurements use the three specified development images; reserved validation images were not opened. The frozen V3.1 source commit and model hash are recorded in the raw results.

## Quality and rate

The complete measurements follow. Rates are actual file bytes divided by source pixels. No interpolation or extrapolation is used. Decode times marked * are fresh-process measurements that include interpreter startup; JPEG decode is measured in-process through Pillow.

### googlewiki.png

| Codec point | Bpp | RGB MSE | Encode s | Decode s* |
|---|---:|---:|---:|---:|
| V4-D low | 0.8853 | 24.835 | 2.70 | 0.48 |
| V4-D intermediate | 1.2947 | 15.618 | 5.03 | 0.62 |
| V4-D high | 7.7271 | 0.030 | 7.75 | 1.91 |
| Frozen V3.1 compression | 0.3507 | 1005.022 | 0.17 | 0.75 |
| Frozen V3.1 balanced | 0.6523 | 845.441 | 0.19 | 0.76 |
| Frozen V3.1 quality | 0.9978 | 781.029 | 0.41 | 0.79 |
| Frozen V3.1 high_quality | 2.1412 | 714.432 | 0.42 | 0.79 |
| JPEG q15 | 0.6379 | 532.740 | 0.02 | 0.004 |
| JPEG q30 | 1.0122 | 384.284 | 0.00 | 0.004 |
| JPEG q45 | 1.2874 | 325.450 | 0.01 | 0.004 |
| JPEG q60 | 1.5492 | 287.429 | 0.01 | 0.004 |
| JPEG q75 | 1.9756 | 246.727 | 0.01 | 0.005 |
| JPEG q90 | 2.9859 | 204.669 | 0.01 | 0.006 |

### cat.jpg

| Codec point | Bpp | RGB MSE | Encode s | Decode s* |
|---|---:|---:|---:|---:|
| V4-D low | 0.7313 | 167.367 | 21.63 | 6.20 |
| V4-D intermediate | 5.9433 | 10.111 | 86.42 | 18.95 |
| V4-D high | 13.9967 | 0.080 | 143.97 | 30.04 |
| Frozen V3.1 compression | 0.0066 | 433.812 | 0.39 | 1.03 |
| Frozen V3.1 balanced | 0.0224 | 352.336 | 0.52 | 1.17 |
| Frozen V3.1 quality | 0.0594 | 299.192 | 0.43 | 1.23 |
| Frozen V3.1 high_quality | 0.0740 | 293.134 | 0.47 | 1.25 |
| JPEG q15 | 0.3273 | 70.720 | 0.04 | 0.049 |
| JPEG q30 | 0.5636 | 38.507 | 0.04 | 0.052 |
| JPEG q45 | 0.7272 | 27.389 | 0.06 | 0.056 |
| JPEG q60 | 0.8764 | 21.383 | 0.05 | 0.053 |
| JPEG q75 | 1.1251 | 14.878 | 0.06 | 0.058 |
| JPEG q90 | 1.9375 | 7.385 | 0.08 | 0.061 |

### coast.jpg

| Codec point | Bpp | RGB MSE | Encode s | Decode s* |
|---|---:|---:|---:|---:|
| V4-D low | 0.0925 | 18.692 | 17.48 | 3.97 |
| V4-D intermediate | 0.4589 | 7.162 | 34.58 | 11.94 |
| V4-D high | 9.6694 | 0.069 | 192.89 | 39.15 |
| Frozen V3.1 compression | 0.0043 | 73.134 | 0.65 | 1.14 |
| Frozen V3.1 balanced | 0.0078 | 48.220 | 0.62 | 1.18 |
| Frozen V3.1 quality | 0.0125 | 40.884 | 0.48 | 1.20 |
| Frozen V3.1 high_quality | 0.0262 | 23.557 | 0.55 | 1.24 |
| JPEG q15 | 0.0728 | 17.790 | 0.05 | 0.064 |
| JPEG q30 | 0.0989 | 7.872 | 0.05 | 0.069 |
| JPEG q45 | 0.1222 | 5.656 | 0.05 | 0.073 |
| JPEG q60 | 0.1490 | 4.745 | 0.05 | 0.067 |
| JPEG q75 | 0.2235 | 3.785 | 0.05 | 0.070 |
| JPEG q90 | 0.6445 | 2.699 | 0.07 | 0.075 |

## Findings

- V4-D improves monotonically from low through high on all three images. The highest setting approaches source fidelity, so the V3 working-resolution quality ceiling is not inherent to this representation. It does so at very high rates: 7.73 bpp on Googlewiki, 14.00 bpp on cat, and 9.67 bpp on coast.
- It does not preserve V3.1’s extreme-low-rate behavior. On coast, V3.1 compression is about 0.0043 bpp and MSE 73.1; V4-D low is 0.0925 bpp and MSE 18.7. This is a quality gain at roughly 21× the rate.
- JPEG remains a strong control. Googlewiki V4-D intermediate is 1.295 bpp / MSE 15.62, while JPEG q45 is 1.287 bpp / MSE 325.45, a clear rate-matched V4 win on this graphics image. Cat’s V4-D low point is 0.731 bpp / MSE 167.37, while JPEG q45 is 0.727 bpp / MSE 27.39, so JPEG dominates there. On coast, JPEG q30 is 0.099 bpp / MSE 7.87; V4-D intermediate is 0.459 bpp / MSE 7.16, similar distortion at roughly 4.6× the rate.
- Ablations show content dependence. On Googlewiki, intermediate DCT-only reaches MSE 15.74 and the combined D point 15.62 at similar rates, so procedural adds little there. On cat, DCT-only and combined are nearly tied (10.05 vs 10.11 MSE), with combined slightly smaller. On coast, procedural-only is 0.161 bpp / MSE 15.71, DCT-only is 0.660 bpp / 16.70, and combined is 0.459 bpp / 7.16; procedural coding is useful and complementary on that image.
- Joint complete-file initialization selection evaluates both caps for each point. Per-cap details are in each raw V4 record. This bounded comparison does not establish that higher initialization resolution is generally preferable; the selected cap varies with image and operating point.
- Conditional taper had a mixed result on the coast diagnostic. With identical hard-selected modes and parameters, file size stayed at 688,860 bytes and SKIP preservation remained exact. Global MSE rose from 7.162 to 7.287, while error in SKIP-adjacent boundary gradients fell from 21.74 to 7.23 (19.55 to 5.91 on source-smooth boundary samples). This is a local numerical improvement with a small global regression; no perceptual seam improvement is claimed.

## Performance and correctness

On the representative coast V4-D intermediate point, three warmed encodes averaged 34.15 s (median 34.15 s); three fresh-process decodes averaged 11.83 s (median 11.87 s). The profile of the coast low point peaked at 782,675,968 bytes working set (~746 MiB). Frozen V3.1 measured encode/decode calls are generally subsecond/under 0.1 s in this runtime. The V4 prototype is therefore substantially slower and heavier. Fresh-process decode includes interpreter startup. NumPy used OpenBLAS 0.3.29 with a 20-thread pool; all measured workers inherited the same environment.

Six synthetic test groups pass, covering odd dimensions, synthetic content, ablations, mode mixtures, deterministic file-only decode, exact protected SKIP and malformed streams. All 18 measured V4 streams and 12 V3.1 conditions decoded in a fresh process with exact pixel agreement.

SSIMULACRA2 was omitted because the configured local executable is not present. No AVIF comparison or reserved validation was run. JPEG timing uses Pillow 11.3.0 with RGB, 4:2:0 subsampling and optimization enabled.

## Decision

The prototype demonstrates original-resolution quality scaling and has a clear graphics-specific rate-distortion result. It fails the broader V4 objective in its current form: it does not retain low-rate V3.1 behavior, its photographic rates are poor against JPEG, and encode/decode cost is high. Stop parameter expansion here. The next useful research decision is whether to redesign the region allocator/transform representation for photographic rate efficiency and decoder cost before opening held-out images.

Raw rows, initialization-candidate statistics, stream hashes, timings, decode hashes and artifact paths are in `outputs/benchmark_raw.json`. Visual comparisons and mode maps are in `visuals/`.
