# Chroma error analysis

## Method

Reference Y/U/V samples were produced by the same pinned libavif 1.3.0 PNG/JPEG input loaders used by avifenc (`avifPNGRead` / `avifJPEGRead`), requested as 8-bit 4:2:0 with automatic chroma downsampling, profiles/EXIF/XMP ignored, and CICP 1/13/1 and full range set before RGB-to-YUV conversion. This matches the encoder's input conversion path; it does not compare decoded YUV420 against an arbitrary YUV444 conversion.

Measured AVIFs were decoded with pinned `avifdec`'s libaom decoder. The analysis tool copied native decoded Y/U/V planes through libavif's public API and made counterfactuals with `avifImageYUVToRGB`, using the decoded AVIF's signaled range, CICP, chroma position and automatic upsampling. Its ordinary decoded RGB matched `avifdec` PNG output exactly (maximum channel difference 0 for all three files). MSE uses float64 channel differences. Edge is the top 20% source Sobel magnitude on linear-light BT.709 luma; smooth is the remaining 80%.

The three reference operating points are the low-rate settings reused for Pass 41: Googlewiki q5, Cat q35 and Coast q80. The corresponding REF files are the actual unmodified Pass 35 release binary; settings are speed 6, one job, 8-bit YUV420, full range, CICP 1/13/1, `aq-mode=0`, `tune=psnr`, `deltaq-mode=0`, `sb-size=64`.

## Native plane error and RGB diagnostics

| Image / q | Bytes | RGB MSE (R/G/B channel MSE) | Native Y MSE | Native U MSE | Native V MSE | RGB MSE if only UV restored | RGB MSE if only Y restored | Source YUV420→RGB floor | Edge / smooth RGB MSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Googlewiki q5 | 23,362 | 568.264 (640.450 / 365.568 / 698.775) | 333.467 | 69.225 | 39.694 | 483.834 | 287.413 | 193.473 | 2172.332 / 166.203 |
| Cat q35 | 290,049 | 43.019 (43.855 / 38.393 / 46.810) | 37.346 | 2.678 | 2.580 | 38.365 | 6.261 | 1.348 | 72.048 / 35.762 |
| Coast q80 | 167,920 | 3.242 (3.250 / 2.683 / 3.794) | 2.363 | 0.340 | 0.347 | 2.887 | 1.135 | 0.725 | 3.977 / 3.059 |

“RGB MSE if only UV restored” replaces decoded U/V with source-derived U/V and keeps decoded Y. “RGB MSE if only Y restored” replaces decoded Y and keeps decoded U/V. Restoring chroma lowers RGB MSE by 84.430 (14.9%) on Googlewiki, 4.654 (10.8%) on Cat and 0.356 (11.0%) on Coast. Restoring luma lowers it by 280.851 (49.4%), 36.758 (85.5%) and 2.108 (65.0%), respectively. These two counterfactual improvements are not additive: color conversion, clipping and channel interactions remain.

The source YUV420→RGB floor is the RGB error after the source image itself passes through the exact 4:2:0 input conversion and back to RGB. It is especially large for Googlewiki's sharp colored graphics (193.473 MSE); it is not AV1 quantization error. The native U/V plane MSE values have one quarter as many samples as Y in 4:2:0 and should be read as per-sample errors, not compared as equally sized SSE budgets.

## Gate B decision

Chroma is not the dominant error source, but it is not negligible: restoring source-derived chroma reduces RGB MSE by about 11–15% at these low-rate points across graphical and photographic content. This passes the feasibility gate for a small existing-control test. It does not by itself justify changing the encoder or prove that extra chroma bits are rate-efficient.

Raw rows, per-plane statistics, exact AVIF properties and counterfactual arrays are retained under ignored `outputs/phase_b_results.json`, `outputs/chroma_error_analysis.json` and `outputs/planes/`.
