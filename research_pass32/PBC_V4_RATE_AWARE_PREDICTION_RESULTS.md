# PBC V4 — Pass 32: Rate-Aware Mode Selection and Generalization

## Result

The fixed-grid predictive modes generalize to these three held-out photographs, but the gains are small and do not meet the screen for a meaningful improvement. R1 changes candidate ranking and yields additional measured frontier points, but it does not materially improve quality at a matched rate. Keep this as a research result; do not promote it as a release change.

The largest repeatable held-out change was on Cathedral: D1 and D each reduced MSE by about 0.76% at rates within 1% of B. Park's corresponding MSE gains were about 0.13%; Seagull's all-mode set was about 0.14% worse at a slightly lower rate. None of the comparisons reached the predeclared 10% screen. The Cat-only R1 model was frozen before the three held-out images were encoded.

## Experimental setup

- Starting revision: `ec52d479399eefed33ca93740b0dd89e44b275f8`; base encoder revision: `8006f30debfc4a57289df06df21636aeb52baffd`.
- One-thread Windows 11 run, Python 3.13.5, Intel Family 6 Model 154 CPU, NumPy 2.2.6, Pillow 11.3.0. The PBC 3.1 Quality base stream and decoded base were frozen per image; Cat, Flower, and Mario reused Pass 29 streams.
- Six images × four mode sets × (three R0 lambdas + five R1 lambdas) = 192 candidates. R0 lambdas: 0, 20, 100. Cat-calibrated R1 lambdas: 0, 20, 100, 400, 1600. XZ preset 6. R1 model SHA256: `6f2a4524d83a89a74e54489a2b20e40ff6d26dfa18a40beeeb86aeb48a717db9`.
- Every candidate was fully serialized and decoded in-process; all 192 output streams then passed an independent fresh-process pixel, mode, and parameter hash check. The mode-map display uses the decoder's seven actual mode IDs and count-checked labels.
- D0 enabled B plus horizontal/vertical prediction without offsets. D1 enabled B plus offset directional prediction only. D enabled all modes. All modes retain the P31 stream header and meanings.

## Held-out photographs

The table pairs each configuration with the B point at the closest measured total file size under the same estimator. Percentages are D0/D1/D relative to B. Negative MSE is better; negative rate means fewer total bytes. These are actual points, not interpolation.

| Image | Estimator | D0: rate / MSE | D1: rate / MSE | D: rate / MSE |
|---|---|---:|---:|---:|
| Cathedral | R0 | −2.63% / −0.37% | +1.10% / −0.86% | +1.16% / −0.86% |
| Cathedral | R1 | −0.08% / −0.47% | −0.83% / −0.76% | −0.76% / −0.76% |
| Park | R0 | −0.45% / −0.03% | −0.71% / −0.13% | −0.79% / −0.13% |
| Park | R1 | −0.19% / −0.03% | −0.45% / −0.13% | −0.52% / −0.13% |
| Seagull | R0 | −0.36% / −0.05% | −0.72% / −0.20% | −0.65% / −0.20% |
| Seagull | R1 | +0.02% / −0.07% | −0.14% / +0.11% | −0.10% / +0.14% |

The 5%-rate screen was met in every comparison except Seagull R1 D1 and D; however, the best rate or MSE changes within that band remained below 10%. A near-rate result on a single image should not be treated as a broad release gain.

## R0, R1, and mode evidence

Across all 192 candidates, R1's predicted enhancement bytes had 27.3% mean absolute percentage error, a +25.0% signed bias, and 0.992 correlation with actual enhancement bytes. R0 had 477.4% mean absolute percentage error and 0.945 correlation. R1 is a substantially better ranking signal here, but it still overestimates rates and is not a calibrated byte predictor. Its histogram was trained on Cat only, with Dirichlet smoothing alpha 0.5; the three held-out images were not used to fit it.

Each R1 configuration contributed five nondominated points per image, versus three for R0; none of the R1 points was strictly dominated by an R0 point in the pooled six-image comparison. This primarily reflects the denser lambda sweep and should not be mistaken for a better same-rate frontier. On the held-out images, R1's best matched-rate MSE shifts were below 1% in magnitude.

The mode map for Cat / D / R1 / lambda 20 was decoded and checked against recorded counts: Skip 7,019; DC 3,228; vertical no-offset 847; horizontal no-offset 2,069; detail 16,598; vertical with-offset 4; horizontal with-offset 3,611. Directional offsets can quantize to zero while still occupying their serialized offset slots; R1 does not eliminate this overhead reliably.

## Runtime and memory

Median encode / decode time over the 32 candidates per image was 11.0 / 1.61 s for Cat, 10.4 / 0.83 s for Flower, 1.23 / 0.14 s for Mario, 15.4 / 2.47 s for Cathedral, 10.8 / 1.87 s for Park, and 6.04 / 1.08 s for Seagull. These are this Windows machine's measurements, not HF Space timings. Decode profiling shows reconstruction dominates entropy parsing; the per-candidate measurements and stage timings are in the summary JSON.

The encoder process peak working set was 559 MB for the two reused photographs/control groups and 781 MB after processing the new large images. Fresh decoder process peaks ranged from 43 MB (Mario) to 148 MB (Cathedral). These are process high-water marks, not incremental per-candidate costs.

## Files and validation

- [Compact candidate summary](PBC_V4_RATE_AWARE_PREDICTION_SUMMARY.json) contains provenance and the 192 candidate rows.
- [Rate–distortion plots](visuals/rate_distortion.png) show measured R0 and R1 points for the held-out photographs.
- [Decoded mode map](visuals/mode_selection_cat_D_R1_lambda20.png) shows the corrected seven-mode display.
- Full raw rows, XZ streams, base images, and detailed analysis are retained locally under `outputs/` and excluded from Git.
- `python -B -m unittest test_pbc32_codec -v`: 10 tests passed (the seven Pass 31 tests plus three Pass 32-specific tests).

## Recommendation

Do not ship the current mode set or R1 cost model as a PBC release improvement. The held-out mode-set gains are too small, R1 retains a substantial positive size bias, and the results do not establish an encode-speed or decode-speed improvement against the actual PBC 3.0 reference. The useful outcome is a frozen, reproducible six-image comparison and an empirically better encoder-side ranking model for further isolated experiments.
