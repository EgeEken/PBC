# PBC Pass 12 — Completed-file results

## Paired screen

Each row compares the frozen preset with only `color_space` changed. MSE is independently measured on the original-size RGB image returned by `PBC3.decompress`; bpp uses complete post-LZMA file bytes divided by original pixels.

| Image | Preset | YCbCr MSE | RGB MSE | MSE change | YCbCr bpp | RGB bpp | bpp change | Result |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cat.jpg | Quality | 299.19 | 291.51 | −2.57% | 0.0594 | 0.1569 | +164.14% | Quality for rate |
| cat.jpg | High Quality | 293.13 | 285.58 | −2.58% | 0.0740 | 0.1781 | +140.54% | Quality for rate |
| flower.jpg | Quality | 43.43 | 30.83 | −29.02% | 0.0272 | 0.0658 | +142.07% | Quality for rate |
| flower.jpg | High Quality | 34.84 | 26.79 | −23.10% | 0.0433 | 0.0872 | +101.56% | Quality for rate |
| monet.jpg | Quality | 96.80 | 80.25 | −17.09% | 0.5100 | 1.5477 | +203.49% | Quality for rate |
| monet.jpg | High Quality | 53.98 | 56.27 | +4.25% | 1.2321 | 2.6203 | +112.67% | Strict loss |
| googlewiki.png | Quality | 781.03 | 712.11 | −8.82% | 0.9978 | 2.5055 | +151.10% | Quality for rate |
| googlewiki.png | High Quality | 714.43 | 710.53 | −0.55% | 2.1412 | 2.6669 | +24.55% | Quality for rate |
| mario64.png | Quality | 135.47 | 137.32 | +1.36% | 0.5209 | 0.8041 | +54.37% | Strict loss |
| mario64.png | High Quality | 122.03 | 109.54 | −10.23% | 0.7190 | 0.9794 | +36.21% | Quality for rate |
| eiffel.png | Quality | 28.76 | 12.77 | −55.58% | 1.6253 | 4.3979 | +170.59% | Quality for rate |
| eiffel.png | High Quality | 15.41 | 7.91 | −48.67% | 2.8966 | 4.6030 | +58.91% | Quality for rate |
| stockpoint.jpg | Quality | 28.21 | 16.98 | −39.80% | 0.1127 | 0.3151 | +179.51% | Quality for rate |
| stockpoint.jpg | High Quality | 16.76 | 12.79 | −23.65% | 0.2282 | 0.6212 | +172.18% | Quality for rate |
| ratedistortion.png | Quality | 47.43 | 12.59 | −73.45% | 0.8804 | 2.2196 | +152.10% | Quality for rate |
| ratedistortion.png | High Quality | 15.61 | 9.54 | −38.90% | 1.5366 | 2.5367 | +65.09% | Quality for rate |

There are **zero strict RD wins**, **14 quality-for-rate trades**, and **two strict losses**. RGB increases bpp in every case. It improves MSE in 7/8 images at each preset, but no case improves both dimensions. RGB never saves bits with a worse image.

## Aggregate measurements

Arithmetic means over the eight fixed images. Timings are local two-repeat means and are descriptive only.

| Preset | MSE YCbCr → RGB | Mean bpp YCbCr → RGB | Encode s YCbCr → RGB | Decode s YCbCr → RGB | Raw bytes YCbCr → RGB | Final bytes YCbCr → RGB | Patches YCbCr → RGB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Quality | 182.54 → 161.80 (−11.36%) | 0.5917 → 1.5016 (+153.77%) | 0.365 → 0.514 (+40.84%) | 0.0334 → 0.0413 (+23.62%) | 111,098 → 254,465 | 47,428 → 117,252 | 32.50 → 48.00 |
| High Quality | 158.27 → 152.37 (−3.73%) | 1.1089 → 1.7866 (+61.12%) | 0.454 → 0.463 (+2.05%) | 0.0431 → 0.0429 (−0.45%) | 260,679 → 365,864 | 84,734 → 148,617 | 21.50 → 23.00 |

The Quality encode/decode changes are large relative to this small timing sample; High Quality is near parity. The maximum per-case encode repeat CV was 31.9%. Do not treat these as stable speed claims. No timing result was transferred to HF.

## Why the RD moved

The color transform alone does not explain the completed-file changes. The no-patch floors differ by at most 2.43 RGB MSE for these images, while the completed-file improvements range from 0.55% to 73.45% in the 14 quality-improvement cases. RGB changes the policy trajectory and patch allocation:

- Quality mean iterative patches rise from 29.5 to 45.0; High Quality rises from 18.5 to 20.0.
- In Quality, pooled iterative channel counts shift from YCbCr `{0:145, 1:63, 2:28}` to RGB `{0:142, 1:117, 2:101}`. High Quality shifts from `{0:95, 1:31, 2:22}` to `{0:54, 1:58, 2:48}`. RGB distributes corrections more evenly across all three channels, especially channels 1 and 2.
- Mean grid cells per iterative patch rise from 7,669 to 9,918 in Quality and 25,444 to 28,273 in High Quality. Higher bitcount selections also rise: Quality’s pooled 4/5-bit grid patches go from 105 to 203; High Quality’s go from 102 to 146.
- Mean raw stream bytes and final bytes both rise at each preset. For Eiffel High Quality, RGB has fewer raw bytes than YCbCr (312,578 vs 332,765) but more final bytes (136,461 vs 85,874), showing that the final LZMA result cannot be inferred from raw size alone.

Photography and graphics/synthetic groups both lower average MSE but raise rate substantially. Photography changes are −10.28% MSE / +194.03% bpp in Quality and −4.33% / +122.28% in High Quality. Graphics/synthetic changes are −11.88% / +146.67% and −3.45% / +47.89%. This is not a class where RGB saves bits.

## Reproducibility limits and stopping decision

The paired run used the same frozen source, model, seed, inputs and all config fields except color space. All inputs match saved Pass 9 hashes; two measured repeats per point emitted identical streams and decoded pixels. RGB v0 files decode through the existing decoder at source dimensions.

The local YCbCr result reproduced the saved Pass 10 hash and patch count in only 6/16 cases. The current local versions (Python 3.13.5, NumPy 2.2.6, Numba 0.62.1) differ from the recorded HF runtime (Python 3.11.17, NumPy 2.4.6, Numba 0.68.0). This screen therefore supports a local paired finding, not a claim of exact HF trajectory parity. No same-runtime HF run was done.

Since final bpp rises for all 16 RGB points and no strict RD win occurs, no matched-rate, oracle selector, adaptive rule, Compression/Balanced extension or holdout confirmation was justified. Global RGB is rejected. No production behavior or HF deployment changed.

Machine-readable result: `paired_results.csv`, `preprocess_floor.csv`, and `screen_summary.json`. Reproduce with `python research_pass12/run_color_space.py` followed by `python research_pass12/analyze_color_space.py`. The detailed repeat JSON files are local and ignored.

Frozen source: [GitHub 8006f30](https://github.com/EgeEken/PBC/commit/8006f30debfc4a57289df06df21636aeb52baffd). HF development baseline: [98ade0c](https://huggingface.co/spaces/EgeEken/PBC/commit/98ade0c8072a4571e9674eaac3d91f861e3b9867). Model SHA-256: `ffaf24aa5447b3602c53a5c4c8df14a68ee16e6bded39331d973f38d0f1ea62f`.
