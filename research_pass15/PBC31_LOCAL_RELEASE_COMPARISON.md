# PBC 3.0 vs PBC 3.1 development — local release comparison

## Method

PBC 3.0 is the immutable release at `2e59f1b06980aa467da0cde94fa33c3f9a0d4d51`. PBC 3.1 development candidate `pbc31_integral_fused` derives from frozen source `8006f30debfc4a57289df06df21636aeb52baffd` (`pbc31_release_rng_baseline`) and changes only `pbc3_ops.integral()` to fused Numba accumulation. Workers load one source tree each in separate persistent Python 3.11.16 processes. They each warm all four presets before measurement, preload the same eight EXIF-transposed RGB images, then run three interleaved repetitions per image and preset. Timers wrap the complete `compress` and `decompress` calls only. Hashing, patch parsing, external RGB MSE, IPC, and image loading are outside timers.
The design records 192 variant observations (each observation times one encode plus one decode), forming 96 matched old/new case-repetition pairs.

Machine: 12th Gen Intel Core i9-12900H (14 cores: 6 E-cores (efficiency class 1) and 8 P-cores (class 0); 20 logical processors); both workers are pinned to logical CPU 12, mapped by Windows processor topology to a P-core. All numerical thread limits are 1. PBC 3.0 and PBC 3.1 use the same Python/dependency runtime, image data, model hash, default seed 2003, release RNG, and internal final-MSE setting.

Runtime: Python 3.11.16; NumPy 2.4.6; Numba 0.68.0; Pillow 11.3.0. Source-isolated process CWDs: `C:\Users\EGE\.codex\worktrees\pbc-research-experiments\PBC\research_pass1\release` and `C:\Users\EGE\.codex\worktrees\pbc-research-experiments\PBC`. Windows reports a 2.5 GHz current/max package clock; per-core boost telemetry is unavailable through psutil on this host, so frequency was not normalized or inferred from that report.

Input file SHA-256 values:

| Image | Input SHA-256 |
|---|---|
| cat.jpg | `75482650c75b9fc17f4ee2f40b093a6065ca9103216650df5da4dbd492915c4e` |
| eiffel.png | `90424a60b193cbbc66a20a4039d12dda76059333a04a6037ac9947276a60c93e` |
| flower.jpg | `a8a33c44cfaf9b251d896181d55e30a8bd1e53fa41574c8f75b74950d73f2f72` |
| googlewiki.png | `cb2cc48d6a03b603d217b59fc4a3d3e53c6bebdff09e0a8172640bf85ccde94e` |
| mario64.png | `19e195de8f46fc4c4ef19925461c21da5769d682b152c967a783d5d3868c00d9` |
| monet.jpg | `ea9dc92627a5226e2bcb1ba9cb029cdb68380ef922f24866e8222ff56a093ea7` |
| ratedistortion.png | `e2fc378ea28312d303c38934d88968913f1018744e7d68f92dfaf7c5bb0a6c10` |
| seagull.jpg | `d44f39a33b0c39d2ab6442dd3a606f6becb5ac3e91165a4a48c37b1aca5c2639` |

## Correctness gate

All 96/96 paired measured streams were byte-identical. All 96/96 decoded RGB outputs, patch counts, final bpp, and independently scored original-resolution RGB MSE values matched exactly. Repeated outputs were stable within each source. Internal MSE remained enabled in both implementations. The frozen model SHA-256 was `ffaf24aa5447b3602c53a5c4c8df14a68ee16e6bded39331d973f38d0f1ea62f`.

## Preset timing results

Values are averages of the 8 per-image medians (three repetitions per case). Reduction is `1 - PBC 3.1 / PBC 3.0`.

| Preset | PBC 3.0 encode (s) | PBC 3.1 encode (s) | Encode reduction | PBC 3.0 decode (s) | PBC 3.1 decode (s) | Decode reduction |
|---|---:|---:|---:|---:|---:|---:|
| Compression | 0.5128 | 0.3745 | 26.98% | 0.0823 | 0.0535 | 35.02% |
| Balanced | 0.6447 | 0.4298 | 33.33% | 0.1189 | 0.0547 | 53.99% |
| Quality | 0.8386 | 0.5234 | 37.59% | 0.1663 | 0.0582 | 64.99% |
| High Quality | 0.9167 | 0.5949 | 35.11% | 0.2857 | 0.0669 | 76.58% |
| Overall | 0.7282 | 0.4807 | 34.00% | 0.1633 | 0.0583 | 64.28% |

## Per-image medians

Each cell averages the four per-preset medians for that image.

| Image | PBC 3.0 encode (s) | PBC 3.1 encode (s) | Encode reduction | PBC 3.0 decode (s) | PBC 3.1 decode (s) | Decode reduction |
|---|---:|---:|---:|---:|---:|---:|
| cat | 0.9170 | 0.6844 | 25.37% | 0.2045 | 0.1172 | 42.68% |
| eiffel | 0.5541 | 0.3320 | 40.09% | 0.1229 | 0.0232 | 81.09% |
| flower | 0.8054 | 0.6473 | 19.63% | 0.1589 | 0.1145 | 27.94% |
| googlewiki | 0.7734 | 0.4430 | 42.72% | 0.2307 | 0.0390 | 83.08% |
| mario64 | 0.8181 | 0.4853 | 40.68% | 0.1521 | 0.0406 | 73.34% |
| monet | 0.6627 | 0.4149 | 37.39% | 0.1196 | 0.0375 | 68.67% |
| ratedistortion | 0.4440 | 0.2680 | 39.64% | 0.1338 | 0.0173 | 87.05% |
| seagull | 0.8511 | 0.5704 | 32.98% | 0.1837 | 0.0772 | 58.00% |

## Variability and outliers

Mean within-case CV across three repetitions: encode 2.61% for PBC 3.0 and 3.35% for PBC 3.1; decode 3.63% and 4.66%. Whole-call reductions across the 32 cases were 0 encode regressions and 0 decode regressions (negative values).

Slowest PBC 3.1 encoder case by median: cat.jpg/balanced (0.7673s median). Slowest PBC 3.1 decoder case by median: cat.jpg/high_quality (0.1230s median). Lowest encode reduction: flower.jpg/balanced (10.08% smallest reduction). Lowest decode reduction: cat.jpg/compression (12.91% smallest reduction).

The complete per-repetition timing rows and SHA/MSE/bpp/patch checks are preserved locally in `local_comparison.json` (ignored by Git).

## Relation to HF results

This local test uses a different machine, dataset and timing protocol from the controlled HF cpu-basic comparison. The HF result was 28.83% lower encoding time and 64.38% lower decoding time with byte-identical files on its matched cases. The two percentages are reported separately and should not be used as direct cross-machine comparisons.
