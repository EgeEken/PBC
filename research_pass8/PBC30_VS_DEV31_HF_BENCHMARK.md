# Actual PBC 3.0 versus integrated development on HF cpu-basic

The actual immutable release and actual development defaults were measured directly in one HF session. Development is faster in aggregate in every preset. It does not establish quality/bitrate dominance: overall RGB MSE is 0.763% higher and final bpp 2.670% higher. This is a development snapshot, not a PBC 3.1 release or tag.

| Preset | PBC 3.0 encode (s) | Dev encode (s) | Encode reduction | PBC 3.0 decode (s) | Dev decode (s) | Decode reduction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Compression | 0.598085 | 0.466540 | +21.99% | 0.096646 | 0.077528 | +19.78% |
| Balanced | 0.674056 | 0.550502 | +18.33% | 0.121539 | 0.078595 | +35.33% |
| Quality | 0.735032 | 0.552388 | +24.85% | 0.152387 | 0.080983 | +46.86% |
| High Quality | 0.849517 | 0.578764 | +31.87% | 0.244027 | 0.080060 | +67.19% |
| Overall | 0.714173 | 0.537049 | +24.80% | 0.153650 | 0.079292 | +48.39% |

Positive time reduction means faster: `100 * (1 - development / release)`. First take each image/preset/source median over three repeats, then average the 20 matched case medians equally within each preset. Overall averages all 80 cases equally. These are ratios of aggregate mean times, not means of percentages.

| Preset | Release MSE | Dev MSE | MSE change | Release bpp | Dev bpp | Bpp change | Release patches | Dev patches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Compression | 274.481629 | 277.694758 | +1.171% | 0.083968 | 0.078661 | -6.320% | 25.65 | 25.65 |
| Balanced | 193.429023 | 195.795584 | +1.223% | 0.168491 | 0.169866 | +0.816% | 33.70 | 36.55 |
| Quality | 160.078407 | 159.939395 | -0.087% | 0.298480 | 0.328058 | +9.910% | 32.80 | 34.80 |
| High Quality | 143.403428 | 143.851532 | +0.312% | 0.532350 | 0.535624 | +0.615% | 22.05 | 20.55 |
| Overall | 192.848122 | 194.320317 | +0.763% | 0.270822 | 0.278052 | +2.670% | 28.55 | 29.39 |

MSE and bpp changes are `100 * (development / release - 1)`. MSE is independently scored RGB from the appropriate decoder at original dimensions. Bpp counts the entire final v0 file after LZMA. Patch totals include three full-image initialization patches in every case. Mean iterative release/development counts: Compression 22.65/22.65, Balanced 30.70/33.55, Quality 29.80/31.80, HQ 19.05/17.55, overall 25.55/26.3875. Individual init/iterative counts and all results are in per_image_results.md and comparison_summary.json.

## Timing variation and regressions

| Preset | Encode wins | Paired encode reduction median [min, max] | Decode wins | Paired decode reduction median [min, max] | Encode CV release/dev | Decode CV release/dev |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Compression | 16/20 | +28.18% [-10.20, +52.77] | 18/20 | +20.33% [-7.41, +71.53] | 13.0/8.7% | 19.1/13.1% |
| Balanced | 18/20 | +17.17% [-38.70, +41.42] | 20/20 | +35.08% [+3.57, +88.87] | 8.8/8.8% | 14.6/11.9% |
| Quality | 18/20 | +23.51% [-1.99, +54.07] | 19/20 | +43.81% [-1.95, +87.68] | 9.4/9.6% | 15.0/16.5% |
| High Quality | 20/20 | +30.38% [+15.85, +48.91] | 20/20 | +65.51% [+38.40, +90.27] | 9.3/8.9% | 12.4/15.8% |
| Overall | 72/80 | +24.69% [-38.70, +54.07] | 77/80 | +41.49% [-7.41, +90.27] | 10.1/9.0% | 15.3/14.3% |

Across 80 cases, encode wins 72/80 and decode wins 77/80. Mean paired encode/decode reductions are 23.61%/44.23%; these differently weighted statistics are separate from the aggregate ratios. All three complete-trial arithmetic means favor development in both metrics; raw trial means are retained. Mean within-case CV is about 10% encode and 14–15% decode. Small per-image differences near that variation are inconclusive; the aggregate direction is consistent across trials.

Largest encode regressions: Eiffel Balanced 38.70% longer (0.335015 → 0.464667 s, total patches 20 → 31); Seagull Balanced 16.08% longer (12 → 39 patches); Cat Compression 10.20% longer (19 → 37 patches). Largest decode regressions: Flower Compression 7.41%, Ducklings Compression 2.02%, Flower Quality 1.95%. The latter small differences are within observed repeat noise. No outlier was removed. Original-resolution conversion/resizing and different patch trajectories remain part of these authentic calls.

There are 38/80 MSE wins and 39/80 bpp wins. Strict RD wins (lower MSE, no higher bpp) occur on nine cases; development is strictly worse in both on twelve. Largest MSE regressions include Field Quality +11.65% with bpp −42.79%, Rate-distortion Compression +10.55% with bpp −19.69%, Monet Compression +10.03% with bpp −14.68%, and Cat Balanced +9.96% with bpp −31.35%. These are substantial operating-point shifts, not equal-rate quality comparisons. Rate-distortion Quality has MSE −22.07% but bpp +30.33%; Coast Quality has MSE −23.23% at bpp +17.66%. Eiffel Balanced is worse in MSE (+1.95%), bpp (+2.46%) and encode time (+38.70%), despite faster decode. Aggregates must not hide these failures.

## Implementations and validity

Measured HF revision: `e3f947bb6b3c3e081ce0f0bb3ea801b8201c3226`, cpu-basic, one running replica. Repository/runtime revision and health were checked before and after. The later UI-only revision `27f6c65225a5496fa065deb586c7e405ecdffc84` changes only static/benchmark.js and its manifest; production codec and benchmark workers are identical.

Release: immutable GitHub `2e59f1b06980aa467da0cde94fa33c3f9a0d4d51`. All six preserved modules (PBC3, ops, heads, types, features, learned filler) were compared with their original Git blobs, including encoder/decoder, RNG use, presets and serialization. They import exclusively from /app/research_pass1/release. Development imports the eight integrated root modules, byte-for-byte codec source from HF `cd0d5691647a4f02cf765454a2d3b0abfde8f3b7`; no disabled-feature ablation substitutes for the release. The original release model blob matches the root model exactly: SHA256 `ffaf24aa5447b3602c53a5c4c8df14a68ee16e6bded39331d973f38d0f1ea62f`. No weights were trained or replaced. Source Git blob IDs/LF hashes and all 20 input hashes are recorded in hf_deployment.json and the measured metadata manifest.

Authentic RNGs are retained: release `numpy.random.default_rng`, PCG64 in this environment; development `pbc3_ops.PBC3Rng`, PCG32 with modulo integers. Seed 2003 for both. Exact default preset dictionaries are retained in metadata.workers.*.presets. Shared defaults include YCbCr working space, automatic 250k cap, LZMA, internal final MSE enabled, learned legacy action/value policy, Quality and Compression/Balanced budgets 50, HQ budget 20; learned quality dials 0.4/0.6/0.8/0.95. Development additionally uses selected-delta reuse, bicubic projection, gradient cell selection and explicit search state. Only the absolute model path is supplied to each config; no RNG override or RD modification is installed.

Twenty existing main images × four presets × two sources × three repeats = 480 measured calls, plus eight full-call warmups. A preceding one-image/eight-call HF smoke also passed. Both persistent processes are pinned to CPU 0, numerical environment threads all one, sequentially interleaved by rotating source order across image/preset/trial. No other full benchmark ran concurrently. A few read-only UI/API verification requests occurred early in the run; browser polling was then closed. Shared HF scheduling/traffic remain uncontrolled noise.

Runtime: Python 3.11.17, NumPy 2.4.6, Numba 0.68.0, Pillow 11.3.0, identical for both workers. Full encode and decode are warmed by one googlewiki call per preset/source; cold/warmup call walls are retained separately and not mixed into the table. Production per-encode initialization, internal metric, output reconstruction and final resizing are inside the encode timer. Decode includes final original-size resizing. Image loading, EXIF transpose/RGB preprocessing, hashing, independent RGB scoring, parsing, logging, IPC and HTTP are outside both timers.

Every file was decoded by its corresponding source implementation. Original dimensions, preview-versus-decoder equality, internal/external MSE consistency and within-source repeat stream/pixel determinism passed. Source namespaces are separate processes; metadata shows their actual import paths and identical model hashes/affinity. All 80 release/development pairs have different streams. Consequently the measured speed ratios include engineering differences and changed authentic patch trajectories. They cannot be attributed solely to vector grid I/O/search-state reuse; those were already proven exact against the preceding HF codec, not against this release. The optional forced-RNG secondary experiment was not needed for the primary default-to-default conclusion and was not run.

No matched-rate frontier or holdout evaluation is implied by these four presets. Encoding and decoding speed improve in aggregate relative to actual PBC 3.0; universal speed wins, quality improvement, bitrate improvement and preservation of every extreme-compression point have not been achieved. The source remains version 3.0/v0 and no release/tag/default was promoted.

## Reproduction

The exact authenticated request is saved in hf_comparison_start.json locally and in comparison_summary.json metadata.spec. It fixes all 20 allowlisted images, four presets, variants [release, development], repeats 3. With explicit operational authorization, `python research_pass8/run_hf.py benchmark` submits it to the existing Space. For a local execution check use `python research_pass8/check_support.py`; local timings are not substituted for HF measurements.

`python research_pass8/analyze_comparison.py` reads ignored hf_comparison.json and regenerates the compact summary and tables. comparison_summary.json retains all 160 per-source cases, three raw timing samples each, output/input hashes, per-image paired changes, trial means, original measured manifest, exact configs, warmup rows and the full-response SHA256. Full raw responses, databases, datasets and generated files remain outside Git. An initial report-write failed on Windows default text encoding for the arrow symbol; UTF-8 output was specified and analysis rerun on the same saved measurements, with no rebenchmark.

All per-image outcomes follow in [per_image_results.md](per_image_results.md). These include the named regressions rather than only favorable averages.
