# HF release-RNG three-way comparison

Optimized development with release RNG reproduces immutable PBC 3.0 exactly in all 100 HF image/preset cases. In the separate paired timing run it reduces aggregate encode time 28.83% and decode time 64.38%, with identical streams and RD. Every preset is faster. Recommend this variant as the PBC 3.1 development foundation; the live public default remains the existing custom RNG.

## Simultaneous full-call performance

Seconds; each source/image/preset cell is the median of three warmed repetitions, followed by equal weighting of the eight case medians within each preset. Overall equally weights all 32 cases. These times come from the same HF session, not historical release latencies.

| Preset | Release encode | Dev release-RNG encode | Dev current encode | Release decode | Dev release-RNG decode | Dev current decode |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Compression | 0.468090 | 0.368027 | 0.360938 | 0.091763 | 0.058645 | 0.065754 |
| Balanced | 0.578009 | 0.426547 | 0.474414 | 0.128845 | 0.054979 | 0.057445 |
| Quality | 0.755442 | 0.550425 | 0.554162 | 0.155399 | 0.062698 | 0.059769 |
| High Quality | 0.915053 | 0.588329 | 0.566305 | 0.292907 | 0.061961 | 0.065506 |
| Overall | 0.679149 | 0.483332 | 0.488955 | 0.167228 | 0.059571 | 0.062119 |

Positive reductions mean faster: `100 * (1 - development_time / release_time)`. Ratios use the aggregate mean of matched case medians, rather than averaging per-image percentages.

| Preset | Release-RNG dev encode reduction vs 3.0 | Release-RNG dev decode reduction vs 3.0 | Current-RNG dev encode reduction vs 3.0 | Current-RNG dev decode reduction vs 3.0 |
| --- | ---: | ---: | ---: | ---: |
| Compression | +21.38% | +36.09% | +22.89% | +28.34% |
| Balanced | +26.20% | +57.33% | +17.92% | +55.42% |
| Quality | +27.14% | +59.65% | +26.64% | +61.54% |
| High Quality | +35.71% | +78.85% | +38.11% | +77.64% |
| Overall | +28.83% | +64.38% | +28.00% | +62.85% |

## RD and workload

The following table uses the same eight-image timing sample. RGB MSE comes from independent decoding at original size; bpp counts the complete final post-LZMA v0 file. Release and release-RNG development are exactly equal on every case, not merely on average.

| Preset | Release MSE | Dev release-RNG MSE | Dev current-RNG MSE | Release bpp | Dev release-RNG bpp | Dev current-RNG bpp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Compression | 340.157264 | 340.157264 | 348.329627 | 0.166459 | 0.166459 | 0.151972 |
| Balanced | 234.532942 | 234.532942 | 239.609904 | 0.339815 | 0.339815 | 0.341072 |
| Quality | 187.106489 | 187.106489 | 187.980867 | 0.588329 | 0.588329 | 0.658589 |
| High Quality | 163.779892 | 163.779892 | 165.031743 | 1.078363 | 1.078363 | 1.089584 |
| Overall | 231.394147 | 231.394147 | 235.238035 | 0.543241 | 0.543241 | 0.560304 |

The 20-image full-main Quality gate has a different sample composition: release and restored development both give MSE 160.07840745282843 and bpp 0.29847982134066314. All 80 main release hashes match Pass 8. Its recorded custom-RNG Quality mean was 0.3280581797472375 bpp (+9.909668%). Restored development has 0% residual release-relative regression, removing 100% of that recorded excess. The new 32 current-RNG sample streams also match Pass 8. Historical RD hashes/bytes are used only for this continuity question; no historical timing enters speed calculations.

Patch means below include three initialization patches in every case; iterative means exclude them. Exact per-image counts and timings are in per_image_results.md and comparison_summary.json.

| Preset | Total patches release / release-RNG / current | Iterative patches release / release-RNG / current |
| --- | ---: | ---: |
| Compression | 19.12 / 19.12 / 17.62 | 16.12 / 16.12 / 14.62 |
| Balanced | 24.62 / 24.62 / 31.75 | 21.62 / 21.62 / 28.75 |
| Quality | 35.62 / 35.62 / 38.62 | 32.62 / 32.62 / 35.62 |
| High Quality | 20.75 / 20.75 / 20.50 | 17.75 / 17.75 / 17.50 |
| Overall | 25.03 / 25.03 / 27.12 | 22.03 / 22.03 / 24.12 |

## Engineering versus RNG trajectory

Release versus dev_release_rng uses the same NumPy PCG64 generator and selected patch payloads/order, final files and decoded pixels. Its 28.83% encode/64.38% decode reductions are combined implementation gains with the release stream preserved. They are not separately attributable to any one runtime mechanism and are not sums of earlier pass percentages.

The two development variants share the same optimized implementation. Their RNG generation costs and search trajectories differ, including stopping points, box geometry, grid sizes and entropy payloads. This is not an isolated RNG microbenchmark. Decoding does not generate random numbers; development decode differences follow different payload work and timing noise, not an RNG cost in the decoder.

| Preset | Release-RNG encode reduction vs current | Release-RNG decode reduction vs current |
| --- | ---: | ---: |
| Compression | -1.96% | +10.81% |
| Balanced | +10.09% | +4.29% |
| Quality | +0.67% | -4.90% |
| High Quality | -3.89% | +5.41% |
| Overall | +1.15% | +4.10% |

These are release-RNG time reductions relative to current RNG. The aggregate estimates favor release RNG by 1.15% encode and 4.10% decode; equivalently, selecting custom RNG increases the aggregate times by about 1.16% and 4.28% relative to restored development. Those small differences do not establish a throughput advantage for either sampler: release RNG wins 17/32 encode case medians and 22/32 decode case medians, while the current RNG has lower whole-trial mean encode time in two of three trials. No material aggregate whole-codec penalty from restoring release RNG is demonstrated. Pure generator overhead is not isolated, and no RNG replacement, seed sweep or additional microbenchmark was run.

Workload shifts can be large in either direction. Balanced emits 24.625 restored versus 31.75 current mean patches and is 10.09% faster with release RNG. Compression emits 19.125 versus 17.625 patches and restored encoding is 1.96% longer; HQ is 3.89% longer with nearly equal patch counts. Seagull Compression is 43.79% longer than current development (37 versus 17 patches), and Flower HQ is 22.66% longer (23 versus 10). Both restored cases remain faster than actual release. Equal patch counts alone do not establish equal geometry/grid/reconstruction work.

## Per-image outcomes and variation

Release-RNG development wins all 32 encode and 32 decode case medians against actual release. Its smallest gains are Flower Compression encode 3.61% and Seagull Compression decode 1.01%; these small effects are within repeat variation and are not universal-speed guarantees. Current development wins 28/32 encode and 30/32 decode medians.

Named encode outliers in this simultaneous session:

| Case | Release s | Release-RNG dev s | Current dev s | Release-RNG reduction vs release | Current time change vs release | Patches release / restored / current |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Cat Compression | 0.743403 | 0.601150 | 0.836706 | +19.14% | +12.55% | 19 / 19 / 37 |
| Eiffel Balanced | 0.410612 | 0.288889 | 0.411273 | +29.64% | +0.16% | 20 / 20 / 31 |
| Seagull Balanced | 0.449799 | 0.361712 | 0.591004 | +19.58% | +31.39% | 12 / 12 / 39 |
| Rate-distortion Quality | 0.433680 | 0.344690 | 0.496320 | +20.52% | +14.44% | 18 / 18 / 53 |

Cat's current-RNG regression persists and disappears with release RNG. Eiffel's prior 38.70% current-RNG slowdown is not reproduced in magnitude here: +0.16% is negligible relative to noise, while restored development is 29.64% faster with release-identical work. Seagull Balanced and Rate-distortion Quality also become wins under restored trajectories. Current decode regressions are Seagull Compression +16.89% and Cat Compression +1.60%; restored development improves them by 1.01% and 36.25% versus release respectively, with the first effect inconclusive at case level.

| Preset | Variant | Encode wins/losses | Decode wins/losses | Mean case encode CV | Mean case decode CV | Encode reduction range vs release | Decode reduction range vs release |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Compression | dev_release_rng | 8/0 | 8/0 | 19.40% | 27.09% | +3.61% to +43.66% | +1.01% to +78.09% |
| Compression | dev_current_rng | 7/1 | 6/2 | 13.78% | 27.01% | -12.55% to +52.22% | -16.89% to +77.42% |
| Balanced | dev_release_rng | 8/0 | 8/0 | 12.96% | 14.13% | +15.81% to +41.80% | +28.08% to +87.74% |
| Balanced | dev_current_rng | 6/2 | 8/0 | 12.02% | 21.46% | -31.39% to +37.07% | +17.81% to +87.13% |
| Quality | dev_release_rng | 8/0 | 8/0 | 12.31% | 27.64% | +10.47% to +48.97% | +13.84% to +81.07% |
| Quality | dev_current_rng | 7/1 | 8/0 | 13.69% | 19.63% | -14.44% to +52.02% | +20.67% to +84.40% |
| High Quality | dev_release_rng | 8/0 | 8/0 | 9.80% | 19.19% | +16.08% to +53.42% | +43.29% to +92.35% |
| High Quality | dev_current_rng | 8/0 | 8/0 | 10.23% | 23.04% | +15.17% to +53.39% | +26.04% to +91.47% |
| Overall | dev_release_rng | 32/0 | 32/0 | 13.62% | 22.01% | +3.61% to +53.42% | +1.01% to +92.35% |
| Overall | dev_current_rng | 28/4 | 30/2 | 12.43% | 22.78% | -31.39% to +53.39% | -16.89% to +91.47% |

Mean within-case encode CV is release 14.37%, restored 13.62%, current 12.43%; decode CV is 20.27%, 22.01%, 22.78%. All three complete-trial mean encode and decode times favor both development variants over release. The restored-versus-current ranking varies by trial. Largest restored encode case CV is 26.69% (googlewiki Compression), and largest current decode CV is 49.95% (googlewiki Compression: 0.023442, 0.052457, 0.023914 s). No observation or outlier was removed. All individual samples and whole-trial means are retained; three repeats on shared HF CPU do not support precise small per-image claims.

## Runtime and operational provenance

Deployed and measured HF revision: [4237d954f4e1a04951c822c738113a53bbd58e9b](https://huggingface.co/spaces/EgeEken/PBC/commit/4237d954f4e1a04951c822c738113a53bbd58e9b), parent 27f6c65225a5496fa065deb586c7e405ecdffc84, one cpu-basic replica. All jobs were idle before deployment. The previously prepared six-file child was reused after verifying its parent, complete diff, matching research tree and 19 source/26 asset manifest entries. No GitHub/HF history merge or protected-ref change was made.

Actual control is immutable GitHub release 2e59f1b06980aa467da0cde94fa33c3f9a0d4d51 under /app/research_pass1/release. Development imports the integrated root under /app, with only explicit per-encode RNG selection added. Both development sources use the same root decoder. The original model is SHA256 ffaf24aa5447b3602c53a5c4c8df14a68ee16e6bded39331d973f38d0f1ea62f. Python 3.11.17, NumPy 2.4.6, Numba 0.68.0, Pillow 11.3.0; every persistent source process pinned CPU 0 and numerical thread variables one. Metadata preserves import paths, exact preset dictionaries, thread settings, input/model/source hashes, warmups and request specifications. Only an absolute path to the unchanged model is supplied as a config override; internal MSE is asserted enabled for every source.

Smoke: cat, Eiffel and googlewiki, four presets, three sources, one repeat (36 measured pairs), followed by the remaining 22 main/holdout images, release/restored pair once (176 pairs). Their 12+88 distinct comparisons form the complete 100-case gate. Timing: cat, Eiffel, flower, googlewiki, Mario64, Monet, ratedistortion and seagull; four presets; three sources; three rotating repeats (288 measured pairs). There were 500 measured encode/decode pairs across these jobs and 32 full-call warmup pairs. Source order rotates by trial/image/preset, and each variant occupies every order position over three timing trials.

Each codec timer covers the entire compress/decompress call. Encode includes preprocessing inside the codec, initialization, search, model/cache access, serialization/LZMA, output reconstruction/final resize and internal MSE. Decode includes entropy parsing, reconstruction, conversion and final original-size resize. File opening/EXIF/RGB materialization, independent scoring, hashes/patch parsing, IPC, HTTP and logging are outside both timers. No new large public benchmark, training or concurrent research job ran. Shared public traffic/scheduling remains possible noise. Windows local results are not pooled with HF timing.

Every stream is independently decoded by its source; preview equality, dimensions, internal/external MSE and repeat determinism pass. All 96 timed release/restored comparisons match. The 32 current-RNG case streams remain different and reproduce historical current defaults. Compatibility also passed fresh public compression with enabled MSE and the retained old v0 public decode input; the default Balanced RGBA stream remains byte-identical to the predeployment response. Invalid/unauthenticated private requests remain rejected (400/403).

The fresh predeployment database export passed quick_check and retains all 6,396 rows (including 996 stopped-run rows), SHA256 ceb40d7366f5c8291b1df2e09b848aed5702b6fb85eae09f0c06b3ffffd833a3. The accepted rebuild loaded the 6,200-row repository database, SHA256 48f73a898571602747ec5ba555473a62ad8bb53138d78ed8f42f630189de20e1. No older backup was restored over live records; exports remain ignored locally. Controlled research results are separate from this public SQLite history.

## Reproduction

The existing implementation/local gate is in [cfac4f84121adc5b825c5d77abe326a84bc5701d](https://github.com/EgeEken/PBC/commit/cfac4f84121adc5b825c5d77abe326a84bc5701d). See README.md for dependencies, input restoration and the archived compatibility stream. The fresh local 100-case gate was not repeated because no codec source changed. With explicit operational authorization and a verified applicable parent, commands used were:

```text
python research_pass9/run_hf.py deploy
python research_pass9/check_public_api.py
python research_pass9/run_hf.py smoke
python research_pass9/run_hf.py equivalence
python research_pass9/run_hf.py benchmark
python research_pass9/analyze.py
```

Do not redeploy or overwrite existing evidence merely to read this report. Raw responses hf_smoke.json, hf_equivalence.json and hf_comparison.json, database copies, images and payloads stay ignored. comparison_summary.json retains all 100 equivalence pairs, 96 timed source/cases with three timing samples each, exact metadata/provenance, whole-trial means and raw-response hashes. Table generation is reproducible without executing codecs using analyze.py against those saved responses. The optional decoder profile and all new RD mechanisms were outside this completion.
