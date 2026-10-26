# Pass 9 completed on HF

Yes: optimized PBC with explicit release RNG reproduces immutable PBC 3.0 exactly on HF in 100/100 image/preset cases (20 main, five holdout; all four presets). Complete file, decoded pixel and ordered patch hashes/counts, RGB MSE and final post-LZMA bpp agree. All 96 repeated timed release/restored comparisons also match.

| Preset | Release encode | Dev release-RNG encode | Dev current encode | Release decode | Dev release-RNG decode | Dev current decode |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Compression | 0.468090 | 0.368027 | 0.360938 | 0.091763 | 0.058645 | 0.065754 |
| Balanced | 0.578009 | 0.426547 | 0.474414 | 0.128845 | 0.054979 | 0.057445 |
| Quality | 0.755442 | 0.550425 | 0.554162 | 0.155399 | 0.062698 | 0.059769 |
| High Quality | 0.915053 | 0.588329 | 0.566305 | 0.292907 | 0.061961 | 0.065506 |
| Overall | 0.679149 | 0.483332 | 0.488955 | 0.167228 | 0.059571 | 0.062119 |

| Preset | Release-RNG dev encode reduction vs 3.0 | Release-RNG dev decode reduction vs 3.0 | Current-RNG dev encode reduction vs 3.0 | Current-RNG dev decode reduction vs 3.0 |
| --- | ---: | ---: | ---: | ---: |
| Compression | +21.38% | +36.09% | +22.89% | +28.34% |
| Balanced | +26.20% | +57.33% | +17.92% | +55.42% |
| Quality | +27.14% | +59.65% | +26.64% | +61.54% |
| High Quality | +35.71% | +78.85% | +38.11% | +77.64% |
| Overall | +28.83% | +64.38% | +28.00% | +62.85% |

These are simultaneous HF cpu-basic measurements: eight fixed images, four presets, three isolated variants, three warmed rotating repeats (288 measured encode/decode pairs). Internal MSE stays enabled. Each table cell averages per-image median full-call times; original-size independent RGB scoring and complete final bytes determine RD. Full tables, per-image regressions, variability and provenance are in PBC_HF_THREE_WAY_BENCHMARK.md and comparison_summary.json.

- Quality bitrate regression is eliminated: all 20 main restored cases match release at 0.29847982134066314 mean bpp and 160.07840745282843 RGB MSE. The recorded Pass 8 custom-RNG excess was 9.909668%; the residual is zero. The eight-image timing subset also matches release exactly in MSE/bpp for every preset.
- Same-stream engineering gains are 28.83% encode and 64.38% decode overall, with substantial aggregate gains in every preset. They combine the retained optimized implementation changes; no historical percentage is added or subtracted.
- Against current-RNG development, restored development has 1.15% lower aggregate median-based encode time and 4.10% lower decode time. These differences are small relative to 13–23% within-case CV and change direction across whole trials. No material whole-codec penalty is demonstrated, but pure RNG overhead is not isolated: different trajectories, patch geometry/counts and payloads change work.
- Restored development wins all 32 encode and 32 decode case medians versus actual release, with no median regressions in this sample. Small wins (Flower Compression encode 3.61%, Seagull Compression decode 1.01%) are within noise. Current RNG retains four encode and two decode median regressions. Cat Compression changes from current +12.55% time (37 patches) to restored −19.14% (19 patches); Eiffel Balanced is restored −29.64% (20 patches), versus current +0.16% (31 patches), not the prior 38.70% slowdown. Seagull Balanced and Rate-distortion Quality also become restored wins.
- Recommend release-compatible RNG as the PBC 3.1 development foundation and primary control for future RD research. Keep the custom generator explicitly selectable for historical reproduction. No public default was changed and no PBC 3.1 release/tag was published.
- This is a faster codec with unchanged release RD, not a strict four-axis improvement. Strictly better RD still needs a separately validated lower-distortion/no-higher-final-rate or lower-rate/no-higher-distortion result against this release-compatible control, including main/holdout and matched-rate evidence. No new compression experiment was started.

HF deployment/measurement: [4237d954f4e1a04951c822c738113a53bbd58e9b](https://huggingface.co/spaces/EgeEken/PBC/commit/4237d954f4e1a04951c822c738113a53bbd58e9b), direct child of 27f6c65225a5496fa065deb586c7e405ecdffc84. The already-prepared minimal six-file commit was reused after ancestry/tree/source checks. Immutable reference: GitHub 2e59f1b06980aa467da0cde94fa33c3f9a0d4d51. Implementation/local gate: [cfac4f84121adc5b825c5d77abe326a84bc5701d](https://github.com/EgeEken/PBC/commit/cfac4f84121adc5b825c5d77abe326a84bc5701d). Earlier deferral/public-run log: [bedfa1409132d77e0e6f145bfd7fb3d13993a54d](https://github.com/EgeEken/PBC/commit/bedfa1409132d77e0e6f145bfd7fb3d13993a54d). Completion reports are published on the same permanent research/experiments branch; their exact commit URL is supplied in the final handoff.

Fresh public compression retains its prior stream and displayed MSE; the retained existing v0 stream decodes identically through the live API. Public presets, model, UI, format and decoder are unchanged. Fresh predeployment database backup preserves 6,396 rows and passed integrity checks; the accepted rebuild reset loaded 6,200 repository rows. No older backup was restored. Raw streams/responses, databases, datasets, screenshot, caches and secrets are excluded from Git. The stopped 9,800-call run was not restarted.

Reproduce the saved analysis with `python research_pass9/analyze.py`; exact authorized deployment/public/smoke/gate/timing commands and dependencies are in README.md and PBC_HF_THREE_WAY_BENCHMARK.md. Future RD controls should explicitly select `rng_mode="release"` until a separate public-default decision is authorized.
