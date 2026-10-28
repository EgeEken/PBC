# Pass 16 exact residual order statistics

Candidate `pbc31_feature_histogram` adds one optimization to the Pass 15 control: integer residual percentiles and quantized unique counting share a bounded histogram. All other feature expressions retain their NumPy definitions and reduction order. The histogram selects the same lower/upper order statistics and uses the same linear interpolation branch as NumPy; `rint(value / 2)` retains ties-to-even rounding. The float32 vector order, normalization, model, presets, search, RNG, patch geometry and v0 framing are unchanged.

Only integer target/canvas arrays with residual range at most 65,535 and magnitudes at most 2**52 take the histogram path. Other inputs retain the original percentile/unique expressions. This keeps fractional feature inputs and unusually large residual ranges exact while bounding allocation. Production target and canvas arrays are int32. The helper uses the existing Numba dependency, no fastmath, no shared mutable state, and no new configuration field. It allocates at most 512 KiB of histogram storage plus small fixed arrays. First-use Numba compilation is outside the warmed speed claim.

## Correctness

810 direct feature cases passed byte equality for all 32 float32 elements and exact logits, value outputs and selected actions. They cover 3/4 channels, integer and floating dtypes, constants, high contrast, negative residuals, clipped/overshooting canvases, fractional values, one-pixel boundaries, full and partial rectangles, and noncontiguous reversed/strided views. Another 78 tests compare float64 percentiles byte-for-byte, including small counts, interpolation boundaries, negative values and large integer offsets. Three wide-range tests exercise the original NumPy path.

The repeated screen passed all 48 paired complete compressed byte strings, SHA-256 values, unpacked ordered-body hashes, patch counts, decoded RGB hashes, final post-LZMA bytes/bpp, internal MSE and independently scored original-resolution RGB MSE. Four additional diagnostic pairs compared every feature vector and policy output plus each sampled proposal and NumPy generator state.

The frozen candidate then passed all 100 main/holdout image-preset pairs in both separate diagnostic and uninstrumented evaluations. Diagnostics compare feature bytes to the original extractor inside each worker and compare ordered feature/policy/proposal/RNG hashes across workers. Complete base64-encoded compressed bytes are compared directly in the parent before discarding that field from saved rows. Identical uncompressed stream bodies establish ordered patch-payload equality. The decoder source is unchanged.

After integration, the feature function and histogram-kernel ASTs matched the frozen experimental implementation, and all 810 feature/logit/action checks passed again. Sixteen additional synthetic RGB/RGBA image-preset cases matched byte strings, decoded pixels, image mode/size and metrics across controls. Serial and two-thread concurrent execution matched in each isolated worker. Each kernel call owns its histogram. Worker RSS/peak working-set readings are saved in integrated_edges.json; they include the whole Python/Numba process and are not claimed as isolated kernel allocation measurements.

## Full encoder screen

Cat, Eiffel, Monet and Seagull; all four untouched presets; three warmed interleaved repetitions. Each variant runs in its own persistent process pinned to logical P-core 12, one numerical thread, Python 3.11.16 / NumPy 2.4.6 / Numba 0.68.0 / Pillow 11.3.0. Workers preload EXIF-transposed RGB images and warm all four presets. Timers wrap the entire compress call, including preparation, internal MSE and LZMA. Loading, IPC, hashing, external scoring and diagnostics are outside the timed call. Values below are means of per-image medians, including the overall row.

| Preset | Integral-fused control (s) | Feature candidate (s) | Encode reduction |
| --- | ---: | ---: | ---: |
| compression | 0.429673 | 0.414341 | 3.57% |
| balanced | 0.463337 | 0.448955 | 3.10% |
| quality | 0.499576 | 0.479800 | 3.96% |
| high_quality | 0.605540 | 0.577574 | 4.62% |
| overall | 0.499532 | 0.480167 | 3.88% |

All 16 case medians improved. Mean within-case CV: control 1.76%, candidate 2.25%. Case reductions range 1.02% to 8.24%. Raw per-repetition times remain in screen.json. An earlier screen overlapped a diagnostic microbenchmark; the entire screen_overlap_discarded.json run is excluded. The table uses its uncontended replacement only.

| Image/preset | Control median (s) | Candidate median (s) | Reduction | Control CV | Candidate CV |
| --- | ---: | ---: | ---: | ---: | ---: |
| cat.jpg/compression | 0.613376 | 0.589644 | 3.87% | 2.22% | 2.12% |
| cat.jpg/balanced | 0.777159 | 0.769218 | 1.02% | 1.24% | 0.78% |
| cat.jpg/quality | 0.649963 | 0.633246 | 2.57% | 0.60% | 1.75% |
| cat.jpg/high_quality | 0.738472 | 0.706505 | 4.33% | 2.25% | 1.39% |
| eiffel.png/compression | 0.295522 | 0.271159 | 8.24% | 2.14% | 2.60% |
| eiffel.png/balanced | 0.246858 | 0.232534 | 5.80% | 1.89% | 0.83% |
| eiffel.png/quality | 0.260616 | 0.247701 | 4.96% | 1.34% | 0.80% |
| eiffel.png/high_quality | 0.547706 | 0.525846 | 3.99% | 1.26% | 1.55% |
| monet.jpg/compression | 0.254689 | 0.247200 | 2.94% | 1.15% | 0.73% |
| monet.jpg/balanced | 0.434866 | 0.424818 | 2.31% | 1.21% | 1.88% |
| monet.jpg/quality | 0.400112 | 0.383783 | 4.08% | 0.50% | 1.59% |
| monet.jpg/high_quality | 0.492236 | 0.456873 | 7.18% | 1.24% | 4.74% |
| seagull.jpg/compression | 0.555106 | 0.549360 | 1.04% | 1.65% | 2.26% |
| seagull.jpg/balanced | 0.394467 | 0.369252 | 6.39% | 1.16% | 10.59% |
| seagull.jpg/quality | 0.687614 | 0.654470 | 4.82% | 4.77% | 1.07% |
| seagull.jpg/high_quality | 0.643747 | 0.621071 | 3.52% | 3.57% | 1.26% |

## Conditional full-set confirmation

The 20 main and five holdout images each ran all four presets. These one-pass timings are diagnostic confirmation, not an additional repeated speed estimate. The repeated screen above is the performance claim.

| Image | Control mean (s) | Candidate mean (s) | Reduction |
| --- | ---: | ---: | ---: |
| 15k_kills.png | 0.308094 | 0.291004 | 5.55% |
| cat.jpg | 0.672870 | 0.654678 | 2.70% |
| cathedral.png | 0.886770 | 0.877928 | 1.00% |
| coast.jpg | 0.847122 | 0.823049 | 2.84% |
| ducklings.jpg | 0.775042 | 0.757885 | 2.21% |
| ducks.jpg | 0.490180 | 0.462890 | 5.57% |
| eiffel.png | 0.336977 | 0.326098 | 3.23% |
| field.jpg | 0.805536 | 0.785975 | 2.43% |
| flag.jpg | 0.643345 | 0.625304 | 2.80% |
| flower.jpg | 0.616596 | 0.610486 | 0.99% |
| flowers.jpg | 0.899140 | 0.865130 | 3.78% |
| googlewiki.png | 0.439375 | 0.417294 | 5.03% |
| mario64.png | 0.463617 | 0.449315 | 3.08% |
| monet.jpg | 0.391359 | 0.367241 | 6.16% |
| monkey.png | 0.417302 | 0.404059 | 3.17% |
| old_man_standing.jpg | 0.085082 | 0.079751 | 6.27% |
| panorama.png | 0.452983 | 0.427550 | 5.61% |
| park.jpg | 0.711211 | 0.698699 | 1.76% |
| perlman.png | 0.450266 | 0.439391 | 2.42% |
| pigeon.JPG | 0.842349 | 0.842496 | -0.02% |
| pigeons.JPG | 0.723739 | 0.701320 | 3.10% |
| ratedistortion.png | 0.262824 | 0.249996 | 4.88% |
| seagull.jpg | 0.547610 | 0.530471 | 3.13% |
| stockpoint.jpg | 0.297517 | 0.284071 | 4.52% |
| sunset.jpg | 0.833683 | 0.799987 | 4.04% |

Full-set case timing wins: 90/100. Individual slower observations: cat.jpg/high_quality (-1.22%); cathedral.png/high_quality (-1.31%); eiffel.png/high_quality (-1.78%); flower.jpg/balanced (-3.20%); mario64.png/high_quality (-0.44%); monkey.png/compression (-2.52%); pigeon.JPG/compression (-0.03%); pigeon.JPG/balanced (-2.45%); pigeon.JPG/high_quality (-0.65%); pigeons.JPG/quality (-1.89%).

## Feature-only timings

These medians of 15 interleaved extractor calls are isolated microbenchmarks; they are not full-encode reductions. They were rerun after all full-codec measurements, with no other diagnostic or benchmark worker running. The original overlapping micro run is retained as micro_overlap_discarded.json and excluded.

| Square region | Original extractor (ms) | Candidate extractor (ms) |
| --- | ---: | ---: |
| 16 | 0.338 | 0.243 |
| 64 | 0.545 | 0.339 |
| 200 | 2.899 | 1.812 |
| 400 | 16.254 | 10.387 |

Integrated worker warm RSS / peak working set: control 97.99 / 101.81 MiB; candidate 98.72 / 102.31 MiB. These small process-level differences are observational; the histogram allocation itself is bounded at 512 KiB.

## Decision

Promote as the next PBC 3.1 development candidate, retaining fused integral construction. The incremental whole-encode reduction is 3.88% against pbc31_integral_fused, with every preset and screen-case median improved and all 100 full-set outputs identical. No sampler experiment was needed. No HF deployment, publication, tag, protected-branch modification, model training or RD change occurred. The previous frozen baseline and Pass 15 source/evidence remain available.

Local staging was updated only after the full correctness, timing and edge gates passed. The complete previous artifact is retained at `C:\Users\EGE\Desktop\Coding\Personal Projects\Completed Projects\Image Processing\Probabilistic Brush Compression\New Demo\HF Space\PBC\PBC3_1_Release_Staging_Pass15_Integral_Fused`. The replacement passed the existing isolated extraction/install, flat-module and package re-export, four-preset, deterministic RGB/RGBA, frozen Cat SHA, and immutable-PBC-3.0 v0 decode validation. New ZIP SHA-256: `08855cb3369000fd50d38fbb41d65920899998d366812c04519302383d873591`.

Exact source/model/data/runtime configurations and ignored raw-evidence SHA-256 values are in provenance.json. Restore the declared datasets through research/prepare_assets.py. Prepare the immutable control before reproduction:

```
research_pass16/.venv/Scripts/python.exe research_pass16/run_benchmark.py --prepare-only
research_pass16/.venv/Scripts/python.exe research_pass16/check_features.py
research_pass16/.venv/Scripts/python.exe research_pass16/check_order_stats.py
research_pass16/.venv/Scripts/python.exe research_pass16/run_benchmark.py
research_pass16/.venv/Scripts/python.exe research_pass16/run_benchmark.py --full
research_pass16/.venv/Scripts/python.exe research_pass16/check_edges.py --integrated
```

Do not overlap diagnostic microbenchmarks with timed workers. build_release_staging.py requires passed local results and an untouched known Pass 15 staging artifact; it refuses existing candidate/archive destinations. It is an explicit local staging action, not part of benchmark execution.
