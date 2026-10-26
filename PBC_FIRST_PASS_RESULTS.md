# PBC first-pass results

## Decision

Use the verified immutable public release as the research baseline. The historical old runtime is equivalent on the tested default-preset sets but is not source-identical. Current root is a combined trajectory/performance variant, not clean release PBC 3.0. No PBC 3.1 release success is established by this pass.

Phases 1–2 passed the default-preset comparison gate. All five requested experiment categories ran locally. A uses the full main and holdout datasets; B profiles three representative images; C/D screen the balanced preset on flower.jpg (smooth photographic/bokeh), monet.jpg (textured art), and googlewiki.png (text/synthetic); E screens these three images across all four presets. No new algorithm was enabled in production. C/D are pilot proposals, not exhaustive optimization of their candidate spaces.

## Attribution diagnostic

Stream inspection checks all 100 first-repeat release cases against old and current. Old equal headers/init/payloads: 100/100. Current matching initial three channel patches: 100/100. Earliest differing patch indices are retained in stream_comparison.json. This localizes differences beyond initialization without calling them numerical noise.

Replacing only current root's RNG constructor with NumPy default_rng in an isolated process produced release-identical files on 12/12 representative image/preset cases. This is a causal ablation for attribution, not a release comparison or proposed improvement. Default current root comparisons still use its real custom RNG. No production RNG was changed.

## A — Working-resolution round trip

Actual release preprocessing was used: EXIF-normalized RGB -> YCbCr -> PIL bicubic working-size image -> RGB -> PIL bicubic source-size output, with no quantization or residual patches. Four offline pixel limits were measured without changing defaults.

This is a preprocessing round-trip diagnostic, not a strict mathematical lower bound on every possible working-grid representation: precompensated samples could differ from the codec's direct downsample, and YCbCr integer conversion adds error even without resizing. Ratios below compare means, not a causal partition of the final codec error.

| Set | Preset | Pixel limit | Roundtrip RGB MSE | Final release MSE | Ratio % |
| --- | --- | --- | --- | --- | --- |
| hpt_data | compression | 250000 | 128.3761 | 278.4012 | 46.11 |
| hpt_data | compression | 500000 | 82.3647 | 278.4012 | 29.58 |
| hpt_data | compression | 1000000 | 42.8490 | 278.4012 | 15.39 |
| hpt_data | compression | 2000000 | 24.1440 | 278.4012 | 8.67 |
| hpt_data | balanced | 250000 | 128.3761 | 193.4861 | 66.35 |
| hpt_data | balanced | 500000 | 82.3647 | 193.4861 | 42.57 |
| hpt_data | balanced | 1000000 | 42.8490 | 193.4861 | 22.15 |
| hpt_data | balanced | 2000000 | 24.1440 | 193.4861 | 12.48 |
| hpt_data | quality | 250000 | 128.3761 | 160.3306 | 80.07 |
| hpt_data | quality | 500000 | 82.3647 | 160.3306 | 51.37 |
| hpt_data | quality | 1000000 | 42.8490 | 160.3306 | 26.73 |
| hpt_data | quality | 2000000 | 24.1440 | 160.3306 | 15.06 |
| hpt_data | high_quality | 250000 | 128.3761 | 143.1334 | 89.69 |
| hpt_data | high_quality | 500000 | 82.3647 | 143.1334 | 57.54 |
| hpt_data | high_quality | 1000000 | 42.8490 | 143.1334 | 29.94 |
| hpt_data | high_quality | 2000000 | 24.1440 | 143.1334 | 16.87 |
| hpt_data_val | compression | 250000 | 27.5209 | 169.2691 | 16.26 |
| hpt_data_val | compression | 500000 | 15.4847 | 169.2691 | 9.15 |
| hpt_data_val | compression | 1000000 | 7.6142 | 169.2691 | 4.50 |
| hpt_data_val | compression | 2000000 | 3.9678 | 169.2691 | 2.34 |
| hpt_data_val | balanced | 250000 | 27.5209 | 99.0308 | 27.79 |
| hpt_data_val | balanced | 500000 | 15.4847 | 99.0308 | 15.64 |
| hpt_data_val | balanced | 1000000 | 7.6142 | 99.0308 | 7.69 |
| hpt_data_val | balanced | 2000000 | 3.9678 | 99.0308 | 4.01 |
| hpt_data_val | quality | 250000 | 27.5209 | 50.6763 | 54.31 |
| hpt_data_val | quality | 500000 | 15.4847 | 50.6763 | 30.56 |
| hpt_data_val | quality | 1000000 | 7.6142 | 50.6763 | 15.03 |
| hpt_data_val | quality | 2000000 | 3.9678 | 50.6763 | 7.83 |
| hpt_data_val | high_quality | 250000 | 27.5209 | 40.2472 | 68.38 |
| hpt_data_val | high_quality | 500000 | 15.4847 | 40.2472 | 38.47 |
| hpt_data_val | high_quality | 1000000 | 7.6142 | 40.2472 | 18.92 |
| hpt_data_val | high_quality | 2000000 | 3.9678 | 40.2472 | 9.86 |

Per-image source/working dimensions, all preset comparisons and round-trip MSE are in resolution_floor.csv/json. Image-type grouping and size diagnostics are in resolution_floor_by_type.json. These controls do not measure full-encode RD or latency at larger limits, and do not justify raising production resolution.

At the production 250k limit, grouped by descriptive image type (main and holdout combined; group counts are small):

| Image type | Images | Mean source MP | Roundtrip MSE | Balanced release MSE |
| --- | --- | --- | --- | --- |
| animal photograph | 5 | 8.089 | 92.4725 | 166.7333 |
| architecture photograph | 2 | 6.116 | 176.8579 | 238.5636 |
| fabric photograph | 1 | 6.469 | 8.1169 | 27.7573 |
| flower/bokeh photograph | 2 | 10.266 | 78.3962 | 138.8248 |
| game/synthetic | 1 | 0.995 | 89.1794 | 185.7115 |
| landscape photograph | 5 | 9.310 | 141.9167 | 198.1573 |
| portrait photograph | 3 | 1.755 | 60.6675 | 107.0955 |
| puppet/meme photograph | 1 | 0.306 | 3.0003 | 35.6971 |
| snow/urban photograph | 1 | 12.008 | 14.3304 | 51.3971 |
| text/synthetic | 3 | 0.345 | 228.5819 | 339.8142 |
| textured artwork | 1 | 0.845 | 40.2973 | 144.3549 |

## B — Production profiling

The actual current root compress/decompress pipeline was warmed and traced twice on each of the three screening images, balanced preset. Existing nested TimingTrace spans separate preprocessing, initialization, residual/integral/anchor work, proposals, features, MLP, quantization/resampling, application, channel errors, serialization, LZMA and output conversion. Exclusive durations below avoid summing nested inclusive spans. Tracing overhead remains; release speed conclusions use the separate uninstrumented benchmark.

| Operation | Phase | Mean exclusive s | Share of traced operation % |
| --- | --- | --- | --- |
| encode | action.search.integral_error | 0.103387 | 21.77 |
| encode | action.search.sample_boxes | 0.039669 | 8.35 |
| encode | encode.metric.final_mse | 0.038719 | 8.15 |
| encode | action.filler.policy.feature_extract | 0.032741 | 6.89 |
| encode | encode.patch.score | 0.031886 | 6.71 |
| encode | action.search.visible_error | 0.025381 | 5.34 |
| encode | encode.serialize.patches | 0.022247 | 4.68 |
| encode | encode.reconstruct.final_resize | 0.021697 | 4.57 |
| encode | prepare.downsample_color | 0.018007 | 3.79 |
| encode | encode.entropy_pack | 0.015749 | 3.32 |
| encode | action.downsample_init.score_candidate | 0.015148 | 3.19 |
| encode | prepare.input_normalize | 0.014244 | 3.00 |
| decode | decode.patch.read | 0.027101 | 42.46 |
| decode | decode.reconstruct.final_resize | 0.022626 | 35.44 |
| decode | decode.patch.resample_apply | 0.010177 | 15.94 |
| decode | decode.reconstruct.canvas_to_image | 0.001551 | 2.43 |
| decode | decode.entropy_unpack | 0.001470 | 2.30 |
| decode | decode.initialize_canvas | 0.000412 | 0.65 |
| decode | decode.patch_loop | 0.000366 | 0.57 |
| decode | head.decoder | 0.000038 | 0.06 |
| decode | decode.parse_header | 0.000017 | 0.03 |
| decode | decode.open_body | 0.000009 | 0.01 |
| decode | decode.to_canvas | 0.000003 | 0.01 |
| decode | decode.validate_magic_version | 0.000002 | 0.00 |

Complete inclusive/exclusive phase tables and raw events/counters are retained in profile_summary.csv and profile_*.json. This identifies where to investigate, not a measured optimization.

Surgical leads: compile the Python proposal sampling/prescore loop together while preserving its chosen RNG sequence exactly; investigate reusing visible-error state after a local patch instead of repeatedly rebuilding full-channel arrays; inspect full feature extraction (the production model uses percentiles and unique) before changing MLP code; specialize bit-grid read/write if serialization/parsing is material. PIL bicubic and NumPy sort/cumsum/BLAS calls already execute native code. A C/Rust wrapper alone does not remove their work. The workspace already contains rejected/reverted integral-fusion experiments; this profile does not justify repeating them without a different mechanism and HF evidence. Any Numba replacement needs byte/pixel equivalence tests and independent HF timing. LZMA stays in place; post-wrapper bytes remain the rate metric. No native rewrite was attempted.

## C — Post-encode pruning

Every available iterative single-patch removal was tested for each of the three balanced trajectories (63 candidates in total). A requested budget of 50 does not imply 50 accepted patches: the production filler can stop earlier when its selected proposal has no positive reduction. Actual emitted patch counts are retained in all benchmark rows. Every candidate had its patch count/header rewritten, was entropy-packed, and was fully decoded before RGB MSE evaluation. Initialization patches were retained.

Strict single removal chooses the smallest final file among deletions with MSE no greater than the full trajectory. Tolerance permits at most +0.1% relative RGB MSE. Greedy is one reverse pass, accepting each deletion only when final MSE does not rise and post-LZMA bytes strictly decrease. These variants start independently from the same complete trajectory. Selection scans and replay/serialization/metric computation are all included in extra encode time; candidate decode measurement is part of this optimization cost.

## D — Frozen-geometry backward refitting

Independent last-5, last-15 and complete backward passes keep patch geometry, order, cell size and maximum palette bitcount fixed. A last-15 pass revisits all available patches when fewer than 15 were emitted. Original palette-mask length and positive-bias semantics are retained; palette bounds, generated mask and indices may be regenerated within that representation.

For each revisited patch, the decoded internal canvas minus that patch's deterministic delta estimates the other fixed contributions. The ordinary bicubic residual-grid/palette/quantization constructor proposes replacement values against the final working target. Every proposal is then serialized and fully decoded, and accepted only for strictly lower original-size RGB MSE with no increase in actual post-LZMA bytes. This is a conservative single refit proposal per patch, not a solved global coefficient optimizer. Subtraction is used only to propose values; final acceptance always uses proper replay. Preset warmup is disabled and the script rejects multistage/warmup streams.

## C/D screening results

Percent changes compare equal-weight means across the three balanced images. Raw/final byte means and mean bpp have different weighting. Decode measurements here are single screened calls and can be noisy; they are not release timing evidence. Baseline encode cost is the median of the three uninstrumented release calls plus all measured optimization overhead.

| Variant | MSE % | bpp % | raw bytes % | final bytes % | encode % | decode % | strict RD image wins | Classification |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C_greedy | -0.025 | -0.174 | -0.356 | -0.421 | +546.157 | +5.988 | 3/3 | Latency tradeoff |
| C_single_strict | -0.020 | -0.147 | -0.130 | -0.222 | +532.238 | -3.265 | 3/3 | Latency tradeoff |
| C_single_tolerance_0.1pct | -0.015 | -0.208 | -0.147 | -0.284 | +532.238 | -1.332 | 2/3 | Latency tradeoff |
| D_refit_15 | -0.013 | -0.005 | +0.000 | -0.055 | +528.799 | +0.318 | 1/3 | Latency tradeoff |
| D_refit_5 | -0.013 | -0.005 | +0.000 | -0.049 | +279.224 | +0.218 | 1/3 | Latency tradeoff |
| D_refit_all | -0.021 | -0.010 | +0.000 | -0.059 | +795.056 | +2.963 | 2/3 | Latency tradeoff |

| Image | Variant | MSE | bpp | raw bytes | final bytes | extra encode s | accepted |
| --- | --- | --- | --- | --- | --- | --- | --- |
| flower.jpg | C_single_strict | 81.6814 | 0.012744 | 24784 | 13581 | 1.720 | 1 |
| flower.jpg | C_single_tolerance_0.1pct | 81.6814 | 0.012744 | 24784 | 13581 | 1.720 | 1 |
| flower.jpg | C_greedy | 81.6335 | 0.012590 | 24384 | 13417 | 1.717 | 4 |
| flower.jpg | D_refit_5 | 81.6000 | 0.012793 | 24967 | 13633 | 1.185 | 2 |
| flower.jpg | D_refit_15 | 81.5963 | 0.012788 | 24967 | 13628 | 2.093 | 3 |
| flower.jpg | D_refit_all | 81.5963 | 0.012788 | 24967 | 13628 | 2.132 | 3 |
| monet.jpg | C_single_strict | 144.3542 | 0.175743 | 23947 | 18565 | 2.829 | 1 |
| monet.jpg | C_single_tolerance_0.1pct | 144.4045 | 0.175231 | 23917 | 18511 | 2.829 | 1 |
| monet.jpg | C_greedy | 144.3538 | 0.175667 | 23932 | 18557 | 2.813 | 2 |
| monet.jpg | D_refit_5 | 144.3549 | 0.175837 | 23961 | 18575 | 0.477 | 0 |
| monet.jpg | D_refit_15 | 144.3549 | 0.175837 | 23961 | 18575 | 1.493 | 0 |
| monet.jpg | D_refit_all | 144.2718 | 0.175799 | 23961 | 18571 | 4.579 | 2 |
| googlewiki.png | C_single_strict | 845.2796 | 0.651202 | 135220 | 53803 | 1.793 | 1 |
| googlewiki.png | C_single_tolerance_0.1pct | 845.2796 | 0.651202 | 135220 | 53803 | 1.793 | 1 |
| googlewiki.png | C_greedy | 845.2796 | 0.651202 | 135220 | 53803 | 1.977 | 1 |
| googlewiki.png | D_refit_5 | 845.4414 | 0.652255 | 135263 | 53890 | 1.665 | 0 |
| googlewiki.png | D_refit_15 | 845.4414 | 0.652255 | 135263 | 53890 | 2.714 | 0 |
| googlewiki.png | D_refit_all | 845.4414 | 0.652255 | 135263 | 53890 | 2.762 | 0 |

Per-image candidates are in screen_results.csv/json; all individual deletion results are in pruning_individual_tests.csv/json. candidate_control_dominance.json compares each retained C/D operating point against the observed prefix/fresh-budget controls. Nondominance against this small observed set is not proof of an improved continuous RD frontier.

| Variant | Dominated by an observed same-image prefix/fresh-budget RD point | Cases |
| --- | --- | --- |
| C_greedy | 0 | 3 |
| C_single_strict | 1 | 3 |
| C_single_tolerance_0.1pct | 1 | 3 |
| D_refit_15 | 0 | 3 |
| D_refit_5 | 0 | 3 |
| D_refit_all | 0 | 3 |

## E — Prefix and patch-budget analysis

Existing completed release trajectories were serialized/decoded at initialization-only and selected iterative budgets through the full trajectory (including late budgets). No search reran for a prefix. Fresh normal encodes were also run at selected identical smaller budgets; changing patch_count changes search interpolation and feature step fraction.

Of 33 same-budget fresh/prefix pairs, 3 are byte-identical. Individual RD points, including poor or counterproductive late prefixes, are in screen_results.csv; exact fresh/prefix comparisons are in prefix_budget_comparison.json. A prefix extracted after a complete encode still pays for the complete search plus extraction. Fresh-budget encode time is reported separately, without inheriting that full trajectory cost. This diagnostic alone is not a new stopping policy.

## Scope, reproducibility and next iteration

No C/D candidate is accepted as a release improvement from this screen. No C/D variant was promoted to full main/holdout evaluation: strict greedy pruning changes mean MSE by only -0.025% and mean bpp by -0.174% while increasing total encode cost by about 546%; complete backward refitting changes MSE by -0.021% and bpp by -0.010% while increasing encode cost by about 795%. These are small RD leads with substantial latency penalties. A three-image balanced pilot cannot establish four-preset success. Broader holdout confirmation is still required before any claimed generalization. No combined prune/refit/stopping variant or large combinatorial sweep was run.

Next work should preserve the immutable baseline and frozen stream artifacts, then use the profile to select one output-preserving optimization with a credible single-core/HF cost reduction. For an RD lead, improve or bound its incremental cost before promoting it to a full dataset run. Do not raise resolution, change RNG, or alter patch budget and describe that alone as a release win.

Commands from repository root, after the baseline: `python research_pass1/experiments.py compare`, `python research_pass1/rng_diagnostic.py`, `python research_pass1/experiments.py floor`, `python research_pass1/experiments.py profile`, `python research_pass1/experiments.py screen`, `python research_pass1/validate_backend.py`, then `python research_pass1/generate_reports.py`. Run sequentially, with no concurrent CPU benchmark. Profiling, controls and search reuse remain isolated opt-in scripts.

Exact commit, expanded configs, model hashes, input lists, repeat/warmup/thread environment and original per-case baselines are in baseline_metadata.json/baseline.jsonl. experiment_provenance.json records actual before/after source/model/dataset/config/environment snapshots for the final diagnostic runs; independent C/D/E variants share a frozen screening-job snapshot window. Profiling uses two traced repeats after one ordinary warmup per image. Screening uses one pass per variant; prefix/fresh-budget evaluations precede pruning/refitting and warm their common runtime paths. Script/source hashes and before/after repository status are included in the artifact manifest. Release source and model remain unchanged. No commit, push or deployment was performed.
