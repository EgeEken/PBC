# Bit-exact performance experiments

All candidates are isolated in research_pass2; production source/config defaults are unchanged. Each candidate is compared with the same current-root RNG and implementation, disabled versus enabled. No representation/search/model changes are combined with these timings.

- `channel_sum`: narrow Numba integer reduction for the existing full-channel visible absolute error. This avoids temporary clip/difference/absolute arrays and preserves exact int64 sums. It does not implement an incrementally maintained residual map or repeat the rejected integral-image fusion. Local clipped-region updates were deferred because hooking mutation/warmup state would be a broader and riskier change than this allocation-free reduction.
- `grid_numpy`: MSB-first vectorized NumPy bit packing/unpacking, preserving every bit alignment and reader/writer suffix state. It introduces temporary bit arrays proportional to grid cells × bitcount, rather than a native rewrite. Production version-0 syntax and generated palette semantics are unchanged.
- `skip_final_mse`: existing `compute_final_mse=False` configuration. The preview is still constructed and resized, bytes/pixels stay identical, and the result's internal MSE is None. The harness independently decodes and scores. Default True and website behavior remain intact. This is an API/runtime tradeoff, not an RD improvement or newly implemented feature.

## Complete-codec results

Six-image screen, all four presets, three repeats:

| Variant | Cases | Encode % | Encode wins | Decode % | Decode wins |
| --- | --- | --- | --- | --- | --- |
| channel_sum | 24 | -2.5784 | 15 | 0.9981 | 12 |
| grid_numpy | 24 | -4.9533 | 16 | -43.0586 | 24 |
| skip_final_mse | 24 | -13.5068 | 21 | -0.9251 | 13 |

Primary confirmation source: performance_confirmation. Values average per-image/preset medians over three warmed interleaved single-core repeats. Equality checks include complete bytes, decoded RGB pixels, patch order/count, bpp and independently scored MSE. No production randomness is replaced for D.

| Variant | Set | Cases | Exact | Baseline enc s | Variant enc s | Encode % | Baseline dec s | Variant dec s | Decode % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| grid_numpy | hpt_data | 80 | 80 | 0.4765 | 0.4504 | -5.4824 | 0.0887 | 0.0539 | -39.3050 |
| grid_numpy | hpt_data_val | 20 | 20 | 0.3824 | 0.3614 | -5.5020 | 0.0765 | 0.0458 | -40.1161 |
| skip_final_mse | hpt_data | 80 | 80 | 0.4765 | 0.4146 | -12.9839 | 0.0887 | 0.0886 | -0.2029 |
| skip_final_mse | hpt_data_val | 20 | 20 | 0.3824 | 0.3230 | -15.5448 | 0.0765 | 0.0744 | -2.7853 |

All screening results, including candidates not promoted, remain in performance_screen.jsonl/_summary.json/_paired.json. Fixed promotion decisions predate confirmation and holdout in performance_promotion.json. Per-image wins, outliers, repeated-call CV, means and extrema are retained in machine-readable tables, rather than hiding small-image regressions in the aggregate. Small percent changes near observed repeat noise are inconclusive.

Actual-call means and mean per-case repeat CV:

| Set | Variant | Encode mean s | Encode CV % | Decode mean s | Decode CV % |
| --- | --- | --- | --- | --- | --- |
| hpt_data | none | 0.4804 | 3.9490 | 0.0896 | 4.1976 |
| hpt_data | grid_numpy | 0.4523 | 3.5991 | 0.0547 | 5.7905 |
| hpt_data | skip_final_mse | 0.4205 | 4.5332 | 0.0897 | 4.1523 |
| hpt_data_val | skip_final_mse | 0.3262 | 4.0972 | 0.0779 | 7.3257 |
| hpt_data_val | none | 0.3844 | 3.6438 | 0.0771 | 4.7557 |
| hpt_data_val | grid_numpy | 0.3642 | 3.5318 | 0.0459 | 4.7783 |

By original image size, pooling main/holdout cases only for this descriptive size comparison:

| Variant | Original size | Cases | Encode % | Decode % |
| --- | --- | --- | --- | --- |
| grid_numpy | <1 MP | 32 | -9.3846 | -71.1852 |
| grid_numpy | >=1 MP | 68 | -4.4239 | -31.0586 |
| skip_final_mse | <1 MP | 32 | -0.6726 | -1.1714 |
| skip_final_mse | >=1 MP | 68 | -16.8812 | -0.5259 |

The MSE option does not change the decoder; its small measured decode differences are timing variation, not an attributed decode improvement. Grid-I/O decoder gains are independently measured on identical files and all 100 cases. Size grouping is descriptive and confounds content type; it is not a controlled causal image-size experiment.

Separate component profiles use three images, balanced, two repeats after warmup: performance_components.json. Timers from those profiled calls include instrumentation overhead; micro/function improvements alone are not a full-codec speed claim. RSS-after fields reflect cumulative process allocators and rotation, not isolated peak-memory deltas. Grid temporaries are O(grid cells × bitcount) bytes (plus the uint16 output and packing temporaries); channel reduction removes O(working pixels) temporary arrays. No unsupported peak-memory claim is made.

| Variant | Function | Profile self s/case | Profile cumulative s/case | Calls/case |
| --- | --- | --- | --- | --- |
| channel_sum | write_grid | 0.0381 | 0.0446 | 41.3333 |
| grid_numpy | write_grid | 0.0031 | 0.0035 | 41.3333 |
| grid_numpy | channel_sum_error | 0.0211 | 0.0370 | 41.3333 |
| none | write_grid | 0.0399 | 0.0460 | 41.3333 |
| none | channel_sum_error | 0.0226 | 0.0384 | 41.3333 |

Decoder phase traces, three images/four presets/two repeats, separate from untraced latency:

| Variant | Phase | Mean inclusive s | Mean exclusive s |
| --- | --- | --- | --- |
| grid_numpy | decode.patch.read | 0.0051 | 0.0051 |
| grid_numpy | decode.patch.resample_apply | 0.0112 | 0.0112 |
| grid_numpy | decode.reconstruct.final_resize | 0.0237 | 0.0237 |
| none | decode.patch.read | 0.0513 | 0.0513 |
| none | decode.patch.resample_apply | 0.0121 | 0.0121 |
| none | decode.reconstruct.final_resize | 0.0238 | 0.0238 |

## Correctness and limitations

optimization_unit_checks.json tests grid widths 0–9 at all eight initial alignments, empty/short/long grids and suffix-state preservation. The channel kernel additionally tests signed-int16 extrema, clipping, odd dimensions and RGB/RGBA. Production encodes provide the independent meaningful correctness gate across real image trajectories. Kernels are warmed before timings; first calls/compilation are retained separately in metadata. No native parser, percentile replacement, approximate channel accounting, combined speed variant or HF deployment is tested.

Local improvements need a later explicitly authorized cpu-basic experiment before deployment claims: run disabled/enabled versions in the same warmed single-thread process on fixed already-approved server-side inputs, interleave repeats, and check byte/pixel hashes and module/dependency identity. No upload or deployment is authorized by this report.

Reproduce: `python research_pass2/performance.py correctness`, `screen`, `confirm`, and `components`, sequentially. Confirmation reads the frozen promotion decision. Source: optimizations.py; raw rows and before/after snapshots are stored beside each experiment.
