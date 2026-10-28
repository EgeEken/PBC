# Pass 16 learned feature extraction profile

Control: `pbc31_integral_fused`, research commit `c0d0339a509ceb4027574fc388cef18a643b14e5`. Before experimentation, all eight runtime files and `patch_policy.npz` matched the commit byte-for-byte. The fused `pbc3_ops.py` SHA-256 is `0aa08cf7e1716cf0f28740aa289e45484b38622b67c45e39b81b199f881a2852`; model SHA-256 is `ffaf24aa5447b3602c53a5c4c8df14a68ee16e6bded39331d973f38d0f1ea62f`.

The deployed policy has `cheap=False`, hidden width 128, no factored heads, and requests the complete 32-element `FEATURE_NAMES` vector. `LearnedFiller._featurize()` calls `extract_features()` once per candidate and stacks its float32 result. All four presets use one candidate and top_k=1. `extract_cheap()` and factored-policy state extraction do not execute. The five percentiles and unique-value calculation do execute.

Active feature order:

```
w_frac h_frac area_frac log_aspect channel step_frac q
res_mean res_std res_abs_mean res_rms res_min res_max
res_p05 res_p25 res_p50 res_p75 res_p95
pos_frac neg_frac zeroish_frac uniq_frac
mean_abs_dx mean_abs_dy grad_energy lap_abs_mean
target_mean target_std canvas_mean canvas_std before_mse before_mae
```

Profile: Cat, Eiffel, Monet and Seagull, Quality and High Quality; one full warmup per case and three uninstrumented full compress calls. A separate TimingTrace encode measured inclusive phase costs; another separate encode used a feature-function-only Python line tracer. Line costs include tracer overhead and are diagnostic, not speed-comparison measurements. Full-call median averaged 543.1 ms across the eight cases. Python 3.11.16, NumPy 2.4.6, Numba 0.68.0, Pillow 11.3.0; i9-12900H, logical P-core 12, numerical threads one.

| Phase | Mean milliseconds/encode | Mean calls/encode |
| --- | ---: | ---: |
| Feature extraction, including stacking/cast | 58.793 | 25 |
| Feature normalization | 0.201 | 25 |
| Policy forward pass | 1.735 | 25 |
| Proposal sampling and box bookkeeping | 100.708 | 25 |

The line-profile totals divided by eight encodes are 15.151 ms for percentiles, 7.434 ms for quantized unique counting, 7.803 ms for both Laplacians, 6.043 ms for both gradients, and 6.593 ms for the three standard deviations. These come from a separate instrumented run and must not be added to the TimingTrace totals as independent work.

Each extraction creates float64 target and canvas regions, their residual, clipped visible residual, absolute residual, and statistics temporaries. Search reuses the per-channel visible absolute-error array through SearchState, but features need signed unclipped residuals and individual box regions. That reused search array cannot replace the feature residual. Normalization and inference are small next to region statistics. The selected experiment targets percentile and unique order statistics together, leaving means, standard deviations, gradients, normalization and the model untouched.

Reproduce from this checkout using the Pass 15 numerical runtime:

```
research_pass16/.venv/Scripts/python.exe research_pass16/run_benchmark.py --prepare-only
research_pass16/.venv/Scripts/python.exe research_pass16/profile.py
research_pass16/.venv/Scripts/python.exe research_pass16/check_features.py
research_pass16/.venv/Scripts/python.exe research_pass16/check_order_stats.py
```

`profile.json`, `feature_checks.json`, and `micro.json` retain raw cases, line costs, hashes and configurations locally and are ignored by Git. `check_order_stats.py` verifies the retained control source files against the control commit before running. An offline uv environment was created with Python 3.11.16 and numpy==2.4.6, numba==0.68.0, pillow==11.3.0, psutil==7.2.2 (llvmlite==0.50.0).
