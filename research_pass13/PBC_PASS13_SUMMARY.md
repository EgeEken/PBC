# PBC Pass 13 summary

The frozen baseline remained unchanged. Work was limited to local research scripts, measurements, and reports.

- The isolated Windows environment (Python 3.11.16, NumPy 2.4.6, Numba 0.68.0, Pillow 11.3.0) loaded the frozen codec/model and reproduced Pass 12 local hashes for 16/16 cases, but only 6/16 saved HF hashes.
- Three HF mismatches and one matching control produced identical arrays, prepared targets, patch trajectories, and full streams across the original and isolated Windows runtimes. This narrows the discrepancy away from those local package-version changes, but HF stage-level artifacts were not saved, so the root cause remains unknown. Future cross-HF reproduction should use the exact Linux HF runtime; local equality is not established.
- Across 16 frozen YCbCr streams, 597/2,402 cross-channel pairs overlapped spatially, but only 56 had IoU ≥ 0.5. There were no exact rectangle or compatible grid-layout matches. Mean correction correlation was −0.112.
- Exact shared geometry and compatible-grid payload coding showed zero measured savings. The total non-grid stream body is only 0.221% of raw bits in this sample. The prior Pass 2 final-size geometry sensitivity was ~0.3%, below the ~3% promotion threshold.
- No offline prototype was justified. No codec, stream, preset, model, HF Space, or public benchmark was changed.

**Recommendation:** stop shared cross-channel coding at the feasibility gate. If another experiment is authorized, prioritize search/candidate reuse that reduces repeated integral-image work while preserving patch semantics and stream output; validate it against the true PBC 3.0 runtime on the main and holdout sets.

Machine-readable case results and provenance are in `measurements.json`. See `PBC_ENVIRONMENT_REPRODUCIBILITY.md` for isolated environment setup and exact commands to regenerate the analysis and four-case diagnostics.
