# PBC Pass 15 summary

## Local PBC 3.0 versus staged PBC 3.1 candidate

The final staged candidate was compared with immutable PBC 3.0 on the i9-12900H using eight images, four presets, three warmed/interleaved repetitions, one pinned P-core, and one numerical thread. Across the 32 image/preset cases, encode time fell 34.00% and decode time fell 64.28%. All 96 paired streams were byte-identical; decoded RGB pixels, patch counts, bpp, and independent MSE also matched. There were no per-case median encode or decode regressions.

The controlled HF cpu-basic comparison remains 28.83% lower encode time and 64.38% lower decode time. These results use different machines and datasets and are reported separately.

## Integral-image optimization

The Numba fused integral implementation is accepted as candidate `pbc31_integral_fused`. Against frozen PBC 3.1 it reduced encode time by 12.00% in a repeated four-image screen, with all four presets faster and 48/48 paired streams, decoded pixels, and MSE identical. On the 20-image main set plus five-image holdout, all 100 image/preset pairs remained byte-identical. The candidate also matched the original NumPy integral output on 96 tested arrays, including varied dtypes/layouts, empty shapes, negative and extreme values, and non-finite float casts. No rate-distortion or stream-format change is claimed.

The staged ZIP was rebuilt from the validated candidate. It passed isolated extraction/install validation, all four preset encode/decode checks, RGB/RGBA, determinism, frozen output hash, and PBC 3.0 v0 decode compatibility. Its SHA-256 is `395e76290965c0e44fe1844b2738c3a0a193d0ce3c3090c6e274af8c10d30312`. The ZIP and flat source tree are in the local `PBC3_1_Release_Staging` directory; the frozen-baseline artifact and intermediate candidate are retained in sibling directories.

The staging source keeps the established top-level GitHub module layout and both import styles. It is not published. No GitHub `main`, release, tag, or HF deployment was changed.

## Reports and reproduction

- `PBC31_LOCAL_RELEASE_COMPARISON.md` — actual PBC 3.0 versus staged candidate method and local timing results.
- `PBC_INTEGRAL_OPTIMIZATION.md` — candidate design, correctness evidence, microbenchmark, full-encode timing, and promotion decision.
- `PBC31_RELEASE_STAGING.md` — staged files, validation, hashes, and retained previous artifacts.
- `run_local_release_comparison.py`, `run_integral_screen.py`, `benchmark_worker.py`, `integral_candidate.py`, `build_local_release_staging.py`, and `validate_local_release_staging.py` — research and packaging scripts.

Raw repetition JSON and local validation JSON are ignored by Git; the reports preserve the aggregate evidence and exact input/source provenance.
