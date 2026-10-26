# Release-compatible development default promotion

This is a development deployment, not a PBC 3.1 release. The production change switches only `compress` and `compress_stream` defaults to `rng_mode="release"`; explicit `"current"` preserves the previous generator. Public callers inherit that default. Stream v0, presets, original model, seed handling, MSE and optimized runtime mechanisms stay unchanged.

Local acceptance: `python research_pass10/check_codec.py` performs 20 completed encodes across immutable release, prepromotion control, implicit/explicit defaults, custom, streaming and overlapping modes. It covers the existing large Cat photo plus three new deterministic synthetic RGBA/graphics/native inputs. `python research_pass10/check_benchmark.py` verifies legacy database preservation, requested coverage, metadata, read-only selection and active call paths. `node research_pass8/test_selector_race.cjs` and `node --check static/benchmark.js` verify the existing selector behavior. The original 100-case Pass 9 gate and 288-pair benchmark are not repeated.

With the task's explicit operational authorization:

```text
python research_pass10/prepare_deployment.py
python research_pass10/run_hf.py deploy
python research_pass10/check_public_api.py
python research_pass10/run_hf.py research-smoke
python research_pass10/run_hf.py benchmark
python research_pass10/analyze.py
```

Preparation checks the default-only codec diff, unchanged supporting runtime and all declared assets, refreshes the active Pass 9 manifest, and creates a five-file HF child using an isolated index. Deployment rechecks idle jobs/current parent, exports and integrity-checks the database, and never restores an old backup. Do not repeat these commands merely to inspect results.

The public benchmark is exactly the ordinary production path: PBC3 only, 20 main images, four presets, three trials, 240 encode/decode pairs. Its new rows retain patches/file hashes, and new run metadata stores requested coverage, source hashes, model and RNG identity. Historical rows retain null provenance. Timing conventions remain those of the public runner; do not calculate release speedups from these absolute times. Controlled Pass 9 remains the version comparison.

Raw responses, PNG fixtures, database exports and payloads stay under ignored outputs. Restore existing input dependencies as described in research/REPRODUCTION.md. Historical scripts must follow research/HISTORICAL_EXECUTION.md. A Git commit plus baseline_manifest.json freezes the baseline; no codec directory copy or new branch is needed.
