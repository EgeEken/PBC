# Release-compatible RNG with optimized runtime

Only `PBC3.compress(..., rng_mode="release")` or `compress_stream` with that option selects NumPy `default_rng(config.random_seed)`. The omitted/default mode remains `"current"`, using the existing custom PCG32 generator. Each call owns its generator; no global replacement or state is used. Presets and the public API do not select the new option.

`release` imports the immutable six-module release under `research_pass1/release`; `dev_release_rng` uses the integrated root with the explicit option; `dev_current_rng` omits the option. All use the immutable original model and internal MSE. No other algorithm is changed.

Restore the 25 ignored inputs with `research/prepare_assets.py` as documented in `research/REPRODUCTION.md`. Use its recorded Windows dependencies locally and the existing HF image for HF. Do not interpret local timing as HF performance.

1. `python research_pass9/prepare_deployment.py` verifies the immutable source/model, unchanged support modules and all input hashes, and writes the runtime manifest.
2. `python research_pass9/check_local.py` runs the small equivalence gate, then the remaining main/holdout cases, and checks default/streaming/concurrent selection and invalid specifications.
3. With explicit HF operational authorization, `python research_pass9/prepare_deployment.py --commit` creates a minimal HF child commit using an isolated index. `python research_pass9/run_hf.py deploy` first exports/checks the database, then pushes that exact child to HF main and waits for it. These commands do not promote the RNG default.
   This completion reused prepared child `4237d954f4e1a04951c822c738113a53bbd58e9b` after verifying its parent/tree and current HF HEAD. The valid local 100-case gate was not repeated. `python research_pass9/check_public_api.py` makes fresh public compression/decode checks, using the retained Pass 1 Balanced v0 stream in the original local archive; it requires the same explicit operational authorization and refuses to reuse a cached response.
4. With the same authorization, run `python research_pass9/run_hf.py smoke`, then `equivalence`, then `benchmark`. The small three-way smoke covers three images/four presets. The extension covers the remaining 22 main/holdout images, release/RNG pair once. The timing run fixes eight main images/four presets/three variants/three repeats (288 calls). Full warmups precede every job; variants rotate by trial/image/preset on the same pinned CPU, numerical threads one.
5. `python research_pass9/analyze.py` reduces the saved responses to concise evidence/tables. Each timing cell is the mean of eight per-image medians; overall equally weights 32 cases. Ratios always use simultaneously measured controls.

The active authenticated helper allows only the declared files, presets and two fixed variant sets. Repeated timing is limited to eight main images and three repeats; one-repeat equivalence permits all 25 images. Existing authorization, lock and 1800-second timeout remain. Hash/patch parsing, image loading, external scoring, IPC and HTTP are excluded from full compress/decompress timers. Internal MSE, full reconstruction/final resize and serialization/LZMA remain inside the calls.

Generated streams, complete responses, database backups, caches and credentials remain ignored. Historical Pass 6/8 manifests describe their original codec revisions; this pass has its own active manifest.

The retained original-v0 compatibility input has SHA256 `2f7ef0a8dcdd22444a65ba03f3e2955dde57844d36db7fa269567ca38bcd9414`. Public checks preserve opaque RGBA; controlled benchmarks consistently use EXIF-normalized RGB. Public HTTP/stage timings are never used in the performance tables.

HF validation is complete at `4237d954f4e1a04951c822c738113a53bbd58e9b`. Both variants remain selectable. For future RD research, explicitly select `rng_mode="release"` as the primary control; use `"current"` to reproduce historical HF work. This recommendation does not change the live public default. See PBC_PASS9_SUMMARY.md for measured outcomes.
