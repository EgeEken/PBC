# Pass 9 findings logged; HF evaluation deferred

The optimized codec with explicit release RNG reproduces immutable PBC 3.0 on 100/100 local cases: 20 main and five holdout images, four presets. Stream/pixel/ordered-patch hashes, counts, original-size RGB MSE and final post-LZMA bpp agree exactly. Mixed concurrent calls retain independent selection. Public defaults remain current PCG32; only `rng_mode="release"` selects NumPy default_rng per encode.

Implementation and local evidence: [cfac4f84121adc5b825c5d77abe326a84bc5701d](https://github.com/EgeEken/PBC/commit/cfac4f84121adc5b825c5d77abe326a84bc5701d). No published release or reference source was changed. The immutable control remains `2e59f1b06980aa467da0cde94fa33c3f9a0d4d51` with the original identical policy weights.

| Requested conclusion | Evidence available now |
| --- | --- |
| Release files reproduced? | Yes, all 100 local cases. HF gate pending. |
| Quality bitrate regression removed? | Local release/RNG bpp matches exactly, including Quality. Removal of the measured HF 9.91% regression is not yet verified. |
| New encode/decode improvements vs actual release? | Unmeasured on HF; no new percentages claimed. |
| Implementation versus trajectory attribution? | Local release/dev-release-RNG streams match, enabling a same-stream engineering comparison. Three-way timed attribution remains pending. Sampler overhead and changed search workload must not be conflated. |
| Eiffel Balanced and Cat Compression speed regressions disappear? | Their release patch counts/order are restored locally; HF speed effects remain unmeasured. |
| Suitable PBC 3.1 baseline? | Candidate only; defer promotion recommendation until HF equivalence and speed gates pass. No default promotion authorized or performed. |
| Strictly better RD than release? | Not demonstrated: exact restoration ties release. A subsequent mechanism would need lower distortion at no higher final rate and separate validation; none was started. |
| HF revision used? | Existing public run observed at `27f6c65225a5496fa065deb586c7e405ecdffc84`; no Pass 9 HF execution. Prepared child `4237d954f4e1a04951c822c738113a53bbd58e9b` was not pushed. |

The initial deployment found the unrelated public run active. At the user's subsequent instruction, stop it and log findings for now, the run was stopped without error at 996/9,800 calls. Its database backup retains every completed row and passed SQLite integrity checks. One complete first trial supports PBC3 means 0.534392 s encode / 0.082203 s decode across 20 images/four presets. The screenshot's PNG bar matches only five images; complete first-trial PNG is 1.878271/0.164118 s. These observational codec averages do not replace the controlled RNG experiment or establish matched-rate dominance.

Reports: PBC_RELEASE_RNG_RESTORATION.md, PBC_HF_THREE_WAY_BENCHMARK.md, PBC_EXISTING_SPEED_RUN_LOG.md. Reproduction: README.md, check_local.py, run_hf.py, analyze.py and log_public_run.py. Compact hashes/config/provenance are tracked; complete responses, image inputs, databases, screenshot, streams and caches are excluded. No secret configuration, model training, unrelated compression mechanism or decoder rewrite occurred.

Next step when the user resumes HF evaluation: reconfirm revision/idle state, back up the current database, review/deploy the opt-in support, run the HF smoke and complete 100-case gate, then the 288-call comparison. Keep the release-compatible default decision separate.
