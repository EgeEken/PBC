Authorized by the user's Pass 5 instruction to commit and push the minimal reviewed runtime and authenticated benchmark changes, preserving public compression defaults and UI behavior, then run a bounded HF benchmark.

Target: existing `https://huggingface.co/spaces/EgeEken/PBC`, `refs/heads/main`, based on `7faa6365186d201f44b842f100645b60273ffd4c`. Local branch: `codex/pass5-hf-runtime`. An isolated Git index avoids including the current checkout's unrelated experimental ancestors or user changes.

Exact deployment files:

- `server.py`: authenticated research branch in the existing benchmark start API.
- `pbc3_benchmark.py`: bounded isolated child worker through existing benchmark guard/status/stop; separate research output, no benchmark DB writes.
- `research_pass5/hf_runtime_job.py`: reused Pass 4 worker, three static variants, eight existing images, four presets, up to three repeats, single CPU and numerical thread, manifest checks, full server-side timing and independent decoded RGB metrics.
- `research_pass5/hf_candidate_grid.py`: exact copy of the validated Pass 4 grid implementation.
- `research_pass5/hf_candidate_search.py`: exact copy of the validated Pass 4 search implementation.
- `research_pass5/hf_runtime_manifest.json`: source/model/input hashes.

Public compression, decode, stream, status, stop and UI behavior are unchanged; their relevant handlers compare identically with HF main. No production codec defaults, model, stream format, dataset, dependency or frontend changes. `all_mse` retains the internal MSE required by public compression; `all` skips it only in the private benchmark child. Baseline is unchanged production code.

Research start is disabled without `PBC_RESEARCH_BENCHMARK_TOKEN`, returns 403 for a missing/wrong `x-pbc-benchmark-token`, and rejects unallowlisted inputs/configuration. No arbitrary paths/code/module/config selection. Original read-only status and ordinary benchmark controls retain their existing access policy. Candidate monkeypatches exist only in a private subprocess and are restored between runs. Secret values are neither committed, printed nor saved locally; the final scoped benchmark secret is derived by domain-separated HMAC from the existing HF credential and revision, held in client memory and installed via HF's secret API. The HF credential itself is never sent to the app. Initial random keys were replaced after local checker failures; see client_correction.json.

Setting the secret may restart the Space; pushing rebuilds/restarts it. Confirm both repository and running revision plus health before timing. Initial smoke: one existing image, balanced preset, three variants, two repeats (six measured calls plus three full warmups). Representative run: eight static images, four presets, three variants, three repeats (288 measured calls plus 12 full warmups), contingent on exact smoke output and no substantial smoke regression. One production compression API check uses the same existing `googlewiki.png` to check stream, preview, file size and MSE. No new image or model is introduced.

Rollback: revert the deployment commit on a clean branch based on current HF main, then push the revert to HF main (no force push/reset). Delete the benchmark secret to disable research start. This restores the previous production source; isolated runtime result files need not be deleted. The commit hash is recorded in `deployment.json` before pushing. Investigate any specific deployment failure before rollback.

Local readiness: six measured smoke rows plus three warmups, byte/pixel/MSE equivalence, correct internal metric availability, 11 rejected unsafe specs, disabled/missing/wrong/valid-token handler checks, unchanged ordinary benchmark arguments, compile checks. Previous correctness suites are reused without repetition. HF-installed dependency versions are captured in the actual worker, since the Docker dependency file is mostly unpinned.

Unrelated user work, `.gitignore`, earlier research artifacts and the current experimental branch/index will not be staged or committed. Only the six named files enter the deployment commit.
