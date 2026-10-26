# HF and local runtime validation

All three runtime mechanisms are compatible and output-preserving locally. HF validation of the new implementation is incomplete: it is not deployed and the existing API cannot select it. Remote work was read-only health/schema/status/source inspection. No remote codec timing, image upload, benchmark start, database mutation, restart or deployment occurred.

## Identity and actual remote observations

The existing [Space](https://egeeken-pbc.hf.space/) answered health/schema/status GETs successfully. HF's [Space metadata](https://huggingface.co/api/spaces/EgeEken/PBC) reported stage `RUNNING`, hardware `cpu-basic`, one replica, repository/runtime revision `7faa6365186d201f44b842f100645b60273ffd4c`. Health boot ID is `1792893987.098`; it is not a commit. Observations are timestamped in hf_observations.json. The date is 2026-10-26 Europe/Paris (recorded timestamps are UTC). Sandbox DNS failures were followed by authorized read-only network access; the failed records remain in hf_observations_sandbox_failure.json.

Nine successfully fetched source/startup files at the reported immutable revision match local text after CRLF normalization: server, broader/Quick RD backends, PBC3, ops, stream, learned filler, Dockerfile and requirements_docker.txt. One mistaken requirements.txt probe returned 404 and is retained. These are reported-revision source checks, not inspection of in-memory imports. The process does not expose installed NumPy/Numba/CPU dispatch, loaded model digest or actual in-memory codec hashes. Docker specifies Python 3.11/Pillow 11.3.0 and mostly unpinned other packages; these declarations do not establish installed versions. Prior Pass 1 model verification at this immutable revision remains historical evidence. No new full remote equivalence assertion is made, and the earlier Quality trajectory discrepancy remains unresolved.

## Existing API/backend inventory

| Method | Path | Operation | Capability/decision |
| --- | --- | --- | --- |
| GET | /api/health; /healthz | Health/boot identity | No codec timing; observed /api/health 200 |
| GET | /openapi.json | Current route/schema inspection | Observed 200; does not expose internal loaded hashes/versions |
| GET | /api/benchmark/status | Read active benchmark state | Observed idle; no start performed |
| GET | /api/benchmark/dataset | Load/describe all main inputs | No image-selection execution; not invoked |
| POST | /api/benchmark/start | Append DB rows for selected codecs/trials | PBC minimum 20 images x 4 presets; no subset/variant selector |
| GET | /api/benchmark/results; /download_db | Historical DB results/file | No fresh source/version identity; not treated as current timing |
| POST | /api/benchmark/stop; /reset_pbc; /upload_db | Stop/delete/replace benchmark state | Not invoked; resets/uploads outside scope |
| GET | /api/quick_rd/status; /results | Status/shared saved points | Status idle; stale output is not timing evidence |
| POST | /api/quick_rd/start | Full default old/current comparison | Overwrites shared JSON; lacks decode timing/independent score; not invoked |
| POST | /api/compress; /stream_compress; /decode | User-supplied image/stream operation | Requires an upload; not invoked |
| GET | /api/sweeps/run/status | Sweep status | Observed idle; no sweep/training started |

The complete live OpenAPI route inventory is remote_endpoint_inventory.json; local server registration paths/lines are local_endpoint_inventory.json. The live schema agrees with the relevant benchmark paths. Current normal start accepts only codecs/n_trials; query filters mp_min/mp_max apply to results, not execution. Authentication is not declared for the existing benchmark routes. The regular backend appends a generated timestamp run_id into the existing SQLite database; callers cannot choose an isolated run ID or a small input/preset subset. Starting PBC alone means at least 80 encodes and 80 decodes. This exceeds the requested small smoke scope, and cannot evaluate undeployed modes, so it was not run. Quick RD is additionally unsuitable because it overwrites shared results and measures the encoder preview/internal encode duration, without independent decode latency.

The broader backend's _eval_pbc3 calls production compress and decompress, separately times them, scores decoded RGB and counts complete file bytes. It preprocesses EXIF/RGB from hpt_data. Its extra resize can hide a size mismatch; the research harness instead asserts original dimensions. Trials are grouped, not interleaved by implementation; preload does not guarantee per-config warmed paths or thread affinity. Its stored rows lack expanded configs/source/model/input/dependency digests and have no experimental implementation selector. Existing BENCHMARK_LOCK coordinates benchmark/sweep/training jobs, but ordinary public compression requests can still contend for CPU. Idle status does not prove absence of every other process. No observed HTTP RTT is reported as codec latency.

## Local equivalence and full timings

Before primary timing, the combined variant matched all 100 image/preset files, decoded pixels, patch signatures/order/counts, original RGB MSE and bpp, with 2612 full proposal-order/RNG comparisons. Sixteen extra RGB/RGBA cases cover all presets, both selected-delta and apply-grid mutation paths, and working-size downsampling. Prior Pass 3 bit layout/malformed/warmup proofs are protected and reused. v0 framing, palettes and decoder syntax are unchanged. Every full repeated current-control file also matches its frozen Pass 1 current file.

Modes: baseline current root; Pass 3 grid+existing MSE opt-out; rectangle reuse alone; all three. No RNG replacement, model change, patch-budget change or allocator is mixed into A. Full confirmation is all 25 images/four presets/three warmed interleaved repeats, 1200 rows. Means below average per-image/preset medians; summary CSV/JSON also retains actual-call means/CV and per-preset values. Rows include after-call RSS, not a controlled peak-RSS difference.

| Set | Candidate | Control | Cases | Exact | Encode base s | Encode s | Encode % | Encode wins | Decode base s | Decode s | Decode % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hpt_data | all | baseline | 80 | 80 | 0.4565 | 0.3402 | -25.4806 | 80 | 0.0843 | 0.0501 | -40.5721 |
| hpt_data_val | all | baseline | 20 | 20 | 0.3646 | 0.2676 | -26.5867 | 18 | 0.0722 | 0.0432 | -40.2441 |
| hpt_data | grid_skip | baseline | 80 | 80 | 0.4565 | 0.3711 | -18.7198 | 80 | 0.0843 | 0.0504 | -40.1848 |
| hpt_data_val | grid_skip | baseline | 20 | 20 | 0.3646 | 0.2905 | -20.3247 | 20 | 0.0722 | 0.0438 | -39.4107 |
| hpt_data | search | baseline | 80 | 80 | 0.4565 | 0.4229 | -7.3725 | 76 | 0.0843 | 0.0839 | -0.5080 |
| hpt_data_val | search | baseline | 20 | 20 | 0.3646 | 0.3434 | -5.8158 | 15 | 0.0722 | 0.0713 | -1.3319 |
| hpt_data | all | grid_skip | 80 | 80 | 0.3711 | 0.3402 | -8.3180 | 75 | 0.0504 | 0.0501 | -0.6476 |
| hpt_data_val | all | grid_skip | 20 | 20 | 0.2905 | 0.2676 | -7.8594 | 16 | 0.0438 | 0.0432 | -1.3754 |
| hpt_data | all | search | 80 | 80 | 0.4229 | 0.3402 | -19.5494 | 80 | 0.0839 | 0.0501 | -40.2687 |
| hpt_data_val | all | search | 20 | 20 | 0.3434 | 0.2676 | -22.0535 | 19 | 0.0713 | 0.0432 | -39.4374 |

Per-preset and variability:

| Set | Preset | Mode | MSE | bpp | Encode s | Encode CV | Decode s | Decode CV |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hpt_data | compression | baseline | 274.7621 | 0.0792 | 0.3971 | 0.0258 | 0.0572 | 0.0410 |
| hpt_data | all | baseline | 193.4631 | 0.2793 | 0.4565 | 0.0234 | 0.0843 | 0.0390 |
| hpt_data | compression | grid_skip | 274.7621 | 0.0792 | 0.3303 | 0.0318 | 0.0475 | 0.0464 |
| hpt_data | all | grid_skip | 193.4631 | 0.2793 | 0.3711 | 0.0251 | 0.0504 | 0.0438 |
| hpt_data | compression | search | 274.7621 | 0.0792 | 0.3636 | 0.0226 | 0.0566 | 0.0447 |
| hpt_data | all | search | 193.4631 | 0.2793 | 0.4229 | 0.0217 | 0.0839 | 0.0354 |
| hpt_data | compression | all | 274.7621 | 0.0792 | 0.2995 | 0.0231 | 0.0474 | 0.0510 |
| hpt_data | all | all | 193.4631 | 0.2793 | 0.3402 | 0.0238 | 0.0501 | 0.0478 |
| hpt_data | balanced | baseline | 195.0596 | 0.1726 | 0.4555 | 0.0249 | 0.0697 | 0.0433 |
| hpt_data | balanced | grid_skip | 195.0596 | 0.1726 | 0.3788 | 0.0243 | 0.0491 | 0.0360 |
| hpt_data | balanced | search | 195.0596 | 0.1726 | 0.4096 | 0.0202 | 0.0685 | 0.0271 |
| hpt_data | balanced | all | 195.0596 | 0.1726 | 0.3385 | 0.0237 | 0.0486 | 0.0495 |
| hpt_data | quality | baseline | 160.0960 | 0.3301 | 0.4584 | 0.0231 | 0.0841 | 0.0339 |
| hpt_data | quality | grid_skip | 160.0960 | 0.3301 | 0.3719 | 0.0224 | 0.0505 | 0.0406 |
| hpt_data | quality | search | 160.0960 | 0.3301 | 0.4177 | 0.0170 | 0.0834 | 0.0420 |
| hpt_data | quality | all | 160.0960 | 0.3301 | 0.3349 | 0.0240 | 0.0499 | 0.0558 |
| hpt_data | high_quality | baseline | 143.9347 | 0.5351 | 0.5151 | 0.0196 | 0.1262 | 0.0376 |
| hpt_data | high_quality | grid_skip | 143.9347 | 0.5351 | 0.4032 | 0.0220 | 0.0545 | 0.0521 |
| hpt_data | high_quality | search | 143.9347 | 0.5351 | 0.5006 | 0.0269 | 0.1269 | 0.0277 |
| hpt_data | high_quality | all | 143.9347 | 0.5351 | 0.3879 | 0.0246 | 0.0544 | 0.0348 |
| hpt_data_val | balanced | baseline | 90.2263 | 0.4243 | 0.3607 | 0.0427 | 0.0599 | 0.1033 |
| hpt_data_val | all | baseline | 91.1562 | 0.7189 | 0.3646 | 0.0313 | 0.0722 | 0.0649 |
| hpt_data_val | balanced | grid_skip | 90.2263 | 0.4243 | 0.2958 | 0.0433 | 0.0422 | 0.0650 |
| hpt_data_val | all | grid_skip | 91.1562 | 0.7189 | 0.2905 | 0.0349 | 0.0438 | 0.0535 |
| hpt_data_val | balanced | search | 90.2263 | 0.4243 | 0.3338 | 0.0351 | 0.0594 | 0.0569 |
| hpt_data_val | all | search | 91.1562 | 0.7189 | 0.3434 | 0.0354 | 0.0713 | 0.0624 |
| hpt_data_val | balanced | all | 90.2263 | 0.4243 | 0.2639 | 0.0585 | 0.0415 | 0.0400 |
| hpt_data_val | all | all | 91.1562 | 0.7189 | 0.2676 | 0.0392 | 0.0432 | 0.0384 |
| hpt_data_val | quality | baseline | 55.5447 | 0.9826 | 0.3863 | 0.0234 | 0.0723 | 0.0812 |
| hpt_data_val | quality | grid_skip | 55.5447 | 0.9826 | 0.3179 | 0.0309 | 0.0433 | 0.0307 |
| hpt_data_val | quality | search | 55.5447 | 0.9826 | 0.3596 | 0.0501 | 0.0720 | 0.0393 |
| hpt_data_val | quality | all | 55.5447 | 0.9826 | 0.2845 | 0.0340 | 0.0437 | 0.0443 |
| hpt_data_val | high_quality | baseline | 41.5028 | 1.2248 | 0.4225 | 0.0223 | 0.1059 | 0.0307 |
| hpt_data_val | high_quality | grid_skip | 41.5028 | 1.2248 | 0.3161 | 0.0327 | 0.0488 | 0.0590 |
| hpt_data_val | high_quality | search | 41.5028 | 1.2248 | 0.4027 | 0.0240 | 0.1051 | 0.1251 |
| hpt_data_val | high_quality | all | 41.5028 | 1.2248 | 0.3074 | 0.0349 | 0.0479 | 0.0504 |
| hpt_data_val | compression | baseline | 177.3511 | 0.2437 | 0.2888 | 0.0369 | 0.0508 | 0.0443 |
| hpt_data_val | compression | grid_skip | 177.3511 | 0.2437 | 0.2321 | 0.0328 | 0.0408 | 0.0593 |
| hpt_data_val | compression | search | 177.3511 | 0.2437 | 0.2774 | 0.0323 | 0.0486 | 0.0281 |
| hpt_data_val | compression | all | 177.3511 | 0.2437 | 0.2148 | 0.0294 | 0.0396 | 0.0190 |

The all-versus-grid_skip pair measures the incremental cache benefit directly. Search state construction, namespace installation, rectangle updates, reductions, allocations and cleanup are inside compress wall time. The private mode-selection/grid hook itself is benchmark setup for all modes; it would become static dispatch in a production design. Small decoder differences between identical decoder paths are noise, not cache benefits. No separately measured percentages are added to claim a combined gain. All full per-image/preset wins/regressions and timing extrema remain in runtime_full_paired.json and runtime_full.jsonl.

Memory is a material tradeoff: int32 cached error maps occupy approximately 3 MB for RGB at 250k and 6 MB at 500k, plus objects/transients. Grid I/O adds O(cells x bitcount) temporary arrays; Pass 3's measured 500k-cell/9-bit tracked write/read peaks were18.0/14.5 MB versus 1.18/1.00 MB scalar. These components do not establish additive whole-codec peak RSS. Metric skipping avoids the internal full-original-image metric allocation/work, but retains the preview and requires independent benchmark scoring. Arbitrary decoder header/resource hardening is outside this change.

## Minimal unapplied integration into this Space

proposed_hf_benchmark.patch changes only server.py and pbc3_benchmark.py, and remains unapplied. It adds an opt-in authenticated branch to the existing POST /api/benchmark/start, retains existing status/stop infrastructure, and changes neither ordinary benchmark execution nor public compression defaults. Stage only hf_runtime_job.py, hf_candidate_grid.py, hf_candidate_search.py and hf_runtime_manifest.json under research_pass4; no Pass 1/2 framework or second Space is needed. Candidate copies are exact Pass 3 implementations. Linux uses standard-library CPU affinity, with no new HF dependency; Windows packaging smoke uses existing psutil.

Static allowlists: images googlewiki.png/flower.jpg, up to two of four presets, modes baseline/grid_skip/search/all, repeats 1..3, at most48 timed calls plus bounded per-mode/preset warmups. No code, module path, filesystem path, shell command or arbitrary config is accepted. baseline is required. A disabled-by-default PBC_RESEARCH_BENCHMARK_TOKEN secret gates the opt-in branch; compare the request's x-pbc-benchmark-token with hmac.compare_digest. Never put it into a URL, report or saved result. The normal route is unchanged. Handler/allowlist tests check disabled404, unauthorized403, invalid400, guarded valid dispatch and ordinary-route arguments. These are local extracted-handler tests, not a live server test.

The guarded backend starts a dedicated bounded subprocess, keeping monkeypatches outside the public FastAPI worker. All modes share one warmed serial worker and allowed CPU. Source/candidate normalized hashes, raw input/model hashes, installed versions, thread settings, module paths, expanded configs and output/pixel hashes accompany results. Full compress/decompress times are distinct from loading, scoring, IPC and HTTP. Each job writes only a new UUID-named research run directory and attaches results to the existing status object; it does not touch the benchmark DB or Quick RD output. Existing stop sets the stop event; the runner terminates the child, with a 20-minute timeout. It does not stop the Space. No new endpoint is introduced.

Future first payload, after authorization and installation: {"research":{"images":["googlewiki.png"], "presets":["balanced"], "variants":["baseline", "grid_skip", "search", "all"], "repeats":2}}. POST it to /api/benchmark/start using the secret header, then GET /api/benchmark/status until complete; inspect research_result and the new isolated results.json. A first run has 8 timed operations plus4 warmups. HTTP request/status RTT stays separate. The job verifies manifest/input/model identity before any benchmark call and exact same-runtime files/pixels/counts/MSE/bpp before reporting success. Public requests can interfere with CPU, so use an agreed quiet interval and retain repeat variance; same-runtime exactness is necessary regardless of unresolved cross-runtime ties.

Required approval is explicit authorization to apply/stage the listed files, commit/push to the existing HF Space, configure its secret and permit the resulting rebuild/restart and bounded job. None was performed. Docker COPY means a push rebuilds the existing image. First verify source/runtime revision and actual package identity in returned metadata. Do not claim optimized HF speedups from current-deployment baseline timings.

Rollback: disable the optional branch by removing the secret (configuration changes/restarts require authorization), or restore only the two server/backend files to the recorded pre-change revision and remove only staged research job files, preserving databases/user work. Git destructive cleanup is unnecessary. No production grid/search default is proposed by this benchmark-only patch.

## UI and promotion limits

server.py still calls float(result.mse), so public compression must keep compute_final_mse=True. Only metric-free private/file benchmark modes opt out; independently decoded RGB MSE remains available in every research row. A future UI that computes the metric elsewhere still pays that work. Do not advertise the full skip-MSE speedup for an unchanged metric-producing UI. Grid plus search are candidates for UI engineering consideration after separate same-runtime HF validation; their unsafe global experimental hooks must remain isolated or be replaced by explicit per-encode ownership for concurrent production use. No PBC 3.1 RD release follows from bit-exact runtime gains.
