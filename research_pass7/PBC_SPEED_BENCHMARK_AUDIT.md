# Speed Benchmark audit

The aggregation defect is confirmed, but it does **not** explain a historical/current PBC3 mixture in the database currently served. There is only one stored PBC3 run. Its unfiltered means are **0.768602 s encode / 0.117958 s decode**, identical to its all-history PBC3 means (difference zero). The existing chart combines this October PBC3 run with other codecs measured in June, so it is not a same-condition codec comparison.

## Stored evidence

Read-only downloads of `/api/benchmark/download_db`, `/results`, and `/status` were taken before changes. No reset, upload, start, stop, database write, restart or HF push was performed. `database_audit.json`, `run_selection_checks.json` and `active_path_audit.json` contain compact evidence; the two database copies and full API responses remain ignored locally.

| Run | Rows | Images | Trials | Points | Codecs |
| --- | ---: | ---: | ---: | ---: | --- |
| 20260624_134736 | 900 | 20 | 1 | 45 | Six non-PBC codecs |
| 20260624_150209 | 4500 | 20 | 5 | 45 | Six non-PBC codecs |
| 20261026_122641 | 800 | 20 | 10 | 4 | PBC3 only |

The PBC3 run spans 2026-10-26 12:26:43–12:40:15 UTC, after Pass 6 confirmed the integrated runtime running at 12:04:36 UTC. Each preset has 200 rows. Public status independently confirms done=total=800, no error, and all ten trials finishing. This run has complete coverage, with no missing rows or duplicate trial/codec/preset/image keys. The public dataset actually contains **20 main images**, not 25 main-plus-holdout images.

Preset encode/decode means in seconds: Compression 0.672669/0.115163; Balanced 0.763215/0.118948; Quality 0.771990/0.115203; HQ 0.866535/0.122517. Trial means range 0.666318–0.870662 encode and 0.099798–0.144179 decode. The first trial is not the slowest. This variation prevents assigning a change in absolute historical means to the codec alone.

The approximately 0.704/0.108 screenshot values are **not reproduced by the current complete unfiltered database**, nor by the database embedded in the HF revision. The screenshot's filter settings and capture time are unavailable; an intermediate run or filtered view cannot be established from that description alone.

The HF revision embeds a different PBC3 run, `20260715_144225`: 800 rows, means 0.579566/0.126183. Those July rows are absent from the live database; October rows have IDs 7481–8280, following the embedded July IDs 6681–7480. This proves that the live and repository databases differ. It does not identify whether a previous reset, upload or other operation removed the July rows. No durable event history records that cause, and the database does not store codec revision, CPU state or requested totals. Deployment-related historical loss cannot be reconstructed reliably.

## Active codec and timing

HF repository and running runtime both report `cd0d5691647a4f02cf765454a2d3b0abfde8f3b7`, cpu-basic, health OK. Nine deployed source files match their pre-Pass-7 GitHub snapshot hashes. The ordinary start route calls `pbc3_benchmark.start`, then `_run` and `_eval_pbc3`, which directly import current root `PBC3` and its integrated dependencies. There is no retained preintegration import or subprocess worker in this ordinary path. The isolated preintegration worker is selected only through the separately authenticated research request. Runtime identity and the completed run's postdeployment timestamp support current-code execution; no invasive process introspection was added.

`preload_numba` runs at server startup and before each ordinary benchmark. It warms learned RGB/RGBA encodes, including the selected-patch resampling kernels. Decode is not separately warmed as a full public call. The ordinary benchmark has no affinity or numerical-thread override. Its order is trial → codec/preset → sorted image; it does not interleave baseline and optimized sources.

Encode timing starts before `Image.fromarray` and wraps complete `PBC3.compress`, including default `compute_final_mse=True`, reconstruction and original-size resizing. Decode wraps full `PBC3.decompress`, including its final original-size resize. The subsequent RGB conversion, redundant same-size resize and independent MSE are outside the decode timer. Dataset reading, SQLite insertion, preload and HTTP are outside the codec timers. Imports/model caches are shared within the server. Initialization, CPU scheduling, traffic and fixed job order differ from the isolated paired method.

| Measurement | Encode s | Decode s | Interpretation |
| --- | ---: | ---: | --- |
| Current public run, all 800 calls | 0.768602 | 0.117958 | 20 images, four presets, ten sequential trials |
| Pass 5 preintegration / retained-MSE optimized | 0.530814 / 0.447193 | 0.098395 / 0.059877 | Same-runtime paired, eight images × four presets |
| Pass 6 preintegration / integrated retained MSE | 0.837813 / 0.690236 | 0.205201 / 0.106760 | Same-runtime paired, three images × Quality/HQ |
| Pass 6 integrated skipped MSE | 0.602111 | 0.098735 | Optional metric-free research control |

Pass 5/6 average per-case medians from three interleaved repeats; the public chart uses count-weighted arithmetic row means. Research workers are persistent isolated processes, pinned to one common CPU, with numerical threads one and consistent warmups. Their image loading, IPC and external scoring are excluded. The public average is consistent with integrated code and these different workloads; numerical proximity alone neither validates nor refutes speedup.

Four controls remain distinct: immutable GitHub release `2e59f1b06980aa467da0cde94fa33c3f9a0d4d51`; HF preintegration codec retained under `research_pass6/baseline`; current integrated runtime; integrated runtime with internal MSE skipped. Pass 6 directly compared the latter three in one live environment. None of its percentages compares fresh calls against stored historical rows or establishes a new RD advantage over the immutable release.

## Correction and deployment decision

`results(run_id=...)` now defaults to the latest run with complete observed coverage, supports an exact run ID and explicit `all`, and uses a read-only SQLite connection. Run metadata is determined before MP filtering. The UI lists run IDs/status, timestamp, trial/image/sample counts, and labels all-history as a composite. Selecting a PBC-only run displays only its records. Existing point filters and database download/upload operations remain available; the database schema and codec are unchanged.

Historical completion is explicitly **inferred coverage**, not a fabricated success flag. Missing points, duplicates, gaps in observed trials and active runs are excluded from latest-complete selection. Entirely missing images/codecs or trailing trials cannot be inferred from rows alone, and this limitation is displayed. The live status log verifies the particular October run independently. No metadata migration was introduced.

`python research_pass7/test_run_selection.py` verifies two deliberately different runs, exact selection, latest selection, active/partial/missing runs, MP filtering, other-codec isolation, the previous weighted all-history result, and an unchanged database hash. `node --check static/benchmark.js` passes. The active research-worker manifest was updated only for the two changed backend hashes; the original Pass 6 outcome retains its measured manifest.

**Not deployed.** The Dockerfile copies the repository database into `/app`; no persistent storage is advertised, and the deployed repository copy lacks the current October records. Rebuilding could replace the live dataset of records. Downloading a backup does not make that deployment non-destructive. Deployment is deferred until persistence or a separately authorized lossless restoration procedure preserves all live rows. HF remains at `cd0d569`; production codec defaults and behavior remain unchanged. No new remote benchmark was necessary.

Reproduce locally: obtain a read-only database export as `research_pass7/database_before.db`; optionally download the revision-embedded copy as `repository_database.db`; run `audit_database.py` and `test_run_selection.py`. `capture_audit.py` performs read-only remote source/metadata checks. Do not upload these databases to GitHub.
