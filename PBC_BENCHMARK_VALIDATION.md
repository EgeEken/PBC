# PBC benchmark validation

## Verdict

The new isolated harness is suitable for controlled local default-preset comparisons. Existing Quick RD/sweep data is not sufficient for release acceptance. The broader benchmark's core PBC metric function is correctly designed, but its stored database rows do not establish current source provenance.

There are 900 measured rows: 25 original images × four presets × three implementations × three repeats, preceded by a 48-row smoke test (two images, all presets, two repeats). Release and old have identical serialized hashes and decoded pixel hashes in **100/100 distinct cases**. Current is byte-identical to release in **0/100**. All 900 encoder previews equal actual decoded output. Every source/image/preset is deterministic across the three repeats.

## Audit findings and local corrections

| Component | Finding | Correction/coverage |
|---|---|---|
| Quick RD, `pbc3_quick_rd.py:86`, `:104` | Current and old evaluators score `res.image`, do not call decompress, use internal encode duration, and omit decode time/raw bytes/source hashes. Runs group old before current; old is restarted per preset, with different warmup. | Replaced for this audit by isolated `research_pass1/benchmark.py`; website runner left unchanged. |
| Old subprocess | Codec imports are isolated, but model default is relative to root cwd in existing Quick RD, so it can load root weights. These weights happen to match here. Old learned_filler alters sys.path. | New workers have implementation-specific cwd and absolute model paths. Loaded helper/module inventories were inspected after learned warmup. |
| Sweep, `pbc3_sweep.py:166` | Uses encoder preview/internal duration; database results can reflect older code or params. | Not used as a release comparison. No database modified. |
| Broader benchmark, `pbc3_benchmark.py:134` | Calls compress and decompress, times complete calls with perf_counter, scores decoded RGB using safe float conversions, counts complete final file. | Locally exercised by `validate_backend.py`; raw artifacts and exact backend coverage are recorded in backend_validation.json. |
| Result properties, `pbc3_types.py:228` | `decode_time` and `decode_seconds` alias the result's single operation duration. On an encoder result this is encode time. | Harness measures calls explicitly and never infers decode timing from an encoder result. Server decoder uses a decompress result, which is valid. |
| Website, `server.py:329` | Auto mode routes to production presets; UI accepts overrides for learned enable/q and forces one learned candidate/top-k. MSE comes from encoder result, displayed size/time rounded. The website preserves an opaque alpha channel; its MSE then averages RGBA, while benchmark loaders discard alpha and measure RGB. | All benchmark previews are empirically equal to decode. Four RGB-normalized local API tests match the harness files, decoded pixels/MSE, and size; an original-RGBA test confirms the different metric. This alpha-policy discrepancy remains in the existing website. Remote GETs alone cannot verify returned compressed files. |
| Provenance and stale output | Quick RD JSON has a creation time but no commit/model/config/input hashes. Deployed `/api/quick_rd/results` reports `new_exists=false`; available DB rows are historical. | New per-image rows include exact input, config, model and stream identifiers. Remote observations stored separately from local timing. |
| Warmup/threads/order | Existing preload helpers differ; neither existing benchmark enforces all numerical thread settings or balanced per-case trial order. | Same complete learned encode/decode warmup per preset, one CPU affinity, six thread variables=1, persistent separate workers, source order rotated by trial/image/preset. |

No codec, model, production default, existing database, or deployed file was repaired or synchronized. Corrections are confined to the new reproducible audit harness.

For the original opaque-RGBA googlewiki.png, the local balanced API reports MSE 637.44; the decoded RGB MSE is 849.914308, and RGBA MSE is 637.435731. A perfect alpha channel lowers the four-channel average to 75% of its RGB value. The original RGBA stream is retained in backend_original_rgba.pbc3. The RGB benchmark does not make a native-RGBA quality claim.

Local TestClient initially stalled in Windows asyncio's fallback socketpair inside the sandbox. The same in-process tests succeeded outside that restriction; those local tests contacted no external host. Startup lifespan/provider probes were excluded, while real form handling, compression, response serialization and decode verification were exercised.

After explicit user authorization, exactly four original googlewiki.png uploads were sent to the HF /api/compress endpoint, one per preset. 3/4 files and decoded pixel arrays match local encodes under the same RGBA policy. All four remote previews, displayed sizes and displayed RGBA MSE values match actual decoded artifacts. Quality differs: RGB MSE 799.084542 remotely versus 796.712163 locally; bytes 79880 versus 81978. The first differing decoded patch is zero-based index 14 (15th stream patch, 11th iterative patch after four initialization patches), with a different selected box. The earlier decoded patches match. The source/model checks match the reported immutable HF revision, but the reason for this cross-runtime trajectory difference is unresolved; it is not dismissed as numerical noise. No additional compression requests were made to investigate it. See deployment_comparison.json and the retained local/remote RGBA streams.

| Preset | Bytes equal | Decoded pixels equal | Raw bytes | Final bytes | bpp | RGB MSE | RGBA MSE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| compression | True | True | 67556 | 28700 | 0.347369 | 1052.578448 | 789.433836 |
| balanced | True | True | 119464 | 51493 | 0.623243 | 849.914308 | 637.435731 |
| quality | False | False | 177361 | 79880 | 0.966824 | 799.084542 | 599.313407 |
| high_quality | True | True | 511658 | 183632 | 2.222583 | 704.168187 | 528.126140 |

## Measurement method

Both sets are exactly the supplied hpt_data (20) and hpt_data_val (5), including uppercase `.JPG` holdout filenames. Input paths and SHA-256 are in baseline_metadata.json. Every source opens the same file, applies EXIF transpose, converts to RGB, and materializes it before the timed call. Alpha is discarded consistently for this RGB benchmark. Native RGBA quality is outside its scope.

Each worker pins itself to logical CPU 0 before warmup and measured encodes. NumPy/BLAS/Numba thread environment settings are established before worker imports. Only one worker executes a requested operation at a time. All sources have the same Python/dependencies/hardware. OS/background scheduling and frequency variation remain possible. No HF timing is merged with these local rows.

Full encode wall time surrounds PBC3.compress including preprocessing, model-object construction/cache lookup, search, serialization, LZMA, final conversion/resize and codec-internal MSE. File loading and the harness's external validation are outside that interval. Decode wall time surrounds PBC3.decompress on the generated bytes, including entropy decoding, bit parsing, reconstruction and original-size conversion. JSON/process IPC, disk artifact writes and independent MSE/hash computation are outside both codec timers. A warmed model-weight cache is retained equally; no prepared image arrays or canvas/search state are reused across encodes.

Decoded image dimensions must equal EXIF-normalized source dimensions; no benchmark-side resize hides mismatches. MSE is mean RGB squared error with float64 subtraction/reduction in baseline rows. bpp is `8 * len(final_data) / (original_width * original_height)`. Raw bytes are the uncompressed bitstream body plus the same six-byte outer header; raw_body_bytes is also retained. Patch count comes from the emitted stream header and includes initialization patches. No patch-bit proxy is substituted for final size.

One full learned encode/decode on googlewiki.png per preset precedes timing. First calls in fresh workers are retained in metadata warmup_rows, but Numba disk caches may already exist; these are not pristine install/cold-boot timings. Final timing claims concern warmed calls only. Three interleaved measured repetitions are retained individually. Tables give both arithmetic means of all calls and averages of each image's median duration. Percent timing comparisons use the latter. CV is population standard deviation / mean over three repeats, then averaged across images; three repeats give only a limited noise estimate.

## Controlled results

| Set | Preset | Source | MSE | bpp | encode mean s | encode median s | decode mean s | decode median s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hpt_data | compression | release | 278.401172 | 0.083143 | 0.441413 | 0.438765 | 0.066077 | 0.066007 |
| hpt_data | compression | old | 278.401172 | 0.083143 | 0.473347 | 0.469549 | 0.066370 | 0.065612 |
| hpt_data | compression | current | 274.762110 | 0.079248 | 0.421089 | 0.418699 | 0.060120 | 0.060123 |
| hpt_data | balanced | release | 193.486148 | 0.169028 | 0.551714 | 0.549167 | 0.086860 | 0.086428 |
| hpt_data | balanced | old | 193.486148 | 0.169028 | 0.598067 | 0.595754 | 0.086847 | 0.086415 |
| hpt_data | balanced | current | 195.059638 | 0.172648 | 0.481474 | 0.478113 | 0.074457 | 0.073503 |
| hpt_data | quality | release | 160.330642 | 0.299595 | 0.597149 | 0.591044 | 0.109015 | 0.108606 |
| hpt_data | quality | old | 160.330642 | 0.299595 | 0.642018 | 0.641408 | 0.110674 | 0.109847 |
| hpt_data | quality | current | 160.096016 | 0.330114 | 0.483982 | 0.476525 | 0.088232 | 0.087265 |
| hpt_data | high_quality | release | 143.133423 | 0.542457 | 0.649485 | 0.642521 | 0.176503 | 0.175502 |
| hpt_data | high_quality | old | 143.133423 | 0.542457 | 0.716726 | 0.708015 | 0.179567 | 0.178952 |
| hpt_data | high_quality | current | 143.934689 | 0.535095 | 0.550930 | 0.545957 | 0.131773 | 0.130338 |
| hpt_data_val | compression | release | 169.269132 | 0.263268 | 0.362329 | 0.364128 | 0.059273 | 0.059284 |
| hpt_data_val | compression | old | 169.269132 | 0.263268 | 0.403553 | 0.395100 | 0.058844 | 0.058460 |
| hpt_data_val | compression | current | 177.351099 | 0.243723 | 0.308245 | 0.307114 | 0.051003 | 0.050387 |
| hpt_data_val | balanced | release | 99.030764 | 0.408693 | 0.356674 | 0.354531 | 0.073128 | 0.072361 |
| hpt_data_val | balanced | old | 99.030764 | 0.408693 | 0.395875 | 0.394168 | 0.072850 | 0.072279 |
| hpt_data_val | balanced | current | 90.226330 | 0.424291 | 0.384296 | 0.378868 | 0.062913 | 0.062740 |
| hpt_data_val | quality | release | 50.676304 | 1.016413 | 0.500775 | 0.491872 | 0.094818 | 0.094325 |
| hpt_data_val | quality | old | 50.676304 | 1.016413 | 0.538430 | 0.537317 | 0.092104 | 0.091245 |
| hpt_data_val | quality | current | 55.544730 | 0.982597 | 0.411665 | 0.407316 | 0.074548 | 0.075567 |
| hpt_data_val | high_quality | release | 40.247215 | 1.174562 | 0.536042 | 0.539782 | 0.152766 | 0.150470 |
| hpt_data_val | high_quality | old | 40.247215 | 1.174562 | 0.602113 | 0.604300 | 0.148393 | 0.148145 |
| hpt_data_val | high_quality | current | 41.502788 | 1.224793 | 0.437743 | 0.438084 | 0.109867 | 0.108712 |

Current relative to immutable release (percent changes of equal-weight image/preset means; lower is better):

| Set | Preset | MSE % | bpp % | raw bytes % | final bytes % | encode median % | decode median % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| hpt_data | compression | -1.307 | -4.685 | +2.323 | +3.214 | -4.573 | -8.915 |
| hpt_data | balanced | +0.813 | +2.142 | -5.213 | -2.899 | -12.938 | -14.955 |
| hpt_data | quality | -0.146 | +10.187 | +0.664 | +3.838 | -19.376 | -19.649 |
| hpt_data | high_quality | +0.560 | -1.357 | -2.932 | -2.181 | -15.029 | -25.734 |
| hpt_data | all | -0.193 | +2.091 | -2.092 | -0.230 | -13.604 | -19.543 |
| hpt_data_val | compression | +4.775 | -7.424 | -1.983 | -9.961 | -15.658 | -15.008 |
| hpt_data_val | balanced | -8.891 | +3.817 | +6.460 | +18.460 | +6.864 | -13.295 |
| hpt_data_val | quality | +9.607 | -3.327 | -4.719 | -0.493 | -17.191 | -19.886 |
| hpt_data_val | high_quality | +3.120 | +4.277 | -4.812 | -4.122 | -18.841 | -27.752 |
| hpt_data_val | all | +1.504 | +0.436 | -3.276 | -0.747 | -12.508 | -20.995 |

Raw/final byte percentage means weight images by their bytes; mean bpp weights image/preset cases equally. They need not have the same percentage change. Both are reported rather than conflated.

Wins / ties / losses against release, with each image/preset counted once and latency based on within-image medians:

| Set | Preset | MSE W/T/L | bpp W/T/L | encode W/T/L | decode W/T/L |
| --- | --- | --- | --- | --- | --- |
| hpt_data | compression | 11/0/9 | 8/0/12 | 11/0/9 | 19/0/1 |
| hpt_data | balanced | 8/0/12 | 12/0/8 | 16/0/4 | 18/0/2 |
| hpt_data | quality | 9/0/11 | 3/0/17 | 17/0/3 | 20/0/0 |
| hpt_data | high_quality | 9/0/11 | 11/0/9 | 20/0/0 | 20/0/0 |
| hpt_data | all | 37/0/43 | 34/0/46 | 64/0/16 | 77/0/3 |
| hpt_data_val | compression | 1/0/4 | 4/0/1 | 4/0/1 | 5/0/0 |
| hpt_data_val | balanced | 2/0/3 | 2/0/3 | 2/0/3 | 4/0/1 |
| hpt_data_val | quality | 2/0/3 | 2/0/3 | 4/0/1 | 5/0/0 |
| hpt_data_val | high_quality | 1/0/4 | 3/0/2 | 5/0/0 | 5/0/0 |
| hpt_data_val | all | 6/0/14 | 11/0/9 | 15/0/5 | 19/0/1 |

Largest MSE regressions:

| Set | Image | Preset | MSE % | bpp % | encode median % |
| --- | --- | --- | --- | --- | --- |
| hpt_data_val | old_man_standing.jpg | high_quality | +65.285 | -0.748 | -26.671 |
| hpt_data_val | old_man_standing.jpg | quality | +38.077 | -13.835 | -24.272 |
| hpt_data | eiffel.png | high_quality | +27.790 | -2.221 | -20.368 |
| hpt_data_val | ducks.jpg | quality | +16.275 | -9.957 | -26.997 |
| hpt_data | coast.jpg | compression | +14.630 | -14.843 | -31.577 |
| hpt_data_val | old_man_standing.jpg | compression | +14.216 | -9.323 | -32.255 |
| hpt_data | ratedistortion.png | compression | +10.551 | -19.694 | -32.336 |
| hpt_data | cat.jpg | balanced | +10.219 | -32.268 | -46.618 |

All per-image changes, including rate outliers, remain in paired_comparisons.json. The current root does not establish a release improvement across quality, final rate, encode and decode jointly. Mean wins must not hide the image/preset regressions above.

Main-set emitted iterative patch counts (three initialization patches excluded):

| Source | Preset | Minimum | Mean | Maximum |
| --- | --- | --- | --- | --- |
| release | compression | 4 | 19.90 | 50 |
| release | balanced | 9 | 32.25 | 50 |
| release | quality | 11 | 33.00 | 50 |
| release | high_quality | 8 | 19.15 | 20 |
| old | compression | 4 | 19.90 | 50 |
| old | balanced | 9 | 32.25 | 50 |
| old | quality | 11 | 33.00 | 50 |
| old | high_quality | 8 | 19.15 | 20 |
| current | compression | 3 | 25.35 | 50 |
| current | balanced | 8 | 32.85 | 50 |
| current | quality | 8 | 30.20 | 50 |
| current | high_quality | 7 | 17.75 | 20 |

The filler can terminate before its requested budget. Different RNG trajectories change both patch contents and stopping points, so the combined root speed difference must not be attributed solely to output-preserving implementation changes.

## Timing variation

| Set | Preset | Source | encode mean CV % | decode mean CV % |
| --- | --- | --- | --- | --- |
| hpt_data | compression | release | 3.96 | 3.95 |
| hpt_data | compression | old | 3.38 | 4.00 |
| hpt_data | compression | current | 3.77 | 3.60 |
| hpt_data | balanced | release | 2.31 | 3.56 |
| hpt_data | balanced | old | 2.71 | 2.56 |
| hpt_data | balanced | current | 2.70 | 5.37 |
| hpt_data | quality | release | 3.33 | 3.12 |
| hpt_data | quality | old | 2.08 | 4.26 |
| hpt_data | quality | current | 3.54 | 4.15 |
| hpt_data | high_quality | release | 3.56 | 2.63 |
| hpt_data | high_quality | old | 3.63 | 2.90 |
| hpt_data | high_quality | current | 3.34 | 4.02 |
| hpt_data_val | compression | release | 2.56 | 6.09 |
| hpt_data_val | compression | old | 3.78 | 6.94 |
| hpt_data_val | compression | current | 2.88 | 5.99 |
| hpt_data_val | balanced | release | 2.99 | 4.38 |
| hpt_data_val | balanced | old | 2.66 | 2.86 |
| hpt_data_val | balanced | current | 5.07 | 5.95 |
| hpt_data_val | quality | release | 4.82 | 3.51 |
| hpt_data_val | quality | old | 1.96 | 2.51 |
| hpt_data_val | quality | current | 2.47 | 3.49 |
| hpt_data_val | high_quality | release | 2.61 | 7.23 |
| hpt_data_val | high_quality | old | 2.09 | 1.85 |
| hpt_data_val | high_quality | current | 1.94 | 2.53 |

No independent HF old-versus-new timing run was launched. Single remote compression checks and HTTP/encode durations are retained separately; they are not controlled, warmed HF latency comparisons and are never pooled with local timings.

## Reproduce

From repository root, run `python research_pass1/benchmark.py --smoke --repeats 2`, inspect results, then `python research_pass1/benchmark.py --repeats 3`. These commands overwrite only their own research output files. The immutable release snapshot is verified against GitHub tree identifiers in github_provenance.json; do not replace it with a current-code ablation.

Raw rows: research_pass1/baseline.jsonl and baseline.csv. Expanded metadata/configs: baseline_metadata.json. Summaries: baseline_summary.json, aggregate_deltas.json, paired_comparisons.json. First-repeat compressed files: streams/{release,old,current}/{hpt_data,hpt_data_val}/. Exact original user status and final artifact manifest are retained separately.
