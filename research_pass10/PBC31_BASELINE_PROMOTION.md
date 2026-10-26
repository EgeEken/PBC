# PBC 3.1 development baseline promotion

Yes: `pbc31_release_rng_baseline` is now the default on the live HF demo, public compression/streaming paths and normal benchmark utilities. This is a development baseline, not an official PBC 3.1 release.

Frozen source: [8006f30debfc4a57289df06df21636aeb52baffd](https://github.com/EgeEken/PBC/commit/8006f30debfc4a57289df06df21636aeb52baffd). Deployed HF: [98ade0c8072a4571e9674eaac3d91f861e3b9867](https://huggingface.co/spaces/EgeEken/PBC/commit/98ade0c8072a4571e9674eaac3d91f861e3b9867), a direct child of 4237d954f4e1a04951c822c738113a53bbd58e9b. `baseline_manifest.json` pins these revisions, complete preset dictionaries, runtime source hashes, original model, historical control and benchmark evidence. The completion report commit is supplied in the final handoff; its codec source is the same frozen source.

## Default and active paths

Only the two API defaults change: `PBC3.compress(..., rng_mode="release")` and `compress_stream(..., rng_mode="release")`. Each call constructs its own `np.random.default_rng(config.random_seed)`. Explicit `rng_mode="current"` still constructs the previous custom `PBC3Rng`, identified as `pbc3_custom_rng_historical` and by active worker label `dev_current_rng`. Invalid modes still raise ValueError.

The production API, streaming, Speed Benchmark, Quick RD, sweep and existing production training-reference calls inherit the new default without redundant overrides. Public form fields cannot select RNG or disable internal MSE; an attempted `rng_mode=current` form field did not change release-equivalent output. User-configured seeds retain their existing path, including the focused seed-17 check. No model, preset, proposal/search algorithm, quantizer, entropy method, palette syntax, stream version or decoder changed.

The active Pass 9 worker now explicitly selects `release` for dev_release_rng and `current` for dev_current_rng. The immutable release imports are unchanged. Its runtime source manifest was refreshed; original saved Pass 9 results retain their original measured manifest. Earlier root-dependent custom-RNG scripts are documented as historical in research/HISTORICAL_EXECUTION.md with revision-pinned execution. The Pass 9 local acceptance assertion follows the new default; no old measurements were rewritten or relabeled.

## Focused correctness and readiness

The completed local gate uses 20 encodes, not a repeated 100-case or three-way timing evaluation. Four cases span all four presets: the existing 3576×2384 Cat RGB photo, a new 96×64 alpha-edge RGBA image, a new 127×95 text/graphics image and a new 17×13 native-size seeded image. Implicit default, explicit release RNG and immutable PBC 3.0 match complete files/pixels/MSE/bpp and counts. Explicit current matches the previous default on RGB and RGBA controls loaded from e6c8eca. Two streaming cases and two overlapping mixed-mode calls agree with their independent references. Invalid RNG rejection, enabled/correct internal MSE and retained old v0 decoding pass.

The first fixture attempt incorrectly used an invalid zero working-pixel cap; the immutable release rejected it. The fixture was corrected to retain the valid default cap, under which the small image is naturally native. No codec correction was needed. Three new synthetic cases match the release exactly. No additional unseen photographic dataset was introduced; broader natural-image/generalization and extreme custom-setting readiness remain outside this limited gate. Existing 100-case main/holdout and edge-condition evidence remains the broader support.

The database fixture preserves all 6,200 historical rows, leaves new provenance columns null for them, detects a missing trailing requested trial, and checks filters/read-only selection. The existing asynchronous selector/stale-response fixture and JS syntax checks pass. No training, new RD algorithm or unrelated codec benchmark ran.

## Deployment and public verification

The five-file HF child contains PBC3.py, pbc3_benchmark.py, static/benchmark.js, research_pass9/worker.py and its runtime manifest. Predeployment revision, health, idle benchmark/Quick RD/sweep/training status and the exact default-only codec diff were checked. The original dirty HF checkout was not changed; deployment uses an isolated Git index and preserves both repository histories.

A fresh predeployment database export contains 6,200 rows, 974,848 bytes, quick_check=ok, SHA256 `48f73a898571602747ec5ba555473a62ad8bb53138d78ed8f42f630189de20e1`. No older backup was restored over live records.

At deployed revision 98ade0c:

- Public Auto compression passes all four presets on normalized RGB googlewiki, matching the frozen HF release files exactly. RGB previews equal independent decode and displayed MSE/size agree.
- The unseen RGBA fixture matches its immutable-release file and preview, with MSE 6.7677001953125, displayed 6.77. The public API preserves alpha; this four-channel metric is distinct from RGB benchmark MSE.
- A native-image Manual request using the actual streaming endpoint's existing configuration matches an immutable-release reference. The streaming final frame matches its independently decoded pixels (five frames observed). The endpoint's existing frame-only response is unchanged.
- The retained old Pass 1 v0 file still decodes with identical pixels through the public decoder.
- The authenticated one-image Balanced check retains release/restored equality and distinct custom output. All three variant rows also match their saved Pass 9 file/pixel/MSE/bpp/count records. Unauthorized research returns 403; invalid authenticated specifications return 400.
- Final health is normal, the deployed revision is exact, all job types are idle, and served /benchmark.js matches the tested source bytes.

No critical public regression was observed and no rollback was needed.

## New public Speed Benchmark baseline

Normal public run **20261026_172335** completed PBC3 only: 20 main images × four presets × three trials = 240 measured encode/decode pairs. Every expected image/preset/trial key is present exactly once, no codec failure was logged, and all 240 file hashes match the 80 frozen HF PBC 3.0 cases. Patch counts and final bpp also match in every row.

| Preset | Mean encode s | Mean decode s | RGB MSE | Final bpp | Mean patches |
| --- | ---: | ---: | ---: | ---: | ---: |
| Compression | 0.676739 | 0.109350 | 274.481629 | 0.083968 | 25.650000 |
| Balanced | 0.732021 | 0.109431 | 193.429023 | 0.168491 | 33.700000 |
| Quality | 0.727759 | 0.104064 | 160.078404 | 0.298480 | 32.800000 |
| High Quality | 0.777287 | 0.107219 | 143.403430 | 0.532350 | 22.050000 |
| Overall | 0.728451 | 0.107516 | 192.848121 | 0.270822 | 28.550000 |

These are ordinary public-run arithmetic means over all 60 rows per preset, with overall equal weighting of all 240 rows. The actual production codec is called. Full encode time includes Image.fromarray and internal MSE; full decode includes original-size reconstruction. Dataset loading and existing preload occur before timing. Independent external float32 RGB MSE, patch-header parsing, file hashing and logging occur outside codec timers. Small decimal MSE differences from Pass 9's float64 reduction are numerical scoring precision, not different files.

This runner keeps its normal trial/preset/image grouping, without pinned affinity, numerical-thread restrictions or paired source interleaving. Mean within-case timing CV is 10.91% encode and 17.48% decode. It uses 20 images, arithmetic means and normal server scheduling/warmup rather than Pass 9's eight-image per-case medians, per-preset full warmups and isolated pinned workers. Different absolute times do not establish a new speed regression or a new release speedup. No historical public average is used as a release comparison, and the controlled three-way benchmark was not repeated.

The established paired comparison remains **28.83% lower encode time, 64.38% lower decode time, exact release RD**, measured at HF 4237d954 and documented in the Pass 9 reports. This promotion selects that already-validated path implicitly.

New Speed Benchmark run metadata stores the development label, baseline ID, release RNG, seed, requested coverage, exact eight-module source hashes and model hash. Source hashes are checked against frozen GitHub 8006f30 and mapped to HF 98ade0c in baseline_manifest.json. New rows also store parsed patch counts and complete-file hashes. Nullable schema additions preserve old records, which receive no inferred RNG/version label. No new bitstream version or public RNG UI control is introduced.

The selector now shows the development label only for known new metadata. Latest complete selects this 240-row PBC-only run; historical runs remain separate and unlabeled. Actual response-driven frontend checks verify the visible label, sample count, PBC-only plots and preserved historical labels; served JS bytes match. The final database has 6,440 rows, all original 6,200 row values unchanged, quick_check=ok, SHA256 `1df81e864c35a20daaabe110f597be5e1b67830596d779083dbdc7f21cfbc56e`. Database exports, raw responses and fixture images remain ignored locally.

## Frozen baseline

| Setting | Value |
| --- | --- |
| Research identifier | pbc31_release_rng_baseline |
| GitHub source | 8006f30debfc4a57289df06df21636aeb52baffd |
| HF deployment | 98ade0c8072a4571e9674eaac3d91f861e3b9867 |
| Original release reference | 2e59f1b06980aa467da0cde94fa33c3f9a0d4d51, unchanged |
| Original model SHA256 | ffaf24aa5447b3602c53a5c4c8df14a68ee16e6bded39331d973f38d0f1ea62f |
| RNG / seed | Per-encode NumPy default_rng (PCG64 in recorded environment), 2003 |
| Working image | Automatic aspect-preserving 250,000-pixel policy; smaller inputs native; final reconstruction at original size |
| Working color / channel cycle | YCbCr / Sum; alpha retained when present |
| Initialization | Existing automatic channel initialization, three patches for RGB/four for RGBA |
| Stream / decoder | Existing PBC3 v0, generated palettes, STORE/raw LZMA2; decoder unchanged |
| Runtime | Vector grid I/O, per-encode exact search-state reuse, selected-delta reuse, contiguous float32 PIL resampling buffers, model-weight cache |
| Recorded HF libraries | Python 3.11.17, NumPy 2.4.6, Numba 0.68.0, Pillow 11.3.0 |
| Historical custom control | pbc3_custom_rng_historical / explicit rng_mode=current / dev_current_rng |

| Preset | Requested iterative patches | Learned quality | Search / proposal / exact depth | Internal MSE |
| --- | ---: | ---: | --- | --- |
| Compression | 50 | 0.4 | 200 / 50 / 10 | enabled |
| Balanced | 50 | 0.6 | 200 / 50 / 10 | enabled |
| Quality | 50 | 0.8 | 200 / 50 / 10 | enabled |
| High Quality | 20 | 0.95 | 200 / 50 / 10 | enabled |

Actual counts can terminate early. Complete definitions, including initialization/palette/search/resolution fields, are in baseline_manifest.json. No additional preset class, codec copy or Git branch was created. Future RD experiments must use this frozen release-compatible development source as their primary control, while retaining the immutable PBC 3.0 reference and explicitly naming custom-RNG comparisons.

## Release objectives and next experiment

| Objective versus PBC 3.0 | Current status |
| --- | --- |
| Better reconstructed quality | Equal on verified cases; strict improvement not achieved |
| Lower final bitrate | Equal on verified cases; strict improvement not achieved |
| Faster encoding | Controlled aggregate time reduced 28.83% |
| Faster decoding | Controlled aggregate time reduced 64.38% |

This establishes a faster release-equivalent development baseline, not a strict four-axis improvement or official release. Remaining readiness limits include the finite default-preset/edge corpus, limited unseen natural-photo coverage, shared HF timing variation and historical tools that require pinned execution. Existing unsupported/extreme configurations are not newly certified.

The highest-priority next RD experiment is to test final-resolution RGB distortion-aware candidate ranking against the frozen baseline while keeping proposal and representation budgets fixed. First compare the current working-space score with actual RGB distortion on frozen candidates; any forward screen must demonstrate matched-rate improvement. No such experiment was started in this task.

## Reproduction and rollback

See README.md for the exact local, deployment, public-check, private correctness and 240-pair public benchmark commands. `python research_pass10/analyze.py` regenerates tables and the frozen manifest from the saved public response and Pass 9 references without running codecs. `node research_pass10/check_ui.cjs` checks the actual result-driven frontend. The compatibility input and generated fixtures are local dependencies, not committed datasets. Historical scripts follow research/HISTORICAL_EXECUTION.md.

If a critical public regression requires rollback, first export the latest database, verify idle jobs/current HEAD, review the reverse diff of HF 98ade0c, and create a normal non-destructive revert on top of the then-current HF history. Push that child without force and verify health, the restored custom default, v0 decode and the prior public reference. Preserve new database rows and nullable columns; never restore an old database merely to undo code. Do not reset, rewrite history, discard unrelated checkout changes or merge GitHub/HF histories. No rollback was performed.
