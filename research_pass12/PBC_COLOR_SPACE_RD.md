# PBC Pass 12 — Color-space RD feasibility

## Decision

RGB is not an RD improvement at either screened preset. It lowers independently decoded RGB MSE on 14/16 image-preset cases, but final bpp rises on all 16 cases by 24.55%–203.49%. There are no strict RD wins: 14 are quality-for-rate trades and two are strict losses. Stop before matched-rate, adaptive-selector, other-preset or holdout work.

This is a paired local screen, not an HF result. The source files and model match the frozen manifest, all eight image hashes match Pass 9, and each RGB config differs from its paired YCbCr config only in `color_space`. However, the local YCbCr stream matched the saved HF Pass 10 stream in only 6/16 cases. The current local runtime is Python 3.13.5 / NumPy 2.2.6 / Numba 0.62.1, versus the recorded HF Python 3.11.17 / NumPy 2.4.6 / Numba 0.68.0; Python 3.13.5 is also outside this repository�s `==3.12.*` requirement. The cause of the trajectory differences was not isolated. Results below describe the local paired comparison and must not be presented as exact HF reproduction.

## Path and compatibility audit

- `PBC3Config.quality()` and `high_quality()` were created from the frozen preset factories. After removing `color_space`, each paired config dict is identical to the other and to the saved baseline preset manifest.
- The same `PBC3.py`, numerical ops, search/filler heads, learned filler, policy weights, seed 2003, explicit release-compatible NumPy RNG, working-size cap, palette/action vocabulary and `use_lzma=True` path are used. The probe confirms those source and model SHA-256 values against `research_pass10/baseline_manifest.json` before encoding.
- `PBC3.COLOR_SPACES` stores RGB and YCbCr identifiers in stream v0. The existing decoder reads that identifier and converts the decoded canvas through the recorded color space. All 16 RGB files were independently decoded by the unchanged `PBC3.decompress`; all decoded dimensions equal the source and all previews equal independent decode pixels.
- No decoder, serializer, model, config default, or production path was changed.

## No-patch preprocessing floor

The floor probe loads and EXIF-normalizes each source, uses the preset’s same downsample policy, treats the resulting working color image as perfectly represented, converts it to RGB, and performs the decoder’s final resize. `roundtrip` measures just input RGB → selected space → RGB at source size. `ideal` measures this complete no-patch working-image reconstruction at source size.

| Image | Working size | YCbCr roundtrip MSE | YCbCr ideal MSE | RGB ideal MSE |
| --- | ---: | ---: | ---: | ---: |
| cat.jpg | 612×408 | 2.243 | 280.575 | 278.654 |
| flower.jpg | 612×408 | 2.206 | 22.121 | 20.120 |
| monet.jpg | 490×511 | 2.221 | 40.297 | 38.275 |
| googlewiki.png | 501×499 | 0.909 | 684.311 | 683.836 |
| mario64.png | 577×433 | 1.548 | 89.179 | 87.837 |
| eiffel.png | 487×487 | 2.425 | 2.425 | 0.000 |
| stockpoint.jpg | 612×408 | 0.983 | 8.588 | 7.658 |
| ratedistortion.png | 509×392 | 0.076 | 0.076 | 0.000 |

RGB removes the small YCbCr roundtrip loss. On downsampled images, the total floor difference ranges from 0.48 to 2.02 MSE; downsampling/resizing dominates the floor. The completed-file MSE shifts are often much larger, so avoiding conversion rounding does not explain the full screen result. The remaining change comes from different residuals, learned actions, channel selection, patch counts and serialized grids; this screen does not isolate their causal shares.

## Run protocol and provenance

`research_pass12/run_color_space.py` performs the fixed 8×2×2 screen and floor probe. It warms each preset/color pair once on a 96×96 synthetic image, then takes two interleaved repeats per case with `rng_mode='release'`. Full `PBC3.compress` and `PBC3.decompress` calls are timed; input loading and independent RGB scoring are outside the timers. Numerical thread variables and Numba threads are set to one, and the process is pinned to CPU 0 where `psutil` is available. Two repeats are too few for fine speed claims; use timings as descriptive screen results.

`research_pass12/analyze_color_space.py` checks paired config identity and input hashes against the Pass 9 equivalence archive, compares YCbCr streams against the saved Pass 10 run, and creates compact CSV/JSON summaries. `paired_results.csv` contains all 16 paired cases; `preprocess_floor.csv` contains the 16 no-patch observations; `screen_summary.json` contains aggregate results and provenance. Detailed repeat rows remain in ignored local `color_space_results.json` and `preprocess_floor.json`.

- Frozen codec source: [GitHub 8006f30](https://github.com/EgeEken/PBC/commit/8006f30debfc4a57289df06df21636aeb52baffd)
- Research branch at run: [GitHub b339f23](https://github.com/EgeEken/PBC/commit/b339f23576cda5c372c3fe9f3e37866e1d274b11)
- HF baseline: [98ade0c](https://huggingface.co/spaces/EgeEken/PBC/commit/98ade0c8072a4571e9674eaac3d91f861e3b9867)
- Baseline ID: `pbc31_release_rng_baseline`; model SHA-256: `ffaf24aa5447b3602c53a5c4c8df14a68ee16e6bded39331d973f38d0f1ea62f`
- Runtime source SHA-256: `PBC3.py` 74fe069600f8190f1a5ac1b4cb01dff29f4b1262d6d36da459236e7a9fb0016c; `pbc3_types.py` c09981d314bb0b176de5ab30050fa2a6b7013f371849b90efcfff1641e9b5c7f; `pbc3_heads.py` d48cfc36ce59e3c1874c92a771c18a4a451b7126f8aeb6b74c82a7ae1a47e482; `pbc3_ops.py` 62243c349dd5f74c6248abad0013a38aeb416984074bd47e3dbdfec6fd11739e; `learned_filler.py` 65d2de14c1fd01be256dbbcdfa98001db8312cfdb7a032f4e40b2c54a58b0c50; `pbc3_stream.py` 4730b8eaebf93e3e5b577ed67ecfd2150eef264b920c9cd64d28c308ba304e94; `pbc3_features.py` f6f19e560072636bc733b5b0885daace4ea724bd0100672dfcd0a2628c293c33.
- Local runtime: Windows 11, Python 3.13.5, NumPy 2.2.6, Numba 0.62.1, Pillow 11.3.0, CPU affinity `[0]`, numerical threads 1.

The saved YCbCr baseline matched the Pass 10 file hashes and patch counts for only 6/16 local cases. Mismatches: Cat HQ; Flower Quality; Monet Quality/HQ; Googlewiki Quality; Mario64 Quality; Eiffel HQ; Stockpoint Quality/HQ; Rate-distortion Quality. The same source, model, config, seed, and image bytes were verified. The local/HF runtime versions differ as listed above; no same-runtime HF check was run because RGB did not pass the RD screen. No HF deployment or benchmark was performed.
