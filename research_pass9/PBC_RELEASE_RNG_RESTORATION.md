# Release RNG restoration

The only codec change adds `rng_mode="release"` to `compress` and `compress_stream`. It constructs `np.random.default_rng(config.random_seed)` inside that encode. The default remains `"current"`, constructing the existing `PBC3Rng`. No shared generator, monkeypatch, preset, model, decoder or other algorithm change is introduced.

The immutable release is `2e59f1b06980aa467da0cde94fa33c3f9a0d4d51`. All six preserved modules were verified against Git objects, and the active original legacy model is byte-identical to the release blob: SHA256 `ffaf24aa5447b3602c53a5c4c8df14a68ee16e6bded39331d973f38d0f1ea62f`, 104,903 bytes, 32 input features, hidden width 128, 30 actions, cheap=False.

The local representative gate passed 12/12 cases, then the extension passed 88/88: 20 main and five holdout images, all four presets. Complete stream hashes/bytes counts, decoded RGB hashes, ordered parsed patch payload hashes/counts, original-size RGB MSE and final post-LZMA bpp agree exactly in every case. Each generated file was independently decoded by its source implementation; previews and internal/external MSE agree. The same v0 file is therefore accepted by both decoders with identical pixels on these cases.

Additional checks passed: omitted/explicit current selection, explicit streaming selection, four overlapping mixed-mode calls, unknown RNG rejection, and 11 invalid bounded API specifications. Local environment and all input/source/model/config hashes are retained in local_equivalence.json. Windows: Python 3.13.5, NumPy 2.2.6, Numba 0.62.1, Pillow 11.3.0; source workers pinned to CPU 0, numerical thread variables one.

Main-set Quality mean bpp is exactly 0.2995946243611457 for both release and optimized release-RNG. The restored variant has zero RD difference from release in every tested local case, rather than a new strict RD improvement. This does not yet establish removal of Pass 8's HF Quality regression: HF equivalence and simultaneous three-way timing remain unperformed.

The initial deployment attempt found an existing public benchmark and stopped at the idle-state guard, before any HF change. The user then authorized stopping that existing run and asked to log findings for now. Its 996 completed rows were preserved; no Pass 9 HF job or deployment was started. See PBC_EXISTING_SPEED_RUN_LOG.md and PBC_HF_THREE_WAY_BENCHMARK.md. The prepared HF child commit is not a deployed revision.

The optimized grid reader/writer, per-encode SearchState, selected-delta reuse, resampling buffers, model caching and enabled internal MSE are retained. No old release module is modified. See README.md for reproduction and the bounded HF operation sequence.
