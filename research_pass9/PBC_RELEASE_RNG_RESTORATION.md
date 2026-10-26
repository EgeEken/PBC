# Release RNG restoration

The only codec change adds `rng_mode="release"` to `compress` and `compress_stream`. It constructs `np.random.default_rng(config.random_seed)` inside that encode. The default remains `"current"`, constructing the existing `PBC3Rng`. No shared generator, monkeypatch, preset, model, decoder or other algorithm change is introduced.

The immutable release is `2e59f1b06980aa467da0cde94fa33c3f9a0d4d51`. All six preserved modules were verified against Git objects, and the active original legacy model is byte-identical to the release blob: SHA256 `ffaf24aa5447b3602c53a5c4c8df14a68ee16e6bded39331d973f38d0f1ea62f`, 104,903 bytes, 32 input features, hidden width 128, 30 actions, cheap=False.

The local representative gate passed 12/12 cases, then the extension passed 88/88: 20 main and five holdout images, all four presets. Complete stream hashes/bytes counts, decoded RGB hashes, ordered parsed patch payload hashes/counts, original-size RGB MSE and final post-LZMA bpp agree exactly in every case. Each generated file was independently decoded by its source implementation; previews and internal/external MSE agree. The same v0 file is therefore accepted by both decoders with identical pixels on these cases.

Additional checks passed: omitted/explicit current selection, explicit streaming selection, four overlapping mixed-mode calls, unknown RNG rejection, and 11 invalid bounded API specifications. Local environment and all input/source/model/config hashes are retained in local_equivalence.json. Windows evidence is separate from HF evidence; HF evaluation is pending in this implementation commit.

The optimized grid reader/writer, per-encode SearchState, selected-delta reuse, resampling buffers, model caching and enabled internal MSE are retained. No old release module is modified. See README.md for reproduction and the bounded HF operation sequence.
