# PBC 3.1 canonical baseline (provisional)

The presumptive compatibility reference is the downloadable GitHub V3.1 release ZIP, SHA-256 `3ee8f68b45adb3a89128251b8905d93cb99a96d911a055f91e124e724abbe069`. The release ZIP’s embedded Python source is used as the source-level oracle; the tagged source commit is not substituted for the shipped archive.

The model SHA-256 is `ffaf24aa5447b3602c53a5c4c8df14a68ee16e6bded39331d973f38d0f1ea62f`. The complete resolved preset dictionaries are referenced from `research_pass10/baseline_manifest.json#/presets` and pinned by normalized JSON SHA-256 `5a2eaab04bbfe373ec0d8e4e1f059b2003b5d90f31d12178382550682780338d`. The model path is normalized to a portable relative name; its contents are pinned separately. RNG, preprocessing and PBC3 v0 format are recorded in [PBC31_CANONICAL_BASELINE.json](PBC31_CANONICAL_BASELINE.json).

The choice remains provisional: Coast Compression from embedded Python matches frozen research exactly, but the Windows executable and the complete four-preset matrix were not executed. The HF screenshot incident cannot be attributed without its exact input/response data. Do not update deployed or released code based solely on this manifest.
