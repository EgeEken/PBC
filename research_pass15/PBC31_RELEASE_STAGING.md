# PBC 3.1 local release staging

The publication directory is prepared locally and has not been published:

`C:\Users\EGE\Desktop\Coding\Personal Projects\Completed Projects\Image Processing\Probabilistic Brush Compression\New Demo\HF Space\PBC\PBC3_1_Release_Staging`

It contains `PBC3_1_Release.zip`, `GitHub_Source/`, `RELEASE_NOTES.md`, `RELEASE_MANIFEST.json`, and `SHA256SUMS.txt`. The ZIP SHA-256 is `395e76290965c0e44fe1844b2738c3a0a193d0ce3c3090c6e274af8c10d30312` (139,588 bytes). Every entry in the checksum file was verified after copying.

The archive has a flat root with the top-level codec modules, model, README, animation script, installation metadata, and supporting modules. It preserves `from PBC3 import PBC3` and `from pbc3_types import PBC3Config`; `from pbc3 import PBC3, PBC3Config` re-exports those same objects. It includes the optimized `pbc3_ops.py` for candidate `pbc31_integral_fused`, with model resolution relative to the installed modules.

The final ZIP was extracted and installed in isolation with Python 3.11.16, NumPy 2.4.6, Numba 0.68.0, and Pillow 11.3.0. Validation passed for module imports, flat and installed APIs, all four preset encode/decode paths with internal MSE, RGB/RGBA handling, repeated-stream determinism, the frozen cat stream SHA-256, PBC 3.0 v0 decode compatibility, and the model hash. The ZIP contains no datasets, research scripts, benchmark databases, or HF server.

The manifest reports the direct local PBC 3.0-versus-staged-candidate result: 34.00% lower encode time and 64.28% lower decode time over eight images, four presets, and three repetitions, with 96/96 byte-identical matched streams. The HF cpu-basic result remains separate at 28.83% and 64.38%. The fused integral candidate separately improved encode time by 12.00% against the frozen PBC 3.1 baseline in its repeated four-image screen and matched outputs across all 100 main/holdout image-preset pairs.

The prior frozen-baseline staging copy remains at:

`C:\Users\EGE\Desktop\Coding\Personal Projects\Completed Projects\Image Processing\Probabilistic Brush Compression\New Demo\HF Space\PBC\PBC3_1_Release_Staging_Frozen_Baseline`

The intermediate candidate package before final metric metadata is preserved at `PBC3_1_Release_Staging_PreFinalMetrics` beside it. This is prepared for a future upload; no GitHub main update, release, tag, or HF deployment was made.
