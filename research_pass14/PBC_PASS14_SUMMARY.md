# PBC Research Pass 14 summary

Pass 14 completed the native runtime audit and prepared the standalone PBC 3.1 Python package. No rate-distortion experiment was run.

- The largest decoder phase is Pillow bicubic final resizing (24.2 ms per case, about 55% of the measured 44.0 ms full decode); patch reconstruction, patch parsing, and liblzma unpacking follow.
- The largest potential native target is proposal box sampling (51.9 ms per case, about 11% of full encode), but preserving NumPy RNG state and call ordering is unresolved. Integral-image work and final resizing already run in compiled libraries.
- The existing Rust decoder built but matched Python pixels on 0/20 v0 streams. It averaged 277 ms versus Python's 45 ms on paired calls; method-2 LZMA spawned a Python fallback. No native optimization was integrated.
- `dist/PBC3_1_Release.zip` contains the Python-only frozen baseline as PBC 3.1.0 while retaining stream v0. SHA-256: `dc069a075836040d45cf7a6375dd3603cb2d3ce0b3f1392784498b68ee522084`. An extracted pip install passed all four preset round trips, RGB/RGBA checks, same-runtime determinism, a frozen cat Quality stream hash, and decoding a PBC 3.0-generated v0 stream.
- A synthetic file also decoded identically through the deployed HF Linux read-only decode endpoint. No Space files or settings were changed.
- The ZIP is ready for publication review, but no public GitHub release or v3.1 tag was created. Windows/HF cross-platform encoder stream identity remains unverified.

See `PBC_NATIVE_PROFILE.md`, `PBC_NATIVE_EXPERIMENT.md`, and `PBC31_RELEASE_READINESS.md`. Raw benchmark rows and reproduction scripts are alongside the reports. The ZIP and its checksum remain in ignored `dist/` and are deliberately excluded from Git history.
