# Pass 27 handoff

PBC 3.2.0rc1 is a local engineering candidate, not a published release. Compression/model/presets/RNG/entropy/v0 format retain actual published 3.1 behavior.

1. Source/wheel pip installs pass: fresh environments, pip check, unrelated CWD, site-packages origin, four reference streams/model SHA. README/INSTALL/MIT license and package-data model placement corrected.
2. Combined directory app is ready for broader tests. UTF-8 errors/Unicode paths, overwrite/malformed behavior, actual RX-only directory/file permissions and user cache pass under non-admin token with Python/compilers off PATH.
3. Frozen Numba cache fallback depended on CWD. Pinned source locator fixes requested/default user cache and different-CWD reuse. No generated caches ship. First-ever encoding still compiles; startup/kernel timings remain separate.
4. Decoder runs without installed Numba/LLVM or stubs/model/encoder imports: 32 full files, 64 partials, 45 invalid/resource cases and concurrency pass. Checked one-pass parsing avoids redundant validation. Directory is 65.6% smaller extracted than combined; optional one-file decoder works.
5. Native per-call backend passes 96 three-way comparisons: 5.98% aggregate reduction versus candidate reference, exact output/MSE, no serious regression. Auto native when DLL present (Windows app); portable wheel uses reference without DLL. Repeated Cat/Flower peaks save about 147 MiB. No global patch/new kernel.
6. Clean source checkout regenerates both apps and identical wheel payloads without existing artifacts. Functional reproducibility, not promised bit-identical executable hashes. Source revision is in packaging/clean_checkout.json.
7. No accessible Windows VM. Before official 3.2: clean Windows/CPU coverage, installer/update/signing decision, final licensing review and real-user testing. No GUI/installer, HF/native Linux speed claim, release tag/upload.

Reports: packaging/PBC32_DISTRIBUTION_RESULTS.md, decoder/PBC_DECODER_ONLY_RESULTS.md, native/PBC_NATIVE_INTEGRATION_RESULTS.md. artifacts.json lists exact executable/ZIP/wheel/sdist/DLL paths/bytes/SHA256. provenance.json records frozen sources, inputs/model, versions/threads and preserved Pass 26/staging. Build dependencies are pinned separately for complete and Numba-free environments.

All new work stays under research_pass27. Binaries/images/streams/clones/caches/logs are ignored. Final commits/GitHub HEAD and URLs are supplied in the chat. Main/rust/v3.1, release ZIP, HF, earlier code/research/staging stay unchanged. No Video/4.0 or unrelated kernel/RD work.
