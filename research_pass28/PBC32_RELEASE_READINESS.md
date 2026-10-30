# PBC 3.2.0 release readiness

**READY WITH LIMITATIONS.** All mandatory acceptance gates pass. Corrected RC1
was accepted before version-only promotion; final artifacts consistently identify
3.2.0 and remain local. Nothing was published, tagged, merged to main or deployed.

- Independent fresh Windows Server 2022 and 2025 hosted runners pass. They retain
  installed development tools; no genuinely clean consumer PC was tested. Local
  non-admin/RX-only app tests pass. Consumer smoke bundle requires no Python/admin.
- All standalone forms run with Python/compiler off PATH; decoder additionally
  runs with Numba/LLVM uninstalled. PBC3 v0 and historical 3.0/3.1 pixels, stored,
  LZMA, warmup, RGBA, partials, determinism and malformed rejection remain correct.
- Compression bytes/quality remain exact. Native retains a 6.96% aggregate encode
  benefit versus reference on 24 cases, about 147 MiB lower large-photo peaks,
  and no serious regression. Decode remains comparable. Default/reused cache and
  final full-command measurements pass; first encoding still compiles kernels.
- Wheel and actual sdist install correctly. Model placement, imports, complete
  notices, embedded one-file notices, source hashes, DLL imports and checksums
  pass. The unused unresolved TBB plugin was removed; no codec feature changed.
- Recommend directory ZIPs; optional one-file-with-notices ZIP; wheel/sdist and
  exact source ZIP. No installer/updater is needed. Executables are unsigned.
- Exact source tree: fd077ba70d6855e043721953d77a65ed43bfd5ea. Build commit:
  d59fd3c20a5284b5cc441dc8275b395d54145012. Independent run:
  https://github.com/EgeEken/PBC/actions/runs/41053705104.

Remaining noncritical limits: no clean consumer-PC/other-CPU or ARM coverage;
unsigned Windows files; Python 3.11-only package target; existing decoder size
bounds; no formal security/legal certification or HF native-speed claim. No
unresolved codec, installation, dependency-import or notice blocker remains.

PBC32_DISTRIBUTION_MANIFEST.json and SHA256SUMS.txt give exact final paths/hashes.
PBC32_RELEASE_ACCEPTANCE.md contains full measurements and all requested answers.
RELEASE_NOTES.md and PUBLICATION_PROCEDURE.md are ready for review. Publication,
release-source/main change and tag creation require separate explicit approval.
