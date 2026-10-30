# PBC 3.2.0 release acceptance

Verdict: **READY WITH LIMITATIONS**. Mandatory functional, packaging, independent
Windows and performance gates pass. Final 3.2.0 artifacts are local and await
explicit publication approval. No release/tag/main/HF change was made.

## Independent Windows validation

https://github.com/EgeEken/PBC/actions/runs/41053705104
Build/test commit: d59fd3c20a5284b5cc441dc8275b395d54145012.

| Runner image | OS | Build Python | Standalone smoke checks | Result |
| --- | --- | --- | --- | --- |
| win25-vs2026 20261021.278.1 | Microsoft Windows NT 10.0.26100.0 | Python 3.11.9 | 75 | PASS |
| win22 20261026.347.1 | Microsoft Windows NT 10.0.20348.0 | Python 3.11.9 | 75 | PASS |

Both hosted jobs independently rebuilt all three applications and wheel/sdist,
installed both Python packages in fresh environments, compared all four RGB/RGBA
presets against the actual published 3.1 runtime in the same environment, and
checked known 3.0/3.1 decoding. Runtime PATH was System32 with Python variables
cleared. No binaries were uploaded. Hosted runner tokens are administrator;
separate local tests passed under a non-admin token with actual RX-only app files
and directories and writable cache/output elsewhere.

A genuinely clean consumer PC was **not** tested. Hosted runners contain installed
development tools even when PATH excludes them. VirtualBox had no accessible
Windows guest. The supplied consumer smoke bundle requires built-in PowerShell
5.1/System.Drawing, no Python, compiler, network or administrator access. It checks
archive SHA-256 before extraction, launches every standalone form, verifies actual
PNG dimensions/raw RGB(A) hashes, Unicode, determinism, overwrite protection and
malformed rejection, then writes a readable JSON report. Work files are retained
for diagnosis. No clean-machine certification is claimed.

## Compatibility and final versions

Actual published 3.1 archive: edec3ec73721c3f56913db21f6089a3b6a1637fb3481e61de5b0475da8eeaede
(130098 bytes). Historical 3.0 stream was produced by research_pass1/release in
Pass 27; it is not an ablation. All 32 retained full streams, 64 partials, 45
malformed/resource inputs and 16 concurrent calls pass using the final Numba-free
reader. Stored/LZMA, warmup, PBC3 v0, original dimensions, RGB/YCbCr/RGBA, one-pixel
edges and signed wrapping remain exact. Final local standalone smoke passes
75 checks; the final directory apps also pass the full saved
stream/permission test. Same-environment deterministic output passes. This does
not promise compressed-file identity on every platform or dependency version.

The six-image all-preset three-way campaign contains 216 measured encode/decode
pairs (24 cases, three repetitions, three variants), with exact bytes, decoded
pixels, bpp and internal/external MSE in every comparison. Final 3.2.0 source,
installed wheel/sdist and frozen apps repeat the relevant comparisons. Native
selection is per call; model, presets, RNG, residual algorithm, entropy and v0
framing are unchanged. The model remains
ffaf24aa5447b3602c53a5c4c8df14a68ee16e6bded39331d973f38d0f1ea62f.

Corrected RC1 first passed both Windows images and all local gates. Only then was
3.2.0 prepared. The only changed runtime files versus Pass 27 are pbc_cli.py
(application version/description and cache namespace) and pbc3/__init__.py
(package version). Codec/native module bodies are identical. README, package
metadata, build names and manifests consistently use 3.2.0; stream v0 is retained.
Accepted RC1 files/evidence remain in artifacts/dist, artifacts/release-source and
artifacts/accepted-rc1. Pass 27 artifacts/source and published 3.1 remain intact.

## Performance acceptance

Windows i9-12900H, logical CPU 12, numerical/Numba threads one. Each full codec
case was warmed once per variant, then three rotated repetitions were measured
in persistent isolated workers. MSE stays enabled. Timing excludes image loading,
hashing and process startup; complete compress/decompress calls are timed.
Raw rows/configs/variation and input conventions are preserved locally and hashed
in acceptance-evidence.json. The complete campaign ran at RC1; stable promotion
changes only the two metadata files above, and final frozen CLI measurements were
rerun. No full historical RD campaign or HF timing was repeated.

| Image | Published 3.1 s | Reference s | Native s | Native reduction vs reference |
| --- | --- | --- | --- | --- |
| cat.jpg | 0.6766 | 0.6906 | 0.6134 | 11.17% |
| flower.jpg | 0.6120 | 0.6194 | 0.5430 | 12.35% |
| googlewiki.png | 0.4167 | 0.4202 | 0.4108 | 2.24% |
| eiffel.png | 0.3161 | 0.3153 | 0.3146 | 0.21% |
| small | 0.1002 | 0.0993 | 0.0989 | 0.43% |
| rgba | 0.2182 | 0.2200 | 0.2195 | 0.23% |

Mean case medians: 0.389967 / 0.394128 /
0.366688 seconds. Native benefit is 6.96%
versus reference and 5.97% versus published 3.1. Native wins 19/24 case
medians and all six image means. Worst slowdown is RGBA Quality, 2.03% versus
reference (about 5 ms, with timing variation); no substantial repeatable
regression was found. No bitrate/quality improvement or general decoder speedup
is claimed. Complete source decode averages remain comparable.

Final 3.2.0 fresh-process medians of three, seconds. OS disk cache was not flushed;
these are process startup/command observations, not cold-disk guarantees. Decode
command includes imports, reading and writing PNG. Published Python uses an
explicit timing shim around the actual released runtime, not the new CLI.

| Runtime | Version startup | Small decode command | RGBA decode command |
| --- | --- | --- | --- |
| published | - | 1.0660 | 1.1106 |
| combined | 0.1278 | 0.2725 | 0.2796 |
| decoder | 0.1216 | 0.2812 | 0.2942 |
| onefile | 1.2175 | 2.2518 | 2.0826 |

Fresh/reused Numba cache uses a fixed 39x37 RGB Quality input, one command per
phase; separate columns prevent confusing JIT/import/process cost with codec time.
The other-CWD phase reuses the same cache. Final permission checks also verify the
default per-user 3.2.0 cache and real .nbc files outside the application directory.

| Runtime/backend | Phase | Full command s | Imports s | Compress s |
| --- | --- | --- | --- | --- |
| published / published_None | empty_cache | 4.750 | 0.777 | 3.600 |
| published / published_None | reused_cache | 1.996 | 0.753 | 0.938 |
| published / published_None | other_cwd | 1.683 | 0.757 | 0.625 |
| combined / combined_reference | empty_cache | 3.728 | 0.373 | 3.065 |
| combined / combined_reference | reused_cache | 0.845 | 0.346 | 0.284 |
| combined / combined_reference | other_cwd | 0.842 | 0.344 | 0.283 |
| combined / combined_native | empty_cache | 3.452 | 0.346 | 2.817 |
| combined / combined_native | reused_cache | 0.847 | 0.353 | 0.277 |
| combined / combined_native | other_cwd | 0.898 | 0.359 | 0.318 |

Final warmed decode medians of three internal calls, milliseconds. Streams and
image hashes are checked. Source reading and process startup are outside these
internal figures; their full command costs are retained in raw evidence.

| File | Published | Combined | Decoder directory | Decoder one-file |
| --- | --- | --- | --- | --- |
| cat | 101.216 | 102.017 | 102.437 | 100.909 |
| flower | 98.432 | 98.898 | 99.870 | 99.102 |
| googlewiki | 37.132 | 37.357 | 36.417 | 38.917 |
| eiffel | 30.431 | 33.183 | 36.397 | 43.939 |
| small | 2.737 | 3.649 | 2.963 | 3.224 |
| rgba | 10.135 | 7.996 | 7.662 | 7.708 |

Two fresh processes per large-image/backend observed peak working set immediately
after complete Quality compression, before hashes. Cat: 434.42 MiB reference to
287.47 MiB native; Flower: 433.59 to 287.04 MiB. This is approximately 147 MiB
saved per process, retaining allocator/import cost. Memory was measured separately
from timings. Startup/JIT remains seconds on first encoding; one-file extraction
adds latency and is optional.

Final frozen Quality encode confirmation (three warmed calls; equal streams):

| Image | Published Python s | Combined reference s | Combined native s |
| --- | --- | --- | --- |
| cat.jpg | 0.6549 | 0.6671 | 0.5634 |
| flower.jpg | 0.6781 | 0.6970 | 0.5989 |

Complete large-image commands were also measured separately for memory in two
fresh commands per mode/image. Process-tree memory is observed every 10 ms;
all polling timings are discarded. The process tree includes the Windows venv launcher and its actual Python child.
The table reports the largest observed sum of OS per-process peak working sets;
component peaks may occur at different times, so this is not an exact concurrent
process-tree peak. Concurrent sampled RSS is also retained in the raw rows.
These high-water observations include startup and image I/O,
including PNG output for the decoder, so they are not the isolated metric table
above. Hashes/pixels match in all 16 commands. MiB, mean observations:

| Image | Published encode | Combined reference encode | Combined native encode | Decoder command |
| --- | --- | --- | --- | --- |
| cat.jpg | 449.04 | 467.61 | 276.79 | 73.61 |
| flower.jpg | 437.75 | 457.92 | 310.10 | 73.88 |

## Packaging, notices and source consistency

Fresh ordinary wheel and sdist installations pass pip check, unrelated-CWD module
origin, all four RGB/RGBA presets, installed model and every declared runtime-file
hash. Both packages have correct 3.2.0 metadata. The pure Python wheel excludes
DLLs/caches; auto falls back to reference. Windows combined contains the optional
C DLL. Decoder contains no Numba/LLVM/model or encoder imports; its build/runtime
environment actually has Numba and llvmlite uninstalled, with no stub.

Combined is 191855909 bytes
extracted; decoder is 66374388
bytes, 65.40% smaller. Download sizes are in the manifest below.

The PE audit covers 192 files with no unresolved imports.
An unused Numba tbbpool plugin originally referenced missing tbb12.dll; it is now
excluded. No new kernel or codec behavior was introduced. All affected build,
installed-source, runtime and independent Windows checks were rerun.

LICENSE, README, INSTALL, THIRD_PARTY_NOTICES, complete installed dependency license
trees, Python's Windows conditions, CPython third-party notices, OpenSSL, XZ,
MinGW, LLVM, NumPy/OpenBLAS/GCC terms, Pillow bundled-library notices and
PyInstaller bootloader/runtime-hook terms are retained. Directory notices match
source bytes; one-file embedded notices are verified entry by entry and a
companion notices ZIP is staged. The notice inventory has no unresolved item;
this engineering audit is not a legal opinion. Third-party code is not relicensed
under PBC's MIT license. Source URLs and hashes are retained in licenses metadata.

The exact source ZIP has 65 files, no compiled binaries/caches/datasets and all
model/build inputs. Its extracted directory rebuilt combined/decoder apps without
previous research source or the 3.1 asset, reproduced every wheel payload byte,
and matched all eight RGB/RGBA reference cases with Python/compiler off runtime
PATH. PE bit identity is not promised across build paths/timestamps.

Common implementation manifest: 65cbd7341f842217a4e33dcd85abe4c077e0b943106a340f8b767b308acede91.
Exact release-source root Git tree: fd077ba70d6855e043721953d77a65ed43bfd5ea.
SOURCE_TREE.json lists every input SHA-256. The eventual tag must identify a
commit containing that exact root tree, not the experimental research tip.
See PUBLICATION_PROCEDURE.md. No main merge, release-source commit/tag or upload
was performed. Protected main/rust/v3.1/HF refs and earlier sources are verified
unchanged. A private object store records the tree without a commit or tag.

## Final local artifacts

All paths are absolute. SHA256SUMS.txt covers these files and the distribution
manifest. PBC32_DISTRIBUTION_MANIFEST.json contains byte counts, final versions,
source identity and paths. No artifact is published.

| Kind | Bytes | SHA-256 | Absolute path |
| --- | --- | --- | --- |
| combined | 72817005 | 439ac835c3e1bfc5bdfb4a83c56647fa7f1fff80bb94fd75e9fad681c1a36425 | C:\Users\EGE\.codex\worktrees\pbc-research-experiments\PBC\research_pass28\artifacts\final\PBC-3.2.0-combined-Windows-x64.zip |
| decoder | 27519330 | 47b5bb0ac0feabe4daa7eaad8fc75a894aecf5336271237572b9c873122c9bee | C:\Users\EGE\.codex\worktrees\pbc-research-experiments\PBC\research_pass28\artifacts\final\PBC-3.2.0-decoder-Windows-x64.zip |
| decoder_onefile | 27352206 | d8b8038ee0e7de8a8b1d0a0ec75ebfa824040f43aef839a5341c8851cdc974f4 | C:\Users\EGE\.codex\worktrees\pbc-research-experiments\PBC\research_pass28\artifacts\final\PBC-3.2.0-decoder-Windows-x64.exe |
| decoder_onefile_with_notices | 27225941 | 0d90e04e4775873f0b2b9ba8cb266d22e2a6b338b86a196d490b780b919b34cb | C:\Users\EGE\.codex\worktrees\pbc-research-experiments\PBC\research_pass28\artifacts\final\PBC-3.2.0-decoder-onefile-with-notices.zip |
| wheel | 260099 | 040d5c75a2daa3285720c1075e3f708759a82c089a08152d5c07ce233a062adb | C:\Users\EGE\.codex\worktrees\pbc-research-experiments\PBC\research_pass28\artifacts\final\pbc_codec-3.2.0-py3-none-any.whl |
| sdist | 233307 | 6dc9391ed6ffa9b16f673252ef3f8e42ff65bc2f8f7b4d1d9e9e21e0bd3905a8 | C:\Users\EGE\.codex\worktrees\pbc-research-experiments\PBC\research_pass28\artifacts\final\pbc_codec-3.2.0.tar.gz |
| release_source | 261606 | 34d338686bc3bb62e65f044dacfff0cc155d16e1cdda63a7bfa0f0584021fd53 | C:\Users\EGE\.codex\worktrees\pbc-research-experiments\PBC\research_pass28\artifacts\final\PBC-3.2.0-release-source.zip |
| consumer_smoke_test | 57088 | 3e83d1b037dac1995a9a4ad721156b481a72f2a33a778f6462942617f460c7a3 | C:\Users\EGE\.codex\worktrees\pbc-research-experiments\PBC\research_pass28\artifacts\final\PBC-3.2.0-consumer-smoke-test.zip |

Recommend combined and decoder directory ZIPs, optional one-file-with-notices ZIP,
Python wheel/sdist and exact source ZIP, with the smoke bundle and checksums.
The raw decoder EXE is also staged. A graphical installer/update system adds no
needed capability for this release. Binaries are unsigned; Windows reputation
prompts may occur. Signing and consumer-PC/other-CPU coverage remain limitations.
The supported Python package target is x64 Python 3.11. Decode bounds remain
32M pixels / 64 MiB input / 128 MiB expanded / 10k patches; larger valid streams
may reject. No Linux/HF native performance claim, ARM build, signing or formal
security certification is made.

PBC 3.2.0 is ready for publication **with these stated limitations**, after explicit
user approval. Final artifacts already identify 3.2.0; no RC filename promotion is
needed. Build/independent test source commit and run are above. The final research
commit/GitHub HEAD containing this evidence is supplied in the chat handoff.
