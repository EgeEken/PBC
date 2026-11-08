# Build and Reproduction

## Pinned sources

- libaom `v3.12.1`, commit `10aece4157eb79315da205f39e19bf6ab3ee30d0`, official origin `https://aomedia.googlesource.com/aom` (BSD 2-Clause and AOM patent license).
- libavif `v1.3.0`, commit `1aadfad932c98c069a1204261b1856f81f3bc199`, official origin `https://github.com/AOMediaCodec/libavif.git` (BSD 2-Clause; upstream `LICENSE` has third-party notices).
- Exact fetched commits for libpng, zlib, libyuv, libsharpyuv/libwebp and libjpeg-turbo are in `SOURCE_MANIFEST.json`.

## Toolchain and build settings

Windows 11 x64; Strawberry MinGW-w64 UCRT GCC/G++ 13.2.0; CMake 3.29.2; Ninja 1.12.0; NASM; Python 3.13.5, Pillow 11.3.0, NumPy 2.2.6. MSVC/Visual Studio was unavailable, so one native x64 GCC toolchain was used for every build. Runtime app PATH was set to `C:\Strawberry\c\bin;C:\Windows\System32;C:\Windows` to avoid loading mismatched MinGW runtime DLLs. The two required runtime DLLs are copied beside the ignored executables.

All libaom builds use Release, `-O3 -DNDEBUG`, x86_64, NASM enabled, AV1 encoder+decoder on, tools/examples/tests off, and static libraries. Reference is clean pinned source. Research-off adds the guarded profile patch with `PBC_PROFILE` undefined. Profile adds `-DPBC_PROFILE`, `CONFIG_COLLECT_PARTITION_STATS=2` and `CONFIG_COLLECT_COMPONENT_TIMING=1`.

libavif uses Release `-O3 -DNDEBUG`, static library, AOM `SYSTEM`, local static AOM, encoder/decoder/apps on, tests off, and locally fetched libyuv, libsharpyuv, libjpeg-turbo, libpng and zlib. The exact AOM include and archive paths are set explicitly. CMake cache and Ninja link command confirm which `libaom.a` is linked. `avifenc --version` reports libavif 1.3.0 and libaom 3.12.1.

## Rebuild and run

`scripts/rebuild.ps1` fetches the pinned commits, applies the instrumentation patch to the separate research source tree, builds reference/research-off/profile libaom libraries, and relinks the same libavif source/build configuration to each archive. The first libavif configure fetches its pinned dependencies and requires network access. Build logs and binaries remain under ignored `research_pass35/build/`, `install/`, and `outputs/`.

From the repository root in PowerShell after the rebuild:

```powershell
$env:PATH = 'C:\Strawberry\c\bin;C:\Windows\System32;C:\Windows'
$py = (Get-Command python).Source
& $py research_pass35/scripts/calibrate.py
& $py research_pass35/scripts/calibrate_high.py
& $py research_pass35/scripts/baseline.py
& $py research_pass35/scripts/profile.py
& $py research_pass35/scripts/validate_profile.py
& $py research_pass35/scripts/profile_overhead.py
```

The baseline uses one warmup pair and three timed fresh-process pairs for each of ten frozen points. It calculates RGB MSE from `avifdec` output. Validation compares profile outputs byte-for-byte with the reference and decodes them through fresh `avifdec` and Pillow AVIF frontend processes. The overhead run interleaves one warmup and three timed samples for each build. Raw files and logs remain in ignored `outputs/`.

The explicit encoding arguments are `--codec aom --qalpha 0 --speed 8 --jobs 1 --depth 8 --yuv 420 --range full --cicp 1/13/1 --ignore-exif --ignore-xmp --ignore-profile`. SharpYUV is not requested; tiling stays automatic. The bounded qcolor grid is 5, 20, 35, 50, 65, extended with 80, 90, 95 for flower and coast.

## Tests and limits

- Both reference and research-off build and encode/decode ordinary AVIF.
- Ten reference/research-off speed-8 outputs are byte-identical.
- Seven profile speed-8/speed-6 outputs match reference bytes and decode in fresh processes.
- Pillow AVIF frontend RGB matches `avifdec` RGB on the seven profile cases.
- Partition and intra counters are nonzero on active paths; RD partition/ML picker zero counts at speed 8 identify inactive paths.
- Source pins and input hashes/dimensions are recorded. No holdout was encoded.

Perl locale and optional Doxygen warnings were non-fatal. Speed-6 timing instrumentation is highly intrusive and must not be used for uninstrumented runtime claims. No algorithmic AV1 modification was selected, so held-out evaluation and Gate B were not run.


scripts/final_parity.py re-encodes the ten points through reference and final instrumentation-off control, decodes both outputs and saves byte/RGB equality rows. scripts/pass34_compare.py decodes the coast qcolor 35/50 calibration files for the overlapping Pass 34 same-rate sanity check.

