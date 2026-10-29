# PBC 3.2 packaging prototype

Both directory and one-file Windows x64 executables work. The codec is the actual published PBC 3.1 ZIP, not the research root or release tag source. Nothing was published.

## Source reconciliation

The live v3.1 tag is ac7566f8b4fe77020bc27ff720fbe9942040cca4; GitHub main was already 17e569049c1b84004540ae89317a2ccdabf89832 at entry. The [attached release ZIP](https://github.com/EgeEken/PBC/releases/tag/v3.1) is 130,098 bytes, SHA-256 edec3ec73721c3f56913db21f6089a3b6a1637fb3481e61de5b0475da8eeaede. Its runtime files match preserved Pass 16 GitHub_Source byte for byte. The original Pass 16 archive hash 08855cb3... identifies a different, fuller ZIP; do not substitute it silently.

The tag still has PBC 2.3.1/Python 3.12 packaging and lacks the split stream/trace modules. The published ZIP has Python 3.11, pbc-codec 3.1.0, the thin pbc3 re-export, corrected dependencies and module-relative model lookup. Compared with research root, PBC3.py has release labeling and pbc3_types.py changes model discovery; numerical/runtime optimization files and model agree. ../baseline.json records comparisons and all preserved staging hashes. The attached ZIP omits README.md/LICENSE although its pyproject references a README and declares MIT. Its wheel data-file layout was not revalidated; this prototype imports the frozen flat runtime and bundles model data explicitly.

## Use and build

See USAGE.txt. Balanced is default; compression, quality and high_quality are supported. Spaces/Unicode, RGB/RGBA, EXIF normalization, ordinary v0 streams, dimensions and model/RNG/presets are retained. Replacement requires --force. Errors return 1, argument errors 2. Inspect reports stored metadata, not inferred preset/MSE. Public API is unchanged.

Use 64-bit Python 3.11 and `python -m pip install -r research_pass26/requirements.txt`. From the research checkout:

```
python research_pass26/packaging/freeze.py
python research_pass26/packaging/build.py
python research_pass26/packaging/build.py --onefile
python research_pass26/packaging/package_directory.py
python research_pass26/packaging/test_distribution.py
python research_pass26/packaging/compatibility.py
python research_pass26/packaging/measure.py
python research_pass26/packaging/memory_measure.py
python research_pass26/packaging/decoder_dependency_check.py
```

Freeze downloads only the pinned release and rejects a digest mismatch. Build uses PyInstaller 6.16.0/hooks 2026.9, not Nuitka. GCC 13.2 and FFmpeg 6.1.1 are available; Rust/MSVC were not on PATH and Nuitka was absent from the validated environment. No alternative toolchain was installed. The custom hook collects CPU Numba dynamic imports and excludes tests, CUDA, pycc, SciPy, setuptools and demo packages. Source .py files accompany frozen modules for the Numba cache locator; model data lives beside them. Python, NumPy, Numba/llvmlite, Pillow, LZMA and Windows runtime DLLs are bundled.

| Form | Bytes | Files | Executable SHA-256 |
| --- | --- | --- | --- |
| onedir | 190957170 | 140 | b5eafdfb5816729e2d9676546ba0abf2a0d8ee35200857f7eeebeced2e7399e8 |
| onefile | 72291627 | 1 | 3295107cc0a34f25711536adcedf41b84ac977a578d1a3ae951afed82d978653 |


Local paths:

- `C:\Users\EGE\.codex\worktrees\pbc-research-experiments\PBC\research_pass26\artifacts\onedir\pbc\pbc.exe`
- `C:\Users\EGE\.codex\worktrees\pbc-research-experiments\PBC\research_pass26\artifacts\onefile\pbc.exe`
- Directory ZIP: `C:\Users\EGE\.codex\worktrees\pbc-research-experiments\PBC\research_pass26\artifacts\PBC32Research-Windows-x64.zip`.
- Independent relocation root: `C:\Users\EGE\AppData\Local\Temp\PBC26 portability 5uw5oy99`.

Keep pbc.exe and _internal together in directory mode. Both forms were copied outside the checkout and launched from another directory without virtualenv activation, with PATH restricted to Windows/System32. This is not a clean Windows VM test with Python uninstalled. Tests run as the current user; locked-down installations, alternate Windows versions and antivirus/signing behavior remain untested.

## Timings

Pinned logical P-core 12, one numerical thread. Fresh-process startup is not a flushed OS/disk-cache experiment. First encode uses a new Numba cache. Final measurements use blocking subprocess waits: no memory polling steals measured CPU. Earlier sampled diagnostics are not used as performance claims.

| Runtime | Version median s (3) | First encode command s | Imports s | First codec s | Second command s |
| --- | --- | --- | --- | --- | --- |
| python | 0.212 | 4.649 | 0.793 | 3.482 | 1.922 |
| onedir | 0.120 | 4.059 | 0.358 | 3.414 | 0.944 |
| onefile | 2.058 | 8.040 | 1.704 | 3.794 | 4.555 |

First/second command encode the same 96x95 RGB Quality fixture. First use includes JIT compilation/loading; the second process reuses that runtime cache. Additional presets/RGBA may need signatures. Frozen execution defaults to writable per-user caching. One-file extraction/imports repeat and are materially slower; directory mode is preferred for interactive use.

| Runtime | Image | Warm encode median s | Warm decode median s | Encode range s | Whole benchmark command s |
| --- | --- | --- | --- | --- | --- |
| python | googlewiki.png | 0.5823 | 0.0512 | 0.5481-0.6567 | 4.301 |
| python | cat.jpg | 0.6769 | 0.1263 | 0.6561-0.6805 | 5.167 |
| onedir | googlewiki.png | 0.6976 | 0.0550 | 0.6451-0.7272 | 7.043 |
| onedir | cat.jpg | 0.6429 | 0.1198 | 0.6360-0.6581 | 7.233 |
| onefile | googlewiki.png | 0.6063 | 0.0507 | 0.5836-0.6137 | 10.012 |
| onefile | cat.jpg | 0.6752 | 0.1201 | 0.6716-0.6964 | 10.884 |

Complete PBC3.compress/decompress calls, three in-process warmed repetitions, internal MSE/LZMA enabled. Whole benchmark command adds imports, initial warmup and all repetitions; it is not one encode latency. Case/runtime order is partially rotated, not a persistent interleaved trial. Small runtime differences are diagnostic, not a claim that bundling accelerates kernels. User decode additionally preflights the stream and writes PNG; command latencies remain in measurements.json. Streams/pixels match the reference exactly.

Separate sampled Cat benchmark process-tree RSS peaks are recorded in memory.json (initial encode plus one warm encode/decode). Python/directory/one-file are about 472.4/507.4/532.1 MiB. Sampling is 50 ms, sums bootloader/worker where present, and those latencies are discarded; this is a one-pass RSS observation, not an exact OS peak or a kernel-memory comparison.

## Dependencies and readiness

Core encoder: NumPy, Numba, llvmlite, Pillow and standard-library LZMA. No pandas, Torch, OpenCV, SciPy, Optuna, Gradio, FastAPI or imageio is needed. Benchmark tooling uses psutil; bundling uses PyInstaller. Animation optionally uses imageio/imageio-ffmpeg. Training/demo dependencies stay outside.

Seven files across all presets/RGBA plus stored/warmup decode with every JIT function replaced by an assertion-raising stub: no Numba kernels execute on those decoder paths. Shared encoder imports still require Numba. A source-level decoder split could remove Numba/LLVM from a decoder-only app; the combined app still needs them. No stub/removal is shipped. NumPy implements bit grids/palettes/canvas; Pillow supplies exact signed interpolation, color conversion and resizing as well as I/O. Eliminating either needs a measured implementation.

Both final binaries pass four RGB presets plus Quality RGBA, reference stream/pixel equality, inspect, overwrite refusal, missing input and seven malformed files. Additional stored-entropy and warmup v0 streams decode identically in Python and both executables. Corrupt LZMA becomes a normal error without traceback. Bodies/dimensions/patch geometry are bounded before allocation. Research limits (32M pixels, 64 MiB file, 128 MiB body, 10k patches) may reject legitimate unusually large files and are documented; this is not a general hostile-file security audit.

Before official 3.2: test clean supported Windows, finalize wheel/model layout and license notices, decide installer/directory versus one-file UX, and verify cache/permission/update behavior. No GUI, installer, signing, release tag or upload. Binaries/caches are ignored; build/test scripts and compact evidence are committed.
