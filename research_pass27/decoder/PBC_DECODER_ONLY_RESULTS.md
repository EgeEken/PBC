# Decoder-only results

Decoder runs with Numba and llvmlite uninstalled, without stubs, model, proposals or learned features. Only NumPy, Pillow and standard-library LZMA are required. Thin directory is 65.65% smaller extracted and 62.44% smaller downloaded than combined. Ready for broader testing.

pbc3_decoder shares existing header/grid framing and unchanged palette/signed-bicubic functions. pbc3_types supplies BitReader; unused config/result definitions import no encoder. Combined CLI uses the same light decode/inspect path. Existing pbc3/PBC3 imports retain the full encoder API/dependency. The full encoder wheel still requires Numba; no separately published Python decoder package was made.

Final reader validates prefix/entropy/header, then checks each patch's geometry, bit width and available grid bits before allocation/reconstruction. Each patch is parsed once and applied immediately. Inspect skips validated grids without pixel reconstruction. Full/partial calls still validate all trailing data/padding. Warmup, int16 wrap, float32 interpolation, clipping, YCbCr/RGBA and original-size resize match the published reader. No stream change or JIT stub.

32 files: saved 3.1 Q/HQ, all presets, RGB/RGBA, actual frozen 3.0 output, stored/LZMA, downsample/warmup, odd/1-pixel dimensions, sign bias and wrap. Full pixels/dimensions/modes, deterministic repeats and 64 partials match. 45 malformed/resource cases reject, including 128 MiB+1 expanded LZMA and oversized sparse file. Sixteen concurrent calls match. Fresh decoder-env asserts Numba/LLVM absent and no encoder imports. Bundles contain no Numba/llvmlite/model and build in that minimal environment.

## Warm core comparison

Three interleaved repetitions in isolated persistent workers, fixed hash seed, logical CPU 12. Middle column uses identical Python/libraries; final column uses actual Numba-free environment. Source reading/hashing is outside timing. Milliseconds; raw rows retain variation.

| File | Published Python ms | Thin same env ms | Thin no-Numba env ms |
| --- | --- | --- | --- |
| cat.jpg.quality | 128.382 | 112.761 | 127.855 |
| eiffel.png.high_quality | 46.972 | 40.043 | 42.548 |
| googlewiki.png.quality | 41.704 | 48.230 | 47.619 |
| python_rgba | 9.927 | 10.082 | 9.059 |

Two-pass validation had small-image overhead. Checked one-pass parsing removes duplicate work without weakening bounds. Final aggregate latency is broadly comparable; individual cases vary, so no universal speedup. The main win is size/import separation. Frozen command and warm-decode rows are in ../packaging/performance.json; shared lazy imports improve combined startup too. One-file extraction is slower than directory mode.

Sampled Cat process-tree RSS is separate in ../packaging/memory.json; polling latencies are discarded. One-pass sampled observation, not exact OS high-water comparison. Core timings exclude startup; user command timings include imports/PNG I/O. No warm-encode algorithm change.

```
pbc-decode input.pbc output.png
pbc-decode inspect input.pbc
python research_pass27/decoder/make_corpus.py
research_pass27/artifacts/decoder-env/Scripts/python.exe research_pass27/decoder/check.py
python research_pass27/decoder/benchmark.py
```

Keep a separate directory download and optional one-file decoder. Clean Windows/other CPU testing remains. Every local path/size/SHA256 is in ../artifacts.json.
