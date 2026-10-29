# Native MSE integration

Promote existing C metric to auto selection in the research Windows candidate. Frozen gate passes: 5.98% aggregate full-encode reduction versus candidate reference, exact streams/pixels/MSE, worst slowdown 2.42%, modest bundle cost and relocated RX-only execution without a compiler on PATH. No production 3.1 default/RD change.

Metric selection is local to compress/compress_stream through mse_backend=reference/native/auto, with no shared monkeypatch. Auto uses optional DLL when present, reference otherwise. Explicit native requires a working DLL; broken dependencies are not silently hidden. Non-byte/unsupported arrays retain reference behavior. Portable Python wheels omit the Windows DLL and work without compiler; Windows app bundles it and uses it by default. Both paths remain available.

32 cases: Cat, Flower, Seagull, Googlewiki, Monet, Eiffel, deterministic small RGB/RGBA; four presets each. One complete warmup/variant/case, three rotated repetitions of published 3.1/reference/native: 96 three-way comparisons, 288 measured full encodes/decodes. Internal MSE enabled. All byte counts, bpp, compressed hash, decoded pixel hash, internal MSE and external source-size RGB MSE match. benchmark.json retains every observation and normalized configs; summary.json includes means/medians/CV/min/max.

| Image | Published s | Reference s | Native s | Native reduction |
| --- | --- | --- | --- | --- |
| cat.jpg | 0.7046 | 0.7044 | 0.6313 | +10.38% |
| eiffel.png | 0.3390 | 0.3316 | 0.3266 | +1.51% |
| flower.jpg | 0.6280 | 0.6461 | 0.5823 | +9.87% |
| googlewiki.png | 0.4403 | 0.4428 | 0.4365 | +1.43% |
| monet.jpg | 0.3856 | 0.3855 | 0.3817 | +0.98% |
| rgba | 0.2430 | 0.2419 | 0.2374 | +1.88% |
| seagull.jpg | 0.5631 | 0.5713 | 0.5252 | +8.07% |
| small | 0.0991 | 0.1024 | 0.1001 | +2.20% |

Mean case medians: published 0.425335s, candidate reference 0.428251s, native 0.402631s. Native is 5.34% below published. Worst native-versus-published case is +3.80% time. Reference candidate aggregate changes +0.69%; individual cases vary. Monet HQ is the worst native/reference regression. No universal/Cat-only claim, no HF timing, no sum with Pass 26 percentages.

Two fresh processes per variant/image, tiny warmup, peak working set immediately after full Quality encode, before hashes:

| Image | Reference MiB | Native MiB | Reduction MiB |
| --- | --- | --- | --- |
| cat.jpg | 433.06 | 286.04 | 147.02 |
| flower.jpg | 432.23 | 285.53 | 146.70 |

Imports/allocator retention included. Both repetitions match stream/MSE. The C kernel avoids two full-image float32 copies; PIL byte materialization remains. uint8 sums stay below 2^53, matching reference arithmetic. 60 type/layout fixtures, signed extremes, 32 concurrent metric calls and four mixed-backend concurrent encodes pass. No new kernel.

DLL: 55172 bytes; SHA256 6c4196caffd0ce311825f2f81af6a3a51cddb96aae82ab405ce3bd74fda459a1. GCC 13.2 C11 -O3/-Wall/-Wextra/-Werror/-shared/-static-libgcc. Dependencies are KERNEL32/UCRT system APIs, no libgcc/MinGW installation. One ctypes crossing/image, int64 shapes/strides, uint64 sum, no native image allocation or mutable native state. Buffers must be legitimate NumPy views, not fabricated out-of-allocation as_strided pointers.

A real relocated non-admin read/execute-only app resolves DLL/model; all four defaults match and Python/GCC are absent from PATH. Added cost is small relative to NumPy/Numba. Default auto is justified for this Windows candidate, not a broader rewrite or official release. Other Windows/CPU coverage remains.

```
python research_pass27/native/build.py
python research_pass27/native/check.py
python research_pass27/native/benchmark.py
python research_pass27/native/memory.py
```
