# Native final-MSE experiment

A compiled C metric preserves every tested stream, decoded pixel and internal MSE. Across the requested three images x two presets, mean case-median encoding time falls 5.70%. Cat improves about 10%; Eiffel does not improve. Preserve an optional path, not a mandatory dependency.

## Profile and mechanism

The actual published ZIP is verified through ../baseline.json. A warmed separate TimingTrace compress/decompress was run for Cat, Googlewiki and Eiffel, Q/HQ, pinned P-core 12 and one numerical thread. profile.json contains inclusive/exclusive phases; instrumented time is diagnostic, not the speed trial.

Cat final_mse takes 145-196 ms and final resize 88-89 ms; proposal sampling takes 77-78 ms. Googlewiki Q sampling is 182 ms, HQ 78 ms; HQ entropy is 246 ms. Eiffel sampling is 65-75 ms; HQ entropy is 201 ms. Learned feature extraction is now about 20-50 ms here, rather than an assumed historical constant. Decoder is mostly final resize for Cat (96-99 ms) and patch read/resample for smaller images. Python rectangle bookkeeping matters, but a native rewrite must preserve conditional NumPy RNG calls. Native interpolation must reproduce Pillow semantics. Neither was attempted in this bounded pass.

Selected candidate: final RGB/RGBA MSE. The release creates two full-image float32 arrays, then performs a compiled Numba sum. This C experiment removes those float arrays and reads uint8 pixels directly. NumPy PIL-to-byte materialization remains. Squared byte differences are integers; the supported sum is below 2^53, so the integer sum/final double division equals reference double accumulation exactly.

mse.c is C11, built by MinGW-w64 GCC 13.2.0 (x86_64 UCRT) with -O3/-Wall/-Wextra/-Werror/-shared/-static-libgcc. kernel.py crosses ctypes once per image, releases the GIL, passes explicit int64 byte strides/shapes and owns array references for the call. It supports reversed/F/strided layouts without copying NumPy pixels. No mutable native state. Empty/mismatched/oversized/unsupported arrays reject; non-uint8 images use the reference through the research switch. The process-local switch is used only in isolated workers, never installed in the server.

## Correctness and timing

28 layout/shape fixtures, 100 deterministic random fixtures, zero/255 endpoints, 32 concurrent read-only calls, six invalid shapes/dtypes and three signed/float fallback cases pass. Signed extremes reject in the uint8-only kernel and reproduce reference results through fallback. Normal/small/degenerate shapes, RGB/RGBA, negative strides and repeated calls are covered. C checks counts; callers must supply legitimate NumPy views, not fabricated out-of-allocation as_strided pointers.

| Image | Input | Reference ms | C binding ms | Speedup |
| --- | --- | --- | --- | --- |
| googlewiki.png | array | 7.346 | 1.870 | 3.93x |
| googlewiki.png | PIL | 10.222 | 4.412 | 2.32x |
| cat.jpg | array | 94.727 | 17.272 | 5.48x |
| cat.jpg | PIL | 159.879 | 69.073 | 2.31x |

Five rotated component repetitions. Both include conversion/call overhead: array rows include reference float copies and ctypes wrapping; PIL rows also include byte materialization. No computation-only timer replaces a full call.

| Image | Preset | Reference encode s | Native encode s | Encode reduction | Reference decode s | Native decode s |
| --- | --- | --- | --- | --- | --- | --- |
| cat.jpg | high_quality | 0.6930 | 0.6204 | +10.48% | 0.1182 | 0.1163 |
| cat.jpg | quality | 0.6131 | 0.5526 | +9.86% | 0.1160 | 0.1162 |
| eiffel.png | high_quality | 0.5010 | 0.5025 | -0.30% | 0.0328 | 0.0308 |
| eiffel.png | quality | 0.2373 | 0.2383 | -0.42% | 0.0215 | 0.0214 |
| googlewiki.png | high_quality | 0.6063 | 0.5716 | +5.72% | 0.0500 | 0.0512 |
| googlewiki.png | quality | 0.5324 | 0.5161 | +3.07% | 0.0408 | 0.0389 |
| rgba | quality | 0.2482 | 0.2488 | -0.26% | 0.0097 | 0.0091 |

Separate source-isolated persistent workers, one full warmup per case, three interleaved repetitions, actual release/config/model and internal MSE enabled. Six main cases plus RGBA give 21 paired measurements. Bytes, pixel hashes, internal MSE and independent RGB MSE all match. benchmark.json retains repetitions and complete configs. Decoder is unchanged; small decode differences are noise. Eiffel/RGBA regressions remain visible.

Fresh-process Cat Quality peak working set: 432.77 MiB reference, 285.98 MiB native. Both begin the full encode at 138.50 MiB RSS after a tiny warmup; stream/MSE match. One isolated observation per variant, including imports/allocator retention, not a repeated memory claim.

## Reproduction and decision

```
python research_pass26/packaging/freeze.py
python research_pass26/native/profile.py
python research_pass26/native/build.py
python research_pass26/native/check.py
python research_pass26/native/benchmark.py
python research_pass26/native/memory.py
```

Use pinned requirements and restored hpt_data. The DLL is ignored artifacts/mse.dll. Production Python/Numba remains intact. Direct uint8 MSE is plausible as an optional photographic optimization with modest binding complexity and lower peak memory. It removes no encoder dependency. Confirm HF/broader sizes before integration. A wider native rewrite is not warranted: much decode work is already Pillow, sampling requires RNG-preserving redesign and entropy is native LZMA. Do not add this gain to earlier pass percentages.
