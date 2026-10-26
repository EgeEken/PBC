# Pass 14 native experiment

## Existing Rust implementation

Built the existing GitHub `rust` branch in a detached isolated worktree at `93e9c0591a710397bb3f09d2d51473313ce23db6`. The protected branch was not changed. The MSVC linker was unavailable, so the existing CLI was built with the installed Windows GNU target using Rust 1.99.1. Build succeeded. Only local debug instrumentation and a timing harness were added in the isolated worktree; no Rust code was promoted to `research/experiments`.

## Correctness

Compared decoded RGBA arrays against the Python production decoder on 20 PBC3 v0 streams: 12 Quality/High Quality photographic cases, plus RGB, RGBA, small native resolution, RGB and YCbCr initialization-only, downsampled initialization, RGB patch interpolation, and a stream produced by the immutable PBC 3.0 runtime. **Dimensions matched for all 20; exact pixel matches were 0/20.** The raw comparison is `rust_correctness.json`.

Focused initialization-only diagnostics found that the encoded initial grid values could be made equal after a local ties-to-even rounding adjustment, but reconstructed canvas values still differed in 2,148 of 2,304 channel values in the small case. Rust uses bilinear `resize_f32` where Python calls Pillow bicubic resampling for signed-grid reconstruction. The first output pixels therefore differ before any complex patch sequence is involved. Other likely incompatibilities include custom YCbCr conversion and integer/clipping details. Fixing all cases would require replacing multiple reconstruction algorithms, not a localized correction.

The Rust LZMA2 path also failed in its native `lzma_raw_buffer_decode` attempt and fell back to spawning `python3`. With Python removed from `PATH`, method-2 decoding failed to spawn the fallback. The method-0 stored stream did not need this fallback. Thus the current Rust CLI is not an all-native decoder for the tested streams.

## Paired performance

Measured the actual decoder calls on the same machine and same 12 Quality/High Quality streams. One warmup and five paired, alternating repetitions were used on CPU 0. Stream loading, process startup, and PNG output were excluded. Python timing used `PBC3.decompress`; Rust timing used the existing `PBC3::decompress` call inside a persistent process. LZMA fallback time was included.

| Mean of per-case median | Python | Rust | Change |
|---|---:|---:|---:|
| Decode seconds | 0.04513 | 0.27742 | Rust 529.3% slower (about 6.15×) |
| Cases Rust faster | — | 0/12 | — |

Paired raw rows and timing variation are in `rust_timing_results.json`. This is a whole-call result; no kernel microbenchmark is used as evidence of improvement.

## Result

The Rust decoder built and ran, but it is neither pixel-compatible nor faster. No native component was improved or replaced, and there is **no whole-codec speed gain**. Keep this implementation experimental. There is no worthwhile localized Rust correction to take further in this pass; matching the reference would require replacing Pillow resampling and auditing color conversion, arithmetic, and fallback behavior together. The Python decoder remains the reference.
