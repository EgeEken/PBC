# Pass 44 benchmark contract

## Frozen identities

- Repository: `EgeEken/PBC`, branch `research/experiments`, starting HEAD `3723dc14de88c6c3dcbd98dd80a097e019a92a9f`.
- V3.1: frozen released source in `research_pass42/outputs/v31_frozen`, source commit `8006f30debfc4a57289df06df21636aeb52baffd`, model SHA-256 `ffaf24aa5447b3602c53a5c4c8df14a68ee16e6bded39331d973f38d0f1ea62f`. Compression, Balanced, Quality and High Quality presets; release RNG mode. Existing Pass 42 streams are hash-verified.
- AVIF: pinned `avifenc.exe` / `avifdec.exe`, libavif 1.3.0 commit `1aadfad932c98c069a1204261b1856f81f3bc199`, libaom 3.12.1 commit `10aece4157eb79315da205f39e19bf6ab3ee30d0`. One frame, qcolor 0 and bounded q grid, speed 8, one encoder/decoder job, 8-bit YUV420 full range, CICP 1/13/1, metadata ignored. qcolor is not a bitrate.
- V4-A: Pass 42 frozen files and codec implementation.
- V4-B: Pass 43 v2 format and optimized decoder. Pass 44 selective streams reuse only existing V4 procedural/transform records and keep the same decoder and stream format.

## Inputs and distortion

Images are normalized with EXIF transpose and converted to 8-bit RGB at source dimensions. SHA-256 values and dimensions are checked against `research_pass35/SOURCE_MANIFEST.json`:

- `googlewiki.png`: 814×812, `cb2cc48d6a03b603d217b59fc4a3d3e53c6bebdff09e0a8172640bf85ccde94e`
- `cat.jpg`: 3576×2384, `75482650c75b9fc17f4ee2f40b093a6065ca9103216650df5da4dbd492915c4e`
- `coast.jpg`: 4240×2832, `7cb3bd541613e2f048a49add80d6e83109686b6744f490d894938ac887765ea2`

RGB MSE uses signed 32-bit channel differences and 64-bit square/accumulation over all original-resolution pixels and three channels. bpp is `completed_file_bytes * 8 / (width * height)`. No resize is used for primary measurements.

## AVIF path and timing limitation

The pinned command-line frontend produced the RD files and is byte-identical to the saved Pass 35 reference at overlapping qcolor points. A Pillow 11.3.0 in-process frontend was rejected: it emitted matrix coefficients 6 rather than required CICP 1/13/1 and did not match the reference AVIF bytes. A small C API timing probe linked against local static libraries crashed before `main` with Windows status `0xC0000379`; its output was not used.

AVIF timings therefore use three warmed, repeated `avifenc`/`avifdec` subprocess calls for the exact selected files. They include CLI process startup, file I/O and PNG output. They are retained as observations but are not compared numerically with in-process PBC timings. AVIF direct speed ratios are withheld. This means Gate A is incomplete despite complete rate/distortion data.

## Timing boundaries

- V3.1: one warmup and three in-process `compress` and `decompress` samples per preset/image. `NUMBA_NUM_THREADS=1`; BLAS thread pool limited to one. Streams and decoded pixel hashes match Pass 42.
- V4-B complete low encode: one warmup and three in-process samples, including frozen P3 prefix generation and V4 mode selection/serialization/XZ. `NUMBA_NUM_THREADS=1`; exact Pass 43 p3003 output hashes required.
- V4 decode: stream preloaded; one warmup and five in-process full-decode samples to original-resolution RGB; BLAS thread pool limited to one. Final selective files also receive fresh-process pixel-hash and protected-SKIP checks.
- V4 selective encode was evaluated as a bounded offline allocator over already-generated V4 regional records. Its number of completed-file evaluations is reported, but this is not a fully integrated source-to-file encoder timing and must not be presented as one.

## Rate matching and claims

The primary near-rate window is ±5% of a V3.1 completed-file target. Every percentage gap is computed from actual file bytes. Unmatched points remain separate, with no interpolation or extrapolation. V4's strict cap is 105% of the actual V3.1 Compression or Quality target. AVIF q0 is reported at its actual completed-file size. No claim is generalized beyond these three development images.
