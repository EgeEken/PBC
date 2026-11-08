# PBC V4-A prototype architecture

## Purpose and status

This is an isolated research format for testing a jointly selected coarse initialization and original-resolution region coding. It does not alter PBC3, its model, released package, GUI, or HF Space. The goals in `research/PBC_V4_GOALS.md` are product direction; this document fixes only the first bounded experiment.

## Reconstruction

1. The encoder normalizes the input to EXIF-oriented RGB.
2. It creates a coarse RGB image with an aspect-preserving cap of either 250,000 or 750,000 samples, using Pillow bicubic reduction. Small images remain native when below a cap.
3. Pillow median-cut quantizes that coarse image to a 64-entry RGB palette with dithering disabled. Palette and packed indices are part of the stream. The decoder maps indices through the palette and upsamples the RGB grid with the specified integer bilinear sampler to original dimensions. This retains PBC's coarse palette initialization idea without embedding a PBC3 stream or its patch metadata.
4. A fixed 16×16 grid covers the original-size residual `source - decoded_init`. Partial right/bottom regions are clipped to valid pixels. Every region has one 2-bit mode: SKIP, procedural, or DCT; mode 3 is invalid.
5. SKIP applies no operation. Its output is exactly the initialization pixels for that region.
6. Procedural mode chooses one local correction subtype: constant RGB residual, or bilinear residual surface represented by four RGB corner controls. Controls are quantized by the signaled procedure step. Integer interpolation is deterministic and clipped at output.
7. DCT mode contains four independent 8×8 blocks per full region, with three RGB channels per block. A hardcoded Q14 orthonormal DCT-II matrix and symmetric integer rounding define forward/inverse arithmetic. Coefficients are quantized by the signaled transform step and RLE/zigzag-varint coded. Partial blocks are zero-padded; only in-bounds pixels are applied.
8. Corrections are region-local and do not overlap. Hard boundaries are the reference. An optional header flag enables a two-pixel taper on active-region edges adjoining SKIP regions. Taper weights are 64/256 and 192/256 inside the active region; SKIP pixels remain untouched.
9. Palette, packed initialization indices, packed mode map, procedural parameters, and transform parameters share one XZ/LZMA stream. The complete file includes the fixed header and XZ framing.

## Bitstream v1

All integers are little-endian. The 32-byte fixed header uses `struct` format `<4sBBHHHHBBBBHIII`:

| Field | Meaning |
|---|---|
| `magic`, `version` | ASCII `PBC4`, version 1 |
| `flags` | entropy id (XZ=1), RGB24 color id, boundary taper bit |
| `width`, `height` | Original dimensions |
| `init_width`, `init_height` | Palette-grid dimensions |
| `region_size` | 16 |
| `palette_count` | 1–64 |
| `procedure_step`, `transform_step` | Positive quantization steps |
| `rd_lambda` | Local proxy in RGB SSE per estimated byte |
| `raw_size`, `compressed_size`, `crc32` | Payload framing and raw-payload integrity |

The uncompressed payload is: palette count RGB triples; packed palette indices using `ceil(log2(palette_count))` bits each; 2-bit row-major region modes; then variable parameter records in row-major mode order. Procedural records start with subtype 0 (three signed 16-bit constant controls) or 1 (four signed 16-bit RGB corner controls). Transform records contain four blocks × three channels, each as zero-run ULEB128 plus signed zigzag ULEB128 nonzero coefficients, ending each 64-coefficient sequence with a zero-run of 64. The shared XZ payload must decompress to exactly `raw_size`; CRC32 is checked before parsing.

Decoder limits dimensions, region count, palette size, payload size and coefficient magnitudes before allocating large arrays. It rejects trailing bytes, truncated records, invalid modes/subtypes, unsupported flags/versions, malformed XZ data, inconsistent lengths, and CRC mismatch. The decoder reads only the file and uses no model or source image.

## Encoder selection

For each initialization cap, the encoder evaluates each enabled mode per region using clipped final RGB SSE plus `lambda × estimated parameter bytes`. SKIP, procedural constants/surfaces, and DCT have fixed or computed raw parameter estimates; the 2-bit mode map is common to all options. It serializes the chosen regions once, compresses the entire payload once, and measures the completed file. Among the two initialization candidates, it chooses the complete file minimizing `total RGB SSE + lambda × actual file bytes`. This is a deliberately approximate local allocator followed by complete-file candidate selection; it does not pretend local estimates equal LZMA bytes.

No decoder-side AI, source image, or PBC3 file is required.
