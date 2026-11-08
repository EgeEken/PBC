# Chroma coding source map

## Control path

Pinned sources: libavif 1.3.0 (`1aadfad932c98c069a1204261b1856f81f3bc199`) and libaom 3.12.1 (`10aece4157eb79315da205f39e19bf6ab3ee30d0`).

In `libavif/apps/avifenc.c`, `-a/--advanced KEY=VALUE` stores an AOM codec-specific option. `libavif/src/codec_aom.c::aomOptionDefs` maps `enable-chroma-deltaq` to `AV1E_SET_ENABLE_CHROMA_DELTAQ`; `avifProcessAOMOptionsPostInit` sends it through `aom_codec_set_option()` (the legacy branch uses the matching control ID). This is an encoder option, not AVIF container metadata.

Libaom's `av1/av1_cx_iface.c` parses that setting into `extra_cfg.enable_chroma_deltaq`, and encoder configuration carries it as `q_cfg.enable_chroma_deltaq`. `av1/encoder/encoder_utils.c` and encoder initialization call `av1_set_quantizer()` with the all-intra flag and the selected tuning mode. The active still-image path is therefore not skipped in the same way as the variance-AQ recode setup found in Pass 39.

## What the option does in 3.12.1

`av1/encoder/av1_quantize.c::av1_set_quantizer` sets U and V DC/AC qindex offsets from the frame's base qindex:

- Disabled: U/V DC and AC offsets are all zero.
- Enabled, all-intra `tune=psnr` (the Pass 41 control): U/V DC and AC offsets are all `+2`. Positive offsets increase chroma quantization relative to luma, usually reducing chroma precision and rate.
- Enabled, all-intra `tune=iq`, 4:2:0: U/V DC and AC are `-clamp((base_qindex / 2) - 14, 0, 16)`. This is a larger chroma-quality boost at higher qindex and ramps down at very low qindex. IQ also changes quantization matrices, CDEF, sharpness, tuning metric and other settings, so it is not an isolated chroma control.
- The source has separate 4:2:2 and 4:4:4 IQ formulas. They are not used in this pass.
- Enabled outside that all-intra IQ case, the fallback is `+2` for U/V DC and AC. The non-IQ rule applies here.

This control is a plane-level quantizer offset, not AV1's spatial `deltaq-mode` map. For a given block, `av1_init_plane_quantizers` uses the current base/block qindex plus the U/V DC or AC delta to select quantizers and dequantizers. With spatial delta-Q active, those block q changes and plane offsets coexist. Pass 41 disables spatial delta-Q (`deltaq-mode=0`) to isolate chroma allocation.

`av1/encoder/bitstream.c::encode_quantization` writes the base qindex and Y DC delta, followed by U DC/AC deltas; if `separate_uv_delta_q` and U/V differ, it also writes V's separate deltas. With this rule U and V match, so it does not need separate U/V values, but nonzero chroma deltas still alter frame-header syntax. The encoded bitstream and bytes can therefore change even if quantized coefficients happen to stay the same.

## Quantizer use and coding decisions

`av1/encoder/av1_quantize.c::av1_init_plane_quantizers` selects per-plane quantizer tables using base qindex plus the applicable plane DC/AC offset. It also selects U/V quantization-matrix levels from the adjusted chroma qindex when matrices are enabled. Coefficient quantization and dequantization therefore change directly.

Chroma has its own transform and rate-distortion decisions. `av1/encoder/intra_mode_search.c` evaluates chroma-from-luma (CfL) alpha independently for U and V and compares their RD costs; CfL predicts chroma from reconstructed luma rather than copying luma values. Chroma offsets can affect residual coefficient costs, chosen transforms/modes and the amount of chroma syntax. The encoder's RD loop includes plane-specific distortion/rate and shares frame/block choices across planes, so a chroma setting can also change luma coding decisions indirectly.

## Color and reconstruction

This experiment encodes opaque 8-bit YUV420 AVIF with full range and CICP 1/13/1 (BT.709 primaries, sRGB transfer, BT.709 matrix), with metadata profiles ignored. Native Y/U/V planes are decoded from the AVIF itself. Source-derived comparison planes come from a lossless AVIF made from the same oriented RGB source with the identical libavif/libaom, CICP, range, subsampling and input conversion path. Counterfactual RGB uses pinned `avifImageYUVToRGB` with automatic chroma upsampling and the decoded AVIF's signaled color/range/siting fields.

## Phase A conclusion

The option is accepted, reaches the all-intra encoder, changes effective U/V DC/AC quantizer indices in the PSNR path, and signals the offsets in the AV1 frame header. Phase C verified that it changes every measured file and decoded planes, but found no consistent RD gain.

## Pinned source references

Line numbers below refer to the Pass 35 checkouts at `research_pass35/sources/` (the exact pinned revisions above):

- `libavif/apps/avifenc.c:294, 1890` — codec-specific `--advanced` argument parsing.
- `libavif/src/codec_aom.c:409–424, 886, 970` — option mapping and post-init application.
- `libaom/av1/av1_cx_iface.c:98, 1224, 1981–1984, 4208–4210, 4613` — option storage, quantizer config forwarding, public control and option parser registration.
- `libaom/av1/encoder/av1_quantize.c:785–843` — per-plane quantizer setup; `866–961` — effective chroma DC/AC delta formulas and `av1_set_quantizer`.
- `libaom/av1/encoder/bitstream.c:2109–2135, 3179–3180` — frame-header quantization syntax and call site.
- `libaom/av1/encoder/intra_mode_search.c:754–961` — CfL alpha RD search and its use by chroma intra-mode selection.

## Phase C activity result

With `tune=psnr`, the active option is the fallback `+2` offset for both U/V DC and AC. The twelve completed-file A/B operating points all produced different AVIF and decoded-RGB hashes. At the three low-rate diagnostic points, A/B native Y, U and V samples all changed. This establishes that the control is active in our still-image path; its RD effects remain small and content-dependent (see `CHROMA_ALLOCATION_RESULTS.md`).
