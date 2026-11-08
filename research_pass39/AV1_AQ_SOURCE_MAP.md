# AV1 spatial allocation source map

Pinned sources: libaom 3.12.1 (`10aece4157eb79315da205f39e19bf6ab3ee30d0`) and libavif 1.3.0 (`1aadfad932c98c069a1204261b1856f81f3bc199`), both under `research_pass35/sources/`.

## Option path and still-image encode path

`libavif/src/codec_aom.c` registers `aq-mode` as `AV1E_SET_AQ_MODE` and `tune` as `AOME_SET_TUNING` (around lines 402–411). The no-tune default is `AOM_TUNE_SSIM`. The CLI accepts `tune=iq` through the pinned AOM codec option interface, even though libavif's older enum table only lists PSNR and SSIM. All five frozen configurations encoded successfully.

In the tested all-intra still-image setup, libaom selects `DISALLOW_RECODE`: `av1/encoder/encoder.c::encode_with_recode_loop_and_filter` calls `encode_without_recode` when that speed-feature setting is active (around lines 3177–3199). The frame setup for `aq-mode=1` and `aq-mode=2` is in `encode_with_recode_loop` around lines 2977–2979. `encode_without_recode` only calls an AQ setup for cyclic refresh; it does not call `av1_vaq_frame_setup` or `av1_setup_in_frame_q_adj`. Diagnostic traces recorded `aq_mode=1` alongside `DISALLOW_RECODE`, with no VAQ setup call. Thus, in this measured path, variance AQ is accepted and stored in the config but never applied to the frame.

## Variance AQ and complexity AQ

`av1/encoder/aq_variance.c::av1_vaq_frame_setup` prepares energy-class segment q offsets from frame average energy and a fixed rate-ratio table. During coding, `partition_search.c::setup_block_rdmult` assigns segment IDs from block energy when `vaq_refresh` is enabled. `av1/encoder/aq_complexity.c::av1_setup_in_frame_q_adj` prepares five complexity segments, and `av1_caq_select_segment` selects among them. Both setup mechanisms are in the recode loop in this pinned source. `aq-mode=2` was source-mapped only and is not in the experiment.

The production configuration's one-pass no-recode path makes these variance and complexity setup routines inactive for the measured still encode. This is a scoped conclusion about this source/build/configuration, not a claim about libaom AQ in video or recode-enabled encodes.

## `tune=iq` and effective qindex

The `AOM_TUNE_IQ` documentation in `aom/aomcx.h` says the tune sets quant-matrix levels 2–10, sharpness 7, `dist-metric=qm-psnr`, adaptive CDEF, chroma delta-Q, and `deltaq-mode=6`; it is restricted to all-intra mode. In `encodeframe.c::setup_delta_q`, mode 6 calls `av1_get_sbq_variance_boost` (around lines 242–291). IQ also selects 64×64 superblocks. `aq_variance.c` computes a robust octile summary from the 64 luma 8×8 subblock variances, and `allintra_vis.c` maps it to a qindex adjustment. This path is separate from variance-AQ segmentation.

Research-only qindex instrumentation in a separate libaom build observed a flat qindex 164 and segment 0 for every A and B superblock root. IQ config C had 2,128 coded superblocks with qindices 108–164; segment ID was 0 throughout because no variance-AQ segmentation was prepared. Configuration D had the same qindex map and AV1 bytes as C. The diagnostic AV1 output was byte-identical to the unmodified reference encoder for REF/A/B/C/D at Cat q35.

The measured D result therefore does not test two active spatial allocators composing: variance AQ is inactive on this encode path, while IQ's superblock delta-Q is active. IQ also changes quant matrices, RD distortion weighting, CDEF and chroma delta-Q, so its total output change cannot be attributed to qindex allocation alone.

## Evidence and limits

`research_pass39/outputs/qmap/` contains diagnostic traces and per-sample maps. The root-only distributions and mask-aligned distortion analysis are in `outputs/benchmark/spatial_analysis.json`; the map plot is `visuals/spatial_allocation.png`. Qindex values are not regional bitrate measurements. AQ-off versus variance-AQ equality was reproduced across the main matrix, and D versus C equality was reproduced at all measured points; both pairs had byte-identical files and reconstructions.
