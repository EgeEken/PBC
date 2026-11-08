# Delta-Q source analysis

## Pinned path

The builds use libaom 3.12.1 at `10aece4157eb79315da205f39e19bf6ab3ee30d0` and libavif 1.3.0 at `1aadfad932c98c069a1204261b1856f81f3bc199`, inherited from Pass 35. The Pass 40 source copy contains the Pass 39 qmap diagnostic patch and the single Pass 40 patch. The reference and macro-off control produced identical AVIF hashes at all seven intervention points.

## What `tune=iq` bundles

In `av1/av1_cx_iface.c::handle_tuning`, `AOM_TUNE_IQ` is accepted for all-intra and enables quantization matrices with IQ levels 2–10, sets loop-filter sharpness to 7, selects QM-PSNR, enables adaptive CDEF and chroma delta-Q, and sets `deltaq_mode=6` (Variance Boost). IQ therefore cannot be used as a proxy for spatial delta-Q alone. Pass 40 set `aq-mode=0`, `tune=psnr`, `deltaq-mode` explicitly, and `sb-size=64` in its delta-Q component ablation.

Variance AQ (`aq-mode=1`) is another mechanism; its segmentation setup runs through the recode path. It is not enabled in these tests. `encoder_utils.c::av1_select_sb_size` forces 64×64 superblocks for mode 6. The ablation also fixes all variants to 64×64, so the mode comparison does not silently change block granularity.

## Variance Boost calculation and use

`av1/encoder/allintra_vis.c::av1_get_sbq_variance_boost` uses a robust variance statistic built from the 64 luma 8×8 blocks in the current 64×64 superblock. It sorts their integer per-pixel variances and combines lower/central/upper weighted octile values in a 1:2:1 ratio. A zero variance is treated as one.

With strength 3, the existing curve is `0.15 × strength × (−log2(variance) + 10) + 1`, clamped to `[1, 8]`; its crossover is variance 1024. The encoder converts base qindex to qstep, divides by this ratio, converts back to a target qindex, and caps the boost at 80 qindex levels. `aom_get_variance_boost_delta_q_res` selects a rounding resolution from the base qindex; `av1_adjust_q_from_delta_q_res` rounds and clamps the result.

In `encodeframe.c::setup_delta_q`, the chosen qindex is used for the current macroblock, quantizers and `rdmult_cur_qindex`, so it affects coefficient quantization and RD decisions such as transform and partition choices. This can alter final bytes and reconstruction even with identical frame-level qcolor.

## Pass 40 intervention

The only treatment change is guarded by `PBC_SPATIAL_REBALANCE` in `av1_get_sbq_variance_boost`, immediately after the existing qindex is computed:

```text
variance < 1024: qindex = min(255, qindex + 8)
otherwise:        qindex = max(1, qindex - 8)
```

It reuses the existing statistic and changes no stream or decoder behavior. Patch SHA-256: `03814c15914290c8fd6dc3ae973f84d7110c59131759af152e111fe692f33488`. The experiment tests a single bounded correction; no parameter was changed after the results were seen.
