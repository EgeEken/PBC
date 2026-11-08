# Experiment design (frozen before main measurements)

## Configurations

Use pinned Pass 35 reference binaries only. Every run uses explicit AOM settings:

| ID | `aq-mode` | `tune` | Role |
|---|---:|---|---|
| REF | 0 | libavif default (`ssim`) | Exact Pass 35 option behavior |
| A | 0 | `psnr` | AQ-off reference |
| B | 1 | `psnr` | Variance AQ |
| C | 0 | `iq` | Existing still-image tuning |
| D | 1 | `iq` | Recorded interaction probe; later source tracing found variance AQ inactive on the selected no-recode path |

D was retained as an interaction probe before the active encode path was traced. The tested all-intra encode uses `DISALLOW_RECODE`, whose frame setup does not call variance-AQ setup. D is interpreted as an IQ-path check with an accepted but inactive `aq-mode=1` option, not a test of two active allocators composing. `aq-mode=2` is source-only.

## Frozen development points

Inputs and expected SHA-256 values are from `research_pass35/SOURCE_MANIFEST.json`; the runner verifies them. Two historical qcolor values per image are fixed from the Pass 35 calibration grid before these measurements:

| Image | Low-rate qcolor | Moderate-rate qcolor | Pass 35 speed-8 bpp at those points (context only) |
|---|---:|---:|---:|
| Googlewiki | 5 | 20 | 0.1935, 0.5153 |
| Mario64 | 50 | 65 | 0.3198, 0.4901 |
| Flower | 80 | 90 | 0.1658, 0.5497 |
| Cat | 35 | 50 | 0.2359, 0.4927 |
| Coast | 80 | 90 | 0.1050, 0.6484 |

The same qcolor points apply across all configurations. These are nominal-quality comparisons only; matched-rate claims will use actual bytes and a small bounded calibration around overlapping endpoints. No held-out images are used unless the development results justify a frozen validation.

## Encoding and measurement

libaom 3.12.1 / libavif 1.3.0, single still AVIF, 8-bit, YUV420, full range, speed 6, one job, qalpha 0, automatic tiling, CICP 1/13/1, metadata ignored. Source images are unchanged and decoded through the pinned `avifdec`. Per-run records include exact command, file bytes and SHA-256, decoded RGB SHA-256, RGB MSE using float64 subtraction/squaring, PSNR, encode and decode duration. MSE arithmetic is checked against a 0-vs-255 fixture. Each principal point gets one warmup and three timed fresh-process repeats in interleaved configuration order. Peak working set is sampled on one representative large image per configuration and rate.

RGB metrics compare original-resolution RGB arrays. bpp is `8 * AVIF bytes / source pixels`. Compatibility requires successful `avifdec` decoding and matching dimensions. Historical Pass 35 MSE is not used as a numeric comparator because its arithmetic overflowed.

SSIMULACRA 2 availability is checked before the main matrix. If a pinned local implementation cannot be used reproducibly, report it unavailable and keep RGB MSE as the only completed-file objective.

## Speed-8 subset

Googlewiki and Cat, one fixed point each (Googlewiki q20; Cat q50), configurations REF/A/B/C only, three timed runs after warmup. D is omitted from this limited speed check.

## Matched-rate analysis

For each source and target, select actual encodes whose final bpp differs by at most 5% where available. Compare MSE at matched bytes; no universal win is inferred from equal qcolor. A bounded additional qcolor calibration is allowed only when existing points do not bracket a near-rate pair; no more than two extra qcolor values per configuration and target. Interpolations, if any, are labeled estimates and never presented as encoded files.

## Rate-calibration addendum (frozen before calibration encodes)

The initial matrix showed that IQ's same-qcolor files were far above the AQ-off rate on several photos and graphics. To get actual near-rate points, encode IQ-only (C) at at most one predicted qcolor per target, with a second allowed only if the first misses the target by more than 5%. Estimate a trial qcolor from the measured endpoint rate trend and round to the nearest integer; choose the nearest measured A bpp as the target. These are settings to encode and measure, not interpolated RD results. Fixed first calibration values: Googlewiki q12/q30, Mario64 q48/q62, Flower q72/q81, Coast q67/q87. Cat needs no added point because both initial C/A pairs are already within 5%. The D outputs matched C byte-for-byte at every initial point, so the added q values are not duplicated for D. Selection uses rates only, not image quality scores.

Second calibration values were frozen from rate-only miss distances: Flower q65 for the low target and q82 for the moderate target; Coast q64 low and q84 moderate. These are the final allowed calibration values for those targets. No more calibration is planned.

## Spatial regions and diagnostic plan

Use the fixed source RGB converted to linear-light luma for deterministic 8x8 local classification: edge is the top 20% of Sobel magnitude; among remaining pixels, texture is the top 20% of local 8x8 luma variance; all others are smooth. Masks are computed once per image and reused for every run. Report RGB MSE, pixel count, area fraction and share of total RGB SSE per class; compare each AQ configuration against A with signed regional MSE differences.

For actual allocation, make a minimal research-only instrumentation patch to the pinned source that emits the final superblock qindex decision during encoding. It must not alter control flow or coded values. Build a separate diagnostic binary, verify byte-for-byte output parity against the reference build for representative REF/A/B/C/D points, then use the qindex map for distributions and aligned overlays. No regional bitrate attribution will be inferred from qindex. If output parity or decode validation fails, discard the diagnostic and mark Gate C incomplete.

## Decision gates

- A: source mechanisms, option forwarding and active paths established.
- B: complete-file sizes, corrected RGB metrics, timing, and decoder checks for all principal points; SSIMULACRA 2 measured or explicitly unavailable.
- C: parity-preserving effective qindex evidence plus fixed-region distortion.
- D: reproducible meaningful RD improvement at actual nearby rates, with content-class regressions reported.
- E: only if D or a specific hypothesis merits it; holdout configs/rates/regions frozen before running.
