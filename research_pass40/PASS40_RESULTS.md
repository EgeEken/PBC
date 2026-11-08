# Pass 40 results

## Finding

The active Variance Boost path is confirmed. At equal qcolor, it lowers RGB MSE but spends more bytes; its rate penalty ranges from 5.7% to 216% in the PSNR component points. Rate calibration reverses that result: 9 of 10 delta-Q comparisons have worse RGB MSE than their delta-Q-off targets, with one effectively neutral. This does not support using the existing boost as a general equal-rate quality improvement.

The frozen spatial redistribution produces a modest but mixed development result. At the seven fixed-q points, EXP averages 12.0% fewer bytes and 5.0% higher MSE than REF; this is a rate/quality trade, not a quality win. A rate-only calibration of at most two qcolors per image produced five pairs within 5%: two Googlewiki, one Mario64 and two Cat. Across those pairs, mean byte gap is 2.52%, mean RGB MSE improves 1.79%, MSE improves on four of five, and SSIMULACRA2 improves on three of five. Median encode time is effectively flat (+0.3%). Flower remains 15.7% off rate and Coast 6.3% off, so they are not treated as matched-quality evidence.

The regional redistribution signal is visible: edge and texture MSE improve in four of five matched pairs, but smooth MSE improves in only two. Perceptual scores and speed are mixed or flat, and two image targets remain unmatched. This is not a clear enough, general development improvement to spend the holdout budget. Gate E was not run; no holdout result is claimed.

## Gate status

- **A — Source and configuration:** pass. The active mode-6 path, forced 64×64 superblock behavior and qindex use are traced in pinned source.
- **B — Controlled component ablation:** pass. PSNR and IQ comparisons isolate delta-Q, and diagnostic qmaps reproduce release AVIF bytes.
- **C — Spatial evidence:** pass for a bounded experiment. Cat region differences justified one frozen correction around the existing variance crossover.
- **D — Frozen intervention:** mixed; no release candidate. Five of seven targets rate-match, with a small average MSE gain but mixed SSIMULACRA2 and no speed improvement.
- **E — Holdout/generalization:** not run because the development evidence is mixed and not a clear improvement.

## Reproducibility and validation

- Worktree: `research/experiments`, base commit `3723dc14de88c6c3dcbd98dd80a097e019a92a9f`.
- Source: libaom `10aece4157eb79315da205f39e19bf6ab3ee30d0`; libavif `1aadfad932c98c069a1204261b1856f81f3bc199`.
- Frozen intervention patch SHA-256: `03814c15914290c8fd6dc3ae973f84d7110c59131759af152e111fe692f33488`.
- REF and CTRL produced identical files at all seven fixed-q points. Fourteen qmap diagnostics (seven REF and seven EXP rate-pair points) matched the corresponding uninstrumented AVIF hashes.
- All 51 component qmaps matched timed AVIF hashes. FFmpeg independently decoded all 20 unique intervention bitstreams and all files in the component ablation.
- Each fixed-q REF/EXP point used one warmup and three timed encodes. Each rate probe used one warmup and three timed encodes. Source images were the pinned `hpt_data` files; MSE was computed from decoded RGB after EXIF transpose.
- Raw measurements, hashes and settings: `outputs/ablation/results.json`, `outputs/spatial_analysis.json`, `outputs/intervention/results.json`, `outputs/intervention_analysis.json`, `outputs/intervention_qmaps.json`, `outputs/ffmpeg_decode.json`, and `outputs/intervention_ffmpeg_decode.json`.

No production defaults, HF files, Pass 35–39 files, stream format, or decoder code were changed. No commit or push was made. Do not continue this patch to a holdout or production default without a new explicit pass decision. A separate future pass could isolate IQ quantization-matrix effects from delta-Q, but this pass does not recommend shipping the spatial correction.

## Frozen intervention points

The EXP qcolor was chosen by measured bytes only. “Matched” means within 5% of the REF file size.

| Image | REF q | REF bytes | REF MSE | EXP q | EXP bytes | EXP MSE | Rate gap | MSE change | SSIMULACRA2 change | Matched |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| googlewiki.png | 5 | 25,740 | 537.348 | 3 | 26,434 | 521.721 | 2.66% | −2.91% | −1.983 | yes |
| googlewiki.png | 20 | 49,592 | 301.940 | 20 | 51,865 | 292.875 | 4.48% | −3.00% | +0.659 | yes |
| mario64.png | 50 | 52,321 | 52.891 | 50 | 50,957 | 53.139 | 2.64% | +0.47% | −0.526 | yes |
| flower.jpg | 80 | 374,005 | 2.666 | 83 | 319,697 | 2.744 | 15.66% | +2.92% | −0.600 | no |
| cat.jpg | 35 | 396,810 | 31.330 | 38 | 399,371 | 31.122 | 0.64% | −0.67% | +0.452 | yes |
| cat.jpg | 50 | 711,866 | 15.215 | 53 | 727,476 | 14.786 | 2.17% | −2.82% | +0.369 | yes |
| coast.jpg | 80 | 530,753 | 2.735 | 84 | 498,150 | 2.766 | 6.34% | +1.12% | −0.140 | no |

The qcolor probes, raw repeat rows and measured encode times are in `outputs/intervention/results.json`.
