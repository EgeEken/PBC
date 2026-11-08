# Component ablation results

## Design

The stock Pass 35 `avifenc.exe` is REF. Pass 40 rebuilt the same pinned libaom source with the diagnostic and intervention patches present, then linked (1) a macro-off CTRL and (2) a `PBC_SPATIAL_REBALANCE` EXP against the same Pass 35 libavif objects. In the component ablation, five configs were measured: REF, A (PSNR with `aq-mode=0`, delta-Q off), B (PSNR with Variance Boost), C (IQ with delta-Q explicitly off), and D (IQ with Variance Boost). Every configuration used `sb-size=64`, speed 6, one job, 8-bit YUV420, full range and CICP 1/13/1. The six initial points were Googlewiki q5/q20, Cat q35/q50, and Coast q80/q90; each has one warmup and three timed encodes.

REF and A were byte-identical at every initial point. Same-q B/A and D/C therefore isolate the delta-Q contribution within PSNR and IQ respectively. They do not measure equal-rate quality.

## Same-q results

| Image | q | PSNR A→B bytes | B/A RGB MSE | IQ C→D bytes | D/C RGB MSE |
|---|---:|---:|---:|---:|---:|
| Googlewiki | 5 | 23,362→25,740 (+10.2%) | 568.264→537.348 (−5.4%) | 13,077→14,530 (+11.1%) | 990.818→952.110 (−3.9%) |
| Googlewiki | 20 | 46,920→49,592 (+5.7%) | 312.291→301.940 (−3.3%) | 32,248→34,337 (+6.5%) | 482.002→464.487 (−3.6%) |
| Cat | 35 | 290,049→396,810 (+36.8%) | 43.019→31.330 (−27.2%) | 221,064→297,309 (+34.5%) | 56.667→44.977 (−20.6%) |
| Cat | 50 | 558,313→711,866 (+27.5%) | 20.315→15.215 (−25.1%) | 461,510→581,492 (+26.0%) | 26.306→20.562 (−21.8%) |
| Coast | 80 | 167,920→530,753 (+216.1%) | 3.242→2.735 (−15.7%) | 211,253→548,378 (+159.6%) | 3.166→2.730 (−13.8%) |
| Coast | 90 | 892,048→1,864,421 (+109.0%) | 2.296→1.607 (−30.0%) | 931,715→1,676,419 (+79.9%) | 2.398→1.819 (−24.2%) |

Mode 6 is active and materially changes the operating point: the qindex maps are non-flat, and both qindex maps and bytes show parity between diagnostic and release outputs. These same-q improvements in MSE come with substantial, image-dependent rate increases.

## Rate-matched comparison

The calibration chose qcolor from measured rate only, using the fixed A target for B and C target for D. Ten pairs fell within 5%: six B/A and four D/C. Nine of ten had worse RGB MSE with delta-Q enabled; the remaining D/C pair was effectively neutral (−0.07%). Mean MSE change was +2.15% (worse); SSIMULACRA2 improved in eight pairs and declined in two. Full rows, selected rates and raw reconstructions are in `outputs/ablation/results.json`; pair and regional values are in `outputs/spatial_analysis.json`.

**Conclusion:** the existing Variance Boost is not a general equal-rate MSE improvement on these points. The same-q quality gains are primarily rate spending, and their size varies sharply by image and qcolor. The source-level hypothesis is confirmed; the existing mechanism is not a release candidate by itself.

## Matched-rate rows

| Enabled / control | Image | Enabled q | Control q | Rate gap | RGB MSE change | SSIMULACRA2 change |
|---|---|---:|---:|---:|---:|---:|
| B / A | cat.jpg | 27 | 33 | 2.13% | +3.01% | +0.544 |
| B / A | cat.jpg | 45 | 51 | 0.08% | +2.46% | +1.030 |
| B / A | coast.jpg | 71 | 82 | 0.12% | +0.39% | -0.014 |
| B / A | googlewiki.png | 3 | 5 | 1.33% | +3.58% | +1.020 |
| B / A | googlewiki.png | 18 | 20 | 4.56% | +4.47% | -0.854 |
| D / C | cat.jpg | 28 | 35 | 1.11% | +3.77% | +0.078 |
| D / C | cat.jpg | 45 | 51 | 0.86% | +1.99% | +0.559 |
| D / C | coast.jpg | 72 | 82 | 1.47% | +0.57% | +0.240 |
| D / C | coast.jpg | 85 | 92 | 3.52% | +1.34% | +0.081 |
| D / C | googlewiki.png | 4 | 6 | 1.93% | -0.07% | +2.169 |
