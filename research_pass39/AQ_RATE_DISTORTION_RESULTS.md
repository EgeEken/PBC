# AQ and tune rate-distortion results

All values below are measured from completed AVIF files at speed 6. Each nominal point has three timed encodes after one warmup. Repeated AVIF hashes were stable and all files decoded successfully. The reported bpp is full AVIF bytes times eight divided by source pixels. RGB MSE uses float64 arithmetic; SSIMULACRA 2 uses libjxl 0.11.2 on 8-bit RGB PNGs from the EXIF-transposed source and pinned `avifdec` reconstruction.

## Near-rate C versus A

Each row picks the actual C file nearest the A rate for that A target, with a maximum 5% rate gap. Positive MSE change means C is worse; positive SSIMULACRA 2 change means C is better.

| Image | A q | C q | Rate gap | A → C MSE | MSE change | SSIMULACRA 2 change |
|---|---:|---:|---:|---:|---:|---:|
| Cat | 35 | 35 | 0.33% | 42.355 → 44.977 | +6.19% | +7.683 |
| Cat | 50 | 50 | 2.71% | 20.108 → 20.562 | +2.26% | +4.985 |
| Coast | 90 | 84 | 4.93% | 2.263 → 2.339 | +3.34% | +0.717 |
| Flower | 90 | 82 | 1.94% | 2.368 → 2.350 | −0.74% | +0.416 |
| Googlewiki | 5 | 12 | 0.21% | 566.657 → 663.718 | +17.13% | +3.017 |
| Googlewiki | 20 | 30 | 0.67% | 311.872 → 351.522 | +12.71% | −2.922 |
| Mario64 | 50 | 48 | 3.22% | 54.944 → 57.812 | +5.22% | +2.645 |
| Mario64 | 65 | 62 | 0.45% | 49.785 → 50.953 | +2.34% | +0.583 |

At all ten nominal same-q operating points, B (`aq-mode=1`, `tune=psnr`) was byte-identical to A (`aq-mode=0`, `tune=psnr`) for every development image. At the ten measured initial matrix points, D was byte-identical to C. These are consistent with the source-traced no-recode path: variance-AQ setup is not called.

REF reproduces the Pass 35 behavior with libavif's default SSIM tune. Only two REF/A pairs fell within 5%: Coast q90 (rate gap 0.26%, MSE +0.68%, SSIMULACRA 2 +0.028 for REF) and Flower q90 (2.26%, MSE +1.73%, SSIMULACRA 2 +0.041 for REF). Elsewhere the measured points did not provide near-rate REF matches.

## Measured operating points

The complete 50-point main matrix, speed-8 subset, calibration encodes, hashes, timing samples, and per-point decoder checks are in `outputs/benchmark/results.json`; aggregates and near-rate pair records are in `outputs/benchmark/summary.json`. The plotted development curves are `visuals/rate_distortion.png`.

On the large Coast q90 memory sample, peak sampled process working set was 202.9–207.9 MB across the five configurations. This is a single sampled encode per configuration and rate, not a memory distribution. The three-run speed-8 subset adds no exception: B still matches A; C's rate/quality movement varies by content. Speed and decode timings are host-specific and do not establish a deployment result.

## Conclusion

Variance AQ was accepted by the CLI but inactive in the measured still-image encode path. IQ does create spatial qindex variation, but after rate calibration it does not give a broad RGB-MSE win: C improves MSE only on Flower among the eight reported pairs, while SSIMULACRA 2 improves on seven of eight and worsens on Googlewiki q20. That mixed metric response, plus the regional edge/texture regression, does not justify a held-out run or a production candidate.
