# Existing chroma allocation control results

## Design

A is the unmodified Pass 35 libaom 3.12.1 reference with `aq-mode=0`, `tune=psnr`, `deltaq-mode=0`, and `sb-size=64`. B uses the same binary and settings plus `enable-chroma-deltaq=1`. Both use speed 6, one encoder thread, 8-bit 4:2:0, full range, CICP 1/13/1, and ignored metadata. The low-rate A points and their warm plus three timed runs were reused from Phase B. All other points have one warmup and three timed repetitions per configuration; high-rate runs were interleaved A/B.

Pairs are within 1.30% of file size, so no bounded rate calibration was needed. Scores are measured on the completed AVIF decoded to RGB. SSIMULACRA2 uses the established Pass 39 binary. Timings are medians of the three timed repetitions. A/B output hashes are deterministic across repetitions.

## Completed-file results

Negative RGB MSE change favors B. These are direct near-matched-rate points; each byte delta is shown explicitly.

| Image | q | Config A→B bytes | Δ bytes | Δ RGB MSE | Δ PSNR | Δ SSIMULACRA2 | Δ encode median | Δ decode median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| googlewiki.png | 5 | 23,362 → 23,384 (+0.094%) | +22 | +0.128% (+0.7271) | -0.0056 dB | -0.1328 | +4.46% | +1.09% |
| googlewiki.png | 20 | 46,920 → 46,859 (-0.130%) | -61 | -0.137% (-0.4276) | +0.0060 dB | -0.0115 | -1.73% | +1.86% |
| cat.jpg | 35 | 290,049 → 289,984 (-0.022%) | -65 | -0.093% (-0.0400) | +0.0040 dB | -0.0606 | -0.29% | +0.55% |
| cat.jpg | 50 | 558,313 → 557,906 (-0.073%) | -407 | +0.237% (+0.0481) | -0.0103 dB | -0.1143 | +1.93% | +10.28% |
| coast.jpg | 80 | 167,920 → 167,137 (-0.466%) | -783 | +0.264% (+0.0085) | -0.0114 dB | +0.0009 | +7.07% | +9.27% |
| coast.jpg | 90 | 892,048 → 903,576 (+1.292%) | +11,528 | +0.981% (+0.0225) | -0.0424 dB | -0.1029 | -1.29% | -4.60% |

The `+2` chroma qindex offsets measurably affect the stream, but completed-file effects are very small and change direction. B improves MSE slightly at Googlewiki q20 and Cat q35; it worsens MSE at the other four points. The largest MSE change is +0.981% at Coast q90, with B also 1.292% larger. There is no consistent RGB or SSIMULACRA2 advantage. Decode-time shifts are noisy and do not indicate a stable decoder cost; encode changes range from −1.73% to +7.07% in the paired medians.

## Native decoded planes at the low-rate points

Plane MSE is against the source-derived reference planes produced by the pinned libavif input loader. A/B sample differences are decoded native samples that differ between configurations. Luma differences confirm that the option can also change coupled encoder decisions; the control is not a pure post-encode chroma replacement.

| Image / q | Y plane MSE A → B | U plane MSE A → B | V plane MSE A → B | Changed native samples Y / U / V |
|---|---:|---:|---:|---:|
| googlewiki.png q5 | 333.4673 → 333.4882 | 69.2248 → 70.0262 | 39.6936 → 39.6728 | 99,861 / 38,487 / 20,306 |
| cat.jpg q35 | 37.3465 → 37.3494 | 2.6781 → 2.6387 | 2.5797 → 2.5870 | 3,219,455 / 813,057 / 472,215 |
| coast.jpg q80 | 2.3626 → 2.3628 | 0.3396 → 0.3430 | 0.3472 → 0.3519 | 917,462 / 239,375 / 261,681 |

AVIF hashes and decoded RGB hashes differ for all six A/B pairs. The option changes output bytes by −0.466% to +1.292%. Pinned `avifdec` and FFmpeg both decoded all 12 distinct AVIF files at their original dimensions.

## Decision

Gate C passes for verified activity: the option reaches the all-intra quantizer, changes native decoded Y/U/V, and changes the bitstream. It does not provide evidence for a useful rate-distortion improvement. The results do not justify Phase D's source modification: a fixed offset would be another chroma-offset experiment without a consistent allocation signal. No third configuration, rate calibration, source patch, or held-out run was made.

Raw rows, file hashes, decoder checks, native planes, and repeated timings are retained in ignored `outputs/phase_c_results.json`, `outputs/phase_c_point_summary.json`, `outputs/planes/`, and `outputs/benchmark/`.
