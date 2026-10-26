# PBC 3.1 development — fused integral-image experiment

## Candidate

Candidate `pbc31_integral_fused` replaces the two NumPy cumulative sums and padding allocation in `pbc3_ops.integral()` with one Numba pass that writes the padded `int64` integral image directly. It does not change RNG calls, proposal ordering, patch selection, the stream, or decoder logic. The frozen comparison source is `8006f30debfc4a57289df06df21636aeb52baffd` (`pbc31_release_rng_baseline`).

## Exactness checks

Before integrating the implementation, its output was compared to the frozen NumPy function on 93 arrays covering five dtypes, empty and non-empty shapes, C/Fortran/strided layouts, negative values, fractional float casts, and `int64` limits. Three additional edge arrays (`uint64` overflow, Boolean, and NaN/infinity float conversion) also matched. Each result had the same shape, `int64` dtype, padding, and values. Non-2D input raises `ValueError` explicitly.

The completed-encode screen used isolated persistent baseline and candidate workers, warmed all four presets, preloaded EXIF-normalized RGB inputs, pinned both workers to logical CPU 12 (an i9-12900H P-core), and set numerical thread limits to one. Across four images, four presets, and three interleaved repetitions, all 48 paired streams, decoded RGB outputs, and independent RGB MSE values were identical. The full-set check repeated the identity comparison for all 20 `hpt_data` images and five `hpt_data_val` images: 100/100 image/preset pairs matched stream hash, decoded pixels, and MSE. Internal final MSE remained enabled and the preset seed remained 2003.

## Timings

The four-image screen reports the mean of per-image medians (three runs per image and preset). Decode timing is included to check for unexpected effects; this candidate only changes an encoder operation.

| Preset | Baseline encode (s) | Candidate encode (s) | Encode reduction |
|---|---:|---:|---:|
| Compression | 0.5037 | 0.4337 | 13.90% |
| Balanced | 0.5475 | 0.4566 | 16.60% |
| Quality | 0.5770 | 0.5042 | 12.61% |
| High Quality | 0.6555 | 0.6157 | 6.08% |
| Overall | 0.5724 | 0.5037 | 12.00% |

Mean decode time was 0.0648 s for the baseline and 0.0642 s for the candidate (0.80% lower, within observed timing noise). One individual encode observation was slower: `seagull.jpg/balanced` in repetition 3. All four preset aggregate medians improved.

After integrating the candidate into the production module, the one-pass 25-image identity run measured 12.10% lower mean encode time. It was not repeated, so this number is diagnostic only; the repeated four-image screen is the timing result. In the full-set run, three individual observations were slower (`old_man_standing.jpg/high_quality`, `old_man_standing.jpg/quality`, and `sunset.jpg/compression`). Mean decode time differed by 0.50% in the candidate's favor. Raw per-observation rows and microbench timings are retained locally in ignored `integral_screen_postpromotion.json` and `integral_full_correctness_postpromotion.json`.

The isolated operation median timings were:

| Array size | NumPy (s) | Fused Numba (s) |
|---|---:|---:|
| 128×128 | 0.000081 | 0.000007 |
| 512×512 | 0.004242 | 0.000544 |
| 1024×1024 | 0.017993 | 0.002309 |
| 2048×2048 | 0.115079 | 0.009367 |

The full-encode gain, rather than the microbenchmark alone, justifies promotion to the PBC 3.1 development candidate. It is a measured result on this i9-12900H and these image sets, not a universal speed guarantee. No rate-distortion change is claimed.

## Runtime and disposition

The run used Python 3.11.16, NumPy 2.4.6, Numba 0.68.0, and Pillow 11.3.0 on Windows, with one pinned P-core and all thread limits set to one. The fused implementation is now the active PBC 3.1 development source under candidate id `pbc31_integral_fused`; the original frozen baseline remains available at its commit and source hashes. The local staging ZIP is rebuilt from this validated candidate and records both its source hashes and the frozen baseline provenance.
