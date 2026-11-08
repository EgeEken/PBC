# Frozen Pass 40 rate calibration

The six initial quality points and completed-file rates have been recorded. Rate-only calibration targets are the initial A rate for B and the initial C rate for D at each corresponding source/quality point. B is PSNR + Variance Boost; D is full IQ. No image distortion or perceptual score is used to choose these qcolor values.

The nearest qcolor is estimated from the measured local log(bpp)-versus-qcolor slope for the delta-Q-enabled configuration. Each selected value below will be encoded as an actual AVIF and measured. These are not interpolated RD results. The maximum is two added qcolor settings for B and two for D per image; no other quality search is planned.

| Image | Target | Config | Frozen added qcolor | Rate target |
|---|---:|---|---:|---|
| Googlewiki | q5 A | B | 3 | A q5 |
| Googlewiki | q20 A | B | 18 | A q20 |
| Googlewiki | q5 C | D | 4 | C q5 |
| Googlewiki | q20 C | D | 18 | C q20 |
| Cat | q35 A | B | 27 | A q35 |
| Cat | q50 A | B | 45 | A q50 |
| Cat | q35 C | D | 28 | C q35 |
| Cat | q50 C | D | 45 | C q50 |
| Coast | q80 A | B | 71 | A q80 |
| Coast | q90 A | B | 84 | A q90 |
| Coast | q80 C | D | 72 | C q80 |
| Coast | q90 C | D | 85 | C q90 |

Main and added encodes keep speed 6, one job, 8-bit YUV420, full range, CICP 1/13/1, automatic tiling and fixed `sb-size=64`. Matched-rate results require actual file rates within 5%; unmatched points remain unmatched, with no extrapolated quality claim.

## Bounded rate-only follow-up

After measuring the first calibration files, some pairs still missed 5%. The remaining budget was frozen using actual bytes only. These are the final added settings; a given mode-off configuration receives no more than two new qcolors per image.

| Image | Config qcolor | Match target |
|---|---:|---|
| Googlewiki | C q6 | D q4 |
| Googlewiki | C q18 | D q18 |
| Cat | A q33 | B q27 |
| Cat | A q51 | B q45 |
| Cat | C q51 | D q45 |
| Coast | A q82 | B q71 |
| Coast | A q91 | B q84 |
| Coast | C q82 | D q72 |
| Coast | C q92 | D q85 |

Each added point is a direct encode. Pair eligibility and any conclusion depend on its measured bytes, not predicted rate.
