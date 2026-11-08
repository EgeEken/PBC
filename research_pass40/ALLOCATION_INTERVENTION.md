# Frozen spatial rebalancing intervention

**Patch SHA-256:** `03814c15914290c8fd6dc3ae973f84d7110c59131759af152e111fe692f33488`

## Evidence and hypothesis

In matched-rate Cat comparisons, standard Variance Boost lowered smooth-region RGB MSE by about 3% while increasing edge-region MSE by about 5–7% and texture-region MSE by about 5–8%, in both PSNR and IQ component pairs. The root maps showed a larger qindex decrease in smooth pixels than in edge/texture pixels. This is consistent with a resource balance that favors smooth blocks; qindex is diagnostic evidence, not a measurement of regional bytes.

The source's current variance crossover is 1024: its qstep ratio reaches 1 there, so normal Variance Boost gives no extra reduction to the qindex above that variance. The intervention moves one bounded amount of quantization in opposite directions around this existing crossover.

## Exact rule

Source function: `av1/encoder/allintra_vis.c::av1_get_sbq_variance_boost`, immediately after its existing bounded `sb_qindex` is computed.

Let `v` be the existing robust superblock variance statistic and `q` be the existing Variance Boost qindex:

```text
if v < 1024: q_exp = min(255, q + 8)
else:        q_exp = max(1,   q - 8)
```

The implementation is compiled only when `PBC_SPATIAL_REBALANCE` is defined. It reuses the already-computed statistic and adds one comparison, one bounded adjustment and no image preprocessing. The `+8` reduces the existing smooth-region boost; the `−8` favors the higher-variance group. Eight qindex levels equal one delta-Q signaling step when the frame base qindex is at least 160. `av1_adjust_q_from_delta_q_res` still applies libaom's resolution rounding and range limits afterward.

This is a redistribution rule, not a promise of rate neutrality. A thresholded split may have unequal pixel or superblock counts, and quantization step size is nonlinear in qindex. Only final AVIF bytes determine rate. The rule may overspend on high-variance blocks, reduce smooth-region quality, or change the encoder's partition decisions.

## Frozen validation

Use REF (stock libaom, existing Variance Boost), CTRL (the patch compiled with its macro off), and EXP (the same patch with its macro on). REF and CTRL must be byte-identical for all seven development points before EXP results are interpreted. Keep `tune=psnr`, `aq-mode=0`, `deltaq-mode=6`, `sb-size=64`, speed 6, one job and the existing AVIF settings fixed.

Use Googlewiki q5/q20, Mario64 q50, Flower q80, Cat q35/q50, and Coast q80. Run three warmed, interleaved uninstrumented Release encodes for REF and EXP. Calibration is limited to two extra qcolors per image/config and selected only by measured rate. A release-quality claim requires actual files within 5% rate; regional results use the fixed Pass 39 masks. No threshold, magnitude or source change may be adjusted after seeing these results.

## Frozen rate-only follow-up (before encodes)

The fixed-q run completed with REF/CTRL byte parity at all seven points. Rate-follow-up is warranted because four points differ from REF by more than 5% in final bytes. Candidate qcolors were selected using only the fixed-q byte ratios and previously measured local REF/B rate slopes; no distortion scores were used.

| Image | Fixed-q target | Probe qcolors | Direction |
|---|---:|---:|---|
| Googlewiki | q5 | 3, 4 | Lower rate to match REF; q20 is already within 5% |
| Flower | q80 | 82, 83 | Raise rate to match REF |
| Cat | q35, q50 | 38, 53 | Raise rate to match REF |
| Coast | q80 | 84, 85 | Raise rate to match REF |
| Mario64 | q50 | none | Within 5% at fixed q |

Each probe is one new EXP qcolor for its image, with one warmup and three measured encodes. After encoding, choose the nearest rate match solely by mean file bytes, requiring a gap no greater than 5%; report unmatched targets as such. No more qcolors or patch changes are allowed.
