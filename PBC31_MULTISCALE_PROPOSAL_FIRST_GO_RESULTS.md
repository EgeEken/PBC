# PBC 3.1 Closure Controls and Multiscale Proposal Replay

## Scope

- Branch: `multistage_pbc31`
- Starting commit: `039174f144981ae09830f98a225299c1299d2067`
- Production presets, stream version, decoder, and model: unchanged
- Closure controls: five-image diagnostic set, all four presets
- Proposal replay: 67 captured baseline residual states
- Proposal full-codec control: five images, all four presets, actual `old_pbc`

## Serialized multistage closure

### Matched rate and quality

The leading serialized candidate was the fixed `0.5x -> 1.0x` plan with a `5/95`
iterative-patch allocation.

| Preset | Comparison point | Codec | MSE | bpp | Encode s | Decode s |
|---|---|---|---:|---:|---:|---:|
| Compression | Matched rate | Multistage | 404.968 | 0.085813 | 0.3865 | 0.0519 |
| Compression | Matched rate | Single p45, q=0.35 | 424.951 | 0.086141 | 0.2904 | 0.0497 |
| Compression | Matched quality | Multistage | 404.968 | 0.085813 | 0.3865 | 0.0519 |
| Compression | Matched quality | Single p45 | 402.494 | 0.097019 | 0.3530 | 0.0571 |
| Balanced | Nearest joint point | Multistage | 337.364 | 0.174690 | 0.5457 | 0.0737 |
| Balanced | Nearest joint point | Single p45 | 326.715 | 0.180735 | 0.3353 | 0.0598 |
| Quality | Matched rate | Multistage | 291.153 | 0.318367 | 0.5004 | 0.0762 |
| Quality | Matched rate | Single p35 | 283.448 | 0.315368 | 0.4405 | 0.0768 |
| Quality | Matched quality | Multistage | 291.153 | 0.318367 | 0.5004 | 0.0762 |
| Quality | Matched quality | Single p30 | 287.979 | 0.309813 | 0.4211 | 0.0759 |
| High quality | Matched rate/quality | Multistage | 238.695 | 0.755123 | 0.5092 | 0.1333 |
| High quality | Matched rate/quality | Single p18 | 238.867 | 0.753146 | 0.4986 | 0.1252 |

Ordinary single-stage dominates the multistage candidate at quality and high
quality. It is substantially faster at the nearest balanced point. Compression
retains a narrow representation signal: at matched bitrate, multistage lowers
MSE by 4.70%, but costs 33.1% encode time. At matched MSE it saves about 11.6%
bitrate, but still encodes slower.

This narrow compression-preset result does not offset the full-main failure
reported previously.

### Coarse initialization only

This control runs initialization at 0.5x, immediately resizes to the normal
final working resolution, then spends the entire iterative patch budget there.

Changes versus the normal current single-stage codec:

| Preset | MSE | bpp | Encode | Decode |
|---|---:|---:|---:|---:|
| Compression | -0.58% | -7.40% | +10.45% | -2.32% |
| Balanced | +6.28% | -6.08% | +2.00% | +2.42% |
| Quality | +1.67% | +1.36% | +18.00% | -0.23% |
| High quality | +2.70% | -4.73% | -10.82% | -9.08% |

Moving initialization is not independently robust. It gives a narrow
compression-preset rate benefit, but balanced and high-quality remain
rate-quality trades, while quality regresses both MSE and bpp.

### Closure decision

The fixed serialized multistage architecture remains rejected. It is retained
only as an isolated research harness on `multistage_pbc31`.

## Multiscale proposal replay

### Design

The proposal experiment keeps:

- normal final-resolution initialization;
- one final-resolution canvas;
- v0 stream framing and decoder;
- unchanged patch construction and serialization;
- unchanged total patch count.

For each captured residual state it constructs a 0.5x BOX-filtered absolute
residual analysis layer and mixes:

- 50% coarse-parent-derived proposals;
- 40% normal final-resolution residual proposals;
- 10% unrestricted spatial exploration.

All candidate boxes are mapped to and exact-scored at final resolution.

Recall means that the top-k set contains the exact baseline box or contains an
equal/better candidate under the production model action and exact RD score.

### Aggregate replay

| Shortlist | Exact box recall | Box-or-equal/better recall | Median relative exact-RD | Nominal exact evaluations avoided vs 10 |
|---|---:|---:|---:|---:|
| Top 1 | 14.9% | 53.7% | 0.0% | 9 |
| Top 4 | 20.9% | 68.7% | +7.16% | 6 |
| Top 10 | 22.4% | 68.7% | +7.73% | 0 |

The production learned path currently exact-scores one selected action, so
top-4/top-10 replay would add exact work rather than save it. The avoided counts
above are relative to a hypothetical ten-candidate exact stage.

Mean proposal-generation time:

| Path | Time |
|---|---:|
| Baseline proposals | 5.76 ms |
| 50/40/10 multiscale proposals | 16.98 ms |
| Change | +194.7% |

### Recall by preset

| Preset | States | Top 1 | Top 4 | Top 10 |
|---|---:|---:|---:|---:|
| Compression | 9 | 22.2% | 44.4% | 44.4% |
| Balanced | 17 | 70.6% | 82.4% | 82.4% |
| Quality | 17 | 58.8% | 82.4% | 82.4% |
| High quality | 24 | 50.0% | 58.3% | 58.3% |

### Recall by image type

| Image type | States | Top 1 | Top 4 | Top 10 | Proposal ms |
|---|---:|---:|---:|---:|---:|
| Smooth/bokeh | 13 | 53.8% | 84.6% | 84.6% | 17.2 |
| Textured art | 13 | 53.8% | 61.5% | 61.5% | 15.5 |
| Text/synthetic | 14 | 21.4% | 35.7% | 35.7% | 16.2 |
| Object/portrait | 10 | 60.0% | 80.0% | 80.0% | 18.6 |
| Landscape | 17 | 76.5% | 82.4% | 82.4% | 17.6 |

Coarse parents supplied the best replay candidate in 37/67 top-1 states,
40/67 top-4 states, and 39/67 top-10 states. Dynamic full-resolution proposals
supplied most of the remainder; unrestricted exploration was best only three
times across the three shortlist sizes.

### Full-codec gate and exact fallback

The mixed-only path did not meet the recall or proposal-time gate and was not
run as a full codec.

An exact fallback control runs the multiscale analysis but returns the untouched
baseline proposal path. It was byte-identical to current v0 for all 20
image-preset pairs.

Diagnostic aggregate:

| Source | MSE | bpp | Encode s | Decode s | Raw bytes | Final bytes |
|---|---:|---:|---:|---:|---:|---:|
| Actual `old_pbc` | 303.178 | 0.356263 | 0.6264 | 0.1199 | 112,165 | 50,151 |
| Current single-stage | 306.050 | 0.355684 | 0.4375 | 0.0810 | 113,181 | 49,905 |
| Multiscale analysis + exact fallback | 306.050 | 0.355684 | 0.8344 | 0.0821 | 113,181 | 49,905 |

Fallback changes versus current:

- MSE: 0.00%;
- bpp: 0.00%;
- encode time: +90.74%;
- decode time: +1.45% timing noise;
- raw and final bytes: 0.00%;
- MSE/bpp wins: 0, losses: 0, ties: 20;
- encode wins: 0/20.

## Decision

The initial multiscale proposal generator is rejected in its current form. It
demonstrates that coarse parents can contain strong candidates, especially for
smooth, portrait, and landscape residuals, but the 50/40/10 replacement loses
too much baseline coverage and costs almost three times the proposal time.

The next defensible iteration is a cheaper coarse-anchor seed added to the
existing full-resolution sampling pass, sharing the already-built residual and
integral data. It should not build and rank three independent proposal pools.
Text/synthetic images need either a bypass or near-complete baseline coverage.
