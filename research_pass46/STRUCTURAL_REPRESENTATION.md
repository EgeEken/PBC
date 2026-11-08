# Source-resolution structural representation screen

## Tested primitive

The probe tested one straight residual stroke at source resolution. A record carries four little-endian 16-bit endpoint coordinates, one width byte, and three signed 8-bit RGB amplitudes (12 raw bytes). Coverage is generated deterministically by fixed-point projection/distance and 8-bit antialias coverage in the bounded stroke box; no pixel mask is transmitted. Parameters were fit to the RGB source-minus-canonical-decoded-base residual. Proposal generation used luma residual and source/base edge gradients, up to 5,000 points, a 30-angle Hough accumulator at 6-degree steps, 4-pixel rho bins, up to 64 line proposals, and widths 1, 2, and 4.

Eight local candidates were retained per image. The runner's executable ranking is by measured actual local SSE gain per compressed packet byte, then selected corrections were applied sequentially and their group gain recomputed. The frozen design prose says “top eight ... by actual local clipped RGB SSE reduction,” which is not precise about this implemented ratio ranking. This protocol mismatch is disclosed rather than retroactively changing the frozen design; the runner and raw candidate ledger define the measurements reported here.

## Local packet results

Each stroke packet was 76 bytes after shared XZ compression. The selected eight-stroke group had 104 raw record bytes but expanded to 164 XZ bytes, including framing. The 164-byte packet is 2.5% of Coast Compression bytes, 0.57% of Googlewiki Compression bytes, and 1.16% of Mario64 Compression bytes. It is only an isolated packet measurement: it does not include any complete V4 header, base, shared-compressor interaction, or rate allocation and is not a completed-file budget result.

| Image | Base MSE | Best single gain / base SSE | Runner-up | 8-stroke recomputed gain / base SSE | 8-stroke local MSE | local SSIMULACRA2 change |
|---|---:|---:|---:|---:|---:|---:|
| Coast | 73.134 | 0.221% | 0.136% | 0.889% | 72.484 | +0.628 |
| Googlewiki | 1,005.022 | 0.106% | 0.068% | 0.327% | 1,001.731 | +0.768 |
| Mario64 | 414.236 | 0.943% | 0.454% | 1.645% | 407.424 | +0.968 |

Only the top Mario64 stroke clears the pre-set 0.5% per-correction threshold; Coast and Googlewiki have no qualifying single candidate. Group gains do not count as multiple qualifying corrections, because the gate requires distinct useful, visually meaningful corrections. Visual inspection found the effects subtle and not a persuasive structural restoration.

## Native controls and interpretation

For every stroke candidate, the comparator used the released `make_patch` implementation at cell1/max2bits and cell2/max3bits, all three RGB channels, with rectangles chunked to the native 400-pixel bound and one shared XZ packet. For each of the 24 shortlisted candidates, **no** native output landed in the required 0.8–1.25× stroke-gain interval. Native controls often removed much more error (for example, Coast's top-line candidate removes 0.221% of base SSE while a native cell2/max3bits packet removes 8.55%). Consequently, the recorded “gain per XZ byte ratio vs best native” values (Coast 5.46, Googlewiki 2.20, Mario64 6.69) are not matched-gain comparisons and cannot satisfy the efficiency gate. They are diagnostic only.

## Result

The bounded primitive did not demonstrate a useful structural mode at the frozen thresholds. No bitstream syntax, decoder mode, or codec integration was added. The probe suggests isolated strokes can make small local corrections, but this representation did not restore softened details strongly enough to justify spending a V4 mode or coding budget. A different source-resolution representation should be considered only after a new hypothesis is supported by a separately frozen experiment.
