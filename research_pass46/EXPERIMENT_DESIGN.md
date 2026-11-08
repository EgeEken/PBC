# Pass 46 frozen experiment design

## Controls and inputs

- Repository branch `research/experiments`, expected start HEAD `3723dc14de88c6c3dcbd98dd80a097e019a92a9f`.
- Primary control: canonical PBC 3.1 release-Python source, model and preset definitions from Pass 45.
- Development images: `hpt_data/coast.jpg`, `hpt_data/googlewiki.png`, and `hpt_data/mario64.png`; source bytes are SHA-256 checked against `research/assets.json` before processing.
- Low-rate target: actual completed PBC 3.1 Compression file bytes per image. Compression and Balanced are both reported. No comparison is labeled rate matched outside ±5%; the initial screen budget is at most 105% of Compression bytes.
- AVIF controls: HF-compatible Pillow 11.3.0 `quality=0` profile; pinned libavif 1.3.0/libaom 3.12.1 profile, qcolor 0, speed 8, one thread, 8-bit 4:2:0 full range, CICP 1/13/1. They remain separate.
- SSIMULACRA2 uses the installed `ssimulacra2.exe` binary, SHA-256 recorded in results, paired with `cjxl v0.11.2`; identical binary and RGB PNG inputs are used for all scores. If a call rejects a format/dimension, the score remains unavailable rather than substituted.

## Structural candidate family

One straight residual stroke consists of endpoints `(x0,y0,x1,y1)` as four little-endian unsigned 16-bit coordinates, one width byte, and three signed 8-bit RGB correction amplitudes. Its parameter record is 12 bytes. No pixel mask is transmitted. Decoder-style coverage is evaluated over only the bounded stroke box with fixed-point projection/distance and 8-bit antialias coverage; amplitudes are least-squares fitted to source-minus-decoded-base RGB residual, rounded and clamped to `[-127,127]`.

Proposal generation is frozen as follows: compute luma residual and source/base edge gradients; retain at most 5,000 source pixels with absolute luma residual at least 24 and source gradient at least 20, ranked by residual magnitude times gradient. A weighted Hough accumulator uses 30 orientations at 6-degree spacing and 4-pixel rho bins. At most 64 non-maximum-suppressed lines with at least 24-pixel support length are retained. Each is evaluated at widths 1, 2 and 4. No curves, thresholds, or geometry sweeps are allowed.

Evaluate the top eight distinct stroke candidates per image by actual local clipped RGB SSE reduction. Candidate corrections are applied sequentially when measuring a selected group; overlapping benefits are recomputed, not summed. Serialize records in a shared header/count packet and measure both raw parameter bytes and actual XZ/LZMA packet bytes. These packet sizes are feasibility diagnostics, not completed-file RD.

## Native PBC controls and local gate

For each shortlisted stroke, construct canonical generated-palette corrections over its padded bounding rectangle using the released PBC `make_patch` implementation at cell size 1/max 2 bits and cell size 2/max 3 bits, on each RGB channel. Apply the decoded grid values and serialize all three channel patches in one packet with the same XZ wrapper. The stronger local native point is selected by measured actual RGB gain/cost; no single weak learned action is used as the only comparator.

A local stroke counts as a useful structural correction only if it removes at least 0.5% of the whole decoded-base RGB SSE and visibly restores an identified structure without introducing a halo or a smooth-region tear. Efficiency is compared only against a native point whose RGB gain is at least 80% of the stroke gain. The representation gate requires at least three useful corrections across at least two content classes and at least 20% better RGB gain per actual compressed packet byte than the best comparable native point. The file-size ceiling is not exceeded in a complete codec; this pass does not infer that isolated packet gains meet that ceiling.

If the gate fails, stop without a bitstream or complete codec. If it passes, only then may a minimal integrated stream be considered under a separately documented strict 105% complete-file budget; no integrated implementation is preauthorized by favorable local packet arithmetic alone.

## Timing, metrics and visual protocol

All timing uses one numerical thread and warmed complete PBC calls from Pass 44/45 or direct timing boundaries for AVIF that exclude CLI launch and file I/O. Structural timings separately include source/base analysis, Hough proposal, candidate fitting, native-patch construction, and stroke rasterization. At least three repetitions are recorded for representative complete calls; this local probe does not claim production speed.

Record completed bytes, bpp, RGB MSE/PSNR, SSIMULACRA2, source/base/error/edge/missed-structure/smooth-error maps, stream and decoded pixel hashes, and exact encoder provenance. Visual comparisons are source-aligned crops at identical scale. No reserved validation images are used.
