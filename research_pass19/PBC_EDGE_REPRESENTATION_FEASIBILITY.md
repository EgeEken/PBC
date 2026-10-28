# Pass 19 signed hard-edge feasibility

Stop after Phase A. The frozen gate required at least three useful regions spanning two images, separately in each domain. Working resolution passed 0/24 regions; original resolution passed 1/16, on Googlewiki only. No complete hybrid file or full-codec candidate was implemented.

## Provenance and scope

Research start: `c545c56686416f59b1c317658034ad93fdb78a7e`, branch research/experiments. Primary control pbc31_feature_histogram; eight runtime sources plus the policy model are byte-checked against Pass 16 commit 1c9c4016fdb213d7d16663542d268c7c3df4ed9b before/after measurement. Pass 17 DCT code, Pass 18 evidence, production v0 decoder/defaults, staged release and protected branches are preserved.

Inputs: Googlewiki, Mario64, Eiffel, Cat. Working states are Quality steps 3/8, reused from Pass 18; no new image encoding. Original states are the independently decoded Quality p10 RGB bases embedded in frozen Pass 17 files. Base byte/pixel hashes match the Pass 17 control rows. This deliberately measures potential for a sparse enhancement over the same reduced base, not a new weak control renamed ordinary HQ. Full ordinary HQ/close-rate comparisons would be required only after a completed file exists.

Python 3.11.16, NumPy 2.4.6, Numba 0.68.0, Pillow 11.3.0, local i9-12900H P-core 12, numerical threads one. Existing SciPy is imported by the historical shared harness, but the new mask implementation uses only NumPy and Python stdlib. PLAN.md, exact configurations, source/model/input/script hashes and packet hashes are retained in feasibility.json. Timings are one-pass offline components, not full encode/decode benchmarks. Raw premeasurement Windows script/plan hashes remain in feasibility.json; provenance.json also records their LF-normalized hashes. Only line endings and a trailing blank line were normalized after the experiment.

## One representation and bounded joint fitting

Each correction stores one channel, x/y/width/height (nine bytes total), one signed int16 negative level and one positive level (four bytes), and a two-bit-per-pixel ternary mask. States are zero/negative/positive; code 3 is invalid. Zero is transparent. Fit starts at signed residual means, runs four Lloyd updates with zero fixed, rounds/clamps levels to [-255,255] and assigns nearest levels with half-step ties toward zero. The reconstructed correction is applied directly without interpolating its mask.

Residuals use the actual current canvas. One analysis constructs residual SSE weighted by bounded horizontal/vertical residual gradients, pooled into 64x64 tiles. Two strongest tile/channel anchors are retained, suppressing adjacent anchors in the same channel. Each yields 64x64 and centered 128x128 rectangles, clipped at image boundaries. No segmentation/model training, binary-source assumption, exhaustive rectangles or new thresholds after observing results.

Original domain tests all four rectangles. Working domain tests the ordinary selected rectangle and the first anchor's two extents, three per state. All proposed geometries are fitted and exact-scored; geometry is not chosen from source color complexity alone. This is a bounded shortlist, not a proven optimal joint geometry/level search. No repeated sequential fitting was run.

A region group is one XZ/LZMA2 stream containing an eight-byte magic/count header and all records/masks, with two-bit packing padded only at each mask's final byte. Packet raw cost is exactly 64 + sum(104 + 2*w*h + up to 6 padding bits); every entropy-frame byte is counted. Individual packets and shared groups are actually compressed and decoded. Metadata has no per-record length because dimensions determine mask length.

These are **correction packets at a known frozen canvas/domain/dimensions**, not complete image files. A later hybrid would also need a base and its image/domain/container framing. No omitted outside bytes are used to claim completed-file savings.

## Native comparisons and reconstruction

Per rectangle, compare three actual native generated-palette grids: the ordinary model action, fixed cell1/max2bits and fixed cell2/max3bits. This prevents a weak single model action from standing in for the whole native representation. Original-resolution model features are outside the ordinary 250k operating geometry, so the two fixed native controls are especially necessary there. No exhaustive 30-action oracle is claimed.

Native patches use production make_patch, palette/quantization, actual bit writer, signed bicubic reconstruction and clipping. For fair local packet comparisons their serialized bodies receive the same eight-byte group/count prefix and one actual XZ frame. Production ordinarily shares raw LZMA across its entire stream; packet bytes are therefore not marginal production-file bytes or per-patch LZMA ownership.

Working candidates are applied to copied int32 canvases, clipped/converted through production YCbCr-to-RGB and resized with the decoder's PIL settings before original RGB SSE scoring. Original candidates act directly on decoded RGB, then clip to uint8. Stored mask packets are decoded before scoring. RGB gains are not inferred from working-channel SSE; both are retained in measurements.

The frozen regional gate requires >=20% higher RGB gain per actual packet bit against the best comparable native point, with gains within a factor 1.25 (>=80% retention). Different working proposal rectangles must also retain >=80% of the actually selected ordinary patch's RGB gain; the original selected geometry already matches. Original-resolution regions must remove >=0.5% of complete base RGB SSE. This excludes tiny efficient corrections that displace a much more useful iteration. Three regions across two images must pass in the same domain. No interpolation, extrapolation or threshold rescue was used.

## Results

Efficiency below is hard-mask gain/encoded-bit divided by the strongest comparable native efficiency; >1 favors the hard mask. Three working regions lack a comparable-gain native point and are excluded from the ratio range, not counted as wins. Model gain retention is descriptive, not rate matched.

| Domain | Image | Regions | Efficiency ratio range | Median gain / model gain | Gate wins |
| --- | --- | ---: | ---: | ---: | ---: |
| working | googlewiki.png | 6 | 0.962-1.095 | 83.3% | 0 |
| working | mario64.png | 6 | 0.592-0.947 | 86.2% | 0 |
| working | eiffel.png | 6 | 0.517-0.974 | 74.2% | 0 |
| working | cat.jpg | 6 | 0.542-0.924 | 82.9% | 0 |
| original | googlewiki.png | 4 | 1.034-1.249 | 89.9% | 1 |
| original | mario64.png | 4 | 0.899-1.101 | 86.0% | 0 |
| original | eiffel.png | 4 | 0.326-0.849 | 76.5% | 0 |
| original | cat.jpg | 4 | 0.804-0.864 | 84.7% | 0 |

The one qualifying original region is Googlewiki channel R, rectangle (640,704,64,64):

- Hard mask levels -119/+41, zero fraction 44.60%; 11,187,729 RGB SSE reduction in 540 actual packet bytes. This removes 0.6303% of the p10 base SSE.
- Native cell1/2-bit: 10,817,393 reduction in 652 bytes. Hard mask gives +3.42% gain at -17.18% packet bytes, a real local dominance point and 24.87% better gain per bit.
- Native model cell1/5-bit: 12,350,722 reduction in 1,980 bytes. Comparing only with this expensive point would greatly inflate the apparent advantage.
- Native cell2/3-bit: 4,225,094 reduction in 464 bytes, much less total gain and outside the comparable-gain interval.

This is one region in one text image. Mario's original masks are only 0.899-1.101 times native efficiency, Eiffel 0.326-0.849, Cat 0.804-0.864. Cat's selected original regions each remove only 0.127-0.300% of base SSE. Those numbers do not demonstrate multiple-class generalization. Flower was not added because the direction already failed on the four-image feasibility screen.

## Mask and shared-context cost

Individual complete packets range from 164 to 6,796 bytes. The qualifying 64x64 mask has 1,024 raw mask bytes plus 13 record bytes and eight group bytes: 1,045 raw bytes -> 540 XZ bytes. This includes explicit signed levels and frame overhead. The four original Googlewiki corrections share 2,604 bytes versus 3,436 bytes in four separate frames. Across all twelve groups, sharing uses 45,044 versus 51,792 bytes (-13.03%).

Shared native group costs for exactly the same rectangle collections:

| Frozen group | Hard masks | Native model | Native cell1/2-bit | Native cell2/3-bit |
| --- | ---: | ---: | ---: | ---: |
| googlewiki.png.quality.3 | 3,132 | 7,956 | 3,548 | 1,984 |
| googlewiki.png.quality.8 | 2,048 | 3,220 | 2,336 | 1,320 |
| googlewiki.png.base_p10 | 2,604 | 10,588 | 3,428 | 2,520 |
| mario64.png.quality.3 | 3,344 | 13,880 | 4,472 | 3,264 |
| mario64.png.quality.8 | 1,808 | 3,912 | 1,716 | 1,280 |
| mario64.png.base_p10 | 1,712 | 9,900 | 2,016 | 1,796 |
| eiffel.png.quality.3 | 4,744 | 16,336 | 3,996 | 2,452 |
| eiffel.png.quality.8 | 7,228 | 2,496 | 504 | 676 |
| eiffel.png.base_p10 | 3,092 | 3,916 | 1,512 | 1,684 |
| cat.jpg.quality.3 | 5,332 | 12,772 | 4,744 | 2,652 |
| cat.jpg.quality.8 | 7,072 | 900 | 2,828 | 2,360 |
| cat.jpg.base_p10 | 2,928 | 17,008 | 4,112 | 2,692 |

These tables are storage diagnostics. Extents overlap, and all masks were fitted to the same unchanged state; summed per-region gains are not the combined output gain of applying a group sequentially. No shared-group quality or completed-file win is inferred from smaller group bytes alone. Native groups also share one context and do not align each patch to a byte. Thus rejecting the mask is not based solely on charging it one entropy frame per tiny patch.

Signed levels improve expressiveness relative to Pass 18's native zero/positive 1-bit palette, but two nonzero magnitudes plus zero still discard multilevel residual structure. Median gain relative to the model is roughly 74-90%, and stronger low-bit native controls remove most apparent wins. Fine native grids can preserve edges at cell size 1; coarse-grid blur alone is not evidence that this mask is superior. Dense two-bit masks also encode fine residual-state boundaries that are not necessarily simple source-image edges. Shared context helps, but does not establish a consistent representation advantage. Metadata alone is not demonstrated as the failure cause.

## Runtime, validation and disposition

Proposal construction in this unoptimized NumPy probe took about 22-29 ms for the 250k working canvases and 23-913 ms for original images, one observation each. Cat's one full residual/gradient scan is already about 0.913 s. Evaluation times additionally include repeated full-canvas copies, RGB rendering/resizing, int64 SSE validation and native comparisons; they are offline instrumentation costs, not a proposed shipping path or full encoder overhead. No completed candidate exists, so encode/decode overhead, practical activation and matched-file RD remain unmeasured. No timing repetitions were run for this failed prototype.

All 40 saved correction records in 12 groups round-trip their masks, signed levels, geometry, packet hashes and recomputed actual RGB gains. Synthetic non-aligned 13x9 corrections verify zero transparency, hard boundaries, +/- levels, final clipping and repeated-byte determinism; six malformed/truncated/bounds cases reject. Empty groups also round-trip. Source/model integrity is checked before/after. This verifies correction-packet reconstruction, not an independently decodable complete hybrid image.

Stop here. Neither domain passes the predeclared multi-image gate; no Phase B/C, new complete container/decoder, candidate runtime screen, additional rate control, holdout or completed-file results report. Do not vary thresholds, region sizes or mask alphabets to rescue this pass.

The strongest next architectural direction is the Pass 17 base/detail lead with joint bitrate allocation and a multiscale residual representation. It already demonstrated completed-file gains across four image types, whereas this mask generally loses to ordinary low-bit grids and only isolates one text-region win. Efficient entropy coding/scale selection may retain that richer residual capacity without dense original-resolution cost on large photos. This is a recommendation, not an experiment launched here; the known DCT runtime/regression limits remain unresolved.
