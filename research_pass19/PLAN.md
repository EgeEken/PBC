# Pass 19 frozen feasibility plan

Start commit c545c56686416f59b1c317658034ad93fdb78a7e; unchanged pbc31_feature_histogram.

One representation: per-channel full-pixel ternary mask with zero, one explicitly stored signed negative level and one positive level. Four Lloyd updates, zero fixed, half-step ties toward zero. Two-bit packed mask; channel, rectangle, two int16 levels stored. One shared XZ context for a collection. No interpolated mask or learned fitting.

Working domain: Googlewiki, Mario64, Eiffel, Cat; Quality steps 3 and 8 from Pass 18 captured arrays. Compare the ordinary selected rectangle and two deterministic new rectangles. Original domain: the same four images, original RGB residual over the frozen Quality p10 base from Pass 17. Four deterministic new rectangles per image. No new image encodes for this phase.

Proposals: one residual/error-and-edge analysis; 64x64 tiles, rank signed residual SSE weighted by residual gradient concentration. For each of the two strongest tile/channel anchors, consider its tile and a centered 128x128 rectangle, clipping to image boundaries. Bounded four-rectangle shortlist, no threshold sweep. Working states use the two strongest candidates in addition to their original selected box.

Native comparators per rectangle: ordinary learned action, fixed cell1/max2bits, fixed cell2/max3bits. Fixed controls prevent attributing an inferior single model decision to the entire native primitive. Compare actual reconstruction, original RGB gain and complete separately entropy-coded region packets. Packet costs are not marginal final-file LZMA costs. Collections share one context and include every geometry/level/count/frame byte.

Regional gate, separately per domain: at least three regions spanning at least two images must retain >=80% of a comparable native grid's RGB gain and improve RGB gain per actual packet byte by >=20%. Compare against the best native efficiency among points within a factor 1.25 of the hard mask's gain, not a deliberately weak comparator. For working states, a new tiny rectangle must additionally retain >=80% of the ordinary chosen patch's RGB gain; an ordinary chosen rectangle already matches geometry. For original resolution, a region must remove >=0.5% of the complete p10 base RGB SSE to avoid crediting tiny irrelevant patches. No extrapolation.

Only a passing domain may proceed to a completed-file prototype. Choose one architecture from these results; do not tune another representation after failure. Full-file scope at most four main images plus Flower if a photographic activation check is needed; two fixed limits on correction count are allowed, not a rate/threshold sweep. No new full-set benchmark or release/HF changes.
