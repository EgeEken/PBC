# Pruning diagnostic design

## Selected condition

Measure the `adaptive_txb_search_level` break in `av1/encoder/tx_search.c::search_tx_type()`. Do not disable the safe coefficient-rate lower bound, transform-set legality, other search features, or the zero-EOB early stop.

## Method and bound

The DIAG-only compile-time patch recognizes at most 128 deterministic sampled events per avifenc process. Sampling uses a stable hash of frame macroblock row/column, transform size, block row/column, plane, and intra prediction mode; a fixed modulus of 16 selects candidates, with a hard cap. An event is sampled only when the original adaptive condition actually fires, the block is luma intra, and more than one transform is legally permitted.

At the original cutoff, DIAG saves the incumbent RD/type as the baseline and continues the remaining legal transforms in the same search call. Later candidates cannot replace the official winner. Their local RD is compared with the saved incumbent; exact RD improvements are counted. The coefficient-rate lower bound remains active because it is mathematically safe relative to the incumbent. If the zero-EOB stop interrupts the shadow, the event is marked incomplete. Candidate counts distinguish post-cutoff transform attempts from candidates reaching full RD calculation.

Expected work is bounded by 128 events per image and no more than the remaining legal candidates for those blocks. Encoding time from the diagnostic build is exploratory only.

## Inputs and frozen settings

- Googlewiki, qcolor 5, speed 6.
- Mario64, qcolor 50, speed 6.
- Cat, qcolor 35, speed 6.
- All use the Pass 37 one-thread AVIF settings: libaom via libavif, 8-bit, 4:2:0, full range, CICP 1/13/1, qalpha 0, metadata omitted, automatic tiling.
- Speed 8 is a path-activity check on Googlewiki q5 and Cat q35; no quality benchmark is planned unless speed 6 establishes missed opportunities.

Each image is a separate avifenc process, resetting the diagnostic cap and counters. Inputs are from the Pass 35 frozen manifest. Event output is bounded to at most 128 rows per process and contains local RD/candidate counts only.

## Correctness safeguards

REF uses the clean pinned libaom build. DIAG uses only the compile-time shadow patch. Verify identical AVIF bytes for every diagnostic point and decode outputs with the unchanged avifdec. Any byte difference invalidates the state-preservation claim and stops interpretation. Diagnostic wall time is excluded from speed claims. This pass does not benchmark an optimization, modify encoder defaults, or use held-out images.

