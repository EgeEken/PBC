# PBC–AV1 Integration Opportunities

These are proposals only. Speed 8 spends most internal time in non-RD partition application and hybrid intra search; speed 6 spends most in RD partition search.

## 1. RD intra candidate ordering

- Existing mechanism: `intra_mode_search.c:av1_rd_pick_intra_sby_mode` applies HOG masks, mode masks, angle checks and model-RD/top-candidate pruning.
- Idea: test whether cheap PBC-like edge/texture features improve ordering or thresholds for remaining modes while leaving AV1 RD cost as final decision.
- Integration: just before `av1_pick_uniform_tx_size_type_yrd`.
- Compatibility/cost: encoder-only and standard AV1; feature cost must stay below the expensive evaluations it avoids.
- Risk/evidence: duplicate current signals or remove a useful mode. Require exact or matched-rate RD, full AVIF bytes, speed and holdout results with instrumentation off.

## 2. Speed-6 RD partition work reuse

- Existing mechanism: `partition_search.c:av1_rd_pick_partition` recurses through partition choices using RD bounds and pruning.
- Idea: identify avoidable repeated candidate/context work and evaluate a bounded reuse optimization that preserves choices.
- Integration: around recursive candidate calls and `av1_prune_partitions`.
- Compatibility/cost: encoder-only if decisions stay identical; caching adds state and memory.
- Risk/evidence: profile timers are intrusive at speed 6. Use counters to locate duplicate evaluations, then paired uninstrumented timings and byte comparisons.

## 3. Local quantizer allocation

- Existing mechanism: AV1 adaptive quantization and segmentation adjust local q/RD multipliers; transform quantization already removes low-value coefficients.
- Idea: test whether PBC's local correction-value history improves an existing AQ decision.
- Integration: AQ/segmentation or delta-q selection, without adding a decoder-side residual layer.
- Compatibility/cost: ordinary AV1 syntax; cheap if based on existing block features.
- Risk/evidence: duplicate variance/complexity AQ or trade rate for distortion. Require matched-rate RD curves and encode-time limits.

## PBC-specific assessment

LibAOM already uses variance/gradient features, HOG pruning, model-RD, mode masks, adaptive quantization, transform evaluation and coefficient quantization. PBC's potential contribution is a signal that predicts where those decisions are wrong, not another copy of them. Sparse-detail results suggest testing a saliency prior, not a sparse decoder representation.

## Small future research list

- JPEG XL offers VarDCT and Modular paths, encoder tool-selection guidance and adaptive quantization; study its tool gating as an analogy, not evidence that AV1 lacks those tools. [JPEG XL white paper](https://ds.jpeg.org/whitepapers/jpeg-xl-whitepaper.pdf).
- WebP documents block prediction, adaptive quantization, transforms and entropy coding, all established coding tools that AV1 extends. [Google WebP compression documentation](https://developers.google.com/speed/webp/docs/compression).
- PICO and PULSE are learned-codec research directions, not drop-in AVIF heuristics. PICO is a learned perceptual codec; PULSE explicitly studies a low-complexity neural receiver. Their decoder-side model requirements do not fit standard-decoder-compatible Pass 35 changes. [PICO research repository](https://github.com/apple-aiml-research/ml-pico), [PULSE paper](https://arxiv.org/abs/2609.18602).
