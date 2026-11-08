# Pass 39 — Adaptive Quantization

This pass maps and measures the existing libaom spatial allocation options. It does not change PBC production code or libaom behavior.

Pinned runtime: libaom 3.12.1 (`10aece4157eb79315da205f39e19bf6ab3ee30d0`) and libavif 1.3.0 (`1aadfad932c98c069a1204261b1856f81f3bc199`), reusing the Pass 35 reference build. Start with [Pass 39 results](PASS39_RESULTS.md), then see the [source map](AV1_AQ_SOURCE_MAP.md), [experiment design](EXPERIMENT_DESIGN.md), [rate-distortion report](AQ_RATE_DISTORTION_RESULTS.md), [spatial analysis](SPATIAL_ALLOCATION_ANALYSIS.md), and machine-readable [summary](PASS39_SUMMARY.json).

Raw encodes and diagnostic intermediates are under ignored `outputs/`. Scripts and reports are kept here.
