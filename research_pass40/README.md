# Pass 40 — Causal spatial bit allocation

This pass isolates libaom's active `deltaq-mode=6` Variance Boost from the other `tune=iq` changes. It reuses the pinned Pass 35 release encoder and Pass 39 byte-parity-checked qindex diagnostic build. All four ablation settings explicitly use 64×64 superblocks so the delta-Q toggle does not also change superblock granularity.

See [source analysis](DELTAQ_SOURCE_ANALYSIS.md), [component ablation](COMPONENT_ABLATION_RESULTS.md), [spatial analysis](SPATIAL_RD_ANALYSIS.md), and [results](PASS40_RESULTS.md). Raw rows and qindex CSVs live under ignored `outputs/`.
