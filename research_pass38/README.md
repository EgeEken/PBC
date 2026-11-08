# PBC V4 Research Pass 38

Bounded same-state shadow test of libaom's adaptive transform-type RD early exit. Source analysis selected the adaptive cutoff, not the coefficient-rate lower-bound check. The diagnostic continued remaining legal transforms at sampled cutoffs and preserved the original winner, producing REF-identical files.

- [Transform search mechanism](TRANSFORM_SEARCH_MECHANISM.md)
- [Diagnostic design](PRUNING_DIAGNOSTIC_DESIGN.md)
- [Pruning results](PRUNING_DIAGNOSTIC_RESULTS.md)
- [Pass results and gates](PASS38_RESULTS.md)
- [Machine-readable summary](PASS38_SUMMARY.json)

Pinned source: libaom 3.12.1, 10aece4157eb79315da205f39e19bf6ab3ee30d0; libavif 1.3.0, 1aadfad932c98c069a1204261b1856f81f3bc199. No production PBC files or decoder code changed. No commit or push.
