# PBC V4 Research Pass 37

Tested stable sorting of AV1 luma intra base modes by their existing syntax cost. The rule reduced transform/RD search entries by 4.24% across seven speed-6 points, but did not improve complete encode time beyond measurement noise. Gate C failed, so held-out images were not used.

- [Intra search mechanism](INTRA_SEARCH_MECHANISM.md)
- [Frozen experiment design](EXPERIMENT_DESIGN.md)
- [Results and gate decisions](PASS37_RESULTS.md)
- [Machine-readable summary](PASS37_SUMMARY.json)
- Source patch: [intra-mode-syntax-cost-order.patch](patches/intra-mode-syntax-cost-order.patch)

The authoritative timing set is in ignored outputs/benchmark_final/. outputs/benchmark/ is an earlier exploratory run that sampled memory during timing and is excluded from the timing conclusion. Pass 35 and Pass 36 remain untouched.
