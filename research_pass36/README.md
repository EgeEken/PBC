# PBC V4 Research Pass 36

Pass 36 used the pinned Pass 35 libaom/libavif checkout to test optimized native sampling and inspect exact reuse opportunities. No encoder modification was made.

Gate A was not passed: one speed-6 Googlewiki configuration was sampled; the other four requested configurations and an independent second capture were not collected. The WPR capture overhead estimate was 13.6%, so its timing is attribution evidence only.

- [Profiling method](PROFILING_METHOD.md)
- [RD search profile](RD_SEARCH_PROFILE.md)
- [Computation reuse candidates](COMPUTATION_REUSE_CANDIDATES.md)
- [Pass results](PASS36_RESULTS.md)
- [Machine-readable summary](PASS36_SUMMARY.json)

`outputs/` is local and ignored. Pass 35 remains untouched and untracked.
