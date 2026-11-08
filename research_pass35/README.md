# PBC V4 Research Pass 35

This pass pins and builds libaom 3.12.1 with libavif 1.3.0, establishes deterministic AVIF output, and profiles the actual all-intra paths. No PBC production source or model changed. The only source patch adds compile-time-guarded profiling counters.

- [Pinned sources and image hashes](SOURCE_MANIFEST.json)
- [Build and reproduction](BUILD_AND_REPRODUCTION.md)
- [Encoder source map](AV1_ENCODER_SOURCE_MAP.md)
- [Baseline measurements](AVIF_BASELINE_RESULTS.md)
- [Search profile](AV1_SEARCH_PROFILE.md)
- [PBC integration opportunities](PBC_AV1_INTEGRATION_OPPORTUNITIES.md)
- [Pass summary](PASS35_SUMMARY.json)

Builds, source checkouts, binaries, outputs, raw logs, and measurement JSON are under ignored paths. The reviewable source-only patch is `patches/libaom-profile-counters.patch`.

Gate A passed. Gate B was not run because profiling did not justify an algorithmic modification. The work remains uncommitted and unpushed.
