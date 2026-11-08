# Pass 41 — Luma–chroma rate-distortion allocation

Investigate whether AV1's frame-level U/V quantizer offsets explain meaningful RGB reconstruction error, then test the existing libaom chroma delta-Q control. Do not carry forward Pass 40's spatial correction.

Pinned source, build and input provenance are inherited from Pass 35. Generated streams, planes and timing rows stay under ignored `outputs/`.

See the [source map](CHROMA_CODING_SOURCE_MAP.md), [error analysis](CHROMA_ERROR_ANALYSIS.md), [allocation results](CHROMA_ALLOCATION_RESULTS.md), and [final results](PASS41_RESULTS.md). Machine-readable measurements and gate decisions are in [PASS41_SUMMARY.json](PASS41_SUMMARY.json). The existing control is active but has no consistent rate-distortion gain; no encoder patch or held-out run was justified.
