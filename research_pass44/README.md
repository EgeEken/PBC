# PBC V4 Pass 44

This pass establishes a completed-file RD baseline against the frozen released PBC V3.1 and pinned AVIF, then tests sparse subsets of existing V4 regional records under strict V3.1-sized budgets.

## Outcome

- V4-B can produce valid files under a 105% V3.1 Compression/Quality byte cap using coarse-only or sparse regional corrections.
- The sparse allocator does not produce a meaningful matched-rate improvement over V3.1 on multiple development images.
- Existing full V4-B encoding and decoding are much slower than V3.1. Selective output encode time was not measured as a complete source-to-file operation.
- AVIF rate and distortion were measured with the pinned command-line build. Its startup-inclusive timing path is not directly comparable to PBC, so Gate A remains incomplete.

## Reports

- `BENCHMARK_CONTRACT.md` — codec identities, source hashes, rate rules and timing limitations.
- `COMPETITIVE_BASELINE.md` — frozen Phase A baseline and low-rate diagnosis.
- `REFINEMENT_RATE_ANALYSIS.md` — candidate ranking and actual completed-file cost.
- `STRICT_BUDGET_RESULTS.md` — coarse-only, frozen-base, joint allocation, timings, correctness and visuals.
- `PASS44_RESULTS.md` — final result using the requested structure.
- `PASS44_SUMMARY.json` — machine-readable rows, hashes, timing samples, decisions and caveats.

All generated streams, AVIF/JPEG/PNG files and raw benchmark outputs are under ignored `outputs/`. Representative crop images are under `visuals/`. Scripts are under `scripts/`.

No production V3.1 source, GUI, HF Space or prior research directory was modified. No Git operation staged, committed, pushed, reset, cleaned or repaired metadata.
