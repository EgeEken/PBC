# Pass 39 results

## Scope and setup

Pass 39 tested existing libaom AV1 options through pinned libavif, with no PBC production or upstream encoder changes. Runtime was libaom 3.12.1 (`10aece4157eb79315da205f39e19bf6ab3ee30d0`) and libavif 1.3.0 (`1aadfad932c98c069a1204261b1856f81f3bc199`). The actual Pass 35 binaries were reused for the primary encodes.

The main matrix covered five `hpt_data` development images at two frozen quality settings and REF/A/B/C/D: 50 points, with one warmup and three timed fresh-process encodes per point. A bounded rate-only IQ calibration added 12 C points. A speed-8 subset added eight points for Googlewiki q20 and Cat q50 (REF/A/B/C). The benchmark ledger contains 285 encode rows, including warmups and memory probes; the five parity-checked qmap encodes are logged separately. Codec settings, source hashes, file hashes, RGB metrics, SSIMULACRA 2, encode/decode timings and decoder checks are in `outputs/benchmark/results.json`.

All runs used 8-bit YUV420, full range, one job, qalpha 0, automatic tiling, CICP 1/13/1 and ignored metadata. bpp is measured from complete AVIF file bytes. Source RGB arrays were promoted to float64 before subtraction and squaring; reconstructions were produced with the pinned `avifdec`. SSIMULACRA 2 was computed with libjxl 0.11.2 on source and decoded 8-bit RGB PNGs. Repeated output hashes were stable. This is a local Windows timing result, not an HF runtime measurement.

## Findings

1. `aq-mode=1` does not affect the tested still-image path. A and B were byte-identical across all ten same-q main points, with identical decoded RGB. Root qindex maps were flat at 164, all segment 0. Source tracing found `DISALLOW_RECODE` selects `encode_without_recode`; variance-AQ frame setup lives only in `encode_with_recode_loop`. D likewise matched C at every initial matrix point, because the variance-AQ part was inactive; IQ's delta-Q path remained active.
2. `tune=iq` changes actual allocation. C's qindices varied from 108 to 164 across the 2,128 64×64 superblocks at Cat q35. It also changes quant matrices, RD weighting, CDEF and chroma delta-Q, so its full effect is not an isolated AQ effect.
3. Near-rate results are mixed. Across the eight A/C matched pairs, C has lower RGB MSE on one, higher on seven; SSIMULACRA 2 improves on seven and worsens on one. Googlewiki is mixed across its two rates; at q20 C has a rate-matched MSE regression (+12.71%) and a lower SSIMULACRA 2 score (−2.922). Full table: `AQ_RATE_DISTORTION_RESULTS.md`.
4. At Cat q35, C lowers smooth-region RGB MSE by 2.99%, while raising edge MSE by 13.27% and texture MSE by 11.57%, at a 0.33% rate increase. B's output equals A.
5. The Coast q90 memory probe measured 202.9–207.9 MB peak sampled working set. This is one encode per configuration. The speed-8 subset shows B still equal to A and C varies by content; no result establishes a generally faster setting.

## Decision gates

| Gate | Result | Evidence |
|---|---|---|
| A — source and option path | Passed, with path qualification | The CLI accepts both options; source tracing and diagnostic logs show variance AQ setup is skipped by the tested no-recode path. IQ's delta-Q is active. |
| B — completed file RD, timing and decode | Passed | Main matrix, calibration, speed-8 subset, stable hashes and successful decoding are recorded. SSIMULACRA 2 is available and measured. |
| C — qindex and spatial evidence | Passed | Separate instrumented build preserves exact AVIF bytes for REF/A/B/C/D at Cat q35; root-only qindex and fixed masks are reported. |
| D — meaningful near-rate improvement | Failed | IQ does not produce a broad RGB-MSE gain; gains in SSIMULACRA 2 are offset by substantial MSE regressions, with edge/texture regressions. Variance AQ is inactive in the selected path. |
| E — held-out validation | Not run | Development evidence did not justify selecting a candidate to validate. |

No production candidate is proposed. No files outside `research_pass39/` were changed, no commit or push was made, and prior untracked Pass 35–38 research folders were preserved.

## Artifacts

- `AV1_AQ_SOURCE_MAP.md` — pinned source mechanisms and active-path finding.
- `EXPERIMENT_DESIGN.md` — frozen matrix, calibration and decision rules.
- `AQ_RATE_DISTORTION_RESULTS.md` — completed-file metrics and near-rate comparisons.
- `SPATIAL_ALLOCATION_ANALYSIS.md` — qindex distributions and fixed-region results.
- `visuals/rate_distortion.png` — measured RD points.
- `visuals/spatial_allocation.png` — source/reconstruction, qindex and error-difference views.
- `PASS39_SUMMARY.json` — machine-readable final results and gate status.
- `outputs/benchmark/results.json` and `outputs/benchmark/summary.json` — ignored raw ledger and aggregates.
