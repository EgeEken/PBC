# Feasibility results

## Summary

Pass 46 tested whether a compact source-resolution straight-stroke residual could restore details lost by the canonical PBC 3.1 Compression reconstruction. The test was intentionally local: it did not define or encode a complete V4 file. Candidate generation was run single-threaded on Coast, Googlewiki, and Mario64. Each image produced 64 Hough peaks from 5,000 retained edge/residual points; eight candidates were fitted.

| Image | Proposals (s) | Candidate fitting (s) | 8-stroke raw bytes | XZ packet bytes | 5% PBC Compression byte allowance | Group SSE gain | Gain / base SSE | Gate finding |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Coast | 0.466 | 1.151 | 104 | 164 | 323 | 23,409,471 | 0.889% | Fail: no single stroke meets 0.5%; visual effect weak; no matched native point |
| Googlewiki | 0.035 | 1.084 | 104 | 164 | 1,448 | 6,526,429 | 0.327% | Fail: no single stroke meets 0.5%; text not visibly restored; no matched native point |
| Mario64 | 0.044 | 1.032 | 104 | 164 | 704 | 20,341,467 | 1.645% | Fail: one single stroke meets numeric 0.5% rule, but effect is subtle and no matched native point |

The XZ packet expands the 104-byte raw group to 164 bytes. Meeting the provisional packet allowance does not prove that a complete stream meets a total-file limit. In particular, Coast's 164-byte group is already just over 5% of its 6,461-byte PBC file. The core local gate failed: there were not three independently useful visual wins across two content classes, and none of the 24 shortlisted strokes had a native output within the specified 80%-to-125% gain interval.

## Timing

The Hough proposal plus candidate fit took 1.62 seconds for Coast, 1.12 seconds for Googlewiki, and 1.08 seconds for Mario64 in these single-run probe timings. They include NumPy/Python proposal and fitting work and are not full-image encoder timings or release speed estimates. Native-patch construction sample timings are preserved in `outputs/feasibility_results.json`; there are 16 native comparisons per image. This probe therefore adds appreciable encoder work for a weak local effect. No stroke rasterizer was integrated into a decoder, so no decoder speed or compatibility statement follows from this phase.

## Visual and metric assessment

Eight strokes reduce local RGB MSE on each base, but the absolute structural improvement was not persuasive in the source-aligned crops. SSIMULACRA2 rose by less than one point per image. The Coast fishing rod remained soft; Googlewiki lettering was not convincingly sharpened; Mario64 fence edges changed subtly. No obvious halo or tear was identified in the inspected crops. Small metric gains are not treated as a perceptual win.

## Protocol note

The frozen design text and executed shortlist ranking differ: the text says top eight by clipped SSE reduction, whereas the runner uses actual gain per compressed packet byte. The implementation did recompute actual gains, serialized packet cost, and sequential group gains. This is recorded as a protocol deviation; it limits the strength of the feasibility conclusion but does not justify selecting or integrating the mode. Any repeat should freeze one ranking rule before rendering candidates.
