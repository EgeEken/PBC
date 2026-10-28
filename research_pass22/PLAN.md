# Pass 22 bounded decoder plan

Start: a0824ff1a319f6cadcfbc16d146f666ec9f91832. Saved P21S and ordinary HQ streams only; no image encoder or reference regeneration. Production, frozen experiments, representation, model and release staging remain unchanged.

First recover validated coefficients for Googlewiki q128, Eiffel q32 and Cat q128. Count per-component and spatial occupancy, clipped source-pixel coverage, and a 4x4 spatial occupancy map. If spatial empty blocks are substantial (at least 20% on two cases), implement one gather/SciPy IDCT/scatter method; otherwise skip it. Report per-component savings independently of spatial coverage. Zero-block fraction is not a full-decoder speed prediction.

Then inspect actual views/copies and test one focused reconstruction method: reuse block output and fuse color/add/round/clip into uint8 output if its float32 arithmetic can reproduce frozen pixels. No fastmath. Stop if exact pixels or meaningful full-call improvement fail; no series of alternate kernels. Retain the reference decoder. Reuse unchanged P21S framing and bounded coefficient reader.

Initial full calls: Googlewiki/Eiffel/Cat, pinned core12, one numerical thread, warmed kernels and SciPy, three rotated repetitions for promising methods. Include unchanged ordinary HQ saved streams. Timers include all parsing/base/recovery/transform/reconstruction; exclude I/O, hashes and scoring. Validate all six saved files, odd dimensions and sparse/dense/extreme fixtures. Measure per-stage timings and actual buffer sharing; isolate memory comparisons in fresh processes with identical warmup. Record RSS baseline and process peaks separately from working-buffer estimates. No broad benchmark or automatic followup.
