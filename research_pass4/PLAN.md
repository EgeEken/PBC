# PBC Pass 4 frozen research plan

Production and previous research are read-only. No commit, push, deployment, restart, uploads, database mutations, training or arbitrary remote execution. Use the existing Space only for read-only inspection and a bounded existing operation if it supports safe subset selection and preserves stored results.

Order: prior integrity; combined-runtime exactness and screen; current HF API audit; nearest-rate diagnostics and bounded fresh ordinary controls; matched-rate evaluation; allocator profiling; at most two exact cost optimizations; qualified full timings; reports and integration proposal.

Runtime: current, grid+skip, rectangle reuse, all three. Six-image screen then all 25 images/four presets/three interleaved warmed repeats if exact. Debug proposal/RNG verification precedes timing; RGB/RGBA and both mutation paths checked. No CPU-intensive runs overlap.

Allocator stays exact Pass 3 weight 0.4, Quality 500k/35, model/seed/geometry policy unchanged. Ordinary controls use fresh encodes. Per-image target is frozen allocator's final bpp, not quality. Use at most six new controls per image: patch budgets 10 and 25 initially, then bounded interpolation/bisection within budgets 3–50; if rates fail to bracket or budget response is irregular, use at most two legitimate learned-q controls (0.7, 0.9) at budget35. Selection uses bitrate only; do not optimize control MSE or retune allocator. Stop once a measured point is within 3% or after six fresh points. Preserve nonmonotonicity and remaining gaps. Prior ordinary points remain available. Main/holdout conclusions and direct/interpolated results are separate.

Cost profiling and candidate choice occur after matched-rate analysis. Prefer shared residual projection and unchanged before-SSE reuse, only if profiles support them. At most two mechanisms; exact full outputs and decisions before speed promotion. Keep the same underlying runtime for all cost comparisons. Optional combined fast-runtime allocator comparison only after a convincing exact cost result.

All timed rows independently decode and score original-resolution RGB, count complete post-LZMA bytes, and retain raw bytes, config, patches, source/model/input provenance. Single numerical thread/CPU, complete per-config warmups, rotated execution, three repeats for primary runtime claims. No prefixes counted as fresh encodes.
