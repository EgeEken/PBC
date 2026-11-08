# Pass 37 results

## Objective and provenance

Test whether stable ordering of the first 13 AV1 luma intra base modes by their existing syntax cost can improve speed-6 AVIF coding. REF is the clean pinned libaom build; CTRL is the patched source with the compile-time switch off; EXP enables the ordering. No PBC production or decoder source was changed.

- Branch/HEAD: research/experiments at 3723dc14de88c6c3dcbd98dd80a097e019a92a9f, unchanged from the expected start.
- Sources: libaom 3.12.1, commit 10aece4157eb79315da205f39e19bf6ab3ee30d0; libavif 1.3.0, commit 1aadfad932c98c069a1204261b1856f81f3bc199.
- Toolchain: Windows 11 x64, Strawberry GCC/G++ 13.2.0; Release -O3 -DNDEBUG; one AV1 thread; static local libaom.
- Patch SHA-256: E2237FC1DE60738E5ED47D06AA87099C6A89570EE8BB719CB83BB96EA87214F4.

## Mechanism and implementation

The pinned av1_rd_pick_intra_sby_mode() visits base modes in a fixed array order, then directional angle variants. It applies configuration and speed masks, HOG and odd-angle pruning, computes intra_model_rd(), and calls prune_intra_y_mode(). That helper updates a per-block top-model-RD list and best-model bound; modes that fail those thresholds skip av1_pick_uniform_tx_size_type_yrd(). The transform search also receives the current best actual RD bound. Candidate order can therefore change full transform/RD work and selected coding decisions.

The rule stably sorts the 13 base modes by ascending bmode_costs[mode], a syntax cost already available from the above/left mode context. Equal costs retain original order. Directional angle deltas keep their existing order. No new feature extraction, state reuse, pruning threshold, transform, syntax, or decoder change was added. The code is compile-time gated by PBC37_EXPERIMENT.

## CTRL parity and EXP decoder checks

REF and CTRL were byte-identical on all nine frozen speed-6 and speed-8 operating points. EXP output was deterministic across five timed repetitions for every point. All development and synthetic files decoded with the unmodified Pass 35 avifdec. FFmpeg 6.1.1 also decoded 16 EXP files and reported the expected dimensions. Seven synthetic cases (constant, gradient, horizontal edge, vertical edge, text-like pattern, odd dimensions, and a 512x512 Cat crop) each encoded twice to identical AVIF hashes and decoded at the original dimensions. No sanitizer run was performed.

## Search behavior

PBC_PROFILE aggregate counters were collected in a separate instrumented build; their timings are excluded. Every RD intra block visited 61 loop entries in CTRL and EXP, so the patch did not drop candidate indices. The loop still visits the same complete candidate set; downstream dynamic pruning and actual RD bounds determine which candidates reach transform search. RD intra block counts can change because altered choices affect later partition decisions.

| Image / qcolor / speed | CTRL -> EXP RD intra blocks | CTRL -> EXP transform/RD entries | Entry change |
|---|---:|---:|---:|
| Googlewiki q5 s6 | 21,967 -> 22,246 | 64,649 -> 64,595 | -0.08% |
| Googlewiki q20 s6 | 23,186 -> 23,313 | 69,009 -> 67,381 | -2.36% |
| Mario64 q50 s6 | 10,127 -> 10,155 | 36,098 -> 35,435 | -1.84% |
| Flower q80 s6 | 41,107 -> 41,092 | 115,870 -> 114,888 | -0.85% |
| Cat q35 s6 | 59,064 -> 59,276 | 152,001 -> 139,480 | -8.24% |
| Cat q50 s6 | 74,315 -> 74,641 | 185,275 -> 171,965 | -7.18% |
| Coast q80 s6 | 38,196 -> 38,195 | 103,499 -> 101,855 | -1.59% |
| Googlewiki q5 s8 | 4,024 -> 4,024 | 11,643 -> 11,316 | -2.81% |
| Cat q35 s8 | 28,059 -> 28,059 | 89,260 -> 75,906 | -14.96% |

Across speed 6, transform/RD entries fell from 726,401 to 695,599 (-4.24%). Per-point reductions ranged from 0.08% to 8.24%. The patch leaves eligibility tests unchanged and the instrumentation verifies all loop entries are still visited. The counts are consistent with the modeled-RD and actual-best-RD bounds being order-sensitive. The aggregate counters do not identify which bound rejected each specific candidate. Speed 8 also entered this RD intra function: 4,024 Googlewiki and 28,059 Cat blocks in CTRL. The ordering was therefore active in these controls.

## Complete encode and rate-distortion results

Five timed repetitions per REF/EXP configuration were interleaved, alternating which binary ran first. Each binary had one warmup per point. Timing is full avifenc process wall time, including process startup, libavif frontend, encoding, and file output. Memory polling was excluded from timed runs. Separate process-RSS probes sampled every 2 ms; CPU affinity was not fixed. Raw rows and all timing samples are in outputs/benchmark_final/benchmark.json.

| Speed-6 point | Bytes REF -> EXP | RGB MSE REF -> EXP | Median seconds REF -> EXP (time reduction) | Transform/RD entry change |
|---|---:|---:|---:|---:|
| Googlewiki q5 | 20,613 -> 20,361 (-1.22%) | 619.155 -> 620.480 (+0.21%) | 0.2265 -> 0.2300 (-1.54%) | -0.08% |
| Googlewiki q20 | 44,550 -> 44,653 (+0.23%) | 319.837 -> 320.421 (+0.18%) | 0.2711 -> 0.2686 (+0.95%) | -2.36% |
| Mario64 q50 | 39,332 -> 39,082 (-0.64%) | 57.886 -> 57.981 (+0.16%) | 0.2461 -> 0.2440 (+0.84%) | -1.84% |
| Flower q80 | 170,486 -> 170,244 (-0.14%) | 3.184 -> 3.186 (+0.06%) | 1.1999 -> 1.1957 (+0.35%) | -0.85% |
| Cat q35 | 262,385 -> 261,979 (-0.15%) | 46.502 -> 46.621 (+0.26%) | 1.7067 -> 1.7104 (-0.21%) | -8.24% |
| Cat q50 | 528,238 -> 528,312 (+0.01%) | 21.686 -> 21.665 (-0.09%) | 2.0736 -> 2.0735 (+0.00%) | -7.18% |
| Coast q80 | 159,713 -> 159,569 (-0.09%) | 3.299 -> 3.298 (-0.04%) | 1.3795 -> 1.3858 (-0.46%) | -1.59% |

The unweighted mean point-median time change was -0.01% reduction and the median was +0.00%; four of seven points were faster. Per-run variability exceeded these changes. The 4.24% reduction in transform/RD entries did not become a measurable complete-encode speedup; per-block sorting and remaining encoder work offset the saved calls. Mean byte change was -0.29%; mean RGB MSE change was +0.11%. Rate and quality changes were mixed and small, not a demonstrated frontier gain.

Pass 37 computes RGB MSE after promoting pixel values to float64 before subtraction and squaring. The Pass 35 baseline helper used int16 arrays for the squared difference; that can overflow and distort its absolute MSE values. Therefore, compare Pass 37 REF and EXP MSE within this pass, and do not compare these absolute values numerically with the historical Pass 35 MSE table.

### Speed-8 controls

| Point | Bytes REF -> EXP | RGB MSE REF -> EXP | Median seconds REF -> EXP (time reduction) |
|---|---:|---:|---:|
| Googlewiki q5 | 15,985 -> 15,919 (-0.41%) | 806.513 -> 804.123 (-0.30%) | 0.0689 -> 0.0662 (+3.86%) |
| Cat q35 | 251,357 -> 251,332 (-0.01%) | 51.484 -> 51.500 (+0.03%) | 0.5970 -> 0.5960 (+0.17%) |

Googlewiki speed 8 improved on all three measurements at this single point, but its 3.86% time change did not generalize to Cat. This does not pass the development gate.

Sampled peak process RSS was essentially unchanged (about 27 MB on Googlewiki, 34 MB on Mario64, and 128-198 MB on photographs); raw per-point values are in benchmark.json. The separate 2 ms sampler may miss short-lived peaks.

## Gate decisions

| Gate | Result | Reason |
|---|---|---|
| A — Ordering-sensitive mechanism | Pass | Active path, dynamic pruning/bounds, and pre-existing syntax-cost signal verified. |
| B — Correct experimental implementation | Pass | REF/CTRL parity; EXP decoded with unchanged avifdec and FFmpeg; deterministic outputs and synthetic checks passed. |
| C — Development RD-speed gain | Fail | Speed-6 complete-encode changes were at timing-noise level; only 4/7 points were faster, with mixed rate/distortion shifts. |
| D — Held-out generalization | Not run | Conditional validation was not justified after Gate C failed. Held-out images were not evaluated. |

## Recommendation

Reject this ordering rule and do not tune it further. The transform-call reduction is measurable, but whole-encode speed did not improve. A later AV1 pass should inspect search_tx_type() in av1/encoder/tx_search.c. That path already orders candidates through txk_map and has a rate-only best-RD skip plus adaptive early termination. First map its existing estimated-RD ranking so a new experiment does not duplicate it.

## Worktree

Pass 35 and Pass 36 remain preserved. No files were staged, committed, or pushed. The only new untracked directory is research_pass37/; generated builds and benchmark outputs are ignored.
