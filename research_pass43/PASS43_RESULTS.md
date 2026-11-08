# PBC V4 Pass 43 results

## Objective and provenance

Diagnose V4-A’s low-rate loss and decoder cost, test one optional coarse stage, and optimize the dominant decoder operation without changing the v1 bitstream. Repository HEAD at start: `3723dc14de88c6c3dcbd98dd80a097e019a92a9f`, branch `research/experiments`. Frozen V3.1 source commit/model and source image hashes are in `PASS43_SUMMARY.json`. Work stayed under `research_pass43/`, except the V4 goals synchronization described below.

## Findings and complete-file comparisons

V3 gets its efficiency from both a compact generated-grid initialization and early coarse residual patches. V4-A spends much more rate on a per-pixel palette/index base, then spends native-resolution regional instructions. On Coast, V3 initialization is 1,783 B / MSE 156.15; V4-A init-only is 32,056 B / MSE 45.23. V3’s first residual reduces MSE to 122.97 at 2,502 B, and three residuals reach 99.53 at 3,142 B. The rate regression is not caused by mode-map bytes; transform payload and regional record count are larger contributors after initialization.

| Image / codec | Complete bytes | Bpp | RGB MSE | RGB PSNR |
|---|---:|---:|---:|---:|
| Googlewiki V3.1 Compression | 28,974 | 0.3507 | 1005.02 | 18.11 |
| Googlewiki V4-A D low | 73,148 | 0.8853 | 24.83 | 34.18 |
| Googlewiki V4-B D low, coarse prefix 3 | 423,996 | 5.1318 | 70.16 | 29.67 |
| Googlewiki V4-A D intermediate | 106,968 | 1.2947 | 15.62 | 36.19 |
| Googlewiki V4-B D intermediate, prefix 3 | 737,264 | 8.923 | 2.546 | 44.07 |
| Googlewiki V4-B D high, prefix 12 | 1,225,860 | 14.837 | 0.0384 | 62.29 |
| Cat V3.1 Compression | 7,036 | 0.0066 | 433.81 | 21.76 |
| Cat V4-A D low | 779,344 | 0.7313 | 167.37 | 25.89 |
| Cat V4-B D low, prefix 3 | 978,704 | 0.9184 | 143.80 | 26.55 |
| Cat V4-A D intermediate | 6,333,468 | 5.9433 | 10.11 | 38.08 |
| Cat V4-B D intermediate, prefix 13 | 6,376,384 | 5.984 | 6.750 | 39.84 |
| Coast V3.1 Compression | 6,461 | 0.0043 | 73.13 | 29.49 |
| Coast V4-A D low | 138,848 | 0.0925 | 18.69 | 35.41 |
| Coast V4-B D low, prefix 3 | 182,676 | 0.1217 | 12.23 | 37.26 |
| Coast V4-A D intermediate | 688,860 | 0.4589 | 7.162 | 39.58 |
| Coast V4-B D intermediate, prefix 3 | 735,768 | 0.4902 | 6.122 | 40.26 |

For V4-B with native modes disabled, complete V3-length coarse prefixes give V4-framed files of 29,180 B / MSE 1005.02 on Googlewiki, 6,448 B / 453.44 on Cat, and 4,436 B / 89.36 on Coast. These are pixel-identical to the equivalent V3 reconstructions. That recovers the V3 operating point, but it does not solve the rate cost of enabling V4 native detail. V4-B-D low is 25.6% larger than V4-A on Cat for 14.1% lower MSE and 31.6% larger on Coast for 34.6% lower MSE; these are not within 5% rate. Googlewiki B is larger and worse than V4-A D-low.

At intermediate rate, Cat V4-B is 0.68% larger than V4-A and has 33.2% lower MSE. Coast V4-B is 6.8% larger and has 14.5% lower MSE, outside the 5% rate-match band. Googlewiki V4-B is 6.9× the V4-A bytes for lower MSE. This is one rate-matched image, not the two-image minimum for Gate D.

All candidate modes, counts, raw field lengths, shared-XZ file lengths, MSE, PSNR, and hashes are in `outputs/v2_rd_comparison.json`; V4-A provenance is in Pass 42 `outputs/benchmark_raw.json`. Raw field sizes must not be summed as final rate.

## Decoder and runtime

V4-A Coast’s dominant decoder cost was Python procedural surfaces: 10.832 s / 73.3% of a 14.771 s instrumented decode across 32,182 surfaces. Upsampling took 1.647 s; coefficient parsing 0.926 s; inverse DCT 0.069 s; XZ 0.039 s. Googlewiki instead spends more relative time unpacking its small-image maps. Pass 42’s 746 MiB high-water working set is from an encode-only profile; Pass 43 observed about 76.8 MB decode RSS increase for Coast.

The semantics-preserving surface kernel is 3.35× faster on Coast (14.305 s to 4.271 s median, five warmed paired repeats) and neutral on Googlewiki (0.254 s to 0.258 s). Coast fresh-process REF/OPT is 15.13 s / 4.21 s with exact hash match. RSS delta changed negligibly. The v2 intermediate Coast file decodes in 2.731 s median with the optimized surface and 2.147 s fresh process; it has 28,211 procedural records, 2,117 transform regions, and exact protected SKIP. Its observed incremental RSS peak was 162.5 MB in the warmed process; this is not a paired memory comparison with the Pass 42 decoder because their loaded process state differs.

Three warmed interleaved end-to-end low-point Coast encodes (including P3 prefix generation for B) measured V4-A at 23.83, 23.03, 24.31 s and V4-B at 8.98, 8.86, 8.51 s. Medians are 23.83 s and 8.86 s. The produced files were 138,848 B and 182,676 B. The B path is faster here because it avoids V4-A’s expensive initialization candidate search; this is one image and one low point, not a general speed claim. Thread pools were limited to one.

## Visual and correctness review

Visuals: [Coast crop](visuals/coast_low_rate_comparison.png) and [Googlewiki text/edge crop](visuals/googlewiki_low_rate_comparison.png). Coast’s smooth sky remains continuous in the coarse base. The detail layer restores foreground/shoreline information, but visible 16×16 correction transitions remain near dark foreground edges in the low point; no taper or new boundary filter was added. Googlewiki shows preserved text/edge detail at much higher rate than V3.1 Compression. Six Pass 42 correctness groups remain the v1 baseline; Pass 43’s three focused groups pass. REF/OPT pixel hashes match, and protected SKIP is exact. The v2 decoder reconstructs the coarse base from the stored payload and does not load a neural model.

## Gates

- **A — Low-rate cause explained: Pass.** High-rate V4 initialization cost, absent coarse residual stage, transform syntax, and local rate allocation are measured separately enough to identify the primary causes.
- **B — Decoder bottleneck identified: Pass.** Procedural surface generation dominates large Coast; small Googlewiki has a different unpack/loop balance.
- **C — Decoder optimization: Pass.** Exact output preserved; 3.35× large-image improvement, neutral small-image result.
- **D — V4-A low-rate RD improvement at comparable rates: Not passed.** Cat/Coast native-detail points are 25–32% larger; Googlewiki B is worse. Coarse-only B is a V3-rate recovery point, not a comparable-rate improvement over V4-A’s much higher-quality low point.
- **E — V3.1 low-rate competitiveness: Not passed overall.** Coarse-only complete-file RD and pixels match V3.1 Compression on all three. End-to-end coarse-only runtime was not separately established, and enabling V4 detail drives rates sharply upward; the full gate includes performance.
- **F — Quality scaling preserved: Pass.** Googlewiki B falls from MSE 70.16 at low to 2.546 intermediate and 0.0384 high with original-resolution corrections.
- **G — Broader validation: Not attempted.** No held-out images, broad AVIF run, or separate 12+ MP matrix. Coast is a 12.0 MP source and was used for the large-image decoder and encoding checks.

## Goals file and Git state

`research/PBC_V4_GOALS.md` was synchronized with the provided Downloads copy, then the AVIF niche stretch-goal text from this Pass 43 request was appended because the supplied file’s actual contents did not include that requested goal. Existing goals were preserved. `AGENTS.md` was not rewritten.

HEAD and branch remained as specified; nothing was staged, committed, or pushed. `git status --short` still fails with “this operation must be run in a work tree”, even though read-only `rev-parse` recognizes the worktree. Read-only index/diff commands work and show the pre-existing modified `AGENTS.md` plus untracked research artifacts; no Git metadata repair was attempted. Details are in `PASS43_SUMMARY.json`.

## Pass 44 direction

Keep the coarse base as an optional low-rate path. The next useful experiment is a genuine global complete-file byte budget for native V4 refinements, with candidates ranked by exact post-quantization gain per measured serialized cost and shared-XZ total size. First target Coast and Cat; do not enable all local transforms solely on regional proxy costs. Keep the exact decoder kernel and test transition seams without taper until mode allocation reduces unnecessary active regions. Defer AVIF until V4 has a rate-matched gain over V3.1 and acceptable single-thread run time.
