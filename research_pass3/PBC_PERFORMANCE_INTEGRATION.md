# Performance integration and HF validation plan

All measurements are local. Production files, defaults, model, UI and running HF Space are unchanged. The exact proposed integration is copied to integration_candidate/pbc3_stream.py and supplied as proposed_grid_io.patch; it is not applied. `git apply --check research_pass3/proposed_grid_io.patch` passes. Full confirmation exercises the proposed module's dispatch path, not just the earlier standalone NumPy prototype.

## Scope and correctness

The current-root baseline, grid only, existing internal-MSE skip only and both combined were screened on six predetermined images/all presets, then confirmed on all 20 main/five holdout images/all four presets, three interleaved warmed repeats. The combined result is measured directly; individual percentages are never added as evidence.

The bit-layout gate covers 1632 cases: widths 0–16, all eight initial offsets, empty/short/long and strided integer inputs. Signed int64 and uint64 extremes preserve low-bit masking. Independent image checks cover 1×1 through 97×131, RGB, opaque and varying-alpha RGBA, original/4096-pixel working limits, file equality, full-channel pixels and internal metric availability. Manual generated-palette fixtures exercise actual precision 1 and 9. Contiguous and strided byte-buffer cases pass. The proposed scalar/NumPy dispatch copy adds 272 bit-layout checks and eight end-to-end mode/metric combinations.

Truncated STORE/LZMA streams, bad version/entropy tags, explicit-palette payload flags, missing declared patches, zero dimensions, zero cell size, invalid RGB channel and out-of-range geometry preserve the baseline's tested accept/reject and exception classes. Valid zero-patch streams remain valid. Full malformed observations are in performance_correctness.json. The helper rejects unsupported negative counts/widths and widths above uint16, and checks required payload bytes before allocating a positive-width output. Noncanonical private helper inputs now have explicit type/range errors; error messages are not promised identical. Valid production patch widths are at most nine.

This is not a decoder security rewrite. Existing header canvas allocation, aggregate dimensions/channel/patch counts, coordinate validation and some NumPy/PIL exception behavior remain independent limitations. Arbitrarily huge headers are not safely stress-tested by attempting huge allocations. Raw zero-bit grids can still imply allocation without consuming payload bits. These limits must not be described as resolved by faster grid I/O.

## Complete-codec results

Times are means of per-image/preset medians. Independent original-size decoded RGB MSE/bpp remain exactly unchanged for every current variant; all emitted bytes, pixels, counts and patch payloads match current baseline and the frozen current streams. Detailed per-image wins/regressions remain in performance_full_paired.json.

| Set | Variant | Cases | MSE | bpp | Encode base s | Encode s | Encode % | Decode base s | Decode s | Decode % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hpt_data | both | 80 | 193.4631 | 0.2793 | 0.4686 | 0.3804 | -18.8176 | 0.0891 | 0.0525 | -41.0986 |
| hpt_data_val | both | 20 | 91.1562 | 0.7189 | 0.3732 | 0.2980 | -20.1445 | 0.0753 | 0.0454 | -39.6669 |
| hpt_data | grid | 80 | 193.4631 | 0.2793 | 0.4686 | 0.4421 | -5.6537 | 0.0891 | 0.0530 | -40.4965 |
| hpt_data_val | grid | 20 | 91.1562 | 0.7189 | 0.3732 | 0.3531 | -5.3789 | 0.0753 | 0.0465 | -38.1924 |
| hpt_data | skip | 80 | 193.4631 | 0.2793 | 0.4686 | 0.4073 | -13.0778 | 0.0891 | 0.0886 | -0.5577 |
| hpt_data_val | skip | 20 | 91.1562 | 0.7189 | 0.3732 | 0.3215 | -13.8680 | 0.0753 | 0.0747 | -0.7807 |

Measured combined effect versus the sum of separately measured absolute savings:

| Set | Separate savings sum s | Measured combined saving s | Offset s |
| --- | --- | --- | --- |
| hpt_data | 0.0878 | 0.0882 | -0.0004 |
| hpt_data_val | 0.0718 | 0.0752 | -0.0033 |

A positive offset means combined saving is smaller than the separate-savings sum; a negative offset means larger. Independent work reductions do not guarantee additive latency because allocations/cache/order and timing noise interact. Metric skipping does not change decoder code; its small decode differences are noise. Grid/both decode gains are the relevant causal comparison.

Actual-call means and repeat variability:

| Set | Variant | Encode mean s | Encode CV % | Decode mean s | Decode CV % |
| --- | --- | --- | --- | --- | --- |
| hpt_data | baseline | 0.4702 | 2.6742 | 0.0898 | 4.8975 |
| hpt_data | grid | 0.4439 | 2.6989 | 0.0534 | 5.4970 |
| hpt_data | skip | 0.4099 | 3.0407 | 0.0893 | 4.2513 |
| hpt_data | both | 0.3807 | 2.4552 | 0.0530 | 4.7399 |
| hpt_data_val | baseline | 0.3760 | 3.3513 | 0.0756 | 4.8439 |
| hpt_data_val | grid | 0.3531 | 3.2127 | 0.0464 | 5.8701 |
| hpt_data_val | skip | 0.3237 | 4.1608 | 0.0749 | 5.1409 |
| hpt_data_val | both | 0.2987 | 3.3847 | 0.0457 | 5.4225 |

## Allocation, dtype and startup implications

MSB bit order, padding and reader state are unchanged for valid inputs. The writer requires a flat integer array; production supplies raveled integer indices. It normalizes to uint16 and masks via selected low bits. The reader uses uint16 arithmetic with supported widths, handles non-byte alignment and normalizes noncontiguous byte buffers only when needed. Ordinary LZMA/STORE byte bodies avoid that copy.

Vectorization allocates O(grid cells × bit width) temporary arrays. Large-grid tracked allocation probes (Python/NumPy tracking, not total native memory):

| Implementation | Operation | Width | Tracked peak bytes |
| --- | --- | --- | --- |
| scalar | write | 1 | 128903 |
| scalar | read | 1 | 1000476 |
| numpy | write | 1 | 2000596 |
| numpy | read | 1 | 2501338 |
| scalar | write | 5 | 657007 |
| scalar | read | 5 | 1000484 |
| numpy | write | 5 | 10000572 |
| numpy | read | 5 | 8501322 |
| scalar | write | 9 | 1183151 |
| scalar | read | 9 | 1000532 |
| numpy | write | 9 | 18000556 |
| numpy | read | 9 | 14501314 |

Small/medium probes also remain in grid_allocation.json. NumPy is already a codec dependency; operator is standard library. There is no new model, GPU or compiled extension dependency. Fresh-process import measurements are in startup.json; OS file caches are warm and these are not pristine cold-machine tests. Primary encoder timers include the complete compress call; interpreter import and worker-mode installation are separate. Warmup rows retain first-invocation/cache-loading effects.

## API hazard and valid caller paths

`compute_final_mse=False` already exists. It leaves the reconstructed preview available and returns internal `result.mse=None`; the artifact and independent decoded metric are identical. The current website executes `float(result.mse)` in server.py, so globally disabling this flag would break the response. Keep public UI configs at True. A file-only caller can opt out. A benchmark requiring a quality value must independently decode/score and report that metric's cost outside encode latency; a UI that still computes MSE elsewhere has not eliminated that work. Do not substitute zero or silently display None. No server.py change is proposed in this patch.

## Concrete unapplied deployment plan

1. Review proposed_grid_io.patch. Its only production file is pbc3_stream.py: retain scalar functions, add tested NumPy functions and a `GRID_IO_MODE` dispatch defaulting to scalar. No PBC3Config, decoder syntax, model, preset or public API changes. The isolated candidate copy has passed exactness checks. Rollout of a NumPy default would be a separate approval after HF measurements.
2. For the first authorized cpu-basic test, production code need not change at all. Stage only hf_cpu_benchmark.py, hf_manifest.json and integration_candidate/pbc3_stream.py, plus the six immutable release Python modules under a private reference directory. Use the existing production model and existing hpt_data/hpt_data_val files after hash checks. Model SHA-256: `ffaf24aa5447b3602c53a5c4c8df14a68ee16e6bded39331d973f38d0f1ea62f`. The manifest validates normalized-LF source hashes, candidate hash, all 25 raw input hashes and model hash before timing.
3. Run the private harness from the HF checkout, after explicit authorization, with `python research_pass3/hf_cpu_benchmark.py --reference PATH_TO_IMMUTABLE_RELEASE --output research_pass3/hf_cpu_basic_run --label HF_CPU_BASIC --repeats 3`. It does not use HTTP, upload images, restart services or alter production source. A local packaging smoke run is retained under hf_plan_smoke and is explicitly labeled LOCAL_PACKAGING_SMOKE; it is not HF evidence.
4. Reference and current source trees require separate persistent subprocesses to prevent module leakage. Both pin to the same first allowed logical CPU (or explicit --cpu N), use numerical thread settings 1 and process requests serially. All four current modes execute on one warmed current worker; no concurrent mode switching in the public FastAPI process is permitted. Model/config/module paths, actual versions, cold worker startup, warmups and raw rows are retained. The authoritative release is the primary old control. If actual old_pbc is additionally audited, use an isolated worker and identify that source honestly; do not silently call it source-identical release.
5. The existing broader benchmark's _eval_pbc3 correctly calls compress/decompress and scores decoded RGB, but lacks these paired modes/provenance. Quick RD lacks independent decode timing and is unsuitable for promotion. The supplied private harness reuses the validated methodology with standalone packaging, rather than changing the public benchmark APIs. It records full encode/decode separately; data loading, metric computation, inspection and IPC stay outside those timers.
6. No HTTP endpoint change is needed for CPU validation. If an approved service experiment is later required, add a private, serial job runner over existing server-side inputs. Record codec encode/decode, metric, preview PNG/base64, serialization and total server times separately. Measure client RTT independently; RTT minus server time includes transport and queuing and must not be called codec time. Never infer encoder latency from a browser progress timer or subtract heterogeneous historical results.
7. Promotion requires zero baseline-versus-optimized stream/pixel/count/MSE/bpp differences on all same-runtime cases and repeated calls, valid malformed/alpha checks, positive repeatable aggregate encode gains and substantial decode gains without a material latency-tail regression. Rerun noisy outliers rather than accepting a single timing. Keep UI MSE enabled. Compare HF only with HF; local speed percentages are not deployment results. Existing cross-runtime encoder trajectory differences remain a separate issue.
8. Rollback is mode=scalar in the private worker. If an authorized source rollout later occurs, restore only pbc3_stream.py from its recorded pre-change backup/revision and restart only with separate approval. Preserve all unrelated files. No Git reset, synchronization or model rollback is part of this plan.

This is ready for a specifically authorized private HF validation test, subject to the limits above. It is not an approval to deploy and is not a claim of validated cpu-basic performance. No commit, push, deployment, restart or remote compression call occurred.
