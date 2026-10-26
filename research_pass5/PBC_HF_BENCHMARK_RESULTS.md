# Existing HF Space runtime benchmark results

Outcome: Validated and ready to promote through a separate production integration. The optimizations ran on the actual existing HF cpu-basic deployment. Public codec defaults and UI remain unchanged. The current deployment exposes the optimized implementation only through the authenticated isolated benchmark worker.

Deployed revision: `eee83b8f10f6f3404bca1c2c178e27fc4e8a1aa0` on [EgeEken/PBC](https://huggingface.co/spaces/EgeEken/PBC). Both HF repository metadata and the running runtime revision were confirmed before and after measurement; health passed. Parent/rollback baseline: `7faa6365186d201f44b842f100645b60273ffd4c`.

## Measured result

The table uses the mean of 32 image/preset medians, each from three interleaved repeats. Negative percentages mean less time. These are complete server-side compress/decompress call walls, including search state creation, controlled-source compilation, updates and cleanup. File loading, independent scoring, IPC and HTTP are outside the timers.

| Variant | Encode seconds | Encode change | Decode seconds | Decode change | Encode/decode wins |
| --- | ---: | ---: | ---: | ---: | ---: |
| Unchanged production baseline | 0.530814 | +0.00% | 0.098395 | +0.00% | 0/0 of 32 |
| Grid + search, internal MSE retained | 0.447193 | -15.75% | 0.059877 | -39.15% | 32/32 of 32 |
| Grid + search, internal MSE skipped | 0.389335 | -26.65% | 0.056356 | -42.72% | 32/32 of 32 |

Public-API-compatible acceleration is the retained-MSE row. The optional skipped-MSE row is suitable only for callers that do not need that internal metric. Both are directly measured; separate speed percentages are not added.

| Preset | Retained MSE encode change | Skipped MSE encode change | Retained/skipped decode change |
| --- | ---: | ---: | ---: |
| compression | -10.93% | -25.59% | -19.53% / -16.46% |
| balanced | -16.76% | -26.77% | -27.65% / -33.27% |
| quality | -17.22% | -27.20% | -38.50% / -43.60% |
| high_quality | -17.00% | -26.82% | -54.70% / -59.05% |

Mean within-case repeat coefficient of variation:

| Variant | Encode CV | Decode CV |
| --- | ---: | ---: |
| baseline | 16.53% | 18.18% |
| all_mse | 17.03% | 20.21% |
| all | 17.58% | 18.46% |

Variation is substantial (roughly 17% encode and 18–20% decode CV), so the percentages should be treated as this bounded run’s estimates. The gain direction is consistent: both optimized modes win all 32 case medians in both metrics and all three complete-trial means.

| Variant | All-call encode mean s / change | All-call decode mean s / change | Per-trial mean encode changes |
| --- | ---: | ---: | --- |
| baseline | 0.553880 / +0.00% | 0.106051 / +0.00% | +0.00%, +0.00%, +0.00% |
| all_mse | 0.465109 / -16.03% | 0.063267 / -40.34% | -16.16%, -17.08%, -14.71% |
| all | 0.410485 / -25.89% | 0.060472 / -42.98% | -27.43%, -24.89%, -25.57% |

Largest encode regressions or smallest improvements:

- all_mse: flower.jpg / high_quality -2.64%, flower.jpg / quality -2.71%, stockpoint.jpg / high_quality -5.95%.
- all: stockpoint.jpg / quality -2.15%, stockpoint.jpg / balanced -12.26%, stockpoint.jpg / compression -13.34%.

## Correctness and service behavior

All 288 representative rows and the six initial HF smoke rows pass exact bitstream, decoded RGB pixels, independently measured original-size MSE, bpp and patch-count checks within the actual HF runtime. Cross-repeat determinism is also checked in the final analysis. Retained internal MSE values match baseline; skipped internal values are None. Final compressed bytes include the full post-LZMA file, not raw stream size. No quality/rate tradeoff is introduced by either optimization.

A separate cross-environment comparison matches 18/32 HF baseline streams/pixels/MSE against Pass 4 Windows baseline records; 14 differ. These are not mismatches between baseline and optimized variants on HF. Source/model/input hashes pass. Pass 1 already documented a current-code Windows-versus-HF trajectory difference on googlewiki Quality (PBC_BENCHMARK_VALIDATION.md, deployment_comparison.json), so cross-runtime identity was not an established baseline guarantee. The rebuild resolved unpinned NumPy/Numba versions as well. The exact cause of each new mismatch is not isolated; platform/library numerical and ordering behavior are hypotheses. Do not attribute it to the optimizations or claim it is harmless numerical noise. Do not compare local and HF files as an optimization pair. See prior_baseline_equivalence.json; the within-HF paired result remains exact.

The public /api/compress check used the existing deployed googlewiki.png and Balanced preset. That input is opaque RGBA, and the existing public handler preserves its alpha channel; benchmark inputs deliberately convert to RGB. The returned RGBA stream matches both a local encode with the unchanged public configuration and the actual predeployment Pass 1 HF Balanced stream (SHA256 2f7ef0a8dcdd22444a65ba03f3e2955dde57844d36db7fa269567ca38bcd9414, 51,493 bytes). Its PNG preview equals independently decoded RGBA pixels, and original dimensions, rounded compressed file size and displayed four-channel MSE are correct. Public compression handler/configs are unchanged. This is a service check, not an HTTP-based speed measurement. No new input was introduced. The optimized remote timing set is RGB only; the public RGBA service check exercises unchanged production code, while the prior local RGBA optimization checks are reused.

Two public requests were made. Initial local verification incorrectly interpreted the preview data-URL prefix, then incorrectly compared the public RGBA stream with the RGB benchmark stream. The checker was corrected and the saved second response was fully verified without another request. Remote codec output was correct; no deployed-source fix was needed. These local-client failures lost their in-memory random benchmark keys and led to two extra secret rotations/restarts. The successful HF smoke is retained and not rerun. client_correction.json records this operational issue.

Remote access checks returned 403 for absent/wrong research tokens and 400 for an unallowlisted path. The valid static variants were selectable and ran. The research start branch is disabled without its Space secret. No arbitrary paths, configs, code or modules can be supplied. Existing read-only status and ordinary benchmark access policies are retained.

## Configuration and limits

Eight predetermined existing main-set images: cat.jpg, monet.jpg, flower.jpg, cathedral.png, googlewiki.png, stockpoint.jpg, mario64.png, eiffel.png. All four unchanged presets; RNG seed 2003; same production model and image bytes, verified by manifest before execution. No allocation, RD, entropy or training experiment.

Representative measurement: one worker PID 18, pinned affinity [0], numerical thread variables all 1. Python 3.11.17 (main, Oct 21 2026, 01:07:38) [GCC 14.2.0]; NumPy 2.4.6, Numba 0.68.0, Pillow 11.3.0. HF reports cpu-basic and one replica. The exact physical CPU model is not exposed by platform.processor on this run; hardware-tier provenance is HF metadata.

Each variant/preset receives one complete encode/decode warmup on googlewiki.png in the same worker (12 full warmups). Measured order rotates per image, preset and trial so every case visits each variant position across three repeats. The initial separate one-image Balanced smoke has six measured calls plus three warmups. No complete previous local suite was repeated: assembled-candidate local smoke has six rows plus three warmups, compile checks, route/auth/allowlist checks and source equivalence checks.

Observed whole-worker cumulative peak RSS reaches 1031.3 MiB. ru_maxrss is a process high-water mark shared across sequential variants; it cannot identify variant-specific incremental memory. Previous tracked vector grid temporary sizes remain relevant; this run shows no OOM or worker failure, but does not prove equal production peak memory. The worker's first warmup includes cache-loading/first-call effects; this is a warmed throughput/latency comparison, not a pristine cold-start benchmark.

Existing benchmark locking excludes simultaneous benchmark/sweep/training jobs; before/after status snapshots are retained. The public Space remains online, so unobserved user traffic and shared-host scheduling are possible sources of variation. Three repeats and interleaving limit this issue; they do not create dedicated hardware. No significant compatibility or reliability failure was observed in the tested calls.

## Decision and promotion boundary

The measured HF result supports promoting exact grid I/O and search-state reuse while retaining internal MSE for the public API. Metric skipping remains an explicit private/file-only option. This task does not switch the production default: the user explicitly required public defaults and behavior to remain unchanged.

The benchmark uses scoped monkeypatches in an isolated process. That installation mechanism must not be copied into a concurrent public request handler: a production integration should use per-encode owned search state and an explicit grid implementation path. The underlying exact runtime optimizations are validated; safe public integration remains a small separate reviewed code change. No allocator or PBC 3.1 all-metric release claim follows from this runtime result.

## Deployment, rollback and repository preservation

Exactly six reviewed files were committed: server.py, pbc3_benchmark.py, research_pass5/hf_runtime_job.py, research_pass5/hf_candidate_grid.py, research_pass5/hf_candidate_search.py, research_pass5/hf_runtime_manifest.json. Candidate grid/search modules are byte-identical to Pass 4; the worker extends only the required static modes/image/preset limits and adds RSS/PID reporting. The endpoint/backend reuse the prepared Pass 4 proposal, fix clearing the stop event before its running check, and return a bounded worker error for diagnosis. No dependency, model, dataset, default, UI, production codec or DB file is in the commit.

The final private benchmark token is a domain-separated HMAC derived in memory from the existing HF credential and deployment revision. Its value is installed through the Space secret API and is never printed, saved locally or committed; the HF credential itself is never sent to the app. An authorized local client can reproduce the scoped token after a retry without unnecessary rotations. The secret remains enabled; rotate/remove it to revoke/disable research starts. Secret configuration and the push restart/rebuild the Space; health and revision were verified afterward.

Rollback: create a clean branch based on current HF main, run `git revert eee83b8f10f6f3404bca1c2c178e27fc4e8a1aa0`, then push that revert to HF main. Do not force-push or reset user work. Optionally delete PBC_RESEARCH_BENCHMARK_TOKEN to disable research access. The previous source is parent 7faa6365186d201f44b842f100645b60273ffd4c; no persistent benchmark DB changes need reversal.

The deployment commit was constructed from HF main with an isolated index and recorded on local codex/pass5-hf-runtime, then pushed to HF main. The original multistage_pbc31 branch and its index stay in place; its unrelated experimental commits were not pushed. The two authorized backend edits remain visible in the original working checkout relative to its original HEAD. Unrelated .gitignore and user files remain untouched. Final status and preservation checks are supplied separately.

## Evidence

- hf_representative.json / hf_rows.csv: all raw measured rows, configs, timestamps in job records, source/model/input hashes, dependency versions, warmups and worker provenance.
- hf_smoke.json, public_api_smoke.json, remote_auth_checks.json: bounded remote smoke, public API and access-control checks.
- timing_summary.json, paired_cases.json / .csv: aggregate, per-preset, per-case medians, repeat variation and outliers.
- deployment.json, deployment.patch, deployment_files.json, push.json: exact six-file commit, source diff, target and rollback.
- hf_before.json, hf_deployed.json, hf_after.json, deployment_observations.json, health_deployed.json, service_before.json, service_after.json: revision, hardware and service state.
- DEPLOYMENT_REVIEW.md, candidate_checks.json, local_smoke.json: reviewed authorization scope and focused local verification.
- changed_files.txt, artifact_manifest.json, final_checks.json, git_status_after.txt: inventory and preservation evidence.
