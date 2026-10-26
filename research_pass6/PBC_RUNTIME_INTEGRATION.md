# Runtime integration

The regular codec now uses vectorized grid I/O and explicit per-encode search-state reuse. Internal MSE stays enabled by default; its existing opt-out remains available. Stream format, presets, model and version are unchanged. This is not a PBC 3.1 release.

`pbc3_stream.py` directly exposes the tested NumPy reader/writer, retaining scalar reference functions. It removes the mutable mode dispatch and the unnecessary int64 index copy. Canonical bit ordering, offsets, low-bit masking, reader state and v0 bytes are preserved. Supported helper widths are 0–16; actual production palettes remain at most nine bits. Positive-width payload size is checked before allocating output.

`pbc3_ops.SearchState` belongs to one compress_stream call. It stores exact int32 visible-error maps and integer totals, binds them to target/canvas identity, and invalidates on warmup replacement. The accepted patch is applied before its rectangle is refreshed. The state supplies an explicit abs_error argument through the heuristic and learned proposal paths; integral construction, sampling, RNG and ordering are unchanged. No global function replacement, inspect/exec, mode switch, shared encode cache, or ctypes channel inference is used in production.

The local gate passed 100 saved default cases across all 20 main/five holdout images and all four presets, matching compressed bytes, decoded pixels, patch counts/order, MSE and bpp. It also passed 20 synthetic RGB/RGBA/native/downsample/init-only/heuristic/delta-reuse/warmup cases, bidirectional old/new v0 decode, 272 width/alignment/count tests plus extreme integer/strided-buffer cases, 9 malformed-stream outcome comparisons, 20 exact proposal/RNG checks, and 4 overlapping calls. Public metric-off behavior remained None; metric-required values matched. Compile/import checks passed.

The first gate run completed its 100 comparisons but a warning from an invalid synthetic warmup setup interrupted JSON IPC before final result export. The synthetic setting was corrected; the completed gate was rerun and saved. No codec output failure occurred. No historical large archive was rebuilt.

HF is running [cd0d5691647a4f02cf765454a2d3b0abfde8f3b7](https://huggingface.co/spaces/EgeEken/PBC/commit/cd0d5691647a4f02cf765454a2d3b0abfde8f3b7). The runtime-only commit changes five codec files, redirects the existing authenticated benchmark to a bounded non-monkeypatch worker, and preserves the prior eight-module codec as an isolated control. No refinement implementation, dataset, weight, dependency, secret, frontend, preset, release tag or version change was deployed. Existing secret authorization was reused; no rotation was needed.

The new helper uses two persistent isolated source processes on the same pinned CPU, executing sequentially with rotated order. This avoids global replacement in either public operations or the remote validation worker. The API allowlist is now at most three images/four presets/three variants/three repeats (108 calls maximum); this pass used a six-call Balanced smoke and 54 timed calls over cat/flower/googlewiki at Quality/HQ. Warmups are complete encode/decode calls per source/configuration. Public compress and existing-v0 decode responses were independently verified, including exact old RGBA stream, preview, dimensions and displayed MSE.

| Mode | Encode s | Encode change | Decode s | Decode change |
| --- | ---: | ---: | ---: | ---: |
| baseline | 0.837813 | +0.00% | 0.205201 | +0.00% |
| all_mse | 0.690236 | -17.61% | 0.106760 | -47.97% |
| all | 0.602111 | -28.13% | 0.098735 | -51.88% |

These are means of six case medians from three interleaved repetitions. all_mse is the actual public-compatible implementation; all skips internal MSE only for the private worker. Full call walls include cache setup and updates; loading, IPC and independent scoring are outside the timers. All 54 streams/pixels/trajectories/MSE/bpp match their same-runtime control. This smaller set differs from Pass 5, so the percentages are not a direct improvement over its eight-image aggregate.

HF worker Python: 3.11.17 (main, Oct 21 2026, 01:07:38) [GCC 14.2.0]; packages: {'numpy': '2.4.6', 'numba': '0.68.0', 'Pillow': '11.3.0'}; numerical thread variables all one, CPU 0. Encode CV baseline/integrated/skip: 6.0%/7.8%/6.7%. Decode CV: 13.6%/17.2%/16.2%. All-call means are retained alongside the medians in hf_runtime_results.json.

All six public-compatible case medians improved in both metrics; no median runtime regression was observed. Encode reductions ranged from 1.59% (flower Quality) to 29.77% (googlewiki Quality); decode reductions ranged from 6.60% to 81.75%. The small flower Quality gain is within timing variation and is not a precise per-image speed claim.

Remaining limits: the benchmark is warmed and runs on a public/shared HF CPU, so scheduling/traffic noise remains. Vectorized bit packing allocates grid-size temporaries; cached error maps cost about 3 MB for RGB/250k pixels. This pass does not claim equal variant-specific peak memory or cold startup latency. Existing unbounded header-canvas allocation behavior is not repaired by grid I/O changes. Canonical v0 compatibility was tested; noncanonical private helper validation is stricter. Prepared target arrays must not be mutated externally during an encode. Historical monkeypatch-based research adapters may need their new abs_error interface forwarded; use the preserved baseline for old experiments.

Rollback is a non-destructive revert of HF commit cd0d5691647a4f02cf765454a2d3b0abfde8f3b7 followed by an authorized push to HF main; never reset/force-push or discard unrelated work. The parent is eee83b8f10f6f3404bca1c2c178e27fc4e8a1aa0. No rollback was required. The actual GitHub integration URL is recorded in PBC_PASS6_SUMMARY.md after committing.
