# PBC baseline audit

Audit date: 2026-10-25. No production source, defaults, model, or reference implementation was changed.

## Inspected states

| State | Repository/ref | Finding |
|---|---|---|
| Immutable public release | `EgeEken/PBC`, `2e59f1b06980aa467da0cde94fa33c3f9a0d4d51` | GitHub identifies this as “PBC 3.0 True True Release”. Seven source/metadata files were retrieved and every downloaded blob checked against its Git tree. The local model also matches the release Git blob. Snapshot: `research_pass1/release/`. |
| Public main | `EgeEken/PBC`, `main`, `162feed4a85b81f0d2494e05bd39d8e159788e7c` | Verified through GitHub. Two commits after the release; only README changed. Runtime is the release runtime. |
| Historical reference | HF workspace `old_pbc/` at checkout HEAD | Last subtree change: `585244b508b1e36d3b4877dd7d14421da1d4387c`. A structurally different predecessor with more configuration options, rather than a source-identical copy of the final release. |
| Current root | HF repository, branch `multistage_pbc31`, `3e9e8c94b7497ecefc5e482a66fde697dfad6e19` | Codec files are clean against HEAD; existing `.gitignore` changes and unrelated untracked user work were preserved. AGENTS.md describes an older state and cannot establish current equivalence. |
| Observable HF deployment | `EgeEken/PBC`, reported repository and runtime SHA `7faa6365186d201f44b842f100645b60273ffd4c` | HF API reports RUNNING on cpu-basic. All 12 checked production codec/model/startup files at this immutable revision match local root after CRLF normalization. The commits between this revision and local HEAD add isolated research files/reports/tests, without changing those files. |

GitHub evidence: [release commit](https://github.com/EgeEken/PBC/commit/2e59f1b06980aa467da0cde94fa33c3f9a0d4d51), [main](https://github.com/EgeEken/PBC/tree/162feed4a85b81f0d2494e05bd39d8e159788e7c), [release-to-main comparison](https://github.com/EgeEken/PBC/compare/2e59f1b06980aa467da0cde94fa33c3f9a0d4d51...162feed4a85b81f0d2494e05bd39d8e159788e7c). HF evidence: [Space metadata](https://huggingface.co/api/spaces/EgeEken/PBC), captured in `research_pass1/space_metadata.json` and `deployment_source_checks.json`.

HF's reported runtime revision is stronger evidence than repository HEAD alone, but the running process does not expose `sys.modules`, installed package versions, or in-memory model hashes. GET health/status requests succeeded; training, Quick RD, and the benchmark runner reported idle at the observation time. OS-level worker inspection was unavailable. This does not prove absence of every external process. The health boot ID is recorded in `space_health.json`; it is not a commit identifier.

After the user authorized four uploads of original googlewiki.png, live `/api/compress` checks matched local RGBA file bytes and decoded pixels for compression, balanced, and high_quality. Quality differed first at decoded patch index 14 (zero-based): the selected box changed, after 14 matching decoded patches including four initialization patches. Its remote RGB MSE was 799.084542 versus local 796.712163; final sizes were 79,880 versus 81,978 bytes. All four remote previews, displayed file sizes, and displayed RGBA MSE values agree with decoding the returned files. This confirms substantial observed alignment but does not establish complete cross-runtime equivalence. The cause of the quality trajectory difference remains unresolved; installed versions and loaded process state are not exposed. Raw artifacts and comparisons are retained in `research_pass1/deployment_comparison.json` and the `space_googlewiki.*.pbc3`/`local_rgba_googlewiki.*.pbc3` files. No extra compression requests beyond the four approved checks were made.

## Runtime identity

All three implementations expose `PBC3.compress`, `compress_stream`, `decompress`, `encode_file`, and `decode_file`. The header is `PBC3`, stream version 0; entropy methods are STORE=0 and raw LZMA2=2. All presets enable LZMA, YCbCr, automatic initialization, seed 2003, and a 250,000-pixel working limit. The first three presets request 50 iterative patches, high_quality 20; the three initial channel patches are additional. Learned q values are 0.4/0.6/0.8/0.95. Exact expanded configs and resolved module/model paths are in `baseline_metadata.json` and each raw benchmark row.

Model SHA-256, identical in release/root/old and the checked deployment-revision model:

`ffaf24aa5447b3602c53a5c4c8df14a68ee16e6bded39331d973f38d0f1ea62f`

The model is the legacy 30-action, hidden-width-128 model with `cheap=False`; production therefore uses the full feature extraction path. The historical reference imports deterministic proposal/action helpers from its own `old_pbc/training_space/`; these are runtime helpers, not a training rollout. Worker module inventories verify that root codec modules do not leak into the old/release encodes. Every benchmark config gives its own model an absolute path.

Local execution: Windows, CPython 3.13.5, NumPy 2.2.6, Numba 0.62.1, Pillow 11.3.0, OpenCV 4.12.0.88, psutil 7.1.0. CPU identifier: Intel64 Family 6 Model 154 Stepping 3, GenuineIntel; 20 logical CPUs visible, workers pinned to logical CPU 0. CIM access to the marketing CPU name was denied. All listed numerical thread environment settings are 1. This is not the HF runtime.

The release's pyproject asks for Python 3.12 and broad dependency ranges (and contains stale package version 2.3.1 metadata). The controlled local comparison runs all sources under the same available Python 3.13 environment; it is not a recreation of the original release environment. HF Dockerfile specifies Python 3.11 and Pillow 11.3.0; most other dependencies are unpinned. Actual deployed installed versions cannot be obtained from the exposed API.

## Substantive source differences

References below are file:line in the inspected copies. Exact unified diffs are `research_pass1/current_*.diff` and `old_*.diff`; machine-readable content hashes are `implementation_hashes.json`.

| Class | Difference | References and implications |
|---|---|---|
| A | RNG and sampled trajectory | Release `PBC3.py:411` constructs `np.random.default_rng`; current `PBC3.py:362` uses `ops.PBC3Rng`, defined at `pbc3_ops.py:28`. This is a different random sequence and bounded-integer sampling mechanism. Seed equality does not imply sampled-box equality. The originating HF commit is `1033abc889460c8081d82cfa2429114bb9cc2d7c`. |
| A | EXIF handling in direct codec calls | Release `_to_image` at `research_pass1/release/PBC3.py:36` returns PIL inputs without EXIF transpose; current `PBC3.py:68` transposes. Benchmark preprocessing explicitly transposes once and converts RGB for every source. Website also transposes before encoding (`server.py:342`). |
| A | Narrower canvas/resampling types | Release canvas is int32 (`release/PBC3.py:388`); current initialization uses int16 (`PBC3.py:29`). Current `_resize_canvas` uses float32/int16 (`PBC3.py:162`); release uses float64/int32 (`release/PBC3.py:242`). Potential behavior differences for warmup/custom extreme patch budgets must not be called performance-only without testing. Defaults disable warmup. Originating canvas change: `637bc3661c4a12babe97755e4695f9af1cc38ea7`. |
| A, opt-in | Residual projection and heuristic cell selection | Current config adds `residual_projection_mode`, `cell_selection_mode`, and `representability_threshold` (`pbc3_types.py:106`). Cell mean projection (`pbc3_ops.py:467`) and hybrid representability filtering (`pbc3_heads.py:150`) can change outputs. Defaults retain bicubic/gradient/0.0. Learned production filling bypasses heuristic selection. No production default was altered during this pass. |
| A, nondefault | Removed old capabilities | Old `_make_patch` (`old_pbc/PBC3.py:502`) supports explicit/auto palettes and dynamic bitcount options absent from final release config; old `_read_patch:471` can read explicit palettes. Final release/current `_read_patch` reject explicit palettes. Old channel scheduling accepts Off/Sum/Max (`old_pbc/PBC3.py:431`); final release/current normalize to Mod/Sum. Nondefault configurations are not interchangeable. |
| B, subject to measured equivalence | Reuse of reconstructed selected delta | Current init/filler returns candidate delta and applies it directly (`PBC3.py:334`, `:391`, `pbc3_heads.py:42`, `learned_filler.py:37`). Release reconstructs the chosen patch again. Current `reuse_selected_delta=True`; this avoids a duplicate resize without intentionally changing the selected values. Introduced by `2b58798396860d1fb8817e43f69014053f99dffb`. |
| B | Grid bit I/O | Release has per-value BitWriter/BitReader calls (`release/PBC3.py:95`); current moves identical framing to `pbc3_stream.py:9`/`:27` with local accumulators. Header, mask, palette fields, cell dimensions, and LZMA framing remain version 0. Valid generated-palette streams are the scope tested here. |
| B | Resampling implementation | Current uses contiguous float32 buffers with PIL `frombuffer` (`pbc3_ops.py:52`), release PIL `fromarray` (`release/pbc3_ops.py:375`). Both use PIL BICUBIC and round signed results to int16. No OpenCV bicubic substitution remains in default production. |
| B/C | Search extraction, anchor dtype, trace scopes, model construction | Search is consolidated at `pbc3_ops.py:585`; integral remains padded int64 NumPy cumsums (`:87`). Current removes redundant anchor float conversions (`:530`). Current caches NPZ arrays and constructs per-encode filler objects (`learned_filler.py:98`, `:142`); old caches mutable filler instances (`old_pbc/learned_filler.py:44`). Current optional nested tracing (`pbc3_trace.py:6`) also incurs small disabled-context overhead. No standalone speed attribution is made from the combined root comparison. |
| C | Module organization and deployment | Old monolith/kernel/helpers become release ops/heads/features; current further extracts stream and trace modules. Current FastAPI presets call production config factories (`server.py:297`); Docker starts `uvicorn server:app` from `/app`. Startup warms production RGB/RGBA learned paths (`PBC3.py:566`, `server.py:931`), whereas release preload uses a tiny heuristic encode (`release/PBC3.py:534`). Different warmup must be controlled explicitly. |
| D | Documentation and unrelated research/UI | GitHub post-release commits only edit README. Local multistage/proposal replay files are not imported by the production codec. Formatting, docstrings, and dead helper removal alone do not establish behavioral changes. |

Palette generation, quantization, additive patch application, initialization candidate recipe, quality schedule, final PIL resize, and LZMA selection retain their default design. `pbc3_features.py` is unchanged in normalized text. This is not a claim that the entire current encoder is equivalent: the RNG is an active default difference.

Prior benchmark JSON files and Git commit messages provide historical leads, not independently attributable validation. This pass validates the whole inspected default pipelines. It does not retroactively validate every optimization, opt-in projection mode, arbitrary alpha/warmup path, or unsupported old palette mode.

## Recommendation

Use the immutable GitHub release snapshot as the authoritative research baseline. Keep `old_pbc/` as a separately identified historical runtime and verify its equivalence on the intended default-preset dataset; do not describe it as source-identical to the final release. Preserve current root as the deployment candidate. Baseline measurements and stream-level equivalence counts are reported in `PBC_BENCHMARK_VALIDATION.md`; causal RNG checks and diagnostics are in `PBC_FIRST_PASS_RESULTS.md`.
