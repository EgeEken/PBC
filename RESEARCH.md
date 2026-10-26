# PBC research snapshot through Pass 5

This snapshot records completed work; branch setup runs no benchmarks and does not begin Pass 6.

| Work | Result | Source |
| --- | --- | --- |
| Pass 1 baseline reconciliation | Immutable GitHub release and actual `old_pbc` outputs agree on 100 local cases. Current HF engineering code has a different RNG trajectory and must remain distinguishable. | `research_pass1/`, `PBC_BASELINE_AUDIT.md`, `PBC_BENCHMARK_VALIDATION.md` |
| Pass 2 resolution, entropy and runtime | Higher resolution/budget changes the RD operating point; no entropy replacement passed the promotion gate. Vector grid I/O and existing internal-MSE opt-out warranted confirmation. | `research_pass2/PBC_PASS2_SUMMARY.md` |
| Pass 3 allocation and search reuse | Frozen rate-aware selector compares the original representation with coarser/lower-precision alternatives, weight 0.4, Quality 500k/35. Exact rectangle search reuse preserves output. | `research_pass3/representation.py`, `variants.py`, `search_reuse.py`, `PBC_PASS3_SUMMARY.md` |
| Pass 4 matched-rate validation | Best actual controls within 5% rate: allocator MSE wins on 16/19 main and 3/4 holdout cases. High-quality holdout coverage remains inadequate. Exact allocator caches saved only 0.90% and failed the 2% promotion gate. | `research_pass4/PBC_PASS4_SUMMARY.md`, `allocator_cost.py`, `variants.py` |
| Pass 5 actual HF cpu-basic timing | Eight images × four presets × three variants × three repeats: 288 measured calls. Retained internal MSE: encode −15.75%, decode −39.15%. Skipped MSE: encode −26.65%, decode −42.72%. All optimized outputs match the same HF baseline. | `research_pass5/PBC_HF_BENCHMARK_RESULTS.md`, `timing_summary.json` |

Pass 5 averages per-image/preset medians. Timing CV was roughly 17–20%; all 32 case medians and all three complete-trial means favored both optimized modes. The public Balanced RGBA response matches the predeployment HF response. Separate Windows/HF baseline differences were already observed in Pass 1 and remain documented; they do not constitute an optimization pair. Memory observations are cumulative process high-water marks, not variant-specific peak comparisons.

The allocator remains experimental. `mario64.png`, `eiffel.png`, and `pigeon.JPG` have measured dominance failures; Monet's direct and interpolated comparisons disagree. Do not promote it as a universal high-quality or all-metric release improvement. The cache cost branch stopped at its negative gate.

The successful grid/search implementations are in `research_pass5/hf_candidate_grid.py` and `hf_candidate_search.py`, with the actual deployed worker in `hf_runtime_job.py`. They remain isolated benchmark variants. The public compression code retains its original defaults and internal quality metric. Production integration should give each encode its own search state instead of applying global monkeypatches across concurrent requests.

Earlier multistage/proposal experiments are preserved as named `pbc31_*` implementations, tests and their two reports. Their original HF commit IDs are recorded in `research/provenance.json`; their generated result dumps and unrelated HF history are not imported.

Detailed reports preserve the original chronology. Later passes supersede earlier confidence claims, especially sparse allocator interpolation and the initially invalid allocator timing labels corrected in Pass 4. Large raw evidence remains in the original local research archive; its selected hashes are in `research/evidence_catalog.json`.
