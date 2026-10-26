# Reproduce Pass 2

Run from the repository root with the recorded Python environment. Scripts set numerical thread limits before numerical imports, and benchmark workers pin themselves to the first available logical CPU (0 here). Run CPU experiments sequentially. No command below commits, pushes, deploys, trains, uploads images, or writes production code.

The immutable release and frozen streams are read from research_pass1. Do not rerun Pass 1 commands or replace that snapshot. Pass 2 commands overwrite only their own output names under research_pass2; copy this directory first if retaining a second independent run. Plans contain absolute input paths from this checkout and must be adjusted explicitly if relocating the repository.

1. `python research_pass2/runner.py initial` — only in a fresh research output directory; refuses to overwrite the initial snapshot.
2. `python research_pass2/runner.py rng`, then `python research_pass2/analyze_rng.py equivalence`.
3. `python research_pass2/runner.py resolution`.
4. `python research_pass2/entropy.py accounting` — inspect before choosing transforms. The retained decision is entropy_representation_decision.json.
5. `python research_pass2/quality.py ablations`, then `python research_pass2/quality.py planned rate_screen`.
6. `python research_pass2/entropy.py screen`. No entropy candidate was promoted. entropy_promotion.json records the negative decision.
7. `python research_pass2/performance.py correctness`, then `python research_pass2/performance.py screen`.
8. `python research_pass2/analyze_rng.py discrepancy`. For the separate CPU variant, start a fresh process with PowerShell `$env:NPY_DISABLE_CPU_FEATURES='AVX2,FMA3'`, run `python research_pass2/analyze_rng.py cpu_variant`, then `Remove-Item Env:\NPY_DISABLE_CPU_FEATURES`. `python research_pass2/analyze_rng.py tie_analysis` replays the saved common prefix/local recorded RNG state without encoding.
9. `python research_pass2/analyze_rng.py profile`; `python research_pass2/performance.py components`; `python research_pass2/performance.py decoder_components`; `python research_pass2/quality.py trajectories`.
10. Run `python research_pass2/entropy.py memory --filter NAME` separately for production_0extreme, preset1, preset6 and preset0extreme_dict64k. These are synthetic allocation probes, not image RD results.
11. `python research_pass2/quality.py planned quality_confirmation`, then `python research_pass2/performance.py confirm`. Both read fixed promotion plans; holdout results are confirmation rather than tuning inputs.
12. `python research_pass2/analyze.py`; `python research_pass2/plot_rd.py`; `python research_pass2/reports.py`; `python research_pass2/final_checks.py`.

JSON/JSONL is authoritative for expanded configurations, patch/accounting lists, dependency/source/model/input identities and module inventories. CSV exports keep scalar columns for analysis. Full per-case repeated latencies remain available even where report tables use means of per-case medians. First-repeat complete files are saved under streams/. Experimental entropy decoders live only in entropy.py and are never registered in the production decoder.

Reports: RNG_SPEED_ATTRIBUTION.md, PBC_QUALITY_SCALING.md, PBC_ENTROPY_RESEARCH.md, PBC_PERFORMANCE_RESULTS.md and PBC_PASS2_SUMMARY.md. The summary explicitly answers all twelve research questions. changed_files.txt and artifact_manifest.json inventory the complete new output directory, excluding generated caches. final_validation.json verifies source/input/Pass 1 integrity, stream hashes, repeated determinism and output equivalence.
