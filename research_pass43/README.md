# PBC V4 Pass 43

This isolated pass diagnoses V4-A’s low-rate regression and decode cost, then tests one integrated V3-style coarse base stage and one exact decoder optimization. No production or frozen V3.1 files were changed. Generated streams, PNGs, and raw benchmark outputs are under the ignored `outputs/` directory.

Start with [PASS43_RESULTS.md](PASS43_RESULTS.md). The experiments use the three Pass 42 images and exact frozen V3.1 source/model. The Pass 42 V4-A files are the controls. The v2 encoder accepts a captured V3 prefix so budget variants can be tested without changing V3.1.

Scripts reproduce the stage probe, decode profile, V4-B coarse budgets, timing, focused checks, and visual sheets. OpenBLAS and OMP threads were set to one for Pass 43 timings. AVIF and reserved validation images were not run.
