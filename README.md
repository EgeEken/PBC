# PBC research experiments

Permanent experimentation branch: [`research/experiments`](https://github.com/EgeEken/PBC/tree/research/experiments). This branch is for source review and reproducible research. It is not intended to be merged directly into `main`.

The current codec, reference runtimes, experimental allocators, runtime optimizations, benchmark tools, and Pass 1–5 reports are included. Production defaults remain unchanged. Read [RESEARCH.md](RESEARCH.md) for the findings, [the reproduction guide](research/REPRODUCTION.md) for dependencies and external inputs, and [the baseline record](research/BASELINE.md) for Git history and HF deployment provenance.

The root runtime and six deployed Pass 5 files correspond to HF revision [`eee83b8`](https://huggingface.co/spaces/EgeEken/PBC/commit/eee83b8f10f6f3404bca1c2c178e27fc4e8a1aa0). Additional experiments are isolated under `research_pass1/` through `research_pass5/` and the named `pbc31_*` modules. They are not enabled as public codec defaults.

Datasets, generated streams, databases, caches, large benchmark results and credentials are excluded. The original production model already exists in GitHub history and is reused unchanged. [File inventory](research/tracked_files.txt), [exclusions](research/EXCLUSIONS.md), and [source/asset manifests](research/) describe the snapshot precisely.

Subsequent experiments should continue on this same branch with focused commits and GitHub commit links in each handoff. GitHub `main`, `rust`, `v3.0`, and HF `main` require separate authorization for changes.
