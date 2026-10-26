# PBC research experiments

Permanent experimentation branch: [`research/experiments`](https://github.com/EgeEken/PBC/tree/research/experiments). This branch is for source review and reproducible research. It is not intended to be merged directly into `main`.

The current codec, reference runtimes, experimental allocators, runtime optimizations, benchmark tools, and Pass 1–5 reports are included. Production defaults remain unchanged. Read [RESEARCH.md](RESEARCH.md) for the findings, [the reproduction guide](research/REPRODUCTION.md) for dependencies and external inputs, and [the baseline record](research/BASELINE.md) for Git history and HF deployment provenance.

The integrated root runtime is deployed as HF revision [`cd0d569`](https://huggingface.co/spaces/EgeEken/PBC/commit/cd0d5691647a4f02cf765454a2d3b0abfde8f3b7). It preserves the prior codec's exact outputs and public MSE behavior. Historical experiments are isolated under the pass directories and named `pbc31_*` modules; unvalidated RD experiments remain disabled in the public codec. See [Pass 6 runtime integration](research_pass6/PBC_RUNTIME_INTEGRATION.md).

Datasets, generated streams, databases, caches, large benchmark results and credentials are excluded. The original production model already exists in GitHub history and is reused unchanged. [File inventory](research/tracked_files.txt), [exclusions](research/EXCLUSIONS.md), and [source/asset manifests](research/) describe the snapshot precisely.

Subsequent experiments should continue on this same branch with focused commits and GitHub commit links in each handoff. GitHub `main`, `rust`, `v3.0`, and HF `main` require separate authorization for changes.
