# Baseline and history

Pass 6 update: the root runtime now includes the exact integration deployed at HF `cd0d5691647a4f02cf765454a2d3b0abfde8f3b7`, parent `eee83b8f10f6f3404bca1c2c178e27fc4e8a1aa0`. The previous eight-module codec is preserved under `research_pass6/baseline/`. The provenance below describes the original research snapshot; it is not a claim that the current integrated root is still the Pass 5 source.

`research/experiments` starts at original GitHub `main`, `162feed4a85b81f0d2494e05bd39d8e159788e7c`, preserving its 65 existing commits. At setup, GitHub `rust` was `93e9c0591a710397bb3f09d2d51473313ce23db6` and `v3.0` was `1ef5155ac6a0037e02b0130b6b5091f0c1f791c3`. These branches are unchanged.

The HF checkout and GitHub repository have unrelated roots and no merge base. HF history is not merged, rewritten, or force-pushed. Snapshot commits make the current source reviewable without importing the HF history's datasets, streams, databases, model-training outputs, or unrelated changes. Existing GitHub history remains the research branch's ancestry.

The source baseline is the current local HF checkout at `3e9e8c94b7497ecefc5e482a66fde697dfad6e19`, plus the two authorized backend edits already deployed as HF commit `eee83b8f10f6f3404bca1c2c178e27fc4e8a1aa0`. The root codec files are unchanged between that local commit and the deployment. The six deployed paths are `server.py`, `pbc3_benchmark.py`, and `research_pass5/{hf_candidate_grid.py,hf_candidate_search.py,hf_runtime_job.py,hf_runtime_manifest.json}`. They match that HF revision with normalized LF text.

The immutable GitHub release control is `2e59f1b06980aa467da0cde94fa33c3f9a0d4d51`; its six runtime modules are copied under `research_pass1/release/`. Actual `old_pbc/` runtime sources and their required action/feature/proposal helpers are included separately. These are not renamed current-code ablations. The original 104,903-byte model is the existing GitHub blob `df8ea77d1c804c8fad35ecb6e3cd0e0267300081`, SHA256 `ffaf24aa5447b3602c53a5c4c8df14a68ee16e6bded39331d973f38d0f1ea62f`. Root, old and release model paths reuse that same object; no new model payload is uploaded.

Source-only snapshots also preserve the local multistage/proposal work from HF commits `039174f144981ae09830f98a225299c1299d2067`, `b2de015caefcc4415c2bb147e7b05eec85d2b316`, and `3e9e8c94b7497ecefc5e482a66fde697dfad6e19`. Original commit hashes are provenance, not ancestors of this GitHub branch.

Snapshot commits are organized as: codec/reference baseline and reproduction policy; Pass 1–2 tooling/reports; allocator, runtime and other isolated experiments from Pass 3–4; deployed Pass 5 package and findings. No new benchmark, allocator investigation, deployment, secret configuration or HF push occurs during this setup.

The original dirty HF checkout and its index stay intact. A separate research checkout tracks GitHub `research/experiments`; its push remote is `github`. The existing HF `origin` URL is retained for provenance, not authorized as a push target. Subsequent research belongs on this same branch. Each handoff should name the exact GitHub commit URL, source/control configs and any remaining data dependencies. Production promotion is a separately reviewed change.
