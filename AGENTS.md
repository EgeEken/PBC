# Research branch workflow

- Work on `research/experiments` in the original `EgeEken/PBC` GitHub repository. Reuse this branch for subsequent experiments; it is not a direct merge candidate for `main`.
- Make focused commits for meaningful research changes and include their GitHub commit URLs in handoffs, as requested by the user. Keep variants distinguishable. Never include unrelated work.
- Use the `github` remote and the explicit `research/experiments` destination. Never push to another branch or remote, especially HF `origin/main`, without separate explicit authorization. Never force-push, rewrite protected history, or merge the unrelated HF history into this branch.
- Keep changes surgical, code simple, comments few, and reports direct and concise. Preserve the source/control distinction and production defaults.
- Read `RESEARCH.md`, `research/BASELINE.md`, and `research/REPRODUCTION.md` before running experiments. No experiment is implied by checking out this branch.
- Keep generated streams, images/datasets, benchmark databases, large raw results, caches, credentials and Space secrets out of commits. Use the manifests and the existing local archive for reproduction.
- Historical scripts that apply patches, commit, push, configure secrets, upload, or execute remote jobs require explicit operational authorization. Do not run them merely to inspect this snapshot.
- Compare current code with the actual isolated `old_pbc` or immutable release where the research question requires it. A disabled feature in current code is an ablation, not the release reference.
- Measure full encode/decode calls, original-size independently decoded RGB MSE, and final post-LZMA bytes. Keep local and HF timing separate; warm consistently, constrain threads, interleave variants, retain raw rows locally, and report variation/outliers.
- Runtime mechanisms are exact within tested environments. The frozen allocator is an experimental content-dependent RD tradeoff, not an accepted PBC 3.1 release improvement. Retain named failures and the negative cache result.
- The public UI needs internal MSE. Leave it enabled. Scoped search/grid monkeypatches belong in isolated workers; public concurrent integration needs per-encode owned state.
