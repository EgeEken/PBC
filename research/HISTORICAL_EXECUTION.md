# Historical RNG and source selection

Pass 10 promotes the root codec default to release RNG. Earlier measurements retain their original RNG/source meaning. Do not regenerate them with the new implicit default.

- Passes 1–8 and the archived `pbc31_*` experiments are historical. Root-dependent custom-RNG execution must use a revision before promotion, such as GitHub `e6c8eca726b3048a92514bd09d31fca0b3e00fd7`, plus the source/control revision and dependencies recorded in the particular report. Earlier runtime/algorithm comparisons may additionally require their preserved source directories or original local archive. Merely running a historical script against today's root is not reproduction.
- Pass 9's three-way worker remains active. `dev_release_rng` explicitly selects `release`; `dev_current_rng` explicitly selects `current`; `release` imports the immutable implementation. Its current runtime manifest was refreshed for the promoted source. Saved Pass 9 measurements retain their original measured manifest in `comparison_summary.json`.
- `research_pass9/check_public_api.py` checks the prepromotion custom-default response and is historical; run it at `e6c8eca`, not against the promoted public default. Use Pass 10 public checks for the new baseline. Pass 9's earlier deployment preparation scripts also target their recorded old HF parent and must not be replayed on the new deployment.
- The immutable `research_pass1/release/`, actual `old_pbc/`, and preserved Pass 6 preceding-HF control are unchanged. They remain distinct controls; no historical measurement has been relabeled as a new default run.

Inspect pinned source with `git show REVISION:path`. Operational historical scripts still require separate authorization. Do not run them merely to inspect this snapshot. For new RD work, use the frozen `pbc31_release_rng_baseline` commit/manifest and explicitly identify any custom-RNG control.
