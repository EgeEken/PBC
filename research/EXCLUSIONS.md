# Included and excluded files

`tracked_files.txt` lists the exact branch tree. `snapshot_manifest.json` hashes that tree, excluding itself, and includes inherited GitHub files as well as the new research source. The only binary objects at the tip are three paths referencing the same production-model object already present on GitHub.

Included: current codec, old/reference runtime sources and unchanged model aliases; benchmark backend and UI source dependencies; all Pass 1–5 Python implementations and report/plan generators; scalar/NumPy grid implementations; exact search reuse; rate-aware representation selector and its cache experiments; source-only multistage/proposal variants and tests; detailed reports plus the concise root summary; small selected decision, manifest and timing-summary JSON; source patches and dependency declarations.

Excluded from new commits and newly introduced history:

- All 25 main/holdout dataset images, frontend sample images and other image/video files; only filenames, dimensions where reported, and hashes remain.
- All generated PBC/PBC3 streams and reconstructed media, including frozen controls. Existing local copies are untouched.
- Benchmark/sweep databases, training checkpoints, student models and training runs; no new model payload is uploaded.
- Large JSON/JSONL/CSV row dumps, per-patch traces, profiler dumps, generated charts, temporary plans containing machine-local snapshots, run/status/build logs and Git index files. A few compact decision/summary JSON records are included intentionally.
- Python/Numba/plot caches, virtual environments, local app attachments, credentials, environment files, Git credentials and all HF secret values. Operational source refers only to secret names or obtains credentials in memory.
- Unrelated untracked user scripts, notes, handoffs and local ignore edits. Only the existing server's direct training-source dependency is retained, without its data or outputs.
- HF Git ancestry and its historical assets. The histories have unrelated roots; importing it would publish unrelated historical payloads even if removed from the final tree.

Existing GitHub history is preserved unchanged. Historical GitHub assets already in that history are not rewritten or purged. No protected GitHub branch or HF branch is changed by creating this research branch.
