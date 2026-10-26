# Reproduce the first-pass audit

Run from the repository root with the same available Python environment. No command below commits, pushes, redeploys, trains, or edits production defaults.

1. Verify the release files against the Git blob IDs in `github_provenance.json`. `release/patch_policy.npz` is the verified local copy of the same release model, not a replacement weight file.
2. Run `python research_pass1/benchmark.py --smoke --repeats 2`.
3. Inspect smoke hashes, module paths, previews versus decoded output, and stderr logs.
4. Run `python research_pass1/benchmark.py --repeats 3` without concurrent CPU workloads. It writes 900 rows and first-repeat streams.
5. Sequentially run `python research_pass1/experiments.py compare`, `python research_pass1/rng_diagnostic.py`, `python research_pass1/experiments.py floor`, `python research_pass1/experiments.py profile`, and `python research_pass1/experiments.py screen`.
6. Run `python research_pass1/validate_backend.py` for local FastAPI and benchmark-function checks. TestClient is used without lifespan startup, so provider probes and startup warming are excluded; actual production codec calls still run.
7. Run `python research_pass1/generate_reports.py` to recreate the benchmark and results reports.

`python research_pass1/deployment_probe.py` performs public read-only metadata/source/status checks. Its `--encode` option uploads googlewiki.png to the public HF Space and must only be run with explicit permission for that transfer. The local benchmark and research experiments do not require that upload.

All generated results remain local to this directory and the three root reports. Re-running overwrites these research outputs. Source copies in `release/` are immutable baseline inputs; do not edit them. Worker processes import one runtime each and resolve its own policy by absolute path. Thread environment settings and logical CPU affinity are recorded in metadata.

The reports distinguish local observations from HF-reported revision/startup state. They do not infer deployed package versions, loaded module paths, or in-memory weights from repository contents alone.
