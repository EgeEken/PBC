# Pass 8 reproduction

Use the existing `requirements-research.txt` environment locally and the unchanged HF Docker environment remotely. Restore declared inputs with `research/prepare_assets.py`; do not commit them.

`check_support.py` exercises independently imported immutable-release/current sources on one image/four presets and rejects invalid API specifications. `test_selector_race.cjs` verifies stale-response rejection, stable options during polling and the chart timing filter. The Pass 7 SQLite fixture remains `research_pass7/test_run_selection.py`.

`prepare_deployment.py`, `run_hf.py deploy`, `prepare_ui_deployment.py`, and `deploy_ui.py` are operational scripts: they create commits/push/restart the HF Space and require explicit authorization. They preserve the independent HF ancestry through isolated indexes. The first deployment exports and validates the live database, adds the six unchanged release modules and allowlisted worker, and deploys the existing run selector. The second changes only its polling behavior and source manifest; it refuses to run before the full comparison has been saved locally.

The authenticated start request accepts only the 20 fixed main image names, four named presets, both fixed sources and 1–3 repeats, at most 480 calls. It accepts no paths, code, overrides or alternative modules. The existing token and benchmark lock remain required; no secret is logged, saved, rotated or committed. `run_hf.py benchmark` submits exactly the requested full workload. `verify_ui_api.py` makes read-only API/database requests and confirms unauthenticated research rejection.

`analyze_comparison.py` reduces saved `hf_comparison.json` to per-case medians, equal-weight tables, timing samples and per-image results. Full remote responses, databases, streams, caches and private job directories stay ignored. The measured result retains its original deployment manifest even if a later UI-only deployment updates the active manifest.

The separate palette experiment replays existing saved current streams with `palette_feasibility.py --archive PATH_TO_RESEARCH_PASS1_STREAMS_CURRENT_HPT_DATA`. It verifies their hashes against Pass 7 controls and performs no proposal search. Its fixed gate permits `screen_palette.py` for one frozen 98th-percentile rule; `analyze_palette.py` reads the saved local screen. This rule failed completed-file screening and is not deployed. No further percentile or broader test is implied.
