# V3.1 release preparation

The Windows GUI, standalone packaging scripts and unchanged Python library are retained here. Work targets github/research/experiments. Public main, tags, releases and HF deployment require separate authorization.

The 25-image campaign is complete. BENCHMARK.md and provenance.json document the saved evidence. Do not rerun benchmark.py for a GUI pass. README-main.patch and public_readme/README.md retain the earlier public-main documentation proposal; the final correction passes did not change it or regenerate charts.

Use Windows x64, Python 3.11 and ../research_pass27/requirements-build.txt. Runtime dependencies are pinned in source/pyproject.toml. Model/DLL assets, fixtures, historical source snapshots, datasets and environments are ignored. Their identities are recorded in library-source.json, provenance.json and ../research/assets.json. prepare.py verifies the prior frozen snapshot and can restore ignored model/native resources; it is an initial preparation tool, not a routine GUI build step.

Current checks and build:

1. Run test_gui.py, test_final_gui.py and layout_check.py with the local saved fixtures available.
2. Run build.py --final and build.py --probe, then verify_payload.py --final. The production output is artifacts/dist-final/PBC/. The tracked hooks/ directory contains the Numba collection hook.
3. The internal probe supports the saved frozen-fixtures.json check, --features <release_prep_v31>, and --latency <release_prep_v31>. Run it with PATH=C:\Windows\System32 and a controlled NUMBA_CACHE_DIR. No installed Python/compiler is required at runtime.
4. profile_ui.py after measures three warmed Cat/Flower operations. profile_ui.py before uses the locally preserved artifacts/resume-finalization/baseline/ copy. UI_TIMINGS.json retains the compact before/after evidence; raw rows remain local.
5. finalize_gui.py checks saved validations and protected identities, creates a clean artifacts/staged-final/ directory, verifies every ZIP member, then replaces the final ZIP. It deliberately refuses an existing staging directory or an unexpected current archive. Earlier finalize.py/package_ui.py are historical preparation scripts.

VALIDATION.md records the final archive, rollback, measured latency, completed checks and limitations. Large artifacts and screenshots remain under artifacts/. No development tools or test evidence are included in the public ZIP.
