# Reproduce Pass 28

The final build/independent Windows source is d59fd3c20a5284b5cc441dc8275b395d54145012
on research/experiments. Later commits add reports and local audit helpers.
Do not use historical main/tag packaging as the 3.1 reference. The complete final
source ZIP and BUILD.md rebuild without this repository or the published asset.

For research-checkout reproduction, use Windows x64/Python 3.11, pinned
research_pass27/requirements-build.txt and GCC UCRT 13.2.0. Prepare a separate
Numba-free environment from research_pass27/requirements-decoder-build.txt.
Full-image tests require hpt_data restored with research/prepare_assets.py and the
preserved Pass 27 corpus; small fixed fixtures are already in smoke/fixtures.json.

```
python research_pass28/prepare.py
gcc -O3 -std=c11 -Wall -Wextra -Werror -shared -static-libgcc research_pass28/artifacts/final-source/pbc3/mse.c -o research_pass28/artifacts/final-source/pbc3/mse.dll
python research_pass28/runtime_check.py
python research_pass28/artifacts/final-source/tools/build_windows.py --kind combined --output research_pass28/artifacts/final
python research_pass28/artifacts/final-source/tools/build_windows.py --kind decoder --decoder-python PATH_TO_NUMBA_FREE_PYTHON --output research_pass28/artifacts/final
python research_pass28/artifacts/final-source/tools/build_windows.py --kind decoder --onefile --decoder-python PATH_TO_NUMBA_FREE_PYTHON --output research_pass28/artifacts/final
python -m build --no-isolation --outdir research_pass28/artifacts/final research_pass28/artifacts/final-source
python research_pass28/install_check.py
PATH_TO_NUMBA_FREE_PYTHON research_pass28/compatibility.py
python research_pass28/assemble.py
python research_pass28/benchmark.py
python research_pass28/measure_cli.py
python research_pass28/memory.py
python research_pass28/memory_cli.py
python research_pass28/permissions.py
python research_pass28/snapshot_check.py
```

Run smoke/Smoke-Test.ps1 separately using smoke/README.md, the generated final
manifest and final three application files. Run report.py only after all current
local checks and the recorded independent CI results are available. It verifies
frozen artifacts, common implementation, source tree, protected refs and hashes,
then generates the required acceptance/readiness reports and flat publication
staging in artifacts/final. Independent evidence is read from GitHub job logs;
never infer success from workflow submission alone.

The performance helpers reuse Pass 27's controlled worker conventions: logical
CPU 12, all numerical/Numba threads one, one warmup per image/preset/variant and
three rotated measurements. This affinity is specific to the recorded i9 laptop;
choose and document an available core explicitly on another CPU. Do not compare
absolute times across hosts. CLI timing uses blocking subprocess waits without
memory polling; memory polling is separate and its timings are discarded. Windows venv
python.exe is a launcher: external memory measurements must include its Python
child. memory_cli.py measures concurrent process-tree RSS for that reason.

CI: .github/workflows/pass28-windows.yml is restricted to source/test pushes on
research/experiments, with Windows 2022/2025 and read-only repository permissions.
It builds with Python 3.11, runs published/reference/native comparisons and fresh
source/wheel installs, then switches runtime PATH to System32 for the Python-free
smoke script. It never uploads binaries, creates releases/tags or touches main.
Hosted development tools remain physically installed. Full results and exact image
versions are in independent-windows.json; corrected-RC1 acceptance precedes stable
promotion in accepted-rc1-windows.json.

All generated apps, DLLs, streams, PNGs, environments, logs, caches, source object
store and raw measurement rows stay in ignored artifacts/. acceptance-evidence.json
records raw absolute paths/bytes/hashes. Binary build hashes vary with paths and
timestamps; the exact prepared source tree and payload behavior are authoritative.
