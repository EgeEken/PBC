# Rebuild the acceptance candidate

Use Windows x64, a writable folder, 64-bit Python 3.11 and GCC targeting
x86_64-w64-mingw32 (canonical build: GCC 13.2.0 UCRT). Do not use the repository's
historical root pyproject. Start in this complete release-source directory.

```
python -m venv .build
.build/Scripts/python.exe -m pip install -r tools/requirements-build.txt
python -m venv .decoder-build
.decoder-build/Scripts/python.exe -m pip install -r tools/requirements-decoder-build.txt
gcc -O3 -std=c11 -Wall -Wextra -Werror -shared -static-libgcc pbc3/mse.c -o pbc3/mse.dll
.build/Scripts/python.exe tools/build_windows.py --kind combined --output dist
.build/Scripts/python.exe tools/build_windows.py --kind decoder --decoder-python .decoder-build/Scripts/python.exe --output dist
.build/Scripts/python.exe tools/build_windows.py --kind decoder --onefile --decoder-python .decoder-build/Scripts/python.exe --output dist
.build/Scripts/python.exe -m build --no-isolation --outdir dist .
```

The decoder build verifies Numba and llvmlite are unavailable in its environment.
The DLL stays out of the pure Python wheel and sdist. IMPLEMENTATION.json pins
all codec source/model inputs common to the wheel and applications. Build tools
are retained here; no earlier research directory or published ZIP is needed to
rebuild this snapshot. Dependency license texts are retained from the canonical
build; verify them when changing dependencies or Python binaries.

PE binaries/ZIP timestamps and build paths can change hashes. Reproduction means
verified common source, model, installed package payloads and output behavior;
bit-for-bit PE reproducibility is not promised. Build native only for baseline
x64 (no -march=native). The first encode compiles Numba for the user's CPU;
compiled caches must never be included in a release.
