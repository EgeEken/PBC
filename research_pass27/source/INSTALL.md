# Python installation

Use 64-bit Python3.11. From this complete candidate source directory:

```
python -m pip install .
```

Or install the built wheel with `python -m pip install pbc_codec-3.2.0rc1-py3-none-any.whl`.
The wheel includes the policy model inside pbc3 next to installed modules and does not depend on the current directory. It has NumPy2.4.6, Numba0.68.0, llvmlite0.50.0 and Pillow11.3.0 runtime requirements (llvmlite is Numba's pinned dependency). No demo/training packages are required.

The standalone decoder has only NumPy/Pillow dependencies. To test the Python decode-only module without the full encoder dependencies, preinstall those two packages and install the wheel with --no-deps; then import pbc3_decoder directly. Importing PBC3/pbc3 selects the complete encoder and requires Numba.

Native C is optional. Ordinary source/wheel installs work without a compiler or DLL. Windows app builds include the precompiled DLL; source developers may compile the retained mse.c using the research build script. No JIT caches are distributed.
