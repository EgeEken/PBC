# Third-party notices

PBC source is under LICENSE (MIT). Dependency and runtime code retains its
respective terms; PBC does not relicense it.

The Windows application includes CPython, Tcl/Tk, NumPy/OpenBLAS, Numba/LLVM,
Pillow and PyInstaller runtime components. The optional native metric DLL is
compiled with MinGW. Retained license texts, third-party notices, copyright
notices and source references are in licenses/. In the standalone application
these are under _internal/licenses/.

This program includes software developed by the OpenSSL Project and code from
XZ Utils. Pillow's license file includes its bundled-library terms, including
FreeType. Python's complete Windows binary conditions and third-party notices
are retained. Microsoft runtime DLLs retain their original filenames and remain
subject to their applicable Windows/Visual C++ terms. No Microsoft endorsement
is implied. This distribution targets Windows x64.

The Python library contains PBC source, the policy model and the optional Windows
native metric DLL and source. pip installs its declared numerical dependencies.
No Python interpreter or generated JIT cache is included in Python/.
