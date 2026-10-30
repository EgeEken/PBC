# Third-party components

PBC code and its unchanged policy model are distributed under LICENSE (MIT).
Dependencies retain their own licenses; the PBC license does not replace them.

Windows applications contain CPython 3.11, NumPy 2.4.6 and Pillow 11.3.0.
The combined application also contains Numba 0.68.0, llvmlite 0.50.0 and the
optional PBC native MSE DLL. The decoder excludes Numba, LLVM, the model and DLL.
PyInstaller 6.16.0 supplies the unmodified bootloader and runtime hooks.

The licenses directory retains the complete license trees from these installed
packages, including bundled OpenBLAS/GCC runtime terms in NumPy, LLVM notices in
llvmlite, and Pillow's bundled image/font/compression library notices. PyInstaller
COPYING includes its bootloader linking exception and runtime-hook terms.
CPYTHON_THIRD_PARTY.rst supplements the installed PYTHON_LICENSE.txt for expat,
libffi, libmpdec, OpenSSL and other CPython components. OpenSSL 3 uses Apache 2.0.
XZ/liblzma and MinGW notices are retained separately. Extra notices do not imply
that every optional upstream component is used by PBC.

Keep LICENSE, this document and licenses together when redistributing Windows
binaries. The single-file decoder embeds them in its temporary extraction and is
also supplied with a companion notices archive. These programs include software
developed by the OpenSSL Project and code from XZ Utils. FreeType is distributed
under its FreeType License option; the Pillow license file contains its text.

Microsoft runtime DLLs and API-set/UCRT files retain their original names and
notices. Their applicable Windows/Visual C++ terms are separate from PBC's MIT
license. The installed Python license's Additional Conditions for this Windows
binary build are retained in full, including its downstream distribution
conditions. These binaries are for Microsoft Windows only. Do not remove or alter
Microsoft notices, imply Microsoft endorsement, or reuse the redistributable
components for another platform or unlawful software.

Python wheel/source packages contain PBC source, model and optional native C
source; pip installs their numerical dependencies separately. The wheel contains
no native DLL, CPython runtime or generated JIT cache.

Dependency source/license URLs and retained-text hashes appear in
licenses/sources.json and licenses/installed-sources.json. This inventory is a
release engineering audit, not a replacement license grant or legal opinion.
