# PBC 3.2.0rc1

Local release candidate; not an official release. PBC3 v0 compression, presets and the policy model are unchanged from the published 3.1 asset.

Windows users: extract the directory ZIP, keep pbc.exe and _internal together, and open a terminal in that folder. No Python or compiler installation is required.

```
.\pbc.exe encode "input image.png" output.pbc --preset quality
.\pbc.exe decode output.pbc decoded.png
.\pbc.exe inspect output.pbc
.\pbc.exe --version
```

Balanced is default. Presets: compression, balanced, quality, high_quality. Add --force to replace a file. PNG preserves RGBA. Input/output paths may contain spaces or Unicode. Errors return1; invalid arguments return2. Decoder-only use: `.\pbc-decode.exe input.pbc output.png` or `.\pbc-decode.exe inspect input.pbc`.

The first encode compiles Numba kernels. A writable per-user cache is used outside the application directory; later commands reuse it. No machine-specific JIT caches are shipped. Decode/inspect do not import Numba, LLVM, the learned model or proposal/feature code. Temporary extraction is needed only for a one-file executable.

Python imports remain `from PBC3 import PBC3`, `from pbc3_types import PBC3Config`, and `from pbc3 import PBC3, PBC3Config`. Default compression is unchanged. The metric defaults to auto (native when the optional DLL is present, otherwise reference). Select `mse_backend="native"` explicitly for a separately built/bundled optional Windows DLL; `reference` retains the release metric and `auto` uses the DLL when present. Backends are selected per call, not by global patching. Non-byte metric inputs retain the reference behavior. Standard wheels are portable Python wheels with no DLL; the native component is included only in the Windows application.

Limits retained from Pass26: 32M pixels,64MiB compressed input,128MiB decompressed body,10k patches. These may reject unusually large valid files. Streams must have valid geometry, complete entropy framing and zero padding. No formal security audit or clean-machine Windows certification is claimed.

See INSTALL.md for Python installation. This candidate is for broader release testing; no release/tag/HF update is authorized.
