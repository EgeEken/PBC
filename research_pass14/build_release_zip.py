import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "dist"
PACKAGE = OUT / "PBC3_1_Release"
BASELINE = ROOT / "research_pass10" / "baseline_manifest.json"
RUNTIME = (
    "PBC3.py",
    "pbc3_ops.py",
    "pbc3_stream.py",
    "pbc3_heads.py",
    "pbc3_features.py",
    "pbc3_types.py",
    "pbc3_trace.py",
    "learned_filler.py",
)

README = """# PBC 3.1 standalone Python codec

This package contains the frozen PBC 3.1 release-RNG development baseline. It keeps the PBC3 v0 bitstream; `PBC3.VERSION == 0` is the stream version, while package metadata reports version 3.1.0.

## Install

Use Python 3.11 and install from this extracted directory:

```sh
python -m pip install .
```

The tested runtime is Python 3.11.16, NumPy 2.4.6, Numba 0.68.0, and Pillow 11.3.0. Those exact runtime dependencies are pinned in `requirements.txt` and `pyproject.toml`. LZMA uses Python's standard-library `lzma` module.

## Encode and decode

```sh
python examples/roundtrip.py input.jpg output.pbc preview.png
```

The API is also available as:

```python
from PIL import Image
from pbc3 import PBC3, PBC3Config

result = PBC3.compress(Image.open("input.jpg"), PBC3Config.balanced())
open("output.pbc", "wb").write(result.data)
preview = PBC3.decompress(result.data).image
preview.save("preview.png")
```

Presets are `compression`, `balanced`, `quality`, and `high_quality`. The default release RNG and final MSE calculation remain enabled. The encoder loads its bundled `patch_policy.npz` relative to the installed package, independent of the process working directory. The decoder does not load the model.

## Compatibility

The package version is PBC 3.1.0; encoded files remain PBC3 stream version 0. Existing v0 files are decoded by the Python decoder. Encoding was checked against the frozen Windows reference runtime, and the HF Linux runtime previously passed its own 100/100 release-equivalence cases. The encoder is deterministic within a validated runtime, but Pass 13 found that Windows and HF Linux do not produce identical output for every input. Cross-platform encoder bitstream identity is unverified. v0 decoding across operating systems is format-compatible; this package was smoke-tested on Windows, and a separate small Linux/HF decode smoke is recorded in the release manifest when available.

The package and its model are licensed under MIT, matching the repository's project metadata.
"""

CHANGELOG = """# PBC 3.1.0

- Packages the frozen release-RNG development baseline with its existing PBC3 v0 stream.
- Includes the runtime fixes already validated against PBC 3.0: vectorized grid I/O, owned per-encode search state, and selected-delta reuse.
- No compression decision, model, preset, palette, decoder arithmetic, or bitstream syntax is intentionally changed by this release package. The package-only model path resolves beside the installed module.
- The baseline matched PBC 3.0 on 100 HF main/holdout cases and measured 28.83% lower encode time and 64.38% lower decode time in the controlled HF comparison.
"""

LICENSE = """MIT License

Copyright (c) 2026 Ege Eken

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

PYPROJECT = """[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "pbc-codec"
version = "3.1.0"
description = "Probabilistic Brush Compression standalone codec"
readme = "README.md"
requires-python = ">=3.11,<3.12"
license = {file = "LICENSE"}
authors = [{name = "Ege Eken", email = "egeeken.tr@gmail.com"}]
dependencies = ["numpy==2.4.6", "numba==0.68.0", "Pillow==11.3.0"]

[tool.setuptools]
packages = ["pbc3"]

[tool.setuptools.package-data]
pbc3 = ["patch_policy.npz"]
"""

EXAMPLE = """import argparse
from pathlib import Path

from PIL import Image

from pbc3 import PBC3, PBC3Config

parser = argparse.ArgumentParser()
parser.add_argument("input")
parser.add_argument("compressed")
parser.add_argument("preview")
args = parser.parse_args()

result = PBC3.compress(Image.open(args.input), PBC3Config.balanced())
Path(args.compressed).write_bytes(result.data)
PBC3.decompress(result.data).image.save(args.preview)
print(f"{len(result.data)} bytes; internal MSE {result.mse:.4f}")
"""


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def package_source():
    manifest = json.loads(BASELINE.read_text(encoding="utf-8"))
    source_hashes = {name: sha(ROOT / name) for name in RUNTIME}
    assert source_hashes == manifest["source_sha256"], "runtime source differs from the frozen Pass 10 baseline"
    model_hash = sha(ROOT / "patch_policy.npz")
    assert model_hash == manifest["model_sha256"]
    if PACKAGE.exists():
        shutil.rmtree(PACKAGE)
    module_dir = PACKAGE / "pbc3"
    module_dir.mkdir(parents=True)
    for name in RUNTIME:
        text = (ROOT / name).read_text(encoding="utf-8")
        text = text.replace("import pbc3_stream as stream", "from . import pbc3_stream as stream")
        text = text.replace("import pbc3_ops as ops", "from . import pbc3_ops as ops")
        text = text.replace("from pbc3_ops import ", "from .pbc3_ops import ")
        text = text.replace("from pbc3_heads import ", "from .pbc3_heads import ")
        text = text.replace("from pbc3_types import ", "from .pbc3_types import ")
        text = text.replace("from pbc3_features import ", "from .pbc3_features import ")
        text = text.replace("from pbc3_trace import ", "from .pbc3_trace import ")
        text = text.replace("from learned_filler import ", "from .learned_filler import ")
        text = text.replace("from PBC3 import PBC3", "from .PBC3 import PBC3")
        if name == "pbc3_types.py":
            text = text.replace("from dataclasses import dataclass", "from dataclasses import dataclass, field\nfrom pathlib import Path")
            text = text.replace('learned_filler_model_path: str = "patch_policy.npz"', 'learned_filler_model_path: str = field(default_factory=lambda: str(Path(__file__).resolve().with_name("patch_policy.npz")))')
        (module_dir / name).write_text(text, encoding="utf-8", newline="\n")
    shutil.copy2(ROOT / "patch_policy.npz", module_dir / "patch_policy.npz")
    (module_dir / "__init__.py").write_text('__version__ = "3.1.0"\nfrom .PBC3 import PBC3\nfrom .pbc3_types import PBC3Config, PBC3Result\n\n__all__ = ["PBC3", "PBC3Config", "PBC3Result", "__version__"]\n', encoding="utf-8")
    (PACKAGE / "examples").mkdir()
    (PACKAGE / "examples" / "roundtrip.py").write_text(EXAMPLE, encoding="utf-8")
    (PACKAGE / "README.md").write_text(README, encoding="utf-8")
    (PACKAGE / "CHANGELOG.md").write_text(CHANGELOG, encoding="utf-8")
    (PACKAGE / "LICENSE").write_text(LICENSE, encoding="utf-8")
    (PACKAGE / "pyproject.toml").write_text(PYPROJECT, encoding="utf-8")
    (PACKAGE / "requirements.txt").write_text("numpy==2.4.6\nnumba==0.68.0\nPillow==11.3.0\n", encoding="utf-8")
    return manifest, source_hashes, model_hash


def build(validation_path=None):
    manifest, source_hashes, model_hash = package_source()
    payload = sorted(p for p in PACKAGE.rglob("*") if p.is_file())
    validation = json.loads(Path(validation_path).read_text(encoding="utf-8")) if validation_path else {"status": "pending"}
    metadata = {
        "package_version": "3.1.0",
        "stream_version": 0,
        "baseline_id": manifest["baseline_id"],
        "frozen_source_commit": manifest["github_source_revision"],
        "hf_baseline_commit": manifest["hf_revision"],
        "immutable_pbc30_commit": manifest["immutable_release_revision"],
        "source_sha256": source_hashes,
        "model_sha256": model_hash,
        "tested_python": "3.11.16 Windows; HF baseline is Python 3.11.17 Linux",
        "runtime_dependencies": ["numpy==2.4.6", "numba==0.68.0", "Pillow==11.3.0"],
        "standard_library": ["lzma"],
        "license": "MIT, as declared by repository pyproject metadata",
        "validation": validation,
        "limitations": ["Windows and HF Linux encoders do not produce matching streams for every case; cross-platform encoder identity is unverified.", "The package smoke test validates its extracted Windows install; no Rust code is included."],
        "files_sha256": {p.relative_to(PACKAGE).as_posix(): sha(p) for p in payload},
    }
    (PACKAGE / "PBC31_RELEASE_MANIFEST.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    zip_path = OUT / "PBC3_1_Release.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(PACKAGE.rglob("*")):
            if path.is_file():
                archive.write(path, (Path("PBC3_1_Release") / path.relative_to(PACKAGE)).as_posix())
    (OUT / "PBC3_1_Release.zip.sha256").write_text(f"{sha(zip_path)}  {zip_path.name}\n", encoding="ascii")
    print(f"{zip_path} {zip_path.stat().st_size} bytes SHA-256 {sha(zip_path)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-json")
    args = parser.parse_args()
    build(args.validation_json)
