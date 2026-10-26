import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = ROOT / "research_pass10" / "baseline_manifest.json"
MAIN_COMMIT = "162feed4a85b81f0d2494e05bd39d8e159788e7c"
PBC31_COMMIT = "8006f30debfc4a57289df06df21636aeb52baffd"
PBC30_COMMIT = "2e59f1b06980aa467da0cde94fa33c3f9a0d4d51"
HF_COMMIT = "98ade0c8072a4571e9674eaac3d91f861e3b9867"
RUNTIME = (
    "PBC3.py", "pbc3_ops.py", "pbc3_stream.py", "pbc3_heads.py",
    "pbc3_features.py", "pbc3_types.py", "pbc3_trace.py", "learned_filler.py",
)
BASELINE_HASH_NAMES = RUNTIME + ("patch_policy.npz",)

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

RELEASE_NOTES = """# PBC 3.1 — Performance Optimization Update

PBC 3.1 keeps the original PBC 3.0 RNG behavior, model, presets, and PBC3 v0 bitstream while reducing runtime through implementation changes. Existing PBC 3.0 v0 files remain decodable.

In a controlled HF cpu-basic comparison, encoding time was 28.83% lower and decoding time was 64.38% lower than PBC 3.0. These are measurements for that CPU and benchmark set, not universal speed guarantees. A separate warmed local comparison on an Intel Core i9-12900H measured 34.00% lower encoding time and 64.28% lower decoding time across eight images and four presets. The matched local files remained byte-identical.

The staged development candidate also uses fused Numba integral-image construction. In a three-repetition local four-image screening benchmark it reduced mean full-encode time by 11.05% while keeping every output byte-identical. It also matched the previous candidate on all 100 image/preset pairs across the 20-image main set and five-image holdout. These measurements are environment-specific; the full-set pass checked output identity, not a separate repeated speed estimate.

No rate-distortion improvement is claimed. PBC 3.1 does not change stream syntax or reconstructed pixels for the validated matched cases.

## Install

Use Python 3.11 with NumPy 2.4.6, Numba 0.68.0, and Pillow 11.3.0. From the extracted source directory run `python -m pip install .`. Python's standard-library `lzma` module handles the stream wrapper.

For examples and the cross-platform encoder reproducibility limit, see `README.md`. Windows and Linux may encode some identical inputs into different streams; universal cross-platform encoder identity has not been verified.
"""

INSTALL = """# PBC 3.1 installation

Supported and validated environment: Python 3.11, NumPy 2.4.6, Numba 0.68.0, and Pillow 11.3.0.

From this directory:

```sh
python -m pip install .
```

The model is installed beside the top-level codec modules. The encoder resolves it relative to `pbc3_types.py`, independent of the current working directory. LZMA uses Python's standard library.

The optional animation export script needs `imageio` and `imageio-ffmpeg`; install them with `python -m pip install .[animation]`.
"""

PYPROJECT = """[build-system]
requires = ["setuptools>=77"]
build-backend = "setuptools.build_meta"

[project]
name = "pbc-codec"
version = "3.1.0"
description = "Probabilistic Brush Compression image codec"
readme = "README.md"
requires-python = ">=3.11,<3.12"
license = "MIT"
authors = [{name = "Ege Eken"}]
dependencies = ["numpy==2.4.6", "numba==0.68.0", "Pillow==11.3.0"]

[project.optional-dependencies]
animation = ["imageio>=2.31", "imageio-ffmpeg>=0.4"]

[tool.setuptools]
packages = ["pbc3"]
py-modules = ["PBC3", "PBC3_animation", "pbc3_ops", "pbc3_stream", "pbc3_heads", "pbc3_features", "pbc3_types", "pbc3_trace", "learned_filler"]

[tool.setuptools.data-files]
"." = ["patch_policy.npz"]
"""

README_TOP = """# Probabilistic Brush Compression (PBC)

PBC 3.1 is a lossy, residual image codec. The encoder selects deterministic patch parameters using a learned policy; the decoder needs no model. PBC 3.1 preserves the PBC 3.0 compression behavior for validated inputs and keeps PBC3 stream version 0.

## Install and use

Validated runtime: Python 3.11, NumPy 2.4.6, Numba 0.68.0, and Pillow 11.3.0. Install with `python -m pip install .` from this directory. Optional animation export dependencies are available with `python -m pip install .[animation]`.

The established module imports continue to work:

```python
from PIL import Image
from PBC3 import PBC3
from pbc3_types import PBC3Config

result = PBC3.compress(Image.open("input.jpg"), PBC3Config.balanced())
result.save("output.pbc")
result.image.save("preview.png")
print(result.mse)
```

`from pbc3 import PBC3, PBC3Config` is also supported as a re-export of the same top-level implementation. The policy model is resolved relative to the installed codec modules, not the current working directory. The stream version is `PBC3.VERSION == 0`.

## PBC 3.1 performance

The optimized implementation preserves the PBC 3.0 RNG, model, presets, stream syntax, and output on the tested matched cases. In a controlled HF cpu-basic comparison, encode time was 28.83% lower and decode time was 64.38% lower. A separate local eight-image test on an Intel Core i9-12900H measured 34.00% and 64.28% reductions. The fused integral-image candidate measured 12.00% lower full-encode time than the frozen PBC 3.1 baseline in a repeated four-image screen and was output-identical across all 25 main and holdout images. These are benchmark-specific measurements, not universal guarantees. No new rate-distortion gain is claimed.

PBC 3.1 decodes existing PBC3 v0 files from PBC 3.0. Windows and Linux can produce different encoder streams for some inputs even with matching source, model, and input hashes. Same-runtime repeat encoding is deterministic in the validated environment; cross-platform encoder bitstream identity is unverified.

See `RELEASE_NOTES.md` and `INSTALL.md` for release and setup details. The original PBC 3.0 visual comparisons and version history follow.

"""


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha_file(path):
    return sha_bytes(path.read_bytes())


def git_main_file(path):
    return subprocess.check_output(["git", "show", f"{MAIN_COMMIT}:{path}"], cwd=ROOT)


def frozen_source_hashes(candidate=False):
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    if baseline["baseline_id"] != "pbc31_release_rng_baseline":
        raise AssertionError("unexpected release baseline")
    hashes = {name: sha_file(ROOT / name) for name in BASELINE_HASH_NAMES}
    expected = {name: baseline["source_sha256"].get(name, baseline["model_sha256"] if name == "patch_policy.npz" else None) for name in RUNTIME}
    expected["patch_policy.npz"] = baseline["model_sha256"]
    compare_hashes = hashes.copy()
    if candidate:
        compare_hashes.pop("pbc3_ops.py")
        expected.pop("pbc3_ops.py")
    if compare_hashes != expected:
        raise AssertionError("frozen PBC 3.1 source or model hash mismatch")
    return baseline, hashes


def make_readme(main_readme):
    demo_start = main_readme.index("### Demo deployed on")
    history_marker = "\n---\n# Development / Version History"
    gallery_end = main_readme.index(history_marker)
    gallery = main_readme[demo_start:gallery_end]
    gallery = gallery.replace("# Current version (V3.0) Demonstration", "# PBC 3.0 visual reference (PBC 3.1 keeps its stream output)")
    history = main_readme[gallery_end:]
    history = history.replace(
        "lossless encoding is not yet available (Intended feature for 3.1 or maybe 3.2 depending on the rate of progress)",
        "lossless encoding is not part of the PBC 3.1 performance release.",
    )
    return README_TOP + gallery + history


def write_source(source):
    source.mkdir(parents=True, exist_ok=False)
    for name in (".gitignore", "PBC3_animation.py"):
        (source / name).write_bytes(git_main_file(name))
    main_readme = git_main_file("README.md").decode("utf-8")
    (source / "README.md").write_text(make_readme(main_readme), encoding="utf-8", newline="\n")

    for name in RUNTIME:
        text = (ROOT / name).read_text(encoding="utf-8")
        if name == "PBC3.py":
            text = text.replace("PBC v3.0 - Probabilistic Brush Compression", "PBC v3.1 - Probabilistic Brush Compression")
        if name == "pbc3_types.py":
            text = text.replace("from dataclasses import dataclass", "from dataclasses import dataclass, field\nfrom pathlib import Path")
            text = text.replace('learned_filler_model_path: str = "patch_policy.npz"', 'learned_filler_model_path: str = field(default_factory=lambda: str(Path(__file__).resolve().with_name("patch_policy.npz")))')
        (source / name).write_text(text, encoding="utf-8", newline="\n")
    shutil.copy2(ROOT / "patch_policy.npz", source / "patch_policy.npz")
    package = source / "pbc3"
    package.mkdir()
    (package / "__init__.py").write_text(
        '__version__ = "3.1.0"\nfrom PBC3 import PBC3\nfrom pbc3_types import PBC3Config, PBC3Result\n\n__all__ = ["PBC3", "PBC3Config", "PBC3Result", "__version__"]\n',
        encoding="utf-8",
        newline="\n",
    )
    (source / "INSTALL.md").write_text(INSTALL, encoding="utf-8", newline="\n")
    (source / "RELEASE_NOTES.md").write_text(RELEASE_NOTES, encoding="utf-8", newline="\n")
    (source / "requirements.txt").write_text(
        "numpy==2.4.6\nnumba==0.68.0\nPillow==11.3.0\n", encoding="utf-8", newline="\n"
    )
    (source / "pyproject.toml").write_text(PYPROJECT, encoding="utf-8", newline="\n")
    (source / "LICENSE").write_text(LICENSE, encoding="utf-8", newline="\n")


def build(output, validation_path, candidate_report_path, full_correctness_path, local_comparison_path):
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing staging output: {output}")
    candidate = candidate_report_path is not None
    baseline, source_hashes = frozen_source_hashes(candidate)
    if candidate:
        candidate_report = json.loads(candidate_report_path.read_text(encoding="utf-8"))
        full_report = json.loads(full_correctness_path.read_text(encoding="utf-8"))
        if candidate_report["status"] != "passed" or not candidate_report["all_outputs_byte_identical"] or candidate_report["repetitions"] != 3:
            raise AssertionError("integral candidate has not passed validation")
        if full_report["status"] != "passed" or not full_report["all_outputs_byte_identical"] or full_report["paired_case_repetitions"] != 100:
            raise AssertionError("integral candidate has not passed full-set correctness")
    else:
        candidate_report = None
    output.mkdir(parents=True)
    source = output / "GitHub_Source"
    write_source(source)
    payload_hashes = {
        path.relative_to(source).as_posix(): sha_file(path)
        for path in sorted(source.rglob("*")) if path.is_file()
    }
    validation = json.loads(validation_path.read_text(encoding="utf-8")) if validation_path else {"status": "pending"}
    manifest = {
        "release": "PBC 3.1 — Performance Optimization Update",
        "package_version": "3.1.0",
        "stream_version": 0,
        "baseline_id": baseline["baseline_id"],
        "base_github_main_revision": MAIN_COMMIT,
        "frozen_development_source_revision": PBC31_COMMIT,
        "hf_measurement_revision": HF_COMMIT,
        "immutable_pbc30_revision": PBC30_COMMIT,
        "frozen_source_sha256": baseline["source_sha256"] | {"patch_policy.npz": baseline["model_sha256"]},
        "staged_source_sha256": source_hashes,
        "development_candidate_id": "pbc31_integral_fused" if candidate else "pbc31_release_rng_baseline",
        "source_files_sha256_excluding_manifest": payload_hashes,
        "model_sha256": baseline["model_sha256"],
        "runtime": {
            "python": "3.11.x",
            "dependencies": ["numpy==2.4.6", "numba==0.68.0", "Pillow==11.3.0"],
            "lzma": "Python standard library",
            "optional_animation": ["imageio", "imageio-ffmpeg"],
        },
        "validated_local_comparison": {
            "machine": "12th Gen Intel Core i9-12900H",
            "encode_time_reduction_percent": 34.00,
            "decode_time_reduction_percent": 64.28,
            "matched_streams": "96/96 byte-identical paired repetitions",
        },
        "controlled_hf_comparison": {
            "cpu": "cpu-basic",
            "encode_time_reduction_percent": 28.83,
            "decode_time_reduction_percent": 64.38,
            "matched_files": "100/100 verified cases",
        },
        "intentional_differences": [
            "pbc3_types.py resolves the default policy model next to its own module instead of relative to the process working directory.",
            "PBC3.py release identification comment and project metadata report PBC 3.1; the bitstream remains version 0.",
            "pbc3 package is a thin re-export of the established top-level API; codec implementation is not duplicated.",
            "The old PDM-specific instructions and pdm.lock were omitted because they described PBC 2.3.1/Python 3.12 and unrelated dependencies; INSTALL.md and pinned runtime dependencies replace them.",
        ],
        "limitations": [
            "Windows and Linux encoder streams are not guaranteed identical; same-runtime determinism is the validated claim.",
            "The benchmark figures are environment-specific and do not claim a rate-distortion improvement.",
        ],
        "validation": validation,
    }
    if candidate:
        manifest["validated_integral_candidate"] = {
            "experiment_report": "research_pass15/PBC_INTEGRAL_OPTIMIZATION.md",
            "screen_encode_reduction_percent": candidate_report["summary"]["overall"]["encode_reduction_percent"],
            "screen_identity": f"{candidate_report['paired_case_repetitions']}/{candidate_report['paired_case_repetitions']} paired cases, 3 repetitions each",
            "full_set_identity": f"{full_report['paired_case_repetitions']}/{full_report['paired_case_repetitions']} image/preset pairs across hpt_data and hpt_data_val",
        }
    if candidate and local_comparison_path:
        comparison = json.loads(local_comparison_path.read_text(encoding="utf-8"))
        if comparison["matched_stream_pairs"] != 96 or comparison["matched_pixel_pairs"] != 96:
            raise AssertionError("final local old-vs-candidate comparison is incomplete or not byte-identical")
        manifest["validated_local_comparison"] = {
            "machine": "12th Gen Intel Core i9-12900H",
            "dataset": "8 images x 4 presets x 3 repetitions",
            "encode_time_reduction_percent": 100 * (1 - comparison["summary"]["overall"]["encode"]["b_mean_case_median"] / comparison["summary"]["overall"]["encode"]["a_mean_case_median"]),
            "decode_time_reduction_percent": 100 * (1 - comparison["summary"]["overall"]["decode"]["b_mean_case_median"] / comparison["summary"]["overall"]["decode"]["a_mean_case_median"]),
            "matched_streams": "96/96 byte-identical paired repetitions",
        }
    manifest_bytes = (json.dumps(manifest, indent=2) + "\n").encode("utf-8")
    (source / "RELEASE_MANIFEST.json").write_bytes(manifest_bytes)
    (output / "RELEASE_MANIFEST.json").write_bytes(manifest_bytes)
    shutil.copy2(source / "RELEASE_NOTES.md", output / "RELEASE_NOTES.md")

    archive_path = output / "PBC3_1_Release.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(source.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(source).as_posix())
    sums = [f"{sha_file(archive_path)}  PBC3_1_Release.zip"]
    sums.extend(f"{sha_file(path)}  GitHub_Source/{path.relative_to(source).as_posix()}" for path in sorted(source.rglob("*")) if path.is_file())
    (output / "SHA256SUMS.txt").write_text("\n".join(sums) + "\n", encoding="ascii", newline="\n")
    print(f"staged {output}; zip sha256={sha_file(archive_path)} bytes={archive_path.stat().st_size}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-json", type=Path)
    parser.add_argument("--candidate-report", type=Path)
    parser.add_argument("--full-correctness", type=Path)
    parser.add_argument("--local-comparison", type=Path)
    args = parser.parse_args()
    if (args.candidate_report is None) != (args.full_correctness is None):
        parser.error("--candidate-report and --full-correctness must be used together")
    if args.candidate_report is not None and args.local_comparison is None:
        parser.error("--local-comparison is required for a promoted candidate")
    build(args.output, args.validation_json, args.candidate_report, args.full_correctness, args.local_comparison)
