import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ARCHIVE = ROOT / "dist" / "PBC3_1_Release.zip"
PYTHON = Path(sys.executable)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    with tempfile.TemporaryDirectory(prefix="pbc31-release-smoke-", dir=ROOT / "dist") as tmp:
        base = Path(tmp)
        extracted = base / "extracted"
        extracted.mkdir()
        with zipfile.ZipFile(ARCHIVE) as archive:
            archive.extractall(extracted)
        package = extracted / "PBC3_1_Release"
        site = base / "site"
        subprocess.run(
            [str(PYTHON), "-m", "pip", "install", "--no-deps", "--no-build-isolation", "--target", str(site), str(package)],
            check=True,
            cwd=base,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        child = base / "check_installed.py"
        child.write_text(
            """import hashlib, json, os, sys
from pathlib import Path
import numpy as np
from PIL import Image
import pbc3
from pbc3 import PBC3, PBC3Config

result = {"version": pbc3.__version__, "stream_version": PBC3.VERSION, "presets": {}, "modes": {}}
assert pbc3.__file__.startswith(os.environ["PBC_SMOKE_SITE"])
assert Path(PBC3Config.balanced().learned_filler_model_path).resolve() == (Path(pbc3.__file__).parent / "patch_policy.npz").resolve()
y, x = np.indices((24, 32))
rgb = np.stack(((x * 7) % 256, (y * 9) % 256, ((x + y) * 5) % 256), axis=-1).astype(np.uint8)
for name in ("compression", "balanced", "quality", "high_quality"):
    config = getattr(PBC3Config, name)(patch_count=1)
    encoded = PBC3.compress(Image.fromarray(rgb), config)
    decoded = PBC3.decompress(encoded.data)
    assert encoded.data[:4] == b"PBC3" and encoded.data[4] == 0
    assert decoded.image.size == (32, 24) and encoded.mse is not None
    result["presets"][name] = {"bytes": len(encoded.data), "mse": encoded.mse}
config = PBC3Config.quality(patch_count=2)
first = PBC3.compress(Image.fromarray(rgb), config).data
second = PBC3.compress(Image.fromarray(rgb), config).data
assert first == second
result["deterministic_same_runtime"] = True
reference = PBC3.compress(Image.open(os.environ["PBC_REFERENCE_IMAGE"]), PBC3Config.quality())
reference_hash = hashlib.sha256(reference.data).hexdigest()
assert reference_hash == os.environ["PBC_REFERENCE_SHA256"]
assert reference.mse is not None
result["frozen_reference"] = {"image": "cat.jpg", "preset": "quality", "stream_sha256": reference_hash, "matches": True}
rgba = np.concatenate((rgb, np.where(x[:, :, None] < 16, 0, 255).astype(np.uint8)), axis=2)
encoded_rgba = PBC3.compress(Image.fromarray(rgba), PBC3Config.quality(patch_count=1))
assert PBC3.decompress(encoded_rgba.data).image.mode == "RGBA"
result["modes"]["RGBA"] = {"bytes": len(encoded_rgba.data), "size": [32, 24]}
old_file = Path(os.environ["PBC_OLD_FIXTURE"])
old = PBC3.decompress(old_file.read_bytes())
assert old.image.size == (64, 48) and old_file.read_bytes()[4] == 0
result["legacy_v0"] = {"bytes": old_file.stat().st_size, "size": list(old.image.size), "pixel_sha256": hashlib.sha256(old.image.tobytes()).hexdigest()}
print(json.dumps(result, sort_keys=True))
""",
            encoding="utf-8",
        )
        env = os.environ.copy()
        env["PYTHONPATH"] = str(site)
        env["PBC_SMOKE_SITE"] = str(site)
        env["PBC_OLD_FIXTURE"] = str(HERE / "native_cases" / "old_v30_quality.pbc")
        env["PBC_REFERENCE_IMAGE"] = str(ROOT / "hpt_data" / "cat.jpg")
        env["PBC_REFERENCE_SHA256"] = "a4e1756ebae5e0ea61cf1dc9601565a6728a80a3388b24b09c3aed569110ac27"
        result = subprocess.run(
            [str(PYTHON), str(child)], check=True, cwd=base, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        smoke = json.loads(result.stdout.splitlines()[-1])
        contents = sorted(zipfile.ZipFile(ARCHIVE).namelist())
        assert not any("server.py" in name or "benchmark.db" in name or "hpt_data" in name for name in contents)
        return {
            "status": "passed",
            "python": sys.version.split()[0],
            "install": "pip install --no-deps --no-build-isolation from extracted ZIP",
            "working_directory": "temporary directory under ignored dist, separate from the codec module directory",
            "zip_sha256": sha(ARCHIVE.read_bytes()),
            "archive_files": contents,
            "smoke": smoke,
        }


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
