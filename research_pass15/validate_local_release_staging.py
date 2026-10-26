import base64
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PYTHON = Path(sys.executable)
THREADS = {key: "1" for key in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
)}
REFERENCE_SHA256 = "a4e1756ebae5e0ea61cf1dc9601565a6728a80a3388b24b09c3aed569110ac27"
OLD_COMMIT = "2e59f1b06980aa467da0cde94fa33c3f9a0d4d51"

SMOKE = r'''import hashlib, importlib, inspect, json, os
from pathlib import Path
import numpy as np
from PIL import Image
import PBC3 as legacy_module
import pbc3
from PBC3 import PBC3
from pbc3_types import PBC3Config

root=Path(os.environ["PBC_EXPECT_ROOT"]).resolve()
assert Path(legacy_module.__file__).resolve().parent == root
assert PBC3 is pbc3.PBC3
assert PBC3Config is pbc3.PBC3Config
assert PBC3.VERSION == 0
assert inspect.signature(PBC3.compress).parameters["rng_mode"].default == "release"
assert pbc3.__version__ == "3.1.0"
modules=("pbc3_ops","pbc3_stream","pbc3_heads","pbc3_features","pbc3_types","pbc3_trace","learned_filler")
for name in modules:
    module=importlib.import_module(name)
    assert Path(module.__file__).resolve().parent == root
model=Path(PBC3Config.balanced().learned_filler_model_path).resolve()
assert model == (root/"patch_policy.npz").resolve() and model.is_file()
y,x=np.indices((32,40))
rgb=np.stack(((x*7)%256,(y*11)%256,((x+y)*5)%256),axis=-1).astype(np.uint8)
checks={"presets":{},"modules":list(modules),"model_path":str(model)}
for preset in ("compression","balanced","quality","high_quality"):
    config=getattr(PBC3Config,preset)()
    assert config.random_seed == 2003 and config.compute_final_mse
    encoded=PBC3.compress(Image.fromarray(rgb),config)
    decoded=PBC3.decompress(encoded.data)
    assert encoded.data[:4] == b"PBC3" and encoded.data[4] == 0
    assert decoded.image.size == (40,32) and encoded.mse is not None
    checks["presets"][preset]={"bytes":len(encoded.data),"mse":float(encoded.mse)}
quality=PBC3Config.quality()
first=PBC3.compress(Image.fromarray(rgb),quality).data
second=PBC3.compress(Image.fromarray(rgb),quality).data
assert first == second
checks["same_runtime_determinism"]=True
rgba=np.concatenate((rgb,np.where(x[:,:,None]<20,0,255).astype(np.uint8)),axis=2)
encoded_rgba=PBC3.compress(Image.fromarray(rgba),PBC3Config.quality())
decoded_rgba=PBC3.decompress(encoded_rgba.data).image
assert decoded_rgba.mode == "RGBA" and decoded_rgba.size == (40,32)
checks["rgba"]={"bytes":len(encoded_rgba.data),"mode":decoded_rgba.mode}
reference=PBC3.compress(Image.open(os.environ["PBC_REFERENCE_IMAGE"]),PBC3Config.quality())
reference_hash=hashlib.sha256(reference.data).hexdigest()
assert reference_hash == os.environ["PBC_REFERENCE_SHA256"] and reference.mse is not None
checks["frozen_reference_sha256"]=reference_hash
legacy_path=Path(os.environ["PBC_OLD_FIXTURE"])
legacy=PBC3.decompress(legacy_path.read_bytes())
assert legacy_path.read_bytes()[4] == 0 and legacy.image.size == (64,48)
legacy_rgba=np.asarray(legacy.image.convert("RGBA"))
checks["old_v30_v0"]={"size":list(legacy.image.size),"pixels_sha256":hashlib.sha256(legacy_rgba.tobytes()).hexdigest()}
print(json.dumps(checks,sort_keys=True))
'''


def sha(data):
    return hashlib.sha256(data).hexdigest()


def run_smoke(root, cwd, env):
    result = subprocess.run(
        [str(PYTHON), "-c", SMOKE], cwd=cwd, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=True,
    )
    return json.loads(result.stdout.splitlines()[-1])


def main():
    parser = __import__("argparse").ArgumentParser()
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--reference-image", type=Path, required=True)
    args = parser.parse_args()
    staging = args.staging.resolve()
    source = staging / "GitHub_Source"
    archive_path = staging / "PBC3_1_Release.zip"
    old_source = ROOT / "research_pass1" / "release"
    old_fixture = ROOT / "research_pass14" / "native_cases" / "old_v30_quality.pbc"
    reference_image = args.reference_image.resolve()
    assert archive_path.is_file() and source.is_dir()
    assert all(path.is_file() for path in (old_fixture, reference_image))

    with tempfile.TemporaryDirectory(prefix="pbc31-stage-check-", dir=staging) as temp:
        temp = Path(temp)
        extracted = temp / "unzipped"
        extracted.mkdir()
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(extracted)
            names = set(archive.namelist())
        assert "PBC3.py" in names and "pbc3/__init__.py" in names
        assert not any("research_pass" in name or "hpt_data" in name or "server.py" in name or name.endswith(".db") for name in names)
        for relative in names:
            assert (source / relative).read_bytes() == (extracted / relative).read_bytes()
        model_hash = sha((extracted / "patch_policy.npz").read_bytes())
        assert model_hash == "ffaf24aa5447b3602c53a5c4c8df14a68ee16e6bded39331d973f38d0f1ea62f"

        env = os.environ.copy()
        env.update(THREADS)
        env.update({
            "PYTHONPATH": str(extracted),
            "PBC_EXPECT_ROOT": str(extracted),
            "PBC_REFERENCE_IMAGE": str(reference_image),
            "PBC_REFERENCE_SHA256": REFERENCE_SHA256,
            "PBC_OLD_FIXTURE": str(old_fixture),
        })
        installed_source = temp / "source-run"
        installed_source.mkdir()
        source_results = run_smoke(extracted, installed_source, env)

        target = temp / "site"
        subprocess.run(
            [str(PYTHON), "-m", "pip", "install", "--no-deps", "--no-build-isolation", "--target", str(target), str(extracted)],
            cwd=temp, env={**env, "PYTHONPATH": ""}, check=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        installed = temp / "installed-run"
        installed.mkdir()
        installed_env = env.copy()
        installed_env["PYTHONPATH"] = str(target)
        installed_env["PBC_EXPECT_ROOT"] = str(target)
        installed_results = run_smoke(target, installed, installed_env)
        source_semantics = {key: value for key, value in source_results.items() if key != "model_path"}
        installed_semantics = {key: value for key, value in installed_results.items() if key != "model_path"}
        assert installed_semantics == source_semantics

        old_env = os.environ.copy()
        old_env.update(THREADS)
        old_env["PYTHONPATH"] = str(old_source)
        old_env["PBC_OLD_FIXTURE"] = str(old_fixture)
        old_run = temp / "old-run"
        old_run.mkdir()
        code = (
            "import hashlib,os; from PBC3 import PBC3; "
            "im=PBC3.decompress(open(os.environ['PBC_OLD_FIXTURE'],'rb').read()).image.convert('RGBA'); "
            "print(hashlib.sha256(im.tobytes()).hexdigest())"
        )
        old_pixels_hash = subprocess.check_output([str(PYTHON), "-c", code], cwd=old_run, env=old_env, text=True).strip()
        assert old_pixels_hash == installed_results["old_v30_v0"]["pixels_sha256"]

    validation = {
        "status": "passed",
        "python": sys.version.split()[0],
        "checks": {
            "flat_github_source_imports": "passed",
            "pip_install_from_extracted_zip": "passed",
            "installed_flat_and_package_api_same_codec_object": "passed",
            "all_runtime_modules_import": "passed",
            "model_loaded_relative_to_modules": "passed",
            "all_four_default_presets_encode_decode_with_mse": "passed",
            "release_rng_default_seed_2003": "passed",
            "rgb_rgba_and_same_runtime_determinism": "passed",
            "pbc30_v0_matches_immutable_30_decoder_pixels": "passed",
            "frozen_cat_quality_stream_sha256": REFERENCE_SHA256,
            "stream_version": 0,
            "no_research_data_or_server_in_archive": "passed",
        },
        "model_sha256": model_hash,
        "frozen_reference_image": "cat.jpg",
        "old_v30_pixels_sha256": old_pixels_hash,
        "source_and_zip_runtime_files_identical": True,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
