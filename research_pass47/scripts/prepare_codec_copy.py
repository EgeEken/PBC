import hashlib
import json
import pathlib
import shutil

ROOT = pathlib.Path(__file__).resolve().parents[2]
PASS = pathlib.Path(__file__).resolve().parents[1]
SOURCE = ROOT / "research_pass45" / "outputs" / "published_release_python"
DEST = PASS / "codec"
MODEL = ROOT / "patch_policy.npz"
EXPECTED = "ffaf24aa5447b3602c53a5c4c8df14a68ee16e6bded39331d973f38d0f1ea62f"


def main():
    if hashlib.sha256(MODEL.read_bytes()).hexdigest() != EXPECTED:
        raise RuntimeError("canonical model hash mismatch")
    DEST.mkdir(parents=True, exist_ok=True)
    source_files = sorted(SOURCE.glob("*.py"))
    for path in source_files:
        shutil.copy2(path, DEST / path.name)
    shutil.copytree(SOURCE / "pbc3", DEST / "pbc3", dirs_exist_ok=True)
    shutil.copy2(MODEL, DEST / MODEL.name)
    manifest = {
        "source_directory": str(SOURCE.relative_to(ROOT)),
        "model_source": str(MODEL.relative_to(ROOT)),
        "model_sha256": EXPECTED,
        "copied_python_files": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files
        },
        "instrumentation": "runtime-only monkeypatches in run_oracle.py; copied codec source is unchanged",
    }
    (PASS / "CODEC_COPY_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
