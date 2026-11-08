import hashlib
import json
import pathlib
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "research_pass45" / "outputs"
ARCHIVE = OUT / "PBC3_1_Release_published.zip"
DEST = OUT / "published_release_python"
PREFIX = "PBC3_1_Release/Python/"

with zipfile.ZipFile(ARCHIVE) as z:
    for info in z.infolist():
        if not info.filename.startswith(PREFIX) or info.is_dir():
            continue
        relative = pathlib.PurePosixPath(info.filename[len(PREFIX):])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe archive path: {info.filename}")
        target = DEST.joinpath(*relative.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        data = z.read(info)
        target.write_bytes(data)

hashes = {str(p.relative_to(DEST)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest() for p in DEST.rglob("*") if p.is_file()}
(OUT / "published_release_python_hashes.json").write_text(json.dumps({"source_asset": "PBC3_1_Release_published.zip", "extracted_root": str(DEST.relative_to(ROOT)), "file_hashes": hashes}, indent=2) + "\n")
print(json.dumps({"files": len(hashes), "root": str(DEST), "model_sha256": hashes.get("pbc3/patch_policy.npz"), "pbc3_sha256": hashes.get("PBC3.py")}, indent=2))
