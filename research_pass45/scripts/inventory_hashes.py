import hashlib
import json
import pathlib
import subprocess
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "research_pass45" / "outputs"
ZIP = OUT / "PBC3_1_Release_published.zip"
REVISIONS = {
    "release_tag": "ac7566f8b4fe77020bc27ff720fbe9942040cca4",
    "release_source_manifest": "8006f30debfc4a57289df06df21636aeb52baffd",
    "hf_head": "98ade0c8072a4571e9674eaac3d91f861e3b9867",
    "research_pass44_reference": "3723dc14de88c6c3dcbd98dd80a097e019a92a9f",
}
SOURCES = ["PBC3.py", "pbc3_types.py", "pbc3_ops.py", "pbc3_heads.py", "pbc3_stream.py", "pbc3_features.py", "learned_filler.py", "patch_policy.npz"]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def git_show(revision, path):
    p = subprocess.run(["git", "show", f"{revision}:{path}"], cwd=ROOT, capture_output=True)
    return None if p.returncode else {"sha256": sha(p.stdout), "bytes": len(p.stdout)}


def main():
    rows = {name: {path: git_show(rev, path) for path in SOURCES} for name, rev in REVISIONS.items()}
    with zipfile.ZipFile(ZIP) as archive:
        names = archive.namelist()
        prefix = "PBC3_1_Release/Python/"
        package = {name[len(prefix):]: sha(archive.read(name)) for name in names if name.startswith(prefix) and not name.endswith("/")}
        relevant = {path: package.get("pbc3/patch_policy.npz" if path == "patch_policy.npz" else path) for path in SOURCES}
        inventory = [{"path": name, "bytes": archive.getinfo(name).file_size} for name in names if not name.endswith("/") and (name.endswith(".json") or name.endswith("PBC.exe") or name.endswith("patch_policy.npz") or name.startswith(prefix))]
    local_zip = ROOT / "dist" / "PBC3_1_Release.zip"
    result = {
        "public_asset": {"path": str(ZIP.relative_to(ROOT)), "bytes": ZIP.stat().st_size, "sha256": sha(ZIP.read_bytes()), "published_asset_sha256_expected": "3ee8f68b45adb3a89128251b8905d93cb99a96d911a055f91e124e724abbe069", "relevant_python_source_hashes": relevant, "files": inventory},
        "local_dist_zip": {"path": str(local_zip.relative_to(ROOT)), "bytes": local_zip.stat().st_size, "sha256": sha(local_zip.read_bytes())},
        "git_revisions": rows,
    }
    (OUT / "implementation_hash_inventory.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"public_asset_sha256": result["public_asset"]["sha256"], "local_dist_sha256": result["local_dist_zip"]["sha256"], "revisions": rows, "public_python_hashes": relevant}, indent=2))


if __name__ == "__main__":
    main()
