import argparse
import os
import shutil
import subprocess
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--avif-build", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ctrl-aom", type=Path, required=True)
    ap.add_argument("--exp-aom", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    build = args.avif_build.resolve()
    profile_aom = (args.repo / "research_pass35/install/profile/lib/libaom.a").resolve().as_posix()
    for variant, archive in (("ctrl", args.ctrl_aom), ("exp", args.exp_aom)):
        dest = args.out / variant
        dest.mkdir(parents=True, exist_ok=True)
        for tool in ("avifenc", "avifdec"):
            commands = subprocess.check_output(["ninja", "-C", str(build), "-t", "commands", tool], text=True).splitlines()
            line = next((s for s in reversed(commands) if f"-o {tool}.exe" in s), None)
            if line is None:
                raise RuntimeError(f"link command not found for {tool}")
            prefix = ' /C "'
            if prefix not in line or not line.endswith('"'):
                raise RuntimeError(f"unexpected generated command wrapper: {line[:100]}")
            body = line.split(prefix, 1)[1][:-1]
            body = body.replace(profile_aom, archive.resolve().as_posix())
            exe = (dest / f"{tool}.exe").resolve().as_posix()
            importlib = (dest / f"lib{tool}.dll.a").resolve().as_posix()
            body = body.replace(f"-o {tool}.exe", f"-o {exe}")
            body = body.replace(f"--out-implib,lib{tool}.dll.a", f"--out-implib,{importlib}")
            log = dest / f"{tool}_link.log"
            with open(log, "w", encoding="utf-8") as f:
                proc = subprocess.run(["cmd.exe", "/D", "/S", "/C", body], cwd=build, stdout=f, stderr=subprocess.STDOUT)
            if proc.returncode:
                raise RuntimeError(f"{tool} {variant} link failed; see {log}")
        runtime = args.repo / "research_pass35/outputs/binaries/reference"
        for dll in ("libgcc_s_seh-1.dll", "libwinpthread-1.dll"):
            shutil.copy2(runtime / dll, dest / dll)

if __name__ == "__main__":
    main()
