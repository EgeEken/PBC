import csv
import hashlib
import pathlib
import subprocess

from benchmark import BIN, CONFIGS, ENV, ROOT, verify_assets

OUT = ROOT / "research_pass39" / "outputs" / "qmap"
DIAG = ROOT / "research_pass39" / "outputs" / "binaries" / "avifenc-qmap.exe"
DEC = BIN / "avifdec.exe"
SOURCE = ROOT / "hpt_data" / "cat.jpg"
Q = 35


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    verify_assets()
    OUT.mkdir(parents=True, exist_ok=True)
    for config in ("REF", "A", "B", "C", "D"):
        output = OUT / f"cat_q35_{config}.avif"
        cmd = [str(DIAG), "--codec", "aom", "--qcolor", str(Q), "--qalpha", "0", "--speed", "6",
               "--jobs", "1", "--depth", "8", "--yuv", "420", "--range", "full", "--cicp", "1/13/1",
               "--ignore-exif", "--ignore-xmp", "--ignore-profile"] + CONFIGS[config] + [str(SOURCE), str(output)]
        proc = subprocess.run(cmd, capture_output=True, text=True, env=ENV)
        if proc.returncode:
            raise RuntimeError(f"qmap encode failed for {config}: {proc.stderr}\n{proc.stdout}")
        (OUT / f"cat_q35_{config}.stdout.txt").write_text(proc.stdout)
        (OUT / f"cat_q35_{config}.stderr.txt").write_text(proc.stderr)
        entries = []
        markers = []
        for line in proc.stderr.splitlines():
            if line.startswith(("PBC_EXTRA_CFG,", "PBC_RECODE_SWITCH,", "PBC_NO_RECODE_AQ,", "PBC_VAQ_CALL,")):
                markers.append(line)
            if line.startswith("PBC_QMAP,"):
                _, row, col, qindex, segment = line.split(",")
                entries.append((int(row), int(col), int(qindex), int(segment)))
        reference = ROOT / "research_pass39" / "outputs" / "benchmark" / f"s6_cat_q35_{config}_timed3.avif"
        if output.read_bytes() != reference.read_bytes():
            raise RuntimeError(f"qmap build changed AVIF bytes for {config}")
        with (OUT / f"cat_q35_{config}.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(("mi_row", "mi_col", "effective_qindex", "segment_id"))
            writer.writerows(entries)
        print(f"{config}: byte-identical={sha(output)==sha(reference)} bytes={output.stat().st_size} map_entries={len(entries)} qindex_range={min(e[2] for e in entries)}..{max(e[2] for e in entries)} segments={sorted(set(e[3] for e in entries))} path_markers={markers}", flush=True)


if __name__ == "__main__":
    main()
