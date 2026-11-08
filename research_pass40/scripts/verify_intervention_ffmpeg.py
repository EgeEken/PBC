import hashlib
import json
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[2]
PASS = ROOT / 'research_pass40'
AVIFS = PASS / 'outputs' / 'intervention'
FFMPEG = pathlib.Path(r'C:\ffmpeg\bin\ffmpeg.exe')
if not FFMPEG.exists():
    raise FileNotFoundError(FFMPEG)
version = subprocess.run([str(FFMPEG), '-version'], capture_output=True, text=True, check=True).stdout.splitlines()[0]
by_hash = {}
for path in AVIFS.glob('*.avif'):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    by_hash.setdefault(digest, path)
rows = []
for digest, path in by_hash.items():
    proc = subprocess.run([str(FFMPEG), '-v', 'error', '-i', str(path), '-f', 'null', '-'], capture_output=True, text=True)
    row = {'file': path.name, 'sha256': digest, 'bytes': path.stat().st_size,
           'decoded': proc.returncode == 0, 'returncode': proc.returncode, 'stderr': proc.stderr.strip()}
    rows.append(row)
    if proc.returncode:
        raise RuntimeError(f'FFmpeg decode failed: {path.name}: {proc.stderr}')
    print(f'decoded {path.name} sha256={digest[:12]}', flush=True)
result = {'ffmpeg': version, 'unique_files': len(rows), 'all_decoded': all(r['decoded'] for r in rows), 'rows': rows}
(PASS / 'outputs' / 'intervention_ffmpeg_decode.json').write_text(json.dumps(result, indent=2) + '\n')
