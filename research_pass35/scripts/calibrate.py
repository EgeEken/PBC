import hashlib, json, os, pathlib, subprocess, time
from PIL import Image, ImageOps

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / 'research_pass35' / 'outputs' / 'calibration'
BIN = ROOT / 'research_pass35' / 'outputs' / 'binaries' / 'reference'
IMAGES = ['googlewiki.png', 'mario64.png', 'flower.jpg', 'cat.jpg', 'coast.jpg']
Q = [5, 20, 35, 50, 65]
ENV = dict(os.environ, PATH=str(BIN) + ';C:\\Strawberry\\c\\bin;C:\\Windows\\System32;C:\\Windows')
rows = []
for name in IMAGES:
    src = ROOT / 'hpt_data' / name
    with Image.open(src) as im:
        rgb = ImageOps.exif_transpose(im).convert('RGB')
        w, h = rgb.size
    sha = hashlib.sha256(src.read_bytes()).hexdigest()
    for q in Q:
        target = OUT / f'{pathlib.Path(name).stem}_q{q}.avif'
        cmd = [str(BIN / 'avifenc.exe'), '--codec', 'aom', '--qcolor', str(q), '--qalpha', '0', '--speed', '8', '--jobs', '1', '--depth', '8', '--yuv', '420', '--range', 'full', '--cicp', '1/13/1', '--ignore-exif', '--ignore-xmp', '--ignore-profile', str(src), str(target)]
        t0 = time.perf_counter()
        p = subprocess.run(cmd, capture_output=True, text=True, env=ENV)
        sec = time.perf_counter() - t0
        if p.returncode:
            raise RuntimeError(f'{name} q={q}: {p.stderr}\n{p.stdout}')
        data = target.read_bytes()
        rows.append({'image': name, 'source_sha256': sha, 'width': w, 'height': h, 'qcolor': q, 'bytes': len(data), 'bpp': len(data) * 8 / (w * h), 'encode_seconds': sec, 'avif_sha256': hashlib.sha256(data).hexdigest()})
        print(f'{name:16} q={q:2} {len(data):8} B {rows[-1]["bpp"]:.4f} bpp {sec:.3f}s')
(OUT / 'calibration.json').write_text(json.dumps({'qcolor_grid': Q, 'settings': {'codec':'aom','qalpha':0,'speed':8,'jobs':1,'depth':8,'yuv':'420','range':'full','cicp':'1/13/1','ignore_metadata':True}, 'rows': rows}, indent=2) + '\n')
