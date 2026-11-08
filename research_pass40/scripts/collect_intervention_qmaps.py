import csv
import hashlib
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
PASS = ROOT / 'research_pass40'
OUT = PASS / 'outputs' / 'intervention'
sys.path.insert(0, str(ROOT / 'research_pass39' / 'scripts'))
import benchmark as p39

DIAGS = {
    'REF': ROOT / 'research_pass39' / 'outputs' / 'binaries' / 'avifenc-qmap.exe',
    'EXP': PASS / 'outputs' / 'binaries' / 'experiment-qmap' / 'avifenc-qmap.exe',
}
CONFIG = ['--advanced', 'aq-mode=0', '--advanced', 'tune=psnr', '--advanced', 'deltaq-mode=6', '--advanced', 'sb-size=64']
POINTS = [('googlewiki.png', 5, 'REF'), ('googlewiki.png', 3, 'EXP'),
          ('googlewiki.png', 20, 'REF'), ('googlewiki.png', 20, 'EXP'),
          ('mario64.png', 50, 'REF'), ('mario64.png', 50, 'EXP'),
          ('flower.jpg', 80, 'REF'), ('flower.jpg', 83, 'EXP'),
          ('cat.jpg', 35, 'REF'), ('cat.jpg', 38, 'EXP'),
          ('cat.jpg', 50, 'REF'), ('cat.jpg', 53, 'EXP'),
          ('coast.jpg', 80, 'REF'), ('coast.jpg', 84, 'EXP')]
OUT.mkdir(parents=True, exist_ok=True)
ledger = json.loads((OUT / 'results.json').read_text())
rows = []
for image, q, config in POINTS:
    binary = DIAGS[config]
    key = f'{pathlib.Path(image).stem}_q{q}_{config}'
    avif = OUT / f'{key}_qmap.avif'
    command = [str(binary), '--codec', 'aom', '--qcolor', str(q), '--qalpha', '0', '--speed', '6',
               '--jobs', '1', '--depth', '8', '--yuv', '420', '--range', 'full', '--cicp', '1/13/1',
               '--ignore-exif', '--ignore-xmp', '--ignore-profile'] + CONFIG + [str(ROOT / 'hpt_data' / image), str(avif)]
    env = dict(p39.os.environ, PATH=str(binary.parent) + ';' + str(ROOT / 'research_pass35' / 'outputs' / 'binaries' / 'reference') + r';C:\Strawberry\c\bin;C:\Windows\System32;C:\Windows')
    proc = subprocess.run(command, capture_output=True, text=True, env=env)
    if proc.returncode:
        raise RuntimeError(f'qmap encode failed {key}: {proc.stderr}\n{proc.stdout}')
    wanted = [r for r in ledger['rows'] if r['image'] == image and r['qcolor'] == q and r['config'] == config and r['phase'] in ('timed', 'parity', 'calibration')]
    digest = hashlib.sha256(avif.read_bytes()).hexdigest()
    if not wanted or digest != wanted[0]['avif_sha256']:
        raise RuntimeError(f'qmap byte parity failed {key}: {digest}')
    samples = []
    for line in proc.stderr.splitlines():
        if line.startswith('PBC_QMAP,'):
            _, row, col, qindex, segment = line.split(',')
            row, col = int(row), int(col)
            if row % 16 == 0 and col % 16 == 0:
                samples.append((row, col, int(qindex), int(segment)))
    path = OUT / f'{key}.csv'
    with path.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(('mi_row', 'mi_col', 'effective_qindex', 'segment_id'))
        writer.writerows(samples)
    rows.append({'image': image, 'qcolor': q, 'config': config, 'sha256': digest,
                 'byte_parity': True, 'root_samples': len(samples), 'qindex_min': min(x[2] for x in samples),
                 'qindex_max': max(x[2] for x in samples), 'csv': path.name,
                 'diagnostic_sha256': p39.digest(binary)})
    print(f'{key}: parity=True roots={len(samples)} q={min(x[2] for x in samples)}..{max(x[2] for x in samples)}', flush=True)
(PASS / 'outputs' / 'intervention_qmaps.json').write_text(json.dumps({'rows': rows}, indent=2) + '\n')
