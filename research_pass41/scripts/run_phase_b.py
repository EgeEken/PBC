import hashlib
import json
import pathlib
import random
import subprocess
import sys

import numpy as np
import PIL

ROOT = pathlib.Path(__file__).resolve().parents[2]
PASS = ROOT / 'research_pass41'
OUT = PASS / 'outputs'
PLANE = OUT / 'binaries' / 'plane_probe.exe'
sys.path.insert(0, str(ROOT / 'research_pass39' / 'scripts'))
import benchmark as p39

REF_BIN = ROOT / 'research_pass35' / 'outputs' / 'binaries' / 'reference'
p39.BIN = REF_BIN
p39.ENV = dict(p39.os.environ, PATH=str(REF_BIN) + r';C:\Strawberry\c\bin;C:\Windows\System32;C:\Windows')
p39.OUT = OUT / 'benchmark'
p39.OUT.mkdir(parents=True, exist_ok=True)
OPTIONS = ['--advanced', 'aq-mode=0', '--advanced', 'tune=psnr', '--advanced', 'deltaq-mode=0', '--advanced', 'sb-size=64']
p39.CONFIGS = {'A': OPTIONS}
POINTS = [('googlewiki.png', 5), ('cat.jpg', 35), ('coast.jpg', 80)]
rows = []
source_hashes = {}
for image, q in POINTS:
    source = ROOT / 'hpt_data' / image
    digest = p39.digest(source)
    source_hashes[image] = digest
    for phase, repeat in [('warm', 0), *[('timed', i) for i in range(1, 4)]]:
        row, _, _ = p39.one(source, image, q, 'A', 6, repeat, f'phaseB_{phase}')
        row['source_sha256'] = digest
        row['binary_sha256'] = p39.digest(REF_BIN / 'avifenc.exe')
        rows.append(row)
        print(f'A {image} q{q} {phase}{repeat}: {row["bytes"]} B MSE {row["rgb_mse"]:.6f}', flush=True)
    prefix = OUT / 'planes' / f'{pathlib.Path(image).stem}_q{q}'
    prefix.parent.mkdir(parents=True, exist_ok=True)
    source_proc = subprocess.run([str(PLANE), 'source', str(source), str(prefix)], capture_output=True, text=True, env=p39.ENV)
    if source_proc.returncode:
        raise RuntimeError(f'pinned avifenc source-loader conversion failed for {image}:\n{source_proc.stderr}')
    lossy = OUT / 'benchmark' / f's6_{pathlib.Path(image).stem}_q{q}_A_phaseB_timed3.avif'
    probe = subprocess.run([str(PLANE), 'compare', str(lossy), str(prefix)], capture_output=True, text=True, env=p39.ENV)
    if probe.returncode:
        raise RuntimeError(f'native-plane probe failed for {image}:\n{probe.stderr}')
    print(f'planes {image} q{q}: {probe.stdout.strip()}', flush=True)
    (OUT / 'phase_b_results.json').write_text(json.dumps({
        'sources': {'libaom_commit': '10aece4157eb79315da205f39e19bf6ab3ee30d0', 'libavif_commit': '1aadfad932c98c069a1204261b1856f81f3bc199'},
        'settings': {'points': POINTS, 'speed': 6, 'jobs': 1, 'depth': 8, 'yuv': '420', 'range': 'full', 'cicp': '1/13/1', 'config': OPTIONS},
        'source_sha256': source_hashes, 'reference_avifenc_sha256': p39.digest(REF_BIN / 'avifenc.exe'),
        'plane_probe_sha256': p39.digest(PLANE), 'python': sys.version.split()[0], 'numpy': np.__version__, 'pillow': PIL.__version__, 'rows': rows}, indent=2) + '\n')
