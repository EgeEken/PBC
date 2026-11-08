import hashlib
import json
import pathlib
import random
import shutil
import sys

import numpy as np
import PIL

ROOT = pathlib.Path(__file__).resolve().parents[2]
PASS = ROOT / 'research_pass40'
OUT = PASS / 'outputs' / 'intervention'
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / 'research_pass39' / 'scripts'))
import benchmark as p39

REFERENCE = ROOT / 'research_pass35' / 'outputs' / 'binaries' / 'reference'
VARIANTS = {name: PASS / 'outputs' / 'binaries' / folder
            for name, folder in [('CTRL', 'control'), ('EXP', 'experiment')]}
for directory in [*VARIANTS.values()]:
    shutil.copy2(REFERENCE / 'avifdec.exe', directory / 'avifdec.exe')

CONFIGS = {
    'REF': ['--advanced', 'aq-mode=0', '--advanced', 'tune=psnr', '--advanced', 'deltaq-mode=6', '--advanced', 'sb-size=64'],
    'CTRL': ['--advanced', 'aq-mode=0', '--advanced', 'tune=psnr', '--advanced', 'deltaq-mode=6', '--advanced', 'sb-size=64'],
    'EXP': ['--advanced', 'aq-mode=0', '--advanced', 'tune=psnr', '--advanced', 'deltaq-mode=6', '--advanced', 'sb-size=64'],
}
POINTS = [('googlewiki.png', 5), ('googlewiki.png', 20), ('mario64.png', 50),
          ('flower.jpg', 80), ('cat.jpg', 35), ('cat.jpg', 50), ('coast.jpg', 80)]
ENV_PATH = ';'.join([str(REFERENCE), str(ROOT / 'research_pass35' / 'outputs' / 'binaries'),
                     r'C:\Strawberry\c\bin', r'C:\Windows\System32', r'C:\Windows'])
p39.OUT = OUT
p39.CONFIGS = CONFIGS
p39.verify_assets()

rows = []


def one(image, q, config, phase, repeat):
    p39.BIN = REFERENCE if config == 'REF' else VARIANTS[config]
    p39.ENV = dict(p39.os.environ, PATH=str(p39.BIN) + ';' + ENV_PATH)
    src = ROOT / 'hpt_data' / image
    row, _, _ = p39.one(src, image, q, config, 6, repeat, phase)
    row['source_sha256'] = p39.digest(src)
    row['binary_sha256'] = p39.digest(p39.BIN / 'avifenc.exe')
    rows.append(row)
    return row


# Verify the macro-off relink before any intervention comparison.
for image, q in POINTS:
    ref = one(image, q, 'REF', 'parity', 0)
    ctrl = one(image, q, 'CTRL', 'parity', 0)
    if ref['avif_sha256'] != ctrl['avif_sha256']:
        raise RuntimeError(f'control/reference parity failed: {image} q{q}: {ref["avif_sha256"]} != {ctrl["avif_sha256"]}')
    print(f'PARITY {image} q{q}: identical {ref["avif_sha256"]}', flush=True)

for image, q in POINTS:
    order = ['REF', 'EXP']
    random.Random(f'p40-intervention-{image}-{q}').shuffle(order)
    for config in order:
        row = one(image, q, config, 'warm', 0)
        print(f'{image} q{q} {config} warm {row["bytes"]}B MSE {row["rgb_mse"]:.6f}', flush=True)
    for repeat in range(1, 4):
        rotated = order[repeat % 2:] + order[:repeat % 2]
        for config in rotated:
            row = one(image, q, config, 'timed', repeat)
            print(f'{image} q{q} {config} r{repeat} {row["bytes"]}B MSE {row["rgb_mse"]:.6f} enc {row["encode_seconds"]:.3f}s', flush=True)
    (OUT / 'results.json').write_text(json.dumps({
        'source': {'libaom': '3.12.1', 'commit': '10aece4157eb79315da205f39e19bf6ab3ee30d0',
                   'libavif': '1.3.0', 'commit_libavif': '1aadfad932c98c069a1204261b1856f81f3bc199'},
        'patch_sha256': '03814c15914290c8fd6dc3ae973f84d7110c59131759af152e111fe692f33488',
        'settings': {'quality_points': POINTS, 'speed': 6, 'jobs': 1, 'depth': 8, 'yuv': '420',
                     'range': 'full', 'cicp': '1/13/1', 'aq_mode': 0, 'tune': 'psnr',
                     'deltaq_mode': 6, 'sb_size': 64, 'warmups': 1, 'timed_repetitions': 3,
                     'mse': 'RGB arrays promoted to float64 before subtraction and squaring'},
        'binaries': {key: p39.digest(path / 'avifenc.exe') for key, path in {'REF': REFERENCE, **VARIANTS}.items()},
        'build': {'python': sys.version.split()[0], 'numpy': np.__version__, 'pillow': PIL.__version__},
        'configurations': CONFIGS, 'rows': rows}, indent=2) + '\n')
    print(f'completed {image} q{q}', flush=True)
