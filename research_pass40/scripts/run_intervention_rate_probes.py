import json
import pathlib
import random
import sys

import numpy as np
import PIL

ROOT = pathlib.Path(__file__).resolve().parents[2]
PASS = ROOT / 'research_pass40'
OUT = PASS / 'outputs' / 'intervention'
sys.path.insert(0, str(ROOT / 'research_pass39' / 'scripts'))
import benchmark as p39

BIN = PASS / 'outputs' / 'binaries' / 'experiment'
p39.BIN = BIN
p39.ENV = dict(p39.os.environ, PATH=str(BIN) + ';' + str(ROOT / 'research_pass35' / 'outputs' / 'binaries' / 'reference') + r';C:\Strawberry\c\bin;C:\Windows\System32;C:\Windows')
p39.OUT = OUT
p39.CONFIGS = {'EXP': ['--advanced', 'aq-mode=0', '--advanced', 'tune=psnr', '--advanced', 'deltaq-mode=6', '--advanced', 'sb-size=64']}
p39.verify_assets()
PROBES = [('googlewiki.png', 3), ('googlewiki.png', 4), ('flower.jpg', 82), ('flower.jpg', 83),
          ('cat.jpg', 38), ('cat.jpg', 53), ('coast.jpg', 84), ('coast.jpg', 85)]
path = OUT / 'results.json'
ledger = json.loads(path.read_text())
rows = ledger['rows']
for image, q in PROBES:
    row, _, _ = p39.one(ROOT / 'hpt_data' / image, image, q, 'EXP', 6, 0, 'calibrationwarm')
    row['source_sha256'] = p39.digest(ROOT / 'hpt_data' / image)
    row['binary_sha256'] = p39.digest(BIN / 'avifenc.exe')
    rows.append(row)
    for rep in range(1, 4):
        row, _, _ = p39.one(ROOT / 'hpt_data' / image, image, q, 'EXP', 6, rep, 'calibration')
        row['source_sha256'] = p39.digest(ROOT / 'hpt_data' / image)
        row['binary_sha256'] = p39.digest(BIN / 'avifenc.exe')
        rows.append(row)
        print(f'{image} q{q} r{rep} {row["bytes"]}B MSE {row["rgb_mse"]:.6f}', flush=True)
    path.write_text(json.dumps(ledger, indent=2) + '\n')
    print(f'completed {image} q{q}', flush=True)
