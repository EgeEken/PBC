import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
PASS1 = ROOT / 'research_pass1'
RELEASE = PASS1 / 'release'
PRESETS = ('compression', 'balanced', 'quality', 'high_quality')
THREAD_ENV = {k: '1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')}
SCREEN = ('cat.jpg', 'monet.jpg', 'flower.jpg', 'cathedral.png', 'googlewiki.png', 'stockpoint.jpg')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def images(dataset='all'):
    folders = ('hpt_data', 'hpt_data_val') if dataset == 'all' else (dataset,)
    return [p for folder in folders for p in sorted((ROOT / folder).iterdir()) if p.suffix.lower() in ('.jpg', '.jpeg', '.png')]


def save(name, value):
    (OUT / (name + '.json')).write_text(json.dumps(value, indent=2), encoding='utf-8')
    if isinstance(value, list) and value:
        keys = list(dict.fromkeys(k for row in value for k, v in row.items() if not isinstance(v, (dict, list))))
        with (OUT / (name + '.csv')).open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(value)


def snapshot():
    manifest = json.loads((PASS1 / 'runtime_source_manifest.json').read_text())
    paths = [ROOT / p for p in manifest]
    paths += [ROOT / 'patch_policy.npz', RELEASE / 'patch_policy.npz'] + images()
    return {'time_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            'status': subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True),
            'hashes': {str(p.relative_to(ROOT)): sha(p.read_bytes()) for p in paths},
            'scripts': {p.name: sha(p.read_bytes()) for p in OUT.glob('*.py')},
            'python': sys.version, 'platform': platform.platform(), 'processor': platform.processor(),
            'versions': {n: importlib.metadata.version(n) for n in ('numpy', 'numba', 'Pillow', 'opencv-python', 'psutil')},
            'thread_env': THREAD_ENV, 'alpha': 'EXIF transpose then RGB for every research input'}


def request(process, job):
    process.stdin.write(json.dumps(job) + '\n')
    process.stdin.flush()
    line = process.stdout.readline()
    if not line:
        raise RuntimeError(f'worker exited {process.poll()}')
    row = json.loads(line)
    if 'error' in row:
        raise RuntimeError(row['error'])
    return row


def run_plan(name, cases, repeats=3):
    before = snapshot()
    sources = sorted({c.get('source', 'current') for c in cases})
    processes, logs, metadata = {}, [], {}
    try:
        for source in sources:
            log = (OUT / (name + '_' + source + '_stderr.log')).open('w')
            logs.append(log)
            processes[source] = subprocess.Popen([sys.executable, str(OUT / 'runner.py'), '--worker', source],
                cwd=RELEASE if source == 'release' else ROOT, env={**os.environ, **THREAD_ENV, 'PYTHONPATH': ''},
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True, bufsize=1)
            metadata[source] = request(processes[source], {'op': 'metadata'})
        warmups = []
        distinct = {(c.get('source', 'current'), c['preset'], c.get('optimization', 'none'), json.dumps(c.get('overrides', {}), sort_keys=True)) for c in cases}
        for source, preset, optimization, overrides in sorted(distinct):
            warmups.append(request(processes[source], dict(op='encode', path=str(ROOT / 'hpt_data' / 'googlewiki.png'),
                preset=preset, optimization=optimization, overrides=json.loads(overrides), warmup=True)))
        save(name + '_metadata', dict(before=before, workers=metadata, repeats=repeats, cases=cases,
            warmup='One complete encode/decode on googlewiki for each source/config/optimization before timed cases; disk caches retained', warmup_rows=warmups))
        with (OUT / (name + '.jsonl')).open('w', encoding='utf-8') as f:
            for trial in range(repeats):
                for image_i, path in enumerate(dict.fromkeys(c['path'] for c in cases)):
                    group = [c for c in cases if c['path'] == path]
                    shift = (trial + image_i) % len(group)
                    for case in group[shift:] + group[:shift]:
                        source = case.get('source', 'current')
                        row = request(processes[source], {**case, 'op': 'encode', 'trial': trial, 'save': trial == 0, 'experiment': name})
                        f.write(json.dumps(row) + '\n')
                        f.flush()
                    print(name, f'{trial + 1}/{repeats}', Path(path).name, flush=True)
    finally:
        for process in processes.values():
            if process.poll() is None:
                request(process, {'op': 'quit'})
                process.wait(timeout=30)
        for log in logs:
            log.close()
        after = snapshot()
        save(name + '_provenance', {'before': before, 'after': after, 'hashes_unchanged': before['hashes'] == after['hashes']})
    rows = [json.loads(line) for line in (OUT / (name + '.jsonl')).read_text().splitlines()]
    save(name, rows)
    return rows
