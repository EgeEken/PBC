from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from threadpoolctl import threadpool_limits

os.environ.setdefault("NUMBA_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
OUT = HERE / 'outputs'
REVISION = '8006f30debfc4a57289df06df21636aeb52baffd'
EXISTING = ('cat.jpg', 'flower.jpg', 'mario64.png')
NEW = ('cathedral.png', 'park.jpg', 'seagull.jpg')
ALL = EXISTING + NEW


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def prepare_release_runtime() -> tuple[Path, dict]:
    manifest = json.loads((ROOT / 'research_pass10/baseline_manifest.json').read_text())
    if manifest['github_source_revision'] != REVISION or manifest['baseline_id'] != 'pbc31_release_rng_baseline':
        raise ValueError('V3.1 release-RNG baseline manifest changed')
    runtime = OUT / 'release_rng_runtime'
    runtime.mkdir(parents=True, exist_ok=True)
    for name, expected in manifest['source_sha256'].items():
        raw = subprocess.run(['git', 'show', f'{REVISION}:{name}'], cwd=ROOT, check=True, capture_output=True).stdout
        if sha(raw) != expected:
            raise ValueError(f'pinned V3.1 source hash mismatch: {name}')
        path = runtime / name
        if not path.exists() or sha(path.read_bytes()) != expected:
            path.write_bytes(raw)
    model = ROOT / 'patch_policy.npz'
    if sha(model.read_bytes()) != manifest['model_sha256']:
        raise ValueError('pinned V3.1 model hash mismatch')
    target = runtime / 'patch_policy.npz'
    if not target.exists() or sha(target.read_bytes()) != manifest['model_sha256']:
        target.write_bytes(model.read_bytes())
    return runtime, manifest


def base_refs() -> dict:
    d = json.loads((ROOT / 'research_pass29/outputs/results.json').read_text())
    return {row['image']: row for row in d['images']}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    limits = threadpool_limits(limits=1)
    runtime, release = prepare_release_runtime()
    refs = base_refs()
    assets = json.loads((ROOT / 'research/assets.json').read_text())['inputs']
    requests = []
    sources = {}
    base_rows = []
    sys.path.insert(0, str(runtime))
    from PIL import Image as PILImage
    from PBC3 import PBC3
    from pbc3_types import PBC3Config

    for name in ALL:
        path = ROOT / 'hpt_data' / name
        source_bytes = path.read_bytes()
        input_ref = assets[f'hpt_data/{name}']
        source_hash = sha(source_bytes)
        if source_hash != input_ref['sha256']:
            raise ValueError(f'benchmark source hash changed: {name}')
        source = np.asarray(ImageOps.exif_transpose(Image.open(path)).convert('RGB'), np.uint8)
        sources[name] = source
        if name in EXISTING:
            ref = refs[name]
            stream_ref = ref['base']
            stream = (ROOT / 'research_pass29/outputs/bases' / f'{name}.pbc3').read_bytes()
            if sha(stream) != stream_ref['sha256'] or source_hash != stream_ref['source_sha256']:
                raise ValueError(f'Pass 29 frozen base/source mismatch: {name}')
            provenance = {'kind': 'cached_pass29_quality', 'pass29_base_sha256': stream_ref['sha256']}
            encode_s = None
        else:
            out_stream = OUT / 'bases' / f'{name}.pbc3'
            if out_stream.exists():
                stream = out_stream.read_bytes()
                if not (OUT / 'manifest.json').exists():
                    raise ValueError(f'unmanifested new base already exists: {name}')
                previous = json.loads((OUT / 'manifest.json').read_text())
                old = next((x for x in previous['images'] if x['image'] == name), None)
                if old is None or old['source_sha256'] != source_hash or old['stream_sha256'] != sha(stream):
                    raise ValueError(f'existing V3.1 base does not match source/provenance: {name}')
                provenance = {'kind': 'generated_once_pinned_v31_release_rng', 'baseline_revision': REVISION, 'reused_existing': True}
                encode_s = old['base_encode_seconds']
            else:
                preset = dict(release['presets']['quality'])
                preset['learned_filler_model_path'] = str(runtime / 'patch_policy.npz')
                config = PBC3Config(**preset)
                start = time.perf_counter()
                result = PBC3.compress(PILImage.fromarray(source), config=config, rng_mode='release')
                encode_s = time.perf_counter() - start
                stream = result.data
                out_stream.parent.mkdir(parents=True, exist_ok=True)
                out_stream.write_bytes(stream)
                provenance = {'kind': 'generated_once_pinned_v31_release_rng', 'baseline_revision': REVISION, 'reused_existing': False}
        stream_sha = sha(stream)
        stream_path = OUT / 'bases' / f'{name}.pbc3'
        if name in EXISTING:
            stream_path = ROOT / 'research_pass29/outputs/bases' / f'{name}.pbc3'
        base_path = OUT / 'bases' / f'{name}.npy'
        base_path.parent.mkdir(parents=True, exist_ok=True)
        requests.append({'image': name, 'stream': str(stream_path.resolve()), 'base_npy': str(base_path.resolve()),
                         'source_shape': list(source.shape), 'source_sha256': source_hash, 'stream_sha256': stream_sha,
                         'source_path': str(path.resolve()), 'provenance': provenance, 'base_encode_seconds': encode_s})

    request_path = OUT / 'base_decode_request.json'
    request_path.write_text(json.dumps({'runtime': str(runtime.resolve()), 'images': requests}))
    run = subprocess.run([sys.executable, str(HERE / 'decode_bases_worker.py'), str(request_path)],
                         cwd=ROOT, check=True, capture_output=True, text=True)
    decoded = json.loads(run.stdout)['images']
    by_name = {x['image']: x for x in decoded}
    for req in requests:
        row = by_name[req['image']]
        if row['shape'] != req['source_shape']:
            raise ValueError(f'decoded base/source dimension mismatch: {req["image"]}')
        base_rows.append({**req, 'decoded_base_sha256': row['decoded_base_sha256'],
                          'base_bytes': Path(req['stream']).stat().st_size,
                          'base_mse': row['mse'], 'width': req['source_shape'][1], 'height': req['source_shape'][0]})
        print(req['image'], row['shape'], 'base bytes', Path(req['stream']).stat().st_size,
              'base mse', round(row['mse'], 4), 'stream sha', req['stream_sha256'], flush=True)

    saved = {'baseline_id': release['baseline_id'], 'baseline_revision': REVISION,
             'rng': release['rng'], 'seed': release['default_seed'], 'model_sha256': release['model_sha256'],
             'release_source_sha256': release['source_sha256'], 'images': base_rows}
    limits.restore_original_limits()
    limits.restore_original_limits()
    (OUT / 'manifest.json').write_text(json.dumps(saved, indent=2))
    print('wrote', OUT / 'manifest.json')


if __name__ == '__main__':
    main()