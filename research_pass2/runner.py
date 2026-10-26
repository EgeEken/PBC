import argparse
import contextlib
import cProfile
import io
import json
import os
from pathlib import Path
import sys
import time

from common import ROOT, OUT, RELEASE, PASS1, PRESETS, THREAD_ENV, sha, save, snapshot, run_plan, images, SCREEN


def worker(source):
    os.environ.update(THREAD_ENV)
    directory = RELEASE if source == 'release' else ROOT
    sys.path.insert(0, str(directory))
    import numpy as np
    import psutil
    from PIL import Image, ImageOps
    from PBC3 import PBC3
    from pbc3_types import PBC3Config, BitReader
    import pbc3_ops as ops

    process = psutil.Process()
    process.cpu_affinity([process.cpu_affinity()[0]])
    if source == 'numpy_root':
        ops.PBC3Rng = np.random.default_rng
    model = directory / 'patch_policy.npz'
    for line in sys.stdin:
        job = json.loads(line)
        try:
            if job['op'] == 'quit':
                print('{}', flush=True)
                break
            if job['op'] == 'metadata':
                reply = {'source': source, 'affinity': process.cpu_affinity(), 'threads': {k: os.environ[k] for k in THREAD_ENV},
                         'model_path': str(model), 'model_sha256': sha(model.read_bytes()),
                         'presets': {p: vars(getattr(PBC3Config, p)()) for p in PRESETS}}
            else:
                from stream_tools import parse, patch_signature
                with Image.open(job['path']) as opened:
                    image = ImageOps.exif_transpose(opened).convert('RGB')
                reference = np.asarray(image)
                cfg = getattr(PBC3Config, job['preset'])(learned_filler_model_path=str(model), **job.get('overrides', {}))
                optimization = job.get('optimization', 'none')
                if optimization != 'none':
                    from optimizations import enabled
                    scope = enabled(optimization)
                else:
                    scope = contextlib.nullcontext()
                profile = cProfile.Profile() if job.get('profile') else None
                with scope, contextlib.redirect_stdout(io.StringIO()):
                    started = time.perf_counter()
                    if profile:
                        result = profile.runcall(PBC3.compress, image, config=cfg)
                    else:
                        result = PBC3.compress(image, config=cfg)
                    encode = time.perf_counter() - started
                    started = time.perf_counter()
                    decoded = PBC3.decompress(result.data)
                    decode = time.perf_counter() - started
                assert decoded.image.size == image.size
                recon = np.asarray(decoded.image.convert('RGB'))
                error = np.subtract(reference, recon, dtype=np.float32)
                np.square(error, out=error)
                header, patches, accounting = parse(result.data)
                frozen = PASS1 / 'streams' / ('release' if source == 'numpy_root' else source) / Path(job['path']).parent.name / (Path(job['path']).name + '.' + job['preset'] + '.pbc3')
                reply = {'label': job.get('label', source), 'source': source, 'optimization': optimization,
                    'dataset': Path(job['path']).parent.name, 'image': Path(job['path']).name, 'preset': job['preset'],
                    'trial': job.get('trial', -1), 'width': image.width, 'height': image.height,
                    'working_width': header[3], 'working_height': header[4], 'mse': float(np.mean(error, dtype=np.float64)),
                    'bpp': 8 * len(result.data) / (image.width * image.height), 'final_bytes': len(result.data),
                    'raw_bytes': accounting['raw_bytes'], 'encode_seconds': encode, 'decode_seconds': decode,
                    'initialization_patches': header[6], 'iterative_patches': len(patches) - header[6], 'patch_count': len(patches),
                    'stream_sha256': sha(result.data), 'pixels_sha256': sha(recon.tobytes()),
                    'preview_equals_decode': bool(np.array_equal(np.asarray(result.image.convert('RGB')), recon)),
                    'frozen_control_equal': result.data == frozen.read_bytes() if frozen.exists() else None,
                    'patch_signatures': [patch_signature(p) for p in patches], 'accounting': accounting,
                    'config': vars(cfg), 'input_sha256': sha(Path(job['path']).read_bytes()), 'model_sha256': sha(model.read_bytes()),
                    'source_revision': '2e59f1b06980aa467da0cde94fa33c3f9a0d4d51' if source == 'release' else '3e9e8c94b7497ecefc5e482a66fde697dfad6e19',
                    'rss_bytes_after': process.memory_info().rss, 'internal_mse': result.mse}
                if profile:
                    import pstats
                    stats = pstats.Stats(profile)
                    reply['profile'] = [{'file': k[0], 'line': k[1], 'function': k[2], 'primitive_calls': v[0],
                        'calls': v[1], 'self_seconds': v[2], 'cumulative_seconds': v[3]} for k, v in stats.stats.items()]
                if job.get('save'):
                    target = OUT / 'streams' / job['experiment'] / reply['label'] / reply['dataset']
                    target.mkdir(parents=True, exist_ok=True)
                    stream_path = target / (reply['image'] + '.' + reply['preset'] + '.pbc3')
                    stream_path.write_bytes(result.data)
                    reply['stream_path'] = str(stream_path.relative_to(OUT))
            reply['modules'] = {n: str(Path(m.__file__).resolve()) for n, m in sys.modules.copy().items()
                if getattr(m, '__file__', None) and (n == 'PBC3' or n == 'learned_filler' or n.startswith('pbc3_'))}
            print(json.dumps(reply), flush=True)
        except Exception:
            import traceback
            print(json.dumps({'error': traceback.format_exc(), 'job': job}), flush=True)


def initial():
    OUT.mkdir(exist_ok=True)
    if (OUT / 'initial_snapshot.json').exists():
        raise RuntimeError('Initial snapshot already exists')
    value = snapshot()
    save('initial_snapshot', value)
    (OUT / 'git_status_before.txt').write_text(value['status'])
    manifest = {str(p.relative_to(ROOT)): sha(p.read_bytes()) for p in PASS1.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.nbc', '.nbi')}
    save('pass1_frozen_manifest', manifest)


def rng():
    cases = [dict(path=str(p), preset=preset, source=s, label=s) for p in images() for preset in PRESETS for s in ('release', 'numpy_root')]
    run_plan('rng_control', cases, 3)


def resolution():
    cases = []
    for name in SCREEN:
        for preset in ('quality', 'high_quality'):
            for source in ('release', 'current'):
                for limit in (250000, 500000, 1000000):
                    cases.append(dict(path=str(ROOT / 'hpt_data' / name), preset=preset, source=source,
                        label=f'{source}_{limit}', overrides={'auto_downsample_max_pixels': limit}))
    run_plan('resolution_screen', cases, 2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', nargs='?', choices=('initial', 'rng', 'resolution'))
    parser.add_argument('--worker', choices=('release', 'current', 'numpy_root'))
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
    else:
        globals()[args.mode]()
