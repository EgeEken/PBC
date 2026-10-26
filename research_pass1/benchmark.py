import argparse
import contextlib
import csv
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
PRESETS = ('compression', 'balanced', 'quality', 'high_quality')
SOURCES = {'release': OUT / 'release', 'old': ROOT / 'old_pbc', 'current': ROOT}
THREAD_ENV = {k: '1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def worker(source):
    sys.path.insert(0, str(SOURCES[source]))
    import numpy as np
    import psutil
    from PIL import Image, ImageOps
    from PBC3 import PBC3
    from pbc3_types import PBC3Config, BitReader

    process = psutil.Process()
    process.cpu_affinity([process.cpu_affinity()[0]])
    model = SOURCES[source] / 'patch_policy.npz'
    for line in sys.stdin:
        job = json.loads(line)
        try:
            if job['op'] == 'quit':
                break
            if job['op'] == 'metadata':
                reply = {'source': source, 'python': sys.version, 'executable': sys.executable,
                         'platform': platform.platform(), 'processor': platform.processor(),
                         'affinity': process.cpu_affinity(), 'threads': {k: os.environ.get(k) for k in THREAD_ENV},
                         'versions': {k: importlib.metadata.version(k) for k in ('numpy', 'numba', 'Pillow', 'opencv-python', 'psutil')},
                         'model_path': str(model), 'model_sha256': sha(model.read_bytes()),
                         'presets': {p: vars(getattr(PBC3Config, p)()) for p in PRESETS}, 'stream_version': PBC3.VERSION}
            else:
                with Image.open(job['path']) as opened:
                    image = ImageOps.exif_transpose(opened).convert('RGB')
                reference = np.asarray(image).astype(np.float64)
                config = getattr(PBC3Config, job['preset'])(learned_filler_model_path=str(model), **job.get('overrides', {}))
                started = time.perf_counter()
                with contextlib.redirect_stdout(io.StringIO()):
                    result = PBC3.compress(image, config=config)
                encode = time.perf_counter() - started
                started = time.perf_counter()
                decoded = PBC3.decompress(result.data)
                decode = time.perf_counter() - started
                assert decoded.image.size == image.size
                recon = np.asarray(decoded.image.convert('RGB'))
                _, body = PBC3._open_body(result.data)
                header = PBC3._read_header(BitReader(body))
                reply = {'source': source, 'dataset': Path(job['path']).parent.name,
                         'image': Path(job['path']).name, 'preset': job['preset'], 'trial': job.get('trial', -1),
                         'width': image.width, 'height': image.height, 'working_width': header[3], 'working_height': header[4],
                         'mse': float(np.mean((reference - recon.astype(np.float64)) ** 2)),
                         'bpp': 8 * len(result.data) / (image.width * image.height),
                         'raw_bytes': len(body) + 6, 'raw_body_bytes': len(body), 'final_bytes': len(result.data),
                         'encode_seconds': encode, 'decode_seconds': decode, 'result_encode_seconds': result.encode_seconds,
                         'patch_count': header[10], 'stream_sha256': sha(result.data), 'pixels_sha256': sha(recon.tobytes()),
                         'preview_equals_decode': bool(np.array_equal(np.asarray(result.image.convert('RGB')), recon)),
                         'config': vars(config), 'input_sha256': sha(Path(job['path']).read_bytes())}
                if job.get('save'):
                    target = OUT / 'streams' / source / Path(job['path']).parent.name
                    target.mkdir(parents=True, exist_ok=True)
                    (target / (Path(job['path']).name + '.' + job['preset'] + '.pbc3')).write_bytes(result.data)
            reply['modules'] = {n: str(Path(m.__file__).resolve()) for n, m in sys.modules.copy().items()
                                if getattr(m, '__file__', None) and (n in ('PBC3', 'pbc3_types', 'learned_filler', 'patch_features', 'patch_actions', 'patch_proposal', 'rl_state') or n.startswith('pbc3_'))}
            print(json.dumps(reply), flush=True)
        except Exception:
            import traceback
            print(json.dumps({'error': traceback.format_exc(), 'job': job}), flush=True)


def request(process, job):
    process.stdin.write(json.dumps(job) + '\n')
    process.stdin.flush()
    line = process.stdout.readline()
    if not line:
        raise RuntimeError(f'Worker exited: {process.poll()}')
    reply = json.loads(line)
    if 'error' in reply:
        raise RuntimeError(reply['error'])
    return reply


def baseline(smoke, repeats):
    env = {**os.environ, **THREAD_ENV, 'PYTHONPATH': ''}
    processes = {}
    log_handles = []
    prefix = 'smoke' if smoke else 'baseline'
    metadata = {'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'branch': subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip(),
                'repeats': repeats, 'warmup': 'One complete learned encode/decode per preset on googlewiki.png before trials; first call recorded separately',
                'alpha_policy': 'EXIF transpose then RGB conversion; alpha discarded consistently', 'sources': {}}
    paths = [p for folder in ('hpt_data', 'hpt_data_val') for p in sorted((ROOT / folder).iterdir())
             if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp', '.bmp')]
    if smoke:
        paths = [p for p in paths if p.name in ('googlewiki.png', 'flower.jpg')]
    metadata['dataset'] = [{'path': str(p), 'sha256': sha(p.read_bytes())} for p in paths]
    try:
        for source, directory in SOURCES.items():
            log = open(OUT / (prefix + '_' + source + '_stderr.log'), 'w')
            log_handles.append(log)
            processes[source] = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--worker', source],
                                                  cwd=directory, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True, bufsize=1)
            metadata['sources'][source] = request(processes[source], {'op': 'metadata'})
            metadata['sources'][source]['warmup_rows'] = [request(processes[source], {'op': 'encode', 'path': str(ROOT / 'hpt_data' / 'googlewiki.png'), 'preset': p}) for p in PRESETS]
            print('warmed', source, flush=True)
        (OUT / (prefix + '_metadata.json')).write_text(json.dumps(metadata, indent=2))
        with open(OUT / (prefix + '.jsonl'), 'w') as output:
            for trial in range(repeats):
                for image_i, path in enumerate(paths):
                    for preset_i, preset in enumerate(PRESETS):
                        order = list(SOURCES)
                        shift = (trial + image_i + preset_i) % len(order)
                        order = order[shift:] + order[:shift]
                        for source in order:
                            row = request(processes[source], {'op': 'encode', 'path': str(path), 'preset': preset, 'trial': trial, 'save': trial == 0})
                            row['commit'] = metadata['commit'] if source != 'release' else '2e59f1b06980aa467da0cde94fa33c3f9a0d4d51'
                            row['model_sha256'] = metadata['sources'][source]['model_sha256']
                            row['config_sha256'] = sha(json.dumps(row['config'], sort_keys=True).encode())
                            output.write(json.dumps(row) + '\n')
                            output.flush()
                    print(f'{prefix} trial {trial + 1}/{repeats} {path.parent.name}/{path.name}', flush=True)
    finally:
        for process in processes.values():
            if process.poll() is None:
                process.stdin.write('{"op":"quit"}\n')
                process.stdin.flush()
                process.wait(timeout=30)
        for log in log_handles:
            log.close()
    summarize(prefix)


def summarize(prefix):
    import numpy as np
    rows = [json.loads(line) for line in (OUT / (prefix + '.jsonl')).read_text().splitlines()]
    flat = [{k: v for k, v in r.items() if not isinstance(v, dict)} for r in rows]
    with open(OUT / (prefix + '.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    summary = []
    for dataset in ('hpt_data', 'hpt_data_val'):
        for preset in PRESETS:
            for source in SOURCES:
                selected = [r for r in rows if (r['dataset'], r['preset'], r['source']) == (dataset, preset, source)]
                if not selected:
                    continue
                s = {'dataset': dataset, 'preset': preset, 'source': source, 'rows': len(selected)}
                for metric in ('mse', 'bpp', 'raw_bytes', 'final_bytes', 'encode_seconds', 'decode_seconds'):
                    s[metric] = float(np.mean([r[metric] for r in selected]))
                for metric in ('encode_seconds', 'decode_seconds'):
                    per_image = [[r[metric] for r in selected if r['image'] == image] for image in sorted({r['image'] for r in selected})]
                    s[metric + '_mean_image_median'] = float(np.mean([np.median(v) for v in per_image]))
                    s[metric + '_mean_image_cv'] = float(np.mean([np.std(v) / np.mean(v) for v in per_image]))
                summary.append(s)
    (OUT / (prefix + '_summary.json')).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', choices=SOURCES)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--summarize')
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
    elif args.summarize:
        summarize(args.summarize)
    else:
        baseline(args.smoke, args.repeats)
