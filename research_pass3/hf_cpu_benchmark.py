import argparse
import contextlib
import csv
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
PRESETS = ('compression', 'balanced', 'quality', 'high_quality')
THREADS = {k: '1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')}
RUNTIME = ('PBC3.py', 'pbc3_types.py', 'pbc3_ops.py', 'pbc3_heads.py', 'pbc3_features.py', 'learned_filler.py', 'pbc3_stream.py', 'pbc3_trace.py')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def module_hashes(directory):
    return {n: digest((directory / n).read_text().encode()) for n in RUNTIME if (directory / n).exists()}


def worker(args):
    os.environ.update(THREADS)
    directory = Path(args.reference).resolve() if args.worker == 'reference' else ROOT
    sys.path.insert(0, str(directory))
    import numpy as np
    import psutil
    from PIL import Image, ImageOps
    from PBC3 import PBC3
    from pbc3_types import PBC3Config, BitReader
    psutil.Process().cpu_affinity([args.cpu])
    candidate = None
    if args.worker == 'current':
        import pbc3_stream as stream
        original_read, original_write, original_class_read = stream.read_grid, stream.write_grid, PBC3._read_grid
        spec = importlib.util.spec_from_file_location('private_candidate_stream', HERE / 'integration_candidate' / 'pbc3_stream.py')
        candidate = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(candidate)
        candidate.GRID_IO_MODE = 'numpy'
    model = ROOT / 'patch_policy.npz'
    for line in sys.stdin:
        job = json.loads(line)
        try:
            if job['op'] == 'quit':
                print('{}', flush=True)
                break
            if job['op'] == 'metadata':
                reply = dict(kind=args.worker, python=sys.version, platform=platform.platform(), cpu=platform.processor(),
                    affinity=psutil.Process().cpu_affinity(), threads={k: os.environ[k] for k in THREADS},
                    versions={k: importlib.metadata.version(k) for k in ('numpy', 'numba', 'Pillow', 'psutil')},
                    model_sha256=digest(model.read_bytes()), source_hashes_lf=module_hashes(directory),
                    module_paths={k: str(Path(m.__file__).resolve()) for k, m in sys.modules.copy().items() if getattr(m, '__file__', None) and (k == 'PBC3' or k == 'learned_filler' or k.startswith('pbc3_'))})
            else:
                variant = job['variant']
                if candidate is not None:
                    stream.read_grid = candidate.read_grid if variant in ('grid', 'both') else original_read
                    stream.write_grid = candidate.write_grid if variant in ('grid', 'both') else original_write
                    PBC3._read_grid = staticmethod(candidate.read_grid if variant in ('grid', 'both') else original_class_read)
                with Image.open(job['path']) as opened:
                    image = ImageOps.exif_transpose(opened).convert('RGB')
                cfg = getattr(PBC3Config, job['preset'])(learned_filler_model_path=str(model), compute_final_mse=variant not in ('skip', 'both'))
                started = time.perf_counter()
                with contextlib.redirect_stdout(io.StringIO()):
                    result = PBC3.compress(image, config=cfg)
                encode = time.perf_counter() - started
                started = time.perf_counter()
                decoded = PBC3.decompress(result.data)
                decode = time.perf_counter() - started
                pixels = np.asarray(decoded.image.convert('RGB'))
                error = np.subtract(np.asarray(image), pixels, dtype=np.float32)
                _, body = PBC3._open_body(result.data)
                header = PBC3._read_header(BitReader(body))
                reply = dict(variant=variant, preset=job['preset'], trial=job.get('trial', -1),
                    dataset=Path(job['path']).parent.name, image=Path(job['path']).name,
                    input_sha256=digest(Path(job['path']).read_bytes()), config=vars(cfg), width=image.width, height=image.height,
                    working_width=header[3], working_height=header[4], patch_count=header[10],
                    final_bytes=len(result.data), raw_bytes=len(body) + 6, bpp=8 * len(result.data) / (image.width * image.height),
                    mse=float(np.mean(error ** 2, dtype=np.float64)), internal_mse=result.mse,
                    encode_seconds=encode, decode_seconds=decode, stream_sha256=digest(result.data),
                    pixels_sha256=digest(pixels.tobytes()), preview_equals_decode=bool(np.array_equal(np.asarray(result.image.convert('RGB')), pixels)))
            print(json.dumps(reply), flush=True)
        except Exception:
            import traceback
            print(json.dumps({'error': traceback.format_exc()}), flush=True)


def request(process, job):
    process.stdin.write(json.dumps(job) + '\n')
    process.stdin.flush()
    line = process.stdout.readline()
    if not line:
        raise RuntimeError('Benchmark worker exited')
    result = json.loads(line)
    if 'error' in result:
        raise RuntimeError(result['error'])
    return result


def run(args):
    import psutil
    if args.cpu is None:
        args.cpu = psutil.Process().cpu_affinity()[0]
    manifest = json.loads((HERE / 'hf_manifest.json').read_text())
    reference = Path(args.reference).resolve()
    assert module_hashes(ROOT) == manifest['current_modules_lf']
    assert module_hashes(reference) == manifest['reference_modules_lf']
    assert digest((ROOT / 'patch_policy.npz').read_bytes()) == manifest['model_sha256']
    assert digest((HERE / 'integration_candidate' / 'pbc3_stream.py').read_text().encode()) == manifest['candidate_lf']
    paths = [ROOT / p for p in manifest['inputs']]
    assert all(digest(p.read_bytes()) == manifest['inputs'][p.relative_to(ROOT).as_posix()] for p in paths)
    if args.smoke:
        paths = [p for p in paths if p.name == 'googlewiki.png']
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    metadata = dict(label=args.label, reference=str(reference), repeats=args.repeats, input_manifest=manifest,
        runtime='This label identifies the host; no HF or HTTP call is made by this script', warmup='one full encode/decode per variant/preset on googlewiki; fresh-process import excluded from primary timers')
    processes, logs, rows = {}, [], []
    try:
        for kind in ('reference', 'current'):
            log = (output / (kind + '_stderr.log')).open('w')
            logs.append(log)
            started = time.perf_counter()
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--worker', kind,
                '--reference', str(reference), '--cpu', str(args.cpu)], cwd=reference if kind == 'reference' else ROOT,
                env={**os.environ, **THREADS, 'PYTHONPATH': ''}, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True, bufsize=1)
            processes[kind] = process
            metadata[kind] = request(process, {'op': 'metadata'})
            metadata[kind]['worker_startup_seconds'] = time.perf_counter() - started
        variants = ('reference', 'baseline', 'grid', 'skip', 'both')
        metadata['warmup_rows'] = [request(processes['reference' if v == 'reference' else 'current'],
            dict(op='encode', variant=v, preset=preset, path=str(ROOT / 'hpt_data' / 'googlewiki.png')))
            for v in variants for preset in PRESETS]
        (output / 'metadata.json').write_text(json.dumps(metadata, indent=2))
        with (output / 'rows.jsonl').open('w') as handle:
            for trial in range(args.repeats):
                for image_i, path in enumerate(paths):
                    for preset_i, preset in enumerate(PRESETS):
                        shift = (trial + image_i + preset_i) % len(variants)
                        order = variants[shift:] + variants[:shift]
                        for variant in order:
                            row = request(processes['reference' if variant == 'reference' else 'current'],
                                dict(op='encode', variant=variant, preset=preset, path=str(path), trial=trial))
                            rows.append(row)
                            handle.write(json.dumps(row) + '\n')
                            handle.flush()
                    print(args.label, trial + 1, path.name, flush=True)
        for row in rows:
            if row['variant'] in ('grid', 'skip', 'both'):
                baseline = next(r for r in rows if (r['variant'], r['dataset'], r['image'], r['preset'], r['trial']) == ('baseline', row['dataset'], row['image'], row['preset'], row['trial']))
                assert all(row[k] == baseline[k] for k in ('stream_sha256', 'pixels_sha256', 'patch_count', 'mse', 'bpp'))
            assert row['preview_equals_decode']
        with (output / 'rows.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=[k for k in rows[0] if k != 'config'], extrasaction='ignore')
            writer.writeheader()
            writer.writerows(rows)
        (output / 'equivalence.json').write_text(json.dumps(dict(rows=len(rows), current_variants_exact=True,
            primary_timing='Full compress/decompress calls only; loading, metrics, framing inspection and IPC excluded', requests=0), indent=2))
    finally:
        for process in processes.values():
            if process.poll() is None:
                request(process, {'op': 'quit'})
                process.wait(timeout=30)
        for log in logs:
            log.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', choices=('reference', 'current'))
    parser.add_argument('--reference', default=str(ROOT / 'research_pass1' / 'release'))
    parser.add_argument('--output', default=str(HERE / 'hf_validation_results'))
    parser.add_argument('--label', default='LOCAL_UNSPECIFIED')
    parser.add_argument('--cpu', type=int)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    worker(args) if args.worker else run(args)
