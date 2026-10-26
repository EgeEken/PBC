import base64
import argparse
import hashlib
import io
import json
from pathlib import Path
import time
import urllib.request
import uuid

from PIL import Image, ImageOps
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
HOST = 'https://egeeken-pbc.hf.space'


def get_json(url):
    with urllib.request.urlopen(url, timeout=45) as response:
        return json.load(response)


def run(encode=False):
    metadata = get_json('https://huggingface.co/api/spaces/EgeEken/PBC')
    (OUT / 'space_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    revision = metadata['runtime']['sha']
    checks = {}
    for name in ('PBC3.py', 'pbc3_ops.py', 'pbc3_heads.py', 'pbc3_types.py', 'pbc3_features.py',
                 'pbc3_stream.py', 'pbc3_trace.py', 'learned_filler.py', 'patch_policy.npz', 'server.py', 'Dockerfile', 'requirements_docker.txt'):
        with urllib.request.urlopen('https://huggingface.co/spaces/EgeEken/PBC/resolve/' + revision + '/' + name, timeout=45) as response:
            remote = response.read()
        local = (ROOT / name).read_bytes()
        checks[name] = {'remote_sha256': hashlib.sha256(remote).hexdigest(), 'local_sha256': hashlib.sha256(local).hexdigest(),
                        'equal_normalized_newlines': remote.replace(b'\r\n', b'\n') == local.replace(b'\r\n', b'\n')}
    (OUT / 'deployment_source_checks.json').write_text(json.dumps({'revision': revision, 'files': checks}, indent=2))
    for name, path in (('space_health', '/api/health'), ('space_train_status', '/api/train/status'),
                       ('space_benchmark_status', '/api/benchmark/status'), ('space_quick_status', '/api/quick_rd/status')):
        (OUT / (name + '.json')).write_text(json.dumps(get_json(HOST + path), indent=2), encoding='utf-8')
    if not encode:
        print('Read-only deployment checks complete', revision)
        return
    rows = []
    for preset in ('compression', 'balanced', 'quality', 'high_quality'):
        boundary = uuid.uuid4().hex
        parts = []
        for key, value in (('mode', 'Auto'), ('auto_config', preset)):
            parts.append(('--' + boundary + '\r\nContent-Disposition: form-data; name="' + key + '"\r\n\r\n' + value + '\r\n').encode())
        parts.append(('--' + boundary + '\r\nContent-Disposition: form-data; name="image"; filename="googlewiki.png"\r\nContent-Type: image/png\r\n\r\n').encode())
        parts.append((ROOT / 'hpt_data' / 'googlewiki.png').read_bytes())
        parts.append(('\r\n--' + boundary + '--\r\n').encode())
        request = urllib.request.Request(HOST + '/api/compress', data=b''.join(parts), headers={'Content-Type': 'multipart/form-data; boundary=' + boundary})
        started = time.perf_counter()
        with urllib.request.urlopen(request, timeout=180) as response:
            result = json.load(response)
        elapsed = time.perf_counter() - started
        data = base64.b64decode(result.pop('pbc_base64'))
        preview = Image.open(io.BytesIO(base64.b64decode(result.pop('reconstructed_image').split(',', 1)[1]))).convert('RGB')
        source = ImageOps.exif_transpose(Image.open(ROOT / 'hpt_data' / 'googlewiki.png')).convert('RGB')
        result.update(preset=preset, stream_sha256=hashlib.sha256(data).hexdigest(), final_bytes=len(data),
                      pixels_sha256=hashlib.sha256(np.asarray(preview).tobytes()).hexdigest(),
                      measured_preview_mse=float(np.mean((np.asarray(source).astype(np.float64) - np.asarray(preview).astype(np.float64)) ** 2)),
                      http_wall_seconds=elapsed)
        (OUT / ('space_googlewiki.' + preset + '.pbc3')).write_bytes(data)
        rows.append(result)
        print(preset, result['stream_sha256'], result['mse'], result['time_seconds'], flush=True)
    (OUT / 'deployment_encodes.json').write_text(json.dumps(rows, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--encode', action='store_true')
    run(parser.parse_args().encode)
