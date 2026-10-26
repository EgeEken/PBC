import base64
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from research_pass5 import run_hf as api
from research_pass9.worker import THREADS
os.environ.update(THREADS)
api.HERE=HERE


def main():
    deployment=json.loads((HERE/'hf_deployment.json').read_text())
    meta=api.metadata()
    assert meta['sha']==meta['runtime']['sha']==deployment['commit']
    assert not (HERE/'public_api_response.json').exists(),'Fresh request required; preserve prior evidence before rerunning'
    api.public_smoke(None)
    result=json.loads((HERE/'public_api_smoke.json').read_text())
    prior=json.loads((ROOT/'research_pass8/public_api_smoke.json').read_text())
    assert result['stream_sha256']==prior['stream_sha256']
    old=Path(r'C:\Users\EGE\Desktop\Coding\Personal Projects\Completed Projects\Image Processing\Probabilistic Brush Compression\New Demo\HF Space\PBC\research_pass1\space_googlewiki.balanced.pbc3').read_bytes()
    assert hashlib.sha256(old).hexdigest()==prior['stream_sha256']
    boundary='PBCPass9V0Compatibility'
    body=(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="existing-v0.pbc3"\r\nContent-Type: application/octet-stream\r\n\r\n').encode()+old+f'\r\n--{boundary}--\r\n'.encode()
    request=urllib.request.Request(api.HOST+'/api/decode',data=body,headers={'Content-Type':'multipart/form-data; boundary='+boundary})
    with urllib.request.urlopen(request,timeout=120) as response:
        decoded=json.load(response)
    from PIL import Image
    import numpy as np
    from PBC3 import PBC3
    pixels=np.asarray(Image.open(io.BytesIO(base64.b64decode(decoded['reconstructed_image'].split(',')[-1]))))
    expected=np.asarray(PBC3.decompress(old).image)
    assert np.array_equal(pixels,expected)
    assert base64.b64decode(decoded['pbc_base64'])==old
    assert decoded['decoded'] and (decoded['width'],decoded['height'])==(expected.shape[1],expected.shape[0])
    api.save('public_compatibility',dict(hf_revision=meta['sha'],health=api.request('/api/health'),
        public_current_rng_stream_unchanged=True,public_internal_mse_enabled=True,
        prior_v0_stream_sha256=hashlib.sha256(old).hexdigest(),prior_v0_decode_pixels_equal=True,
        prior_v0_pixels_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(),dimensions=[decoded['width'],decoded['height']],
        prior_v0_source='Retained Pass 1 live Balanced RGBA response',public_requests=2,http_latency_used_for_benchmark=False))
    print('Fresh public compression, enabled MSE, unchanged stream and existing v0 API decode passed.',flush=True)


if __name__=='__main__':
    main()
