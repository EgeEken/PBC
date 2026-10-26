import base64
import io
import json
import os
import sys

from benchmark import ROOT, OUT, PRESETS, THREAD_ENV

os.environ.update(THREAD_ENV)
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image, ImageOps
from fastapi.testclient import TestClient
from PBC3 import PBC3
import pbc3_benchmark
import server


if __name__ == '__main__':
    client = TestClient(server.app)
    path = ROOT / 'hpt_data' / 'googlewiki.png'
    source = np.asarray(ImageOps.exif_transpose(Image.open(path)).convert('RGB'))
    normalized = io.BytesIO()
    Image.fromarray(source).save(normalized, format='PNG')
    rows = []
    for preset in PRESETS:
        response = client.post('/api/compress', data={'mode': 'Auto', 'auto_config': preset}, files={'image': ('googlewiki.png', normalized.getvalue(), 'image/png')})
        assert response.status_code == 200, response.text
        result = response.json()
        data = base64.b64decode(result['pbc_base64'])
        preview = np.asarray(Image.open(io.BytesIO(base64.b64decode(result['reconstructed_image'].split(',', 1)[1]))).convert('RGB'))
        decoded = np.asarray(PBC3.decompress(data).image.convert('RGB'))
        value = float(np.mean((source.astype(np.float64) - decoded.astype(np.float64)) ** 2))
        reference = (OUT / 'streams' / 'current' / 'hpt_data' / ('googlewiki.png.' + preset + '.pbc3')).read_bytes()
        assert data == reference
        assert np.array_equal(preview, decoded)
        assert result['mse'] == round(value, 2)
        assert result['compressed_kb'] == round(len(data) / 1024, 2)
        measured = pbc3_benchmark._eval_pbc3(source, preset)
        assert np.isclose(measured[2], 8 * len(data) / (source.shape[0] * source.shape[1]))
        assert np.isclose(measured[3], value, rtol=1e-6)
        rows.append({'preset': preset, 'api_bytes_equal_current_harness': True, 'api_preview_equals_decode': True,
                     'api_mse': result['mse'], 'decoded_mse': value, 'final_bytes': len(data),
                     'benchmark_bpp': measured[2], 'benchmark_mse': measured[3]})
    (OUT / 'backend_validation.json').write_text(json.dumps(rows, indent=2))
    response = client.post('/api/compress', data={'mode': 'Auto', 'auto_config': 'balanced'}, files={'image': ('googlewiki.png', path.read_bytes(), 'image/png')})
    assert response.status_code == 200, response.text
    result = response.json()
    data = base64.b64decode(result['pbc_base64'])
    decoded = np.asarray(PBC3.decompress(data).image.convert('RGBA')).astype(np.float64)
    source_rgba = np.asarray(ImageOps.exif_transpose(Image.open(path)).convert('RGBA')).astype(np.float64)
    rgba_mse = float(np.mean((source_rgba - decoded) ** 2))
    rgb_mse = float(np.mean((source_rgba[:, :, :3] - decoded[:, :, :3]) ** 2))
    assert result['mse'] == round(rgba_mse, 2)
    (OUT / 'backend_original_rgba.pbc3').write_bytes(data)
    (OUT / 'backend_alpha_observation.json').write_text(json.dumps({'image': path.name, 'preset': 'balanced', 'source_mode': 'RGBA',
                                                                 'alpha_extrema': [255, 255], 'displayed_mse': result['mse'],
                                                                 'decoded_rgba_mse': rgba_mse, 'decoded_rgb_mse': rgb_mse,
                                                                 'note': 'Website preserves opaque alpha; its MSE averages four channels whereas the RGB benchmark averages three.'}, indent=2))
    print('Four local API presets and benchmark metric checks passed')
