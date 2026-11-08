import hashlib
import json
import pathlib
import sys

import numpy as np
from PIL import Image, ImageOps

HERE = pathlib.Path(__file__).resolve().parents[1]
ROOT = HERE.parent
OUT = HERE / 'outputs'
sys.path.insert(0, str(ROOT / 'research_pass42'))
from pbc4_codec import encode

PASS42 = json.loads((ROOT / 'research_pass42' / 'outputs' / 'benchmark_raw.json').read_text())


def main():
    OUT.mkdir(exist_ok=True)
    rows = {}
    for name in ('googlewiki.png', 'cat.jpg', 'coast.jpg'):
        item = PASS42['v4'][f'{name}|D|low']
        path = ROOT / 'hpt_data' / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != PASS42['inputs'][name]['sha256']:
            raise RuntimeError(f'input hash mismatch: {name}')
        result = encode(path, variant='A', point='low', init_caps=(item['selected_init_cap'],))
        stream_path = OUT / f'{pathlib.Path(name).stem}_v4_A_low_matched.pbc'
        stream_path.write_bytes(result['data'])
        image_path = stream_path.with_suffix('.png')
        Image.fromarray(result['reconstruction']).save(image_path)
        rows[name] = {
            'input_sha256': PASS42['inputs'][name]['sha256'],
            'matched_init_cap': item['selected_init_cap'],
            'variant': 'A', 'point': 'low', 'bytes': len(result['data']),
            'bpp': result['bpp'] if 'bpp' in result else len(result['data']) * 8 / (result['width'] * result['height']),
            'rgb_mse': result['mse'], 'file_sha256': result['file_sha256'],
            'pixel_sha256': result['pixel_sha256'], 'mode_counts': result['mode_counts'],
            'byte_accounting': result['byte_accounting'],
            'stream_path': str(stream_path), 'image_path': str(image_path),
        }
        print(f"{name}: matched V4-A base {len(result['data']):,} bytes, MSE {result['mse']:.3f}", flush=True)
    (OUT / 'v4_base_probe.json').write_text(json.dumps(rows, indent=2, allow_nan=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
