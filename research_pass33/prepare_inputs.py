from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
NAMES = ('googlewiki.png', 'mario64.png', 'flower.jpg', 'cat.jpg', 'cathedral.png', 'park.jpg')


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    p29 = json.loads((ROOT / 'research_pass29' / 'outputs' / 'results.json').read_text())
    p31 = json.loads((ROOT / 'research_pass31' / 'outputs' / 'summary.json').read_text())
    p32 = json.loads((ROOT / 'research_pass32' / 'outputs' / 'manifest.json').read_text())
    bases = {r['image']: r for r in p29['images']}
    later_bases = {r['image']: r for r in p32['images']}
    output = {'pass29_head': p29['head'], 'pass31_head': p31['provenance']['starting_head'], 'images': []}
    for name in NAMES:
        source = ROOT / 'hpt_data' / name
        source_sha = digest(source.read_bytes())
        image = ImageOps.exif_transpose(Image.open(source)).convert('RGB')
        src = np.asarray(image, np.uint8)
        if name in bases:
            expected_p29 = bases[name]
            expected_p31 = p31['images'][name]
            stream = ROOT / 'research_pass29' / 'outputs' / 'bases' / f'{name}.pbc3'
            base_npy = ROOT / 'research_pass31' / 'outputs' / 'bases' / f'{name}.npy'
            source_expected = expected_p29['base']['source_sha256']
            stream_expected = expected_p29['base']['sha256']
            base_mse = expected_p31['base_mse']['mse']
            decoded_base_sha = None
            dims_expected = expected_p29['dimensions']
            if source_sha != expected_p31['source_sha256'] or stream_expected != expected_p31['base_sha256']:
                raise ValueError(f'Pass 29/31 base provenance disagreement for {name}')
            base_pass = 29
        else:
            expected_p32 = later_bases[name]
            stream = Path(expected_p32['stream'])
            base_npy = Path(expected_p32['base_npy'])
            source_expected = expected_p32['source_sha256']
            stream_expected = expected_p32['stream_sha256']
            base_mse = expected_p32['base_mse']
            decoded_base_sha = expected_p32['decoded_base_sha256']
            dims_expected = [expected_p32['width'], expected_p32['height']]
            base_pass = 32
        stream_data = stream.read_bytes()
        stream_sha = digest(stream_data)
        base = np.load(base_npy, allow_pickle=False)
        if (source_sha != source_expected or stream_sha != stream_expected or base.shape != src.shape or
            dims_expected != [image.width, image.height]):
            raise ValueError(f'frozen input provenance mismatch for {name}')
        decoded_sha = digest(np.ascontiguousarray(base).tobytes())
        if decoded_base_sha and decoded_sha != decoded_base_sha:
            raise ValueError(f'frozen decoded base hash mismatch for {name}')
        output['images'].append({'image': name, 'source_path': str(source), 'source_sha256': source_sha,
            'width': image.width, 'height': image.height, 'base_stream_path': str(stream),
            'base_stream_sha256': stream_sha, 'base_bytes': len(stream_data), 'base_npy_path': str(base_npy),
            'decoded_base_rgb_sha256': decoded_sha, 'base_mse': base_mse, 'base_source_pass': base_pass,
            'source_shape': list(src.shape)})
    out = HERE / 'outputs' / 'input_manifest.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2) + '\n')
    print(out)
    for row in output['images']:
        print(row['image'], row['width'], row['height'], row['base_bytes'], row['base_stream_sha256'])


if __name__ == '__main__':
    main()
