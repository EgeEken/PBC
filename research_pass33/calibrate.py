from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

import pbc33_codec as codec

HERE = Path(__file__).resolve().parent
OUT = HERE / 'outputs'
LAMBDA_SCREEN = (100.0, 400.0, 1600.0)


def main() -> None:
    manifest = json.loads((OUT / 'input_manifest.json').read_text())
    info = next(r for r in manifest['images'] if r['image'] == 'googlewiki.png')
    source = np.asarray(ImageOps.exif_transpose(Image.open(info['source_path'])).convert('RGB'), np.uint8)
    base = np.load(info['base_npy_path'], allow_pickle=False)
    rows = []
    for result in codec.encode_family(source, base, 'S', LAMBDA_SCREEN, 64):
        rec, modes, params = codec.decode(result['data'], base)
        if not np.array_equal(rec, result['reconstruction']) or not np.array_equal(modes, result['modes']) or params != result['params']:
            raise AssertionError('calibration candidate did not roundtrip')
        path = OUT / 'calibration' / f"googlewiki_S_q64_l{result['lambda']:g}.p33s"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(result['data'])
        err = source.astype(np.int32) - rec.astype(np.int32)
        rows.append({'lambda': result['lambda'], 'enhancement_bytes': len(result['data']),
            'total_bytes': info['base_bytes'] + len(result['data']),
            'total_bpp': 8 * (info['base_bytes'] + len(result['data'])) / (source.shape[0] * source.shape[1]),
            'mse': float(np.square(err, dtype=np.int64).mean()), 'mode_counts': result['mode_counts'],
            'active_blocks': result['active_blocks'], 'coefficients': result['coefficients'],
            'stream_sha256': codec.sha(result['data']), 'pixels_sha256': codec.sha(rec.tobytes()),
            'fresh_decode_in_process': True})
    record = {'image': 'googlewiki.png', 'role': 'single-image development calibration; not independent holdout',
        'qstep': 64, 'lambda_candidates': list(LAMBDA_SCREEN), 'rule': 'select three logarithmically spaced penalties spanning the A/B total-byte interval; preserve the lowest-penalty S point, then select the two higher penalties that bring S into or near the measured A/B range',
        'A_B_total_bytes': [83207, 90967], 'rows': rows}
    path = OUT / 'calibration' / 'rate_screen.json'
    path.write_text(json.dumps(record, indent=2) + '\n')
    for row in rows:
        print(row['lambda'], row['total_bytes'], round(row['mse'], 3), row['active_blocks'], row['coefficients'])


if __name__ == '__main__':
    main()
