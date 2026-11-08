import hashlib
import json
import pathlib
import sys
import time

import numpy as np
from PIL import Image

HERE = pathlib.Path(__file__).resolve().parents[1]
ROOT = HERE.parent
OUT = HERE / 'outputs'
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / 'research_pass42'))
import pbc4_codec as v1
import pbc4_codec_v2 as v2

IMAGES = ('googlewiki.png', 'cat.jpg', 'coast.jpg')
PREFIX_COUNTS = (3, 6, 13)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def save_png(path, pixels):
    Image.fromarray(pixels, 'RGB').save(path)


def main():
    OUT.mkdir(exist_ok=True)
    stage = json.loads((OUT / 'v3_stage_probe.json').read_text())
    results = {}
    for image_name in IMAGES:
        source_path = ROOT / 'hpt_data' / image_name
        source = v1.load_rgb(source_path)
        stem = pathlib.Path(image_name).stem
        results[image_name] = {'source_sha256': sha(source_path.read_bytes()), 'points': {}}
        for point in ('low', 'intermediate'):
            p3_row = stage[image_name]
            candidates = []
            for requested in PREFIX_COUNTS:
                count = min(requested, p3_row['init_patch_count'] + p3_row['refinement_patch_count'])
                prefix_file = OUT / f'{stem}_v31_prefix_{count:03d}.pbc'
                if not prefix_file.exists():
                    count = max(k for k in PREFIX_COUNTS if (OUT / f'{stem}_v31_prefix_{k:03d}.pbc').exists())
                    prefix_file = OUT / f'{stem}_v31_prefix_{count:03d}.pbc'
                start = time.perf_counter()
                result = v2.encode_from_prefix(source, prefix_file.read_bytes(), 'D', point)
                elapsed = time.perf_counter() - start
                out_path = OUT / f'{stem}_v4_v2_{point}_p3{count:03d}.pbc'
                out_path.write_bytes(result['data'])
                save_png(OUT / f'{stem}_v4_v2_{point}_p3{count:03d}.png', result['reconstruction'])
                candidates.append({k: val for k, val in result.items() if k not in ('data', 'reconstruction', 'base')})
                candidates[-1].update({'coarse_patch_count': count, 'encode_seconds': elapsed,
                                       'stream_path': str(out_path), 'base_mse': result['base_mse']})
                print(f'{image_name} {point} coarse={count}: {result["bytes"]} B MSE={result["mse"]:.3f} encode={elapsed:.2f}s', flush=True)
            # Complete-file score includes the selected coarse syntax and all native modes.
            chosen = min(candidates, key=lambda x: x['score'])
            results[image_name]['points'][point] = {'candidates': candidates, 'selected': chosen}

        # Verify the disabled mechanism directly reproduces the stored Pass 42 control.
        stored = (ROOT / 'research_pass42' / 'outputs' / f'{stem}_v4_D_low.pbc').read_bytes()
        control = v1.encode(source_path, variant='D', point='low')['data']
        if control != stored:
            raise RuntimeError(f'V1 disabled control changed for {image_name}')
        results[image_name]['v1_control_sha256'] = sha(control)
        results[image_name]['v1_control_exact'] = True
    (OUT / 'v2_rd_comparison.json').write_text(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
