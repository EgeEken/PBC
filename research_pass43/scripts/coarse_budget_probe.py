import json
import pathlib
import sys
import time

from PIL import Image

HERE = pathlib.Path(__file__).resolve().parents[1]
ROOT = HERE.parent
OUT = HERE / 'outputs'
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / 'research_pass42'))
import pbc4_codec_v2 as v2


def main():
    stage = json.loads((OUT / 'v3_stage_probe.json').read_text())
    rows = {}
    for name in ('googlewiki.png', 'cat.jpg', 'coast.jpg'):
        stem = pathlib.Path(name).stem
        source_path = ROOT / 'hpt_data' / name
        source = v2.v1.load_rgb(source_path)
        row = stage[name]
        rows[name] = []
        for requested in (3, 6, 13):
            count = min(requested, row['init_patch_count'] + row['refinement_patch_count'])
            p3 = (OUT / f'{stem}_v31_prefix_{count:03d}.pbc').read_bytes()
            start = time.perf_counter()
            result = v2.encode_from_prefix(source, p3, variant='A', point='low')
            elapsed = time.perf_counter() - start
            path = OUT / f'{stem}_v4_v2_low_base_p3{count:03d}.pbc'
            path.write_bytes(result['data'])
            Image.fromarray(result['reconstruction']).save(OUT / f'{stem}_v4_v2_low_base_p3{count:03d}.png')
            metrics = {k: val for k, val in result.items() if k not in ('data', 'reconstruction', 'base')}
            metrics.update({'coarse_patch_count': count, 'encode_seconds': elapsed, 'stream_path': str(path)})
            rows[name].append(metrics)
            print(f'{name} A low coarse={count}: {result["bytes"]} B MSE={result["mse"]:.3f} {elapsed:.2f}s', flush=True)
    (OUT / 'coarse_budget_probe.json').write_text(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
