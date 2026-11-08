import json
import pathlib
import statistics
import sys
import time

import threadpoolctl

HERE = pathlib.Path(__file__).resolve().parents[1]
ROOT = HERE.parent
OUT = HERE / 'outputs'
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / 'research_pass42'))
import pbc4_codec as v1
import pbc4_codec_v2 as v2


def main():
    image = ROOT / 'hpt_data' / 'coast.jpg'
    source = v1.load_rgb(image)
    prefix = (OUT / 'coast_v31_prefix_003.pbc').read_bytes()
    expected_a = (ROOT / 'research_pass42' / 'outputs' / 'coast_v4_D_low.pbc').read_bytes()
    rows = {'warmups_per_encoder': 1, 'repetitions': 3, 'order': [], 'samples_seconds': {'A': [], 'B': []},
            'files': {}}
    with threadpoolctl.threadpool_limits(limits=1, user_api='blas'):
        warm_a = v1.encode(image, variant='D', point='low')['data']
        warm_b = v2.encode_from_prefix(source, prefix, variant='D', point='low')['data']
        if warm_a != expected_a:
            raise RuntimeError('V4-A timing run does not match stored Pass 42 control')
        rows['files']['A'] = {'bytes': len(warm_a), 'sha256': v1.hashlib.sha256(warm_a).hexdigest()}
        rows['files']['B'] = {'bytes': len(warm_b), 'sha256': v1.hashlib.sha256(warm_b).hexdigest()}
        for i in range(3):
            for label in (('A', 'B') if i % 2 == 0 else ('B', 'A')):
                start = time.perf_counter()
                data = (v1.encode(image, variant='D', point='low')['data'] if label == 'A'
                        else v2.encode_from_prefix(source, prefix, variant='D', point='low')['data'])
                elapsed = time.perf_counter() - start
                if label == 'A' and data != expected_a:
                    raise RuntimeError('V4-A timing stream changed')
                rows['order'].append(label)
                rows['samples_seconds'][label].append(elapsed)
    rows['median_seconds'] = {k: statistics.median(v) for k, v in rows['samples_seconds'].items()}
    (OUT / 'v2_encode_benchmark.json').write_text(json.dumps(rows, indent=2) + '\n')
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
