from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import pbc33_codec as codec

HERE = Path(__file__).resolve().parent
OUT = HERE / 'outputs'
REPEATS = 3


def main() -> None:
    path = OUT / 'summary.json'
    data = json.loads(path.read_text())
    for name, image in data['images'].items():
        height, width = image['height'], image['width']
        n = ((height + 15) // 16) * ((width + 15) // 16)
        mode_len = (n * codec.MODE_BITS + 7) // 8
        for row in image['rows']:
            file_path = OUT / 'streams' / name / f"{row['config']}_q{row['qstep']}_l{row['lambda']:g}.p33s"
            stream = file_path.read_bytes()
            raw = __import__('lzma').decompress(stream[codec.HEADER.size:])
            modes = codec.unpack_modes(raw[:mode_len], n)
            params = raw[mode_len:]
            if len(stream) != row['enhancement_bytes'] or codec.sha(stream) != row['stream_sha256']:
                raise ValueError(f'candidate stream changed: {file_path}')
            samples = []
            for _ in range(REPEATS):
                start = time.perf_counter()
                encoded = codec._emit({'modes': modes, 'params': params}, width, height, row['qstep'])
                samples.append(time.perf_counter() - start)
                if encoded != stream:
                    raise AssertionError(f'non-deterministic candidate serialization: {file_path}')
            mean = statistics.mean(samples)
            emission = {'samples': samples, 'mean': mean, 'median': statistics.median(samples),
                        'cv_percent': 100 * statistics.pstdev(samples) / mean if mean else 0}
            row['serialization_seconds'] = emission
            row['encode_seconds_complete_amortized'] = row['family_analysis_seconds'] / len((0, 20, 100)) + emission['median']
    path.write_text(json.dumps(data, indent=2))
    print('profiled', sum(len(i['rows']) for i in data['images'].values()), 'complete files; deterministic serialization verified')


if __name__ == '__main__':
    main()
