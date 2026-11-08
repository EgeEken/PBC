import hashlib
import json
import pathlib
import statistics
import subprocess
import sys
import threading
import time

import psutil
import threadpoolctl

HERE = pathlib.Path(__file__).resolve().parents[1]
ROOT = HERE.parent
OUT = HERE / 'outputs'
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / 'research_pass42'))
import pbc4_codec_v2


def main():
    path = OUT / 'coast_v4_v2_intermediate_p3003.pbc'
    if not path.exists():
        path = OUT / 'coast_v4_v2_intermediate_p3003.pbc'
    data = path.read_bytes()
    with threadpoolctl.threadpool_limits(limits=1, user_api='blas'):
        warm, info = pbc4_codec_v2.decode(data, return_info=True)
        expected = hashlib.sha256(warm.tobytes()).hexdigest()
        samples = []
        rss_peaks = []
        process = psutil.Process()
        for _ in range(5):
            baseline = process.memory_info().rss
            peak = {'rss': baseline}
            stop = threading.Event()

            def sample():
                while not stop.wait(0.005):
                    peak['rss'] = max(peak['rss'], process.memory_info().rss)

            monitor = threading.Thread(target=sample, daemon=True)
            monitor.start()
            start = time.perf_counter()
            try:
                pixels, _ = pbc4_codec_v2.decode(data, return_info=True)
                samples.append(time.perf_counter() - start)
            finally:
                stop.set()
                monitor.join()
            rss_peaks.append(max(0, peak['rss'] - baseline))
            if hashlib.sha256(pixels.tobytes()).hexdigest() != expected:
                raise RuntimeError('repeated v2 decoding is nondeterministic')
            del pixels
    fresh = subprocess.run([sys.executable, str(HERE / 'scripts' / 'v2_fresh_worker.py'), str(path)],
                           cwd=ROOT, capture_output=True, text=True)
    if fresh.returncode:
        raise RuntimeError(fresh.stderr)
    fresh_row = json.loads(fresh.stdout)
    if fresh_row['pixel_sha256'] != expected or not fresh_row['skip_exact']:
        raise RuntimeError('fresh-process coarse decode mismatch or SKIP violation')
    row = {'stream_path': str(path), 'stream_sha256': hashlib.sha256(data).hexdigest(),
           'warmups': 1, 'repetitions': 5, 'samples_seconds': samples,
           'median_seconds': statistics.median(samples), 'pixel_sha256': expected,
           'peak_observed_rss_delta_bytes': max(rss_peaks),
           'mode_counts': info['mode_counts'], 'skip_exact': info['skip_exact'],
           'fresh_process': fresh_row}
    (OUT / 'v2_decode_benchmark.json').write_text(json.dumps(row, indent=2) + '\n')
    print(json.dumps(row, indent=2))


if __name__ == '__main__':
    main()
