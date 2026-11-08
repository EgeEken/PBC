import gc
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
import pbc4_codec
import pbc4_decode_opt

FILES = {
    'coast.jpg': ROOT / 'research_pass42' / 'outputs' / 'coast_v4_D_intermediate.pbc',
    'googlewiki.png': ROOT / 'research_pass42' / 'outputs' / 'googlewiki_v4_D_intermediate.pbc',
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def measured_decode(decoder, data):
    process = psutil.Process()
    baseline = process.memory_info().rss
    peak = {'rss': baseline}
    stop = threading.Event()

    def sample():
        while not stop.wait(0.005):
            try:
                peak['rss'] = max(peak['rss'], process.memory_info().rss)
            except psutil.Error:
                return

    monitor = threading.Thread(target=sample, daemon=True)
    monitor.start()
    start = time.perf_counter()
    try:
        pixels = decoder(data)
        elapsed = time.perf_counter() - start
    finally:
        stop.set()
        monitor.join()
    return pixels, elapsed, max(0, peak['rss'] - baseline)


def main():
    OUT.mkdir(exist_ok=True)
    output = {}
    process = psutil.Process()
    for name, path in FILES.items():
        data = path.read_bytes()
        decoders = {'ref': pbc4_codec.decode, 'opt': pbc4_decode_opt.decode}
        with threadpoolctl.threadpool_limits(limits=1, user_api='blas'):
            warm = {label: decoder(data) for label, decoder in decoders.items()}
            hashes = {label: sha(pixels.tobytes()) for label, pixels in warm.items()}
            if hashes['ref'] != hashes['opt']:
                raise RuntimeError(f'warmed decoder pixel mismatch: {name}')
            del warm
            gc.collect()
            samples = {'ref': [], 'opt': []}
            memory = {'ref': [], 'opt': []}
            final_hashes = {}
            for repetition in range(5):
                order = ('ref', 'opt') if repetition % 2 == 0 else ('opt', 'ref')
                for label in order:
                    pixels, seconds, delta = measured_decode(decoders[label], data)
                    samples[label].append(seconds)
                    memory[label].append(delta)
                    final_hashes[label] = sha(pixels.tobytes())
                    del pixels
                gc.collect()
        fresh = subprocess.run([sys.executable, str(HERE / 'scripts' / 'decoder_fresh_worker.py'),
                                str(path), str(OUT)], cwd=ROOT, capture_output=True, text=True)
        if fresh.returncode:
            raise RuntimeError(f'fresh decoder comparison failed {name}: {fresh.stderr}')
        fresh_rows = json.loads(fresh.stdout)
        if fresh_rows['ref']['sha256'] != fresh_rows['opt']['sha256'] or fresh_rows['ref']['sha256'] != final_hashes['ref']:
            raise RuntimeError(f'fresh-process pixel mismatch: {name}')
        record = {
            'stream_path': str(path), 'stream_sha256': sha(data),
            'warmups_per_decoder': 1, 'timing_repetitions': 5,
            'threadpool': {'openblas_threads': 1},
            'ref_seconds': samples['ref'], 'opt_seconds': samples['opt'],
            'ref_median_seconds': statistics.median(samples['ref']),
            'opt_median_seconds': statistics.median(samples['opt']),
            'speedup_median': statistics.median(samples['ref']) / statistics.median(samples['opt']),
            'ref_peak_observed_rss_delta_bytes': max(memory['ref']),
            'opt_peak_observed_rss_delta_bytes': max(memory['opt']),
            'ref_pixel_sha256': final_hashes['ref'], 'opt_pixel_sha256': final_hashes['opt'],
            'exact_pixels': final_hashes['ref'] == final_hashes['opt'],
            'fresh_process': fresh_rows,
        }
        output[name] = record
        (OUT / 'decoder_benchmark.json').write_text(json.dumps(output, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        print(f"{name}: REF {record['ref_median_seconds']:.3f}s, OPT {record['opt_median_seconds']:.3f}s, {record['speedup_median']:.2f}x", flush=True)


if __name__ == '__main__':
    main()
