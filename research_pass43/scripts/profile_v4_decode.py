import hashlib
import json
import pathlib
import sys
import threading
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parents[1]
ROOT = HERE.parent
OUT = HERE / 'outputs'
FILES = {
    'coast.jpg': ROOT / 'research_pass42' / 'outputs' / 'coast_v4_D_intermediate.pbc',
    'googlewiki.png': ROOT / 'research_pass42' / 'outputs' / 'googlewiki_v4_D_intermediate.pbc',
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    import psutil
    import threadpoolctl
    sys.path.insert(0, str(ROOT / 'research_pass42'))
    import pbc4_codec as codec
    import pbc4_stream as stream

    rows = {}
    proc = psutil.Process()
    for name, path in FILES.items():
        data = path.read_bytes()
        header, payload = stream.unpack_file(data)
        count = ((header['width'] + 15) // 16) * ((header['height'] + 15) // 16)
        idx_len = (header['init_width'] * header['init_height'] * max(1, (header['palette_count'] - 1).bit_length()) + 7) // 8
        mode_pos = 1 + header['palette_count'] * 3 + idx_len
        modes = np.asarray(stream.unpack_bits(payload[mode_pos:mode_pos + (count * 2 + 7) // 8], count, 2), dtype=np.uint8)
        mode_counts = np.bincount(modes, minlength=4).tolist()
        target = codec.decode(data)
        del target

        stages = {key: 0.0 for key in ('unpack_total', 'header_unpack', 'xz_decompress', 'bit_unpack',
                                      'bilinear_upsample', 'procedural_surface', 'coefficient_parse',
                                      'inverse_matmul', 'taper', 'decode_other')}
        counts = {key: 0 for key in ('bit_unpack_calls', 'resize_calls', 'surface_calls',
                                    'coefficient_reads', 'matmul_calls', 'taper_calls')}
        original = {key: getattr(codec, key) for key in ('unpack_file', 'unpack_bits', '_resize_bilinear',
                    '_surface_from_controls', '_read_coefficients', '_taper_correction')}
        original_matmul = np.matmul
        original_decompressor = stream.lzma.LZMADecompressor

        def timed(name, func, counter=None):
            def wrapped(*args, **kwargs):
                t = time.perf_counter()
                try:
                    return func(*args, **kwargs)
                finally:
                    stages[name] += time.perf_counter() - t
                    if counter:
                        counts[counter] += 1
            return wrapped

        class TimedDecompressor:
            def __init__(self, *args, **kwargs):
                t = time.perf_counter()
                self.inner = original_decompressor(*args, **kwargs)
                stages['xz_decompress'] += time.perf_counter() - t

            def decompress(self, *args, **kwargs):
                t = time.perf_counter()
                try:
                    return self.inner.decompress(*args, **kwargs)
                finally:
                    stages['xz_decompress'] += time.perf_counter() - t

            def __getattr__(self, key):
                return getattr(self.inner, key)

        def timed_unpack_file(data):
            t = time.perf_counter()
            h = stream.HEADER.unpack(memoryview(data)[:stream.HEADER.size])
            stages['header_unpack'] += time.perf_counter() - t
            t = time.perf_counter()
            try:
                return original['unpack_file'](data)
            finally:
                stages['unpack_total'] += time.perf_counter() - t

        def timed_matmul(*args, **kwargs):
            t = time.perf_counter()
            try:
                return original_matmul(*args, **kwargs)
            finally:
                stages['inverse_matmul'] += time.perf_counter() - t
                counts['matmul_calls'] += 1

        codec.unpack_file = timed_unpack_file
        codec.unpack_bits = timed('bit_unpack', original['unpack_bits'], 'bit_unpack_calls')
        codec._resize_bilinear = timed('bilinear_upsample', original['_resize_bilinear'], 'resize_calls')
        codec._surface_from_controls = timed('procedural_surface', original['_surface_from_controls'], 'surface_calls')
        codec._read_coefficients = timed('coefficient_parse', original['_read_coefficients'], 'coefficient_reads')
        codec._taper_correction = timed('taper', original['_taper_correction'], 'taper_calls')
        np.matmul = timed_matmul
        stream.lzma.LZMADecompressor = TimedDecompressor

        stop = threading.Event()
        memory = {'baseline_bytes': proc.memory_info().rss, 'peak_bytes': proc.memory_info().rss}

        def sample_memory():
            while not stop.wait(0.005):
                try:
                    memory['peak_bytes'] = max(memory['peak_bytes'], proc.memory_info().rss)
                except psutil.Error:
                    break

        monitor = threading.Thread(target=sample_memory, daemon=True)
        monitor.start()
        start = time.perf_counter()
        try:
            with threadpoolctl.threadpool_limits(limits=1, user_api='blas'):
                decoded = codec.decode(data)
            elapsed = time.perf_counter() - start
        finally:
            stop.set()
            monitor.join()
            for key, value in original.items():
                setattr(codec, key, value)
            np.matmul = original_matmul
            stream.lzma.LZMADecompressor = original_decompressor
        pixel_hash = sha(decoded.tobytes())
        stage_sum = stages['unpack_total'] + stages['bit_unpack'] + stages['bilinear_upsample'] + stages['procedural_surface'] + stages['coefficient_parse'] + stages['inverse_matmul'] + stages['taper']
        stages['decode_other'] = max(0.0, elapsed - stage_sum)
        stages['header_unpack'] = min(stages['header_unpack'], stages['unpack_total'])
        rows[name] = {
            'stream_path': str(path), 'stream_sha256': sha(data),
            'dimensions': [header['width'], header['height']], 'file_bytes': len(data),
            'mode_counts': {'skip': mode_counts[0], 'procedural': mode_counts[1], 'transform': mode_counts[2]},
            'transform_block_count': mode_counts[2] * 4,
            'inverse_block_channel_count': mode_counts[2] * 12,
            'instrumented_decode_seconds': elapsed, 'stages_seconds': stages,
            'stages_fraction': {k: v / elapsed for k, v in stages.items() if k not in ('unpack_total', 'header_unpack', 'decode_other')},
            'operation_counts': counts,
            'rss_before_bytes': memory['baseline_bytes'], 'rss_peak_bytes': memory['peak_bytes'],
            'rss_observed_delta_bytes': max(0, memory['peak_bytes'] - memory['baseline_bytes']),
            'instrumented_pixel_sha256': pixel_hash,
        }
        print(f"{name}: {elapsed:.3f}s instrumented; matmul {stages['inverse_matmul']:.3f}s / {counts['matmul_calls']} calls; coeff parse {stages['coefficient_parse']:.3f}s", flush=True)
    (OUT / 'decoder_profile.json').write_text(json.dumps(rows, indent=2, allow_nan=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
