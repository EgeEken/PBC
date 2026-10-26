import argparse
import contextlib
import json
import os
import sys
import time
import tracemalloc

from support import ROOT, OUT, PASS1, PASS2, SCREEN, PRESETS, THREAD_ENV, images, save, snapshot, run_plan, read, sha


def plan(dataset):
    paths = [ROOT / 'hpt_data' / n for n in SCREEN] if dataset == 'screen' else images()
    cases = [dict(path=str(path), preset=preset, source='current', label=method,
        retain_stream=dataset == 'screen',
        optimization='grid_safe' if method in ('grid', 'both') else 'none',
        overrides={'compute_final_mse': False} if method in ('skip', 'both') else {})
        for path in paths for preset in PRESETS for method in ('baseline', 'grid', 'skip', 'both')]
    save('performance_' + dataset + '_plan', dict(decision=snapshot(), cases=cases, repeats=3))


def correctness():
    os.environ.update(THREAD_ENV)
    sys.path.insert(0, str(ROOT))
    import numpy as np
    import psutil
    from PIL import Image
    from PBC3 import PBC3
    from pbc3_types import BitReader, BitWriter, PBC3Config
    import pbc3_stream as stream
    from grid_io import read_grid, write_grid
    from variants import enabled
    from stream_tools import parse

    before = snapshot()
    psutil.Process().cpu_affinity([0])
    rng = np.random.default_rng(2003)
    unit = []
    for width in range(17):
        for offset in range(8):
            for count in (0, 1, 7, 9, 31, 1024):
                for layout in ('contiguous', 'strided'):
                    source = rng.integers(0, 1 << width, max(1, count * 2), dtype=np.uint32)
                    values = source[:count] if layout == 'contiguous' else source[:2 * count:2]
                    a, b = BitWriter(), BitWriter()
                    a.write((1 << offset) - 1, offset)
                    b.write((1 << offset) - 1, offset)
                    stream.write_grid(a, values, width)
                    write_grid(b, values, width)
                    a.write(37, 6)
                    b.write(37, 6)
                    assert a.finish() == b.finish()
                    a, b = BitReader(a.finish()), BitReader(b.finish())
                    a.read(offset)
                    b.read(offset)
                    assert np.array_equal(stream.read_grid(a, count, width), read_grid(b, count, width))
                    assert (a.i, a.acc, a.nbits) == (b.i, b.acc, b.nbits)
                    assert a.read(6) == b.read(6) == 37
                    unit.append(dict(width=width, offset=offset, count=count, layout=layout, equal=True))
    end_to_end = []
    for shape in ((1, 1), (3, 5), (17, 19), (97, 131)):
        for mode in ('RGB', 'RGBA_opaque', 'RGBA_varying'):
            channels = 3 if mode == 'RGB' else 4
            array = rng.integers(0, 256, (*shape, channels), dtype=np.uint8)
            if mode == 'RGBA_opaque':
                array[:, :, 3] = 255
            image = Image.fromarray(array)
            for limit in (250000, 4096):
                cfg = PBC3Config.quality(patch_count=3, auto_downsample_max_pixels=limit,
                    learned_filler_model_path=str(ROOT / 'patch_policy.npz'))
                baseline = PBC3.compress(image, config=cfg)
                if mode == 'RGB' and shape == (97, 131) and limit == 4096:
                    malformed_fixture = baseline.data
                expected = np.asarray(PBC3.decompress(baseline.data).image)
                for method in ('grid', 'skip', 'both'):
                    with enabled('grid_safe') if method in ('grid', 'both') else contextlib.nullcontext():
                        result = PBC3.compress(image, config=cfg, compute_final_mse=method not in ('skip', 'both'))
                        decoded = PBC3.decompress(result.data)
                    assert result.data == baseline.data and np.array_equal(np.asarray(decoded.image), expected)
                    assert np.array_equal(np.asarray(result.image), expected)
                    assert (result.mse is None) == (method in ('skip', 'both'))
                    end_to_end.append(dict(shape=shape, mode=mode, limit=limit, method=method, equal=True,
                        independent_rgba_mse=float(np.mean((array.astype(np.float64) - expected.astype(np.float64)) ** 2))))
    malformed = []
    palette_checks = []
    import pbc3_ops as ops
    for precision in (1, 9):
        bits = ops.resolve_palette_bitcount([1, 1, 1, 1], precision, 255, 255, True)
        patch = dict(channel=0, x=0, y=0, w=17, h=19, cell_size=1, mask=[1, 1, 1, 1],
            neg=255, pos=255, max_bitcount=precision, bitcount=bits,
            indices=rng.integers(0, 1 << bits, (19, 17), dtype=np.uint16))
        bw = BitWriter()
        PBC3._write_header(bw, 17, 19, None, None, False, 0, 3, 2, True, False, 1, [128] * 3)
        PBC3._write_patch(bw, patch, 2)
        payload = b'PBC3\x00\x00' + bw.finish()
        expected = np.asarray(PBC3.decompress(payload).image)
        with enabled('grid_safe'):
            decoded = PBC3.decompress(payload)
        assert np.array_equal(expected, np.asarray(decoded.image))
        palette_checks.append(dict(max_bitcount=precision, actual_bitcount=bits, equal=True))
    buffer_checks = []
    for values in (np.array([-2**63, -65536, -1, 0, 511, 65535, 65536, 2**63 - 1], dtype=np.int64),
                   np.array([0, 65535, 2**64 - 1], dtype=np.uint64)):
        for width in (1, 5, 9, 16):
            a, b = BitWriter(), BitWriter()
            stream.write_grid(a, values, width)
            write_grid(b, values, width)
            assert a.finish() == b.finish()
            buffer_checks.append(dict(kind=str(values.dtype) + '_extremes', width=width, equal=True))
    raw = b'\x12\x34\x56\x78'
    interleaved = b''.join(bytes((v, 0)) for v in raw)
    for kind, data in (('bytes', raw), ('bytearray', bytearray(raw)), ('memoryview', memoryview(raw)), ('strided_memoryview', memoryview(interleaved)[::2])):
        a, b = BitReader(data), BitReader(data)
        a.read(3)
        b.read(3)
        assert np.array_equal(stream.read_grid(a, 5, 5), read_grid(b, 5, 5))
        assert (a.i, a.acc, a.nbits) == (b.i, b.acc, b.nbits)
        buffer_checks.append(dict(kind=kind, equal=True))
    data = malformed_fixture
    _, raw = PBC3._open_body(data)
    stored = b'PBC3\x00\x00' + raw
    mutations = [(f'truncated_{n}', stored[:n]) for n in sorted({0, 1, 4, 5, 6, 7, len(stored) // 2, len(stored) - 1})]
    mutations += [('lzma_truncated', data[:-1]), ('bad_version', data[:4] + b'\x7f' + data[5:]), ('bad_entropy', data[:5] + b'\xff' + data[6:])]
    header, patches, _ = parse(data)
    br = BitReader(raw)
    PBC3._read_header(br)
    flag_position = br.i * 8 - br.nbits + header[7] + 64
    corrupt = bytearray(raw)
    corrupt[flag_position // 8] |= 1 << (7 - flag_position % 8)
    mutations.append(('explicit_palette_payload', b'PBC3\x00\x00' + bytes(corrupt)))
    bw = BitWriter()
    PBC3._write_header(bw, header[3], header[4], header[1], header[2], header[0], 1, header[6], header[7], header[8], header[9], 0, header[11])
    mutations.append(('zero_patch_count_valid', b'PBC3\x00\x00' + bw.finish()))
    for width, height, count in ((0, 19, 0), (17, 0, 0), (17, 19, 50)):
        bw = BitWriter()
        PBC3._write_header(bw, width, height, None, None, False, 0, 3, 2, True, False, count, [128] * 3)
        mutations.append((f'dimensions_count_{width}_{height}_{count}', b'PBC3\x00\x00' + bw.finish()))
    for field, replacement in (('cell_size', 0), ('channel', 3), ('w', 0), ('x', 65535)):
        changed = [{**p} for p in patches]
        changed[0][field] = replacement
        bw = BitWriter()
        PBC3._write_header(bw, header[3], header[4], header[1], header[2], header[0], 1, header[6], header[7], header[8], header[9], len(changed), header[11])
        try:
            for patch in changed:
                PBC3._write_patch(bw, patch, header[7])
            mutations.append((field + '_invalid', b'PBC3\x00\x00' + bw.finish()))
        except Exception as error:
            malformed.append(dict(case=field + '_writer', baseline_error=type(error).__name__))
    for name, payload in mutations:
        outcomes = []
        for method in ('baseline', 'grid'):
            try:
                with enabled('grid_safe') if method == 'grid' else contextlib.nullcontext():
                    decoded = PBC3.decompress(payload)
                outcome = {'accepted': True, 'shape': list(np.asarray(decoded.image).shape), 'pixels_sha256': sha(np.asarray(decoded.image).tobytes())}
            except Exception as error:
                outcome = {'accepted': False, 'error': type(error).__name__, 'message': str(error)}
            outcomes.append(outcome)
        assert outcomes[0].get('accepted') == outcomes[1].get('accepted'), (name, outcomes)
        if outcomes[0]['accepted']:
            assert outcomes[0] == outcomes[1]
        else:
            assert outcomes[0]['error'] == outcomes[1]['error'], (name, outcomes)
        malformed.append(dict(case=name, baseline=outcomes[0], candidate=outcomes[1]))
    for count, width in ((10**12, 9), (-1, 2), (1, -1), (1, 17)):
        try:
            read_grid(BitReader(b''), count, width)
        except (ValueError, EOFError) as error:
            malformed.append(dict(case='grid_arguments', count=count, width=width, candidate_error=type(error).__name__))
        else:
            raise AssertionError('Invalid grid arguments were accepted')
    save('performance_correctness', dict(bit_layout=unit, images=end_to_end, malformed=malformed, palette_extremes=palette_checks, buffers=buffer_checks))
    save('performance_correctness_provenance', dict(before=before, after=snapshot()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('correctness', 'screen_plan', 'full_plan'))
    args = parser.parse_args()
    if args.mode == 'correctness':
        correctness()
    else:
        plan(args.mode.split('_')[0])
