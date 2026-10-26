import argparse
from collections import Counter
from contextlib import contextmanager
import json
import lzma
import os
from pathlib import Path
import sys
import time

from common import ROOT, OUT, PASS1, SCREEN, PRESETS, THREAD_ENV, save, sha, snapshot

os.environ.update(THREAD_ENV)
sys.path.insert(0, str(ROOT))
import numpy as np
import psutil
from PBC3 import PBC3
from pbc3_types import BitReader
import pbc3_ops as ops
from stream_tools import parse, serialize
from optimizations import enabled

FILTERS = {
    'production_0extreme': PBC3._LZMA_FILTERS,
    'preset1': [{'id': lzma.FILTER_LZMA2, 'preset': 1}],
    'preset6': [{'id': lzma.FILTER_LZMA2, 'preset': 6}],
    'preset0extreme_dict64k': [{'id': lzma.FILTER_LZMA2, 'preset': lzma.PRESET_EXTREME, 'dict_size': 65536}],
}
MAGICS = {'horizontal_delta': b'P2D1', 'column_order': b'P2T1'}


def raw_fields(accounting):
    fields = [('image_header', 0, accounting['header_bits'])]
    for p in accounting['patches']:
        offset = p['start_bit']
        for name in ('geometry_bits', 'representation_flag_bits', 'mask_length_bits', 'mask_bits', 'bounds_bits', 'max_bitcount_bits', 'cell_size_bits', 'indices_bits'):
            fields.append((name, offset, p[name]))
            offset += p[name]
    return fields


def selected_streams(dataset, screen):
    for source in ('release', 'current'):
        for path in sorted((PASS1 / 'streams' / source / dataset).glob('*.pbc3')):
            image, preset, _ = path.name.rsplit('.', 2)
            if not screen or image in SCREEN:
                yield source, path, image, preset


def accounting():
    before = snapshot()
    rows, patch_rows, symbols = [], [], Counter()
    for source, path, image, preset in selected_streams('hpt_data', True):
        data = path.read_bytes()
        header, patches, a = parse(data)
        _, raw = PBC3._open_body(data)
        original_bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8))
        categories = sorted({name for name, _, _ in raw_fields(a)})
        impact = {}
        for category in categories:
            modified = original_bits.copy()
            for name, start, length in raw_fields(a):
                if name == category:
                    modified[start:start + length] = 0
            _, packed = PBC3._entropy_pack(np.packbits(modified).tobytes())
            impact[category] = len(data) - (len(packed) + 6)
        totals = {category: sum(length for name, _, length in raw_fields(a) if name == category) for category in categories}
        assert sum(totals.values()) + a['padding_bits'] + 48 == a['raw_bytes'] * 8
        for patch, ledger in zip(patches, a['patches']):
            idx = patch['indices'].astype(np.int32)
            pal = ops.palette_generator(patch['mask'], patch['max_bitcount'], patch['neg'], patch['pos'], header[8])
            horizontal = idx[:, 1:] - idx[:, :-1]
            vertical = idx[1:, :] - idx[:-1, :]
            flat = idx.ravel()
            row = dict(source=source, image=image, preset=preset, **ledger,
                zero_index_fraction=float(np.mean(flat == 0)), zero_value_fraction=float(np.mean(pal[flat] == 0)),
                horizontal_repeat_fraction=float(np.mean(horizontal == 0)) if horizontal.size else None,
                vertical_repeat_fraction=float(np.mean(vertical == 0)) if vertical.size else None,
                flat_repeat_fraction=float(np.mean(flat[1:] == flat[:-1])) if len(flat) > 1 else None,
                horizontal_correlation=float(np.corrcoef(idx[:, 1:].ravel(), idx[:, :-1].ravel())[0, 1]) if horizontal.size and idx[:, 1:].std() and idx[:, :-1].std() else None,
                vertical_correlation=float(np.corrcoef(idx[1:, :].ravel(), idx[:-1, :].ravel())[0, 1]) if vertical.size and idx[1:, :].std() and idx[:-1, :].std() else None)
            patch_rows.append(row)
            for value, count in zip(*np.unique(flat, return_counts=True)):
                symbols[(source, preset, patch['channel'], patch['bitcount'], int(value))] += int(count)
        rows.append(dict(source=source, image=image, preset=preset, final_bytes=len(data), raw_bytes=a['raw_bytes'],
            header=header, input_stream_sha256=sha(data), syntax_bits=totals, padding_bits=a['padding_bits'],
            zeroing_compressed_byte_effect_nonadditive=impact))
    save('bitstream_accounting', rows)
    save('index_statistics', patch_rows)
    save('symbol_distributions', [dict(source=s, preset=p, channel=c, bitcount=b, symbol=v, count=n) for (s, p, c, b, v), n in symbols.items()])
    save('entropy_accounting_provenance', dict(before=before, after=snapshot()))


def transform(indices, bits, method, inverse=False):
    indices = indices.astype(np.int32)
    if method == 'horizontal_delta':
        if inverse:
            return (np.cumsum(indices, axis=1) & ((1 << bits) - 1)).astype(np.uint16)
        result = indices.copy()
        result[:, 1:] -= indices[:, :-1]
        return (result & ((1 << bits) - 1)).astype(np.uint16)
    if method == 'column_order':
        return indices.ravel().reshape(indices.shape[::-1]).T.copy().astype(np.uint16) if inverse else indices.T.ravel().reshape(indices.shape).astype(np.uint16)
    raise ValueError(method)


class AlternateCodec(PBC3):
    method = None

    @classmethod
    def _read_patch(cls, br, channel_bits, positive_bias=True):
        c = br.read(channel_bits)
        x, y, w, h = (br.read(16) for _ in range(4))
        if br.read(1):
            raise ValueError('explicit palette is unsupported')
        mask = [br.read(1) for _ in range(br.read(10))]
        neg, pos, max_bits, cell = br.read(8), br.read(8), br.read(4), br.read(16)
        bits = ops.resolve_palette_bitcount(mask, max_bits, neg, pos, positive_bias)
        pal = ops.palette_generator(mask, max_bits, neg, pos, positive_bias)
        gw, gh = ops.ceil_div(w, cell), ops.ceil_div(h, cell)
        indices = PBC3._read_grid(br, gw * gh, bits).reshape(gh, gw)
        indices = transform(indices, bits, cls.method, inverse=True)
        return c, x, y, w, h, cell, pal[indices], bits


def encode_alternate(header, patches, method):
    changed = [{**p, 'indices': transform(p['indices'], p['bitcount'], method)} for p in patches]
    result = serialize(header, changed)
    return MAGICS[method] + result[4:]


def decode_alternate(data, method):
    AlternateCodec.MAGIC = MAGICS[method]
    AlternateCodec.method = method
    return AlternateCodec.decompress(data)


def evaluate(dataset, screen):
    before = snapshot()
    rows = []
    controls = [json.loads(line) for line in (PASS1 / 'baseline.jsonl').read_text().splitlines()]
    promotion = json.loads((OUT / 'entropy_promotion.json').read_text()) if not screen else None
    if not screen and not (promotion['representations'] or promotion['lzma']):
        raise RuntimeError('No entropy candidate was promoted; do not rerun control-only searches')
    psutil.Process().cpu_affinity([0])
    for source, path, image, preset in selected_streams(dataset, screen):
        data = path.read_bytes()
        header, patches, a = parse(data)
        _, raw = PBC3._open_body(data)
        baseline_pixels = np.asarray(PBC3.decompress(data).image)
        assert serialize(header, patches) == data
        representation_methods = ['control'] + (['horizontal_delta', 'column_order'] if screen else promotion['representations'])
        filter_methods = list(FILTERS) if screen else ['production_0extreme'] + promotion['lzma']
        for method in representation_methods[1:]:
            assert np.array_equal(np.asarray(decode_alternate(encode_alternate(header, patches, method), method).image), baseline_pixels)
        for method in filter_methods:
            packed = lzma.compress(raw, format=lzma.FORMAT_RAW, filters=FILTERS[method])
            assert lzma.decompress(packed, format=lzma.FORMAT_RAW, filters=FILTERS[method]) == raw
        for trial in range(3):
            methods = representation_methods
            shift = trial % len(methods)
            for method in methods[shift:] + methods[:shift]:
                started = time.perf_counter()
                alternative = serialize(header, patches) if method == 'control' else encode_alternate(header, patches, method)
                pack = time.perf_counter() - started
                started = time.perf_counter()
                decoded = PBC3.decompress(alternative) if method == 'control' else decode_alternate(alternative, method)
                unpack = time.perf_counter() - started
                equal = bool(np.array_equal(np.asarray(decoded.image), baseline_pixels))
                assert equal
                if method != 'control':
                    _, changed, _ = parse(b'PBC3' + alternative[4:])
                    assert all(np.array_equal(transform(q['indices'], q['bitcount'], method, True), p['indices']) for p, q in zip(patches, changed))
                    try:
                        PBC3.decompress(alternative)
                    except ValueError:
                        pass
                    else:
                        raise AssertionError('Production decoder accepted experimental magic')
                rows.append(dict(dataset=dataset, source=source, image=image, preset=preset, trial=trial,
                    kind='representation', method=method, baseline_bytes=len(data), final_bytes=len(alternative),
                    raw_bytes=a['raw_bytes'], bpp=8 * len(alternative) / ((header[1] or header[3]) * (header[2] or header[4])),
                    representation_pack_seconds=pack, decode_seconds=unpack, pixels_equal=equal,
                    semantic_equal=True, input_stream_sha256=sha(data)))
            filters = filter_methods
            shift = trial % len(filters)
            for method in filters[shift:] + filters[:shift]:
                started = time.perf_counter()
                compressed = lzma.compress(raw, format=lzma.FORMAT_RAW, filters=FILTERS[method])
                packed = compressed if len(compressed) < len(raw) else raw
                pack = time.perf_counter() - started
                started = time.perf_counter()
                unpacked = lzma.decompress(packed, format=lzma.FORMAT_RAW, filters=FILTERS[method]) if packed is compressed else packed
                unpack = time.perf_counter() - started
                assert unpacked == raw
                if method == 'production_0extreme':
                    assert packed == data[6:]
                # Raw LZMA2 dictionary size is supplied externally by the existing decoder.
                compatible = True
                try:
                    assert PBC3._entropy_unpack(2 if packed is compressed else 0, packed) == raw
                except (lzma.LZMAError, AssertionError):
                    compatible = False
                original_filters = PBC3._LZMA_FILTERS
                try:
                    PBC3._LZMA_FILTERS = FILTERS[method]
                    candidate_data = data[:5] + bytes((2 if packed is compressed else 0,)) + packed
                    started = time.perf_counter()
                    decoded = PBC3.decompress(candidate_data)
                    full_decode = time.perf_counter() - started
                finally:
                    PBC3._LZMA_FILTERS = original_filters
                assert np.array_equal(np.asarray(decoded.image), baseline_pixels)
                rows.append(dict(dataset=dataset, source=source, image=image, preset=preset, trial=trial,
                    kind='lzma', method=method, baseline_bytes=len(data), final_bytes=len(packed) + 6, raw_bytes=len(raw) + 6,
                    bpp=8 * (len(packed) + 6) / ((header[1] or header[3]) * (header[2] or header[4])),
                    entropy_pack_seconds=pack, entropy_unpack_seconds=unpack, raw_equal=True,
                    decode_seconds=full_decode, pixels_equal=True,
                    production_decoder_compatible=compatible, filters=FILTERS[method], input_stream_sha256=sha(data)))
        print('entropy', dataset, source, image, preset, flush=True)
    for row in rows:
        base = next(r for r in controls if (r['source'], r['dataset'], r['image'], r['preset'], r['trial']) == (row['source'], row['dataset'], row['image'], row['preset'], 0))
        row.update(phase='screen' if screen else 'confirmation', mse=base['mse'], input_sha256=base['input_sha256'],
            model_sha256=base['model_sha256'], source_revision=base['commit'], config=base['config'],
            initialization_patches=3, iterative_patches=base['patch_count'] - 3,
            original_dimensions=[base['width'], base['height']], working_dimensions=[base['working_width'], base['working_height']],
            frozen_full_encode_seconds=base['encode_seconds'], frozen_full_decode_seconds=base['decode_seconds'])
    prefix = 'entropy_screen' if screen else 'entropy_' + dataset
    save(prefix, rows)
    save(prefix + '_provenance', dict(before=before, after=snapshot(), filters=FILTERS,
        timing='three repeats rotated per case; untimed control and each alternate encode/decode plus each entropy filter; frozen search excluded'))


def memory(method):
    provenance_before = snapshot()
    path = max((PASS1 / 'streams' / 'current' / 'hpt_data').glob('*.pbc3'), key=lambda p: p.stat().st_size)
    _, raw = PBC3._open_body(path.read_bytes())
    sample = (raw * (4 * 1024 * 1024 // len(raw) + 1))[:4 * 1024 * 1024]
    process = psutil.Process()
    before = process.memory_info()
    compressor = lzma.LZMACompressor(format=lzma.FORMAT_RAW, filters=FILTERS[method])
    constructed = process.memory_info()
    output = compressor.compress(sample) + compressor.flush()
    after = process.memory_info()
    assert lzma.decompress(output, format=lzma.FORMAT_RAW, filters=FILTERS[method]) == sample
    prop = lzma._encode_filter_properties(FILTERS[method][0])[0]
    dictionary = (2 | (prop & 1)) << (prop // 2 + 11) if prop < 40 else 2**32 - 1
    save('memory_' + method, dict(filters=FILTERS[method], dictionary_bytes=dictionary,
        rss_before=before.rss, rss_constructed=constructed.rss, rss_after_pack=after.rss,
        process_peak_wset=getattr(after, 'peak_wset', None), sample_bytes=len(sample), packed_bytes=len(output),
        scope='Separate single-core process, 4 MiB repeated frozen raw body for allocation visibility; not a codec RD point',
        filter_properties_byte=prop, source_stream=str(path.relative_to(ROOT)), raw_body_sha256=sha(raw), sample_sha256=sha(sample)))
    save('memory_' + method + '_provenance', dict(before=provenance_before, after=snapshot(), filters=FILTERS[method]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('accounting', 'screen', 'main', 'holdout', 'memory'))
    parser.add_argument('--filter', choices=FILTERS)
    args = parser.parse_args()
    psutil.Process().cpu_affinity([0])
    if args.mode == 'memory':
        memory(args.filter)
    elif args.mode == 'accounting':
        accounting()
    else:
        evaluate('hpt_data_val' if args.mode == 'holdout' else 'hpt_data', args.mode == 'screen')
