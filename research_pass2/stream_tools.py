import hashlib

import numpy as np


def patch_signature(patch):
    h = hashlib.sha256()
    for key, value in patch.items():
        h.update(key.encode())
        h.update(value.tobytes() if isinstance(value, np.ndarray) else str(value).encode())
    return h.hexdigest()


def parse(data):
    from PBC3 import PBC3
    from pbc3_types import BitReader
    import pbc3_ops as ops

    _, body = PBC3._open_body(data)
    br = BitReader(body)
    header = PBC3._read_header(br)
    assert not header[12], 'Only single-stage streams are supported'
    header_bits = br.i * 8 - br.nbits
    patches, ledger = [], []
    for index in range(header[10]):
        start = br.i * 8 - br.nbits
        c = br.read(header[7])
        x, y, w, h = (br.read(16) for _ in range(4))
        assert br.read(1) == 0
        mask = [br.read(1) for _ in range(br.read(10))]
        neg, pos, max_bits, cell = br.read(8), br.read(8), br.read(4), br.read(16)
        bits = ops.resolve_palette_bitcount(mask, max_bits, neg, pos, header[8])
        gw, gh = ops.ceil_div(w, cell), ops.ceil_div(h, cell)
        indices = PBC3._read_grid(br, gw * gh, bits).reshape(gh, gw)
        patch = dict(channel=c, x=x, y=y, w=w, h=h, mask=mask, neg=neg, pos=pos,
            max_bitcount=max_bits, cell_size=cell, indices=indices, bitcount=bits)
        patches.append(patch)
        end = br.i * 8 - br.nbits
        ledger.append({'patch': index, 'start_bit': start, 'initialization': index < header[6], 'channel': c,
            'geometry_bits': 64 + header[7], 'representation_flag_bits': 1, 'mask_length_bits': 10,
            'mask_bits': len(mask), 'bounds_bits': 16, 'max_bitcount_bits': 4, 'cell_size_bits': 16,
            'indices_bits': gw * gh * bits, 'total_bits': end - start,
            'w': w, 'h': h, 'cell_size': cell, 'bitcount': bits, 'grid_cells': gw * gh})
    padding = len(body) * 8 - (br.i * 8 - br.nbits)
    assert 0 <= padding < 8
    return header, patches, {'wrapper_bits': 48, 'header_bits': header_bits, 'padding_bits': padding,
        'raw_bytes': len(body) + 6, 'patches': ledger}


def serialize(header, patches):
    from PBC3 import PBC3
    from pbc3_types import BitWriter
    bw = BitWriter()
    PBC3._write_header(bw, header[3], header[4], header[1], header[2], header[0],
        PBC3.COLOR_SPACES[header[5]], header[6], header[7], header[8], header[9], len(patches), header[11])
    for patch in patches:
        PBC3._write_patch(bw, patch, header[7])
    body = bw.finish()
    method, packed = PBC3._entropy_pack(body)
    return PBC3.MAGIC + bytes((PBC3.VERSION, method)) + packed
