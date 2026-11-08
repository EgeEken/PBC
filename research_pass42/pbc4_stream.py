import lzma
import struct
import zlib

HEADER = struct.Struct('<4sBBHHHHBBBBHIII')
MAGIC = b'PBC4'
VERSION = 1
MAX_DIMENSION = 8192
MAX_PIXELS = 50_000_000
MAX_PAYLOAD = 1_000_000_000


def pack_bits(values, bit_count):
    out = bytearray()
    acc = nbits = 0
    mask = (1 << bit_count) - 1
    for value in values:
        value = int(value)
        if value < 0 or value > mask:
            raise ValueError('value does not fit packed field')
        acc = (acc << bit_count) | value
        nbits += bit_count
        while nbits >= 8:
            nbits -= 8
            out.append((acc >> nbits) & 255)
            acc &= (1 << nbits) - 1
    if nbits:
        out.append((acc << (8 - nbits)) & 255)
    return bytes(out)


def unpack_bits(data, count, bit_count):
    if count < 0 or bit_count < 1 or bit_count > 16:
        raise ValueError('invalid packed field dimensions')
    need = (count * bit_count + 7) // 8
    if len(data) != need:
        raise ValueError('packed field length mismatch')
    out = [0] * count
    acc = nbits = pos = 0
    mask = (1 << bit_count) - 1
    for i in range(count):
        while nbits < bit_count:
            acc = (acc << 8) | data[pos]
            pos += 1
            nbits += 8
        nbits -= bit_count
        out[i] = (acc >> nbits) & mask
        acc &= (1 << nbits) - 1
    if acc:
        raise ValueError('nonzero packed-field padding')
    return out


def put_uleb(out, value):
    value = int(value)
    if value < 0:
        raise ValueError('negative ULEB128 value')
    while value >= 128:
        out.append((value & 127) | 128)
        value >>= 7
    out.append(value)


def get_uleb(data, pos, max_value=0x7fffffff):
    value = shift = 0
    for _ in range(5):
        if pos >= len(data):
            raise ValueError('truncated ULEB128')
        byte = data[pos]
        pos += 1
        value |= (byte & 127) << shift
        if not byte & 128:
            if value > max_value:
                raise ValueError('ULEB128 value out of range')
            return value, pos
        shift += 7
    raise ValueError('overlong ULEB128')


def put_sleb_zigzag(out, value):
    value = int(value)
    put_uleb(out, value * 2 if value >= 0 else -value * 2 - 1)


def get_sleb_zigzag(data, pos, max_abs=32767):
    code, pos = get_uleb(data, pos, max_abs * 2 + 1)
    value = code // 2 if code % 2 == 0 else -(code // 2) - 1
    if abs(value) > max_abs:
        raise ValueError('signed coefficient out of range')
    return value, pos


def pack_file(width, height, init_width, init_height, palette_count,
              procedure_step, transform_step, rd_lambda, payload, taper=False):
    if not (1 <= width <= MAX_DIMENSION and 1 <= height <= MAX_DIMENSION):
        raise ValueError('image dimensions out of range')
    if width * height > MAX_PIXELS:
        raise ValueError('image pixel count out of range')
    if not (1 <= init_width <= width and 1 <= init_height <= height):
        raise ValueError('initialization dimensions out of range')
    if not (1 <= palette_count <= 64 and 1 <= procedure_step <= 255 and 1 <= transform_step <= 255):
        raise ValueError('invalid codec parameter')
    if not (0 <= rd_lambda <= 65535 and len(payload) <= MAX_PAYLOAD):
        raise ValueError('rate or payload out of range')
    compressed = lzma.compress(payload, format=lzma.FORMAT_XZ, preset=6)
    flags = 1 | (16 if taper else 0)
    header = HEADER.pack(MAGIC, VERSION, flags, width, height, init_width,
                         init_height, 16, palette_count, procedure_step,
                         transform_step, rd_lambda, len(payload), len(compressed),
                         zlib.crc32(payload) & 0xffffffff)
    return header + compressed


def unpack_file(data):
    data = memoryview(data)
    if len(data) < HEADER.size:
        raise ValueError('truncated PBC4 header')
    (magic, version, flags, width, height, init_width, init_height,
     region_size, palette_count, procedure_step, transform_step, rd_lambda,
     raw_size, compressed_size, checksum) = HEADER.unpack(data[:HEADER.size])
    if magic != MAGIC or version != VERSION:
        raise ValueError('unsupported PBC4 format')
    if flags & 3 != 1 or flags & 0xe0:
        raise ValueError('unsupported codec flags')
    taper = bool(flags & 16)
    if flags & 0x0c:
        raise ValueError('unsupported color representation')
    if not (1 <= width <= MAX_DIMENSION and 1 <= height <= MAX_DIMENSION and width * height <= MAX_PIXELS):
        raise ValueError('image dimensions out of range')
    if not (1 <= init_width <= width and 1 <= init_height <= height):
        raise ValueError('initialization dimensions out of range')
    if region_size != 16 or not (1 <= palette_count <= 64):
        raise ValueError('unsupported grid or palette size')
    if not (1 <= procedure_step <= 255 and 1 <= transform_step <= 255):
        raise ValueError('invalid quantization step')
    if raw_size > MAX_PAYLOAD or compressed_size != len(data) - HEADER.size:
        raise ValueError('stream length mismatch')
    try:
        decoder = lzma.LZMADecompressor(format=lzma.FORMAT_XZ, memlimit=128 * 1024 * 1024)
        payload = decoder.decompress(data[HEADER.size:], max_length=raw_size + 1)
    except lzma.LZMAError as exc:
        raise ValueError('invalid XZ payload') from exc
    if not decoder.eof or decoder.unused_data or len(payload) != raw_size:
        raise ValueError('truncated or oversized XZ payload')
    if zlib.crc32(payload) & 0xffffffff != checksum:
        raise ValueError('payload checksum mismatch')
    header = {'width': width, 'height': height, 'init_width': init_width,
              'init_height': init_height, 'palette_count': palette_count,
              'procedure_step': procedure_step, 'transform_step': transform_step,
              'rd_lambda': rd_lambda, 'taper': taper}
    return header, payload
