import lzma
import struct
import zlib

from pbc4_stream import HEADER, MAX_DIMENSION, MAX_PAYLOAD, MAX_PIXELS, MAGIC

VERSION = 2
FLAGS = 5


def pack_v2(width, height, coarse_width, coarse_height, procedure_step,
            transform_step, rd_lambda, payload):
    if not (1 <= width <= MAX_DIMENSION and 1 <= height <= MAX_DIMENSION and width * height <= MAX_PIXELS):
        raise ValueError('image dimensions out of range')
    if not (1 <= coarse_width <= width and 1 <= coarse_height <= height):
        raise ValueError('coarse dimensions out of range')
    if not (1 <= procedure_step <= 255 and 1 <= transform_step <= 255 and 0 <= rd_lambda <= 65535):
        raise ValueError('invalid V4 parameters')
    if len(payload) > MAX_PAYLOAD:
        raise ValueError('payload too large')
    compressed = lzma.compress(payload, format=lzma.FORMAT_XZ, preset=6)
    header = HEADER.pack(MAGIC, VERSION, FLAGS, width, height, coarse_width,
                         coarse_height, 16, 0, procedure_step, transform_step,
                         rd_lambda, len(payload), len(compressed), zlib.crc32(payload) & 0xffffffff)
    return header + compressed


def unpack_v2(data):
    data = memoryview(data)
    if len(data) < HEADER.size:
        raise ValueError('truncated PBC4 v2 header')
    (magic, version, flags, width, height, coarse_width, coarse_height,
     region_size, coarse_mode, procedure_step, transform_step, rd_lambda,
     raw_size, compressed_size, checksum) = HEADER.unpack(data[:HEADER.size])
    if magic != MAGIC or version != VERSION or flags != FLAGS or coarse_mode != 0:
        raise ValueError('unsupported PBC4 v2 stream')
    if not (1 <= width <= MAX_DIMENSION and 1 <= height <= MAX_DIMENSION and width * height <= MAX_PIXELS):
        raise ValueError('image dimensions out of range')
    if not (1 <= coarse_width <= width and 1 <= coarse_height <= height and region_size == 16):
        raise ValueError('invalid coarse-grid geometry')
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
    return {
        'width': width, 'height': height, 'coarse_width': coarse_width,
        'coarse_height': coarse_height, 'procedure_step': procedure_step,
        'transform_step': transform_step, 'rd_lambda': rd_lambda,
    }, payload
