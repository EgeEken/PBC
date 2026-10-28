import lzma
import struct
import numpy as np
from numba import njit

HEADER=struct.Struct('<4sQQ')
MAX_COEFFICIENTS=50331648


@njit(cache=True)
def _varint(data,offset,limit):
    value=0
    for shift in range(0,limit*7,7):
        if offset>=data.size:raise ValueError('truncated varint')
        byte=int(data[offset]);offset+=1
        value|=(byte&127)<<shift
        if byte<128:
            if shift and byte==0:raise ValueError('noncanonical varint')
            return value,offset
    raise ValueError('varint overflow')


@njit(cache=True)
def _pack_runs(q,nnz):
    output=np.empty(nnz*8,dtype=np.uint8)
    offset=0;previous=-1;position=0
    for c in range(3):
        for u in range(8):
            for v in range(8):
                for by in range(q.shape[0]):
                    for bx in range(q.shape[1]):
                        value=int(q[by,bx,c,u,v])
                        if value:
                            gap=position-previous-1
                            while gap>=128:
                                output[offset]=(gap&127)|128;offset+=1;gap>>=7
                            output[offset]=gap;offset+=1
                            signed=(value<<1)^(value>>15)
                            while signed>=128:
                                output[offset]=(signed&127)|128;offset+=1;signed>>=7
                            output[offset]=signed;offset+=1
                            previous=position
                        position+=1
    return output[:offset].copy()


@njit(cache=True)
def _unpack_runs(data,n,nnz):
    output=np.zeros(n,dtype=np.int16)
    offset=0;position=-1
    for _ in range(nnz):
        gap,offset=_varint(data,offset,5)
        value,offset=_varint(data,offset,3)
        position+=gap+1
        if position>=n or gap>4294967295:raise ValueError('position outside coefficients')
        if value==0 or value>65535:raise ValueError('invalid nonzero value')
        output[position]=(value>>1)^-(value&1)
    if offset!=data.size:raise ValueError('trailing coefficient bytes')
    return output


def pack(q):
    if q.dtype!=np.int16 or q.ndim!=5 or q.shape[2:]!=(3,8,8) or q.size>MAX_COEFFICIENTS:raise ValueError('coefficient shape/type')
    nnz=int(np.count_nonzero(q))
    return HEADER.pack(b'R21V',q.size,nnz)+_pack_runs(q,nnz).tobytes()


def counts(prefix,expected):
    if len(prefix)!=HEADER.size:raise ValueError('short sparse header')
    magic,n,nnz=HEADER.unpack(prefix)
    if magic!=b'R21V' or n!=expected or n>MAX_COEFFICIENTS or nnz>n:raise ValueError('sparse coefficient counts')
    return n,nnz


def unpack(raw,expected):
    n,nnz=counts(raw[:HEADER.size],expected)
    if not 2*nnz<=len(raw)-HEADER.size<=8*nnz:raise ValueError('sparse body size')
    return _unpack_runs(np.frombuffer(raw,dtype=np.uint8,offset=HEADER.size),n,nnz)


def decompress(payload,expected):
    if not 0<=expected<=MAX_COEFFICIENTS:raise ValueError('coefficient resource limit')
    reader=lzma.LZMADecompressor(format=lzma.FORMAT_XZ)
    prefix=reader.decompress(payload,max_length=HEADER.size)
    n,nnz=counts(prefix,expected)
    body=b'' if reader.eof else reader.decompress(b'',max_length=8*nnz+1)
    if not reader.eof or reader.unused_data or not 2*nnz<=len(body)<=8*nnz:raise ValueError('sparse XZ framing')
    return prefix+body
