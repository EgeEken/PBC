import io
import struct
import numpy as np
from PIL import Image
from PBC3 import PBC3


def encode(prep,quality):
    image=Image.fromarray(prep['arr'],prep['color_space'])
    stream=io.BytesIO();image.save(stream,format='JPEG',quality=quality,subsampling=2,optimize=True,progressive=False)
    data=stream.getvalue();comment=b'P25REF'+struct.pack('>HH',prep['original_w'],prep['original_h'])
    return data[:2]+b'\xff\xfe'+struct.pack('>H',len(comment)+2)+comment+data[2:]


def decode_working(data):
    if data[:6]!=b'\xff\xd8\xff\xfe\0\x0c' or data[6:12]!=b'P25REF':raise ValueError('JPEG reference size metadata')
    size=struct.unpack_from('>HH',data,12)
    with Image.open(io.BytesIO(data)) as image:working=image.convert('RGB')
    return working,size


def decode(data):
    working,size=decode_working(data)
    if working.size!=size:working=working.resize(size,PBC3.RESAMPLE_FILTER,reducing_gap=PBC3.RESAMPLE_REDUCING_GAP)
    return working
