import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import *
from container import decode,HEADER,RECORD
import subprocess
import zlib
import io
results=json.loads((PASS/'video/results.json').read_text())
actual=json.loads(subprocess.check_output([sys.executable,str(PASS/'video/decode_worker.py')],text=True))
for row in actual:
    match=next(r for r in results if row['file']==r['clip']+'.'+r['mode']+'.p26v')
    assert row['frame_hashes']==match['frame_hashes']
source=(PASS/'artifacts/video/moving_shape.temporal.p26v').read_bytes()
bad=[b'',source[:10],source[:-1],source+b'X',b'xxxx'+source[4:],source[:4]+b'\x02'+source[5:]]
for index in (5,7,9,11,13,15):
    b=bytearray(source);b[index:index+2]=b'\0\0';bad.append(bytes(b))
b=bytearray(source);b[HEADER.size+RECORD.size]^=1;bad.append(bytes(b))
header=HEADER.pack(b'P26V',1,1,1,1,12,1,8)
for kind,payload in ((1,zlib.compress(bytes(6))),(9,b'x'),(0,b'PBC3\0\x02x'),(2,b'invalid PNG')):
    bad.append(header+RECORD.pack(kind,len(payload),zlib.crc32(payload))+payload)
png=io.BytesIO();Image.fromarray(np.array([[[255,0,128]]],np.uint8)).save(png,format='PNG')
key=png.getvalue()
head=HEADER.pack(b'P26V',1,1,1,2,12,1,8)
def fixture(delta):
    payload=zlib.compress(np.array(delta,dtype='<i2').tobytes())
    return head+RECORD.pack(2,len(key),zlib.crc32(key))+key+RECORD.pack(1,len(payload),zlib.crc32(payload))+payload
extreme,_=decode(fixture([-255,255,-128]))
assert extreme[-1].tolist()==[[[0,255,0]]]
bad.extend((fixture([-256,0,0]),fixture([1,0,0]),fixture([-32768,32767,0])))
count=0
for data in bad:
    try: decode(data)
    except (ValueError,OSError,EOFError): count+=1
    else: raise AssertionError('accepted malformed video')
save('video/validation.json',dict(file_only_model_and_encoder_forbidden=actual,malformed_rejected=count,signed_endpoint_fixture=True,all_frame_hashes_exact=True))
print('video validation passed',len(actual),count)
