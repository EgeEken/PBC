import json
import lzma
import struct
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research_pass21.common import *
from research_pass21 import sparse_hybrid as codec,zero_runs


def main():
    setup();checked=[]
    rows=json.loads((HERE/'files.json').read_text())['rows']
    for row in rows:
        data=(HERE/row['file']).read_bytes()
        assert len(data)==row['bytes'] and sha(data)==row['stream_sha256']
        params,base,payload=codec.split(data);w,h,step,n=params
        old,old_base,expected=frozen(row['image'],row['step'])
        assert base==old_base and sha(base)==row['base_sha256']
        q=codec.read_coefficients(payload,w,h,n)
        assert np.array_equal(q,coefficients(row['image'],row['step']))
        assert zero_runs.pack(q)==zero_runs.decompress(payload,n)
        output=codec.decode(data)
        assert np.array_equal(output,h17.decode(old)) and sha(output.tobytes())==row['pixels_sha256']==expected['pixels_sha256']
        image=load(row['image']);error=np.asarray(image,dtype=np.float32)-output.astype(np.float32)
        assert float(np.mean(error**2,dtype=np.float64))==row['rgb_mse']==expected['rgb_mse']
        assert (w,h)==image.size and zero_runs.HEADER.unpack(zero_runs.decompress(payload,n)[:20])[1:]==(n,row['nonzero_count'])
        dense=(ARTIFACTS/(row['image']+f'.q{int(step)}.dense_normal.p17')).read_bytes()
        assert sha(dense)==row['dense_normal_sha256'] and np.array_equal(h17.decode(dense),output)
        checked.append(dict(image=row['image'],step=step,bytes=len(data),coefficient_count=n,nonzero_count=row['nonzero_count'],pixels_sha256=row['pixels_sha256'],exact_coefficients=True,exact_rgb=True))
    arrays=[]
    arrays.append(np.zeros((0,0,3,8,8),dtype=np.int16))
    arrays.append(np.zeros((2,3,3,8,8),dtype=np.int16))
    endpoints=np.zeros((1,1000,3,8,8),dtype=np.int16);endpoints[0,0,0,0,0]=-32768;endpoints[0,-1,2,7,7]=32767;arrays.append(endpoints)
    rng=np.random.default_rng(2003)
    arrays.append(rng.integers(-32768,32768,(2,3,3,8,8),dtype=np.int16))
    for q in arrays:
        raw=zero_runs.pack(q);payload=lzma.compress(raw,format=lzma.FORMAT_XZ,preset=0)
        assert zero_runs.decompress(payload,q.size)==raw
        assert np.array_equal(zero_runs.unpack(raw,q.size),q.transpose(2,3,4,0,1).ravel())
        assert zero_runs.pack(q)==raw
    fixtures=[]
    for w,h in ((13,9),(35,37)):
        image=Image.fromarray(rng.integers(0,256,(h,w,3),dtype=np.uint8))
        for step in (32,128):
            sparse=codec.encode(image,step)
            assert sparse==codec.encode(image,step)
            output=codec.decode(sparse)
            dense=h17.encode(image,step)
            assert np.array_equal(output,h17.decode(dense)) and output.shape==(h,w,3)
            fixtures.append(dict(size=[w,h],step=step,stream_sha256=sha(sparse),pixels_sha256=sha(output.tobytes())))
    rejected=0
    raw_cases=[(zero_runs.HEADER.pack(b'BAD!',192,0),192),(zero_runs.HEADER.pack(b'R21V',193,0),192),(zero_runs.HEADER.pack(b'R21V',1,2)+b'\0\1\0\1',1),(zero_runs.HEADER.pack(b'R21V',zero_runs.MAX_COEFFICIENTS+1,0),zero_runs.MAX_COEFFICIENTS+1)]
    for body in (b'\0\0',b'\x80\0\1',b'\0\x81\0',b'\0\x80',b'\x80'*5+b'\1',b'\0\x80\x80\4',b'\xc0\1\1',b'\0\1\0'):
        raw_cases.append((zero_runs.HEADER.pack(b'R21V',192,1)+body,192))
    for raw,n in raw_cases:
        try:zero_runs.unpack(raw,n)
        except ValueError:rejected+=1
        else:raise AssertionError('invalid sparse raw accepted')
    empty=zero_runs.HEADER.pack(b'R21V',192,0)
    empty_payload=lzma.compress(empty,format=lzma.FORMAT_XZ,preset=0)
    for payload,n in ((empty_payload[:-1],192),(empty_payload+b'junk',192),(empty_payload+empty_payload,192),(lzma.compress(empty+b'\0'*1000000,format=lzma.FORMAT_XZ,preset=0),192),(empty_payload,zero_runs.MAX_COEFFICIENTS+1)):
        try:zero_runs.decompress(payload,n)
        except (ValueError,EOFError,lzma.LZMAError):rejected+=1
        else:raise AssertionError('invalid XZ/resource stream accepted')
    valid=(HERE/rows[0]['file']).read_bytes();fields=list(codec.HEADER.unpack_from(valid))
    invalid=[valid[:-1],valid+b'junk',valid[:10]]
    for index,value in ((0,b'BAD!'),(1,0),(1,65536),(1,4096),(5,float('nan')),(5,0),(6,4),(7,2)):
        altered=fields.copy();altered[index]=value
        if index==1 and value==4096:altered[2]=4097
        invalid.append(codec.HEADER.pack(*altered)+valid[codec.HEADER.size:])
    for data in invalid:
        try:codec.decode(data)
        except (ValueError,EOFError,lzma.LZMAError):rejected+=1
        else:raise AssertionError('invalid container accepted')
    class Oversized:
        def __len__(self):return codec.MAX_FILE_BYTES+1
    try:codec.split(Oversized())
    except ValueError:rejected+=1
    else:raise AssertionError('file resource limit ignored')
    for decoder in (PBC3.decompress,h17.decode):
        try:decoder(valid)
        except ValueError:rejected+=1
        else:raise AssertionError('foreign decoder accepted sparse magic')
    code="import sys,hashlib;sys.path.insert(0,sys.argv[1]);from learned_filler import LearnedFiller;LearnedFiller.load=classmethod(lambda *a,**k:(_ for _ in ()).throw(AssertionError('policy loaded')));from research_pass21.sparse_hybrid import decode;from pathlib import Path;print(hashlib.sha256(decode(Path(sys.argv[2]).read_bytes()).tobytes()).hexdigest())"
    for name in ('googlewiki.png','flower.jpg'):
        row=next(r for r in rows if r['image']==name)
        result=subprocess.check_output([sys.executable,'-c',code,str(ROOT),str(HERE/row['file'])],cwd=ROOT,text=True).strip()
        assert result==row['pixels_sha256']
    write_json('validation.json',dict(provenance=meta(),files=checked,dense_normal_files=6,raw_array_fixtures=len(arrays),raw_fixture_cases=['empty','all zero','long zero run with int16 endpoint extremes','dense random int16 extremes'],nonaligned_determinism=fixtures,malformed_resource_or_magic_rejections=rejected,fresh_process_file_only_policy_forbidden=2,production_unchanged=True))
    print('validated',len(checked),'sparse and six dense files, four raw fixtures, four repeated odd-size fixtures,',rejected,'rejections and two file-only subprocesses',flush=True)


if __name__=='__main__':main()
