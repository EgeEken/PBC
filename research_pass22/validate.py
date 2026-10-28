import lzma
import subprocess
import sys
from common import *
from research_pass22 import decoder,fixtures


def main():
    setup()
    PBC3.compress=classmethod(lambda *a,**k:(_ for _ in ()).throw(AssertionError('image encoder called')))
    from learned_filler import LearnedFiller
    LearnedFiller.load=classmethod(lambda *a,**k:(_ for _ in ()).throw(AssertionError('model loaded')))
    variants={'sparse_only':lambda d:decoder.decode(d,reuse=False),'reuse_only':lambda d:decoder.decode(d,sparse=False),'optimized':decoder.decode}
    checked=[];fixture_rows=[]
    rows=json.loads((ROOT/'research_pass21/files.json').read_text())['rows']
    for row in rows:
        data,record=saved(row['image'])
        expected=reference.decode(data)
        assert sha(expected.tobytes())==row['pixels_sha256']
        for label,fn in variants.items():
            output=fn(data)
            assert output.dtype==np.uint8 and output.shape==expected.shape and np.array_equal(output,expected)
            assert np.array_equal(output,fn(data))
        error=np.asarray(load(row['image']),dtype=np.float32)-output.astype(np.float32)
        mse=float(np.mean(error**2,dtype=np.float64));assert mse==row['rgb_mse']
        checked.append(dict(image=row['image'],shape=list(output.shape),stream_sha256=sha(data),pixels_sha256=sha(output.tobytes()),rgb_mse=mse,exact_all_three_paths=True,repeated_determinism=True))
        print('exact',row['image'],flush=True)
    rng=np.random.default_rng(2003)
    for w,h in ((1,1),(8,8),(13,9),(35,37)):
        for pattern in ('zero','single_dc','mixed','extremes'):
            q=np.zeros(((h+7)//8,(w+7)//8,3,8,8),dtype=np.int16)
            if pattern=='single_dc':q[-1,-1,2,0,0]=-1
            elif pattern=='mixed':
                q[0,0,0]=rng.integers(-16,17,(8,8),dtype=np.int16)
                q[-1,-1,1,7,7]=3
            elif pattern=='extremes':
                q[:]=rng.integers(-32768,32768,q.shape,dtype=np.int16)
                q[0,0,0,0,0]=-32768;q[-1,-1,2,7,7]=32767
            for step in (32,128):
                data=fixtures.container(w,h,q,step)
                expected=reference.decode(data)
                for label,fn in variants.items():
                    output=fn(data)
                    assert output.dtype==np.uint8 and output.shape==(h,w,3) and np.array_equal(output,expected),(w,h,pattern,step,label,int(np.count_nonzero(output!=expected)))
                    assert np.array_equal(output,fn(data))
                fixture_rows.append(dict(size=[w,h],pattern=pattern,step=step,stream_sha256=sha(data),pixels_sha256=sha(expected.tobytes())))
    fixture=fixtures.container(8,8,np.zeros((1,1,3,8,8),dtype=np.int16))
    params,base,payload=reference.split(fixture)
    raw_cases=[zero_runs.HEADER.pack(b'BAD!',192,0),zero_runs.HEADER.pack(b'R21V',193,0),zero_runs.HEADER.pack(b'R21V',192,193)+b'\0\1'*193,zero_runs.HEADER.pack(b'R21V',zero_runs.MAX_COEFFICIENTS+1,0)]
    for body in (b'\0\0',b'\x80\0\1',b'\0\x81\0',b'\0\x80',b'\x80'*5+b'\1',b'\0\x80\x80\4',b'\xc0\1\1',b'\0\1\0'):
        raw_cases.append(zero_runs.HEADER.pack(b'R21V',192,1)+body)
    payloads=[lzma.compress(raw,format=lzma.FORMAT_XZ,preset=0) for raw in raw_cases]
    empty=zero_runs.HEADER.pack(b'R21V',192,0)
    payloads.extend((payload[:-1],payload+b'junk',payload+payload,lzma.compress(empty+b'\0'*1000000,format=lzma.FORMAT_XZ,preset=0)))
    invalid=[]
    for bad_payload in payloads:invalid.append(reference.HEADER.pack(b'P21S',8,8,len(base),len(bad_payload),128,8,1)+base+bad_payload)
    invalid.extend((fixture[:-1],fixture+b'junk',fixture[:10]))
    fields=list(reference.HEADER.unpack_from(fixture))
    for index,value in ((0,b'BAD!'),(1,0),(1,65536),(1,4096),(5,float('nan')),(5,0),(5,float('inf')),(6,4),(7,2),(3,5)):
        changed=fields.copy();changed[index]=value
        if index==1 and value==4096:changed[2]=4097
        invalid.append(reference.HEADER.pack(*changed)+fixture[reference.HEADER.size:])
    invalid.append(fixtures.container(13,9,np.zeros((2,2,3,8,8),dtype=np.int16),base_size=(12,9)))
    rejected=0
    for data in invalid:
        for fn in (reference.decode,*variants.values()):
            try:fn(data)
            except (ValueError,EOFError,lzma.LZMAError):pass
            else:raise AssertionError('malformed input accepted')
        rejected+=1
    class Oversized:
        def __len__(self):return reference.MAX_FILE_BYTES+1
    for fn in (reference.decode,*variants.values()):
        try:fn(Oversized())
        except ValueError:pass
        else:raise AssertionError('file resource limit bypassed')
    rejected+=1
    code="import sys,hashlib;sys.path.insert(0,sys.argv[1]);from learned_filler import LearnedFiller;LearnedFiller.load=classmethod(lambda *a,**k:(_ for _ in ()).throw(AssertionError('model loaded')));from PBC3 import PBC3;PBC3.compress=classmethod(lambda *a,**k:(_ for _ in ()).throw(AssertionError('encoder called')));from research_pass22.decoder import decode;from pathlib import Path;print(hashlib.sha256(decode(Path(sys.argv[2]).read_bytes()).tobytes()).hexdigest())"
    for name in ('googlewiki.png','flower.jpg'):
        row=next(r for r in rows if r['image']==name)
        result=subprocess.check_output([sys.executable,'-c',code,str(ROOT),str(ROOT/'research_pass21'/row['file'])],cwd=ROOT,text=True).strip()
        assert result==row['pixels_sha256']
    write('validation.json',dict(provenance=meta(),files=checked,fixtures=fixture_rows,malformed_and_resource_cases=rejected,rejected_in_all_four_paths=True,decoder_only_model_and_encoder_forbidden=True,fresh_processes=2,notes='No image encoder used: synthetic fixture base headers have zero patches. All-zero, one occupied component, mixed sparse/dense and signed int16 extremes at boundaries, odd sizes and1x1 included. All three candidate paths equal frozen float32 RGB and repeat deterministically.'))
    print('passed',len(checked),'files',len(fixture_rows),'fixtures',rejected,'malformed/resource cases',flush=True)


if __name__=='__main__':main()
