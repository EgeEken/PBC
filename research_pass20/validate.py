import json
import lzma
import struct
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research_pass20.common import *
from research_pass20 import multiscale as codec


def main():
    setup()
    screen=json.loads((HERE/'screen.json').read_text())
    costs=json.loads((HERE/'costs.json').read_text())
    details=json.loads((HERE/'cost_details.json').read_text())
    files=[]
    for row in screen['rows']:
        data=(HERE/row['file']).read_bytes()
        assert len(data)==row['bytes'] and sha(data)==row['stream_sha256']
        output,parts,coarse,fine,pixels=codec.decode(data,details=True)
        assert sha(output.tobytes())==row['pixels_sha256'] and mse(load(row['image']),output)==row['rgb_mse']
        params,base,qc,qf=codec.split(data)
        w,h,cw,ch,sc,sf=params
        assert (w,h)==load(row['image']).size and (sc,sf)==codec.SETTINGS[row['setting']]
        assert len(base)==row['base_bytes'] and sha(base)==row['base_sha256']
        if row['base_budget']==10: assert base==frozen_base(row['image'])
        assert np.count_nonzero(qf[:,:,:,~codec.BAND])==0
        assert np.array_equal(output,np.clip(np.rint(pixels+coarse+fine),0,255).astype(np.uint8))
        assert np.array_equal(coarse,codec.resize_float(h17.correction(qc,sc,ch,cw),w,h))
        first,second=codec.layer_bytes(qc,qf)
        payload=data[codec.HEADER.size+len(base):]
        assert lzma.decompress(payload,format=lzma.FORMAT_XZ)==first+second
        assert parts['coarse_raw_bytes']==len(first) and parts['fine_raw_bytes']==len(second)
        files.append(dict(file=row['file'],stream_sha256=sha(data),pixels_sha256=row['pixels_sha256'],bytes=len(data),rgb_mse=row['rgb_mse']))
    auxiliary=0
    for row in costs['rows']:
        for band in row['band_repacks']:
            data=(ARTIFACTS/(row['image']+'.'+band['kind']+'.p17')).read_bytes()
            assert sha(data)==band['stream_sha256'] and len(data)==band['bytes']
            out=h17.decode(data)
            assert sha(out.tobytes())==band['pixels_sha256'] and mse(load(row['image']),out)==band['rgb_mse']
            auxiliary+=1
    for row in details['normal_full_rows']:
        data=(ARTIFACTS/(row['image']+'.normal0.p17')).read_bytes()
        assert sha(data)==row['stream_sha256'] and len(data)==row['bytes']
        output=h17.decode(data)
        assert sha(output.tobytes())==row['pixels_sha256'] and mse(load(row['image']),output)==row['rgb_mse']
        auxiliary+=1
    rng=np.random.default_rng(2003)
    qc=rng.integers(-32000,32001,(1,1,3,8,8),dtype=np.int16)
    qf=rng.integers(-32000,32001,(2,2,3,8,8),dtype=np.int16);qf[:,:,:,~codec.BAND]=0
    first,second=codec.layer_bytes(qc,qf)
    recovered=codec.read_layers(lzma.compress(first+second,format=lzma.FORMAT_XZ,preset=0),13,9,4,3)
    assert np.array_equal(qc,recovered[0]) and np.array_equal(qf,recovered[1])
    fixtures=[]
    for width,height in ((13,9),(35,37)):
        image=Image.fromarray(rng.integers(0,256,(height,width,3),dtype=np.uint8))
        for setting,budget in (('lean',5),('rich',10)):
            data=codec.encode(image,setting,budget)
            repeated=codec.encode(image,setting,budget)
            output=codec.decode(data)
            assert data==repeated and output.shape==(height,width,3)
            fixtures.append(dict(size=[width,height],setting=setting,budget=budget,stream_sha256=sha(data),pixels_sha256=sha(output.tobytes())))
    failures=0
    valid=(ARTIFACTS/'googlewiki.png.p10.lean.p20').read_bytes()
    fields=list(codec.HEADER.unpack_from(valid))
    invalid=[valid[:-1],valid+b'junk',valid[:20]]
    for index,value in ((0,b'BAD!'),(1,0),(5,fields[5]+1),(7,float('nan')),(8,0),(9,4),(10,2),(11,3),(12,0)):
        changed=fields.copy();changed[index]=value
        invalid.append(codec.HEADER.pack(*changed)+valid[codec.HEADER.size:])
    for bad in invalid:
        try:codec.decode(bad)
        except (ValueError,EOFError,lzma.LZMAError):failures+=1
        else:raise AssertionError('malformed stream accepted')
    params,base,qc,qf=codec.split(valid);first,second=codec.layer_bytes(qc,qf)
    for payload in (lzma.compress(first+second+b'\0\0',format=lzma.FORMAT_XZ,preset=0),valid[codec.HEADER.size+len(base):]+b'junk'):
        changed=fields.copy();changed[4]=len(payload)
        bad=codec.HEADER.pack(*changed)+base+payload
        try:codec.decode(bad)
        except (ValueError,EOFError,lzma.LZMAError):failures+=1
        else:raise AssertionError('coefficient framing accepted')
    for decoder in (PBC3.decompress,h17.decode):
        try:decoder(valid)
        except ValueError:failures+=1
        else:raise AssertionError('foreign decoder accepted new magic')
    code="import sys,hashlib;sys.path.insert(0,sys.argv[1]);from learned_filler import LearnedFiller;LearnedFiller.load=classmethod(lambda *a,**k:(_ for _ in ()).throw(AssertionError('policy loaded')));from research_pass20.multiscale import decode;from pathlib import Path;print(hashlib.sha256(decode(Path(sys.argv[2]).read_bytes()).tobytes()).hexdigest())"
    for name,budget,setting in (('googlewiki.png',10,'lean'),('cat.jpg',5,'rich')):
        path=ARTIFACTS/(name+f'.p{budget}.{setting}.p20')
        result=subprocess.check_output([sys.executable,'-c',code,str(ROOT),str(path)],text=True,cwd=ROOT).strip()
        expected=next(r['pixels_sha256'] for r in screen['rows'] if r['image']==name and r['base_budget']==budget and r['setting']==setting)
        assert result==expected
    provenance=meta();provenance['timing']='Correctness verification only; no timing observations used.'
    write_json('validation.json',dict(provenance=provenance,files=files,auxiliary_p17_files=auxiliary,signed_int16_readback=True,nonaligned_deterministic_fixtures=fixtures,malformed_or_magic_rejections=failures,fresh_process_file_only_policy_forbidden=2,production_unchanged=True))
    print('validated',len(files),'multiscale files,',auxiliary,'P17 repacks,',len(fixtures),'repeat fixtures,',failures,'rejections and two file-only subprocesses',flush=True)


if __name__=='__main__':main()
