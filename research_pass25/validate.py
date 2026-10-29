import lzma
import subprocess
import sys
from common import *
from research_pass25 import base,hybrid,jpeg_reference
from research_pass21 import zero_runs
from pbc3_types import BitWriter
from learned_filler import LearnedFiller


def main():
    setup();a=json.loads((HERE/'feasibility.json').read_text());screen=json.loads((HERE/'screen.json').read_text());checked=[];fixtures=[]
    assert a['passed'] and not screen['gate']['passed']
    for row in a['rows']+screen['rows']:
        data=(HERE/row['file']).read_bytes();assert sha(data)==row['stream_sha256'] and len(data)==row['bytes']
        variant=row['variant']
        decoder=(lambda data:PBC3.decompress(data).image) if variant=='ordinary' else jpeg_reference.decode if variant=='jpeg' else base.decode if variant=='transform' else hybrid.decode
        output=decoder(data);assert sha(output.tobytes())==row['pixels_sha256'] and output.tobytes()==decoder(data).tobytes()
        assert output.size==load(row['image']).size and mse(np.asarray(load(row['image'])),np.asarray(output))==row['rgb_mse']
        if variant in ('hybrid','base_only'):
            initial,raw,count=hybrid.split(data);assert count==row['spatial_patches'] and len(initial)==row['base_bytes'] and len(raw)==row['spatial_raw_bytes']
            canvas,info=hybrid.decode_canvas(data);assert base.render(canvas,info).tobytes()==output.tobytes()
            if variant=='hybrid':assert row['policy_class']=='LearnedFiller'
        checked.append(dict(image=row['image'],variant=variant,preset=row.get('preset'),step=row.get('step'),stream_sha256=row['stream_sha256'],pixels_sha256=row['pixels_sha256'],rgb_mse=row['rgb_mse']))
    rng=np.random.default_rng(2003)
    for w,h in ((1,1),(13,9),(35,37)):
        image=Image.fromarray(rng.integers(0,256,(h,w,3),dtype=np.uint8))
        prep=PBC3.prepare(image,PBC3Config.quality())
        for step in (64,128,256):
            data=base.encode(prep,step);assert data==base.encode(prep,step)
            canvas,info=base.decode_canvas(data);assert canvas.dtype==np.int16 and canvas.shape==(h,w,3)
            framed=hybrid.frame(data,[])[0];assert hybrid.decode(framed).tobytes()==base.decode(data).tobytes()
            fixtures.append(dict(size=[w,h],step=step,base_sha256=sha(data),pixels_sha256=sha(base.decode(data).tobytes())))
    q=np.zeros((1,1,3,8,8),dtype=np.int16);q[0,0,0,0,0]=-32768;q[0,0,1,0,0]=32767
    payload=lzma.compress(zero_runs.pack(q),format=lzma.FORMAT_XZ,preset=0)
    endpoints=base.HEADER.pack(b'P25I',8,8,8,8,0,128,128,128,1,len(payload))+payload
    canvas,_=base.decode_canvas(endpoints);assert np.all(canvas[:,:,0]==-3968) and np.all(canvas[:,:,1]==4224)
    fixtures.append(dict(case='signed_coefficient_endpoints',pixels_sha256=sha(base.decode(endpoints).tobytes())))
    q[:]=0;q[0,0,0,0,0]=32767;payload=lzma.compress(zero_runs.pack(q),format=lzma.FORMAT_XZ,preset=0)
    maximum=base.HEADER.pack(b'P25I',8,8,8,8,0,0,0,0,8,len(payload))+payload
    canvas,_=base.decode_canvas(maximum);assert np.all(canvas[:,:,0]==32767)
    patch=dict(channel=0,x=0,y=0,w=8,h=8,mask=[0,1],neg=0,pos=1,max_bitcount=1,cell_size=8,indices=np.ones((1,1),dtype=np.uint16),bitcount=1)
    wrapped=hybrid.frame(maximum,[patch])[0];result,_=hybrid.decode_canvas(wrapped)
    assert np.all(result[:,:,0]==-32768)
    fixtures.append(dict(case='additive_int16_wrap',pixels_sha256=sha(hybrid.decode(wrapped).tobytes())))
    valid=(HERE/next(r['file'] for r in screen['rows'] if r['variant']=='hybrid')).read_bytes()
    initial,raw,count=hybrid.split(valid)
    malformed_base=[initial[:-1],initial+b'junk',initial[:10],base.HEADER.pack(b'P25I',8,8,8,8,0,0,0,0,9,len(payload))+payload]
    fields=list(base.HEADER.unpack_from(initial))
    for index,value in ((0,b'BAD!'),(1,0),(1,65535),(3,65535),(5,3),(9,0),(9,float('nan')),(10,0)):
        changed=fields.copy();changed[index]=value
        if index==3:changed[4]=65535
        malformed_base.append(base.HEADER.pack(*changed)+initial[base.HEADER.size:])
    n=((fields[2]+7)//8)*((fields[1]+7)//8)*192
    for bad in (zero_runs.HEADER.pack(b'R21V',n+1,0),zero_runs.HEADER.pack(b'R21V',n,1)+b'\0\0',zero_runs.HEADER.pack(b'R21V',n,1)+b'\0\x80',zero_runs.HEADER.pack(b'R21V',n,0)+b'\0'*1000000):
        payload=lzma.compress(bad,format=lzma.FORMAT_XZ,preset=0);changed=fields.copy();changed[-1]=len(payload);malformed_base.append(base.HEADER.pack(*changed)+payload)
    payload=initial[base.HEADER.size:]
    for bad_payload in (payload[:-1],payload+b'junk',payload+payload):
        changed=fields.copy();changed[-1]=len(bad_payload);malformed_base.append(base.HEADER.pack(*changed)+bad_payload)
    rejected=0
    for data in malformed_base:
        try:base.decode(data)
        except (ValueError,EOFError,lzma.LZMAError):rejected+=1
        else:raise AssertionError('bad initializer accepted')
    invalid=[valid[:-1],valid+b'junk',valid[:10]];fields=list(hybrid.HEADER.unpack_from(valid))
    for index,value in ((0,b'BAD!'),(1,0),(1,len(valid)),(2,hybrid.MAX_RAW_BYTES+1),(3,1),(4,hybrid.MAX_PATCHES+1)):
        changed=fields.copy();changed[index]=value;invalid.append(hybrid.HEADER.pack(*changed)+valid[hybrid.HEADER.size:])
    tiny=base.encode(PBC3.prepare(Image.new('RGB',(8,8)),PBC3Config.quality()),128)
    for channel,x,y,w,h,cell in ((3,0,0,1,1,1),(0,8,0,1,1,1),(0,0,0,9,1,1),(0,0,0,1,1,0)):
        patch=dict(channel=channel,x=x,y=y,w=w,h=h,mask=[0,1],neg=0,pos=1,max_bitcount=1,cell_size=cell,indices=np.zeros((1,1),dtype=np.uint16),bitcount=1)
        invalid.append(hybrid.frame(tiny,[patch])[0])
    invalid.append(hybrid.HEADER.pack(b'P25B',len(tiny),1,0,0)+tiny+b'\0')
    invalid.append(hybrid.HEADER.pack(b'P25B',len(tiny),0,0,0)+tiny+b'\0')
    for data in invalid:
        try:hybrid.decode(data)
        except (ValueError,EOFError,lzma.LZMAError):rejected+=1
        else:raise AssertionError('bad hybrid accepted')
    class Oversized:
        def __len__(self):return base.MAX_FILE_BYTES+1
    for decoder in (base.decode,hybrid.decode):
        try:decoder(Oversized())
        except ValueError:rejected+=1
        else:raise AssertionError('file cap ignored')
    for data in (initial,valid):
        try:PBC3.decompress(data)
        except ValueError:rejected+=1
        else:raise AssertionError('ordinary decoder accepted research magic')
    code="import sys,hashlib;from pathlib import Path;sys.path.insert(0,sys.argv[1]);from learned_filler import LearnedFiller;LearnedFiller.load=classmethod(lambda *a,**k:(_ for _ in ()).throw(AssertionError('policy loaded')));from PBC3 import PBC3;PBC3.compress=classmethod(lambda *a,**k:(_ for _ in ()).throw(AssertionError('ordinary encoder called')));from research_pass25.hybrid import decode;print(hashlib.sha256(decode(Path(sys.argv[2]).read_bytes()).tobytes()).hexdigest())"
    for name in ('googlewiki.png','cat.jpg'):
        row=next(r for r in screen['rows'] if r['image']==name and r['preset']=='high_quality' and r['variant']=='hybrid')
        digest=subprocess.check_output([sys.executable,'-c',code,str(ROOT),str(HERE/row['file'])],cwd=ROOT,text=True).strip();assert digest==row['pixels_sha256']
    policy=LearnedFiller.load(str(ROOT/'patch_policy.npz'));assert type(policy) is LearnedFiller and not hasattr(policy,'begin_encode')
    write('validation.json',dict(provenance=meta(),files=checked,base_determinism_fixtures=fixtures,malformed_resource_or_magic_rejections=rejected,fresh_process_model_and_ordinary_encoder_forbidden=2,loaded_policy_class=type(policy).__name__,legacy_init_metadata_dependency=False,notes='No new full-image encodes. All saved initializer/JPEG/ordinary/hybrid pixels, sizes, MSE and deterministic repeated decodes checked; base-only matches A. Existing legacy model needs no init-patch history; no fake patches. Bounded sparse entropy and hybrid raw entropy, patch geometry/cell/channel, dimensions, caps, truncation/extra data and foreign magic reject. Odd/1x1 fixtures remain signed int16.'))
    print('validated',len(checked),'files',len(fixtures),'fixtures',rejected,'rejections and2 file-only decoders',flush=True)


if __name__=='__main__':main()
