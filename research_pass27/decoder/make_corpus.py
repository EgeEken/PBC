import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import *
sys.path.insert(0,str(PUBLISHED))
from PBC3 import PBC3
from pbc3_types import PBC3Config,BitWriter
import subprocess
import lzma
out=PASS/'artifacts/corpus';out.mkdir(exist_ok=True)
rows=[]
def record(name,data,origin):
    path=out/(name+'.pbc');path.write_bytes(data)
    decoded=PBC3.decompress(data).image
    rows.append(dict(name=name,file=str(path),origin=origin,stream_sha256=sha(data),bytes=len(data),mode=decoded.mode,size=decoded.size,pixel_sha256=sha(decoded.tobytes()),partial={str(n):sha(PBC3.decompress(data,max_patches=n).image.tobytes()) for n in (0,1)}))
for file in sorted((ROOT/'research_pass26/artifacts/packaging_tests').glob('python_*.pbc')):
    record(file.stem,file.read_bytes(),'published31_pass26')
for name in ('stored','warmup'): record(name,(ROOT/'research_pass26/artifacts/packaging_tests'/(name+'.pbc')).read_bytes(),'published31_pass26')
for file in sorted((ROOT/'research_pass26/artifacts/architecture').glob('*.pbc')):
    record(file.stem,file.read_bytes(),'published31_pass26')
for preset in ('quality','high_quality'):
    record('eiffel.png.'+preset,PBC3.compress(load('eiffel.png'),config=getattr(PBC3Config,preset)()).data,'published31_fresh')
rng=np.random.default_rng(2701)
for i,(w,h,channels) in enumerate(((1,1,3),(1,17,4),(19,1,3),(37,31,4),(65,67,3))):
    image=Image.fromarray(rng.integers(0,256,(h,w,channels),dtype=np.uint8))
    for color in ('RGB','YCbCr'):
        cfg=PBC3Config.quality(patch_count=5,color_space=color,positive_bias=bool(i%2),use_lzma=bool(i%2))
        record(f'boundary_{i}_{color}',PBC3.compress(image,config=cfg).data,'published31_fresh')
for sign in (-1,1):
    bw=BitWriter();PBC3._write_header(bw,1,1,1,1,False,0,3,2,True,False,130,[0,0,0])
    for _ in range(130):
        patch=dict(channel=0,x=0,y=0,w=1,h=1,mask=[0,1],neg=255 if sign<0 else 0,pos=255 if sign>0 else 0,max_bitcount=8,cell_size=1,indices=np.array([[0 if sign<0 else 254]],np.uint16),bitcount=8)
        PBC3._write_patch(bw,patch,2)
    record('signed_wrap_'+str(sign),b'PBC3\0\0'+bw.finish(),'published31_signed_wrap')
old=ROOT/'research_pass1/release'
code="import sys;sys.path.insert(0,sys.argv[1]);from PBC3 import PBC3;from pbc3_types import PBC3Config;from PIL import Image;from pathlib import Path;assert Path(sys.modules['PBC3'].__file__).parent==Path(sys.argv[1]);r=PBC3.compress(Image.open(sys.argv[2]),config=PBC3Config.quality(patch_count=5,learned_filler_enabled=False,color_space='RGB'));Path(sys.argv[3]).write_bytes(r.data)"
image=load('eiffel.png').resize((65,63));image.save(out/'historical.png')
subprocess.run([sys.executable,'-c',code,str(old),str(out/'historical.png'),str(out/'historical.pbc')],check=True)
record('historical30',(out/'historical.pbc').read_bytes(),'actual_frozen30')
invalid=[b'',b'PBC3',b'PBC3\1\0',b'PBC3\0\7',b'PBC3\0\2x',rows[0] and (out/(rows[0]['name']+'.pbc')).read_bytes()[:17]]
valid=(out/(rows[0]['name']+'.pbc')).read_bytes();invalid+=[valid+b'junk',valid[:-1]]
for args in ((0,1,3,0),(65535,65535,3,0),(1,1,2,0),(1,1,3,10001)):
    w,h,ch,count=args;bw=BitWriter();PBC3._write_header(bw,w,h,1,1,False,0,ch,2,True,False,count,[0]*ch)
    invalid.append(b'PBC3\0\0'+bw.finish())
bw=BitWriter();PBC3._write_header(bw,1,1,65535,65535,True,0,3,2,True,False,0,[0]*3);invalid.append(b'PBC3\0\0'+bw.finish())
for i in range(30): invalid.append(valid[:max(1,len(valid)*i//30)])
invalid.append(b'PBC3\0\2'+lzma.compress(bytes(128*1024*1024+1),format=lzma.FORMAT_RAW,filters=[{'id':lzma.FILTER_LZMA2,'preset':0}]))
for i,data in enumerate(invalid): (out/f'invalid_{i}.pbc').write_bytes(data)
save('decoder/corpus.json',dict(valid=rows,invalid=[str(out/f'invalid_{i}.pbc') for i in range(len(invalid))]))
print('corpus',len(rows),'valid',len(invalid),'invalid')
