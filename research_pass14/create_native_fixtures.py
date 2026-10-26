import hashlib,json,os,subprocess,sys
from pathlib import Path
os.environ.update({k:'1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS','VECLIB_MAXIMUM_THREADS')})
ROOT=Path(__file__).resolve().parents[1]
OUT=Path(__file__).resolve().parent/'native_cases'
OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
from PBC3 import PBC3
from pbc3_ops import ceil_div
from pbc3_types import BitReader,PBC3Config
sha=lambda b:hashlib.sha256(b).hexdigest()

def meta(path,name,origin):
 data=path.read_bytes(); body=PBC3._open_body(data)[1]; br=BitReader(body); h=PBC3._read_header(br)
 patches=[]
 for i in range(h[10]):
  c,x,y,w,hh,cell,values,bits=PBC3._read_patch(br,h[7],h[8])
  patches.append(dict(index=i,channel=int(c),cell=int(cell),grid_width=int(values.shape[1]),grid_height=int(values.shape[0]),bitcount=int(bits)))
 return dict(name=name,file=name+'.pbc',origin=origin,stream_sha256=sha(data),file_bytes=len(data),format_version=data[4],entropy_method=data[5],working_size=[int(h[3]),int(h[4])],original_size=[int(h[1] or h[3]),int(h[2] or h[4])],color_space=h[5],channels=h[6],patch_count=h[10],initialization_patches=h[6],iterative_patches=h[10]-h[6],patches=patches)

def save(name,image,config,origin='frozen-development'):
 p=OUT/(name+'.pbc'); result=PBC3.compress(image,config=config); p.write_bytes(result.data); return meta(p,name,origin)
rows=[]
for image in ('cat.jpg','googlewiki.png','monet.jpg','mario64.png','eiffel.png','seagull.jpg'):
 with Image.open(ROOT/'hpt_data'/image) as im: source=im.convert('RGB')
 for preset in ('quality','high_quality'):
  rows.append(save(image.rsplit('.',1)[0]+'_'+preset,source,getattr(PBC3Config,preset)()))
with Image.open(ROOT/'hpt_data'/'cat.jpg') as im: cat=im.convert('RGB')
rows.append(save('rgb_patches',cat,PBC3Config.quality(color_space='RGB',patch_count=8)))
y,x=np.indices((96,128)); rgba=np.stack(((x*2)%256,(y*3)%256,((x+y)*2)%256, np.where(x<64,0,255)),axis=-1).astype(np.uint8)
rows.append(save('rgba_alpha_edge',Image.fromarray(rgba),PBC3Config.quality(patch_count=8)))
small=np.random.default_rng(314159).integers(0,256,size=(13,17,3),dtype=np.uint8)
rows.append(save('small_native',Image.fromarray(small),PBC3Config.quality(patch_count=6)))
init=np.random.default_rng(926535).integers(0,256,size=(24,32,3),dtype=np.uint8)
rows.append(save('initialization_only_rgb',Image.fromarray(init),PBC3Config.quality(color_space='RGB',patch_count=0,downsample_init_cell_size=1)))
rows.append(save('initialization_only_ycbcr',Image.fromarray(init),PBC3Config.quality(color_space='YCbCr',patch_count=0,downsample_init_cell_size=1)))
rows.append(save('downsampled_rgb_init_only',cat,PBC3Config.quality(color_space='RGB',patch_count=0,downsample_init_cell_size=1)))
y,x=np.indices((96,128))
rgb=np.stack(((x*2)%256,(y*3)%256,((x+y)*2)%256),axis=-1).astype(np.uint8)
rows.append(save('rgb_patch_interpolation',Image.fromarray(rgb),PBC3Config.quality(color_space='RGB',patch_count=8)))
old=ROOT/'research_pass1'/'release'; oldfile=OUT/'old_v30_quality.pbc'
code="""
from PIL import Image
import numpy as np
from PBC3 import PBC3
from pbc3_types import PBC3Config
im=Image.fromarray(np.random.default_rng(7654321).integers(0,256,size=(48,64,3),dtype=np.uint8),'RGB')
r=PBC3.compress(im,config=PBC3Config.quality(patch_count=7))
open(r'"""+str(oldfile)+"""','wb').write(r.data)
"""
subprocess.run([sys.executable,'-c',code],cwd=old,check=True)
rows.append(meta(oldfile,'old_v30_quality','immutable-PBC-3.0-runtime'))
(OUT/'fixtures.json').write_text(json.dumps(rows,indent=2)+'\n',encoding='utf-8')
print(json.dumps([{k:r[k] for k in ('name','origin','file_bytes','color_space','channels','patch_count','iterative_patches','format_version','entropy_method')} for r in rows],indent=2))
