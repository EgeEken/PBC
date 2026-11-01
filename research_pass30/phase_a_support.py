import hashlib, math
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps

ROOT=Path(__file__).resolve().parents[1]
HERE=Path(__file__).resolve().parent
OUT=Path(__file__).resolve().parent/'outputs'
VIS=Path(__file__).resolve().parent/'visuals'
NAMES=('googlewiki.png','mario64.png','flower.jpg','cat.jpg')
CELL=16; GRID=4; DC_STEP=8
PATTERN_STEPS=(8,16)
COEFF_NUM=np.array([-4,-2,-1,0,1,2,4],np.int16)

def sha(b): return hashlib.sha256(b).hexdigest()
def rnd(x): return np.sign(x)*np.floor(np.abs(x)+.5)

def cells(image):
    h,w=image.shape[:2]; rows,cols=math.ceil(h/CELL),math.ceil(w/CELL)
    out=np.zeros((rows*cols,CELL,CELL,3),np.int16); valid=np.zeros((rows*cols,CELL,CELL),bool)
    for y in range(rows):
        for x in range(cols):
            i=y*cols+x; ch,cw=min(CELL,h-y*CELL),min(CELL,w-x*CELL)
            out[i,:ch,:cw]=image[y*CELL:y*CELL+ch,x*CELL:x*CELL+cw]; valid[i,:ch,:cw]=1
    return out,valid,rows,cols

def bilinear(p):
    p=np.asarray(p,np.int32); c=np.linspace(0,GRID-1,CELL); lo=np.floor(c).astype(int); hi=np.minimum(lo+1,GRID-1); f=c-lo
    a=p[lo[:,None],lo[None,:]]; b=p[lo[:,None],hi[None,:]]; d=p[hi[:,None],hi[None,:]]; e=p[hi[:,None],lo[None,:]]
    wx=f[None,:,None]; wy=f[:,None,None]
    return np.clip(rnd((a*(1-wx)+b*wx)*(1-wy)+(e*(1-wx)+d*wx)*wy),-32768,32767).astype(np.int16)

def low_projection(c): return c.reshape(len(c),4,4,4,4,3).mean((2,4)).astype(np.float32)

def structural_groups(low):
    centered=low-low.mean((1,2),keepdims=True); lum=centered@np.array([.299,.587,.114],np.float32)
    norm=np.sqrt((lum*lum).mean((1,2))); z=lum/np.maximum(norm[:,None,None],1)
    gx=np.abs(np.diff(lum,axis=2)).mean((1,2))/np.maximum(norm,1); gy=np.abs(np.diff(lum,axis=1)).mean((1,2))/np.maximum(norm,1)
    f=np.column_stack((z.reshape(len(z),-1),gx,gy)); s=f.std(0); s[s<1e-8]=1; z=(f-f.mean(0))/s
    centers=[int(np.argmax((z*z).sum(1)))]; d=((z-z[centers[0]])**2).sum(1)
    for _ in range(3): j=int(np.argmax(d)); centers.append(j); d=np.minimum(d,((z-z[j])**2).sum(1))
    c=z[centers].copy()
    for _ in range(32):
        lab=((z[:,None]-c[None])**2).sum(2).argmin(1); nxt=c.copy()
        for g in range(4):
            if np.any(lab==g): nxt[g]=z[lab==g].mean(0)
        if np.array_equal(nxt,c): break
        c=nxt
    lab=((z[:,None]-c[None])**2).sum(2).argmin(1); order=sorted(range(4),key=lambda g:tuple(np.round(c[g],12)))
    remap=np.empty(4,np.uint8); remap[order]=np.arange(4,dtype=np.uint8); return remap[lab]

def qdc(low): return np.clip(rnd(low.mean((1,2))/DC_STEP)*DC_STEP,-256,256).astype(np.int16)

def reconstruct(base,dc,patterns=None,coeff=None,assign=None):
    h,w=base.shape[:2]; rows,cols=math.ceil(h/CELL),math.ceil(w/CELL); n=len(dc)
    delta=np.broadcast_to(dc[:,None,None,:],(n,CELL,CELL,3)).astype(np.int32).copy()
    if patterns is not None:
        for g in range(len(patterns)):
            ids=np.flatnonzero(((assign==g) if len(patterns)>1 else np.ones(n,bool))&(coeff!=3))
            if len(ids): delta[ids]+=rnd(COEFF_NUM[coeff[ids],None,None,None]*bilinear(patterns[g])[None].astype(np.float32)/2).astype(np.int32)
    patch=delta.reshape(rows,cols,CELL,CELL,3).transpose(0,2,1,3,4).reshape(rows*CELL,cols*CELL,3)[:h,:w]
    return np.clip(base.astype(np.int32)+patch,0,255).astype(np.uint8)

def fit(low,dc,assign,groups,q):
    target=low-dc[:,None,None,:]; p=np.zeros((groups,4,4,3),np.int16); coef=np.full(len(low),3,np.uint8)
    for g in range(groups):
        ids=np.flatnonzero(assign==g) if groups>1 else np.arange(len(low))
        if not len(ids): continue
        energy=(target[ids]**2).sum((1,2,3)); pat=np.clip(rnd(target[ids[int(np.argmax(energy))]]/q)*q,-256,256).astype(np.int16)
        for _ in range(4):
            up=bilinear(pat).astype(np.float32).reshape(4,4,4,4,3).mean((1,3))
            err=np.stack([((target[ids]-up[None]*(x/2))**2).sum((1,2,3)) for x in COEFF_NUM]); ci=err.argmin(0).astype(np.uint8); coef[ids]=ci
            a=COEFF_NUM[ci].astype(np.float64)/2; den=(a*a).sum()
            if den: pat=np.clip(rnd(np.einsum('n,nhwc->hwc',a,target[ids])/den/q)*q,-256,256).astype(np.int16)
        p[g]=pat
    return p,coef

def metric(src,rec):
    e=(src.astype(np.float32)-rec.astype(np.float32))**2; lum=src.mean(2).astype(np.float32); gy,gx=np.gradient(lum); grad=np.hypot(gx,gy)
    return {'mse':float(e.mean()),'edge_top10_mse':float(e[grad>=np.quantile(grad,.9)].mean()),'smooth_bottom50_mse':float(e[grad<=np.quantile(grad,.5)].mean())}

def cell_sse(r,corr,valid): return ((r.astype(np.float32)-corr.astype(np.float32))**2*valid[...,None]).sum((1,2,3))

def load_case(name):
    src=np.asarray(ImageOps.exif_transpose(Image.open(ROOT/'hpt_data'/name)).convert('RGB'),np.uint8)
    stream=(ROOT/'research_pass29/outputs/bases'/f'{name}.pbc3').read_bytes()
    import sys
    sys.path.insert(0,str(ROOT))
    from PBC3 import PBC3
    base=np.asarray(PBC3.decompress(stream).image.convert('RGB'),np.uint8)
    arr=np.load(ROOT/'research_pass29/outputs/arrays'/f'{name}.npz')
    return {'src':src,'base':base,'base_bytes':stream,'dist_assign':arr['structure_assignments'],'source_hash':sha((ROOT/'hpt_data'/name).read_bytes()),'base_hash':sha(stream)}
