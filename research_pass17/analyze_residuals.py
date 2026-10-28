import dataclasses
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

os.environ.update({k:'1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS')})
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT))
import numpy as np
import numba
import PIL
from PIL import Image, ImageOps
import psutil
import scipy
from scipy.fft import dctn
from scipy.ndimage import sobel
from PBC3 import PBC3
from pbc3_types import PBC3Config

IMAGES=('cat.jpg','flower.jpg','monet.jpg','googlewiki.png','mario64.png','eiffel.png')
PRESETS=('quality','high_quality')
RUNTIME=('PBC3.py','pbc3_ops.py','pbc3_features.py','learned_filler.py','pbc3_heads.py','pbc3_types.py','pbc3_stream.py','pbc3_trace.py','patch_policy.npz')
COMMIT='1c9c4016fdb213d7d16663542d268c7c3df4ed9b'
ARTIFACTS=HERE/'artifacts'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def setup():
    psutil.Process().cpu_affinity([12]); numba.set_num_threads(1)
    ARTIFACTS.mkdir(exist_ok=True)


def load(name):
    with Image.open(ROOT/'hpt_data'/name) as image:
        return ImageOps.exif_transpose(image).convert('RGB')


def measure(image,config):
    start=time.perf_counter(); encoded=PBC3.compress(image,config=config); enc=time.perf_counter()-start
    start=time.perf_counter(); decoded=PBC3.decompress(encoded.data).image.convert('RGB'); dec=time.perf_counter()-start
    original=np.asarray(image,dtype=np.uint8); output=np.asarray(decoded,dtype=np.uint8)
    error=original.astype(np.float32)-output.astype(np.float32)
    row={'bytes':len(encoded.data),'bpp':8*len(encoded.data)/(image.width*image.height),'rgb_mse':float(np.mean(error**2,dtype=np.float64)),'encode_seconds':enc,'decode_seconds':dec,'stream_sha256':sha(encoded.data),'pixels_sha256':sha(output.tobytes()),'internal_mse':encoded.mse,'config':dataclasses.asdict(config)}
    return encoded.data,output,row


def blocks(array,size=8):
    h,w,c=array.shape
    padded=np.pad(array,((0,(-h)%size),(0,(-w)%size),(0,0)),mode='edge')
    return padded.reshape(padded.shape[0]//size,size,padded.shape[1]//size,size,c).transpose(0,2,4,1,3)


def residual_analysis(original,output,prepared):
    residual=original.astype(np.float32)-output.astype(np.float32)
    coeff=dctn(blocks(residual),axes=(-2,-1),norm='ortho',workers=1)
    energy=np.sum(coeff.astype(np.float64)**2,axis=(0,1,2))
    u,v=np.indices((8,8)); bands=u+v
    luminance=original.astype(np.float32).mean(axis=2)
    gradient=sobel(luminance,axis=0,mode='reflect')**2+sobel(luminance,axis=1,mode='reflect')**2
    pixel_sse=np.sum(residual**2,axis=2,dtype=np.float64)
    edge=gradient>=np.percentile(gradient,90)
    smooth=gradient<=np.percentile(gradient,50)
    tile_sse=np.pad(pixel_sse,((0,(-pixel_sse.shape[0])%32),(0,(-pixel_sse.shape[1])%32)))
    tiles=tile_sse.reshape(tile_sse.shape[0]//32,32,tile_sse.shape[1]//32,32).sum(axis=(1,3)).ravel()
    ideal=PBC3._canvas_to_image(prepared['target'],prepared['color_space'],False)
    if ideal.size!=(original.shape[1],original.shape[0]):
        ideal=ideal.resize((original.shape[1],original.shape[0]),PBC3.RESAMPLE_FILTER,reducing_gap=PBC3.RESAMPLE_REDUCING_GAP)
    ideal=np.asarray(ideal,dtype=np.float32)
    down=original.astype(np.float32)-ideal
    represented=ideal-output.astype(np.float32)
    total=float(pixel_sse.sum())
    result={'rgb_sse':total,'rgb_mse':total/original.size,'dct_low_uplusv_le2_fraction':float(energy[bands<=2].sum()/energy.sum()),'dct_high_uplusv_ge4_fraction':float(energy[bands>=4].sum()/energy.sum()),'top10pct_tiles_error_fraction':float(np.sort(tiles)[-max(1,int(np.ceil(tiles.size*.1))):].sum()/tiles.sum()),'edge_pixel_fraction':float(edge.mean()),'edge_sse_fraction':float(pixel_sse[edge].sum()/total),'smooth_pixel_fraction':float(smooth.mean()),'smooth_rgb_mse':float(pixel_sse[smooth].mean()/3),'working_size':[prepared['w'],prepared['h']],'source_roundtrip_mse':float(np.mean(down**2,dtype=np.float64)),'working_representable_residual_mse':float(np.mean(represented**2,dtype=np.float64)),'decomposition_cross_mse':float(2*np.mean(down*represented,dtype=np.float64))}
    assert abs(result['source_roundtrip_mse']+result['working_representable_residual_mse']+result['decomposition_cross_mse']-result['rgb_mse'])<1e-6
    return result


def provenance():
    hashes={name:sha((ROOT/name).read_bytes()) for name in RUNTIME}
    for name in RUNTIME:
        assert (ROOT/name).read_bytes()==subprocess.check_output(['git','show',COMMIT+':'+name],cwd=ROOT),name
    return {'control_commit':COMMIT,'control_id':'pbc31_feature_histogram','source_sha256':hashes,'image_sha256':{n:sha((ROOT/'hpt_data'/n).read_bytes()) for n in IMAGES},'python':sys.version,'numpy':np.__version__,'numba':numba.__version__,'pillow':PIL.__version__,'scipy':scipy.__version__,'platform':platform.platform(),'cpu':'i9-12900H logical P-core 12','threads':1,'seed':2003,'internal_mse':True,'timing':'One warmed deterministic call per configuration; feasibility timing only. Full compress/decompress calls; loading, hashing, analysis and external RGB scoring excluded.'}


def main():
    setup(); meta=provenance()
    warm=Image.new('RGB',(40,32),(100,140,170))
    for preset in PRESETS: PBC3.compress(warm,config=getattr(PBC3Config,preset)())
    rows=[]
    for name in IMAGES:
        image=load(name); original=np.asarray(image)
        for preset in PRESETS:
            config=getattr(PBC3Config,preset)()
            data,output,row=measure(image,config)
            path=ARTIFACTS/(name+'.'+preset+'.pbc'); path.write_bytes(data)
            row.update({'image':name,'preset':preset,'file':str(path.relative_to(HERE)),'analysis':residual_analysis(original,output,PBC3.prepare(image,config))})
            rows.append(row)
            print(name,preset,row['bytes'],round(row['rgb_mse'],3),round(row['analysis']['dct_high_uplusv_ge4_fraction'],3),flush=True)
    (HERE/'baseline.json').write_text(json.dumps({'provenance':meta,'rows':rows},indent=2)+'\n')

if __name__=='__main__': main()
