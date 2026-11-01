from __future__ import annotations
import hashlib,json,io,time,statistics,subprocess,sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageOps
ROOT=Path(__file__).resolve().parents[1]; HERE=Path(__file__).resolve().parent; OUT=HERE/'outputs'
def sha(b): return hashlib.sha256(b).hexdigest()
def encode(source,q):
    b=io.BytesIO(); t=time.perf_counter()
    Image.fromarray(source).save(b,format='AVIF',quality=int(q),speed=8,max_threads=1,subsampling='4:2:0',codec='aom',range='full')
    return b.getvalue(),time.perf_counter()-t
def decode(data):
    t=time.perf_counter()
    with Image.open(io.BytesIO(data)) as im: a=np.asarray(im.convert('RGB'),np.uint8)
    return a,time.perf_counter()-t
def mse(a,b): return float(np.square(a.astype(np.int32)-b.astype(np.int32),dtype=np.int64).mean())
def main():
    import PIL
    from PIL import _avif
    codec_info={'pillow':PIL.__version__,'libavif':getattr(_avif,'libavif_version','unknown'),'codec_versions':_avif.codec_versions(),'encoder_aom_available':_avif.encoder_codec_available('aom'),'quality':None,'speed':8,'max_threads':1,'subsampling':'4:2:0','codec':'aom','range':'full','decode':'Pillow/libavif'}
    selection=json.loads((HERE/'dataset_selection.json').read_text())
    summary=json.loads((OUT/'codec_summary.json').read_text())
    by_image={x['image']:x for x in summary['images']}
    rows=[]
    for selected in selection['images']:
        name=selected['filename']; src=np.asarray(ImageOps.exif_transpose(Image.open(ROOT/'hpt_data'/name)).convert('RGB'),np.uint8)
        base_bytes=by_image[name]['base_bytes']
        fixed=next(r for r in by_image[name]['rows'] if r['config']=='S' and r['band']=='F' and r['qstep']==32 and r['lambda']==400.0)
        targets=[('base',int(base_bytes)),('representative_F_q32_l400',int(fixed['total_bytes']))]
        cache={}
        def at(q):
            q=int(q)
            if q not in cache:
                data,et=encode(src,q);rec,dt=decode(data)
                path=OUT/'avif'/name/f'q{q}.avif';path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
                cache[q]={'quality':q,'bytes':len(data),'sha256':sha(data),'mse':mse(src,rec),'decoded_sha256':sha(rec.tobytes()),'encode_seconds':et,'preliminary_decode_seconds':dt,'stream_path':str(path.relative_to(HERE)),'decoded_shape':list(rec.shape)}
            return cache[q]
        at(0);at(100)
        brackets={}
        for target_name,target in targets:
            lo,hi=0,100
            s0,s100=at(lo)['bytes'],at(hi)['bytes']
            if (s0-target)*(s100-target)>0:
                brackets[target_name]={'target_bytes':target,'q0_bytes':s0,'q100_bytes':s100,'bracketed':False};continue
            for _ in range(4):
                mid=(lo+hi)//2
                point=at(mid)
                if point['bytes']<=target: lo=mid
                else: hi=mid
            brackets[target_name]={'target_bytes':target,'q_below':lo,'below':at(lo),'q_above':hi,'above':at(hi),'bracketed':True}
        mf=OUT/'avif'/name/'decode_manifest.json'
        mf.write_text(json.dumps({'source':str(ROOT/'hpt_data'/name),'files':[{'quality':p['quality'],'path':str((HERE/p['stream_path']).resolve())} for p in cache.values()]},indent=2)+'\n')
        proc=subprocess.run([sys.executable,'-B',str(HERE/'worker_avif_decode.py'),str(mf)],cwd=ROOT,check=True,capture_output=True,text=True)
        fresh={r['quality']:r for r in json.loads(proc.stdout)['rows']}
        for point in cache.values():
            check=fresh[point['quality']]
            if check['decoded_sha256']!=point['decoded_sha256'] or abs(check['mse']-point['mse'])>1e-12: raise ValueError('fresh-process AVIF decode mismatch')
            point['fresh_process_verified']=True;point['decode_seconds']=check['decode_seconds'];point['fresh_process_peak_working_set_bytes']=check['peak_working_set_bytes']
        rows.append({'image':name,'width':src.shape[1],'height':src.shape[0],'targets':brackets,'points':list(cache.values())})
        print('AVIF',name,'qualities',sorted(cache),'fresh decoded',flush=True)
    result={'provenance':{'selection_sha256':sha((HERE/'dataset_selection.json').read_bytes()),'codec':codec_info,'max_bisection_rounds':4},'images':rows}
    (OUT/'avif_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print('AVIF summary complete',len(rows))
if __name__=='__main__':main()
