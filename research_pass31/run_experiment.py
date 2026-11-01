from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import psutil
from PIL import Image, ImageOps
from threadpoolctl import threadpool_limits, threadpool_info

import pbc31_codec as codec

ROOT=Path(__file__).resolve().parents[1]
HERE=Path(__file__).resolve().parent
OUT=HERE/'outputs'
NAMES=('googlewiki.png','mario64.png','flower.jpg','cat.jpg')
CONFIGS=('A','B','C','D')
LAMBDAS=(0.0,20.0,100.0)
CLASSES={'googlewiki.png':'text','mario64.png':'synthetic graphics','flower.jpg':'smooth photography','cat.jpg':'detailed photography'}

def sha(data: bytes) -> str: return hashlib.sha256(data).hexdigest()

def load_refs():
    return {x['image']:x for x in json.loads((ROOT/'research_pass29/outputs/results.json').read_text())['images']}

def load_image(name, refs):
    source_path=ROOT/'hpt_data'/name
    stream_path=ROOT/'research_pass29/outputs/bases'/f'{name}.pbc3'
    source=np.asarray(ImageOps.exif_transpose(Image.open(source_path)).convert('RGB'),np.uint8)
    stream=stream_path.read_bytes(); ref=refs[name]['base']
    if sha(stream)!=ref['sha256'] or sha(source_path.read_bytes())!=ref['source_sha256']:
        raise ValueError(f'Pass 29 hash mismatch: {name}')
    sys.path.insert(0,str(ROOT))
    from PBC3 import PBC3
    base=np.asarray(PBC3.decompress(stream).image.convert('RGB'),np.uint8)
    if source.shape!=base.shape: raise ValueError(f'source/base dimensions differ: {name}')
    return source,base,stream,sha(source_path.read_bytes()),sha(stream)

def edge_masks(source):
    lum=source.astype(np.float32).mean(2)
    gy,gx=np.gradient(lum)
    mag=np.hypot(gx,gy)
    high=np.quantile(mag,.9); low=np.quantile(mag,.5)
    return mag>=high,mag<=low

def rgb_metrics(source,recon,edge,smooth):
    diff=source.astype(np.int16)-recon.astype(np.int16)
    per_pixel=np.square(diff.astype(np.int32)).sum(2,dtype=np.int64)
    return {'mse':float(per_pixel.sum()/(source.shape[0]*source.shape[1]*3)),
            'edge_top10_mse':float(per_pixel[edge].mean()/3),
            'smooth_bottom50_mse':float(per_pixel[smooth].mean()/3)}

def timing_decode(data,base,expected_modes,expected_params):
    image,modes,params=codec.decode(data,base)
    if not np.array_equal(modes,expected_modes) or params!=expected_params:
        raise AssertionError('decoder did not recover the encoded decisions exactly')
    samples=[]
    for _ in range(3):
        t=time.perf_counter(); got,_,_=codec.decode(data,base); samples.append(time.perf_counter()-t)
        if not np.array_equal(got,image): raise AssertionError('non-deterministic decode')
    return image,{'samples_s':samples,'median_s':float(np.median(samples))}

def summarize_record(name,source,base,base_stream,edge,smooth,config,lam,encoded,path):
    path.write_bytes(encoded['data'])
    stored=path.read_bytes()
    if stored!=encoded['data']: raise AssertionError('written stream changed')
    image,decode_time=timing_decode(stored,base,encoded['modes'],encoded['params'])
    if not np.array_equal(image,encoded['reconstruction']): raise AssertionError('serialized decode differs from encoder reconstruction')
    metrics=rgb_metrics(source,image,edge,smooth)
    modes=encoded['modes']; counts=np.bincount(modes,minlength=7)
    dir_mask=np.isin(modes,(2,3,5,6))
    direction_selected_gain=int((encoded['skip_sse'][dir_mask]-encoded['chosen_sse'][dir_mask]).sum())
    valid_dir=encoded['best_direction_sse']>=0
    direction_opportunity=int((encoded['skip_sse'][valid_dir]-encoded['best_direction_sse'][valid_dir]).sum())
    n=len(modes); mode_bytes=math.ceil(n*codec.MODE_BITS/8)
    actual_bits=len(stored)*8
    rec={'key':f'{config}|{lam:g}','config':config,'lambda':lam,'enhancement_bytes':len(stored),
         'base_bytes':len(base_stream),'total_bytes':len(stored)+len(base_stream),
         'total_bpp':8*(len(stored)+len(base_stream))/(source.shape[0]*source.shape[1]),
         'enhancement_bpp':8*len(stored)/(source.shape[0]*source.shape[1]),'metrics':metrics,
         'mode_counts':{'skip':int(counts[0]),'dc':int(counts[1]),'vertical_no_offset':int(counts[2]),
             'horizontal_no_offset':int(counts[3]),'local_detail':int(counts[4]),
             'vertical_with_offset':int(counts[5]),'horizontal_with_offset':int(counts[6])},
         'estimated_bits':encoded['estimated_bits'],'actual_enhancement_bits':actual_bits,
         'actual_minus_estimated_bits':actual_bits-encoded['estimated_bits'],
         'estimated_bits_per_cell':encoded['estimated_bits']/n,
         'mode_signal_raw_bytes':mode_bytes,'parameter_raw_bytes':len(encoded['params']),
         'raw_payload_bytes':encoded['raw_payload_bytes'],'header_bytes':encoded['header_bytes'],
         'entropy_payload_bytes':encoded['entropy_payload_bytes'],
         'xz_reduction_pct':100*(1-encoded['entropy_payload_bytes']/max(1,encoded['raw_payload_bytes'])),
         'encode_seconds':encoded['encode_seconds'],'decode':decode_time,
         'directional_cells':int(dir_mask.sum()),'directional_selected_sse_gain':direction_selected_gain,
         'directional_available_cell_sse_gain':direction_opportunity,
         'stream_sha256':sha(stored),'pixels_sha256':sha(image.tobytes()),
         'modes_sha256':sha(modes.tobytes()),'params_sha256':sha(encoded['params']),
         'fresh_process_verified':False}
    percell=OUT/'per_cell'/name/config/f'{lam:g}.npz'; percell.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(percell,modes=modes,skip_sse=encoded['skip_sse'],chosen_sse=encoded['chosen_sse'],
                        best_direction_sse=encoded['best_direction_sse'])
    return rec

def fresh_decode_check(name,base_path,records):
    manifest=OUT/'fresh'/f'{name}.json'; manifest.parent.mkdir(parents=True,exist_ok=True)
    entries=[]
    for r in records:
        entries.append({'key':r['key'],'stream':str((OUT/'streams'/name/f"{r['key'].replace('|','_')}.p31r").resolve())})
    manifest.write_text(json.dumps({'base_npy':str(base_path.resolve()),'streams':entries}))
    run=subprocess.run([sys.executable,str(HERE/'worker_decode.py'),str(manifest)],check=True,capture_output=True,text=True)
    result=json.loads(run.stdout)
    by_key={x['key']:x for x in result['rows']}
    for r in records:
        decoded=by_key[r['key']]
        for field in ('pixels_sha256','modes_sha256','params_sha256'):
            if decoded[field]!=r[field]: raise AssertionError(f'fresh process mismatch {name} {r["key"]} {field}')
        r['fresh_process_verified']=True
    return True

def run():
    OUT.mkdir(parents=True,exist_ok=True)
    limits=threadpool_limits(limits=1)
    refs=load_refs()
    summary={'provenance':{'starting_head':'fa86060175889e8ec40d3dd3b938b7982daa93dd',
        'python':sys.version.split()[0],'platform':platform.platform(),'cpu':platform.processor(),
        'numpy':np.__version__,'pillow':Image.__version__ if hasattr(Image,'__version__') else None,
        'threadpools':threadpool_info(),'thread_limit':1},
        'config':{'cell':16,'dc_step':8,'detail_step':16,'detail_grid':'4x4 RGB; align-corners bilinear fixed-point, ties away from zero',
          'modes':{'0':'skip','1':'dc','2':'vertical prediction','3':'horizontal prediction','4':'local 4x4 detail',
                   '5':'vertical prediction + DC offset','6':'horizontal prediction + DC offset'},
          'rate_estimator_bits_per_cell':{'skip':3,'dc':27,'vertical_no_offset':3,'horizontal_no_offset':3,
              'vertical_with_offset':27,'horizontal_with_offset':27,'detail':387},
          'operating_points':list(LAMBDAS),'xz_preset':6,'configuration_masks':codec.MASKS},'images':{}}
    for name in NAMES:
        print('image',name,flush=True)
        source,base,base_stream,source_hash,base_hash=load_image(name,refs)
        base_path=OUT/'bases'/f'{name}.npy'; base_path.parent.mkdir(parents=True,exist_ok=True); np.save(base_path,base,allow_pickle=False)
        edge,smooth=edge_masks(source); results=[]
        (OUT/'streams'/name).mkdir(parents=True,exist_ok=True)
        for config in CONFIGS:
            for lam in LAMBDAS:
                print(' encode',config,lam,flush=True)
                encoded=codec.encode(source,base,config,lam)
                stream_path=OUT/'streams'/name/f"{config}_{lam:g}.p31r"
                rec=summarize_record(name,source,base,base_stream,edge,smooth,config,lam,encoded,stream_path)
                results.append(rec); print('  bytes',rec['enhancement_bytes'],'mse',round(rec['metrics']['mse'],4),rec['mode_counts'],flush=True)
        fresh_decode_check(name,base_path,results)
        summary['images'][name]={'class':CLASSES[name],'width':source.shape[1],'height':source.shape[0],
            'source_sha256':source_hash,'base_sha256':base_hash,'base_bytes':len(base_stream),
            'base_mse':rgb_metrics(source,base,edge,smooth),'results':results,
            'process_peak_working_set_bytes':psutil.Process().memory_info().peak_wset}
        (OUT/'summary.partial.json').write_text(json.dumps(summary,indent=2)); print(' fresh-process verified',name,flush=True)
    limits.restore_original_limits()
    return summary

if __name__=='__main__':
    result=run()
    (OUT/'summary.json').write_text(json.dumps(result,indent=2))
