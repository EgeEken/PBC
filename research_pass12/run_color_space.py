import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path
import statistics
import sys
import time

THREADS={k:'1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS','VECLIB_MAXIMUM_THREADS')}
os.environ.update(THREADS)
ROOT=Path(__file__).resolve().parents[1]
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image, ImageOps
import numba
numba.set_num_threads(1)
from PBC3 import PBC3
from pbc3_types import BitReader, PBC3Config

IMAGES=('cat.jpg','flower.jpg','monet.jpg','googlewiki.png','mario64.png','eiffel.png','stockpoint.jpg','ratedistortion.png')
PRESETS=('quality','high_quality')
SPACES=('YCbCr','RGB')
SOURCES=('PBC3.py','pbc3_types.py','pbc3_heads.py','pbc3_ops.py','learned_filler.py','pbc3_stream.py','pbc3_features.py')
sha=lambda data:hashlib.sha256(data).hexdigest()

def load_image(path):
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert('RGB')

def rgb_mse(a,b):
    delta=np.subtract(np.asarray(a,dtype=np.int16),np.asarray(b.convert('RGB'),dtype=np.int16)).astype(np.int64)
    return float(np.mean(delta*delta,dtype=np.float64))

def floor_rows():
    rows=[]
    for name in IMAGES:
        path=ROOT/'hpt_data'/name
        src=load_image(path)
        src_rgb=np.asarray(src,dtype=np.int16)
        for space in SPACES:
            cfg=PBC3Config.quality(color_space=space)
            prep=PBC3.prepare(src,cfg)
            converted=src.convert(space)
            roundtrip=converted.convert('RGB')
            ideal=PBC3._canvas_to_image(prep['arr'][:,:,:3],space,False)
            if ideal.size!=src.size:
                ideal=ideal.resize(src.size,PBC3.RESAMPLE_FILTER,reducing_gap=PBC3.RESAMPLE_REDUCING_GAP)
            rows.append(dict(image=name,color_space=space,input_sha256=sha(path.read_bytes()),width=src.width,height=src.height,
                working_width=prep['w'],working_height=prep['h'],working_pixels=prep['w']*prep['h'],
                colorspace_roundtrip_rgb_mse=rgb_mse(src,roundtrip),ideal_working_reconstruction_rgb_mse=rgb_mse(src,ideal)))
    return rows

def parse_stream(data, decoded, original_size):
    method,body=PBC3._open_body(data)
    br=BitReader(body); header=PBC3._read_header(br)
    downsampled,ow,oh,w,h,color_space,channels,channel_bits,positive_bias,has_alpha,patch_count,base_values,warmup_on,warm_w,warm_h,warm_split=header
    patches=[]
    for _ in range(patch_count):
        c,x,y,pw,ph,cell,values,bits=PBC3._read_patch(br,channel_bits,positive_bias)
        patches.append(dict(channel=int(c),w=int(pw),h=int(ph),cell=int(cell),grid_w=(int(pw)+int(cell)-1)//int(cell),grid_h=(int(ph)+int(cell)-1)//int(cell),bits=int(bits)))
    init_count=channels
    init=patches[:init_count]; iterative=patches[init_count:]
    def hist(key,items):
        values={}
        for patch in items:
            k=str(patch[key]); values[k]=values.get(k,0)+1
        return dict(sorted(values.items(),key=lambda kv:int(kv[0])))
    hbytes=6
    decoded_pixels=np.asarray(decoded.convert('RGB'))
    return dict(stream_version=int(data[4]),entropy_method='LZMA_RAW' if method==PBC3.ENTROPY_LZMA else 'STORE',
      color_space=color_space,original_size=list(original_size),decoded_size=list(decoded.size),working_size=[w,h],
      patch_count=int(patch_count),initialization_patches=len(init),iterative_patches=len(iterative),
      initialization_channel_counts=hist('channel',init),iterative_channel_counts=hist('channel',iterative),
      mean_grid_width=(statistics.mean(p['grid_w'] for p in iterative) if iterative else 0),
      mean_grid_height=(statistics.mean(p['grid_h'] for p in iterative) if iterative else 0),
      mean_grid_cells=(statistics.mean(p['grid_w']*p['grid_h'] for p in iterative) if iterative else 0),
      grid_bitcount_histogram=hist('bits',iterative),cell_size_histogram=hist('cell',iterative),
      raw_body_bytes=len(body),raw_stream_bytes=hbytes+len(body),final_file_bytes=len(data),
      raw_to_final_ratio=(hbytes+len(body))/len(data),decoded_pixels_sha256=sha(decoded_pixels.tobytes()))

def main():
    try:
        import psutil
        proc=psutil.Process(); allowed=proc.cpu_affinity(); core=min(allowed); proc.cpu_affinity([core])
    except Exception:
        core=None
    model_path=ROOT/'patch_policy.npz'
    source_hashes={name:sha((ROOT/name).read_bytes()) for name in SOURCES}
    source_hashes['patch_policy.npz']=sha(model_path.read_bytes())
    manifest=json.loads((OUT.parent/'research_pass10'/'baseline_manifest.json').read_text(encoding='utf-8'))
    for name,digest in source_hashes.items():
        if name in manifest['source_sha256']:
            assert digest==manifest['source_sha256'][name],f'frozen source hash mismatch: {name}'
    assert source_hashes['patch_policy.npz']==manifest['model_sha256']
    floor=floor_rows()
    (OUT/'preprocess_floor.json').write_text(json.dumps(floor,indent=2)+'\n',encoding='utf-8')
    cases={}
    configs={}
    for preset in PRESETS:
        for space in SPACES:
            cfg=getattr(PBC3Config,preset)(color_space=space)
            configs[preset,space]=cfg
    for preset in PRESETS:
        for space in SPACES:
            cfg=configs[preset,space]
            PBC3.compress(Image.new('RGB',(96,96),(53,117,191)),config=cfg,rng_mode='release')
    rows=[]
    reference_path=OUT.parent/'research_pass10'/'public_benchmark.json'
    reference=json.loads(reference_path.read_text(encoding='utf-8'))
    pass9=json.loads((OUT.parent/'research_pass9'/'comparison_summary.json').read_text(encoding='utf-8'))
    saved_inputs={(r['dataset'],r['image']):r['input_sha256'] for r in pass9['equivalence']}
    for name in IMAGES:
        assert sha((ROOT/'hpt_data'/name).read_bytes())==saved_inputs['hpt_data',name],f'input SHA mismatch: {name}'
    for preset in PRESETS:
        yc=vars(configs[preset,'YCbCr']); rgb=vars(configs[preset,'RGB'])
        assert {k:v for k,v in yc.items() if k!='color_space'}=={k:v for k,v in rgb.items() if k!='color_space'}
    selected_run=next(r for r in reference['runs'] if r['run_id']==reference['selected_run'])
    ref={(r['image'],r['variant']):r for r in reference['rows'] if r['run_id']==reference['selected_run'] and r['trial']==1}
    for image_index,name in enumerate(IMAGES):
        path=ROOT/'hpt_data'/name; src=load_image(path); input_hash=sha(path.read_bytes())
        for preset in PRESETS:
            order=('YCbCr','RGB') if (image_index+PRESETS.index(preset))%2==0 else ('RGB','YCbCr')
            samples={(space):[] for space in SPACES}
            for repeat in range(2):
                run_order=order if repeat==0 else tuple(reversed(order))
                for space in run_order:
                    cfg=configs[preset,space]
                    started=time.perf_counter(); encoded=PBC3.compress(src,config=cfg,rng_mode='release'); enc=time.perf_counter()-started
                    started=time.perf_counter(); decoded=PBC3.decompress(encoded.data).image; dec=time.perf_counter()-started
                    assert decoded.size==src.size
                    assert np.array_equal(np.asarray(encoded.image.convert('RGB')),np.asarray(decoded.convert('RGB')))
                    mse=rgb_mse(src,decoded)
                    stream=parse_stream(encoded.data,decoded,src.size)
                    assert stream['color_space']==space and stream['stream_version']==0
                    assert stream['decoded_size']==list(src.size)
                    assert stream['decoded_pixels_sha256']==sha(np.asarray(decoded.convert('RGB')).tobytes())
                    samples[space].append(dict(encode_seconds=enc,decode_seconds=dec,mse=mse,bpp=8*len(encoded.data)/(src.width*src.height),
                      stream_sha256=sha(encoded.data),input_sha256=input_hash,**stream))
            for space in SPACES:
                group=samples[space]
                assert len({r['stream_sha256'] for r in group})==1
                assert len({r['decoded_pixels_sha256'] for r in group})==1
                first=group[0]
                baseline_match=None
                if space=='YCbCr':
                    expected=ref[name,preset]
                    baseline_match=first['stream_sha256']==expected['stream_sha256'] and first['patch_count']==expected['patches']
                    baseline_ref=dict(stream_sha256=expected['stream_sha256'],patch_count=expected['patches'],bpp=expected['bpp'],mse=expected['mse'])
                else:
                    baseline_ref=None
                rows.append(dict(image=name,preset=preset,color_space=space,input_sha256=input_hash,input_size=[src.width,src.height],
                  config_sha256=sha(json.dumps(vars(configs[preset,space]),sort_keys=True,separators=(',',':')).encode()),
                  config=vars(configs[preset,space]),baseline_match_pass10_release=baseline_match,pass10_reference=baseline_ref,
                  mean_encode_seconds=statistics.mean(r['encode_seconds'] for r in group),median_encode_seconds=statistics.median(r['encode_seconds'] for r in group),
                  mean_decode_seconds=statistics.mean(r['decode_seconds'] for r in group),median_decode_seconds=statistics.median(r['decode_seconds'] for r in group),
                  mse=first['mse'],bpp=first['bpp'],stream_sha256=first['stream_sha256'],repeats=len(group),
                  repeat_encode_seconds=[r['encode_seconds'] for r in group],repeat_decode_seconds=[r['decode_seconds'] for r in group],**{k:first[k] for k in ('stream_version','entropy_method','working_size','decoded_size','patch_count','initialization_patches','iterative_patches','initialization_channel_counts','iterative_channel_counts','mean_grid_width','mean_grid_height','mean_grid_cells','grid_bitcount_histogram','cell_size_histogram','raw_body_bytes','raw_stream_bytes','final_file_bytes','raw_to_final_ratio','decoded_pixels_sha256')}))
    metadata=dict(github_baseline_source='8006f30debfc4a57289df06df21636aeb52baffd',research_head='b339f23576cda5c372c3fe9f3e37866e1d274b11',hf_baseline='98ade0c8072a4571e9674eaac3d91f861e3b9867',model_sha256=source_hashes['patch_policy.npz'],source_sha256=source_hashes,baseline_manifest_id=manifest['baseline_id'],
      python=sys.version,platform=platform.platform(),processor=platform.processor(),numpy=np.__version__,numba=importlib.metadata.version('numba'),pillow=Image.__version__,thread_env=THREADS,cpu_affinity=[core] if core is not None else None,
      warmups='one untimed 96x96 RGB encode per preset/color-space pair; two measured interleaved repeats per image/preset/color-space',
      timing_scope='full PBC3.compress and PBC3.decompress calls; source loading and independent RGB scoring outside timers',rng='NumPy default_rng seed 2003, explicitly rng_mode=release',model_path='patch_policy.npz',cases=rows,floor=floor)
    (OUT/'color_space_results.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(rows=len(rows),floor_rows=len(floor),cpu=metadata['cpu_affinity'],baseline_rows=sum(r['baseline_match_pass10_release'] is True for r in rows if r['color_space']=='YCbCr')),indent=2))
if __name__=='__main__': main()
