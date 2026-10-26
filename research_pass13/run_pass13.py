import hashlib
import importlib.metadata
import json
import math
import os
import platform
from collections import Counter, defaultdict
from pathlib import Path
import statistics
import sys

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
from pbc3_ops import signed_resample, ceil_div
from pbc3_types import BitReader, PBC3Config

IMAGES=('cat.jpg','flower.jpg','monet.jpg','googlewiki.png','mario64.png','eiffel.png','stockpoint.jpg','ratedistortion.png')
PRESETS=('quality','high_quality')
SOURCE_FILES=('PBC3.py','pbc3_types.py','pbc3_heads.py','pbc3_ops.py','learned_filler.py','pbc3_stream.py','pbc3_features.py')
sha=lambda b:hashlib.sha256(b).hexdigest()

def load_rgb(path):
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert('RGB')

def entropy(values):
    counts=Counter(values)
    n=sum(counts.values())
    if not n: return 0.0
    return -sum((v/n)*math.log2(v/n) for v in counts.values())

def pearson(a,b):
    a=np.asarray(a,dtype=np.float64).ravel(); b=np.asarray(b,dtype=np.float64).ravel()
    if len(a)<2 or np.std(a)==0 or np.std(b)==0: return None
    return float(np.corrcoef(a,b)[0,1])

def rect_iou(a,b):
    x0=max(a['x'],b['x']); y0=max(a['y'],b['y'])
    x1=min(a['x']+a['w'],b['x']+b['w']); y1=min(a['y']+a['h'],b['y']+b['h'])
    inter=max(0,x1-x0)*max(0,y1-y0)
    union=a['w']*a['h']+b['w']*b['h']-inter
    return (inter/union if union else 0.0),(x0,y0,x1,y1)

def support_overlap(a,b,intersection):
    x0,y0,x1,y1=intersection
    if x1<=x0 or y1<=y0: return None
    da=a['delta'][y0-a['y']:y1-a['y'],x0-a['x']:x1-a['x']]
    db=b['delta'][y0-b['y']:y1-b['y'],x0-b['x']:x1-b['x']]
    sa=da!=0; sb=db!=0
    union=int(np.count_nonzero(sa|sb))
    common=int(np.count_nonzero(sa&sb))
    return dict(jaccard=(common/union if union else 1.0),both_nonzero=common,union_nonzero=union,
        support_a=int(np.count_nonzero(sa)),support_b=int(np.count_nonzero(sb)),
        pixel_corr=pearson(da,db),active_corr=pearson(da[sa&sb],db[sa&sb]))

def parse_stream(data):
    method,body=PBC3._open_body(data)
    br=BitReader(body)
    header=PBC3._read_header(br)
    _,ow,oh,w,h,color,channels,channel_bits,positive_bias,has_alpha,patch_count,_,warmup_on,_,_,_=header
    header_end=br.i*8-br.nbits
    patches=[]
    for index in range(patch_count):
        start=br.i*8-br.nbits
        channel,x,y,pw,ph,cell,values,bits=PBC3._read_patch(br,channel_bits,positive_bias)
        end=br.i*8-br.nbits
        grid_h,grid_w=values.shape
        delta=signed_resample(values,ph,pw)
        patches.append(dict(index=index,kind='initialization' if index<channels else 'iterative',channel=int(channel),
            x=int(x),y=int(y),w=int(pw),h=int(ph),cell=int(cell),grid_w=int(grid_w),grid_h=int(grid_h),
            cells=int(grid_w*grid_h),bitcount=int(bits),raw_bits=int(end-start),grid_bits=int(grid_w*grid_h*bits),
            values=np.asarray(values,dtype=np.int16),delta=np.asarray(delta),palette_value_histogram={str(int(k)):int(v) for k,v in zip(*np.unique(values,return_counts=True))}))
    return dict(method=int(method),body_bytes=len(body),raw_bytes=6+len(body),final_bytes=len(data),header_bits=header_end,
        width=int(w),height=int(h),original_size=[int(ow or w),int(oh or h)],color=color,channel_bits=int(channel_bits),
        channels=int(channels),patch_count=int(patch_count),initialization_patches=int(channels),iterative_patches=int(patch_count-channels),
        warmup=bool(warmup_on),patches=patches)

def analyze_stream(parsed):
    patches=[p for p in parsed['patches'] if p['kind']=='iterative']
    totals=dict(iterative_patches=len(patches),cross_channel_pairs=0,overlapping_rectangle_pairs=0,
        iou_ge_050_pairs=0,exact_rectangle_pairs=0,exact_grid_pairs=0,aligned_grid_pairs=0)
    ious=[]; support_j=[]; pixel_corr=[]; active_corr=[]; temporal=[]; exact_pairs=[]
    for i,a in enumerate(patches):
        for b in patches[i+1:]:
            if a['channel']==b['channel']: continue
            totals['cross_channel_pairs']+=1
            iou,intersection=rect_iou(a,b)
            if iou<=0: continue
            totals['overlapping_rectangle_pairs']+=1; ious.append(iou)
            if iou>=0.5: totals['iou_ge_050_pairs']+=1
            dt=abs(a['index']-b['index']); temporal.append(dt)
            s=support_overlap(a,b,intersection)
            if s:
                support_j.append(s['jaccard'])
                if s['pixel_corr'] is not None: pixel_corr.append(s['pixel_corr'])
                if s['active_corr'] is not None: active_corr.append(s['active_corr'])
            exact=(a['x'],a['y'],a['w'],a['h'])==(b['x'],b['y'],b['w'],b['h'])
            if exact:
                totals['exact_rectangle_pairs']+=1
                compatible=a['cell']==b['cell'] and a['grid_w']==b['grid_w'] and a['grid_h']==b['grid_h']
                if compatible:
                    totals['exact_grid_pairs']+=1
                    aligned=((a['x']-b['x'])%a['cell']==0 and (a['y']-b['y'])%a['cell']==0)
                    if aligned: totals['aligned_grid_pairs']+=1
                    corr=pearson(a['values'],b['values'])
                    j_a=a['values']!=0; j_b=b['values']!=0
                    union=int(np.count_nonzero(j_a|j_b)); common=int(np.count_nonzero(j_a&j_b))
                    exact_pairs.append(dict(patch_a=a,patch_b=b,pearson=corr,grid_support_jaccard=(common/union if union else 1.0),
                        temporal_distance=dt,iou=iou,aligned=aligned))
    for key,values in (('rectangle_iou',ious),('decoded_support_jaccard',support_j),('decoded_delta_pearson',pixel_corr),('active_delta_pearson',active_corr),('temporal_distance',temporal)):
        totals[key+'_mean']=statistics.mean(values) if values else None
        totals[key+'_median']=statistics.median(values) if values else None
        totals[key+'_p90']=float(np.quantile(values,.9)) if values else None
    totals['compatible_grid_value_pearson_mean']=statistics.mean([p['pearson'] for p in exact_pairs if p['pearson'] is not None]) if any(p['pearson'] is not None for p in exact_pairs) else None
    totals['compatible_grid_support_jaccard_mean']=statistics.mean([p['grid_support_jaccard'] for p in exact_pairs]) if exact_pairs else None
    totals['exact_pair_temporal_distance_mean']=statistics.mean([p['temporal_distance'] for p in exact_pairs]) if exact_pairs else None
    return totals,exact_pairs

def multi_information(arrays):
    arrays=[np.asarray(a).ravel().tolist() for a in arrays]
    if not arrays or any(len(a)!=len(arrays[0]) for a in arrays): return dict(samples=0,savings_bits=0.0)
    n=len(arrays[0])
    marginals=[entropy(a) for a in arrays]
    joint=entropy(list(zip(*arrays)))
    return dict(samples=n,marginal_entropy_sum_bits_per_cell=sum(marginals),joint_entropy_bits_per_cell=joint,
        total_correlation_bits_per_cell=max(0.0,sum(marginals)-joint),ideal_payload_saving_bits=max(0.0,sum(marginals)-joint)*n)

def stream_metadata(image,preset,source,config,data,parsed,expected,pass9_expected):
    result=PBC3.decompress(data)
    decoded=np.asarray(result.image.convert('RGB'))
    reference=np.asarray(source)
    error=np.subtract(reference,decoded,dtype=np.int16).astype(np.int64)
    mse=float(np.mean(error*error,dtype=np.float64))
    parsed_summary={k:v for k,v in parsed.items() if k!='patches'}
    raw_index_bits=sum(p['grid_bits'] for p in parsed['patches'])
    geometry_bits=sum(64 for p in parsed['patches'])
    parsed_summary.update(raw_index_bits=raw_index_bits,geometry_coordinate_bits=geometry_bits,
        raw_index_share_percent=100*raw_index_bits/(parsed['body_bytes']*8),
        geometry_coordinate_share_percent=100*geometry_bits/(parsed['body_bytes']*8),
        bpp=8*len(data)/(source.width*source.height),mse=mse,stream_sha256=sha(data),
        decoded_pixels_sha256=sha(decoded.tobytes()),input_sha256=sha((ROOT/'hpt_data'/image).read_bytes()),
        input_size=[source.width,source.height],preview_equal=bool(np.array_equal(np.asarray(result.image),decoded)),
        config_sha256=sha(json.dumps(vars(config),sort_keys=True,separators=(',',':')).encode()),
        expected_hf_stream_sha256=expected['stream_sha256'],expected_hf_patch_count=int(expected['patches']),
        hf_stream_match=(sha(data)==expected['stream_sha256'] and parsed['patch_count']==int(expected['patches'])),
        pass9_stream_match=(sha(data)==pass9_expected['release']['stream_sha256'] and parsed['patch_count']==pass9_expected['release']['patches']),
        pass9_trajectory_sha256=pass9_expected['release']['trajectory_sha256'])
    assert result.image.size==source.size and parsed['color']=='YCbCr' and parsed['channels']==3 and parsed['warmup'] is False
    assert parsed_summary['preview_equal']
    return parsed_summary

def main():
    manifest=json.loads((ROOT/'research_pass10'/'baseline_manifest.json').read_text(encoding='utf-8'))
    source_hashes={n:sha((ROOT/n).read_bytes()) for n in SOURCE_FILES}
    source_hashes['patch_policy.npz']=sha((ROOT/'patch_policy.npz').read_bytes())
    assert all(source_hashes[n]==manifest['source_sha256'][n] for n in SOURCE_FILES)
    assert source_hashes['patch_policy.npz']==manifest['model_sha256']
    p9=json.loads((ROOT/'research_pass9'/'comparison_summary.json').read_text(encoding='utf-8'))
    inputs={(r['dataset'],r['image']):r['input_sha256'] for r in p9['equivalence']}
    pass9={(r['dataset'],r['image'],r['preset']):r for r in p9['equivalence']}
    p10=json.loads((ROOT/'research_pass10'/'public_benchmark.json').read_text(encoding='utf-8'))
    expected={(r['image'],r['variant']):r for r in p10['rows'] if r['run_id']==p10['selected_run'] and r['trial']==1}
    run_rows=[]; streams=[]
    for image in IMAGES:
        path=ROOT/'hpt_data'/image; src=load_rgb(path)
        input_hash=sha(path.read_bytes())
        assert input_hash==inputs['hpt_data',image],f'input hash mismatch: {image}'
        for preset in PRESETS:
            cfg=getattr(PBC3Config,preset)()
            assert vars(cfg)==manifest['presets'][preset],f'config mismatch: {preset}'
            encoded=PBC3.compress(src,config=cfg,rng_mode='release')
            expected_hf=expected[image,preset]
            expected_p9=pass9['hpt_data',image,preset]
            parsed=parse_stream(encoded.data)
            meta=stream_metadata(image,preset,src,cfg,encoded.data,parsed,expected_hf,expected_p9)
            pair_stats,exact_pairs=analyze_stream(parsed)
            meta.update(pair_stats)
            meta['exact_compatible_pairs']=[dict(channel_a=p['patch_a']['channel'],channel_b=p['patch_b']['channel'],
                index_a=p['patch_a']['index'],index_b=p['patch_b']['index'],x=p['patch_a']['x'],y=p['patch_a']['y'],w=p['patch_a']['w'],h=p['patch_a']['h'],
                cell=p['patch_a']['cell'],grid_w=p['patch_a']['grid_w'],grid_h=p['patch_a']['grid_h'],raw_grid_bits_a=p['patch_a']['grid_bits'],
                raw_grid_bits_b=p['patch_b']['grid_bits'],grid_value_pearson=p['pearson'],grid_support_jaccard=p['grid_support_jaccard'],
                temporal_distance=p['temporal_distance'],aligned=p['aligned']) for p in exact_pairs]
            run_rows.append(dict(image=image,preset=preset,**meta))
            streams.append((image,preset,parsed,exact_pairs))
            print(json.dumps(dict(image=image,preset=preset,hf_match=meta['hf_stream_match'],patches=meta['patch_count'],pairs=meta['cross_channel_pairs'],exact=meta['exact_grid_pairs'])),flush=True)
    summary=[]; total_geometry_saved=0; total_payload_mi=0.0; total_raw=0; total_final=0; total_iter=0; total_cross=0; total_exact=0; total_iou=0
    for image,preset,parsed,pairs in streams:
        bykey=defaultdict(lambda:defaultdict(list))
        for p in parsed['patches']:
            if p['kind']=='iterative': bykey[(p['x'],p['y'],p['w'],p['h'],p['cell'],p['grid_w'],p['grid_h'])][p['channel']].append(p)
        group_stats=[]; saved=0; mi_bits=0.0
        for key,channels in bykey.items():
            selected=[v[0] for v in channels.values()]
            if len(selected)<2: continue
            n=len(selected)
            geometry_saving=max(0,(n-1)*(64+16)+n*parsed['channel_bits']-(parsed['channels']+1))
            saved+=geometry_saving
            mi=multi_information([p['values'] for p in selected])
            mi_bits+=mi['ideal_payload_saving_bits']
            group_stats.append(dict(x=key[0],y=key[1],w=key[2],h=key[3],cell=key[4],channels=sorted(channels),patch_indices=[p['index'] for p in selected],
                geometry_saving_upper_bound_bits=geometry_saving,**mi))
        st=next(r for r in run_rows if r['image']==image and r['preset']==preset)
        total_geometry_saved+=saved; total_payload_mi+=mi_bits; total_raw+=st['raw_bytes']; total_final+=st['final_bytes']; total_iter+=st['iterative_patches']
        total_cross+=st['cross_channel_pairs']; total_exact+=st['exact_grid_pairs']; total_iou+=st['iou_ge_050_pairs']
        summary.append(dict(image=image,preset=preset,iterative_patches=st['iterative_patches'],cross_channel_pairs=st['cross_channel_pairs'],
            exact_rectangle_pairs=st['exact_rectangle_pairs'],exact_grid_pairs=st['exact_grid_pairs'],iou_ge_050_pairs=st['iou_ge_050_pairs'],
            mean_rectangle_iou=st['rectangle_iou_mean'],mean_support_jaccard=st['decoded_support_jaccard_mean'],
            mean_delta_correlation=st['decoded_delta_pearson_mean'],mean_aligned_grid_value_correlation=st['compatible_grid_value_pearson_mean'],
            exact_groups=len(group_stats),geometry_saving_upper_bound_bits=saved,ideal_payload_mutual_information_bits=mi_bits,
            raw_bytes=st['raw_bytes'],final_bytes=st['final_bytes'],stream_sha256=st['stream_sha256'],hf_stream_match=st['hf_stream_match'],groups=group_stats))
    env=dict(python=sys.version,platform=platform.platform(),numpy=np.__version__,numba=importlib.metadata.version('numba'),pillow=Image.__version__,numba_threads=numba.get_num_threads(),thread_env=THREADS)
    output=dict(schema='pbc-pass13-cross-channel-v1',baseline_id=manifest['baseline_id'],github_baseline_source=manifest['github_source_revision'],research_head_at_run='60c058bff597422026b636234ecbb10e62c3e51f',hf_baseline=manifest['hf_revision'],model_sha256=source_hashes['patch_policy.npz'],source_sha256=source_hashes,environment=env,
        cases=run_rows,summary=summary,aggregate=dict(streams=len(streams),hf_stream_matches=sum(r['hf_stream_match'] for r in run_rows),hf_streams=len(run_rows),iterative_patches=total_iter,
            cross_channel_pairs=total_cross,exact_grid_pairs=total_exact,iou_ge_050_pairs=total_iou,geometry_saving_upper_bound_bits=total_geometry_saved,
            geometry_saving_upper_bound_percent_raw=100*total_geometry_saved/(total_raw*8),geometry_saving_upper_bound_percent_final=100*total_geometry_saved/(total_final*8),
            payload_mutual_information_upper_bound_bits=total_payload_mi,payload_mutual_information_upper_bound_percent_raw=100*total_payload_mi/(total_raw*8),
            payload_mutual_information_upper_bound_percent_final=100*total_payload_mi/(total_final*8),total_raw_bytes=total_raw,total_final_bytes=total_final))
    (OUT/'channel_analysis.json').write_text(json.dumps(output,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(output['aggregate'],indent=2))
if __name__=='__main__': main()
