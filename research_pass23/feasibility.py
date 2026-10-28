import time
import tracemalloc
from scipy.ndimage import sobel,minimum_filter,maximum_filter
from common import *
from research_pass23.working_target import adjust,reconstruct


def main():
    setup();rows=[]
    config=PBC3Config.quality()
    for name in IMAGES:
        image=load(name);prep=PBC3.prepare(image,config)
        source=np.asarray(image,dtype=np.uint8)
        luma=source.astype(np.float32).mean(axis=2)
        gradient=sobel(luma,axis=0,mode='reflect')**2+sobel(luma,axis=1,mode='reflect')**2
        edges=gradient>=np.percentile(gradient,90)
        local_min=minimum_filter(source,size=(3,3,1),mode='nearest').astype(np.int16)
        local_max=maximum_filter(source,size=(3,3,1),mode='nearest').astype(np.int16)
        for setting,(step,iterations,limit) in SETTINGS.items():
            adjust(prep,step,iterations,limit)
            diagnostics=[]
            start=time.perf_counter();target=adjust(prep,step,iterations,limit,diagnostics);seconds=time.perf_counter()-start
            tracemalloc.start();traced=adjust(prep,step,iterations,limit);current,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
            assert np.array_equal(target,traced) and target.dtype==np.int32 and target.shape==prep['target'].shape and target.min()>=0 and target.max()<=255
            assert np.max(np.abs(target-prep['target']))<=limit if iterations else np.array_equal(target,prep['target'])
            output=reconstruct(target,prep)
            error=source.astype(np.float32)-output.astype(np.float32)
            sse=np.sum(error**2,axis=2,dtype=np.float64)
            delta=target-prep['target']
            overshoot=np.maximum(np.maximum(local_min-output.astype(np.int16),output.astype(np.int16)-local_max),0)
            spatial=[[float(sse[np.ix_(ys,xs)].mean()/3) for xs in np.array_split(np.arange(source.shape[1]),4)] for ys in np.array_split(np.arange(source.shape[0]),4)]
            variation=float(np.mean(np.abs(np.diff(target,axis=0))))+float(np.mean(np.abs(np.diff(target,axis=1))))
            row=dict(image=name,setting=setting,source_size=list(image.size),working_size=[prep['w'],prep['h']],rgb_mse=float(sse.mean()/3),edge_rgb_mse=float(sse[edges].mean()/3),nonedge_rgb_mse=float(sse[~edges].mean()/3),edge_fraction=float(edges.mean()),rgb_mse_4x4=spatial,target_mean_abs_change=float(np.mean(np.abs(delta))),target_rmse=float(np.sqrt(np.mean(delta.astype(np.float64)**2))),target_max_abs_change=int(np.max(np.abs(delta))),target_p99_abs_change=float(np.percentile(np.abs(delta),99)),target_component_change_fraction=float(np.mean(delta!=0)),target_total_variation=variation,output_saturated_component_fraction=float(np.mean((output==0)|(output==255))),edge_local_3x3_envelope_overshoot_mean=float(overshoot[edges].mean()),edge_local_3x3_envelope_overshoot_max=int(overshoot[edges].max()),edge_local_3x3_envelope_overshoot_fraction=float(np.mean(overshoot[edges]>0)),preconditioning_seconds=seconds,tracemalloc_current=current,tracemalloc_peak=peak,target_bytes=target.nbytes,source_uint8_bytes=source.nbytes,float_error_layer_bytes=source.shape[0]*source.shape[1]*4,diagnostics=diagnostics,target_sha256=sha(target.tobytes()),pixels_sha256=sha(output.tobytes()))
            rows.append(row);print(name,setting,round(row['rgb_mse'],4),round(seconds,4),flush=True)
        ordinary=next(r for r in rows if r['image']==name and r['setting']=='ordinary')
        for row in rows:
            if row['image']==name:row['mse_change_percent']=100*(row['rgb_mse']/ordinary['rgb_mse']-1)
    gates={setting:sum(r['mse_change_percent']<=-5 for r in rows if r['setting']==setting)>=2 and all(r['mse_change_percent']<=1 for r in rows if r['setting']==setting) for setting in ('conservative','stronger')}
    chosen=next((setting for setting in ('conservative','stronger') if gates[setting]),None)
    write('feasibility.json',dict(provenance=meta(),config=config.__dict__,rows=rows,gate=gates,chosen=chosen,timing='One untraced warmed target-only preconditioning call per image/setting; separate traced call measures NumPy/Python temporary allocations, not all PIL native buffers or total process RSS. Source loading/ordinary prepare/final scoring excluded; candidate includes all feedback reconstructions and signed projections. No image encoding.'))
    print('target gate',gates,'chosen',chosen,flush=True)


if __name__=='__main__':main()
