import json, time, sys
from pathlib import Path
import numpy as np
import phase_a_support as e

OUT=e.OUT

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    summary={'config':{'cell':16,'grid':4,'dc_step':8,'pattern_steps':[8,16],'coefficients':e.COEFF_NUM.tolist(),'iterations':4},'images':{}}
    names=sys.argv[1:] or e.NAMES
    for name in names:
        t0=time.perf_counter(); item=e.load_case(name); src,base=item['src'],item['base']
        r,valid,rows,cols=e.cells(src.astype(np.int16)-base.astype(np.int16)); low=e.low_projection(r); dc=e.qdc(low); n=len(dc)
        dc_rec=e.reconstruct(base,dc); dc_mse=e.metric(src,dc_rec)['mse']
        dc_full=np.broadcast_to(dc[:,None,None,:],(n,16,16,3)).astype(np.int32)
        dc_sse=float(e.cell_sse(r,dc_full,valid).sum())
        groups=[('global',1,np.zeros(n,np.uint8)),('pass29_distribution_structure',4,item['dist_assign']),('residual_shape',4,e.structural_groups(low))]
        per=[]; fitted={}
        for label,g,assign in groups:
            for q in e.PATTERN_STEPS:
                t=time.perf_counter(); pats,coef=e.fit(low,dc,assign,g,q); fit_s=time.perf_counter()-t
                rec=e.reconstruct(base,dc,pats,coef,assign if g>1 else None); mse=e.metric(src,rec)['mse']
                action=np.zeros_like(r,dtype=np.int32)
                for group in range(g):
                    ids=np.flatnonzero(((assign==group) if g>1 else np.ones(n,bool))&(coef!=3))
                    if len(ids): action[ids]=e.rnd(e.COEFF_NUM[coef[ids],None,None,None]*e.bilinear(pats[group])[None].astype(np.float32)/2).astype(np.int32)
                exact_sse=float(e.cell_sse(r,dc_full+action,valid).sum())
                ideal_action=np.zeros_like(r,dtype=np.int32)
                for group in range(g):
                    ids=np.flatnonzero((assign==group) if g>1 else np.ones(n,bool)); pat=e.bilinear(pats[group]).astype(np.float32)
                    den=float((pat*pat).sum())
                    if den:
                        scalars=np.clip(np.einsum('nhwc,hwc->n',r[ids].astype(np.float32)-dc[ids,None,None,:],pat)/den,-2,2)
                        ideal_action[ids]=e.rnd(scalars[:,None,None,None]*pat[None]).astype(np.int32)
                ideal_sse=float(e.cell_sse(r,dc_full+ideal_action,valid).sum())
                used=[int(np.count_nonzero((assign==k)&(coef!=3))) for k in range(g)]
                per.append({'name':label,'q':q,'mse':mse,'improvement_vs_dc_pct':100*(dc_mse-mse)/dc_mse,
                    'residual_sse':exact_sse,'residual_sse_reduction_beyond_dc_pct':100*(dc_sse-exact_sse)/dc_sse,
                    'ideal_unpriced_coefficient_sse':ideal_sse,'ideal_unpriced_reduction_beyond_dc_pct':100*(dc_sse-ideal_sse)/dc_sse,
                    'active_cells':int(np.count_nonzero(coef!=3)),'group_sizes':np.bincount(assign,minlength=g).tolist(),
                    'used_by_group':used,'fit_seconds':fit_s})
                fitted[f'{label}|{q}']={'patterns':pats,'coeff':coef,'assign':assign}
        # Matched low-resolution independent cells: same DC and quantization step, exact decoder output.
        r_cells,valid,rows,cols=e.cells(src.astype(np.int16)-base.astype(np.int16))
        base_sse=e.cell_sse(r_cells,dc_full,valid); lowres=low-dc[:,None,None,:]
        for q in e.PATTERN_STEPS:
            p=np.clip(e.rnd(lowres/q)*q,-256,256).astype(np.int16); expanded=np.stack([e.bilinear(x) for x in p])
            active=e.cell_sse(r_cells,dc_full+expanded.astype(np.int32),valid)<base_sse; p[~active]=0
            delta=np.broadcast_to(dc[:,None,None,:],(n,16,16,3)).astype(np.int32).copy(); delta[active]+=np.stack([e.bilinear(x) for x in p[active]]).astype(np.int32)
            patch=delta.reshape(rows,cols,16,16,3).transpose(0,2,1,3,4).reshape(rows*16,cols*16,3)[:src.shape[0],:src.shape[1]]
            rec=np.clip(base.astype(np.int32)+patch,0,255).astype(np.uint8); mse=e.metric(src,rec)['mse']
            independent_sse=float(e.cell_sse(r_cells,dc_full+np.stack([e.bilinear(x) for x in p]).astype(np.int32),valid).sum())
            per.append({'name':'independent_4x4','q':q,'mse':mse,'improvement_vs_dc_pct':100*(dc_mse-mse)/dc_mse,
                'residual_sse':independent_sse,'residual_sse_reduction_beyond_dc_pct':100*(dc_sse-independent_sse)/dc_sse,
                'active_cells':int(active.sum()),'group_sizes':[],'used_by_group':[]})
        summary['images'][name]={'width':src.shape[1],'height':src.shape[0],'cells':n,'source_sha256':item['source_hash'],
            'base_stream_sha256':item['base_hash'],'base_stream_bytes':len(item['base_bytes']),'dc_mse':dc_mse,
            'dc_residual_sse':dc_sse,'no_enhancement_mse':e.metric(src,base)['mse'],'methods':per,'phase_a_seconds':time.perf_counter()-t0}
        np.savez_compressed(OUT/f'{name}.fit.npz',**{f'{k}.{field}':arr for k,obj in fitted.items() for field,arr in obj.items()})
        (OUT/'phase_a.json').write_text(json.dumps(summary,indent=2)); print(name,per,flush=True)
    wins=sum(any(m['name']=='residual_shape' and m['residual_sse_reduction_beyond_dc_pct']>=10 and max(m['used_by_group'],default=0)>=2 for m in d['methods']) for d in summary['images'].values())
    passes=wins>=2
    summary['gate']={'content_classes_with_structural_shared_action_at_least_10pct':wins,'pass':passes}
    (OUT/'phase_a.json').write_text(json.dumps(summary,indent=2))
    print('PHASE_A_GATE',json.dumps(summary['gate']),flush=True)

if __name__=='__main__':main()
