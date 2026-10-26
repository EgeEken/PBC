import json, os, sys, time
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
ROOT=Path(__file__).resolve().parents[1]
os.environ.update({k:'1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS','VECLIB_MAXIMUM_THREADS')})
sys.path.insert(0,str(ROOT))
import learned_filler as lf
import pbc3_ops as ops
from PBC3 import PBC3
from pbc3_types import PBC3Config
IMAGES=('cat.jpg','ducklings.jpg','flower.jpg','monet.jpg','panorama.png','seagull.jpg')
PRESETS=('quality','high_quality'); SAMPLE_STEPS={3,8,13}
orig_builder=lf.build_action; orig_forward=lf.LearnedFiller._forward; orig_select=lf.LearnedFiller.select_patch
state={}
def final_sse(canvas):
    img=PBC3._canvas_to_image(canvas,'YCbCr',False)
    if img.size!=state['source'].size: img=img.resize(state['source'].size,PBC3.RESAMPLE_FILTER,reducing_gap=PBC3.RESAMPLE_REDUCING_GAP)
    d=np.asarray(img,dtype=np.int16)-state['rgb']; return int(np.sum(d.astype(np.int64)**2))
def forward(self,x,trace=None):
    result=orig_forward(self,x,trace); state['logits']=result[0][0]; return result
def select(self,target,canvas,config,rng,channel_bits,step,current_channel,trace=None,abs_error=None):
    state['step']=step; return orig_select(self,target,canvas,config,rng,channel_bits,step,current_channel,trace,abs_error)
def builder(target,canvas,box,config,channel_bits,action_index,trace=None):
    built=orig_builder(target,canvas,box,config,channel_bits,action_index,trace)
    step=state.get('step')
    if step not in SAMPLE_STEPS or step not in state['wanted'] or state.get('logits') is None: return built
    patch,values,delta,reduction,bits=built
    order=np.argsort(state['logits'])[::-1]
    alt_index=next((int(i) for i in order if int(i)!=int(action_index)),None)
    if alt_index is None: return built
    start=time.perf_counter(); alt=orig_builder(target,canvas,box,config,channel_bits,alt_index,trace); alt_build=time.perf_counter()-start
    ap,av,ad,ared,abits=alt
    start=time.perf_counter(); before=final_sse(canvas)
    selected_canvas=canvas.copy(); ops.apply_delta(selected_canvas[:,:,patch['channel']],patch['x'],patch['y'],patch['w'],patch['h'],delta)
    selected_after=final_sse(selected_canvas)
    alternate_canvas=canvas.copy(); ops.apply_delta(alternate_canvas[:,:,ap['channel']],ap['x'],ap['y'],ap['w'],ap['h'],ad)
    alternate_after=final_sse(alternate_canvas); score_time=time.perf_counter()-start
    denom=3*state['rgb'].shape[0]*state['rgb'].shape[1]; sg=before-selected_after; ag=before-alternate_after
    state['rows'].append(dict(image=state['image'],preset=state['preset'],step=step,
      selected=dict(box=[patch['channel'],patch['x'],patch['y'],patch['w'],patch['h']],action=int(action_index),cell=patch['cell_size'],bits=bits,working_sse_reduction=float(reduction),final_rgb_sse_reduction=sg,final_rgb_mse_delta=float(-sg/denom)),
      alternative=dict(box=[ap['channel'],ap['x'],ap['y'],ap['w'],ap['h']],action=alt_index,cell=ap['cell_size'],bits=abits,working_sse_reduction=float(ared),final_rgb_sse_reduction=ag,final_rgb_mse_delta=float(-ag/denom)),
      selected_action_rank=int(np.where(order==action_index)[0][0])+1,alternative_action_rank=int(np.where(order==alt_index)[0][0])+1,equal_raw_bits=bits==abits,
      working_vs_rgb_order_disagrees=bool((reduction-ared)*(sg-ag)<0),alternative_build_seconds=round(alt_build,6),final_rgb_scoring_seconds=round(score_time,6),additional_candidate_seconds=round(alt_build+score_time,6)))
    return built
lf.LearnedFiller._forward=forward; lf.LearnedFiller.select_patch=select; lf.build_action=builder
out=ROOT/'research_pass11'/'rgb_probe.jsonl'
if out.exists(): out.unlink()
for image_name in IMAGES:
  with Image.open(ROOT/'hpt_data'/image_name) as opened: src=ImageOps.exif_transpose(opened).convert('RGB')
  for preset in PRESETS:
    state.clear(); state.update(image=image_name,preset=preset,source=src,rgb=np.asarray(src,dtype=np.int16),rows=[],wanted=SAMPLE_STEPS,logits=None,step=0)
    cfg=getattr(PBC3Config,preset)(learned_filler_model_path=str(ROOT/'patch_policy.npz'))
    PBC3.compress(src,config=cfg,rng_mode='release')
    with out.open('a',encoding='utf-8') as f:
      for row in state['rows']: f.write(json.dumps(row,separators=(',',':'))+'\n')
    print(json.dumps(dict(image=image_name,preset=preset,captured=len(state['rows']))),flush=True)
