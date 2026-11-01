from __future__ import annotations
import hashlib,json,subprocess,sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageOps
from scipy.ndimage import sobel
ROOT=Path(__file__).resolve().parents[1];HERE=Path(__file__).resolve().parent;OUT=HERE/'outputs'
sys.path.insert(0,str(HERE.parent/'research_pass33'));sys.path.insert(0,str(ROOT))
import pbc34_codec as p34
def sha(b):return hashlib.sha256(b).hexdigest()
def masks(src):
 g=src.astype(np.float32).mean(2);m=np.hypot(sobel(g,axis=0),sobel(g,axis=1));return m>=np.quantile(m,.9),m<=np.quantile(m,.5)
def metrics(src,rec,e,s):
 err=np.square(src.astype(np.int32)-rec.astype(np.int32),dtype=np.int64).sum(2)
 return {'mse':float(err.mean()/3),'edge_top10_mse':float(err[e].mean()/3),'smooth_bottom50_mse':float(err[s].mean()/3)}
def main():
 summary=json.loads((OUT/'codec_summary.json').read_text()); selection=json.loads((HERE/'dataset_selection.json').read_text()); rows=[]
 for image in summary['images']:
  name=image['image'];src=np.asarray(ImageOps.exif_transpose(Image.open(ROOT/'hpt_data'/name)).convert('RGB'),np.uint8);edge,smooth=masks(src)
  full=next(r for r in image['rows'] if r['band']=='F' and r['qstep']==32 and r['lambda']==400.0)
  stream=(HERE/full['stream_path']).resolve();data=stream.read_bytes();base=np.load(OUT/'bases'/f'{name}.base.npy',allow_pickle=False)
  fullrec=np.load(OUT/'reconstructions'/name/'S_F_q32_lambda400.npy',allow_pickle=False)
  candidates=[]
  for remove in ('L','H'):
   cf=p34.counterfactual_remove(data,remove);keep='H' if remove=='L' else 'L';path=OUT/'counterfactuals'/name/f'remove_{remove}_keep_{keep}.p34s';path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(cf)
   decoded,modes,params=p34.decode(cf,base);acc=p34.accounting(cf);acc.pop('lambda',None);acc.pop('base_bytes',None)
   candidates.append({'key':f'CF_remove_{remove}_keep_{keep}','remove_band':remove,'kept_band':keep,'stream':str(path),'type':'p34','base_npy':str(OUT/'bases'/f'{name}.base.npy'),'expected_pixels_sha256':sha(decoded.tobytes()),'expected_modes_sha256':sha(modes.tobytes()),'expected_params_sha256':sha(params),'metrics':metrics(src,decoded,edge,smooth),'enhancement_bytes':len(cf),'total_bytes':image['base_bytes']+len(cf),'total_bpp':(image['base_bytes']+len(cf))*8/(src.shape[0]*src.shape[1]),'conditional_delta_from_full':{'mse':metrics(src,decoded,edge,smooth)['mse']-full['metrics']['mse'],'enhancement_bytes':len(cf)-full['enhancement_bytes'],'total_bytes':image['base_bytes']+len(cf)-full['total_bytes'],'active_blocks':acc['active_blocks']-full['active_blocks'],'coefficients':acc['coefficients']-full['coefficients']},'accounting':acc})
  manifest=OUT/'counterfactuals'/name/'decode_manifest.json';manifest.write_text(json.dumps({'files':[{k:v for k,v in c.items() if k in ('key','stream','type','base_npy','expected_pixels_sha256','expected_modes_sha256','expected_params_sha256')} for c in candidates]},indent=2)+'\n')
  proc=subprocess.run([sys.executable,'-B',str(HERE/'worker_decode.py'),str(manifest)],cwd=ROOT,check=True,capture_output=True,text=True);fresh={r['key']:r for r in json.loads(proc.stdout)['rows']}
  for c in candidates:
   c['fresh_process_verified']=fresh[c['key']]['fresh_process_verified']
   c['fresh_decode_profile']=fresh[c['key']]['profile']
   rec=np.load(OUT/'reconstructions'/name/(c['key']+'.npy'),allow_pickle=False)
   if sha(rec.tobytes())!=c['expected_pixels_sha256']:raise ValueError('fresh counterfactual mismatch')
  rows.append({'image':name,'full_reference_key':full['key'],'full_reference_bytes':full['enhancement_bytes'],'counterfactuals':candidates})
  print('counterfactuals fresh decoded',name,flush=True)
 result={'protocol':'Conditional ablation: remove L or H coefficients from one fixed F q32 lambda400 completed stream; remaining modes/blocks/cell masks are reserialized and shared XZ recompressed. These are conditional deltas, not independent band encodes.','images':rows}
 (OUT/'counterfactual_summary.json').write_text(json.dumps(result,indent=2)+'\n')
 print('conditional ablations complete',len(rows))
if __name__=='__main__':main()
