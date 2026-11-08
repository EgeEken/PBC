import argparse
import json
import os
import re
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
CASES=[("googlewiki.png",5,6),("googlewiki.png",20,6),("mario64.png",50,6),("flower.jpg",80,6),("cat.jpg",35,6),("cat.jpg",50,6),("coast.jpg",80,6),("googlewiki.png",5,8),("cat.jpg",35,8)]

def parse(path):
    text=path.read_text(encoding='utf-8',errors='replace')
    totals={'rd_intra_blocks':0,'candidate_visits':0,'transform_search_entries':0,'pruned_before_transform':0,'invalid_transform_results':0,'selected_mode_records':0,'hybrid_rd_calls':0,'hybrid_nonrd_calls':0}
    by_mode={}
    for m in re.finditer(r'^PBC_PROFILE rd_intra_bsize=\d+ blocks=(\d+)',text,re.M): totals['rd_intra_blocks']+=int(m.group(1))
    for m in re.finditer(r'^PBC_PROFILE rd_intra_mode bsize=\d+ mode=(\d+) considered=(\d+) evaluated=(\d+) pruned=(\d+) invalid=(\d+) selected=(\d+)',text,re.M):
        mode,considered,evaluated,pruned,invalid,selected=map(int,m.groups())
        vals=by_mode.setdefault(str(mode),[0,0,0,0,0])
        vals[0]+=considered; vals[1]+=evaluated; vals[2]+=pruned; vals[3]+=invalid; vals[4]+=selected
        totals['candidate_visits']+=considered; totals['transform_search_entries']+=evaluated; totals['pruned_before_transform']+=pruned; totals['invalid_transform_results']+=invalid; totals['selected_mode_records']+=selected
    hm=re.search(r'^PBC_PROFILE hybrid_intra_rd=(\d+) hybrid_intra_nonrd=(\d+)',text,re.M)
    if hm: totals['hybrid_rd_calls'],totals['hybrid_nonrd_calls']=map(int,hm.groups())
    if not totals['rd_intra_blocks']: raise RuntimeError(f'no RD intra counters in {path}')
    return {'totals':totals,'by_mode':{k:{'considered':v[0],'transform_search_entries':v[1],'pruned':v[2],'invalid':v[3],'selected':v[4]} for k,v in by_mode.items()}}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--ctrl',type=Path,required=True); ap.add_argument('--exp',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); a=ap.parse_args()
    a.out.mkdir(parents=True,exist_ok=True); rows=[]
    for image,q,speed in CASES:
        case=f'{Path(image).stem}_q{q}_s{speed}'; src=ROOT/'hpt_data'/image
        case_dir=(a.out/case).resolve(); case_dir.mkdir(parents=True,exist_ok=True); data={}
        for label,exe in [('CTRL',a.ctrl),('EXP',a.exp)]:
            avif=case_dir/f'{label.lower()}.avif'; log=case_dir/f'{label.lower()}.log'; env=os.environ.copy()
            env['PATH']=str(exe.parent)+os.pathsep+r'C:\Strawberry\c\bin;C:\Windows\System32;C:\Windows'+os.pathsep+env.get('PATH','')
            cmd=[str(exe),'--codec','aom','--qcolor',str(q),'--qalpha','0','--speed',str(speed),'--jobs','1','--depth','8','--yuv','420','--range','full','--cicp','1/13/1','--ignore-exif','--ignore-xmp','--ignore-profile',str(src),str(avif)]
            with open(log,'wb') as f: p=subprocess.run(cmd,stdout=f,stderr=f,env=env,cwd=case_dir)
            if p.returncode or not avif.is_file(): raise RuntimeError(f'{label} profile encode failed {case}: {p.returncode}; {log}')
            parsed=parse(log); data[label]={'exit_code':p.returncode,'bytes':avif.stat().st_size,'counters':parsed}
        c=data['CTRL']['counters']['totals']; e=data['EXP']['counters']['totals']
        data['transform_search_change_percent']=(e['transform_search_entries']-c['transform_search_entries'])/c['transform_search_entries']*100
        data['candidate_visit_change_percent']=(e['candidate_visits']-c['candidate_visits'])/c['candidate_visits']*100
        rows.append({'image':image,'qcolor':q,'speed':speed,'results':data})
    out={'instrumentation':'Pass 35 PBC_PROFILE aggregate counters; timing data ignored','cases':rows}
    (a.out/'search_counts.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
    print(json.dumps([{'image':r['image'],'qcolor':r['qcolor'],'speed':r['speed'],'ctrl_tx':r['results']['CTRL']['counters']['totals']['transform_search_entries'],'exp_tx':r['results']['EXP']['counters']['totals']['transform_search_entries'],'tx_change_percent':r['results']['transform_search_change_percent'],'ctrl_blocks':r['results']['CTRL']['counters']['totals']['rd_intra_blocks'],'exp_blocks':r['results']['EXP']['counters']['totals']['rd_intra_blocks'],'hybrid_rd':r['results']['EXP']['counters']['totals']['hybrid_rd_calls']} for r in rows],indent=2))
if __name__=='__main__': main()

