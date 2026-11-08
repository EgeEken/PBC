import json,subprocess
from pathlib import Path
import hashlib
ROOT=Path(__file__).resolve().parents[2]
bench=json.loads((ROOT/'research_pass37/outputs/benchmark_final/benchmark.json').read_text())
cases=[]
for r in bench['rows']:
    label=f"{Path(r['image']).stem}_q{r['qcolor']}_s{r['speed']}"
    p=ROOT/'research_pass37/outputs/benchmark_final'/label/'exp_r1.avif'
    cases.append((label,p,r['width'],r['height']))
syn=json.loads((ROOT/'research_pass37/outputs/synthetic_smoke/synthetic_results.json').read_text())
for name in {r['case'] for r in syn['cases']}:
    match=next(r for r in syn['cases'] if r['case']==name)
    cases.append((name,ROOT/'research_pass37/outputs/synthetic_smoke'/f'{name}_r1.avif',match['width'],match['height']))
rows=[]
for name,p,w,h in cases:
    cmd=['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height','-of','csv=s=x:p=0',str(p)]
    probe=subprocess.run(cmd,capture_output=True,text=True)
    decode=subprocess.run(['ffmpeg','-v','error','-i',str(p),'-frames:v','1','-f','null','-'],capture_output=True,text=True)
    dims=probe.stdout.strip()
    if probe.returncode or decode.returncode or dims!=f'{w}x{h}':
        raise RuntimeError(f'FFmpeg check failed {name} dims={dims} probe={probe.returncode} decode={decode.returncode} err={decode.stderr}')
    rows.append({'case':name,'avif_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'ffprobe_dimensions':dims,'decode_exit':decode.returncode})
result={'ffmpeg_version':subprocess.check_output(['ffmpeg','-version'],text=True).splitlines()[0],'ffprobe_version':subprocess.check_output(['ffprobe','-version'],text=True).splitlines()[0],'rows':rows,'passed':True}
(ROOT/'research_pass37/outputs/ffmpeg_decode.json').write_text(json.dumps(result,indent=2)+'\n')
print(result['ffmpeg_version'], 'passed',len(rows),'files')
