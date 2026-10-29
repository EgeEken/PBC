import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import *
import subprocess
import statistics as stats
modes=('published','reference','native')
workers={mode:subprocess.Popen([sys.executable,str(PASS/'native/worker.py'),mode],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True) for mode in modes}
rows=[]
try:
    for name in ('cat.jpg','flower.jpg','seagull.jpg','googlewiki.png','monet.jpg','eiffel.png','small','rgba'):
        for preset in ('compression','balanced','quality','high_quality'):
            request=json.dumps(dict(image=name,preset=preset))+'\n'
            for w in workers.values():
                w.stdin.write(request);w.stdin.flush();json.loads(w.stdout.readline())
            for repeat in range(3):
                pair={}
                for mode in modes[repeat:]+modes[:repeat]:
                    w=workers[mode];w.stdin.write(request);w.stdin.flush();pair[mode]=json.loads(w.stdout.readline())
                for key in ('bytes','stream_sha256','pixel_sha256','internal_mse','rgb_mse'):
                    assert len(set(r[key] for r in pair.values()))==1,(name,preset,key)
                rows.append(dict(image=name,preset=preset,repeat=repeat,**pair))
            print(name,preset,flush=True)
finally:
    for w in workers.values(): w.stdin.close();w.wait(timeout=30)
configs={}
for row in rows:
    for mode in modes:
        config=row[mode].pop('config');key=mode+'/'+row['preset']
        if key in configs: assert configs[key]==config
        configs[key]=config
save('native/benchmark.json',dict(rows=rows,configs=configs))
summary=[]
for image,preset in sorted(set((r['image'],r['preset']) for r in rows)):
    subset=[r for r in rows if (r['image'],r['preset'])==(image,preset)]
    values={m:stats.median(r[m]['encode_seconds'] for r in subset) for m in modes}
    summary.append(dict(image=image,preset=preset,median_seconds=values,native_gain_percent=100*(1-values['native']/values['reference']),candidate_change_percent=100*(values['reference']/values['published']-1),variation={m:dict(mean=stats.mean(r[m]['encode_seconds'] for r in subset),cv=stats.stdev(r[m]['encode_seconds'] for r in subset)/stats.mean(r[m]['encode_seconds'] for r in subset),minimum=min(r[m]['encode_seconds'] for r in subset),maximum=max(r[m]['encode_seconds'] for r in subset)) for m in modes}))
aggregate={m:stats.mean(r['median_seconds'][m] for r in summary) for m in modes}
gain=100*(1-aggregate['native']/aggregate['reference']);worst=min(r['native_gain_percent'] for r in summary)
save('native/summary.json',dict(cases=summary,aggregate_mean_medians=aggregate,gain_percent=gain,worst_case_gain_percent=worst,gate_numeric_pass=gain>=3 and worst>=-5,streams_exact=True))
print('aggregate',gain,'worst',worst)
