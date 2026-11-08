import json,statistics
from pathlib import Path
d=json.loads(Path('research_pass37/outputs/benchmark_final/benchmark.json').read_text())
rows=[r for r in d['rows'] if r['speed']==6]
for r in rows:
    a,b=r['configs']['REF'],r['configs']['EXP']
    pair=[round((x-y)/x*100,2) for x,y in zip(a['timed_seconds'],b['timed_seconds'])]
    print(r['image'],f"q{r['qcolor']}",f"median {a['median_seconds']:.4f}->{b['median_seconds']:.4f}s",f"reduction {r['time_reduction_percent']:.2f}%",f"bytes {a['bytes']}->{b['bytes']} ({r['byte_change_percent']:.2f}%)",f"MSE {a['rgb_mse']:.4f}->{b['rgb_mse']:.4f} ({r['mse_change_percent']:.2f}%)",f"SD {a['stdev_seconds']:.4f}/{b['stdev_seconds']:.4f}",f"paired% {pair}")
print('mean/median time reduction:',statistics.mean(r['time_reduction_percent'] for r in rows),statistics.median(r['time_reduction_percent'] for r in rows))
print('faster points:',sum(r['time_reduction_percent']>0 for r in rows),'/',len(rows))
print('mean byte change %, mean MSE change %:',statistics.mean(r['byte_change_percent'] for r in rows),statistics.mean(r['mse_change_percent'] for r in rows))
print('memory REF/EXP sampled RSS MB:')
for r in rows:
    key=f"{Path(r['image']).stem}_q{r['qcolor']}_s{r['speed']}"
    m=d['memory_probes'][key]
    print(key,round(m['REF']['peak_sampled_rss_bytes']/1e6,1),round(m['EXP']['peak_sampled_rss_bytes']/1e6,1))
