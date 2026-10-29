import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import *
import subprocess
import time
pack=json.loads((PASS/'packaging/measurements.json').read_text())
perf=json.loads((PASS/'packaging/performance.json').read_text())
portable=Path(pack['portable_directory']);cache=Path(perf['cache_root'])
commands={'python':[sys.executable,str(PASS/'packaging/cli.py')],'onedir':[str(portable/'directory/PBC/pbc.exe')],'onefile':[str(portable/'single/pbc.exe')]}
rows=[]
for mode,command in commands.items():
    env=dict(os.environ,PATH=r'C:\Windows\System32',NUMBA_CACHE_DIR=str(cache/mode))
    p=subprocess.Popen(command+['benchmark',str(ROOT/'hpt_data/cat.jpg'),'--preset','quality','--repeats','1'],cwd=portable,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    peak=0;samples=0
    while p.poll() is None:
        try:
            process=psutil.Process(p.pid)
            peak=max(peak,sum(c.memory_info().rss for c in [process]+process.children(recursive=True) if c.is_running()))
            samples+=1
        except psutil.Error: pass
        time.sleep(.05)
    stdout,stderr=p.communicate();assert p.returncode==0,stderr
    rows.append(dict(mode=mode,sampled_peak_process_tree_rss_bytes=peak,samples=samples,stream_sha256=json.loads(stdout)['stream_sha256']))
assert len(set(r['stream_sha256'] for r in rows))==1
save('packaging/memory.json',dict(workload='cat.jpg Quality initial encode plus one warmed encode/decode',sample_interval_seconds=.05,latencies_discarded=True,rows=rows))
print(rows)
