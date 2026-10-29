import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import *
import subprocess
import time
root=Path(json.loads((PASS/'packaging/validation.json').read_text())['root'])
commands={'published_python':[sys.executable,str(PASS/'packaging/reference_cli.py')],'combined':[str(root/'combined/pbc/pbc.exe')],'decoder':[str(root/'decoder/pbc-decode/pbc-decode.exe')],'decoder_onefile':[str(root/'single/pbc-decode.exe')]}
rows=[]
for mode,command in commands.items():
    env=dict(os.environ,PATH=r'C:\Windows\System32',TEMP=str(root/'tmp'),TMP=str(root/'tmp'))
    p=subprocess.Popen(command+['benchdecode',str(PASS/'artifacts/corpus/cat.jpg.quality.pbc'),'--repeats','3'],cwd=root/'cwd',env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    peak=0
    while p.poll() is None:
        try:
            process=psutil.Process(p.pid);peak=max(peak,sum(c.memory_info().rss for c in [process]+process.children(recursive=True) if c.is_running()))
        except psutil.Error:pass
        time.sleep(.02)
    out,err=p.communicate();assert p.returncode==0,err
    rows.append(dict(mode=mode,sampled_peak_process_tree_rss=peak,pixel_sha256=json.loads(out)['pixel_sha256']))
assert len(set(r['pixel_sha256'] for r in rows))==1
save('packaging/memory.json',dict(rows=rows,poll_interval_seconds=.02,timings_discarded=True,workload='Cat Quality warm decode, initial plus3'))
print(rows)
