import json
import os
import pathlib
import subprocess
import sys
import time

import psutil
ROOT=pathlib.Path(__file__).resolve().parents[1]
OUT=ROOT/'research_pass42'/'outputs'
script="""import hashlib,json,pathlib,sys,time;sys.path.insert(0,r'%s');from pbc4_codec import encode; p=pathlib.Path(sys.argv[1]);t=time.perf_counter();r=encode(p,'D','low');elapsed=time.perf_counter()-t;f=pathlib.Path(sys.argv[2]);f.write_bytes(r['data']);print(json.dumps({'encode_seconds':elapsed,'bytes':len(r['data']),'mse':r['mse'],'modes':r['mode_counts'],'init':[r['init_width'],r['init_height']],'sha256':hashlib.sha256(r['data']).hexdigest()}))""" % (ROOT/'research_pass42')
env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
proc=subprocess.Popen([sys.executable,'-c',script,str(ROOT/'hpt_data'/'coast.jpg'),str(OUT/'large_coast_screen.pbc')],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,env=env,cwd=ROOT)
handle=psutil.Process(proc.pid);peak=0
while proc.poll() is None:
    try: peak=max(peak,handle.memory_info().rss)
    except psutil.Error: pass
    time.sleep(0.05)
out,err=proc.communicate()
if proc.returncode: raise RuntimeError(err or out)
result=json.loads(out);result['peak_working_set_bytes']=peak
(OUT/'large_screen.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
