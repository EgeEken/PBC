import json
import os
from pathlib import Path
import subprocess
HERE=Path(__file__).resolve().parent
OUT=HERE/'artifacts'
env=os.environ.copy()
env['PATH']=r'C:\Windows\System32'
for key in ('PYTHONPATH','PYTHONHOME','VIRTUAL_ENV','NUMBA_CACHE_LOCATOR_CLASSES'):
    env.pop(key,None)
env['NUMBA_CACHE_DIR']=str(OUT/'ui-polish/frozen-cache')
exe=OUT/'probe-dist/PBC-probe/PBC-probe.exe'
for phase in ('first-use','cached'):
    with (OUT/('frozen-'+phase+'.log')).open('w',encoding='utf-8') as log:
        subprocess.run([str(exe),str(OUT/'frozen-fixtures.json'),str(OUT/('frozen-'+phase+'.json'))],
            cwd=os.environ['TEMP'],env=env,stdout=log,stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW,check=True,timeout=180)
    data=json.loads((OUT/('frozen-'+phase+'.json')).read_text())
    print(phase,dict(cases=len(data['checks']),valid=data['known_streams'],partials=data['partial_checks'],
        invalid=data['invalid_rejected'],cache_files=data['cached_files']),flush=True)
