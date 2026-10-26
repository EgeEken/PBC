import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from research_pass5 import run_hf as api
api.HERE=HERE


def main():
    result=json.loads((HERE/'hf_comparison.json').read_text())
    assert len(result['rows'])==480 and result['deterministic']
    deployment=json.loads((HERE/'hf_ui_deployment.json').read_text())
    meta=api.metadata()
    assert meta['sha']==meta['runtime']['sha']==deployment['base']
    for endpoint in ('/api/benchmark/status','/api/quick_rd/status','/api/sweeps/run/status','/api/train/status'):
        assert not api.request(endpoint).get('running')
    data=urllib.request.urlopen(api.HOST+'/api/benchmark/download_db').read()
    path=HERE/'benchmark_before_ui_fix.db'
    with path.open('xb') as f:
        f.write(data)
    con=sqlite3.connect(path.as_uri()+'?mode=ro',uri=True)
    assert con.execute('PRAGMA quick_check').fetchone()[0]=='ok'
    runs=con.execute('SELECT run_id,COUNT(*) FROM results GROUP BY run_id').fetchall(); con.close()
    api.save('ui_fix_database_backup',dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),runs=runs,readable=True))
    subprocess.run(['git','push','origin',deployment['commit']+':refs/heads/main'],check=True)
    deadline=time.monotonic()+1200
    while time.monotonic()<deadline:
        meta=api.metadata()
        print(meta['runtime']['stage'],meta['runtime'].get('sha'),flush=True)
        if meta['runtime']['stage']=='RUNNING' and meta['runtime'].get('sha')==deployment['commit']:
            api.save('ui_fix_deployed',dict(sha=meta['sha'],runtime=meta['runtime'],health=api.request('/api/health')))
            return
        assert meta['runtime']['stage'] not in ('RUNTIME_ERROR','BUILD_ERROR')
        time.sleep(15)
    raise RuntimeError('UI deployment not ready')


if __name__=='__main__':
    main()
