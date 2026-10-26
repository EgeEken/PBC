import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from research_pass5 import run_hf as api
from research_pass8.run_hf import token,job
from research_pass9.hf_release_job import INPUTS,HOLDOUT,PRESETS,VARIANTS,EQUIVALENCE
api.HERE=HERE
SMOKE=('cat.jpg','eiffel.png','googlewiki.png')
TIMING=('cat.jpg','eiffel.png','flower.jpg','googlewiki.png','mario64.png','monet.jpg','ratedistortion.png','seagull.jpg')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('operation',choices=('deploy','smoke','equivalence','benchmark'))
    operation=parser.parse_args().operation
    deployment=json.loads((HERE/'hf_deployment.json').read_text())
    meta=api.metadata()
    assert meta['runtime']['stage']=='RUNNING' and meta['sha']==meta['runtime']['sha']
    assert meta['runtime']['hardware']['current']=='cpu-basic'
    for endpoint in ('/api/benchmark/status','/api/quick_rd/status','/api/sweeps/run/status','/api/train/status'):
        assert not api.request(endpoint).get('running')
    if operation=='deploy':
        assert meta['sha']==deployment['base']
        data=urllib.request.urlopen(api.HOST+'/api/benchmark/download_db',timeout=60).read()
        path=HERE/'benchmark_before.db'
        with path.open('xb') as stream:
            stream.write(data)
        con=sqlite3.connect(path.as_uri()+'?mode=ro',uri=True)
        assert con.execute('PRAGMA quick_check').fetchone()[0]=='ok'
        runs=con.execute('SELECT run_id,COUNT(*) FROM results GROUP BY run_id').fetchall()
        con.close()
        api.save('database_backup',dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),runs=runs,quick_check='ok'))
        api.save('hf_before',dict(sha=meta['sha'],runtime=meta['runtime']))
        subprocess.run(['git','push','origin',deployment['commit']+':refs/heads/main'],check=True)
        deadline=time.monotonic()+1200
        while time.monotonic()<deadline:
            meta=api.metadata()
            print(meta['runtime']['stage'],meta['runtime'].get('sha'),flush=True)
            if meta['runtime']['stage']=='RUNNING' and meta['runtime'].get('sha')==deployment['commit']:
                api.save('hf_deployed',dict(sha=meta['sha'],runtime=meta['runtime'],health=api.request('/api/health')))
                return
            assert meta['runtime']['stage'] not in ('RUNTIME_ERROR','BUILD_ERROR')
            time.sleep(15)
        raise RuntimeError('Deployment not ready')
    assert meta['sha']==deployment['commit']
    key=token()
    if operation=='smoke':
        try:
            api.request('/api/benchmark/start',{'research':{} })
        except urllib.error.HTTPError as error:
            assert error.code==403
        else:
            raise AssertionError('Unauthenticated research accepted')
        try:
            api.request('/api/benchmark/start',{'research':{}},key)
        except urllib.error.HTTPError as error:
            assert error.code==400
        else:
            raise AssertionError('Invalid authenticated specification accepted')
        images=list(SMOKE); variants=list(VARIANTS); repeats=1
    elif operation=='equivalence':
        smoke=json.loads((HERE/'hf_smoke.json').read_text())
        assert len(smoke['equivalent'])==12
        images=[x for x in INPUTS+HOLDOUT if x not in SMOKE]; variants=list(EQUIVALENCE); repeats=1
    else:
        gate=json.loads((HERE/'hf_equivalence.json').read_text())
        assert len(gate['equivalent'])==88
        images=list(TIMING); variants=list(VARIANTS); repeats=3
    result=job('hf_'+('comparison' if operation=='benchmark' else operation),dict(images=images,presets=list(PRESETS),variants=variants,repeats=repeats),key)
    after=api.metadata()
    assert after['sha']==after['runtime']['sha']==deployment['commit']
    api.save('hf_after_'+operation,dict(sha=after['sha'],runtime=after['runtime']))
    print(len(result['rows']),'measured calls;',len(result['equivalent']),'equivalent release/RNG pairs',flush=True)


if __name__=='__main__':
    main()
