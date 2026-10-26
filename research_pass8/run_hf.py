import argparse
import datetime
import hashlib
import hmac
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
from research_pass8.hf_release_job import INPUTS,PRESETS
api.HERE=HERE


def token():
    response=subprocess.run(['git','credential','fill'],input='protocol=https\nhost=huggingface.co\n\n',capture_output=True,text=True,check=True)
    fields=dict(line.split('=',1) for line in response.stdout.splitlines() if '=' in line)
    return hmac.new(fields['password'].encode(),b'PBC scoped benchmark token v1 / EgeEken/PBC / eee83b8f10f6f3404bca1c2c178e27fc4e8a1aa0',hashlib.sha256).hexdigest()


def job(name,spec,key):
    assert not api.request('/api/benchmark/status')['running']
    response=api.request('/api/benchmark/start',{'research':spec},key)
    assert response.get('started'),response
    api.save(name+'_start',dict(spec=spec,response=response,utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    deadline=time.monotonic()+1850
    while time.monotonic()<deadline:
        time.sleep(15)
        state=api.request('/api/benchmark/status')
        print(name,'running' if state['running'] else 'finished',flush=True)
        if not state['running']:
            assert not state['error'],state['error']
            result=state['research_result']
            assert result['deterministic'] and result['metadata']['spec']==spec
            api.save(name,result)
            return result
    raise RuntimeError('Bounded benchmark did not complete')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('operation',choices=('deploy','benchmark'))
    operation=parser.parse_args().operation
    deployment=json.loads((HERE/'hf_deployment.json').read_text())
    key=token()
    states={p:api.request(p) for p in ('/api/health','/api/benchmark/status','/api/quick_rd/status','/api/sweeps/run/status','/api/train/status')}
    assert not any(s.get('running') for s in states.values())
    if operation=='deploy':
        meta=api.metadata()
        assert meta['sha']==meta['runtime']['sha']==deployment['base']
        data=urllib.request.urlopen(api.HOST+'/api/benchmark/download_db',timeout=60).read()
        path=HERE/'benchmark_before.db'
        with path.open('xb') as f:
            f.write(data)
        con=sqlite3.connect(path.as_uri()+'?mode=ro',uri=True)
        assert con.execute('PRAGMA quick_check').fetchone()[0]=='ok'
        runs=con.execute('SELECT run_id,COUNT(*) FROM results GROUP BY run_id').fetchall()
        con.close()
        api.save('database_backup',dict(path=str(path),bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),quick_check='ok',
            rows=sum(r[1] for r in runs),runs=runs,utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
        api.save('hf_before',meta)
        try:
            api.request('/api/benchmark/start',{'research':{}},key)
        except urllib.error.HTTPError as error:
            assert error.code==400
        else:
            raise AssertionError('Invalid private spec was accepted')
        subprocess.run(['git','push','origin',deployment['commit']+':refs/heads/main'],check=True)
        observations=[]
        deadline=time.monotonic()+1200
        while time.monotonic()<deadline:
            meta=api.metadata()
            observations.append(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),sha=meta['sha'],runtime=meta['runtime']))
            api.save('deployment_observations',observations)
            print(meta['runtime']['stage'],meta['runtime'].get('sha'),flush=True)
            if meta['runtime']['stage']=='RUNNING' and meta['runtime'].get('sha')==deployment['commit']:
                break
            assert meta['runtime']['stage'] not in ('RUNTIME_ERROR','BUILD_ERROR')
            time.sleep(15)
        else:
            raise RuntimeError('Deployment not ready')
        api.save('hf_deployed',meta)
        api.save('health',api.request('/api/health'))
        api.public_smoke(None)
        job('hf_smoke',dict(images=['googlewiki.png'],presets=list(PRESETS),variants=['release','development'],repeats=1),key)
    else:
        meta=api.metadata()
        assert meta['sha']==meta['runtime']['sha']==deployment['commit'] and meta['runtime']['hardware']['current']=='cpu-basic'
        result=job('hf_comparison',dict(images=list(INPUTS),presets=list(PRESETS),variants=['release','development'],repeats=3),key)
        assert len(result['rows'])==480
        api.save('hf_after',api.metadata())


if __name__=='__main__':
    main()
