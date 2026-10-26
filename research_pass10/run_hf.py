import argparse
import datetime
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
from research_pass8.run_hf import token,job
api.HERE=HERE


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('operation',choices=('deploy','research-smoke','benchmark'))
    operation=parser.parse_args().operation
    deployment=json.loads((HERE/'hf_deployment.json').read_text())
    meta=api.metadata()
    assert meta['sha']==meta['runtime']['sha'] and meta['runtime']['stage']=='RUNNING'
    assert meta['runtime']['hardware']['current']=='cpu-basic' and api.request('/api/health')['ok']
    for path in ('/api/benchmark/status','/api/quick_rd/status','/api/sweeps/run/status','/api/train/status'):
        assert not api.request(path).get('running'),path
    if operation=='deploy':
        assert meta['sha']==deployment['base']
        data=urllib.request.urlopen(api.HOST+'/api/benchmark/download_db',timeout=60).read()
        path=HERE/'benchmark_before.db'
        with path.open('xb') as stream:
            stream.write(data)
        con=sqlite3.connect(path.as_uri()+'?mode=ro',uri=True)
        assert con.execute('PRAGMA quick_check').fetchone()[0]=='ok'
        runs=con.execute('SELECT run_id,COUNT(*) FROM results GROUP BY run_id').fetchall(); con.close()
        api.save('database_backup',dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),rows=sum(x[1] for x in runs),runs=runs,quick_check='ok'))
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
        raise RuntimeError('Deployment did not become ready')
    assert meta['sha']==deployment['commit']
    if operation=='research-smoke':
        result=job('hf_variants',dict(images=['googlewiki.png'],presets=['balanced'],variants=['release','dev_release_rng','dev_current_rng'],repeats=1),token())
        rows={r['source']:r for r in result['rows']}
        assert rows['release']['stream_sha256']==rows['dev_release_rng']['stream_sha256']!=rows['dev_current_rng']['stream_sha256']
        api.save('variant_checks',dict(equivalent=True,distinct_custom=True,rows=[{k:v for k,v in r.items() if k not in ('trajectory','encode_seconds','decode_seconds')} for r in result['rows']],workers=result['metadata']['workers']))
        return
    assert json.loads((HERE/'public_checks.json').read_text())['passed']
    assert json.loads((HERE/'variant_checks.json').read_text())['distinct_custom']
    response=api.request('/api/benchmark/start',dict(codecs=['PBC3'],n_trials=3))
    assert response.get('started') and response['codecs']==['PBC3'] and response['n_trials']==3
    api.save('benchmark_start',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),response=response,hf_revision=meta['sha']))
    deadline=time.monotonic()+1200
    while time.monotonic()<deadline:
        time.sleep(15)
        state=api.request('/api/benchmark/status')
        print(state['done'],'/',state['total'],'running' if state['running'] else 'finished',flush=True)
        if not state['running']:
            assert not state['error'],state['error']
            assert state['done']==state['total']==240
            assert not any('failed on' in line for line in state['log'])
            result=api.request('/api/benchmark/results?run_id='+state['run_id'])
            assert len(result['rows'])==240 and {r['codec'] for r in result['rows']}=={'PBC3'}
            api.save('public_benchmark',result)
            return
    raise RuntimeError('PBC-only benchmark exceeded its bound')


if __name__=='__main__':
    main()
