import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import urllib.error
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from research_pass5 import run_hf as api


def main():
    latest=api.request('/api/benchmark/results')
    name=latest['selected_run']
    exact=api.request('/api/benchmark/results?run_id='+name)
    history=api.request('/api/benchmark/results?run_id=all')
    limited=api.request('/api/benchmark/results?run_id='+name+'&mp_min=10')
    missing=api.request('/api/benchmark/results?run_id=not-a-run')
    assert latest['rows']==exact['rows'] and {r['run_id'] for r in exact['rows']}=={name}
    assert {r['codec'] for r in exact['rows']}=={'PBC3'}
    assert len(history['runs'])==3 and history['count']==6200
    assert all(r['mp']>=10 and r['run_id']==name for r in limited['rows'])
    assert limited['count']==sum(r['mp']>=10 for r in exact['rows'])
    assert missing['count']==0 and 'missing' in missing['selection_status']
    script=urllib.request.urlopen(api.HOST+'/benchmark.js').read()
    assert script.decode().replace('\r\n','\n')==(ROOT/'static/benchmark.js').read_text(encoding='utf-8')
    response=urllib.request.urlopen(api.HOST+'/api/benchmark/download_db').read()
    path=HERE/'benchmark_after.db'; path.write_bytes(response)
    con=sqlite3.connect(path.as_uri()+'?mode=ro',uri=True)
    assert con.execute('PRAGMA quick_check').fetchone()[0]=='ok'
    counts=con.execute('SELECT run_id,COUNT(*) FROM results GROUP BY run_id').fetchall(); con.close()
    try:
        api.request('/api/benchmark/start',{'research':{}})
    except urllib.error.HTTPError as error:
        assert error.code==403
    else:
        raise AssertionError('Unauthenticated research request accepted')
    result=dict(latest_run=name,latest_count=latest['count'],latest_status=latest['selection_status'],
        latest_timestamp=next(r['first_ts'] for r in latest['runs'] if r['run_id']==name),
        exact_matches_latest=True,pbc_only=True,all_history_count=history['count'],all_history_runs=history['runs'],
        mp_min_10_count=limited['count'],missing_labeled=True,served_javascript_exact=True,
        unauthenticated_research_status=403,database_download_readable=True,
        database_after_sha256=hashlib.sha256(response).hexdigest(),database_after_bytes=len(response),database_after_runs=counts,
        restored_backup=False,health=api.request('/api/health'))
    (HERE/'ui_api_checks.json').write_text(json.dumps(result,indent=2)+'\n',newline='\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
