import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pbc3_benchmark as benchmark


def main():
    with tempfile.TemporaryDirectory(dir=ROOT/'research_pass7') as directory:
        benchmark.DB_PATH = str(Path(directory)/'fixture.db')
        con = benchmark._connect()
        for name, ts, timing, trials in [('old',1,1.,1),('new',2,9.,2)]:
            for trial in range(1,trials+1):
                for preset in benchmark.PBC3_PRESETS:
                    for image,mp in [('a',1),('b',2)]:
                        benchmark._insert(con,(ts,name,trial,'PBC3',preset,None,image,mp,timing,timing/10,.5,10.,30.))
        benchmark._insert(con,(3,'partial',1,'PBC3','quality',None,'a',1.,99.,99.,.5,10.,30.))
        benchmark._insert(con,(.5,'other',1,'PNG','lossless',0,'a',1.,100.,100.,4.,0.,99.))
        con.commit()
        con.close()
        before = hashlib.sha256(Path(benchmark.DB_PATH).read_bytes()).hexdigest()
        for name,timing,count in [('old',1.,8),('new',9.,16)]:
            result=benchmark.results(run_id=name)
            assert len(result['rows'])==count
            assert {r['run_id'] for r in result['rows']}=={name}
            assert all(p['encode_seconds']==timing and p['codec']=='PBC3' for p in result['points'])
        assert benchmark.results()['selected_run']=='new'
        assert benchmark.results(mp_min=2,run_id='new')['count']==8
        assert benchmark.results(mp_max=1,run_id='new')['count']==8
        assert benchmark.results(mp_min=3,run_id='new')['count']==0
        assert benchmark.results(run_id='partial')['selection_status']=='incomplete coverage'
        assert benchmark.results(run_id='absent')['count']==0
        assert 'missing' in benchmark.results(run_id='absent')['selection_status']
        all_result=benchmark.results(run_id='all')
        pbc=[p for p in all_result['points'] if p['codec']=='PBC3']
        weighted=sum(p['encode_seconds']*p['count'] for p in pbc)/sum(p['count'] for p in pbc)
        assert abs(weighted-(8+16*9+99)/25)<1e-12
        benchmark._STATE.update(running=True,run_id='new')
        assert benchmark.results()['selected_run']=='old'
        benchmark._STATE.update(running=False)
        assert hashlib.sha256(Path(benchmark.DB_PATH).read_bytes()).hexdigest()==before
    benchmark.DB_PATH=str(ROOT/'research_pass7/database_before.db')
    actual=benchmark.results()
    outcome=dict(fixture_passed=True,database_unchanged=True,weighted_seconds=weighted,
        live_copy_runs=actual['runs'],latest_selected=actual['selected_run'],latest_samples=actual['count'])
    (ROOT/'research_pass7/run_selection_checks.json').write_text(json.dumps(outcome,indent=2)+'\n')
    print(json.dumps(outcome,indent=2))


if __name__=='__main__':
    main()
