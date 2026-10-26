import ast
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import pbc3_benchmark as benchmark


def main():
    (HERE/'hf_runs').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=HERE/'hf_runs') as directory:
        path=Path(directory)/'fixture.db'
        shutil.copyfile(ROOT/'research_pass9/benchmark_after_deploy.db',path)
        benchmark.DB_PATH=str(path)
        con=benchmark._connect()
        assert con.execute('SELECT COUNT(*) FROM results').fetchone()[0]==6200
        assert con.execute('SELECT COUNT(*) FROM results WHERE patches IS NOT NULL OR stream_sha256 IS NOT NULL').fetchone()[0]==0
        meta=dict(n_trials=3,images=['a','b'],codecs=['PBC3'],baseline_id='pbc31_release_rng_baseline',label='PBC 3.1 development — release-compatible RNG',rng_mode='release')
        con.execute('INSERT INTO run_metadata VALUES (?,?)',('promoted',json.dumps(meta)))
        for trial in (1,2):
            for preset in benchmark.PBC3_PRESETS:
                for image in meta['images']:
                    benchmark._insert(con,(9e9,'promoted',trial,'PBC3',preset,None,image,1.,.5,.05,.3,10.,30.,19,'frozen-hash'))
        con.commit()
        assert benchmark.results(run_id='promoted')['selection_status']=='incomplete coverage'
        for preset in benchmark.PBC3_PRESETS:
            for image in meta['images']:
                benchmark._insert(con,(9e9,'promoted',3,'PBC3',preset,None,image,1.,.5,.05,.3,10.,30.,19,'frozen-hash'))
        con.commit(); con.close()
        before=hashlib.sha256(path.read_bytes()).hexdigest()
        result=benchmark.results()
        assert result['selected_run']=='promoted' and result['count']==24
        run=next(x for x in result['runs'] if x['run_id']=='promoted')
        assert run['status']=='complete' and run['metadata']==meta
        assert all(x['metadata'] is None for x in result['runs'] if x['run_id']!='promoted')
        assert benchmark.results(mp_min=2)['count']==0
        assert benchmark.results(run_id='all')['count']==6224
        assert hashlib.sha256(path.read_bytes()).hexdigest()==before
    entry_points={}
    for name in ('server.py','pbc3_benchmark.py','pbc3_quick_rd.py','pbc3_sweep.py','pbc3_rl_a20.py'):
        tree=ast.parse((ROOT/name).read_text(encoding='utf-8'))
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
            and isinstance(n.func.value,ast.Name) and n.func.value.id=='PBC3' and n.func.attr in ('compress','compress_stream')]
        assert calls and all('rng_mode' not in [k.arg for k in n.keywords] for n in calls)
        entry_points[name]=[dict(line=n.lineno,call=n.func.attr,inherits_default=True) for n in calls]
    source=(ROOT/'server.py').read_text(encoding='utf-8')
    fields=source[source.index('PBC3_FIELDS ='):source.index('@app.get("/api/multlist")')]
    assert 'rng_mode' not in fields and 'compute_final_mse' not in fields
    outcome=dict(historical_rows_preserved=6200,historical_provenance_not_relabelled=True,nullable_schema_upgrade=True,
        new_coverage_verified=True,incomplete_trailing_trial_rejected=True,read_queries_do_not_write=True,mp_filter_preserved=True,
        public_form_cannot_select_rng_or_disable_mse=True,entry_points=entry_points)
    (HERE/'benchmark_checks.json').write_text(json.dumps(outcome,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(outcome,indent=2))


if __name__=='__main__':
    main()
