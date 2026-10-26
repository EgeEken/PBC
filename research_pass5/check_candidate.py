import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from hf_runtime_job import validate


def main():
    old = ast.parse(subprocess.check_output(['git','show','origin/main:server.py'],text=True,encoding='utf-8'))
    current = ast.parse((ROOT/'server.py').read_text(encoding='utf-8'))
    names = ('compress','decode','stream_compress','benchmark_status','benchmark_stop')
    for name in names:
        before = next(n for n in old.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==name)
        after = next(n for n in current.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==name)
        assert ast.dump(before)==ast.dump(after),name
    from fastapi import HTTPException
    handler=next(n for n in current.body if isinstance(n,ast.AsyncFunctionDef) and n.name=='benchmark_start')
    handler.decorator_list=[]
    class Request:
        def __init__(self,body,token=''):
            self.body,self.headers=body,{'x-pbc-benchmark-token':token}
        async def json(self):
            return self.body
    namespace=dict(Request=Request,os=os,HTTPException=HTTPException,
        pbc3_benchmark=SimpleNamespace(start_research='research',start='ordinary',status='status'),
        _guarded_benchmark_start=lambda *args:args)
    exec(compile(ast.Module(body=[handler],type_ignores=[]),'<assembled-handler>','exec'),namespace)
    def call(request):
        coroutine=namespace['benchmark_start'](request)
        try:
            coroutine.send(None)
        except StopIteration as done:
            return done.value
    spec=dict(images=['googlewiki.png'],presets=['balanced'],variants=['baseline','all_mse','all'],repeats=2)
    invalid=[None,{},dict(spec,images=['../patch_policy.npz']),dict(spec,images=[{}]),dict(spec,images=[['cat.jpg']]),
        dict(spec,variants=['eval']),dict(spec,variants=['all']),dict(spec,repeats=True),dict(spec,repeats=4),
        dict(spec,code='print(1)'),dict(spec,images=['cat.jpg']*2)]
    for value in invalid:
        try:
            validate(value)
        except ValueError:
            pass
        else:
            raise AssertionError(value)
    previous=os.environ.pop('PBC_RESEARCH_BENCHMARK_TOKEN',None)
    checks=[]
    try:
        for token,expected in (('',404),('wrong',403),('test',None)):
            if expected!=404:
                os.environ['PBC_RESEARCH_BENCHMARK_TOKEN']='test'
            try:
                result=call(Request({'research':spec},token))
                assert expected is None and result==('benchmark','research','status',spec)
            except HTTPException as e:
                assert e.status_code==expected
            checks.append(expected or 200)
        assert call(Request({'codecs':['PBC3'],'n_trials':1}))==('benchmark','ordinary','status',['PBC3'],1)
    finally:
        os.environ.pop('PBC_RESEARCH_BENCHMARK_TOKEN',None)
        if previous is not None:
            os.environ['PBC_RESEARCH_BENCHMARK_TOKEN']=previous
    smoke=json.loads((HERE/'local_smoke.json').read_text(encoding='utf-8-sig'))
    assert smoke['exact'] and len(smoke['rows'])==6
    for name in ('hf_candidate_grid.py','hf_candidate_search.py'):
        assert (HERE/name).read_bytes()==(ROOT/'research_pass4'/name).read_bytes()
    manifest=json.loads((HERE/'hf_runtime_manifest.json').read_text())
    for name,value in manifest['source_lf'].items():
        assert hashlib.sha256((ROOT/name).read_text(encoding='utf-8').encode()).hexdigest()==value,name
    result=dict(public_handlers_ast_unchanged=list(names),ordinary_benchmark_arguments_unchanged=True,
        invalid_specs_rejected=len(invalid),auth_statuses=checks,local_smoke_rows=6,exact=True,
        candidate_modules_identical_to_pass4=True)
    (HERE/'candidate_checks.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result))


if __name__=='__main__':
    main()
