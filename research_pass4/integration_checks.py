import ast
import os
import sys
from types import SimpleNamespace

from support import ROOT, OUT, save
from hf_runtime_job import validate


def main():
    sys.path.insert(0,str(ROOT))
    valid=dict(images=['googlewiki.png'],presets=['balanced'],variants=['baseline','all'],repeats=2)
    invalid=[None,{},dict(valid,images=['../patch_policy.npz']),dict(valid,presets=['manual']),
        dict(valid,variants=['eval']),dict(valid,variants=['all']),dict(valid,repeats=True),
        dict(valid,repeats=4),dict(valid,code='print(1)'),dict(valid,images=['googlewiki.png']*2),
        dict(valid,presets=['compression','balanced','quality'])]
    for spec in invalid:
        try:
            validate(spec)
        except (ValueError,TypeError):
            pass
        else:
            raise AssertionError(spec)
    assert validate(valid)==valid
    from fastapi import HTTPException
    tree=ast.parse((OUT/'integration_candidate'/'server.py').read_text(encoding='utf-8'))
    handler=next(n for n in tree.body if isinstance(n,ast.AsyncFunctionDef) and n.name=='benchmark_start')
    handler.decorator_list=[]
    class Request:
        def __init__(self,body,token=''):
            self.body,self.headers=body,{'x-pbc-benchmark-token':token}
        async def json(self):
            return self.body
    namespace=dict(Request=Request,os=os,HTTPException=HTTPException,
        pbc3_benchmark=SimpleNamespace(start_research='research_start',start='normal_start',status='status'),
        _guarded_benchmark_start=lambda *args:args)
    exec(compile(ast.Module(body=[handler],type_ignores=[]),'<unapplied-api-handler>','exec'),namespace)
    def call(body,token=''):
        coro=namespace['benchmark_start'](Request(body,token))
        try:
            coro.send(None)
        except StopIteration as done:
            return done.value
    previous=os.environ.pop('PBC_RESEARCH_BENCHMARK_TOKEN',None)
    results=[]
    try:
        for token,expected in (('',404),('wrong',403),('local-test',None)):
            if expected!=404:
                os.environ['PBC_RESEARCH_BENCHMARK_TOKEN']='local-test'
            try:
                result=call({'research':valid},token)
                assert expected is None and result[1]=='research_start' and result[-1]==valid
                results.append(dict(case='authorized',status='guarded research start'))
            except HTTPException as error:
                assert error.status_code==expected
                results.append(dict(case='disabled' if expected==404 else 'unauthorized',status=expected))
        try:
            call({'research':dict(valid,images=['invalid'])},'local-test')
        except HTTPException as error:
            assert error.status_code==400
            results.append(dict(case='invalid allowlist',status=400))
        else:
            raise AssertionError('invalid accepted')
        assert call({'codecs':['PBC3'],'n_trials':1})==('benchmark','normal_start','status',['PBC3'],1)
        results.append(dict(case='ordinary route',status='original arguments retained'))
    finally:
        os.environ.pop('PBC_RESEARCH_BENCHMARK_TOKEN',None)
        if previous is not None:
            os.environ['PBC_RESEARCH_BENCHMARK_TOKEN']=previous
    save('integration_checks',dict(invalid_specs=len(invalid),valid_spec=True,handler_cases=results,
        scope='Extracted route handler and allowlist validation only; no server startup, HTTP, job execution or deployment.'))


if __name__=='__main__':
    main()
