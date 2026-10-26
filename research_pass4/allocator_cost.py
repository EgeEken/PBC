from contextlib import contextmanager
import inspect

import numpy as np


@contextmanager
def action_cache(mode,counters):
    import learned_filler as filler
    import pbc3_ops as ops
    projection=mode in ('projection','both')
    scoring=mode in ('before','both')
    projected={}
    before_sse=None
    original_project,original_score=ops.project_residual_to_grid,filler._score_patch
    def project(values,cell_size,projection_mode='bicubic'):
        key=int(cell_size),projection_mode
        if key not in projected:
            projected[key]=original_project(values,cell_size,projection_mode)
        else:
            counters['avoided_projection_calls']=counters.get('avoided_projection_calls',0)+1
        return projected[key]
    def before(target,canvas,box):
        nonlocal before_sse
        if before_sse is None:
            c,x,y,w,h=box
            error=target[y:y+h,x:x+w,c]-np.clip(canvas[y:y+h,x:x+w,c],0,255)
            before_sse=float(np.sum(error.astype(np.int64)**2))
        else:
            counters['avoided_before_sse_reductions']=counters.get('avoided_before_sse_reductions',0)+1
        return before_sse
    try:
        if projection:
            ops.project_residual_to_grid=project
        if scoring:
            filler._score_patch=_cached_score(before)
        yield
    finally:
        ops.project_residual_to_grid,filler._score_patch=original_project,original_score


_score_code=None
_score_globals=None


def prepare_score():
    global _score_code,_score_globals
    import learned_filler as filler
    code=inspect.getsource(filler._score_patch)
    needle='        before = target_box - np.clip(canvas_box, 0, 255)\n'
    assert code.count(needle)==1
    code=code.replace(needle,'')
    needle='        before_sse = float(np.sum(before.astype(np.int64) ** 2))'
    assert code.count(needle)==1
    code=code.replace(needle,'        before_sse = cached_before(target, canvas, box)')
    namespace=dict(vars(filler),cached_before=None)
    exec(compile(code,__file__,'exec'),namespace)
    _score_code=namespace['_score_patch'].__code__
    _score_globals=namespace


def _cached_score(before):
    from types import FunctionType
    namespace=dict(_score_globals,cached_before=before)
    return FunctionType(_score_code,namespace,'_score_patch', (None,))
