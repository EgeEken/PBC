import json,time
from pathlib import Path
import numpy as np
import psutil
from threadpoolctl import threadpool_limits,threadpool_info
import phase_a_support as e

def samples(fn):
    fn(); out=[]
    for _ in range(3):
        t=time.perf_counter(); fn(); out.append(time.perf_counter()-t)
    return {'samples_s':out,'median_s':float(np.median(out))}

def main():
    limits=threadpool_limits(limits=1)
    result={'runtime':{'python':__import__('sys').version.split()[0],'numpy':np.__version__,'threadpools':threadpool_info(),'thread_limit':1}}
    for name in e.NAMES:
        item=e.load_case(name); src,base=item['src'],item['base']; r,valid,rows,cols=e.cells(src.astype(np.int16)-base.astype(np.int16))
        low=e.low_projection(r); dc=e.qdc(low); assign=e.structural_groups(low)
        fit=lambda:e.fit(low,dc,assign,4,8)
        pats,coef=fit()
        reconstruct=lambda:e.reconstruct(base,dc,pats,coef,assign)
        result[name]={'encoder_pattern_fit':samples(fit),'decoder_equivalent_reconstruction':samples(reconstruct),
                      'process_peak_working_set_bytes':psutil.Process().memory_info().peak_wset}
        print(name,result[name],flush=True)
    Path(e.OUT/'timing.json').write_text(json.dumps(result,indent=2))
    limits.restore_original_limits()

if __name__=='__main__':main()
