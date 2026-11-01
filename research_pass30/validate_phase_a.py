import hashlib,json
from pathlib import Path
import numpy as np
import phase_a_support as e

def main():
    summary=json.loads((e.OUT/'phase_a.json').read_text()); rows=[]
    for name,entry in summary['images'].items():
        item=e.load_case(name); src,base=item['src'],item['base']; residual=src.astype(np.int16)-base.astype(np.int16)
        cells,valid,_,_=e.cells(residual); dc=e.qdc(e.low_projection(cells)); fits=np.load(e.OUT/f'{name}.fit.npz')
        for method in ('global','pass29_distribution_structure','residual_shape'):
            for q in e.PATTERN_STEPS:
                key=f'{method}|{q}'; pats=fits[f'{key}.patterns']; coef=fits[f'{key}.coeff']; assign=fits[f'{key}.assign']
                if np.any(pats%q) or np.any(coef>6) or np.any(assign>3): raise AssertionError(f'invalid symbols {name} {key}')
                rec=e.reconstruct(base,dc,pats,coef,assign); again=e.reconstruct(base,dc,pats,coef,assign)
                if not np.array_equal(rec,again): raise AssertionError(f'non-deterministic reconstruction {name} {key}')
                mse=e.metric(src,rec)['mse']; saved=next(x for x in entry['methods'] if x['name']==method and x['q']==q)['mse']
                if abs(mse-saved)>1e-4: raise AssertionError(f'MSE mismatch {name} {key}: {mse} != {saved}')
                rows.append({'image':name,'variant':key,'mse':mse,'reconstruction_sha256':hashlib.sha256(rec.tobytes()).hexdigest()})
    result={'candidate_reconstructions':len(rows),'all_deterministic':True,'all_summary_mse_matched':True,'rows':rows}
    (e.OUT/'validation.json').write_text(json.dumps(result,indent=2)); print(json.dumps({k:v for k,v in result.items() if k!='rows'}))

if __name__=='__main__':main()
