import json
from pathlib import Path
import numpy as np
import pbc31_codec as codec
from run_experiment import OUT,edge_masks,load_image,rgb_metrics

LAMBDAS=(0.0,20.0,100.0)

def main():
    refs={x['image']:x for x in json.loads((Path(__file__).resolve().parents[1]/'research_pass29/outputs/results.json').read_text())['images']}
    source,base,base_stream,source_hash,base_hash=load_image('googlewiki.png',refs)
    edge,smooth=edge_masks(source); rows=[]
    directory=OUT/'calibration'; directory.mkdir(parents=True,exist_ok=True)
    for lam in LAMBDAS:
        e=codec.encode(source,base,'D',lam); path=directory/f'googlewiki_D_{lam:g}.p31r'; path.write_bytes(e['data'])
        data=path.read_bytes(); rec,modes,params=codec.decode(data,base)
        if not np.array_equal(rec,e['reconstruction']) or not np.array_equal(modes,e['modes']) or params!=e['params']:
            raise AssertionError('calibration stream round-trip mismatch')
        rows.append({'lambda':lam,'enhancement_bytes':len(data),'mode_counts':np.bincount(modes,minlength=5).tolist(),
                     'metrics':rgb_metrics(source,rec,edge,smooth),'estimated_bits':e['estimated_bits'],
                     'encode_seconds':e['encode_seconds'],'stream_sha256':codec.sha(data)})
    sizes=[r['enhancement_bytes'] for r in rows]
    distinct=len(set(sizes))
    informative=distinct==3 and max(sizes)>1.05*min(sizes)
    result={'image':'googlewiki.png','source_sha256':source_hash,'base_sha256':base_hash,'base_bytes':len(base_stream),
        'initial_lambda_candidates':list(LAMBDAS),'fixed_before_four_image_screen':list(LAMBDAS),
        'informative_by_predefined_rule':informative,'one_correction_permitted':not informative,'results':rows}
    (OUT/'calibration.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))

if __name__=='__main__':main()
