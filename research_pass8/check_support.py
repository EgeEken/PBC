import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(HERE))
from hf_release_job import validate,run


def main():
    spec=dict(images=['googlewiki.png'],presets=['compression','balanced','quality','high_quality'],variants=['release','development'],repeats=1)
    assert validate(spec)==spec
    bad=[dict(spec,images=['../secret']),dict(spec,images=['cat.jpg','cat.jpg']),dict(spec,presets=['custom']),
        dict(spec,variants=['baseline','all']),dict(spec,variants=['release']),dict(spec,repeats=4),dict(spec,repeats=True),
        dict(spec,code='x'),dict(spec,path='x'),dict(spec,images=[]),None,[]]
    for value in bad:
        try:
            validate(value)
        except (ValueError,TypeError):
            pass
        else:
            raise AssertionError(value)
    result=run(spec)
    assert len(result['rows'])==8 and result['deterministic']
    for source,worker in result['metadata']['workers'].items():
        for name,path in worker['modules'].items():
            if source=='release':
                assert Path(path).parent==ROOT/'research_pass1/release',(name,path)
            else:
                assert Path(path).parent==ROOT,(name,path)
    result['invalid_specifications_rejected']=len(bad)
    (HERE/'local_support_checks.json').write_text(json.dumps(result,indent=2)+'\n',newline='\n')
    print('Two isolated sources, four presets, independent decode/MSE passed; 12 invalid specs rejected.')


if __name__=='__main__':
    main()
