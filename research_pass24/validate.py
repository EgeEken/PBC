import subprocess
import sys
from common import *
from research_pass2.stream_tools import patch_signature


def main():
    setup();measured=json.loads((HERE/'omissions.json').read_text());checked=[]
    assert not measured['gate']['passed']
    parent={}
    for record in measured['files']:
        data,expected=saved(record['image'],record['preset'])
        header,patches,accounting=parse(data)
        parent[(record['image'],record['preset'])]=(header,[patch_signature(p) for p in patches])
        assert record['exact_roundtrip'] and sha(data)==record['stream_sha256']
        assert select(descriptions(header,patches,accounting),header[6])==record['selected']
    for row in measured['rows']:
        data=(HERE/row['file']).read_bytes()
        assert sha(data)==row['stream_sha256'] and len(data)==row['bytes']
        header,patches,accounting=parse(data)
        original_header,signatures=parent[(row['image'],row['preset'])]
        assert row['patch']>=original_header[6]
        expected_header=list(original_header);expected_header[10]-=1
        assert tuple(expected_header)==header
        expected=signatures[:row['patch']]+signatures[row['patch']+1:]
        assert [patch_signature(p) for p in patches]==expected
        assert len(patches)==row['remaining_patches']
        checked.append(dict(image=row['image'],preset=row['preset'],patch=row['patch'],stream_sha256=row['stream_sha256'],initialization_and_surviving_order_exact=True,header_count_exact=True))
    selections=[r for r in measured['rows'] if r['category']=='strict_rd_improvement']
    for row in selections:
        source=np.asarray(load(row['image']),dtype=np.uint8)
        output=np.asarray(PBC3.decompress((HERE/row['file']).read_bytes()).image.convert('RGB'))
        assert sha(output.tobytes())==row['pixels_sha256'] and rgb_mse(source,output)==row['rgb_mse']
    code="import sys,hashlib;from pathlib import Path;root=Path(sys.argv[1]);sys.path.insert(0,str(root));sys.path.insert(0,str(root/'research_pass1/release') if sys.argv[3]=='release' else str(root));from PBC3 import PBC3;PBC3.compress=classmethod(lambda *a,**k:(_ for _ in ()).throw(AssertionError('encoder called')));from learned_filler import LearnedFiller;LearnedFiller.load=classmethod(lambda *a,**k:(_ for _ in ()).throw(AssertionError('model loaded')));print(hashlib.sha256(PBC3.decompress(Path(sys.argv[2]).read_bytes()).image.convert('RGB').tobytes()).hexdigest())"
    for row in selections:
        for decoder in ('practical','release'):
            result=subprocess.check_output([sys.executable,'-c',code,str(ROOT),str(HERE/row['file']),decoder],cwd=ROOT,text=True).strip()
            assert result==row['pixels_sha256']
    write('validation.json',dict(provenance=meta(),original_roundtrip_files=len(measured['files']),omission_header_order_checks=checked,independently_scored_omissions=len(measured['rows']),strict_files_rescored=len(selections),model_and_encoder_forbidden_subprocesses=2*len(selections),decoders=['practical v0','immutable PBC3.0 release runtime'],notes='All80 omissions independently decoded/scored during measurement. Validation reparses saved files and verifies exactly one intended iterative patch removed, initialization/remaining patch order unchanged, correct header count/dimensions/flags, saved stream SHA and length. Three strict files independently rechecked and decoded in fresh practical/release processes. No proxy/subtraction implementation, integrated encoder, syntax change or extra omission search.'))
    print('validated',len(checked),'omission streams and',len(measured['files']),'roundtrip records;',2*len(selections),'independent model-free decoder processes',flush=True)


if __name__=='__main__':main()
