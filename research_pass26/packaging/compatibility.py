import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import *
import subprocess
sys.path.insert(0,str(PASS/'packaging'))
from cli import checked_body
pack=json.loads((PASS/'packaging/measurements.json').read_text())
portable=Path(pack['portable_directory'])
commands={'python':[sys.executable,str(PASS/'packaging/cli.py')],'onedir':[str(portable/'directory/PBC/pbc.exe')],'onefile':[str(portable/'single/pbc.exe')]}
image=load('eiffel.png').resize((96,95));out=PASS/'artifacts/packaging_tests';rows=[]
for case,cfg in [('stored',PBC3Config.quality(use_lzma=False)),('warmup',PBC3Config.quality(patch_count=8,downsample_rate=2,warmup_ratio=.5,warm_downsample_max_pixels=0))]:
    encoded=PBC3.compress(image,config=cfg);source=out/(case+'.pbc');source.write_bytes(encoded.data)
    meta=checked_body(encoded.data)
    assert (case!='stored' or meta['entropy']==0) and (case!='warmup' or meta['warmup'])
    expected=PBC3.decompress(encoded.data).image
    for mode,command in commands.items():
        target=out/(mode+'_'+case+'.png')
        env=dict(os.environ,PATH=r'C:\Windows\System32')
        p=subprocess.run(command+['decode',str(source),str(target),'--force'],cwd=portable,env=env,capture_output=True,text=True)
        assert p.returncode==0,(mode,case,p.stderr)
        assert Image.open(target).tobytes()==expected.tobytes()
        rows.append(dict(mode=mode,case=case,stream_sha256=sha(encoded.data),pixel_sha256=sha(expected.tobytes()),metadata=meta))
save('packaging/compatibility.json',rows)
print('stored/warmup compatibility passed',len(rows))
