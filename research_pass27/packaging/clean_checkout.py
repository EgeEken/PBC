import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import *
import subprocess
import tempfile
import zipfile
root=Path(tempfile.mkdtemp(prefix='p27-clean-'));clone=root/'repo'
commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
subprocess.run(['git','clone','--shared','--no-checkout','--branch','research/experiments',str(ROOT),str(clone)],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
patterns='/research_pass27/\n/research_pass26/native/mse.c\n/research_pass26/packaging/hooks/hook-numba.py\n'
subprocess.run(['git','-C',str(clone),'sparse-checkout','set','--no-cone','--stdin'],input=patterns,text=True,check=True)
subprocess.run(['git','-C',str(clone),'checkout','research/experiments'],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
assert subprocess.check_output(['git','-C',str(clone),'rev-parse','HEAD'],text=True).strip()==commit
assert not subprocess.check_output(['git','-C',str(clone),'status','--porcelain'],text=True).strip()
base=clone/'research_pass27';clean=base/'artifacts/candidate'
with (PASS/'artifacts/clean-build.log').open('w') as log:
    for args in (['packaging/prepare.py'],['native/build.py'],['packaging/build.py','--kind','combined']):
        subprocess.run([sys.executable,str(base/args[0])]+args[1:],cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
    decoder_env=base/'artifacts/decoder-env'
    subprocess.run(['uv','venv','--no-project',str(decoder_env),'--python',sys.executable],check=True,stdout=log,stderr=subprocess.STDOUT)
    subprocess.run(['uv','pip','install','--python',str(decoder_env/'Scripts/python.exe'),'-r',str(base/'requirements-decoder-build.txt')],check=True,stdout=log,stderr=subprocess.STDOUT)
    subprocess.run([sys.executable,str(base/'packaging/build.py'),'--kind','decoder'],cwd=root,check=True,stdout=log,stderr=subprocess.STDOUT)
    subprocess.run([sys.executable,'-m','build','--wheel','--no-isolation','--outdir',str(root/'packages'),str(clean)],check=True,stdout=log,stderr=subprocess.STDOUT)
local=next((PASS/'artifacts/packages').glob('*.whl'));rebuilt=next((root/'packages').glob('*.whl'))
with zipfile.ZipFile(local) as a,zipfile.ZipFile(rebuilt) as b:
    aa={n:sha(a.read(n)) for n in a.namelist()};bb={n:sha(b.read(n)) for n in b.namelist()};assert aa==bb,'wheel payloads differ'
image=load('eiffel.png').resize((43,41));image.save(root/'input.png')
sys.path.insert(0,str(PUBLISHED))
from PBC3 import PBC3
from pbc3_types import PBC3Config
expected=PBC3.compress(image,config=PBC3Config.quality())
env=dict(os.environ,PATH=r'C:\Windows\System32',NUMBA_CACHE_DIR=str(root/'cache'))
for name in ('PYTHONPATH','PYTHONHOME','VIRTUAL_ENV'):env.pop(name,None)
combined=base/'artifacts/combined/onedir/pbc/pbc.exe';decoder=base/'artifacts/decoder/onedir/pbc-decode/pbc-decode.exe'
subprocess.run([str(combined),'encode',str(root/'input.png'),str(root/'out.pbc'),'--preset','quality'],cwd=root,env=env,check=True,stdout=subprocess.PIPE)
assert (root/'out.pbc').read_bytes()==expected.data
subprocess.run([str(decoder),str(root/'out.pbc'),str(root/'out.png')],cwd=root,env=env,check=True,stdout=subprocess.PIPE)
assert Image.open(root/'out.png').tobytes()==PBC3.decompress(expected.data).image.tobytes()
assert not subprocess.check_output(['git','-C',str(clone),'status','--porcelain'],text=True).strip()
save('packaging/clean_checkout.json',dict(root=str(root),commit=commit,tracked_checkout_clean=True,published_asset_downloaded_and_verified=True,no_existing_artifacts_reused=True,wheel_payloads_identical=True,combined_encode_exact=True,decoder_pixels_exact=True,runtime_no_python_or_compiler_on_path=True,combined_exe=dict(path=str(combined),sha256=sha(combined.read_bytes())),decoder_exe=dict(path=str(decoder),sha256=sha(decoder.read_bytes()))))
print('clean source checkout build passed')
