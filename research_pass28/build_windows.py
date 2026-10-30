import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
source=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--kind',choices=('combined','decoder'),required=True);p.add_argument('--onefile',action='store_true');p.add_argument('--decoder-python',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
output=a.output.resolve();output.mkdir(parents=True,exist_ok=True)
exe=a.decoder_python.resolve() if a.kind=='decoder' else Path(sys.executable)
if a.kind=='decoder':
    subprocess.run([str(exe),'-c',"import importlib.util;assert importlib.util.find_spec('numba') is None;assert importlib.util.find_spec('llvmlite') is None"],check=True)
name='pbc-decode' if a.kind=='decoder' else 'pbc'
mode='onefile' if a.onefile else 'onedir'
dist=output/a.kind/mode
cmd=[str(exe),'-m','PyInstaller','--noconfirm','--clean','--'+mode,'--name',name,'--paths',str(source),'--distpath',str(dist),'--workpath',str(output/('build_'+a.kind+'_'+mode)),'--specpath',str(output)]
for module in ('scipy','psutil','matplotlib','setuptools','pkg_resources'):cmd+=['--exclude-module',module]
for package in ('numpy','Pillow'):cmd+=['--copy-metadata',package]
for filename in ('LICENSE','README.md','INSTALL.md','THIRD_PARTY_NOTICES.md','IMPLEMENTATION.json'):
    cmd+=['--add-data',str(source/filename)+';.']
cmd+=['--add-data',str(source/'licenses')+';licenses']
if a.kind=='combined':
    cmd+=['--hidden-import','pbc3_cache','--additional-hooks-dir',str(source/'tools/hooks')]
    for package in ('numba','llvmlite'):cmd+=['--copy-metadata',package]
    for module in ('numba.tests','numba.cuda','numba.pycc','numba.np.ufunc.tbbpool'):cmd+=['--exclude-module',module]
    for filename in ('pbc3_ops.py','pbc3_features.py'):cmd+=['--add-data',str(source/filename)+';.']
    cmd+=['--add-data',str(source/'pbc3/patch_policy.npz')+';pbc3','--add-binary',str(source/'pbc3/mse.dll')+';pbc3']
else:
    for module in ('numba','llvmlite','PBC3','pbc3_ops','pbc3_heads','pbc3_features','learned_filler','pbc3_native','pbc3_cache','pbc3'):
        cmd+=['--exclude-module',module]
entry=source/('pbc_decode_cli.py' if a.kind=='decoder' else 'pbc_cli.py')
subprocess.run(cmd+[str(entry)],check=True)
if not a.onefile:
    folder=dist/name
    for filename in ('LICENSE','README.md','INSTALL.md','THIRD_PARTY_NOTICES.md','IMPLEMENTATION.json'):shutil.copy2(source/filename,folder/filename)
    shutil.copytree(source/'licenses',folder/'licenses',dirs_exist_ok=True)
    assert not any(p.suffix in ('.nbc','.nbi') for p in folder.rglob('*'))
    print(shutil.make_archive(str(output/('PBC-3.2.0-'+a.kind+'-Windows-x64')),'zip',dist,name))
