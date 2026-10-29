import argparse
import importlib.metadata
from pathlib import Path
import shutil
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2];PASS=ROOT/'research_pass27'
p=argparse.ArgumentParser();p.add_argument('--kind',choices=('combined','decoder'),required=True);p.add_argument('--onefile',action='store_true');a=p.parse_args()
source=PASS/'artifacts/candidate';mode='onefile' if a.onefile else 'onedir'
exe=PASS/'artifacts/decoder-env/Scripts/python.exe' if a.kind=='decoder' else Path(sys.executable)
name='pbc-decode' if a.kind=='decoder' else 'pbc'
output=PASS/'artifacts'/a.kind/mode
cmd=[str(exe),'-m','PyInstaller','--noconfirm','--clean','--'+mode,'--name',name,'--paths',str(source),'--distpath',str(output),'--workpath',str(PASS/'artifacts'/('build_'+a.kind+'_'+mode)),'--specpath',str(PASS/'artifacts'),
     '--exclude-module','scipy','--exclude-module','psutil','--exclude-module','matplotlib','--exclude-module','setuptools','--exclude-module','pkg_resources']
for package in ('numpy','Pillow'): cmd+=['--copy-metadata',package]
if a.kind=='combined':
    cmd+=['--hidden-import','pbc3_cache']
    cmd+=['--additional-hooks-dir',str(ROOT/'research_pass26/packaging/hooks')]
    for package in ('numba','llvmlite'): cmd+=['--copy-metadata',package]
    for module in ('numba.tests','numba.cuda','numba.pycc'): cmd+=['--exclude-module',module]
    for filename in ('pbc3_ops.py','pbc3_features.py'): cmd+=['--add-data',str(source/filename)+';.']
    cmd+=['--add-data',str(source/'pbc3/patch_policy.npz')+';pbc3','--add-binary',str(source/'pbc3/mse.dll')+';pbc3']
else:
    for module in ('numba','llvmlite','PBC3','pbc3_ops','pbc3_heads','pbc3_features','learned_filler','pbc3_native','pbc3_cache','pbc3'):
        cmd+=['--exclude-module',module]
entry=source/('pbc_decode_cli.py' if a.kind=='decoder' else 'pbc_cli.py')
subprocess.run(cmd+[str(entry)],check=True)
if not a.onefile:
    folder=output/name
    for filename in ('README.md','INSTALL.md','LICENSE'): shutil.copy2(source/filename,folder/filename)
    license_file=Path(sys.base_prefix)/'LICENSE.txt'
    if license_file.exists(): shutil.copy2(license_file,folder/'PYTHON_LICENSE.txt')
    for path in folder.rglob('*'):
        if path.suffix in ('.nbc','.nbi'): raise RuntimeError('JIT caches must not be shipped')
    archive=shutil.make_archive(str(PASS/'artifacts'/('PBC32rc1-'+a.kind+'-Windows-x64')),'zip',output,name)
    print(archive)
