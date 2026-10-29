import argparse
from pathlib import Path
import subprocess
import sys
from freeze import freeze

p=argparse.ArgumentParser();p.add_argument('--onefile',action='store_true');a=p.parse_args()
source=freeze()
root=Path(__file__).resolve().parent
out=root.parent/'artifacts'
mode='onefile' if a.onefile else 'onedir'
cmd=[sys.executable,'-m','PyInstaller','--noconfirm','--clean','--'+mode,'--name','pbc',
     '--distpath',str(out/mode),'--workpath',str(out/('build_'+mode)),'--specpath',str(out),
     '--paths',str(source),'--additional-hooks-dir',str(root/'hooks'),'--exclude-module','numba.tests','--exclude-module','numba.cuda','--exclude-module','numba.pycc',
     '--exclude-module','setuptools','--exclude-module','pkg_resources',
     '--exclude-module','scipy','--exclude-module','psutil','--exclude-module','matplotlib',
     '--add-data',str(source/'patch_policy.npz')+';'+'.']
for path in source.glob('*.py'):
    if path.name!='PBC3_animation.py': cmd+=['--add-data',str(path)+';'+'.']
subprocess.run(cmd+[str(root/'cli.py')],check=True)
