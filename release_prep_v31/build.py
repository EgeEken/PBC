import json
from pathlib import Path
import shutil
import subprocess
import sys
HERE=Path(__file__).resolve().parent
SOURCE=HERE/'source'
OUT=HERE/'artifacts'
final='--final' in sys.argv
probe='--probe' in sys.argv
name='PBC-probe' if probe else 'PBC'
cmd=[sys.executable,'-m','PyInstaller','--noconfirm','--clean','--onedir','--console' if probe else '--windowed',
     '--name',name,'--manifest',str(HERE/'app.manifest'),'--version-file',str(HERE/'version.txt'),'--paths',str(SOURCE),'--distpath',str(OUT/('probe-dist' if probe else 'dist-final' if final else 'dist')),
     '--workpath',str(OUT/('probe-build' if probe else 'build')),'--specpath',str(OUT),
     '--additional-hooks-dir',str(HERE/'hooks'),
     '--hidden-import','pbc3_cache']
for module in ('scipy','psutil','matplotlib','setuptools','pkg_resources','numba.tests','numba.cuda','numba.pycc','numba.np.ufunc.tbbpool'):
    cmd+=['--exclude-module',module]
for package in ('numpy','Pillow','numba','llvmlite'):
    cmd+=['--copy-metadata',package]
for name in ('pbc3_ops.py','pbc3_features.py'):
    cmd+=['--add-data',str(SOURCE/name)+';.']
cmd+=['--add-data',str(SOURCE/'pbc3/patch_policy.npz')+';pbc3',
      '--add-binary',str(SOURCE/'pbc3/mse.dll')+';pbc3',
      '--add-data',str(SOURCE/'licenses')+';licenses',
      '--add-data',str(SOURCE/'LICENSE')+';licenses/PBC',
      '--add-data',str(SOURCE/'THIRD_PARTY_NOTICES.md')+';licenses']
if probe:
    cmd+=['--hidden-import','pbc_gui','--hidden-import','test_final_gui','--hidden-import','profile_ui','--paths',str(HERE)]
subprocess.run(cmd+[str(HERE/('frozen_probe.py' if probe else 'pbc_gui.py'))],check=True)
if probe:
    print(OUT/'probe-dist/PBC-probe/PBC-probe.exe')
    sys.exit(0)
app=OUT/('dist-final/PBC' if final else 'dist/PBC')
shutil.copytree(SOURCE,app/'Python',ignore=shutil.ignore_patterns('__pycache__','*.egg-info','build','dist'),dirs_exist_ok=True)
(app/'README.txt').write_text("""PBC V3.1 - Windows x64

Open PBC.exe. Choose an image, select a preset, and click Compress.
The encoder prepares in the background at startup; the window stays responsive.\nSome manual settings may compile additional kernels.
Save the .pbc file or reconstructed PNG. Decode opens historical PBC3 v0 files.
Registry keeps the latest 30 results for this session.

Python 3.11 library: python -m pip install ./Python
Usage: Python/README.md. Licenses: _internal/licenses and Python/LICENSE.
Demo: https://huggingface.co/spaces/EgeEken/PBC

No installer, administrator access or Python installation is needed for PBC.exe.
""",encoding='utf-8',newline='\n')
assert (app/'_internal/pbc3/mse.dll').is_file()
assert not any(p.suffix in ('.nbc','.nbi') for p in app.rglob('*'))
print(app/'PBC.exe')
