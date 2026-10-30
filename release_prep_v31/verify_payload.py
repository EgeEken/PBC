from pathlib import Path
import json
import marshal
import types
import sys
from PyInstaller.archive.readers import CArchiveReader
HERE=Path(__file__).resolve().parent
OUT=HERE/'artifacts'
DIST=OUT/('dist-final/PBC' if '--final' in sys.argv else 'dist/PBC')
app=CArchiveReader(str(DIST/'PBC.exe'))
probe=CArchiveReader(str(OUT/'probe-dist/PBC-probe/PBC-probe.exe'))
a=app.open_embedded_archive('PYZ.pyz')
b=probe.open_embedded_archive('PYZ.pyz')
def signature(c):
    return (c.co_code,c.co_names,c.co_varnames,c.co_freevars,c.co_cellvars,c.co_argcount,c.co_kwonlyargcount,c.co_flags,
        tuple(signature(v) if isinstance(v,types.CodeType) else v for v in c.co_consts))
modules=('PBC3','pbc3_ops','pbc3_features','pbc3_types','pbc3_heads','pbc3_stream','learned_filler',
         'pbc_ui','pbc_windows','pbc_startup','pbc3_native','pbc3_cache','pbc3_decoder','pbc3_decode_ops','pbc3_trace')
for name in modules:
    assert signature(a.extract(name))==signature(b.extract(name)),name
assert signature(marshal.loads(app.extract('pbc_gui')))==signature(b.extract('pbc_gui'))
assert signature(marshal.loads(app.extract('pbc_gui')))==signature(compile((HERE/'pbc_gui.py').read_text(encoding='utf-8'),'pbc_gui.py','exec'))
assert signature(a.extract('pbc_ui'))==signature(compile((HERE/'pbc_ui.py').read_text(encoding='utf-8'),'pbc_ui.py','exec'))
for name in ('pbc3/mse.dll','pbc3/patch_policy.npz'):
    assert (DIST/'_internal'/name).read_bytes()==(OUT/'probe-dist/PBC-probe/_internal'/name).read_bytes()
(OUT/'frozen-payload-equivalence.json').write_text(json.dumps(dict(codec_modules=list(modules),
    gui_entry_bytecode_exact=True,current_gui_sources_exact=True,native_dll_exact=True,model_exact=True),indent=2)+'\n')
print('Production GUI and test executable payloads match')
