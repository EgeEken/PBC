import ast
import hashlib
import json
from pathlib import Path
import shutil
import sys
import urllib.request
import zipfile
ROOT=Path(__file__).resolve().parents[2]
PASS=ROOT/'research_pass27'

def prepare():
    archive=PASS/'artifacts/published31.zip';archive.parent.mkdir(parents=True,exist_ok=True)
    if not archive.exists():
        cached=ROOT/'research_pass26/artifacts/released.zip'
        if cached.exists(): shutil.copy2(cached,archive)
        else: urllib.request.urlretrieve('https://github.com/EgeEken/PBC/releases/download/v3.1/PBC3_1_Release.zip',archive)
    if hashlib.sha256(archive.read_bytes()).hexdigest()!='edec3ec73721c3f56913db21f6089a3b6a1637fb3481e61de5b0475da8eeaede':
        raise ValueError('published asset hash mismatch')
    source=PASS/'artifacts/published'
    with zipfile.ZipFile(archive) as z: z.extractall(source)
    dest=PASS/'artifacts/candidate';dest.mkdir(parents=True,exist_ok=True)
    names=('PBC3','pbc3_ops','pbc3_heads','pbc3_features','pbc3_stream','pbc3_types','pbc3_trace','learned_filler')
    for name in names: shutil.copy2(source/(name+'.py'),dest/(name+'.py'))
    package=dest/'pbc3';package.mkdir(exist_ok=True)
    shutil.copy2(source/'patch_policy.npz',package/'patch_policy.npz')
    shutil.copy2(ROOT/'research_pass26/native/mse.c',package/'mse.c')
    package.joinpath('__init__.py').write_text('from PBC3 import PBC3\nfrom pbc3_types import PBC3Config, PBC3Result\n__version__="3.2.0rc1"\n',encoding='utf-8',newline='\n')
    for path in (PASS/'source').iterdir():
        if path.is_file(): shutil.copy2(path,dest/path.name)
    ops=(dest/'pbc3_ops.py').read_text();lines=ops.splitlines(keepends=True)
    wanted=('bicubic_resample_2d','ceil_div','range_counts','range_for_mask_index','active_value_count','resolve_palette_bitcount','palette_generator','signed_resample')
    nodes=[n for n in ast.parse(ops).body if isinstance(n,ast.FunctionDef) and n.name in wanted]
    assert len(nodes)==len(wanted)
    pure='import math\nimport numpy as np\nfrom PIL import Image\n\n'+'\n\n'.join(''.join(lines[n.lineno-1:n.end_lineno]) for n in nodes)
    (dest/'pbc3_decode_ops.py').write_text(pure+'\n',encoding='utf-8',newline='\n')
    for node in reversed(nodes): del lines[node.lineno-1:node.end_lineno]
    (dest/'pbc3_ops.py').write_text(''.join(lines)+'\nfrom pbc3_decode_ops import '+', '.join(wanted)+'\n',encoding='utf-8',newline='\n')
    path=dest/'pbc3_stream.py';path.write_text(path.read_text().replace('import pbc3_ops as ops','import pbc3_decode_ops as ops'),encoding='utf-8',newline='\n')
    path=dest/'pbc3_types.py';path.write_text(path.read_text().replace('Path(__file__).resolve().with_name("patch_policy.npz")','Path(__file__).resolve().parent / "pbc3" / "patch_policy.npz"'),encoding='utf-8',newline='\n')
    path=dest/'PBC3.py';text=path.read_text()
    default='auto'
    text=text.replace('rng_mode="release", **kwargs)',f'rng_mode="release", mse_backend="{default}", **kwargs)')
    text=text.replace('rng_mode="release", frame_every:',f'rng_mode="release", mse_backend="{default}", frame_every:')
    text=text.replace('rng_mode=rng_mode, frame_every=0','rng_mode=rng_mode, mse_backend=mse_backend, frame_every=0')
    text=text.replace('        config = _config(config, kwargs)\n        owns_trace', '        from pbc3_native import select_metric\n        metric = select_metric(mse_backend, ops.final_mse)\n        config = _config(config, kwargs)\n        owns_trace')
    text=text.replace('ops.final_mse(orig_compare, out_img)','metric(orig_compare, out_img)')
    assert text.count('mse_backend=')==3 and 'metric(orig_compare' in text
    path.write_text(text,encoding='utf-8',newline='\n')
    return dest

if __name__=='__main__': print(prepare())
