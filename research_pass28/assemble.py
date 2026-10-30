import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import zipfile
import pefile
from PyInstaller.archive.readers import CArchiveReader
ROOT=Path(__file__).resolve().parents[1];PASS=ROOT/'research_pass28';source=PASS/'artifacts/release-source';dist=PASS/'artifacts/dist'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
implementation=json.loads((source/'IMPLEMENTATION.json').read_text())
for name,digest in implementation['files'].items():
    assert sha(source/name)==digest,name
    assert sha(source/name)==sha(ROOT/'research_pass27/artifacts/candidate'/name),name
wheel=next(dist.glob('*.whl'));sdist=next(dist.glob('*.tar.gz'))
with zipfile.ZipFile(wheel) as z:
    for name,digest in implementation['files'].items():assert hashlib.sha256(z.read(name)).hexdigest()==digest,name
    assert json.loads(z.read('pbc3/IMPLEMENTATION.json'))==implementation
    assert not any(n.endswith(('.dll','.nbc','.nbi')) for n in z.namelist())
    assert any(n.endswith('/licenses/THIRD_PARTY_NOTICES.md') for n in z.namelist())
with tarfile.open(sdist) as t:
    for name,digest in implementation['files'].items():
        item=next(n for n in t.getnames() if n.endswith('/'+name));assert hashlib.sha256(t.extractfile(item).read()).hexdigest()==digest,name
for name in ('combined','decoder'):
    app=dist/name/'onedir'/('pbc' if name=='combined' else 'pbc-decode')
    for filename in ('LICENSE','README.md','INSTALL.md','THIRD_PARTY_NOTICES.md','IMPLEMENTATION.json'):
        assert sha(app/filename)==sha(source/filename)
    assert json.loads((app/'IMPLEMENTATION.json').read_text())==implementation
    assert not any(p.suffix in ('.nbc','.nbi') for p in app.rglob('*'))
    if name=='decoder':
        assert not (app/'_internal/numba').exists() and not (app/'_internal/llvmlite').exists()
        assert not list(app.rglob('*.npz'))
    else:assert sha(app/'_internal/pbc3/patch_policy.npz')==implementation['files']['pbc3/patch_policy.npz']
reader=CArchiveReader(str(dist/'decoder/onefile/pbc-decode.exe'))
assert all(n in reader.toc for n in ('LICENSE','THIRD_PARTY_NOTICES.md','IMPLEMENTATION.json'))
pyz=reader.open_embedded_archive('PYZ.pyz')
assert not any(n=='numba' or n.startswith(('numba.','llvmlite.')) for n in pyz.toc)
assert not list((dist/'combined/onedir/pbc').rglob('tbbpool*.pyd'))
pe_rows=[];unresolved=[]
for kind in ('combined','decoder'):
    app=dist/kind/'onedir'/('pbc' if kind=='combined' else 'pbc-decode')
    available={p.name.lower() for p in app.rglob('*') if p.is_file()}
    for p in app.rglob('*'):
        if p.suffix.lower() not in ('.exe','.dll','.pyd'):continue
        image=pefile.PE(str(p),fast_load=True);image.parse_data_directories(directories=[1,13])
        imports=sorted({entry.dll.decode().lower() for attr in ('DIRECTORY_ENTRY_IMPORT','DIRECTORY_ENTRY_DELAY_IMPORT') for entry in getattr(image,attr,[])})
        missing=[n for n in imports if n not in available and not (Path(os.environ['SystemRoot'])/'System32'/n).exists() and not n.startswith(('api-ms-','ext-ms-'))]
        pe_rows.append(dict(kind=kind,path=str(p.relative_to(app)),imports=imports,missing=missing))
        unresolved+=missing;image.close()
assert not unresolved,unresolved
(PASS/'artifacts/pe-imports.json').write_text(json.dumps(pe_rows,indent=2)+'\n')
# An exact root tree is recorded without creating any commit, branch or tag.
(source/'.gitattributes').write_text('* -text\n',newline='\n')
files=[p for p in source.rglob('*') if p.is_file() and not any(part in ('build','__pycache__') or part.endswith('.egg-info') for part in p.relative_to(source).parts) and p.suffix not in ('.dll','.nbc','.nbi','.pyc')]
files.sort(key=lambda p:p.relative_to(source).as_posix())
repo=PASS/'artifacts/source-objects.git'
if not repo.exists():subprocess.run(['git','init','--bare',str(repo)],check=True,stdout=subprocess.PIPE)
cmd=['git','--git-dir',str(repo)]
env=dict(os.environ,GIT_INDEX_FILE=str(PASS/'artifacts/source-tree.index'))
subprocess.run(cmd+['read-tree','--empty'],check=True,env=env)
records=[]
for p in files:
    blob=subprocess.check_output(cmd+['hash-object','-w','--stdin'],input=p.read_bytes()).decode().strip()
    records.append('100644 '+blob+'\t'+p.relative_to(source).as_posix()+'\n')
subprocess.run(cmd+['update-index','--index-info'],input=''.join(records),text=True,check=True,env=env)
tree=subprocess.check_output(cmd+['write-tree'],env=env,text=True).strip()
archive=dist/'PBC32rc1-release-source.zip'
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
    for p in files:
        info=zipfile.ZipInfo('PBC-3.2.0rc1-source/'+p.relative_to(source).as_posix(),date_time=(1980,1,1,0,0,0));info.external_attr=0o100644<<16;info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,p.read_bytes())
source_tree=dict(git_tree_sha1=tree,files={p.relative_to(source).as_posix():sha(p) for p in files},archive_sha256=sha(archive),implementation_sha256=sha(source/'IMPLEMENTATION.json'))
(PASS/'SOURCE_TREE.json').write_text(json.dumps(source_tree,indent=2)+'\n',newline='\n')
exe=dist/'PBC32rc1-decoder-Windows-x64.exe';shutil.copy2(dist/'decoder/onefile/pbc-decode.exe',exe)
companion=PASS/'artifacts/onefile-download';companion.mkdir(exist_ok=True)
shutil.copy2(exe,companion/'pbc-decode.exe')
for name in ('LICENSE','README.md','INSTALL.md','THIRD_PARTY_NOTICES.md'):shutil.copy2(source/name,companion/name)
shutil.copytree(source/'licenses',companion/'licenses',dirs_exist_ok=True)
convenience=Path(shutil.make_archive(str(dist/'PBC32rc1-decoder-onefile-with-notices'),'zip',companion))
paths={'combined':dist/'PBC32rc1-combined-Windows-x64.zip','decoder':dist/'PBC32rc1-decoder-Windows-x64.zip','decoder_onefile':exe,'decoder_onefile_with_notices':convenience,'wheel':wheel,'sdist':sdist,'release_source':archive}
rows=[dict(kind=k,path=str(p.resolve()),filename=p.name,bytes=p.stat().st_size,sha256=sha(p)) for k,p in paths.items()]
smoke=PASS/'artifacts/smoke-bundle';smoke.mkdir(exist_ok=True)
for p in (PASS/'smoke').iterdir():
    if p.is_file():shutil.copy2(p,smoke/p.name)
(smoke/'smoke-checksums.json').write_text(json.dumps(dict(version='3.2.0rc1',artifacts=[r for r in rows if r['kind'] in ('combined','decoder','decoder_onefile')]),indent=2)+'\n')
smokezip=Path(shutil.make_archive(str(dist/'PBC32rc1-consumer-smoke-test'),'zip',smoke))
rows.append(dict(kind='consumer_smoke_test',path=str(smokezip.resolve()),filename=smokezip.name,bytes=smokezip.stat().st_size,sha256=sha(smokezip)))
manifest=dict(version='3.2.0rc1',promoted_to_stable=False,codec_implementation_commit='1b207e1c67fee4c911c57c9f7d0d4c0845e82952',acceptance_build_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),implementation_sha256=sha(source/'IMPLEMENTATION.json'),release_source_git_tree=tree,source_snapshot_path=str(source.resolve()),artifacts=rows)
(PASS/'PBC32_DISTRIBUTION_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n',newline='\n')
checksums=''.join(r['sha256']+'  '+r['filename']+'\n' for r in rows)+sha(PASS/'PBC32_DISTRIBUTION_MANIFEST.json')+'  PBC32_DISTRIBUTION_MANIFEST.json\n'
(PASS/'SHA256SUMS.txt').write_text(checksums,newline='\n')
(PASS/'artifacts/packaging-audit.json').write_text(json.dumps(dict(common_implementation_exact=True,wheel_sdist_exact=True,notice_inventory_complete=True,pe_files_audited=len(pe_rows),unresolved_imports=unresolved,decoder_no_numba_llvm_model=True,onefile_licenses_embedded=True,source_tree=tree),indent=2)+'\n')
print('PACKAGING_AUDIT_PASS',len(pe_rows),'PE files;',len(files),'source files; tree',tree)
