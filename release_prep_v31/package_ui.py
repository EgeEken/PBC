import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
OUT=HERE/'artifacts'
EVIDENCE=OUT/'ui-polish'
OLD=OUT/'rollback/pre-ui-polish'
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
assert sha(OLD/'PBC3_1_Release.zip')=='13ad674d58e68593bb96c0e6d19c1877179a4b6ba026eaef06eecac8f8505c31'
assert sha(OUT/'measurements.jsonl')=='98a7dbeba061b0fd6441171777c79be3bdcb6075a78b80b561aa438cf56ac1d8'
for name,digest in json.loads((OLD/'library-source.json').read_text()).items():
    assert sha(HERE/'source'/name)==digest,name
before=ast.parse((OLD/'pbc_gui.py').read_text(encoding='utf-8'))
after=ast.parse((HERE/'pbc_gui.py').read_text(encoding='utf-8'))
for name in ('configuration','load_image','encode','decode','export'):
    a=next(n for n in before.body if isinstance(n,ast.FunctionDef) and n.name==name)
    b=next(n for n in after.body if isinstance(n,ast.FunctionDef) and n.name==name)
    assert ast.dump(a)==ast.dump(b),name
for phase in ('first-use','cached'):
    data=json.loads((OUT/f'frozen-{phase}.json').read_text())
    assert len(data['checks'])==8 and data['known_streams']==32 and data['partial_checks']==64 and data['invalid_rejected']==45
    assert data['native_dll_sha256']=='0def06d35fb5d4f97949ba2bc1ddb55651afeb2d007971a7c626c951cce928f3'
    assert data['path'].lower()==r'c:\windows\system32'
assert json.loads((OUT/'frozen-payload-equivalence.json').read_text())['gui_entry_bytecode_exact']
assert 'Traceback' not in (EVIDENCE/'test-gui.log').read_text(encoding='utf-8')
assert len(json.loads((EVIDENCE/'layout-tests.json').read_text()))==18
expected_images=('01-encode-empty-final.png','02-encode-selected-final.png','03-encode-completed-final.png',
    '04-decode-completed-final.png','05-registry-populated-final.png','06-manual-final.png',
    '07-manual-minimum-125.png','08-encode-maximized.png','09-encode-minimum-100.png','10-encode-empty-100.png')
for name in expected_images:
    assert (EVIDENCE/'screenshots'/name).is_file(),name
stage=OUT/'staged-ui/PBC3_1_Release'
shutil.copytree(OUT/'dist/PBC',stage,ignore=shutil.ignore_patterns('__pycache__','*.nbc','*.nbi'),dirs_exist_ok=True)
assert sorted(p.name for p in stage.iterdir())==['PBC.exe','Python','README.txt','_internal']
assert [p.name for p in stage.rglob('*.exe')]==['PBC.exe']
assert not any(p.suffix in ('.nbc','.nbi') for p in stage.rglob('*'))
assert not any('matplotlib' in str(p.relative_to(stage)).lower() for p in stage.rglob('*'))
files=sorted(p for p in stage.rglob('*') if p.is_file())
candidate=EVIDENCE/'PBC3_1_Release.zip'
with zipfile.ZipFile(candidate,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
    for p in files:
        z.write(p,p.relative_to(stage.parent).as_posix())
with zipfile.ZipFile(candidate) as z:
    assert z.testzip() is None
    for item in z.infolist():
        assert hashlib.sha256(z.read(item)).hexdigest()==sha(stage.parent/item.filename),item.filename
archive=OUT/'final/PBC3_1_Release.zip'
shutil.copy2(candidate,archive)
assert sha(archive)==sha(candidate)
candidate.unlink()
identity=json.loads((HERE/'artifacts/public-main/identity.json').read_text())
old_bytes=(OLD/'PBC3_1_Release.zip').stat().st_size
manifest=dict(version='3.1.0',zip=str(archive),zip_bytes=archive.stat().st_size,zip_sha256=sha(archive),
    extracted_bytes=sum(p.stat().st_size for p in files),file_count=len(files),
    executable=str(stage/'PBC.exe'),exe_sha256=sha(stage/'PBC.exe'),
    previous_zip=str(OLD/'PBC3_1_Release.zip'),previous_zip_sha256=sha(OLD/'PBC3_1_Release.zip'),
    size_change_bytes=archive.stat().st_size-old_bytes,size_change_percent=100*(archive.stat().st_size/old_bytes-1),
    runtime_dependency_changes=[],main_readme_commit=identity['main_commit'],
    screenshots={name:sha(EVIDENCE/'screenshots'/name) for name in expected_images},
    gui_source={name:sha(HERE/name) for name in ('pbc_gui.py','pbc_ui.py','build.py','app.manifest')},
    benchmark_rerun=False,codec_functions_exact=True,python_library_files_exact=True)
(EVIDENCE/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
(OUT/'final-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
index=['# Desktop GUI screenshots','',
'Actual Tkinter application windows, using local Cat/Flower inputs. No image mockups. Initial inspection led to aligned preview areas, larger technical labels, and responsive Manual-panel height.',
'',
'The Windows display setting remained 150% (144 DPI). The 100%/125% cases use per-process Tk rendering scales; global Windows display settings were not changed. Those checks do not certify switching the Windows display setting or moving across monitors. Native-scale captures use the actual display DPI. The capture service normalizes its image coordinates to desktop logical pixels.',
'']
for name in expected_images:
    index.append(f'- [{name}]({name})')
(EVIDENCE/'screenshots/README.md').write_text('\n'.join(index)+'\n',encoding='utf-8')
report=f"""# V3.1 UI and README validation

Prepared locally on research/experiments at 5f95490a6f630f739111d5d7fc0b8d181711378a. No commit, push, deployment, release edit or tag change. The research README and public main are unchanged.

README: README-main.patch updates only the V3.1 section of public main README at {identity['main_commit']}. public_readme/README.md is a self-contained preview. The complete patch with three PNG assets and docs/benchmarks/v31.md is artifacts/ui-polish/README-main-with-assets.patch; it passes git apply --check against the saved public main. Charts use actual saved image/preset medians, with assertions for every plotted value. The definitive 25-image benchmark was not rerun. Claims remain 41.03% lower encode time and 53.59% lower decode time versus V3.0, plus 10.28% lower encode time versus published V3.1. The 0.37% decode change is neutral.

The former third image was a historical cross-codec comparison with obsolete PBC timings. Its conventional-codec series has no matching definitive measurements, so it was replaced with the measured three-way PBC overview. No mixed-dataset cross-codec claim was introduced. Historical README sections are byte-identical.

GUI: compact branded header/navigation, aligned source → circular action → output panels, restrained rounded surfaces, dashed source selector, readable statistics and persistent exports, segmented modes/presets, grouped scrollable Manual fields, responsive Registry cards and per-item actions. Hover, keyboard focus, selected, disabled, processing and error states are styled. High-DPI awareness is explicit. No new runtime framework or dependency; Matplotlib is an internal plotting dependency only.

Validation: source GUI suite passes four presets with RGB/RGBA, actual Manual and Semi-Auto overrides, Browse selection, image-change result reset, export/overwrite handling, Unicode paths, invalid files, Registry limit/reopen, and worker heartbeats. Focused keyboard testing verifies Manual fields scroll into view. Layout checks pass 18 rendering-scale/size/mode combinations with visible actions and aligned previews. Frozen first-use and cached tests pass with PATH=C:\\Windows\\System32: eight exact preset outputs, 32 historical streams, 64 partial checks and 45 invalid/resource rejections. Auto native C MSE is active. Production/test executable GUI and codec bytecode, DLL and model match. Python library files and encoder/decoder/configuration functions match the preserved functional baseline. Full-size Cat/Flower visual-review streams match saved benchmark candidate hashes.

Visual review: six required real states plus minimum, maximized and 100%/125% rendering checks were inspected and saved under artifacts/ui-polish/screenshots/; see its README.md. Source/output preview areas were aligned after inspection, technical labels enlarged, and Manual height made responsive. No clipped primary actions were found. Windows itself remained at 150% scaling; 100%/125% checks were per-process rendering checks, not global OS-setting changes.

Distribution: {manifest['zip_bytes']:,} bytes compressed ({manifest['zip_bytes']/1_000_000:.2f} MB), {manifest['extracted_bytes']:,} bytes extracted, {manifest['file_count']} files. Change from the preserved ZIP: {manifest['size_change_bytes']:+,} bytes ({manifest['size_change_percent']:+.3f}%). Layout remains PBC.exe / _internal / Python / README.txt; one executable, no public tests/reports/tools, and unchanged legal notices. Every ZIP member and CRC verifies.

- ZIP: {archive}
- ZIP SHA-256: {manifest['zip_sha256']}
- Executable: {manifest['executable']}
- Executable SHA-256: {manifest['exe_sha256']}
- Rollback: {manifest['previous_zip']}
- Rollback SHA-256: {manifest['previous_zip_sha256']}

Changed implementation/build/test files: pbc_gui.py, new pbc_ui.py, app.manifest, build.py, test_gui.py, layout_check.py, verify_payload.py and run_frozen.py. Supporting files: plot_readme.py, requirements-plot.txt, visual_session.py, package_ui.py, README-main.patch and public_readme/. The three charts and public benchmark report are staged in the research index for review. No codec/library source or model was edited.

Limitations: unsigned Windows x64 build, validated on this Windows 11 machine with developer tools installed; Python/compiler were removed from runtime PATH. No fresh OS, Windows 10, cross-monitor DPI or HF validation was added. Small Manual windows intentionally reduce preview size while preserving actions. Drag-and-drop and persistent history remain absent. Publication remains unauthorized.
"""
production=EVIDENCE/'production-validation.json'
if production.exists():
    proof=json.loads(production.read_text())
    if proof.get('exe_sha256')==manifest['exe_sha256'] and proof.get('exact'):
        report=report.replace('Production/test executable GUI and codec bytecode, DLL and model match.',
            'Production/test executable GUI and codec bytecode, DLL and model match. The actual production executable was launched with Python/compiler off PATH; a full-size Cat Quality encode completed from a fresh cache, its DLL was confirmed loaded, and stream/pixel hashes matched the saved candidate.')
(HERE/'VALIDATION.md').write_text(report,encoding='utf-8',newline='\n')
print(json.dumps(manifest,indent=2))
