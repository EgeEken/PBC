import hashlib
import json
from pathlib import Path
import shutil
import statistics
import zipfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
OUT=HERE/'artifacts'
EVIDENCE=OUT/'resume-finalization'
ROLLBACK=OUT/'rollback/pre-final-corrections/PBC3_1_Release.zip'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

assert sha(ROLLBACK)=='37cd53fab5cba11f419ab232cae40c632b7c51fc1cea8bcf47251930f46ee97d'
for name,digest in json.loads((EVIDENCE/'protected.json').read_text()).items():
    assert sha(ROOT/name)==digest,name
for name,digest in json.loads((HERE/'library-source.json').read_text()).items():
    assert sha(HERE/'source'/name)==digest,name
proof=json.loads((EVIDENCE/'frozen-correctness.json').read_text())
assert len(proof['checks'])==8 and proof['known_streams']==32 and proof['partial_checks']==64 and proof['invalid_rejected']==45
assert proof['native_dll_sha256']=='0def06d35fb5d4f97949ba2bc1ddb55651afeb2d007971a7c626c951cce928f3'
assert proof['path'].lower()==r'c:\windows\system32'
assert json.loads((OUT/'frozen-payload-equivalence.json').read_text())['current_gui_sources_exact']
features=json.loads((EVIDENCE/'features-frozen.json').read_text())
assert features['click_regions_bounded'] and features['native_drop_message_routes'] and features['clipboard_rgba_exact']
assert len(json.loads((EVIDENCE/'layout-tests.json').read_text()))==27
for name in ('features-source.log','test-existing.log','layout.log','frozen-features.log'):
    assert 'Traceback' not in (EVIDENCE/name).read_text(encoding='utf-8'),name
before=json.loads((EVIDENCE/'latency-before.json').read_text())
after=json.loads((EVIDENCE/'latency-after.json').read_text())
frozen=json.loads((EVIDENCE/'latency-after-frozen.json').read_text())
for rows in (after,frozen):
    for old,new in zip(before,rows):
        assert old['stream_sha256']==new['stream_sha256'] and old['pixels_sha256']==new['pixels_sha256']
        assert new.get('decompress_calls',0)==0
summary={}
for phase,rows in (('before',before),('after',after),('after_frozen',frozen)):
    summary[phase]={}
    for image in ('cat.jpg','flower.jpg'):
        group=[r for r in rows if r['image']==image]
        keys=('worker_begin','compress','decompress','save','stage','queue_delivery','loaded','add_result','preview_render','displayed')
        stats={key:statistics.median(r.get(key,0) for r in group) for key in keys}
        stats['non_codec']=statistics.median(r['displayed']-r['compress'] for r in group)
        summary[phase][image]=stats
(HERE/'UI_TIMINGS.json').write_text(json.dumps(dict(repetitions=3,units='seconds',aggregation='median per image',
    end_point='Tk event processing and idle repaint completed, followed by DwmFlush; excludes human reaction and initial JIT',
    note='Stage measurements can overlap; preview/thumbnail/PhotoImage costs are nested in callback costs. Shared desktop CPU timings vary.',
    results=summary),indent=2)+'\n')
stage=OUT/'staged-final/PBC3_1_Release'
assert not stage.exists(),'Use a new clean staging folder; do not merge stale files'
shutil.copytree(OUT/'dist-final/PBC',stage,ignore=shutil.ignore_patterns('__pycache__','*.nbc','*.nbi'))
assert sorted(p.name for p in stage.iterdir())==['PBC.exe','Python','README.txt','_internal']
assert [p.name for p in stage.rglob('*.exe')]==['PBC.exe']
with zipfile.ZipFile(ROLLBACK) as old:
    for item in old.infolist():
        relative=Path(*Path(item.filename).parts[1:])
        if 'Python' in relative.parts[:1] or relative.parts[:2]==('_internal','licenses'):
            assert old.read(item)==(stage/relative).read_bytes(),str(relative)
files=sorted(p for p in stage.rglob('*') if p.is_file())
assert not any(p.suffix.lower() in ('.nbc','.nbi','.pbc') for p in files)
assert not any('matplotlib' in str(p.relative_to(stage)).lower() for p in files)
candidate=EVIDENCE/'PBC3_1_Release.zip'
with zipfile.ZipFile(candidate,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
    for path in files:archive.write(path,path.relative_to(stage.parent).as_posix())
with zipfile.ZipFile(candidate) as archive:
    assert archive.testzip() is None
    for item in archive.infolist():
        assert hashlib.sha256(archive.read(item)).hexdigest()==sha(stage.parent/item.filename),item.filename
final=OUT/'final/PBC3_1_Release.zip'
assert sha(final)==sha(ROLLBACK),'Current archive changed since rollback was preserved'
shutil.copy2(candidate,final)
assert sha(final)==sha(candidate)
candidate.unlink()
manifest=dict(zip=str(final),zip_bytes=final.stat().st_size,zip_sha256=sha(final),
    extracted_bytes=sum(p.stat().st_size for p in files),file_count=len(files),
    executable=str(stage/'PBC.exe'),exe_sha256=sha(stage/'PBC.exe'),
    previous_zip=str(ROLLBACK),previous_zip_sha256=sha(ROLLBACK),
    size_change_bytes=final.stat().st_size-ROLLBACK.stat().st_size,
    size_change_percent=100*(final.stat().st_size/ROLLBACK.stat().st_size-1),
    runtime_dependency_changes=[],bundled_extension_additions=[dict(path='_internal/PIL/_imagingft.cp311-win_amd64.pyd',bytes=2010624,component='Pillow FreeType support, collected with ImageDraw')],
    python_library_exact=True,legal_notices_exact=True,
    full_benchmark_rerun=False,codec_sources_unchanged=True)
(EVIDENCE/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
(OUT/'final-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest,indent=2))
