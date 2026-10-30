import ctypes as ct
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import zipfile
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
OUT=HERE/'artifacts'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
summary=json.loads((OUT/'summary.json').read_text())
assert summary['measured_pairs']==900 and summary['exact_cases']==100
for phase in ('first-use','cached'):
    result=json.loads((OUT/('frozen-'+phase+'.json')).read_text())
    assert len(result['checks'])==8 and result['known_streams']==32 and result['partial_checks']==64 and result['invalid_rejected']==45
    assert result['cached_files']>0
assert json.loads((OUT/'frozen-payload-equivalence.json').read_text())['gui_entry_bytecode_exact']
expected=json.loads((OUT/'probe-candidate/validation.json').read_text())['rows']
for mode in ('source','installed'):
    assert json.loads((OUT/('probe-'+mode)/'validation.json').read_text())['rows']==expected
for p in (HERE/'source').glob('*.py'):
    if p.name not in ('pbc_cli.py','setup.py'):
        assert p.read_bytes()==(OUT/'sources/candidate'/p.name).read_bytes(),p.name
frozen=ROOT/'research_pass28/artifacts/final-source'
for name,digest in json.loads((ROOT/'research_pass28/SOURCE_TREE.json').read_text())['files'].items():
    assert sha(frozen/name)==digest
stage=OUT/'staged/PBC3_1_Release'
shutil.copytree(OUT/'dist/PBC',stage,ignore=shutil.ignore_patterns('__pycache__','*.nbc','*.nbi'),dirs_exist_ok=True)
assert sorted(p.name for p in stage.iterdir())==['PBC.exe','Python','README.txt','_internal']
assert [p.name for p in stage.rglob('*.exe')]==['PBC.exe']
assert not any(p.suffix in ('.nbc','.nbi') for p in stage.rglob('*'))
assert not (stage/'Python/requirements.txt').exists()
assert not (stage/'Python/PBC3_animation.py').exists()
assert sha(stage/'_internal/pbc3/mse.dll')==sha(frozen/'pbc3/mse.dll')
assert sha(stage/'Python/pbc3/mse.dll')==sha(frozen/'pbc3/mse.dll')
archive=OUT/'final/PBC3_1_Release.zip'
archive.parent.mkdir(exist_ok=True)
files=sorted(p for p in stage.rglob('*') if p.is_file())
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
    for path in files:z.write(path,str(path.relative_to(stage.parent)).replace('\\','/'))
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    assert len(z.infolist())==len(files)
    for info in z.infolist():
        path=stage.parent/Path(info.filename)
        assert hashlib.sha256(z.read(info)).hexdigest()==sha(path),info.filename
source_hashes={str(p.relative_to(HERE/'source')).replace('\\','/'):sha(p) for p in (HERE/'source').rglob('*')
    if p.is_file() and '__pycache__' not in p.parts and not any(n in p.parts for n in ('build','pbc_codec.egg-info'))}
(HERE/'library-source.json').write_text(json.dumps(source_hashes,indent=2)+'\n')
evidence={str(p.relative_to(OUT)).replace('\\','/'):dict(bytes=p.stat().st_size,sha256=sha(p))
    for p in OUT.glob('*.json') if p.name not in ('final-manifest.json','evidence.json')}
evidence['measurements.jsonl']=dict(bytes=(OUT/'measurements.jsonl').stat().st_size,sha256=sha(OUT/'measurements.jsonl'))
(OUT/'evidence.json').write_text(json.dumps(evidence,indent=2)+'\n')
manifest=dict(version='3.1.0',branding='PBC V3.1',zip=str(archive),zip_bytes=archive.stat().st_size,zip_sha256=sha(archive),
    directory=str(stage),executable=str(stage/'PBC.exe'),exe_sha256=sha(stage/'PBC.exe'),files=len(files),
    extracted_bytes=sum(p.stat().st_size for p in files),native_dll_sha256=sha(stage/'_internal/pbc3/mse.dll'),
    non_admin=not bool(ct.windll.shell32.IsUserAnAdmin()),head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
(OUT/'final-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
overall=summary['overall']
enc=overall['encode_seconds']['comparisons']
dec=overall['decode_seconds']['comparisons']
(HERE/'RELEASE_NOTES.md').write_text(
f"""PBC V3.1 update: {enc['candidate_vs_original']['reduction_percent']:.1f}% less encode time and {dec['candidate_vs_original']['reduction_percent']:.1f}% less decode time than V3.0 in a local 25-image, four-preset benchmark. Encoding used {enc['candidate_vs_published']['reduction_percent']:.1f}% less time than the previous V3.1 download. All tested files and decoded pixels were unchanged.

Native C MSE optimization and a standalone Windows GUI with Encode, Decode and Registry. One download includes the Python library.

Desktop: extract and open PBC.exe. Python 3.11: run python -m pip install ./Python; usage is in Python/README.md.

[Hugging Face demo](https://huggingface.co/spaces/EgeEken/PBC)
""",encoding='utf-8',newline='\n')
focus=json.loads((OUT/'focused-frozen.json').read_text())['medians']
lines=['# V3.1 local release validation','',
'Prepared locally; nothing committed, pushed, published or deployed. Research HEAD remains '+manifest['head']+'.',
'',
'Frozen sources: original 3.0 commit 2e59f1b06980aa467da0cde94fa33c3f9a0d4d51 (research_pass1/release; source bytes verified against Git); actual published 3.1 ZIP edec3ec73721c3f56913db21f6089a3b6a1637fb3481e61de5b0475da8eeaede (130,098 bytes); final Pass 28 source tree fd077ba70d6855e043721953d77a65ed43bfd5ea. Pass 28 ZIP SHA-256 34d338686bc3bb62e65f044dacfff0cc155d16e1cdda63a7bfa0f0584021fd53. Original artifacts are unchanged. provenance.json records every source/input hash and installed version.',
'',
'Benchmark: all 25 manifest-hashed images × four presets × three implementations × three measured repetitions: 900 complete encode/decode pairs, plus 300 warmup pairs. Persistent source-isolated workers, preloaded EXIF-transposed RGB images, numerical/Numba threads one, i9-12900H logical CPU 10 (verified SMT P-core). CPU 12 in earlier reports is an E-core. Python 3.11.16, NumPy 2.4.6, Numba 0.68.0, Pillow 11.3.0. Full-call timers retain final MSE and entropy coding. Each case contributes its median of three equally to aggregates. No build/test workload was launched during measurement.',
'',
'| Comparison | Encode time reduction | Decode time reduction |',
'| --- | ---: | ---: |']
for key,label in [('published_vs_original','Published 3.1 / original 3.0'),('candidate_vs_published','Optimized / published 3.1'),('candidate_vs_original','Optimized / original 3.0')]:
    lines.append(f"| {label} | {enc[key]['reduction_percent']:.2f}% | {dec[key]['reduction_percent']:.2f}% |")
lines+=['',
'All 100 cases have exact three-way streams, decoded pixels, dimensions/mode, size, bpp, patch count and internal/external MSE. All 300 implementation/cases repeat deterministically. Mean case-median encode times: 0.481247 / 0.316292 / 0.283785 s; decode: 0.104033 / 0.048461 / 0.048281 s. Versus published 3.1, encode wins 86/100 cases (worst −4.95%); decode wins 47/100 (worst −11.32%). Versus 3.0, encode wins 100/100 and decode 99/100 (worst −0.83%). The additional 0.37% decoder change is treated as neutral. No quality or bitrate improvement is claimed. BENCHMARK.md contains per-preset/per-image results, variability and outliers; artifacts/summary.json retains all case medians and dataset aggregates.',
'',
'Validation: candidate/source/clean installed package match on 17 functional cases, including all four presets on RGB/RGBA and 1×7 / 317×5 images, plus a manual config. Fresh non-editable installation, pip check and isolated imports from an unrelated CWD pass; model and native DLL resolve in site-packages. All codec module bodies match Pass 28; only CLI/package version metadata changes. The pbc3 package remains a supported re-export and resource location. Animation/demo utilities are absent. The internal wheel is correctly tagged CPython 3.11 / Windows x64.',
'',
'GUI: Tkinter/ttk, exactly Encode | Decode | Registry; Quality default, validated Auto/Semi-Auto/Manual settings, background worker and indeterminate activity, direct PIL previews, PBC/PNG saves and 30-entry session history. Source tests pass Unicode/spaces, manual overrides, repeat operations, registry eviction/reopen, export, source-overwrite rejection and native Save-dialog confirmation request. Default/minimum-size and Manual layouts keep action controls visible. Oversized image headers reject before allocation. GUI file input uses the existing bounded pbc3_decoder, while the Python PBC3.decompress API remains unchanged.',
'',
'Frozen validation: separate internal test executable runs the same GUI/codec bytecode, DLL and model as production (verify_payload.py asserts equivalence). All eight RGB/RGBA preset outputs match the candidate, 32 retained streams and 64 partials pass, and 45 malformed/resource checks reject. First-use and cached launches with PATH=C:\\Windows\\System32 pass; JIT files are outside the app. Source tests cover the registry limit and parameter errors. Production GUI startup, RGB/RGBA Quality encoding, native status and reconstructed data were also observed directly; footer layout was visually checked. These checks are on this Windows 11 Home 26200 consumer machine with developer tools installed, '+('under a non-admin token.' if manifest['non_admin'] else 'under the current user token.'),
'',
'Focused frozen-runtime check: two full-size images, Quality, one warmup and three interleaved repetitions against the candidate, CPU 10 / one numerical thread, exact streams/pixels. No encode/decode regression was observed. Median seconds (candidate → frozen):']
for image,data in focus.items():
    lines.append(f"- {image}: encode {data['encode_seconds']['candidate']:.6f} → {data['encode_seconds']['frozen']:.6f}; decode {data['decode_seconds']['candidate']:.6f} → {data['decode_seconds']['frozen']:.6f}.")
lines+=['',
f"One public download: {archive}. {manifest['zip_bytes']:,} bytes ({manifest['zip_bytes']/1_000_000:.2f} MB / {manifest['zip_bytes']/2**20:.2f} MiB); {manifest['extracted_bytes']:,} bytes extracted across {manifest['files']} files. ZIP CRC and every member SHA-256 match staging. Top level: PBC.exe, _internal/, Python/, README.txt. One executable; runtime dependencies are bundled once. No research, benchmarks, smoke suite, separate decoder, wheel, source ZIP, installer or checksum files are in the download. Licenses, including Tcl/Tk terms, are retained unobtrusively.",
'',
'| Artifact | Absolute path | SHA-256 |','| --- | --- | --- |',
f"| ZIP | {archive} | {manifest['zip_sha256']} |",
f"| Executable | {stage/'PBC.exe'} | {manifest['exe_sha256']} |",
f"| Native DLL | {stage/'_internal/pbc3/mse.dll'} | {manifest['native_dll_sha256']} |",
'',
'Source/build scripts and concise reports are under '+str(HERE)+'. Raw rows, warmups, worker origins/configs, source copies, probe results, focused timing, installed environments and internal builds remain ignored under artifacts/. artifacts/evidence.json records evidence hashes; artifacts/final-manifest.json records final paths/sizes/hashes.',
'',
'Limitations: unsigned Windows x64 build; tested locally on Windows 11 with Python/compiler off PATH, not on a fresh OS, Windows 10 or another consumer CPU. No HF/native Linux performance claim. Drag-and-drop is omitted to avoid another GUI dependency. First-use JIT remains noticeable; session history is not persistent. Decode bounds are 32M pixels / 64 MiB input / 128 MiB expanded / 10k patches. Use the bounded reader for untrusted streams; the preserved legacy PBC3.decompress API is not newly hardened. No codec or bitstream change was made.',
'',
'Publication decisions require explicit approval: a source commit/target branch, the v3.1 tag/source relationship, release-asset replacement and any HF deployment. Options are to preserve the historical tag and link the exact updated source commit explicitly, or deliberately retarget it after review. Neither was performed. The complete research tip must not become release source by accident; the prepared library/GUI/build inputs are the reviewable source.']
relocation=OUT/'relocation-process.json'
if relocation.exists():
    record=json.loads(relocation.read_text())
    if record.get('zip_sha256')==manifest['zip_sha256'] and record.get('startup_observed'):
        lines=[line.replace('footer layout was visually checked.',
            'footer layout was visually checked. The exact final ZIP was extracted to a Unicode directory, launched from an unrelated CWD with Python off PATH, and its Registry export controls were visually confirmed.') for line in lines]
(HERE/'VALIDATION.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(manifest,indent=2))
