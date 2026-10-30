import hashlib
import json
from pathlib import Path
import platform
import statistics as stats
import subprocess
import sys
import importlib.metadata as metadata
import numpy as np
from PIL import Image,ImageOps
ROOT=Path(__file__).resolve().parents[1];PASS=ROOT/'research_pass28';A=PASS/'artifacts'
load=lambda name:json.loads((A/name).read_text(encoding='utf-8-sig'))
manifest=json.loads((PASS/'PBC32_DISTRIBUTION_MANIFEST.json').read_text());summary=load('summary.json');cli=load('cli-performance.json')['rows'];memory=load('memory.json');compat=load('compatibility.json');permissions=load('permission-validation.json');smoke=load('smoke-final.json');snapshot=load('snapshot-check.json');audit=load('packaging-audit.json');installs=load('install-check.json');ci=json.loads((PASS/'independent-windows.json').read_text(encoding='utf-8-sig'))
assert ci['conclusion']=='success' and all(r['status']=='PASS' for r in ci['environments'])
assert snapshot['archive_sha256']==next(r['sha256'] for r in manifest['artifacts'] if r['kind']=='release_source')
assert all(r['implementation_sha256']==manifest['implementation_sha256'] for r in ci['environments'])
assert smoke['status']=='PASS' and permissions['non_administrator'] and snapshot['wheel_payloads_identical'] and summary['gate_numeric_pass'] and not audit['unresolved_imports']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for name in ('independent-windows.json','accepted-rc1-windows.json'):
    path=PASS/name;data=json.loads(path.read_text(encoding='utf-8-sig'));path.write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8',newline='\n')
source=Path(manifest['source_snapshot_path']);implementation=json.loads((source/'IMPLEMENTATION.json').read_text());original=ROOT/'research_pass27/artifacts/candidate'
changed=[n for n in implementation['files'] if sha(source/n)!=sha(original/n)]
assert sorted(changed)==['pbc3/__init__.py','pbc_cli.py']
protected=subprocess.check_output(['git','ls-remote','github','refs/heads/main','refs/heads/rust','refs/tags/v3.1','refs/heads/research/experiments'],cwd=ROOT,text=True)
hf=subprocess.check_output(['git','ls-remote','origin','refs/heads/main'],cwd=ROOT,text=True)
assert 'd938d003f68063e8df607957d8929c5ba8e74344' in protected and '93e9c0591a710397bb3f09d2d51473313ce23db6' in protected and 'ac7566f8b4fe77020bc27ff720fbe9942040cca4' in protected and '98ade0c8072a4571e9674eaac3d91f861e3b9867' in hf
assert not subprocess.check_output(['git','diff','--name-only','2e0ad08','--','research_pass27','research_pass26','PBC3.py','pbc3_ops.py','pbc3_types.py','pbc3_stream.py','patch_policy.npz','old_pbc'],cwd=ROOT,text=True).strip()
frozen=json.loads((ROOT/'research_pass27/artifacts.json').read_text())
for kind in ('combined','decoder'):assert sha(Path(frozen[kind]['zip']))==frozen[kind]['zip_sha256']
for kind,key in (('decoder_onefile','executable'),('.whl','path'),('.gz','path'),('native_dll','path')):assert sha(Path(frozen[kind][key]))==frozen[kind]['sha256']
evidence_files=['benchmark.json','summary.json','cli-performance.json','memory.json','compatibility.json','runtime-check.json','install-check.json','packaging-audit.json','permission-validation.json','smoke-final.json','snapshot-check.json','pe-imports.json','cli-memory.json']
evidence=dict(version='3.2.0',verdict='READY WITH LIMITATIONS',source_build_commit=manifest['acceptance_build_commit'],implementation_sha256=manifest['implementation_sha256'],source_git_tree=manifest['release_source_git_tree'],metadata_only_runtime_files=changed,performance_campaign_commit='e6cd616361d9ea9a6a552524383f84183d25a622',performance_campaign_version='3.2.0rc1; unchanged codec/native bodies; final frozen command timings rerun at 3.2.0',cpu='Intel Core i9-12900H; logical CPU 12',threads=1,python=sys.version,platform=platform.platform(),dependencies={n:metadata.version(n) for n in ('numpy','Pillow','numba','llvmlite','pyinstaller','pyinstaller-hooks-contrib')},model_sha256=implementation['files']['pbc3/patch_policy.npz'],raw_evidence={n:dict(path=str((A/n).resolve()),bytes=(A/n).stat().st_size,sha256=sha(A/n)) for n in evidence_files},protected_refs=protected.splitlines(),hf_ref=hf.strip(),pass27_artifacts_unchanged=True)
evidence['input_images']=[]
for name in ('cat.jpg','flower.jpg','googlewiki.png','eiffel.png','small','rgba'):
    if name in ('small','rgba'):
        image=Image.fromarray(np.random.default_rng(2702).integers(0,256,(61,79,4 if name=='rgba' else 3),dtype=np.uint8));item=dict(name=name,generator='numpy.default_rng(2702).integers(0,256,(61,79,C),uint8)',channels=4 if name=='rgba' else 3)
    else:
        path=ROOT/'hpt_data'/name;image=ImageOps.exif_transpose(Image.open(path)).convert('RGB');item=dict(name=name,path=str(path.resolve()),file_sha256=sha(path))
    evidence['input_images'].append(item|dict(size=list(image.size),mode=image.mode,pixel_sha256=hashlib.sha256(image.tobytes()).hexdigest()))
evidence['warmup_per_case_variant']=1;evidence['measured_repeats']=3
(PASS/'acceptance-evidence.json').write_text(json.dumps(evidence,indent=2)+'\n',newline='\n')
per_image=[]
for image in ('cat.jpg','flower.jpg','googlewiki.png','eiffel.png','small','rgba'):
    subset=[c for c in summary['cases'] if c['image']==image];v={m:stats.mean(c['median_seconds'][m] for c in subset) for m in ('published','reference','native')}
    per_image.append(f"| {image} | {v['published']:.4f} | {v['reference']:.4f} | {v['native']:.4f} | {100*(1-v['native']/v['reference']):.2f}% |")
startup=[]
for mode in ('published','combined','decoder','onefile'):
    subset=[r for r in cli if r['mode']==mode]
    value=lambda op: f"{stats.median(r['command_seconds'] for r in subset if r['operation']==op):.4f}" if any(r['operation']==op for r in subset) else '-'
    startup.append(f"| {mode} | {value('startup')} | {value('full_decode_command_small')} | {value('full_decode_command_rgba')} |")
cache=[]
for r in cli:
    if r['phase']:cache.append(f"| {r['mode']} / {r['operation'].removeprefix('first_')} | {r['phase']} | {r['command_seconds']:.3f} | {r['result']['import_seconds']:.3f} | {r['result']['codec_seconds']:.3f} |")
warm_decode=[]
for image in ('cat','flower','googlewiki','eiffel','small','rgba'):
    v={r['mode']:stats.median(r['result']['seconds'])*1000 for r in cli if r['operation']=='warm_decode_'+image}
    warm_decode.append(f"| {image} | {v['published']:.3f} | {v['combined']:.3f} | {v['decoder']:.3f} | {v['onefile']:.3f} |")
frozen_encode=[]
for image in ('cat.jpg','flower.jpg'):
    subset=[r for r in cli if r['operation'].startswith('warm_encode_'+image)]
    assert len(set(r['result']['stream_sha256'] for r in subset))==1
    values={}
    for r in subset:
        key='published' if r['mode']=='published' else ('native' if r['operation'].endswith('_native') else 'reference')
        values[key]=stats.median(r['result']['seconds'] if r['mode']=='published' else [v['encode_seconds'] for v in r['result']['rows']])
    frozen_encode.append(f"| {image} | {values['published']:.4f} | {values['reference']:.4f} | {values['native']:.4f} |")
cli_memory=[]
for image in ('cat.jpg','flower.jpg'):
    values={m:stats.mean(r['peak_wset_observed'] for r in load('cli-memory.json')['rows'] if r['image']==image and r['mode']==m)/1048576 for m in ('published','reference','native','decoder')}
    cli_memory.append(f"| {image} | {values['published']:.2f} | {values['reference']:.2f} | {values['native']:.2f} | {values['decoder']:.2f} |")
artifact_rows=['| '+r['kind']+' | '+str(r['bytes'])+' | '+r['sha256']+' | '+r['path']+' |' for r in manifest['artifacts']]
ci_rows=[f"| {r['image']} {r['image_version']} | {r['os']} | {r['python']} | {r['checks']} | {r['status']} |" for r in ci['environments']]
aggregate=summary['aggregate_mean_medians'];native_old=100*(1-aggregate['native']/aggregate['published'])
report=f"""# PBC 3.2.0 release acceptance

Verdict: **READY WITH LIMITATIONS**. Mandatory functional, packaging, independent
Windows and performance gates pass. Final 3.2.0 artifacts are local and await
explicit publication approval. No release/tag/main/HF change was made.

## Independent Windows validation

{ci['environments'][0]['run_url']}
Build/test commit: {manifest['acceptance_build_commit']}.

| Runner image | OS | Build Python | Standalone smoke checks | Result |
| --- | --- | --- | --- | --- |
{chr(10).join(ci_rows)}

Both hosted jobs independently rebuilt all three applications and wheel/sdist,
installed both Python packages in fresh environments, compared all four RGB/RGBA
presets against the actual published 3.1 runtime in the same environment, and
checked known 3.0/3.1 decoding. Runtime PATH was System32 with Python variables
cleared. No binaries were uploaded. Hosted runner tokens are administrator;
separate local tests passed under a non-admin token with actual RX-only app files
and directories and writable cache/output elsewhere.

A genuinely clean consumer PC was **not** tested. Hosted runners contain installed
development tools even when PATH excludes them. VirtualBox had no accessible
Windows guest. The supplied consumer smoke bundle requires built-in PowerShell
5.1/System.Drawing, no Python, compiler, network or administrator access. It checks
archive SHA-256 before extraction, launches every standalone form, verifies actual
PNG dimensions/raw RGB(A) hashes, Unicode, determinism, overwrite protection and
malformed rejection, then writes a readable JSON report. Work files are retained
for diagnosis. No clean-machine certification is claimed.

## Compatibility and final versions

Actual published 3.1 archive: edec3ec73721c3f56913db21f6089a3b6a1637fb3481e61de5b0475da8eeaede
(130098 bytes). Historical 3.0 stream was produced by research_pass1/release in
Pass 27; it is not an ablation. All 32 retained full streams, 64 partials, 45
malformed/resource inputs and 16 concurrent calls pass using the final Numba-free
reader. Stored/LZMA, warmup, PBC3 v0, original dimensions, RGB/YCbCr/RGBA, one-pixel
edges and signed wrapping remain exact. Final local standalone smoke passes
{len(smoke['checks'])} checks; the final directory apps also pass the full saved
stream/permission test. Same-environment deterministic output passes. This does
not promise compressed-file identity on every platform or dependency version.

The six-image all-preset three-way campaign contains 216 measured encode/decode
pairs (24 cases, three repetitions, three variants), with exact bytes, decoded
pixels, bpp and internal/external MSE in every comparison. Final 3.2.0 source,
installed wheel/sdist and frozen apps repeat the relevant comparisons. Native
selection is per call; model, presets, RNG, residual algorithm, entropy and v0
framing are unchanged. The model remains
{evidence['model_sha256']}.

Corrected RC1 first passed both Windows images and all local gates. Only then was
3.2.0 prepared. The only changed runtime files versus Pass 27 are pbc_cli.py
(application version/description and cache namespace) and pbc3/__init__.py
(package version). Codec/native module bodies are identical. README, package
metadata, build names and manifests consistently use 3.2.0; stream v0 is retained.
Accepted RC1 files/evidence remain in artifacts/dist, artifacts/release-source and
artifacts/accepted-rc1. Pass 27 artifacts/source and published 3.1 remain intact.

## Performance acceptance

Windows i9-12900H, logical CPU 12, numerical/Numba threads one. Each full codec
case was warmed once per variant, then three rotated repetitions were measured
in persistent isolated workers. MSE stays enabled. Timing excludes image loading,
hashing and process startup; complete compress/decompress calls are timed.
Raw rows/configs/variation and input conventions are preserved locally and hashed
in acceptance-evidence.json. The complete campaign ran at RC1; stable promotion
changes only the two metadata files above, and final frozen CLI measurements were
rerun. No full historical RD campaign or HF timing was repeated.

| Image | Published 3.1 s | Reference s | Native s | Native reduction vs reference |
| --- | --- | --- | --- | --- |
{chr(10).join(per_image)}

Mean case medians: {aggregate['published']:.6f} / {aggregate['reference']:.6f} /
{aggregate['native']:.6f} seconds. Native benefit is {summary['gain_percent']:.2f}%
versus reference and {native_old:.2f}% versus published 3.1. Native wins 19/24 case
medians and all six image means. Worst slowdown is RGBA Quality, 2.03% versus
reference (about 5 ms, with timing variation); no substantial repeatable
regression was found. No bitrate/quality improvement or general decoder speedup
is claimed. Complete source decode averages remain comparable.

Final 3.2.0 fresh-process medians of three, seconds. OS disk cache was not flushed;
these are process startup/command observations, not cold-disk guarantees. Decode
command includes imports, reading and writing PNG. Published Python uses an
explicit timing shim around the actual released runtime, not the new CLI.

| Runtime | Version startup | Small decode command | RGBA decode command |
| --- | --- | --- | --- |
{chr(10).join(startup)}

Fresh/reused Numba cache uses a fixed 39x37 RGB Quality input, one command per
phase; separate columns prevent confusing JIT/import/process cost with codec time.
The other-CWD phase reuses the same cache. Final permission checks also verify the
default per-user 3.2.0 cache and real .nbc files outside the application directory.

| Runtime/backend | Phase | Full command s | Imports s | Compress s |
| --- | --- | --- | --- | --- |
{chr(10).join(cache)}

Final warmed decode medians of three internal calls, milliseconds. Streams and
image hashes are checked. Source reading and process startup are outside these
internal figures; their full command costs are retained in raw evidence.

| File | Published | Combined | Decoder directory | Decoder one-file |
| --- | --- | --- | --- | --- |
{chr(10).join(warm_decode)}

Two fresh processes per large-image/backend observed peak working set immediately
after complete Quality compression, before hashes. Cat: 434.42 MiB reference to
287.47 MiB native; Flower: 433.59 to 287.04 MiB. This is approximately 147 MiB
saved per process, retaining allocator/import cost. Memory was measured separately
from timings. Startup/JIT remains seconds on first encoding; one-file extraction
adds latency and is optional.

Final frozen Quality encode confirmation (three warmed calls; equal streams):

| Image | Published Python s | Combined reference s | Combined native s |
| --- | --- | --- | --- |
{chr(10).join(frozen_encode)}

Complete large-image commands were also measured separately for memory in two
fresh commands per mode/image. Process-tree memory is observed every 10 ms;
all polling timings are discarded. The process tree includes the Windows venv launcher and its actual Python child.
The table reports the largest observed sum of OS per-process peak working sets;
component peaks may occur at different times, so this is not an exact concurrent
process-tree peak. Concurrent sampled RSS is also retained in the raw rows.
These high-water observations include startup and image I/O,
including PNG output for the decoder, so they are not the isolated metric table
above. Hashes/pixels match in all 16 commands. MiB, mean observations:

| Image | Published encode | Combined reference encode | Combined native encode | Decoder command |
| --- | --- | --- | --- | --- |
{chr(10).join(cli_memory)}

## Packaging, notices and source consistency

Fresh ordinary wheel and sdist installations pass pip check, unrelated-CWD module
origin, all four RGB/RGBA presets, installed model and every declared runtime-file
hash. Both packages have correct 3.2.0 metadata. The pure Python wheel excludes
DLLs/caches; auto falls back to reference. Windows combined contains the optional
C DLL. Decoder contains no Numba/LLVM/model or encoder imports; its build/runtime
environment actually has Numba and llvmlite uninstalled, with no stub.

Combined is {manifest['windows_directories']['combined']['extracted_bytes']} bytes
extracted; decoder is {manifest['windows_directories']['decoder']['extracted_bytes']}
bytes, {100*(1-manifest['windows_directories']['decoder']['extracted_bytes']/manifest['windows_directories']['combined']['extracted_bytes']):.2f}% smaller. Download sizes are in the manifest below.

The PE audit covers {audit['pe_files_audited']} files with no unresolved imports.
An unused Numba tbbpool plugin originally referenced missing tbb12.dll; it is now
excluded. No new kernel or codec behavior was introduced. All affected build,
installed-source, runtime and independent Windows checks were rerun.

LICENSE, README, INSTALL, THIRD_PARTY_NOTICES, complete installed dependency license
trees, Python's Windows conditions, CPython third-party notices, OpenSSL, XZ,
MinGW, LLVM, NumPy/OpenBLAS/GCC terms, Pillow bundled-library notices and
PyInstaller bootloader/runtime-hook terms are retained. Directory notices match
source bytes; one-file embedded notices are verified entry by entry and a
companion notices ZIP is staged. The notice inventory has no unresolved item;
this engineering audit is not a legal opinion. Third-party code is not relicensed
under PBC's MIT license. Source URLs and hashes are retained in licenses metadata.

The exact source ZIP has 65 files, no compiled binaries/caches/datasets and all
model/build inputs. Its extracted directory rebuilt combined/decoder apps without
previous research source or the 3.1 asset, reproduced every wheel payload byte,
and matched all eight RGB/RGBA reference cases with Python/compiler off runtime
PATH. PE bit identity is not promised across build paths/timestamps.

Common implementation manifest: {manifest['implementation_sha256']}.
Exact release-source root Git tree: {manifest['release_source_git_tree']}.
SOURCE_TREE.json lists every input SHA-256. The eventual tag must identify a
commit containing that exact root tree, not the experimental research tip.
See PUBLICATION_PROCEDURE.md. No main merge, release-source commit/tag or upload
was performed. Protected main/rust/v3.1/HF refs and earlier sources are verified
unchanged. A private object store records the tree without a commit or tag.

## Final local artifacts

All paths are absolute. SHA256SUMS.txt covers these files and the distribution
manifest. PBC32_DISTRIBUTION_MANIFEST.json contains byte counts, final versions,
source identity and paths. No artifact is published.

| Kind | Bytes | SHA-256 | Absolute path |
| --- | --- | --- | --- |
{chr(10).join(artifact_rows)}

Recommend combined and decoder directory ZIPs, optional one-file-with-notices ZIP,
Python wheel/sdist and exact source ZIP, with the smoke bundle and checksums.
The raw decoder EXE is also staged. A graphical installer/update system adds no
needed capability for this release. Binaries are unsigned; Windows reputation
prompts may occur. Signing and consumer-PC/other-CPU coverage remain limitations.
The supported Python package target is x64 Python 3.11. Decode bounds remain
32M pixels / 64 MiB input / 128 MiB expanded / 10k patches; larger valid streams
may reject. No Linux/HF native performance claim, ARM build, signing or formal
security certification is made.

PBC 3.2.0 is ready for publication **with these stated limitations**, after explicit
user approval. Final artifacts already identify 3.2.0; no RC filename promotion is
needed. Build/independent test source commit and run are above. The final research
commit/GitHub HEAD containing this evidence is supplied in the chat handoff.
"""
(PASS/'PBC32_RELEASE_ACCEPTANCE.md').write_text(report,encoding='utf-8',newline='\n')
readiness=f"""# PBC 3.2.0 release readiness

**READY WITH LIMITATIONS.** All mandatory acceptance gates pass. Corrected RC1
was accepted before version-only promotion; final artifacts consistently identify
3.2.0 and remain local. Nothing was published, tagged, merged to main or deployed.

- Independent fresh Windows Server 2022 and 2025 hosted runners pass. They retain
  installed development tools; no genuinely clean consumer PC was tested. Local
  non-admin/RX-only app tests pass. Consumer smoke bundle requires no Python/admin.
- All standalone forms run with Python/compiler off PATH; decoder additionally
  runs with Numba/LLVM uninstalled. PBC3 v0 and historical 3.0/3.1 pixels, stored,
  LZMA, warmup, RGBA, partials, determinism and malformed rejection remain correct.
- Compression bytes/quality remain exact. Native retains a 6.96% aggregate encode
  benefit versus reference on 24 cases, about 147 MiB lower large-photo peaks,
  and no serious regression. Decode remains comparable. Default/reused cache and
  final full-command measurements pass; first encoding still compiles kernels.
- Wheel and actual sdist install correctly. Model placement, imports, complete
  notices, embedded one-file notices, source hashes, DLL imports and checksums
  pass. The unused unresolved TBB plugin was removed; no codec feature changed.
- Recommend directory ZIPs; optional one-file-with-notices ZIP; wheel/sdist and
  exact source ZIP. No installer/updater is needed. Executables are unsigned.
- Exact source tree: {manifest['release_source_git_tree']}. Build commit:
  {manifest['acceptance_build_commit']}. Independent run:
  {ci['environments'][0]['run_url']}.

Remaining noncritical limits: no clean consumer-PC/other-CPU or ARM coverage;
unsigned Windows files; Python 3.11-only package target; existing decoder size
bounds; no formal security/legal certification or HF native-speed claim. No
unresolved codec, installation, dependency-import or notice blocker remains.

PBC32_DISTRIBUTION_MANIFEST.json and SHA256SUMS.txt give exact final paths/hashes.
PBC32_RELEASE_ACCEPTANCE.md contains full measurements and all requested answers.
RELEASE_NOTES.md and PUBLICATION_PROCEDURE.md are ready for review. Publication,
release-source/main change and tag creation require separate explicit approval.
"""
(PASS/'PBC32_RELEASE_READINESS.md').write_text(readiness,encoding='utf-8',newline='\n')
extra=['PBC32_DISTRIBUTION_MANIFEST.json','SOURCE_TREE.json','acceptance-evidence.json','independent-windows.json','accepted-rc1-windows.json','PBC32_RELEASE_ACCEPTANCE.md','PBC32_RELEASE_READINESS.md','RELEASE_NOTES.md','PUBLICATION_PROCEDURE.md','REPRODUCE.md']
checksums=''.join(r['sha256']+'  '+r['filename']+'\n' for r in manifest['artifacts'])+''.join(sha(PASS/n)+'  '+n+'\n' for n in extra)
(PASS/'SHA256SUMS.txt').write_text(checksums,newline='\n')
import shutil
for name in extra+['SHA256SUMS.txt']:shutil.copy2(PASS/name,A/'final'/name)
print('ACCEPTANCE_REPORTS_READY')
