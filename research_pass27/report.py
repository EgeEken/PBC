import json
from pathlib import Path
import statistics as st
PASS=Path(__file__).resolve().parent

def read(name):return json.loads((PASS/name).read_text())
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,row))+' |' for row in rows])+'\n'
def write(name,text):
    for a,b in {'Use64':'Use 64','uses96':'uses 96','all32':'all 32','All32':'All 32','remain32':'remain 32','pixels,64MiB':'pixels, 64 MiB','input,128MiB':'input, 128 MiB','body,10k':'body, 10k','Numba0.68':'Numba 0.68','saved3.1':'saved 3.1','frozen3.0':'frozen 3.0','and64':'and 64','including128MiB':'including 128 MiB','logicalCPU12':'logical CPU 12','production3.1':'production 3.1','published3.1':'published 3.1','native:96':'native: 96','comparisons,288':'comparisons, 288','below2^53':'below 2^53','arithmetic.60':'arithmetic. 60','extremes,32':'extremes, 32','GCC13.2':'GCC 13.2','PBC3.2':'PBC 3.2','imports:32':'imports: 32','files,64':'files, 64','partials,45':'partials, 45','passes96':'passes 96','about147MiB':'about 147 MiB','official3.2':'official 3.2','one-file3.2':'one-file 3.2','Pass26':'Pass 26'}.items():text=text.replace(a,b)
    (PASS/name).write_text(text.strip()+'\n',encoding='utf-8',newline='\n')
art=read('artifacts.json');perf=read('packaging/performance.json')['rows'];validation=read('packaging/validation.json');native=read('native/summary.json');memory=read('native/memory.json');decode=read('decoder/benchmark.json');install=read('packaging/installations.json');clean=read('packaging/clean_checkout.json')
size=[]
for kind in ('combined','decoder'):
    a=art[kind];size.append([kind,a['files'],a['extracted_bytes'],a['zip_bytes'],a['exe_sha256']])
a=art['decoder_onefile'];size.append(['decoder one-file',1,a['bytes'],a['bytes'],a['sha256']])
commands=[]
for mode in ('pass26_combined','combined','decoder','decoder_onefile','published_python'):
    values=[]
    for operation in ('version','inspect','decode_command'):
        data=[r['latency_seconds'] for r in perf if r['mode']==mode and r['operation']==operation]
        values.append(f'{st.median(data):.4f}' if data else '-')
    commands.append([mode,*values])
cache=[]
for row in perf:
    if 'phase' in row:cache.append([row['mode'],row['operation'].split('_')[-1],row['phase'],f'{row["latency_seconds"]:.3f}',f'{row["result"]["import_seconds"]:.3f}',f'{row["result"]["codec_seconds"]:.3f}'])
paths='\n'.join('- '+str(v.get('executable',v.get('path',v.get('zip',''))))+'; SHA256 '+str(v.get('exe_sha256',v.get('sha256',''))) for v in art.values())
write('packaging/PBC32_DISTRIBUTION_RESULTS.md',f'''# PBC 3.2.0rc1 distribution results

Combined Windows directory distribution and Python source/wheel are ready for broader testing, not publication. The decoder-only directory and optional one-file application also work. Published 3.1, its tag/ZIP, main, HF and earlier research/staging are preserved.

Baseline: actual published ZIP edec3ec73721c3f56913db21f6089a3b6a1637fb3481e61de5b0475da8eeaede. prepare.py verifies it and generates a complete candidate from tracked overlays plus the unchanged model. Decode-operation bodies are extracted unchanged. Old root PBC 2.3/tag packaging metadata is not used. Paths, exact bytes and SHA256 are in ../artifacts.json and ../provenance.json.

## Packages and clean reproduction

Ordinary `python -m pip install .` and wheel installation pass in separate fresh Python 3.11 environments, including pip check. From unrelated directories, flat PBC3/types imports and pbc3 re-exports resolve installed site-packages; all four presets match the published reference. Model placement/hash are correct: pbc3/patch_policy.npz is setuptools package-data, not prefix-level data-files. README, INSTALL and retained MIT LICENSE are included. Portable Python wheel contains no Windows DLL or JIT caches. NumPy/Numba/Pillow remain core requirements; LLVM is Numba's dependency. Native C source is retained as data. Demo/training/animation stacks are not mandatory.

Clean sparse checkout {clean['commit']} downloaded the pinned asset, reused no existing candidate/model/build directories, rebuilt both applications and a wheel, and stayed clean in Git. Wheel payloads match entry by entry; ZIP timestamps may differ. Both rebuilt apps run from an unrelated directory without Python/compiler on PATH, retaining reference bytes/pixels. This is functional reproducibility, not bit-identical PE builds: paths/timestamps can change executable hashes.

Use64-bit Python 3.11 with requirements-build.txt. Decoder build uses a separate environment with requirements-decoder-build.txt and no Numba/LLVM:

```
python research_pass27/packaging/prepare.py
python research_pass27/native/build.py
uv venv --no-project research_pass27/artifacts/decoder-env --python PATH_TO_PYTHON311
uv pip install --python research_pass27/artifacts/decoder-env/Scripts/python.exe -r research_pass27/requirements-decoder-build.txt
python research_pass27/packaging/build.py --kind combined
python research_pass27/packaging/build.py --kind decoder
python research_pass27/packaging/build.py --kind decoder --onefile
python research_pass27/packaging/install_test.py
python research_pass27/packaging/install_verify.py
python research_pass27/packaging/validate_distribution.py
python research_pass27/packaging/measure.py
python research_pass27/packaging/memory.py
python research_pass27/packaging/clean_checkout.py
```

Restore hpt_data and preserved Pass26 streams for validation. Builds need tracked source and the verified published asset. Exact dependency freezes/toolchains accompany provenance. End users need neither compiler nor separate Python. Keep pbc.exe and _internal together. Overwrite requires --force; stdout/errors are UTF-8, including Unicode missing-file paths.

## Distribution and command costs

'''+table(['Form','Files','Extracted bytes','Download bytes','EXE SHA256'],size)+'''
Fresh-process figures are medians of three calls, not flushed disk-cache startup. Decode command uses96x95 RGB and includes PNG output; inspect uses the same file. Python includes interpreter/imports. Combined and thin apps share the lazy decode path.

'''+table(['Runtime','Version s','Inspect s','Decode command s'],commands)+'''
First-use measurements use the same small Quality input. Complete command, imports and compress-call time are separate:

'''+table(['Runtime','Backend','Phase','Command s','Imports s','Compress s'],cache)+'''
Old frozen Numba ignored the requested NUMBA_CACHE_DIR and used a CWD-derived UserWide cache. Candidate uses a frozen-source locator: shipped source files are found independently of CWD and the requested versioned user cache is populated. Different-CWD commands reuse it. No machine-generated nbc/nbi files are distributed. First-ever compilation still costs seconds. Savings are local observations, not guarantees across processors. Cache validity retains Numba's executable stamp/CPU/code keys. Application relocation/build changes may require compilation. Combined one-file3.2 was not produced; directory mode is intended. One-file decoder has no JIT.

## Permissions and remaining work

Actual non-admin token: app copies have inheritance removed and user RX plus ACL-management rights for cleanup. Directory creation and existing-file writes fail, while read/execute works with cache/output elsewhere. ACLs return to writable afterward. PATH is System32; PYTHONHOME/PYTHONPATH/VIRTUAL_ENV are cleared. Unicode app/input/output names work. Four defaults and RGBA retain reference bytes. All32 files decode through thin app; stored/warmup/historical also pass all three binaries. Invalid/truncated/header/resource inputs fail without traceback. Isolated one-file TEMP leaves no _MEI folders after exit.

VirtualBox is installed but no accessible Windows guest exists. No clean VM with Python physically absent, administrator-owned Program Files installation, other CPU/Windows version, installer/signing or formal security audit. Limits remain32M pixels,64MiB input,128MiB body,10k patches; valid files beyond them reject. Keep Numba0.68 pinned and revalidate locator before cache-API changes. Before official release: clean Windows/CPU coverage, installer/update decision and final licensing review. No release/tag/assets/HF deployment.

## Local artifacts

'''+paths)

core=[]
for name in sorted(set(r['image'] for r in decode)):
    subset=[r for r in decode if r['image']==name]
    core.append([name]+[f'{1000*st.median(r[m]["seconds"] for r in subset):.3f}' for m in ('published','candidate_same_environment','candidate_without_numba')])
reduction=100*(1-art['decoder']['extracted_bytes']/art['combined']['extracted_bytes']);download=100*(1-art['decoder']['zip_bytes']/art['combined']['zip_bytes'])
write('decoder/PBC_DECODER_ONLY_RESULTS.md',f'''# Decoder-only results

Decoder runs with Numba and llvmlite uninstalled, without stubs, model, proposals or learned features. Only NumPy, Pillow and standard-library LZMA are required. Thin directory is {reduction:.2f}% smaller extracted and {download:.2f}% smaller downloaded than combined. Ready for broader testing.

pbc3_decoder shares existing header/grid framing and unchanged palette/signed-bicubic functions. pbc3_types supplies BitReader; unused config/result definitions import no encoder. Combined CLI uses the same light decode/inspect path. Existing pbc3/PBC3 imports retain the full encoder API/dependency. The full encoder wheel still requires Numba; no separately published Python decoder package was made.

Final reader validates prefix/entropy/header, then checks each patch's geometry, bit width and available grid bits before allocation/reconstruction. Each patch is parsed once and applied immediately. Inspect skips validated grids without pixel reconstruction. Full/partial calls still validate all trailing data/padding. Warmup, int16 wrap, float32 interpolation, clipping, YCbCr/RGBA and original-size resize match the published reader. No stream change or JIT stub.

32 files: saved3.1 Q/HQ, all presets, RGB/RGBA, actual frozen3.0 output, stored/LZMA, downsample/warmup, odd/1-pixel dimensions, sign bias and wrap. Full pixels/dimensions/modes, deterministic repeats and64 partials match. 45 malformed/resource cases reject, including128MiB+1 expanded LZMA and oversized sparse file. Sixteen concurrent calls match. Fresh decoder-env asserts Numba/LLVM absent and no encoder imports. Bundles contain no Numba/llvmlite/model and build in that minimal environment.

## Warm core comparison

Three interleaved repetitions in isolated persistent workers, fixed hash seed, logicalCPU12. Middle column uses identical Python/libraries; final column uses actual Numba-free environment. Source reading/hashing is outside timing. Milliseconds; raw rows retain variation.

'''+table(['File','Published Python ms','Thin same env ms','Thin no-Numba env ms'],core)+'''
Two-pass validation had small-image overhead. Checked one-pass parsing removes duplicate work without weakening bounds. Final aggregate latency is broadly comparable; individual cases vary, so no universal speedup. The main win is size/import separation. Frozen command and warm-decode rows are in ../packaging/performance.json; shared lazy imports improve combined startup too. One-file extraction is slower than directory mode.

Sampled Cat process-tree RSS is separate in ../packaging/memory.json; polling latencies are discarded. One-pass sampled observation, not exact OS high-water comparison. Core timings exclude startup; user command timings include imports/PNG I/O. No warm-encode algorithm change.

```
pbc-decode input.pbc output.png
pbc-decode inspect input.pbc
python research_pass27/decoder/make_corpus.py
research_pass27/artifacts/decoder-env/Scripts/python.exe research_pass27/decoder/check.py
python research_pass27/decoder/benchmark.py
```

Keep a separate directory download and optional one-file decoder. Clean Windows/other CPU testing remains. Every local path/size/SHA256 is in ../artifacts.json.
''')
rows=[]
for image in sorted(set(r['image'] for r in native['cases'])):
    subset=[r for r in native['cases'] if r['image']==image]
    a=st.mean(r['median_seconds']['published'] for r in subset);b=st.mean(r['median_seconds']['reference'] for r in subset);c=st.mean(r['median_seconds']['native'] for r in subset)
    rows.append([image,f'{a:.4f}',f'{b:.4f}',f'{c:.4f}',f'{100*(1-c/b):+.2f}%'])
mem=[]
for name in ('cat.jpg','flower.jpg'):
    a=st.median(r['peak_wset'] for r in memory if r['image']==name and r['mode']=='reference')/1048576;b=st.median(r['peak_wset'] for r in memory if r['image']==name and r['mode']=='native')/1048576
    mem.append([name,f'{a:.2f}',f'{b:.2f}',f'{a-b:.2f}'])
pub=native['aggregate_mean_medians']['published'];base=native['aggregate_mean_medians']['reference'];accelerated=native['aggregate_mean_medians']['native'];worst=max(100*(r['median_seconds']['native']/r['median_seconds']['published']-1) for r in native['cases'])
write('native/PBC_NATIVE_INTEGRATION_RESULTS.md',f'''# Native MSE integration

Promote existing C metric to auto selection in the research Windows candidate. Frozen gate passes: {native['gain_percent']:.2f}% aggregate full-encode reduction versus candidate reference, exact streams/pixels/MSE, worst slowdown {-native['worst_case_gain_percent']:.2f}%, modest bundle cost and relocated RX-only execution without a compiler on PATH. No production3.1 default/RD change.

Metric selection is local to compress/compress_stream through mse_backend=reference/native/auto, with no shared monkeypatch. Auto uses optional DLL when present, reference otherwise. Explicit native requires a working DLL; broken dependencies are not silently hidden. Non-byte/unsupported arrays retain reference behavior. Portable Python wheels omit the Windows DLL and work without compiler; Windows app bundles it and uses it by default. Both paths remain available.

32 cases: Cat, Flower, Seagull, Googlewiki, Monet, Eiffel, deterministic small RGB/RGBA; four presets each. One complete warmup/variant/case, three rotated repetitions of published3.1/reference/native:96 three-way comparisons,288 measured full encodes/decodes. Internal MSE enabled. All byte counts, bpp, compressed hash, decoded pixel hash, internal MSE and external source-size RGB MSE match. benchmark.json retains every observation and normalized configs; summary.json includes means/medians/CV/min/max.

'''+table(['Image','Published s','Reference s','Native s','Native reduction'],rows)+f'''
Mean case medians: published {pub:.6f}s, candidate reference {base:.6f}s, native {accelerated:.6f}s. Native is {100*(1-accelerated/pub):.2f}% below published. Worst native-versus-published case is {worst:+.2f}% time. Reference candidate aggregate changes {100*(base/pub-1):+.2f}%; individual cases vary. Monet HQ is the worst native/reference regression. No universal/Cat-only claim, no HF timing, no sum with Pass26 percentages.

Two fresh processes per variant/image, tiny warmup, peak working set immediately after full Quality encode, before hashes:

'''+table(['Image','Reference MiB','Native MiB','Reduction MiB'],mem)+f'''
Imports/allocator retention included. Both repetitions match stream/MSE. The C kernel avoids two full-image float32 copies; PIL byte materialization remains. uint8 sums stay below2^53, matching reference arithmetic.60 type/layout fixtures, signed extremes,32 concurrent metric calls and four mixed-backend concurrent encodes pass. No new kernel.

DLL: {art['native_dll']['bytes']} bytes; SHA256 {art['native_dll']['sha256']}. GCC13.2 C11 -O3/-Wall/-Wextra/-Werror/-shared/-static-libgcc. Dependencies are KERNEL32/UCRT system APIs, no libgcc/MinGW installation. One ctypes crossing/image, int64 shapes/strides, uint64 sum, no native image allocation or mutable native state. Buffers must be legitimate NumPy views, not fabricated out-of-allocation as_strided pointers.

A real relocated non-admin read/execute-only app resolves DLL/model; all four defaults match and Python/GCC are absent from PATH. Added cost is small relative to NumPy/Numba. Default auto is justified for this Windows candidate, not a broader rewrite or official release. Other Windows/CPU coverage remains.

```
python research_pass27/native/build.py
python research_pass27/native/check.py
python research_pass27/native/benchmark.py
python research_pass27/native/memory.py
```
''')
write('PBC_PASS27_SUMMARY.md',f'''# Pass 27 handoff

PBC3.2.0rc1 is a local engineering candidate, not a published release. Compression/model/presets/RNG/entropy/v0 format retain actual published3.1 behavior.

1. Source/wheel pip installs pass: fresh environments, pip check, unrelated CWD, site-packages origin, four reference streams/model SHA. README/INSTALL/MIT license and package-data model placement corrected.
2. Combined directory app is ready for broader tests. UTF-8 errors/Unicode paths, overwrite/malformed behavior, actual RX-only directory/file permissions and user cache pass under non-admin token with Python/compilers off PATH.
3. Frozen Numba cache fallback depended on CWD. Pinned source locator fixes requested/default user cache and different-CWD reuse. No generated caches ship. First-ever encoding still compiles; startup/kernel timings remain separate.
4. Decoder runs without installed Numba/LLVM or stubs/model/encoder imports:32 full files,64 partials,45 invalid/resource cases and concurrency pass. Checked one-pass parsing avoids redundant validation. Directory is {reduction:.1f}% smaller extracted than combined; optional one-file decoder works.
5. Native per-call backend passes96 three-way comparisons: {native['gain_percent']:.2f}% aggregate reduction versus candidate reference, exact output/MSE, no serious regression. Auto native when DLL present (Windows app); portable wheel uses reference without DLL. Repeated Cat/Flower peaks save about147MiB. No global patch/new kernel.
6. Clean source checkout regenerates both apps and identical wheel payloads without existing artifacts. Functional reproducibility, not promised bit-identical executable hashes. Source revision is in packaging/clean_checkout.json.
7. No accessible Windows VM. Before official3.2: clean Windows/CPU coverage, installer/update/signing decision, final licensing review and real-user testing. No GUI/installer, HF/native Linux speed claim, release tag/upload.

Reports: packaging/PBC32_DISTRIBUTION_RESULTS.md, decoder/PBC_DECODER_ONLY_RESULTS.md, native/PBC_NATIVE_INTEGRATION_RESULTS.md. artifacts.json lists exact executable/ZIP/wheel/sdist/DLL paths/bytes/SHA256. provenance.json records frozen sources, inputs/model, versions/threads and preserved Pass26/staging. Build dependencies are pinned separately for complete and Numba-free environments.

All new work stays under research_pass27. Binaries/images/streams/clones/caches/logs are ignored. Final commits/GitHub HEAD and URLs are supplied in the chat. Main/rust/v3.1, release ZIP, HF, earlier code/research/staging stay unchanged. No Video/4.0 or unrelated kernel/RD work.
''')
print('Pass27 reports generated')
