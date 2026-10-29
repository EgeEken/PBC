import json
from pathlib import Path
import statistics as stats
PASS=Path(__file__).resolve().parent

def read(name):
    return json.loads((PASS/name).read_text())

def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])+'\n'

def write(name,text):
    (PASS/name).write_text(text.strip()+'\n',encoding='utf-8',newline='\n')

pack_memory=read('packaging/memory.json')
pack=read('packaging/measurements.json'); perf=read('packaging/performance.json'); native=read('native/benchmark.json'); memory=read('native/memory.json'); video=read('video/results.json'); compared=read('architecture/comparisons.json'); diagnostics=read('architecture/diagnostics.json')
pack_peak_text='/'.join(format(r['sampled_peak_process_tree_rss_bytes']/1048576,'.1f') for r in pack_memory['rows'])
size_table=table(['Form','Bytes','Files','Executable SHA-256'],[(k,v['total_bytes'],v['files'],v['exe_sha256']) for k,v in pack['distributions'].items()])
startup=[];warm=[]
for mode in ('python','onedir','onefile'):
    subset=[r for r in perf['rows'] if r['mode']==mode]
    version=[r['latency_seconds'] for r in subset if r['operation']=='version']
    first=next(r for r in subset if r['operation']=='first_use_fresh_cache');second=next(r for r in subset if r['operation']=='second_process_same_cache')
    startup.append([mode,f'{stats.median(version):.3f}',f'{first["latency_seconds"]:.3f}',f'{first["result"]["import_seconds"]:.3f}',f'{first["result"]["codec_seconds"]:.3f}',f'{second["latency_seconds"]:.3f}'])
    for row in subset:
        if row['operation']!='warmed_codec': continue
        vals=row['result']['rows']
        warm.append([mode,row['image'],f'{stats.median(v["encode_seconds"] for v in vals):.4f}',f'{stats.median(v["decode_seconds"] for v in vals):.4f}',f'{min(v["encode_seconds"] for v in vals):.4f}-{max(v["encode_seconds"] for v in vals):.4f}',f'{row["latency_seconds"]:.3f}'])
write('packaging/PBC32_PACKAGING_RESULTS.md',f'''# PBC 3.2 packaging prototype

Both directory and one-file Windows x64 executables work. The codec is the actual published PBC 3.1 ZIP, not the research root or release tag source. Nothing was published.

## Source reconciliation

The live v3.1 tag is ac7566f8b4fe77020bc27ff720fbe9942040cca4; GitHub main was already 17e569049c1b84004540ae89317a2ccdabf89832 at entry. The [attached release ZIP](https://github.com/EgeEken/PBC/releases/tag/v3.1) is 130,098 bytes, SHA-256 edec3ec73721c3f56913db21f6089a3b6a1637fb3481e61de5b0475da8eeaede. Its runtime files match preserved Pass 16 GitHub_Source byte for byte. The original Pass 16 archive hash 08855cb3... identifies a different, fuller ZIP; do not substitute it silently.

The tag still has PBC 2.3.1/Python 3.12 packaging and lacks the split stream/trace modules. The published ZIP has Python 3.11, pbc-codec 3.1.0, the thin pbc3 re-export, corrected dependencies and module-relative model lookup. Compared with research root, PBC3.py has release labeling and pbc3_types.py changes model discovery; numerical/runtime optimization files and model agree. ../baseline.json records comparisons and all preserved staging hashes. The attached ZIP omits README.md/LICENSE although its pyproject references a README and declares MIT. Its wheel data-file layout was not revalidated; this prototype imports the frozen flat runtime and bundles model data explicitly.

## Use and build

See USAGE.txt. Balanced is default; compression, quality and high_quality are supported. Spaces/Unicode, RGB/RGBA, EXIF normalization, ordinary v0 streams, dimensions and model/RNG/presets are retained. Replacement requires --force. Errors return 1, argument errors 2. Inspect reports stored metadata, not inferred preset/MSE. Public API is unchanged.

Use 64-bit Python 3.11 and `python -m pip install -r research_pass26/requirements.txt`. From the research checkout:

```
python research_pass26/packaging/freeze.py
python research_pass26/packaging/build.py
python research_pass26/packaging/build.py --onefile
python research_pass26/packaging/package_directory.py
python research_pass26/packaging/test_distribution.py
python research_pass26/packaging/compatibility.py
python research_pass26/packaging/measure.py
python research_pass26/packaging/memory_measure.py
python research_pass26/packaging/decoder_dependency_check.py
```

Freeze downloads only the pinned release and rejects a digest mismatch. Build uses PyInstaller 6.16.0/hooks 2026.9, not Nuitka. GCC 13.2 and FFmpeg 6.1.1 are available; Rust/MSVC were not on PATH and Nuitka was absent from the validated environment. No alternative toolchain was installed. The custom hook collects CPU Numba dynamic imports and excludes tests, CUDA, pycc, SciPy, setuptools and demo packages. Source .py files accompany frozen modules for the Numba cache locator; model data lives beside them. Python, NumPy, Numba/llvmlite, Pillow, LZMA and Windows runtime DLLs are bundled.

{size_table}

Local paths:

'''+ '\n'.join(f'- `{v["executable"]}`' for v in pack['distributions'].values())+f'''
- Directory ZIP: `{PASS/'artifacts/PBC32Research-Windows-x64.zip'}`.
- Independent relocation root: `{pack['portable_directory']}`.

Keep pbc.exe and _internal together in directory mode. Both forms were copied outside the checkout and launched from another directory without virtualenv activation, with PATH restricted to Windows/System32. This is not a clean Windows VM test with Python uninstalled. Tests run as the current user; locked-down installations, alternate Windows versions and antivirus/signing behavior remain untested.

## Timings

Pinned logical P-core 12, one numerical thread. Fresh-process startup is not a flushed OS/disk-cache experiment. First encode uses a new Numba cache. Final measurements use blocking subprocess waits: no memory polling steals measured CPU. Earlier sampled diagnostics are not used as performance claims.

'''+table(['Runtime','Version median s (3)','First encode command s','Imports s','First codec s','Second command s'],startup)+'''
First/second command encode the same 96x95 RGB Quality fixture. First use includes JIT compilation/loading; the second process reuses that runtime cache. Additional presets/RGBA may need signatures. Frozen execution defaults to writable per-user caching. One-file extraction/imports repeat and are materially slower; directory mode is preferred for interactive use.

'''+table(['Runtime','Image','Warm encode median s','Warm decode median s','Encode range s','Whole benchmark command s'],warm)+f'''
Complete PBC3.compress/decompress calls, three in-process warmed repetitions, internal MSE/LZMA enabled. Whole benchmark command adds imports, initial warmup and all repetitions; it is not one encode latency. Case/runtime order is partially rotated, not a persistent interleaved trial. Small runtime differences are diagnostic, not a claim that bundling accelerates kernels. User decode additionally preflights the stream and writes PNG; command latencies remain in measurements.json. Streams/pixels match the reference exactly.

Separate sampled Cat benchmark process-tree RSS peaks are recorded in memory.json (initial encode plus one warm encode/decode). Python/directory/one-file are about {pack_peak_text} MiB. Sampling is 50 ms, sums bootloader/worker where present, and those latencies are discarded; this is a one-pass RSS observation, not an exact OS peak or a kernel-memory comparison.

## Dependencies and readiness

Core encoder: NumPy, Numba, llvmlite, Pillow and standard-library LZMA. No pandas, Torch, OpenCV, SciPy, Optuna, Gradio, FastAPI or imageio is needed. Benchmark tooling uses psutil; bundling uses PyInstaller. Animation optionally uses imageio/imageio-ffmpeg. Training/demo dependencies stay outside.

Seven files across all presets/RGBA plus stored/warmup decode with every JIT function replaced by an assertion-raising stub: no Numba kernels execute on those decoder paths. Shared encoder imports still require Numba. A source-level decoder split could remove Numba/LLVM from a decoder-only app; the combined app still needs them. No stub/removal is shipped. NumPy implements bit grids/palettes/canvas; Pillow supplies exact signed interpolation, color conversion and resizing as well as I/O. Eliminating either needs a measured implementation.

Both final binaries pass four RGB presets plus Quality RGBA, reference stream/pixel equality, inspect, overwrite refusal, missing input and seven malformed files. Additional stored-entropy and warmup v0 streams decode identically in Python and both executables. Corrupt LZMA becomes a normal error without traceback. Bodies/dimensions/patch geometry are bounded before allocation. Research limits (32M pixels, 64 MiB file, 128 MiB body, 10k patches) may reject legitimate unusually large files and are documented; this is not a general hostile-file security audit.

Before official 3.2: test clean supported Windows, finalize wheel/model layout and license notices, decide installer/directory versus one-file UX, and verify cache/permission/update behavior. No GUI, installer, signing, release tag or upload. Binaries/caches are ignored; build/test scripts and compact evidence are committed.
''')

cases=[];base=[];cand=[]
for key in sorted(set((r['image'],r['preset']) for r in native['rows'])):
    subset=[r for r in native['rows'] if (r['image'],r['preset'])==key]
    a=stats.median(r['reference']['encode_seconds'] for r in subset);b=stats.median(r['native']['encode_seconds'] for r in subset)
    da=stats.median(r['reference']['decode_seconds'] for r in subset);db=stats.median(r['native']['decode_seconds'] for r in subset)
    cases.append([*key,f'{a:.4f}',f'{b:.4f}',f'{100*(1-b/a):+.2f}%',f'{da:.4f}',f'{db:.4f}'])
    if key[0]!='rgba':base.append(a);cand.append(b)
gain=100*(1-stats.mean(cand)/stats.mean(base))
component=[]
for row in native['micro']:
    a=stats.median(row['seconds']['reference']);b=stats.median(row['seconds']['native'])
    component.append([row['image'],row['input'],f'{1000*a:.3f}',f'{1000*b:.3f}',f'{a/b:.2f}x'])
write('native/PBC_NATIVE_KERNEL_RESULTS.md',f'''# Native final-MSE experiment

A compiled C metric preserves every tested stream, decoded pixel and internal MSE. Across the requested three images x two presets, mean case-median encoding time falls {gain:.2f}%. Cat improves about 10%; Eiffel does not improve. Preserve an optional path, not a mandatory dependency.

## Profile and mechanism

The actual published ZIP is verified through ../baseline.json. A warmed separate TimingTrace compress/decompress was run for Cat, Googlewiki and Eiffel, Q/HQ, pinned P-core 12 and one numerical thread. profile.json contains inclusive/exclusive phases; instrumented time is diagnostic, not the speed trial.

Cat final_mse takes 145-196 ms and final resize 88-89 ms; proposal sampling takes 77-78 ms. Googlewiki Q sampling is 182 ms, HQ 78 ms; HQ entropy is 246 ms. Eiffel sampling is 65-75 ms; HQ entropy is 201 ms. Learned feature extraction is now about 20-50 ms here, rather than an assumed historical constant. Decoder is mostly final resize for Cat (96-99 ms) and patch read/resample for smaller images. Python rectangle bookkeeping matters, but a native rewrite must preserve conditional NumPy RNG calls. Native interpolation must reproduce Pillow semantics. Neither was attempted in this bounded pass.

Selected candidate: final RGB/RGBA MSE. The release creates two full-image float32 arrays, then performs a compiled Numba sum. This C experiment removes those float arrays and reads uint8 pixels directly. NumPy PIL-to-byte materialization remains. Squared byte differences are integers; the supported sum is below 2^53, so the integer sum/final double division equals reference double accumulation exactly.

mse.c is C11, built by MinGW-w64 GCC 13.2.0 (x86_64 UCRT) with -O3/-Wall/-Wextra/-Werror/-shared/-static-libgcc. kernel.py crosses ctypes once per image, releases the GIL, passes explicit int64 byte strides/shapes and owns array references for the call. It supports reversed/F/strided layouts without copying NumPy pixels. No mutable native state. Empty/mismatched/oversized/unsupported arrays reject; non-uint8 images use the reference through the research switch. The process-local switch is used only in isolated workers, never installed in the server.

## Correctness and timing

28 layout/shape fixtures, 100 deterministic random fixtures, zero/255 endpoints, 32 concurrent read-only calls, six invalid shapes/dtypes and three signed/float fallback cases pass. Signed extremes reject in the uint8-only kernel and reproduce reference results through fallback. Normal/small/degenerate shapes, RGB/RGBA, negative strides and repeated calls are covered. C checks counts; callers must supply legitimate NumPy views, not fabricated out-of-allocation as_strided pointers.

'''+table(['Image','Input','Reference ms','C binding ms','Speedup'],component)+'''
Five rotated component repetitions. Both include conversion/call overhead: array rows include reference float copies and ctypes wrapping; PIL rows also include byte materialization. No computation-only timer replaces a full call.

'''+table(['Image','Preset','Reference encode s','Native encode s','Encode reduction','Reference decode s','Native decode s'],cases)+f'''
Separate source-isolated persistent workers, one full warmup per case, three interleaved repetitions, actual release/config/model and internal MSE enabled. Six main cases plus RGBA give 21 paired measurements. Bytes, pixel hashes, internal MSE and independent RGB MSE all match. benchmark.json retains repetitions and complete configs. Decoder is unchanged; small decode differences are noise. Eiffel/RGBA regressions remain visible.

Fresh-process Cat Quality peak working set: {memory[0]['peak_wset']/1048576:.2f} MiB reference, {memory[1]['peak_wset']/1048576:.2f} MiB native. Both begin the full encode at {memory[0]['before_rss']/1048576:.2f} MiB RSS after a tiny warmup; stream/MSE match. One isolated observation per variant, including imports/allocator retention, not a repeated memory claim.

## Reproduction and decision

```
python research_pass26/packaging/freeze.py
python research_pass26/native/profile.py
python research_pass26/native/build.py
python research_pass26/native/check.py
python research_pass26/native/benchmark.py
python research_pass26/native/memory.py
```

Use pinned requirements and restored hpt_data. The DLL is ignored artifacts/mse.dll. Production Python/Numba remains intact. Direct uint8 MSE is plausible as an optional photographic optimization with modest binding complexity and lower peak memory. It removes no encoder dependency. Confirm HF/broader sizes before integration. A wider native rewrite is not warranted: much decode work is already Pillow, sampling requires RNG-preserving redesign and entropy is native LZMA. Do not add this gain to earlier pass percentages.
''')

video_rows=[]
for row in video:
    video_rows.append([row['clip'],row['mode'],row['bytes'],f'{row["bpp_per_frame"]:.4f}',f'{row["mean_mse"]:.3f}',f'{row["worst_mse"]:.3f}',f'{row["encode_fps"]:.1f}',f'{row["decode_fps"]:.1f}'])
write('video/PBC_VIDEO_FEASIBILITY.md','''# PBC Video feasibility

The independently decodable container works. This temporal representation fails rate efficiency on every clip; preserve it as a negative result, not an effective video codec.

container.py implements P26V v1: 17-byte little-endian <4sB6H> magic/version, width, height, frame count, frame-rate numerator/denominator and key interval. Each frame has <BII> type, complete payload length and CRC32 (9 bytes), followed by normal PBC v0 (type 0), zlib signed residual (1) or PNG reference keyframe (2). Source frames/encoder state/model are not needed at decode. Limits: 64 MiB file, 512 frames, 2,073,600 pixels/frame and 32M total frame pixels. CRC/length/type/shape/trailing checks and bounded inflation apply.

Three deterministic 16-frame 256x144 RGB clips at 12 fps: repeated resized Flower, a 32x40 colored rectangle moving 9 px/frame on a fixed gradient, and a moving 256x144 crop of resized Cat (2 px horizontal/1 px vertical per frame). The photo pan is synthetic camera motion, not captured natural footage. Generated inputs/files remain in ignored ../artifacts/video.

Intra: every frame is ordinary Balanced PBC. Temporal: PBC keyframes at 0/8; subsequent prediction is the previous independently decoded frame, with little-endian int16 RGB differences [-255,255] compressed by zlib level 6. Residuals are lossless, not biased eight-bit images. Encoder reconstructs from the decoded reference and stored delta before advancing. A fully lossless PNG-keyframe/delta reference uses the same prediction/interval/container. H.264 is installed FFmpeg 6.1.1 libx264, CRF 28, veryfast, yuv420p, one encoder thread, complete MP4, no audio.

'''+table(['Clip','Mode','File bytes','bpp/frame','Mean RGB MSE','Worst MSE','Encode fps','Decode fps'],video_rows)+'''
All metrics score every independently decoded frame and complete bytes. PBC timing includes real encode/decode and predictive reference reconstruction; FFmpeg includes process startup. One warmed feasibility pass per mode/clip, not a repeated speed benchmark. results.json retains sources/file/frame hashes, frame MSE, seconds, interval, payload ledger and residual fraction. Peak memory was not isolated for these small clips.

Temporal/intra size ratios are about 2.01x static, 5.11x moving shape and 19.70x photo pan. The first predicted frame after each lossy key must transmit a dense correction to make prediction lossless. Later static deltas become zero, but correction frames outweigh saved keys. Uncompensated motion makes photo deltas large. PNG keys can be cheaper than small lossy keys followed by lossless repairs, explaining the smaller fully lossless static/moving-shape references.

Temporal keys retain PBC distortion; every predicted frame is source-exact. Mean MSE is therefore much lower than intra at far greater rate, not an equal-quality comparison. Worst errors occur at keys and final-frame MSE is zero for temporal/lossless. No accumulating encoder/decoder drift: both advance from the same decoded reference. H.264 gives much smaller files, faster encode throughput and lower MSE than intra on all clips. Temporal has lower average MSE than H.264 at vastly more bytes; no equal-rate claim.

Nine files decode in a fresh process with PBC3.compress/model constructor forbidden and reproduce all frame hashes. Twenty malformed/header/CRC/entropy/type/range/truncation cases reject; a [-255,+255,-128] endpoint fixture reconstructs exactly. MP4 files independently decode all 16 frames. No original/cached residual is used by the decoder.

```
python research_pass26/video/experiment.py
python research_pass26/video/validate.py
```

Stop this formulation. A future project needs jointly lossy temporal residual coding with explicit quantization/RD, skip blocks and at least global/block motion; otherwise lossy-key repairs and motion dominate. This is a new research project, not a 3.2 feature or promised H.264 advantage. No interval/mapping rescue, second representation, long footage, GPU, audio or neural flow was attempted.
''')

avif=[]
for target in compared:
    best=min(target['codecs']['AVIF']['dominating'],key=lambda p:p['mse'])
    avif.append([target['image'],target['preset'],target['pbc_bytes'],f'{target["pbc_bpp"]:.4f}',f'{target["pbc_mse"]:.2f}',best['setting'],best['bytes'],f'{best["bpp"]:.4f}',f'{best["mse"]:.2f}'])
resolution=[[r['image'],r['working_size'],f'{r["working_target_rgb_mse"]:.2f}',f'{100*min(v["raw_metadata_fraction"] for v in r["streams"]):.3f}-{100*max(v["raw_metadata_fraction"] for v in r["streams"]):.3f}%'] for r in diagnostics]
write('architecture/PBC4_ARCHITECTURAL_DIRECTIONS.md','''# PBC 4.0 architectural directions

No new architecture was implemented: this conventional-codec screen does not expose a compelling, demonstrated PBC representation advantage to build on. The result is a concrete competitive baseline and bounded next design, not another transform-plus-patch rescue.

## Completed-file comparison

Five originals: Googlewiki text, Mario synthetic graphics, Monet artwork, detailed Cat and smooth Flower. Ordinary released Q/HQ keeps default presets/250k planning, internal MSE and LZMA. JPEG: Pillow/libjpeg-turbo, 4:2:0, optimize; WebP: method 4; AVIF: installed Pillow/libavif, speed 8, max_threads 1, 4:2:0; JPEG XL: installed FFmpeg/libjxl effort 3, distance 8/15, thread 1. No conventional codec/video toolchain was installed.

JPEG/WebP/AVIF use bounded measured-rate searches, endpoints plus at most four bisections per PBC target, shared across Q/HQ (at most ten points per image/codec). JPEG XL has just the named two distances; do not infer equal-rate results where they fail to bracket. With ten PBC points and explicit working controls, 238 complete files were produced. This is a comparative screen, not PBC default tuning.

Full-resolution conventional files preserve source dimensions. Separate working_JPEG/working_WEBP controls encode the same PBC working target and store original dimensions in counted EXIF ImageDescription text; file-only reconstruction performs defined RGB bicubic upscale. These are explicit resizing wrappers, not full-resolution ordinary codec points or a new PBC 4 bitstream. Metadata counts toward bytes. results.json retains actual sizes, independently decoded original RGB MSE/pixel hashes and encode/decode seconds/file hashes. comparisons.json separates actual points within 5% rate, nearest points outside that range, strict dominance and adjacent measured interpolation. Never extrapolate; an estimate is not an achievable file.

These are actual AVIF files with both smaller size and lower MSE than the named PBC file, without interpolation or rate tolerance:

'''+table(['Image','Preset','PBC bytes','PBC bpp','PBC MSE','AVIF q','AVIF bytes','AVIF bpp','AVIF MSE'],avif)+'''
AVIF strictly dominates all ten PBC points. Full WebP strictly dominates nine (Flower Q is below WebP's measured minimum rate). JPEG strictly dominates the six text/synthetic/art points. PBC has real low-rate dominance over JPEG q1 on Cat Q and Flower Q/HQ, but it does not survive against AVIF here. Working-resolution conventional wrappers also dominate most photographic points. Cat HQ beats nearby q1 JPEG MSE at 3.55% more bytes, a near-rate comparison rather than dominance. No universal or perceptual PBC advantage is established.

Full-resolution JPEG/WebP/AVIF brackets favor conventional methods wherever valid. Some nearest AVIF points (Monet Q, Cat Q) remain over 5% lower-rate; direct dominance remains valid, but they are not called near-rate. JPEG XL is better on selected text/Mario/Monet points at fewer bytes, but photographic distances do not reach PBC rates. No photographic JXL equal-rate claim.

Times are one-pass complete-codec observations with common source loading excluded and no memory polling. FFmpeg JXL includes process startup; Pillow/PBC are in-process, so totals cannot rank their arithmetic speed. No repeated cross-codec speed or isolated memory claim. RGB MSE does not establish a perceptual ranking. Source/version/thread controls are in ../provenance.json.

## Structural evidence

'''+table(['Image','Working size','Direct working-target RGB MSE','Raw metadata share Q/HQ range'],resolution)+'''
Direct reconstruction uses the actual unquantized working target and production color/resize. Cat Q/HQ MSE 299/293 is near this target's 281, while AVIF reaches about 116/104 at fewer bits. Google HQ is similarly near the direct-target 684 at MSE 714, versus full AVIF 196. Fixed working resolution materially limits detail on these cases. This is not a mathematical lower bound for every possible working canvas: decoder-aware optimization can change it, and Pass 23 already tested one unsuccessful route. Mario/Monet also leave substantial patch approximation error.

Raw image/patch metadata occupies about 0.073-1.105% of unwrapped bodies; grid indices dominate. This does not attribute post-LZMA bytes, whose contexts are coupled, but argues against header micro-optimization as the main RD remedy. Initializer efficiency differs from completed refinement: Pass 25 found cheap photo initialization yet inefficient refinement. No policy ablation establishes that learned selection is essential.

Prior evidence: Pass 17 original-resolution DCT detail has selected strong RD gains; Pass 21 makes sparse storage fast; Pass 22 still has substantial photographic inverse reconstruction/memory cost. Pass 18 low-bit patches, Pass 19 ternary masks, Pass 20 tested multiscale, Pass 23 target preconditioning and Pass 24 pruning did not deliver general completed-file gains. Pass 25 transform bases followed by existing patches failed. These reject specific formulations, not all mixed representations; intermediate residual improvement is insufficient evidence.

## Distinct future designs

1. Full-resolution heterogeneous region tree with shared entropy. Use mutually exclusive source-space modes: compact smooth surface, local detail transform, or sparse/lossless graphics predictor. Select modes by completed-file marginal RD and jointly code tree/mode/symbol contexts. Unlike Pass 17 global enhancement or Pass 25 mandatory refinement, an expensive representation can be omitted rather than always paying both. Potential benefit is sharp detail without a global low-resolution ceiling and cheap truly smooth regions. Costs include partition/mode search, context simulation, boundary handling and a new format. Decoder traverses local regions/buffers without AI. Tree, mode, quantizer and context state count toward rate. Earlier split-border failures require continuous/lapped boundaries, not independent hard tiles. AVIF/JXL superiority is unproved.

2. Progressive full-resolution predictive wavelet/bitplane codec. Replace greedy overlapping patches with a scale hierarchy and embedded significance/refinement decisions. A complete stream stops at a rate budget; shared contexts avoid independently compressed residual arrays. Unlike Pass 20, rate allocation/entropy are intrinsic to one representation rather than layers appended to a PBC base. Potential benefit is controlled rate and bounded inverse work. Costs are a new predictor/context model/decoder; existing wavelet competitors already exploit this structure. Encode/decode should be roughly linear per coded bitplane, with less proposal orchestration but possible entropy overhead. RD advantage remains unproved.

Recommended first substantial experiment: design 1, retaining this conventional baseline. Freeze a minimal format with tree/mode/quantizer plus one shared entropy stream. Start with continuous smooth-surface and full-resolution detail modes, without the existing filler by default. Address the 250k ceiling and inefficient mandatory refinement. Encode performs multiple local trials; decode should need one traversal/local inverse, without full-image enhancement float copies. Count every header/mode/entropy byte. Compare actual nearby rates against full AVIF/JXL and resized conventional controls. Continue only for at least 10% matched-rate MSE reduction on two classes including one photographic and one graphics/art case, without a large decode/memory penalty. Otherwise stop before policy training, native optimization or more modes.

No prototype is forced now: neither design has demonstrated that advantage. Packaging and the optional metric are the practical engineering work; a broader rewrite must pass an RD gate.

```
python research_pass26/architecture/compare.py
python research_pass26/architecture/analyze.py
python research_pass26/architecture/diagnose.py
```
''')

write('PBC_PASS26_SUMMARY.md',f'''# Pass 26 handoff

Both standalone Windows forms work. An optional exact C metric lowers six-case mean median encode time by {gain:.2f}% and reduces one Cat encode peak by about 147 MiB. Video-container correctness succeeds, temporal rate efficiency fails. Conventional controls do not support another incremental RD change.

- Packaging: frozen actual published PBC 3.1 ZIP (edec3ec7...), not old-tag source. CLI encode/decode/inspect, four presets, RGB/RGBA, spaces/Unicode, overwrite/error and relocation checks pass. Both executable forms are local/ignored; Python off PATH, no virtualenv activation. Directory mode is preferred for latency. Paths, hashes, size, startup/warmed timing and reproduction: packaging/PBC32_PACKAGING_RESULTS.md; USAGE.txt accompanies the directory ZIP.
- Native: C11/GCC metric eliminates two full-size float32 copies. All 21 paired streams/pixels/MSE match across three repeated requested cases x two presets plus RGBA. Cat about 10% faster, Google 3-6%, Eiffel neutral/slightly slower. Optional worker switch, no production dependency. See native/PBC_NATIVE_KERNEL_RESULTS.md.
- Video: P26V intra/temporal/PNG-lossless modes independently decode. Three 16-frame clips; signed deltas use decoded references. Temporal files are 2.01x/5.11x/19.70x intra size. Predicted frames are exact, but repairing lossy keys is expensive. H.264 is much smaller/faster. Stop this formulation. See video/PBC_VIDEO_FEASIBILITY.md.
- Architecture: 238 completed image files across five classes; measured nearby rates/brackets for JPEG/WebP/AVIF, two installed JPEG XL points and counted resizing wrappers. Smaller actual AVIF files beat all ten PBC Q/HQ MSE points. Resolution and spatial payload matter more than raw headers. No forced 4.0 prototype. Recommend full-resolution heterogeneous modes/shared entropy with a competitive gate; progressive wavelet/bitplanes is the distinct alternative. See architecture/PBC4_ARCHITECTURAL_DIRECTIONS.md.

Validation includes final binary workflows/reference equality including stored and warmup v0, nine fresh model/encoder-forbidden video decodes, 20 malformed video cases/signed endpoints and native layout/random/concurrent/fallback tests. No historical suite replay, HF timing, clean Windows VM/installer/GUI/signing, captured natural video, repeated conventional speed trial or new 4.0 codec. Startup/JIT and warmed codec times are separate; timing waits do not poll memory. Memory observations are separately labeled.

New source/build/test scripts, compact rows and reports are confined to research_pass26/. Binaries/DLLs/ZIPs/frames/streams/raw images/caches remain ignored under artifacts/; independent portable copies/cache roots are recorded in packaging measurements. Use Python 3.11, requirements.txt, pinned published asset and restored hpt_data. provenance.json records hashes/versions/environment. Final commits/verified HEAD and GitHub URLs are supplied in the chat after pushing only github/research/experiments.

Published PBC 3.1 asset/tag, rust, HF, root codec/model/defaults, previous research and release staging are unchanged by this pass. GitHub main was 17e5690 at entry and moved externally to d938d003 during the pass; no write to main was performed here. No 3.2 release/tag, asset replacement, deployment or protected-branch operation was performed by this pass.
''')
print('reports generated')
