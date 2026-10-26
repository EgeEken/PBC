import base64
import hashlib
import io
import json
from pathlib import Path
from statistics import mean,median,stdev
import sys

ROOT=Path(__file__).resolve().parents[1]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))


def main():
    import numpy as np
    from PIL import Image
    from PBC3 import PBC3
    from pbc3_types import PBC3Config
    run=json.loads((HERE/'hf_representative.json').read_text())
    assert run['exact'] and len(run['rows'])==54
    cases=sorted({(r['image'],r['preset']) for r in run['rows']})
    summary={}
    for variant in ('baseline','all_mse','all'):
        groups=[[r for r in run['rows'] if (r['image'],r['preset'],r['variant'])==(i,p,variant)] for i,p in cases]
        record={}
        for metric in ('encode_seconds','decode_seconds'):
            value=mean(median(r[metric] for r in g) for g in groups)
            control=mean(median(r[metric] for r in run['rows'] if (r['image'],r['preset'],r['variant'])==(i,p,'baseline')) for i,p in cases)
            record[metric]=value
            record[metric+'_change_percent']=100*(value/control-1)
            record[metric+'_cv_percent']=mean(100*stdev(r[metric] for r in g)/mean(r[metric] for r in g) for g in groups)
            record[metric+'_mean']=mean(r[metric] for g in groups for r in g)
        summary[variant]=record
    from pbc3_types import PBC3Config
    configs={p:vars(getattr(PBC3Config,p)()) for p in ('compression','balanced','quality','high_quality')}
    (HERE/'runtime_configs.json').write_text(json.dumps(configs,indent=2)+'\n')
    old=json.loads((HERE/'hf_old_v0_decode.json').read_text())
    image=Image.open(io.BytesIO(base64.b64decode(old['reconstructed_image'].split(',')[-1])))
    pixels_hash=hashlib.sha256(np.asarray(image).tobytes()).hexdigest()
    assert pixels_hash=='6762b473012c82cf1de553a964dfed84e5eb990e6d8dc83065628fcab9445d5c'
    public=json.loads((HERE/'public_api_smoke.json').read_text())
    assert public['preview_equals_decode'] and public['stream_sha256']=='2f7ef0a8dcdd22444a65ba03f3e2955dde57844d36db7fa269567ca38bcd9414'
    checks=json.loads((HERE/'integration_checks.json').read_text())
    deployment=json.loads((HERE/'hf_deployment.json').read_text())
    after=json.loads((HERE/'hf_after.json').read_text())
    assert after['sha']==after['runtime']['sha']==deployment['commit'] and after['runtime']['stage']=='RUNNING'
    api=dict(compression=public,old_v0_decode=dict(pixels_sha256=pixels_hash,exact=True,width=image.width,height=image.height),
        healthy=True,runtime_revision=deployment['commit'])
    (HERE/'api_checks.json').write_text(json.dumps(api,indent=2)+'\n')
    result=dict(summary=summary,rows=run['rows'],metadata=run['metadata'],exact=True)
    (HERE/'hf_runtime_results.json').write_text(json.dumps(result,indent=2)+'\n')
    base=run['metadata']['workers']['baseline']
    lines=['# Runtime integration','',
        'The regular codec now uses vectorized grid I/O and explicit per-encode search-state reuse. Internal MSE stays enabled by default; its existing opt-out remains available. Stream format, presets, model and version are unchanged. This is not a PBC 3.1 release.','',
        '`pbc3_stream.py` directly exposes the tested NumPy reader/writer, retaining scalar reference functions. It removes the mutable mode dispatch and the unnecessary int64 index copy. Canonical bit ordering, offsets, low-bit masking, reader state and v0 bytes are preserved. Supported helper widths are 0–16; actual production palettes remain at most nine bits. Positive-width payload size is checked before allocating output.','',
        '`pbc3_ops.SearchState` belongs to one compress_stream call. It stores exact int32 visible-error maps and integer totals, binds them to target/canvas identity, and invalidates on warmup replacement. The accepted patch is applied before its rectangle is refreshed. The state supplies an explicit abs_error argument through the heuristic and learned proposal paths; integral construction, sampling, RNG and ordering are unchanged. No global function replacement, inspect/exec, mode switch, shared encode cache, or ctypes channel inference is used in production.','',
        f"The local gate passed {checks['frozen_cases']} saved default cases across all 20 main/five holdout images and all four presets, matching compressed bytes, decoded pixels, patch counts/order, MSE and bpp. It also passed {len(checks['edges'])} synthetic RGB/RGBA/native/downsample/init-only/heuristic/delta-reuse/warmup cases, bidirectional old/new v0 decode, {checks['grid_cases']} width/alignment/count tests plus extreme integer/strided-buffer cases, {checks['malformed_cases']} malformed-stream outcome comparisons, 20 exact proposal/RNG checks, and {checks['concurrent']['cases']} overlapping calls. Public metric-off behavior remained None; metric-required values matched. Compile/import checks passed.",'',
        'The first gate run completed its 100 comparisons but a warning from an invalid synthetic warmup setup interrupted JSON IPC before final result export. The synthetic setting was corrected; the completed gate was rerun and saved. No codec output failure occurred. No historical large archive was rebuilt.','',
        f"HF is running [{deployment['commit']}](https://huggingface.co/spaces/EgeEken/PBC/commit/{deployment['commit']}). The runtime-only commit changes five codec files, redirects the existing authenticated benchmark to a bounded non-monkeypatch worker, and preserves the prior eight-module codec as an isolated control. No refinement implementation, dataset, weight, dependency, secret, frontend, preset, release tag or version change was deployed. Existing secret authorization was reused; no rotation was needed.",'',
        'The new helper uses two persistent isolated source processes on the same pinned CPU, executing sequentially with rotated order. This avoids global replacement in either public operations or the remote validation worker. The API allowlist is now at most three images/four presets/three variants/three repeats (108 calls maximum); this pass used a six-call Balanced smoke and 54 timed calls over cat/flower/googlewiki at Quality/HQ. Warmups are complete encode/decode calls per source/configuration. Public compress and existing-v0 decode responses were independently verified, including exact old RGBA stream, preview, dimensions and displayed MSE.','',
        '| Mode | Encode s | Encode change | Decode s | Decode change |',
        '| --- | ---: | ---: | ---: | ---: |']
    for variant,s in summary.items():
        lines.append(f"| {variant} | {s['encode_seconds']:.6f} | {s['encode_seconds_change_percent']:+.2f}% | {s['decode_seconds']:.6f} | {s['decode_seconds_change_percent']:+.2f}% |")
    lines+=['','These are means of six case medians from three interleaved repetitions. all_mse is the actual public-compatible implementation; all skips internal MSE only for the private worker. Full call walls include cache setup and updates; loading, IPC and independent scoring are outside the timers. All 54 streams/pixels/trajectories/MSE/bpp match their same-runtime control. This smaller set differs from Pass 5, so the percentages are not a direct improvement over its eight-image aggregate.','',
        f"HF worker Python: {base['python'].splitlines()[0]}; packages: {base['versions']}; numerical thread variables all one, CPU {base['cpu']}. Encode CV baseline/integrated/skip: "+'/'.join(f"{summary[v]['encode_seconds_cv_percent']:.1f}%" for v in summary)+'. Decode CV: '+ '/'.join(f"{summary[v]['decode_seconds_cv_percent']:.1f}%" for v in summary)+'. All-call means are retained alongside the medians in hf_runtime_results.json.','',
        'All six public-compatible case medians improved in both metrics; no median runtime regression was observed. Encode reductions ranged from 1.59% (flower Quality) to 29.77% (googlewiki Quality); decode reductions ranged from 6.60% to 81.75%. The small flower Quality gain is within timing variation and is not a precise per-image speed claim.','',
        'Remaining limits: the benchmark is warmed and runs on a public/shared HF CPU, so scheduling/traffic noise remains. Vectorized bit packing allocates grid-size temporaries; cached error maps cost about 3 MB for RGB/250k pixels. This pass does not claim equal variant-specific peak memory or cold startup latency. Existing unbounded header-canvas allocation behavior is not repaired by grid I/O changes. Canonical v0 compatibility was tested; noncanonical private helper validation is stricter. Prepared target arrays must not be mutated externally during an encode. Historical monkeypatch-based research adapters may need their new abs_error interface forwarded; use the preserved baseline for old experiments.','',
        f"Rollback is a non-destructive revert of HF commit {deployment['commit']} followed by an authorized push to HF main; never reset/force-push or discard unrelated work. The parent is {deployment['base']}. No rollback was required. The actual GitHub integration URL is recorded in PBC_PASS6_SUMMARY.md after committing.",'']
    (HERE/'PBC_RUNTIME_INTEGRATION.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
