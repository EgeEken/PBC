import json,statistics,hashlib
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image,ImageOps
H=Path(__file__).resolve().parent; R=H.parent; O=H/'outputs'; F=H/'visuals';F.mkdir(exist_ok=True)
c=json.loads((O/'codec_summary.json').read_text());cf=json.loads((O/'counterfactual_summary.json').read_text());av=json.loads((O/'avif_summary.json').read_text());base=json.loads((O/'base_manifest.json').read_text());mem=json.loads((O/'codec_encoder_memory.json').read_text())
def avg(v):return statistics.mean(v)
def get(im,b,q,l):return next(r for r in im['rows'] if r['band']==b and r['qstep']==q and r['lambda']==float(l))
def image_mse(name,arr):
 s=np.asarray(ImageOps.exif_transpose(Image.open(R/'hpt_data'/name)).convert('RGB'),np.uint8)
 return float(np.square(s.astype(np.int32)-arr.astype(np.int32),dtype=np.int64).mean())
bm={i['image']:image_mse(i['image'],np.load(O/'bases'/f"{i['image']}.base.npy",allow_pickle=False)) for i in c['images']}
# completed-file RD figure
fig,axs=plt.subplots(2,2,figsize=(13,9),constrained_layout=True);bc={'F':'#c43c39','L':'#26834a','H':'#8053a6'}
for ax,im in zip(axs.flat,c['images']):
 n=im['image'];ax.scatter(im['base_bytes']*8/(im['width']*im['height']),bm[n],marker='s',color='#222',s=40,label='PBC base')
 for cfg,col,mk in [('A','#3976aa','o'),('B','#db8b27','D')]:
  rr=[r for r in im['rows'] if r['config']==cfg];ax.scatter([r['total_bpp'] for r in rr],[r['metrics']['mse'] for r in rr],color=col,marker=mk,s=22,label='control '+cfg)
 for b,col in bc.items():
  rr=[r for r in im['rows'] if r['band']==b];ax.scatter([r['total_bpp'] for r in rr],[r['metrics']['mse'] for r in rr],color=col,s=20,alpha=.65,label='P34 '+b)
 ai=next(x for x in av['images'] if x['image']==n);ax.scatter([p['bytes']*8/(im['width']*im['height']) for p in ai['points']],[p['mse'] for p in ai['points']],color='black',marker='^',s=24,label='AVIF AOM')
 ax.set(title=n,xlabel='Total bits per pixel',ylabel='RGB MSE');ax.set_xscale('log');ax.grid(alpha=.25)
h,l=axs.flat[0].get_legend_handles_labels();fig.legend(h,l,loc='outside lower center',ncol=7,frameon=False);fig.suptitle('Completed-file rate–distortion');fig.savefig(F/'rate_distortion.png',dpi=180,bbox_inches='tight');plt.close(fig)
# conditional ablation figure
fig,ax=plt.subplots(figsize=(10,6),constrained_layout=True);cc={'L':'#26834a','H':'#8053a6'};mkmap={'ratedistortion.png':'o','coast.jpg':'s','ducklings.jpg':'D','sunset.jpg':'^'}
for im in cf['images']:
 for x in im['counterfactuals']:
  d=x['conditional_delta_from_full'];saved=-d['enhancement_bytes'];delta=d['mse'];ax.scatter(saved,delta,color=cc[x['remove_band']],marker=mkmap[im['image']],s=55,edgecolors='white',linewidths=.4)
ax.axhline(0,color='#555',lw=.8);ax.axvline(0,color='#555',lw=.8);ax.grid(alpha=.25);ax.set(xlabel='Enhancement bytes saved',ylabel='MSE increase vs F q32 lambda400',title='Conditional band deletion and XZ recompression');ax.legend(handles=[plt.Line2D([],[],marker='o',ls='',color=cc[b],label='Remove '+b) for b in ('L','H')]+[plt.Line2D([],[],marker=m,ls='',color='#444',label=name) for name,m in mkmap.items()],ncol=2);fig.savefig(F/'conditional_bands.png',dpi=180,bbox_inches='tight');plt.close(fig)
# frequency profile
fig,ax=plt.subplots(figsize=(10,5),constrained_layout=True)
for im in c['images']:
 r=get(im,'F',32,400);z=np.array(r['frequency_diagonal_counts'],float);p=100*z/z.sum() if z.sum() else z;ax.plot(range(15),p,marker='o',ms=3,label=im['image'])
ax.axvline(3.5,color='#555',ls='--',label='L/H boundary');ax.set(xticks=range(15),xlabel='DCT frequency diagonal u+v',ylabel='Selected coefficients (%)',title='Full q32 lambda400 coefficient distribution');ax.grid(alpha=.25);ax.legend();fig.savefig(F/'frequency_diagonals.png',dpi=180,bbox_inches='tight');plt.close(fig)
fields=('key','band','qstep','lambda','enhancement_bytes','total_bytes','total_bpp','metrics','mode_counts','cell_mode_counts','active_cells','active_blocks','coefficients','frequency_diagonal_counts','header_bytes','mode_bytes','dc_bytes','detail_grid_bytes','block_mask_bytes','block_count_bytes','coefficient_position_bytes','coefficient_value_varint_bytes','raw_parameter_bytes','raw_payload_bytes','xz_payload_bytes','xz_vs_raw_delta_bytes','family_wall_seconds','family_analysis_seconds','analysis_amortized_seconds','serialization_amortized_seconds','decode_seconds','fresh_decode_profile','fresh_process_verified','fresh_process_peak_working_set_bytes')
out={'pass':34,'decision':'Reject sparse-transform PBC as a release candidate; next research should target encoder-only AV1/libaom search and allocation.','provenance':c['provenance'],'avif_provenance':av['provenance'],'encoder_peak_working_set_bytes':mem['peak_working_set_bytes'],'base_mse':bm,'images':[],'conditional_ablation':cf,'avif':av}
for im in c['images']:
 out['images'].append({'image':im['image'],'width':im['width'],'height':im['height'],'base_bytes':im['base_bytes'],'base_mse':bm[im['image']],'fresh_process_verified_count':im['fresh_process_verified_count'],'candidates':[{k:r.get(k) for k in fields} for r in im['rows']]})
(H/'PBC_V4_SPARSE_VALIDATION_SUMMARY.json').write_text(json.dumps(out,indent=2)+'\n')
lines=['# PBC V4 — Research Pass 34','',
'## Decision','',
'Reject the sparse-frequency PBC variant as a release candidate. It reduces base error, but costs substantially more bytes on the photographic images, and fixed AVIF points beat its MSE at equal or lower completed-file size on all four images. Direct the next experiment to encoder-side AV1/libaom search and bit allocation. Keep production PBC unchanged. This is a four-image screen, not a population-level claim. No next pass is started automatically.','',
'## Protocol','',
'- Starting revision 39fe28e33bcff10d1d0481ec61f987c676ad9611, branch research/experiments. Image selection and SHA-256 values were frozen before enhancement runs in dataset_selection.json. All four are established hpt_data benchmark images; none was part of the Pass 33 sparse-transform screen, but they are not globally unseen.',
'- Each image used one PBC Quality base encoded through PBC3.compress with rng_mode=release and model SHA-256 '+base['model_sha256']+'. The baseline is pinned to pbc31_release_rng_baseline at 8006f30debfc4a57289df06df21636aeb52baffd. Root pbc3_ops.py and pbc3_features.py include later semantics-preserving runtime kernels; see outputs/base_manifest.json.',
'- The fixed matrix contains 6 A/B controls per image (A/B × lambda 0, 20, 100) and 18 sparse candidates (F/L/H × q16/32/64 × lambda400/1600): 96 total. L accepts u+v<4; H accepts u+v>=4; F accepts all frequencies. Eligibility is filtered before ranking/shortlisting, and the signaled band is checked during decode.',
'- All 96 PBC streams passed fresh-process pixel, mode, and parameter hash checks. The eight conditional streams and each generated AVIF point also passed fresh-process decode. The A/B controls matched Pass 31 mode, parameter, and reconstruction hashes.',
'- MSE, edge-top-10% MSE, and smooth-bottom-50% MSE use the complete decoded base plus enhancement. Bpp uses finalized bytes divided by source pixels. Region masks use source Sobel magnitude.',
'- PBC encode time is shared family analysis amortized across two lambda outputs plus serialization amortized across outputs. Three decoder timings were sampled. Encoding was not repeated; these single-run times are exploratory and not deployment speed claims. Encoder peak working set sampled over the run: '+f"{mem['peak_working_set_bytes']/1e9:.2f} GB"+'.',
'- AVIF uses Pillow 11.3.0, libavif 1.3.0, libaom 3.12.1, speed 8, one thread, 4:2:0, full range. Up to four quality bisections bracket the PBC base size and fixed F q32 lambda400 total size. Sunset did not bracket the base target; the smallest tested AVIF file is reported.',
'- Environment: '+base['environment']['python']+'; '+base['environment']['platform']+'; NumPy '+base['environment']['numpy']+', SciPy 1.16.2, Numba '+base['environment']['numba']+'. OMP/OpenBLAS/MKL/NumExpr/Numba thread limits were one.','',
'## Frozen images and base reconstruction','',
'| Image | Dimensions | Prior use | Base bytes | Base MSE |','|---|---:|---|---:|---:|']
prior={'ratedistortion.png':'Pass 10; Pass 11/12 set, chart/graphics','coast.jpg':'Pass 10 benchmark','ducklings.jpg':'Pass 10; Pass 11 RGB probe','sunset.jpg':'Pass 10 benchmark'}
for im in c['images']:lines.append(f"| {im['image']} | {im['width']}×{im['height']} | {prior[im['image']]} | {im['base_bytes']:,} | {bm[im['image']]:.2f} |")
lines += ['', '## Sparse screen','',
'The following four-image means give each image equal weight. Per-image detail for all 96 completed files is in PBC_V4_SPARSE_VALIDATION_SUMMARY.json.','',
'| Band | q | lambda | Mean total bytes | Mean MSE | Edge MSE | Smooth MSE | Amortized encode seconds |','|---|---:|---:|---:|---:|---:|---:|---:|']
for b in ('F','L','H'):
 for q in (16,32,64):
  for lam in (400,1600):
   rr=[get(im,b,q,lam) for im in c['images']]
   lines.append(f"| {b} | {q} | {lam} | {avg([r['total_bytes'] for r in rr]):,.0f} | {avg([r['metrics']['mse'] for r in rr]):.2f} | {avg([r['metrics']['edge_top10_mse'] for r in rr]):.2f} | {avg([r['metrics']['smooth_bottom50_mse'] for r in rr]):.2f} | {avg([r['analysis_amortized_seconds']+r['serialization_amortized_seconds'] for r in rr]):.2f} |")
lines += ['', '### Representative F q32 lambda400','',
'| Image | Base bytes / MSE | PBC total bytes / MSE | bpp | AVIF below-rate point: q / bytes / MSE | AVIF above-rate point: q / bytes / MSE |','|---|---:|---:|---:|---|---|']
for im in c['images']:
 r=get(im,'F',32,400);ai=next(x for x in av['images'] if x['image']==im['image']);t=ai['targets']['representative_F_q32_l400']
 lo=t.get('below');hi=t.get('above')
 lo_s=f"q{lo['quality']} / {lo['bytes']:,} / {lo['mse']:.2f}" if lo else '—';hi_s=f"q{hi['quality']} / {hi['bytes']:,} / {hi['mse']:.2f}" if hi else '—'
 lines.append(f"| {im['image']} | {im['base_bytes']:,} / {bm[im['image']]:.2f} | {r['total_bytes']:,} / {r['metrics']['mse']:.2f} | {r['total_bpp']:.4f} | {lo_s} | {hi_s} |")
lines += ['', 'F q32 lambda400 lowers base MSE on all four, but adds 4.5% (ratedistortion), 143% (coast), 110% (ducklings), and 416% (sunset) to base bytes. q refinement is not monotonic in rate or MSE. The amortized F q32 lambda400 encoder times on the 4K photos were 23.33 s, 28.50 s, and 63.40 s; A control times were 1.13 s, 1.26 s, and 1.76 s.','',
'### Conditional coefficient removal','',
'Each counterfactual starts from the same completed F q32 lambda400 stream, removes a band, recalculates active cells/blocks, reserializes, and recompresses the shared XZ stream. These are conditional deltas, not independent band encodes.','',
'| Image | Remove | Bytes saved | MSE increase | Blocks removed | Coefficients removed |','|---|---|---:|---:|---:|---:|']
for im in cf['images']:
 for r in im['counterfactuals']:
  d=r['conditional_delta_from_full'];lines.append(f"| {im['image']} | {r['remove_band']} | {-d['enhancement_bytes']:,} | {d['mse']:.2f} | {-d['active_blocks']:,} | {-d['coefficients']:,} |")
lines += ['', 'Removing L creates the larger MSE loss on all four images. Removing H is near-neutral on coast and ducklings, but increases sunset MSE by 22.0. This is useful diagnostic evidence for the low/high split, but does not reverse the AVIF comparison.','',
'### Exact stream accounting: F q32 lambda400','',
'XZ compresses the raw fields jointly, so savings are not assigned to individual fields.','',
'| Image | Header | Raw payload | XZ payload | XZ minus raw | Enhancement bytes | Blocks | Coefficients |','|---|---:|---:|---:|---:|---:|---:|---:|']
for im in c['images']:
 r=get(im,'F',32,400);lines.append(f"| {im['image']} | {r['header_bytes']} | {r['raw_payload_bytes']:,} | {r['xz_payload_bytes']:,} | {r['xz_vs_raw_delta_bytes']:+,} | {r['enhancement_bytes']:,} | {r['active_blocks']:,} | {r['coefficients']:,} |")
lines += ['', '## AVIF comparison','',
'Both neighboring AVIF points beat PBC MSE on each representative F q32 lambda400 file. On sunset, AVIF q6 is 21.1% smaller and slightly lower MSE; q12 is larger and much lower MSE. AVIF brackets the PBC base size for ratedistortion, coast, and ducklings. Sunset q0 is 95,484 bytes, above its 61,132-byte base target, so that target was not bracketed.','',
'AVIF stores AV1 images in HEIF. libavif provides the format library and tools; libaom documents partition/mode/transform rate-distortion search. [AVIF spec](https://aomediacodec.github.io/av1-avif/v1.2.0.html), [libavif](https://aomedia.googlesource.com/libavif/), [libaom encoder guide](https://aomedia.googlesource.com/aom/+/refs/heads/main/doc/dev_guide/av1_encoder.dox).','',
'## Figures','',
'![Completed-file rate-distortion](visuals/rate_distortion.png)','',
'![Conditional frequency-band deletion](visuals/conditional_bands.png)','',
'![Full-spectrum frequency profile](visuals/frequency_diagonals.png)','',
'## Transition strategy','',
'Do not continue the PBC sparse-transform codec path. Start the next experiment inside an AV1 encoder, preserving ordinary AVIF output and decoder compatibility. AV1 encoders already search and prune partitions, prediction modes, transforms, and rate-distortion candidates.','',
'1. **PBC-feature-guided AV1 candidate pruning.** Feed cheap residual-frequency and edge summaries into libaom intra partition/mode/transform candidate ranking. Compare against libaom statistical, ML, and estimated-cost pruning at matched rate and speed. Evidence gap: libaom already has these pruning methods, so PBC features must demonstrate a net gain without hiding RD losses. [libaom encoder guide](https://aomedia.googlesource.com/aom/+/refs/heads/main/doc/dev_guide/av1_encoder.dox).','',
'2. **PBC residual diagnostics for AV1 adaptive quantization.** Use smooth/edge and low/high residual measurements to propose AQ or segment weights, while libaom still emits all AV1 syntax. Validate matched file size, edge and smooth errors, and the existing 20-image main plus 5-image holdout. Evidence gap: conditional band value varied by image, so averaged MSE may hide local damage. [AV1 tool description](https://aomedia.org/docs/AV1_ToolDescription_v11-clean.pdf).','',
'3. **Content-aware selection of AV1 speed and quality settings.** Use image-level PBC analysis to select existing libaom speed features and quantizer/RDO settings. Gate on lower one-thread encode time without worse rate-distortion on the main and holdout sets. This changes encoder decisions only, not the stream.',
'Keep WebP and JPEG XL as separately measured conventional baselines: WebP uses prediction, residual transforms, quantization, and entropy coding; JPEG XL uses adaptive quantization and transform residuals. [WebP compression guide](https://developers.google.com/speed/webp/docs/compression), [JPEG XL format overview](https://github.com/libjxl/libjxl/blob/main/doc/xl_overview.md).','',
'Keep learned codecs as an independent architecture track. Apple PICO and Microsoft PULSE use their own model/decoder stacks and are not AVIF integrations. PICO documents Apple Sample Code License for code and CC-BY-NC-ND for shared study data; GenCodec/PULSE documents MIT licensing and model-dependent .pulse files. Pin code and model revisions and compare them with equal images, rates, metrics, and CPU limits. [PICO](https://github.com/apple-aiml-research/ml-pico), [PULSE](https://github.com/microsoft/GenCodec/tree/main/PULSE), [PULSE paper](https://arxiv.org/abs/2609.18602).','',
'## Validation and limits','',
'- PBC34 tests: 8 passed. Pass33 regression tests: 8 passed.',
'- All 96 PBC candidates, 8 conditional streams, and every generated AVIF point passed fresh-process decoding. The PBC full spectrum matches Pass 33 exactly under tests.',
'- AVIF quality q is not a cross-codec quality scale; comparisons use final bytes and decoded RGB MSE. No subjective study or perceptual metric was run.',
'- Four images are too few for a population-level result; three have prior general benchmark use. Encoder timings were one-run samples and are not deployment claims.',
'- No production, model, or HF Space files changed. No follow-on research pass was started.','']
(H/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
print('wrote report, summary, and 3 figures')
