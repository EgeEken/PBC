import hashlib,json,statistics,subprocess
from pathlib import Path
root=Path.cwd(); p=root/'research_pass37'
bench=json.loads((p/'outputs/benchmark_final/benchmark.json').read_text())
counts=json.loads((p/'outputs/search_counts/search_counts.json').read_text())
parity=json.loads((p/'outputs/ctrl_parity/ctrl_parity.json').read_text())
syn=json.loads((p/'outputs/synthetic_smoke/synthetic_results.json').read_text())
ff=json.loads((p/'outputs/ffmpeg_decode.json').read_text())
manifest=json.loads((root/'research_pass35/SOURCE_MANIFEST.json').read_text())
assets={x['name']:x for x in manifest['assets']}
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
rows=[]
for r in bench['rows']:
    c=next(x for x in counts['cases'] if (x['image'],x['qcolor'],x['speed'])==(r['image'],r['qcolor'],r['speed']))
    def metrics(label):
        v=r['configs'][label]
        return {k:v[k] for k in ('bytes','bpp','avif_sha256','decoded_rgb_sha256','rgb_mse','median_seconds','mean_seconds','stdev_seconds')}
    rows.append({'image':r['image'],'qcolor':r['qcolor'],'speed':r['speed'],'source_sha256':r['source_sha256'],'dimensions':[r['width'],r['height']],
      'ref':metrics('REF'),'ctrl_byte_identical':r['ref_ctrl_byte_identical'],'exp':metrics('EXP'),
      'time_reduction_percent':r['time_reduction_percent'],'speedup':r['speedup'],'byte_change_percent':r['byte_change_percent'],'mse_change_percent':r['mse_change_percent'],
      'exp_deterministic':r['exp_deterministic'],
      'search_counts':{label:{k:c['results'][label]['counters']['totals'][k] for k in ('rd_intra_blocks','candidate_visits','transform_search_entries','pruned_before_transform','invalid_transform_results','selected_mode_records','hybrid_rd_calls','hybrid_nonrd_calls')} for label in ('CTRL','EXP')},
      'transform_search_change_percent':c['results']['transform_search_change_percent']})
s6=[r for r in rows if r['speed']==6]
patch=p/'patches/intra-mode-syntax-cost-order.patch'
summary={
 'pass':37,
 'repository':{'name':'EgeEken/PBC','branch':'research/experiments','starting_head':'3723dc14de88c6c3dcbd98dd80a097e019a92a9f','ending_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()},
 'sources':{'libaom':{'version':'3.12.1','commit':'10aece4157eb79315da205f39e19bf6ab3ee30d0'},'libavif':{'version':'1.3.0','commit':'1aadfad932c98c069a1204261b1856f81f3bc199'}},
 'patch':{'path':'patches/intra-mode-syntax-cost-order.patch','sha256':sha(patch),'summary':'Stable ascending syntax-cost order for the first 13 base intra modes; angle order unchanged; gated by PBC37_EXPERIMENT.'},
 'build':{'compiler':'Strawberry MinGW-w64 GCC/G++ 13.2.0','optimization':'Release -O3 -DNDEBUG','control_macro':'PBC37_EXPERIMENT undefined','experiment_macro':'PBC37_EXPERIMENT defined','linkage':'static local libaom archives','ctrl_archive_sha256':sha(p/'outputs/build/aom-ctrl/libaom.a'),'exp_archive_sha256':sha(p/'outputs/build/aom-exp/libaom.a'),'ref_avifenc_sha256':sha(root/'research_pass35/outputs/binaries/reference/avifenc.exe'),'ctrl_avifenc_sha256':sha(p/'outputs/binaries/ctrl/avifenc.exe'),'exp_avifenc_sha256':sha(p/'outputs/binaries/exp/avifenc.exe'),'instrumented_timings_used':False},
 'settings':{'depth':8,'yuv':'420','range':'full','cicp':'1/13/1','threads':1,'metadata':'omitted','timed_repetitions':5,'timing_boundary':bench['timing_boundary'],'memory_probes':bench['memory_method'],'speed6_and_speed8_cases':[{'image':r['image'],'qcolor':r['qcolor'],'speed':r['speed']} for r in rows]},
 'ctrl_parity':{'case_count':len(parity['cases']),'all_byte_identical':parity['all_byte_identical'],'details':'outputs/ctrl_parity/ctrl_parity.json'},
 'search_counter_method':'Pass 35 PBC_PROFILE aggregate counters; all instrumented timings excluded.',
 'results':rows,
 'summary_speed6':{'point_count':len(s6),'mean_point_median_time_reduction_percent':statistics.mean(r['time_reduction_percent'] for r in s6),'median_point_median_time_reduction_percent':statistics.median(r['time_reduction_percent'] for r in s6),'faster_points':sum(r['time_reduction_percent']>0 for r in s6),'mean_byte_change_percent':statistics.mean(r['byte_change_percent'] for r in s6),'mean_mse_change_percent':statistics.mean(r['mse_change_percent'] for r in s6),'ctrl_transform_entries':sum(r['search_counts']['CTRL']['transform_search_entries'] for r in s6),'exp_transform_entries':sum(r['search_counts']['EXP']['transform_search_entries'] for r in s6)},
 'correctness':{'exp_timed_runs_deterministic':all(r['exp_deterministic'] for r in rows),'unmodified_avifdec':'all nine development points and seven synthetic cases decoded; dimensions checked','ffmpeg':{'version':ff['ffmpeg_version'],'passed_files':len(ff['rows']),'details':'outputs/ffmpeg_decode.json'},'synthetic_case_count':len({r['case'] for r in syn['cases']}),'synthetic_all_passed':syn['all_passed'],'synthetic_details':'outputs/synthetic_smoke/synthetic_results.json','sanitizer_run':False,'decoder_modified':False},
 'held_out_validation':{'status':'not_run','reason':'Gate C failed; development results did not justify conditional validation.'},
 'gates':{'A':'passed','B':'passed','C':'failed','D':'not_run'},
 'mse_calculation':'RGB uint8 source/reconstruction arrays promoted to float64 before subtraction and squaring; historical Pass 35 MSE used int16 and may overflow, so absolute cross-pass MSE comparisons are invalid.',
 'recommendation':'Reject the syntax-cost ordering. Later inspect search_tx_type candidate ranking and its rate-only best-RD skip/adaptive early termination, accounting for the existing txk_map estimated-RD ordering.'
}
(p/'PASS37_SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
print('summary bytes', (p/'PASS37_SUMMARY.json').stat().st_size, 'rows',len(rows),'head',summary['repository']['ending_head'])

