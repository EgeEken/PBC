import csv
import json
import statistics
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
HERE=Path(__file__).resolve().parent
DATA=json.loads((HERE/'color_space_results.json').read_text(encoding='utf-8'))
PAIRS=DATA['cases']
PRESETS=('quality','high_quality')
IMAGES=tuple(dict.fromkeys(r['image'] for r in PAIRS))

def mean(rows,key): return statistics.mean(r[key] for r in rows)
def classification(y,r):
    dm=r['mse']-y['mse']; db=r['bpp']-y['bpp']
    if dm<=0 and db<=0 and (dm<0 or db<0): return 'strict_rgb_win'
    if dm>=0 and db>=0 and (dm>0 or db>0): return 'strict_rgb_loss'
    if dm<0 and db>0: return 'rgb_quality_for_rate'
    if dm>0 and db<0: return 'rgb_rate_for_quality'
    return 'tie'

def main():
    pass9=json.loads((ROOT/'research_pass9'/'comparison_summary.json').read_text(encoding='utf-8'))
    pass10=json.loads((ROOT/'research_pass10'/'public_benchmark.json').read_text(encoding='utf-8'))
    inputs={(r['dataset'],r['image']):r['input_sha256'] for r in pass9['equivalence']}
    by={(r['image'],r['preset'],r['color_space']):r for r in PAIRS}
    for (image,preset,space),row in by.items():
        assert row['input_sha256']==inputs['hpt_data',image],f'input hash mismatch: {image}'
        if space=='YCbCr':
            expected=next(r for r in pass10['rows'] if r['run_id']==pass10['selected_run'] and r['trial']==1 and r['image']==image and r['variant']==preset)
            assert row['baseline_match_pass10_release']==(row['stream_sha256']==expected['stream_sha256'] and row['patch_count']==expected['patches'])
    paired=[]
    for image in IMAGES:
        for preset in PRESETS:
            y=by[image,preset,'YCbCr']; r=by[image,preset,'RGB']
            assert {k:v for k,v in y['config'].items() if k!='color_space'}=={k:v for k,v in r['config'].items() if k!='color_space'}
            dm=r['mse']-y['mse']; db=r['bpp']-y['bpp']
            paired.append(dict(image=image,preset=preset,input_sha256=y['input_sha256'],input_size=json.dumps(y['input_size']),classification=classification(y,r),ycbcr_config_sha256=y['config_sha256'],rgb_config_sha256=r['config_sha256'],
                ycbcr_mse=y['mse'],rgb_mse=r['mse'],mse_change=dm,mse_change_percent=100*dm/y['mse'],
                ycbcr_bpp=y['bpp'],rgb_bpp=r['bpp'],bpp_change=db,bpp_change_percent=100*db/y['bpp'],
                ycbcr_encode_seconds=y['mean_encode_seconds'],rgb_encode_seconds=r['mean_encode_seconds'],encode_change_percent=100*(r['mean_encode_seconds']/y['mean_encode_seconds']-1),
                ycbcr_decode_seconds=y['mean_decode_seconds'],rgb_decode_seconds=r['mean_decode_seconds'],decode_change_percent=100*(r['mean_decode_seconds']/y['mean_decode_seconds']-1),
                ycbcr_patches=y['patch_count'],rgb_patches=r['patch_count'],ycbcr_iterative_patches=y['iterative_patches'],rgb_iterative_patches=r['iterative_patches'],
                ycbcr_raw_bytes=y['raw_stream_bytes'],rgb_raw_bytes=r['raw_stream_bytes'],ycbcr_final_bytes=y['final_file_bytes'],rgb_final_bytes=r['final_file_bytes'],
                ycbcr_stream_sha256=y['stream_sha256'],rgb_stream_sha256=r['stream_sha256'],ycbcr_decoded_size=json.dumps(y['decoded_size']),rgb_decoded_size=json.dumps(r['decoded_size']),ycbcr_entropy_method=y['entropy_method'],rgb_entropy_method=r['entropy_method'],ycbcr_matches_pass10=bool(y['baseline_match_pass10_release']),
                repeat_encode_ycbcr=json.dumps(y['repeat_encode_seconds']),repeat_encode_rgb=json.dumps(r['repeat_encode_seconds']),
                repeat_decode_ycbcr=json.dumps(y['repeat_decode_seconds']),repeat_decode_rgb=json.dumps(r['repeat_decode_seconds']),
                ycbcr_channels=json.dumps(y['iterative_channel_counts'],sort_keys=True),rgb_channels=json.dumps(r['iterative_channel_counts'],sort_keys=True),
                ycbcr_grid_bitcount=json.dumps(y['grid_bitcount_histogram'],sort_keys=True),rgb_grid_bitcount=json.dumps(r['grid_bitcount_histogram'],sort_keys=True),
                ycbcr_cell_size_histogram=json.dumps(y['cell_size_histogram'],sort_keys=True),rgb_cell_size_histogram=json.dumps(r['cell_size_histogram'],sort_keys=True),
                ycbcr_mean_grid_width=y['mean_grid_width'],rgb_mean_grid_width=r['mean_grid_width'],ycbcr_mean_grid_height=y['mean_grid_height'],rgb_mean_grid_height=r['mean_grid_height'],
                ycbcr_cells_per_patch=y['mean_grid_cells'],rgb_cells_per_patch=r['mean_grid_cells']))
    fields=list(paired[0])
    with (HERE/'paired_results.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader(); writer.writerows(paired)
    floor_fields=list(DATA['floor'][0])
    with (HERE/'preprocess_floor.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=floor_fields); writer.writeheader(); writer.writerows(DATA['floor'])
    out={'baseline_id':DATA['baseline_manifest_id'],'baseline_source':DATA['github_baseline_source'],'research_head_at_run':DATA['research_head'],
      'hf_baseline':DATA['hf_baseline'],'model_sha256':DATA['model_sha256'],'source_sha256':DATA['source_sha256'],
      'environment':{k:DATA[k] for k in ('python','platform','processor','numpy','numba','pillow','thread_env','cpu_affinity','warmups','timing_scope','rng')},
      'cases':len(paired),'main_holdout_confirmation':'not run: screen had no strict RD wins and RGB bpp increased on every screen case',
      'baseline_reproduction':{'ycbcr_matches_pass10':sum(r['ycbcr_matches_pass10'] for r in paired),'ycbcr_cases':len(paired),'mismatches':[dict(image=r['image'],preset=r['preset']) for r in paired if not r['ycbcr_matches_pass10']]},
      'counts':dict(Counter(r['classification'] for r in paired)), 'preset_summary':{},'class_summary':{}}
    for preset in PRESETS:
        group=[r for r in paired if r['preset']==preset]
        out['preset_summary'][preset]={}
        for key in ('ycbcr_mse','rgb_mse','ycbcr_bpp','rgb_bpp','ycbcr_encode_seconds','rgb_encode_seconds','ycbcr_decode_seconds','rgb_decode_seconds','ycbcr_raw_bytes','rgb_raw_bytes','ycbcr_final_bytes','rgb_final_bytes','ycbcr_patches','rgb_patches','ycbcr_iterative_patches','rgb_iterative_patches','ycbcr_cells_per_patch','rgb_cells_per_patch'):
            out['preset_summary'][preset][key]=mean(group,key)
        out['preset_summary'][preset]['mse_change_percent']=100*(mean(group,'rgb_mse')/mean(group,'ycbcr_mse')-1)
        out['preset_summary'][preset]['bpp_change_percent']=100*(mean(group,'rgb_bpp')/mean(group,'ycbcr_bpp')-1)
        out['preset_summary'][preset]['encode_change_percent']=100*(mean(group,'rgb_encode_seconds')/mean(group,'ycbcr_encode_seconds')-1)
        out['preset_summary'][preset]['decode_change_percent']=100*(mean(group,'rgb_decode_seconds')/mean(group,'ycbcr_decode_seconds')-1)
        out['preset_summary'][preset]['mse_improvements']=sum(r['mse_change']<0 for r in group)
        out['preset_summary'][preset]['bpp_increases']=sum(r['bpp_change']>0 for r in group)
        out['preset_summary'][preset]['categories']=dict(Counter(r['classification'] for r in group))
        out['preset_summary'][preset]['channel_counts']={}
        out['preset_summary'][preset]['bitcount_histogram']={}
        for space in ('YCbCr','RGB'):
            channels=Counter(); precision=Counter()
            for r in PAIRS:
                if r['preset']==preset and r['color_space']==space:
                    channels.update({k:int(v) for k,v in r['iterative_channel_counts'].items()})
                    precision.update({k:int(v) for k,v in r['grid_bitcount_histogram'].items()})
            out['preset_summary'][preset]['channel_counts'][space]=dict(channels)
            out['preset_summary'][preset]['bitcount_histogram'][space]=dict(precision)
    classes={'photography':{'cat.jpg','flower.jpg','monet.jpg','stockpoint.jpg'},'graphics_synthetic':{'googlewiki.png','mario64.png','eiffel.png','ratedistortion.png'}}
    for name,names in classes.items():
        out['class_summary'][name]={}
        for preset in PRESETS:
            group=[r for r in paired if r['preset']==preset and r['image'] in names]
            out['class_summary'][name][preset]={'mse_change_percent':100*(mean(group,'rgb_mse')/mean(group,'ycbcr_mse')-1),'bpp_change_percent':100*(mean(group,'rgb_bpp')/mean(group,'ycbcr_bpp')-1),'cases':len(group)}
    (HERE/'screen_summary.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'rows':len(paired),'baseline_match':out['baseline_reproduction'],'categories':out['counts'],'summary':out['preset_summary']},indent=2))
if __name__=='__main__': main()
