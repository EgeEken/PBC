import json
from pathlib import Path
import phase_a_support as e

def main():
    src=json.loads((e.OUT/'phase_a.json').read_text())
    p29=json.loads((e.ROOT/'research_pass29/outputs/results.json').read_text())
    p29_images={x['image']:x for x in p29['images']}
    timings=json.loads((e.OUT/'timing.json').read_text())
    classes={'googlewiki.png':'text','mario64.png':'synthetic graphics','flower.jpg':'smooth photography','cat.jpg':'detailed photography'}
    images={}
    for name,d in src['images'].items():
        base=p29_images[name]['base']
        d['base_stream_bytes']=base['bytes']; d['base_stream_sha256']=base['sha256']; d['source_sha256']=base['source_sha256']
        methods=d['methods']
        def get(label,q): return next(m for m in methods if m['name']==label and m['q']==q)
        shared=get('residual_shape',8); independent=get('independent_4x4',8)
        images[name]={'class':classes[name],'dimensions':[d['width'],d['height']],'cells':d['cells'],
            'base_stream_bytes':base['bytes'],'base_stream_sha256':base['sha256'],'source_sha256':base['source_sha256'],
            'base_mse':d['no_enhancement_mse'],'dc_only_mse':d['dc_mse'],'dc_residual_sse':d['dc_residual_sse'],
            'best_shared_shape_q8':{k:shared[k] for k in ('mse','residual_sse','residual_sse_reduction_beyond_dc_pct','ideal_unpriced_coefficient_sse','ideal_unpriced_reduction_beyond_dc_pct','active_cells','group_sizes','used_by_group')},
            'independent_q8':{k:independent[k] for k in ('mse','residual_sse','residual_sse_reduction_beyond_dc_pct','active_cells')},
            'fixed_setting_screen':[{'method':m['name'],'q':m['q'],'mse':m['mse'],
                'residual_sse_reduction_beyond_dc_pct':m['residual_sse_reduction_beyond_dc_pct'],
                'ideal_unpriced_reduction_beyond_dc_pct':m.get('ideal_unpriced_reduction_beyond_dc_pct'),
                'active_cells':m['active_cells']} for m in methods],
            'timings':timings[name]}
    src['provenance']={'starting_head':'9d0bc00fbc5e7a2d7bb21596dc02574705f7bb5a','pbc3_source':'Pass 29 cached base streams','runtime':timings['runtime']}
    (e.OUT/'phase_a.json').write_text(json.dumps(src,indent=2))
    compact={'experiment':'PBC V4 Pass 30 shared-action Phase A','phase_a_pass':src['gate']['pass'],
        'phase_a_classes_over_10pct':src['gate']['content_classes_with_structural_shared_action_at_least_10pct'],
        'phase_b_executed':False,'phase_b_reason':'Phase A stop gate passed on only one of four image classes',
        'config':src['config'],'provenance':src['provenance'],'images':images,
        'actual_enhancement_bytes':None,'rate_distortion_results':None,
        'nominal_uncompressed_signaling':{'global_pattern_bytes':96,'four_pattern_bytes':384,
            'four_group_map_bytes':'ceil(2 * cell_count / 8)','coefficient_bytes':'ceil(3 * cell_count / 8)',
            'dc_bytes':'3 * cell_count','independent_activation_bytes':'ceil(cell_count / 8)',
            'independent_grid_bytes':'48 * active_cell_count','measured':False}}
    (e.HERE/'PBC_V4_SHARED_ACTION_SUMMARY.json').write_text(json.dumps(compact,indent=2))

if __name__=='__main__':main()
