import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
raw=json.loads((HERE/'channel_analysis.json').read_text(encoding='utf-8'))
p12=json.loads((HERE.parent/'research_pass12'/'color_space_results.json').read_text(encoding='utf-8'))
d11=json.loads((HERE/'diagnostic_311.json').read_text(encoding='utf-8'))
d13=json.loads((HERE/'diagnostic_313.json').read_text(encoding='utf-8'))
old={(r['image'],r['preset']):r['stream_sha256'] for r in p12['cases'] if r['color_space']=='YCbCr'}
rows=[]
for r in raw['cases']:
 rows.append({k:r[k] for k in ('image','preset','hf_stream_match','pass9_stream_match','stream_sha256','expected_hf_stream_sha256','patch_count','iterative_patches','cross_channel_pairs','overlapping_rectangle_pairs','iou_ge_050_pairs','exact_rectangle_pairs','exact_grid_pairs','rectangle_iou_mean','decoded_support_jaccard_mean','decoded_delta_pearson_mean','temporal_distance_mean','raw_bytes','final_bytes')})
local={(r['image'],r['preset']):r for r in d11['cases']}
previous={(r['image'],r['preset']):r for r in d13['cases']}
diag=[]
for key,r in local.items():
 q=previous[key]
 first=next((i for i,(a,b) in enumerate(zip(r['trajectory'],q['trajectory'])) if a!=b),None)
 diag.append(dict(image=key[0],preset=key[1],hf_match=next(z['hf_stream_match'] for z in raw['cases'] if (z['image'],z['preset'])==key),
  stream_equal_311_313=r['stream_sha256']==q['stream_sha256'],input_pixels_equal_311_313=r['input_pixels_sha256']==q['input_pixels_sha256'],prepared_array_equal_311_313=r['prepared_array_sha256']==q['prepared_array_sha256'],prepared_target_equal_311_313=r['prepared_target_sha256']==q['prepared_target_sha256'],first_local_trajectory_difference=first))
summary={
 'schema':'pbc-pass13-summary-v1','baseline_id':raw['baseline_id'],'research_head_at_run':raw['research_head_at_run'],'frozen_source_revision':raw['github_baseline_source'],'hf_revision':raw['hf_baseline'],'model_sha256':raw['model_sha256'],
 'environment':raw['environment'],'reproducibility':{'cases':len(rows),'hf_stream_matches':sum(r['hf_stream_match'] for r in rows),'pass12_local_stream_matches':sum(r['stream_sha256']==old[r['image'],r['preset']] for r in rows),'unresolved_hf_mismatches':[{'image':r['image'],'preset':r['preset']} for r in rows if not r['hf_stream_match']],'bounded_runtime_diagnostics':diag},
 'cross_channel':{'streams':len(rows),'iterative_patches':sum(r['iterative_patches'] for r in rows),'cross_channel_patch_pairs':sum(r['cross_channel_pairs'] for r in rows),'overlapping_rectangle_pairs':sum(r['overlapping_rectangle_pairs'] for r in rows),'iou_at_least_050_pairs':sum(r['iou_ge_050_pairs'] for r in rows),'exact_rectangle_pairs':sum(r['exact_rectangle_pairs'] for r in rows),'exact_grid_pairs':sum(r['exact_grid_pairs'] for r in rows),'mean_per_stream_rectangle_iou_overlapping_pairs':sum(r['rectangle_iou_mean'] for r in rows if r['rectangle_iou_mean'] is not None)/sum(r['rectangle_iou_mean'] is not None for r in rows),'mean_per_stream_decoded_support_jaccard':sum(r['decoded_support_jaccard_mean'] for r in rows if r['decoded_support_jaccard_mean'] is not None)/sum(r['decoded_support_jaccard_mean'] is not None for r in rows),'mean_per_stream_decoded_delta_pearson':sum(r['decoded_delta_pearson_mean'] for r in rows if r['decoded_delta_pearson_mean'] is not None)/sum(r['decoded_delta_pearson_mean'] is not None for r in rows),'mean_per_stream_temporal_distance':sum(r['temporal_distance_mean'] for r in rows if r['temporal_distance_mean'] is not None)/sum(r['temporal_distance_mean'] is not None for r in rows),'exact_shared_geometry_payload_upper_bound_raw_bits':raw['aggregate']['geometry_saving_upper_bound_bits'],'payload_mutual_information_saving_bits_for_exact_compatible_groups':raw['aggregate']['payload_mutual_information_upper_bound_bits'],'raw_bytes':raw['aggregate']['total_raw_bytes'],'final_bytes':raw['aggregate']['total_final_bytes'],'prior_pass2_geometry_raw_bit_share_percent':0.204,'prior_pass2_geometry_final_size_sensitivity_percent':0.3},
 'cases':rows}
(HERE/'measurements.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
print(json.dumps(summary['reproducibility'],indent=2))
print(json.dumps(summary['cross_channel'],indent=2))
