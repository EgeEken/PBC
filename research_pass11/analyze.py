import json, statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PASS9=ROOT/'research_pass9'/'comparison_summary.json'
PASS10=ROOT/'research_pass10'/'public_benchmark.json'
PROBE=ROOT/'research_pass11'/'rgb_probe.jsonl'
EIGHT=('cat.jpg','eiffel.png','flower.jpg','googlewiki.png','mario64.png','monet.jpg','ratedistortion.png','seagull.jpg')
PRESETS=('compression','balanced','quality','high_quality')

def stats(values):
    values=sorted(values)
    def percentile(q):
        x=(len(values)-1)*q; lo=int(x); hi=min(lo+1,len(values)-1); return values[lo]+(values[hi]-values[lo])*(x-lo)
    return {'mean':statistics.mean(values),'median':statistics.median(values),'p90':percentile(.9),'max':max(values)}

def main():
    p9=json.loads(PASS9.read_text()); p10=json.loads(PASS10.read_text()); rows=p10['rows']; meta=next(r for r in p10['runs'] if r['run_id']==p10['selected_run'])['metadata']
    result={'run_id':p10['selected_run'],'pass10_rows':len(rows),'same8':{},'all20':{},'trial_means':{},'pass9_same8':{},'probe':None}
    for scope,images in [('all20',meta['images']),('same8',EIGHT)]:
        result[scope]={}
        for preset in PRESETS+('overall',):
            selected=[r for r in rows if (preset=='overall' or r['variant']==preset) and r['image'] in images]
            per_image={im:[r for r in selected if r['image']==im] for im in images}
            result[scope][preset]={}
            for metric in ('encode_seconds','decode_seconds'):
                image_means=[statistics.mean(r[metric] for r in group) for group in per_image.values()]
                image_medians=[statistics.median(r[metric] for r in group) for group in per_image.values()]
                result[scope][preset][metric]={'row_mean':statistics.mean(r[metric] for r in selected),'image_mean_distribution':stats(image_means),'mean_image_median':statistics.mean(image_medians)}
    for trial in (1,2,3):
        selected=[r for r in rows if r['trial']==trial]
        result['trial_means'][str(trial)]={m:statistics.mean(r[m] for r in selected) for m in ('encode_seconds','decode_seconds')}
    for preset in PRESETS+('overall',):
        selected=[r for r in p9['rows'] if r['source']=='dev_release_rng' and r['dataset']=='hpt_data' and r['image'] in EIGHT and (preset=='overall' or r['preset']==preset)]
        result['pass9_same8'][preset]={m:statistics.mean(r[m] for r in selected) for m in ('encode_seconds','decode_seconds')}
    if PROBE.exists():
        probe=[json.loads(line) for line in PROBE.read_text().splitlines() if line.strip()]
        result['probe']={'pairs':len(probe),'ordering_disagreements':sum(r['working_vs_rgb_order_disagrees'] for r in probe),'equal_bit_pairs':sum(r['equal_raw_bits'] for r in probe),'equal_bit_alternative_wins':sum(r['equal_raw_bits'] and r['alternative']['final_rgb_sse_reduction']>r['selected']['final_rgb_sse_reduction'] for r in probe),'mean_extra_candidate_seconds':statistics.mean(r['additional_candidate_seconds'] for r in probe),'median_extra_candidate_seconds':statistics.median(r['additional_candidate_seconds'] for r in probe),'max_extra_candidate_seconds':max(r['additional_candidate_seconds'] for r in probe)}
    print(json.dumps(result,indent=2))
if __name__=='__main__': main()
