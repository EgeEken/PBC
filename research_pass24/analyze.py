import statistics
from common import *
from scipy.stats import spearmanr
from research_pass17.report import interpolate


def main():
    setup();measured=json.loads((HERE/'omissions.json').read_text());rows=measured['rows'];summaries=[]
    controls=json.loads((ROOT/'research_pass23/comparisons.json').read_text())['controls']
    for record in measured['files']:
        subset=[r for r in rows if r['image']==record['image'] and r['preset']==record['preset']]
        strict=[r for r in subset if r['category']=='strict_rd_improvement']
        low_loss=[r for r in subset if r['byte_change']<0 and r['mse_change_percent']<=1]
        best=min(low_loss,key=lambda r:r['rate_change_percent']) if low_loss else None
        summaries.append(dict(image=record['image'],preset=record['preset'],samples=len(subset),available_iterative=record['iterative_patches'],rgb_mse_improving=sum(r['mse_change']<0 for r in subset),nonincreasing_mse=sum(r['mse_change']<=0 for r in subset),strict_count=len(strict),best_low_loss_patch=best['patch'] if best else None,best_low_loss_rate_change_percent=best['rate_change_percent'] if best else None,best_low_loss_mse_change_percent=best['mse_change_percent'] if best else None,evaluation_total_seconds=sum(r['evaluation_seconds'] for r in subset),evaluation_median_seconds=statistics.median(r['evaluation_seconds'] for r in subset),repack_median_seconds=statistics.median(r['repack_seconds'] for r in subset),decode_median_seconds=statistics.median(r['decode_seconds'] for r in subset),score_median_seconds=statistics.median(r['score_seconds'] for r in subset)))
    groups={}
    groups.update({preset:[r for r in rows if r['preset']==preset] for preset in PRESETS})
    groups.update({name:[r for r in rows if r['image']==name] for name in IMAGES})
    groups.update(early=[r for r in rows if r['iterative_fraction']<1/3],middle=[r for r in rows if 1/3<=r['iterative_fraction']<2/3],late=[r for r in rows if r['iterative_fraction']>=2/3],overlap_at_least_half=[r for r in rows if r['later_same_component_overlap']>=.5],overlap_below_half=[r for r in rows if r['later_same_component_overlap']<.5],effective_bits_1to3=[r for r in rows if r['effective_bit_depth']<=3],effective_bits_4plus=[r for r in rows if r['effective_bit_depth']>=4])
    aggregate={label:dict(count=len(group),strict=sum(r['category']=='strict_rd_improvement' for r in group),mse_improving=sum(r['mse_change']<0 for r in group),low_loss_saving=sum(r['byte_change']<0 and r['mse_change_percent']<=1 for r in group),median_mse_change_percent=statistics.median(r['mse_change_percent'] for r in group),median_rate_change_percent=statistics.median(r['rate_change_percent'] for r in group)) for label,group in groups.items()}
    correlations={key:float(spearmanr([r[key] for r in rows],[r['mse_change_percent'] for r in rows]).statistic) for key in ('iterative_fraction','later_same_component_overlap','effective_bit_depth','total_bits')}
    chosen=[]
    for record in measured['files']:
        subset=[r for r in rows if r['image']==record['image'] and r['preset']==record['preset']]
        strict=[r for r in subset if r['category']=='strict_rd_improvement']
        low_loss=[r for r in subset if r['byte_change']<0 and r['mse_change_percent']<=1]
        selected={r['patch']:r for r in strict}
        if low_loss:
            best=min(low_loss,key=lambda r:r['rate_change_percent']);selected[best['patch']]=best
        near_threshold=[r for r in subset if r['rate_change_percent']<=-5 and r['mse_change_percent']<=2]
        for r in near_threshold:selected[r['patch']]=r
        for r in selected.values():
            points=[c for c in controls if c['image']==r['image']]
            bracket=interpolate(points,r['bytes'],'bytes','rgb_mse')
            near=[c for c in points if abs(r['bytes']/c['bytes']-1)<=.05]
            best=min(near,key=lambda c:c['rgb_mse']) if near else None
            chosen.append(dict(image=r['image'],preset=r['preset'],patch=r['patch'],category=r['category'],bytes=r['bytes'],rgb_mse=r['rgb_mse'],rate_change_percent=r['rate_change_percent'],mse_change_percent=r['mse_change_percent'],dominated_by=[c['preset'] for c in points if c['bytes']<=r['bytes'] and c['rgb_mse']<=r['rgb_mse']],dominates=[c['preset'] for c in points if c['bytes']>=r['bytes'] and c['rgb_mse']>r['rgb_mse']],ordinary_rate_bracket=bracket,interpolated_mse_change_percent=100*(r['rgb_mse']/bracket['value']-1) if bracket else None,best_measured_within5pct=None if best is None else dict(preset=best['preset'],bytes=best['bytes'],rgb_mse=best['rgb_mse'],signed_rate_gap_percent=100*(r['bytes']/best['bytes']-1),mse_change_percent=100*(r['rgb_mse']/best['rgb_mse']-1))))
    write('analysis.json',dict(provenance=meta(),summary=summaries,groups=aggregate,rank_correlation_with_mse_increase=correlations,chosen_control_comparisons=chosen,prior_controls=controls,prior_control_manifest_sha256=sha((ROOT/'research_pass23/comparisons.json').read_bytes()),sample_count=len(rows),total_iterative_patches=sum(r['iterative_patches'] for r in measured['files']),category_counts={key:sum(r['category']==key for r in rows) for key in sorted(set(r['category'] for r in rows))},passed=measured['gate']['passed'],scope='Structural sample deliberately enriches large/overlapped/early/late patches; descriptive associations are not population estimates or a validated cheap predictor. Reused ordinary controls were independently verified in Pass23 with the identical practical decoder, preserved source and hashes; current saved parent controls independently checked in this pass. Conservative bracket estimates do not extrapolate and are not achievable equal-rate files. No C/D/E stages.'))
    print('sample',len(rows),'of',sum(r['iterative_patches'] for r in measured['files']),'iterative patches; categories', {key:sum(r['category']==key for r in rows) for key in sorted(set(r['category'] for r in rows))},flush=True)
    print('descriptive associations',correlations,flush=True)


if __name__=='__main__':main()
