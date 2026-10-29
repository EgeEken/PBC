import struct
from common import *
from research_pass25 import base,hybrid
from research_pass17.report import interpolate


def main():
    setup();a=json.loads((HERE/'feasibility.json').read_text());screen=json.loads((HERE/'screen.json').read_text());controls=[];base_points=[]
    for filename in ('research_pass17/baseline.json','research_pass17/extra_controls.json','research_pass17/targeted_controls.json'):
        for row in json.loads((ROOT/filename).read_text())['rows']:
            if row['image'] not in IMAGES:continue
            if row['preset']=='quality_p10_base':
                full=(ROOT/'research_pass17/artifacts'/(row['image']+'.dct_q128.p17')).read_bytes();fields=struct.unpack_from('<4sIIIIfBBI',full);data=full[30:30+fields[3]]
            else:data=(ROOT/'research_pass17'/row['file']).read_bytes()
            assert sha(data)==row['stream_sha256']
            controls.append(dict(image=row['image'],preset=row['preset'],bytes=row['bytes'],rgb_mse=row['rgb_mse'],stream_sha256=row['stream_sha256'],source=filename))
    extra=json.loads((ROOT/'research_pass18/representation.json').read_text())['googlewiki']['higher_resolution']
    data=(ROOT/'research_pass18/artifacts/googlewiki.png.high_quality_500k.pbc').read_bytes();assert sha(data)==extra['stream_sha256']
    controls.append(dict(image='googlewiki.png',preset='high_quality_500k',bytes=extra['bytes'],rgb_mse=extra['rgb_mse'],stream_sha256=extra['stream_sha256'],source='research_pass18/representation.json'))
    for row in a['rows']:
        if row['variant']!='transform':continue
        initial=(HERE/row['file']).read_bytes();data,raw,packed=hybrid.frame(initial,[])
        pixels=hybrid.decode(data);assert sha(pixels.tobytes())==row['pixels_sha256']
        path=ARTIFACTS/(row['image']+f".q{row['step']}.base_only.p25b");path.write_bytes(data)
        base_points.append(dict(image=row['image'],step=row['step'],bytes=len(data),rgb_mse=row['rgb_mse'],stream_sha256=sha(data),pixels_sha256=row['pixels_sha256'],file=str(path.relative_to(HERE))))
    comparisons=[]
    for row in screen['rows']:
        if row['variant']!='hybrid':continue
        name=row['image'];points=[r for r in controls if r['image']==name];bracket=interpolate(points,row['bytes'],'bytes','rgb_mse');near=[r for r in points if abs(row['bytes']/r['bytes']-1)<=.05];best=min(near,key=lambda r:r['rgb_mse']) if near else None
        jpeg=[r for r in screen['rows'] if r['image']==name and r['variant']=='jpeg']+[r for r in a['rows'] if r['image']==name and r['variant']=='jpeg']
        jpeg_bracket=interpolate([dict(r,preset='jpeg_q'+str(r.get('quality',r.get('step')))) for r in jpeg],row['bytes'],'bytes','rgb_mse')
        comparisons.append(dict(image=name,preset=row['preset'],bytes=row['bytes'],rgb_mse=row['rgb_mse'],ordinary_dominated_by=[r['preset'] for r in points if r['bytes']<=row['bytes'] and r['rgb_mse']<=row['rgb_mse']],ordinary_dominates=[r['preset'] for r in points if r['bytes']>=row['bytes'] and r['rgb_mse']>row['rgb_mse']],ordinary_rate_bracket=bracket,interpolated_ordinary_mse_change_percent=100*(row['rgb_mse']/bracket['value']-1) if bracket else None,actual_ordinary_within5pct=None if best is None else dict(**best,signed_rate_gap_percent=100*(row['bytes']/best['bytes']-1),mse_change_percent=100*(row['rgb_mse']/best['rgb_mse']-1)),existing_base_steps_dominating=[r['step'] for r in base_points if r['image']==name and r['bytes']<=row['bytes'] and r['rgb_mse']<=row['rgb_mse']],jpeg_rate_bracket=jpeg_bracket,interpolated_jpeg_mse_change_percent=100*(row['rgb_mse']/jpeg_bracket['value']-1) if jpeg_bracket else None))
    write('analysis.json',dict(provenance=meta(),ordinary_controls=controls,wrapped_base_only_points=base_points,comparisons=comparisons,control_evidence_sha256={name:sha((ROOT/name).read_bytes()) for name in ('research_pass17/baseline.json','research_pass17/extra_controls.json','research_pass17/targeted_controls.json','research_pass18/representation.json')},method='Saved ordinary controls checked by SHA with unchanged source; fresh screen defaults exactly reproduce them. Reframing existing A base streams adds17-byte hybrid header with zero patches, no extra transform/image encode or quantizer search. Actual dominance/within5% gaps separate from minimum measured-chord estimates, never extrapolate. JPEG q1/q75 endpoints are sparse and usually far from candidate bytes.'))
    print('base-only domination',[r for r in comparisons if r['existing_base_steps_dominating']],flush=True)


if __name__=='__main__':main()
