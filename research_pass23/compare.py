import struct
from common import *
from research_pass17.report import interpolate


def main():
    setup();screen=json.loads((HERE/'screen.json').read_text());controls=[]
    input_paths=('research_pass17/baseline.json','research_pass17/extra_controls.json','research_pass17/targeted_controls.json','research_pass18/representation.json')
    for filename in input_paths[:3]:
        prior=json.loads((ROOT/filename).read_text())
        for original in prior['rows']:
            if original['image'] not in IMAGES:continue
            row=original.copy();name=row['image']
            if row['preset']=='quality_p10_base':
                full=(ROOT/'research_pass17/artifacts'/(name+'.dct_q128.p17')).read_bytes()
                fields=struct.unpack_from('<4sIIIIfBBI',full)
                data=full[30:30+fields[3]]
            else:data=(ROOT/'research_pass17'/row['file']).read_bytes()
            assert sha(data)==row['stream_sha256']
            output=np.asarray(PBC3.decompress(data).image.convert('RGB'))
            mse=float(np.mean((np.asarray(load(name),dtype=np.float32)-output.astype(np.float32))**2,dtype=np.float64))
            assert sha(output.tobytes())==row['pixels_sha256'] and mse==row['rgb_mse'] and len(data)==row['bytes']
            controls.append(dict(image=name,preset=row['preset'],bytes=len(data),rgb_mse=mse,stream_sha256=row['stream_sha256'],pixels_sha256=row['pixels_sha256'],source=filename))
    row=json.loads((ROOT/input_paths[3]).read_text())['googlewiki']['higher_resolution']
    data=(ROOT/'research_pass18/artifacts/googlewiki.png.high_quality_500k.pbc').read_bytes()
    output=np.asarray(PBC3.decompress(data).image.convert('RGB'))
    mse=float(np.mean((np.asarray(load('googlewiki.png'),dtype=np.float32)-output.astype(np.float32))**2,dtype=np.float64))
    assert len(data)==row['bytes'] and sha(data)==row['stream_sha256'] and sha(output.tobytes())==row['pixels_sha256'] and mse==row['rgb_mse']
    controls.append(dict(image='googlewiki.png',preset='high_quality_500k',bytes=row['bytes'],rgb_mse=mse,stream_sha256=row['stream_sha256'],pixels_sha256=row['pixels_sha256'],source=input_paths[3],config=row['config']))
    comparisons=[]
    for candidate in [r for r in screen['rows'] if r['variant']=='candidate']:
        points=[r for r in controls if r['image']==candidate['image']]
        bracket=interpolate(points,candidate['bytes'],'bytes','rgb_mse')
        near=[r for r in points if abs(candidate['bytes']/r['bytes']-1)<=.05]
        strongest=min(near,key=lambda r:r['rgb_mse']) if near else None
        comparisons.append(dict(image=candidate['image'],preset=candidate['preset'],bytes=candidate['bytes'],rgb_mse=candidate['rgb_mse'],dominated_by=[r['preset'] for r in points if r['bytes']<=candidate['bytes'] and r['rgb_mse']<=candidate['rgb_mse']],dominates=[r['preset'] for r in points if r['bytes']>=candidate['bytes'] and r['rgb_mse']>candidate['rgb_mse']],ordinary_rate_bracket=bracket,interpolated_matched_rate_mse_change_percent=100*(candidate['rgb_mse']/bracket['value']-1) if bracket else None,strongest_actual_within5pct=None if strongest is None else dict(**strongest,signed_rate_gap_percent=100*(candidate['bytes']/strongest['bytes']-1),mse_change_percent=100*(candidate['rgb_mse']/strongest['rgb_mse']-1))))
    write('comparisons.json',dict(provenance=meta(),controls=controls,rows=comparisons,input_evidence_sha256={name:sha((ROOT/name).read_bytes()) for name in input_paths},method='All saved controls independently decoded and hashes/MSE verified against current unchanged decoder. Best actual within5% retains signed rate gap. Interpolation takes lowest MSE over all measured bracketing chords, including Googlewiki500k, never extrapolates; it is an estimate, not an achievable equal-rate file. Candidate remains250k.'))
    for row in comparisons:print(row['image'],row['preset'],'interpolated MSE%',row['interpolated_matched_rate_mse_change_percent'],'actual near',row['strongest_actual_within5pct']['preset'] if row['strongest_actual_within5pct'] else None,flush=True)


if __name__=='__main__':main()
