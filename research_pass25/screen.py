import argparse
import dataclasses
import time
from common import *
from research_pass25 import base,hybrid,jpeg_reference


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--extend',action='store_true');args=parser.parse_args()
    setup();ARTIFACTS.mkdir(exist_ok=True)
    feasibility=json.loads((HERE/'feasibility.json').read_text());assert feasibility['passed'] and feasibility['frozen_integrated_step']==128
    if args.extend:
        prior=json.loads((HERE/'screen.json').read_text());assert prior['gate']['passed'];names=('mario64.png','eiffel.png')
    else:names=IMAGES
    warm=Image.new('RGB',(40,32),(100,140,170))
    for preset in ('quality','high_quality'):
        config=getattr(PBC3Config,preset)();PBC3.compress(warm,config=config);hybrid.encode(warm,config)
    rows=[]
    for i,name in enumerate(names):
        image=load(name);source=np.asarray(image,dtype=np.uint8)
        for j,preset in enumerate(('quality','high_quality')):
            config=getattr(PBC3Config,preset)();variants=['ordinary','base_only','hybrid']
            if (i+j)%2:variants=variants[::-1]
            for variant in variants:
                start=time.perf_counter()
                if variant=='ordinary':result=PBC3.compress(image,config=config,rng_mode='release');details={}
                else:result,details=hybrid.encode(image,dataclasses.replace(config,patch_count=0) if variant=='base_only' else config,step=128)
                enc=time.perf_counter()-start
                start=time.perf_counter();decoded=PBC3.decompress(result.data).image if variant=='ordinary' else hybrid.decode(result.data);dec=time.perf_counter()-start
                output=np.asarray(decoded,dtype=np.uint8);error=mse(source,output)
                assert error==result.mse and decoded.tobytes()==result.image.tobytes()
                if variant=='ordinary':
                    old,record=saved(name,preset);assert old==result.data and error==record['rgb_mse']
                    hh,pp,ledger=parse(result.data)
                    details=dict(spatial_patches=len(pp)-3,initial_patches=3,raw_bytes=ledger['raw_bytes'])
                else:
                    initial,raw,count=hybrid.split(result.data)
                    if variant=='base_only':
                        expected=next((r for r in feasibility['rows'] if r['image']==name and r['variant']=='transform' and r['step']==128),None)
                        if expected:assert sha(initial)==expected['stream_sha256'] and sha(output.tobytes())==expected['pixels_sha256']
                    assert count==details['spatial_patches']
                path=ARTIFACTS/(name+'.'+preset+'.'+variant+('.pbc' if variant=='ordinary' else '.p25b'));path.write_bytes(result.data)
                row=dict(image=name,preset=preset,variant=variant,config=result.config.__dict__,rng_mode='release',bytes=len(result.data),bpp=8*len(result.data)/(image.width*image.height),rgb_mse=error,internal_mse=result.mse,encode_seconds=enc,decode_seconds=dec,source_size=list(image.size),working_size=[result.working_width,result.working_height],input_sha256=sha((ROOT/'hpt_data'/name).read_bytes()),stream_sha256=sha(result.data),pixels_sha256=sha(output.tobytes()),file=str(path.relative_to(HERE)),**details)
                rows.append(row);print(name,preset,variant,row['bytes'],round(error,3),round(enc,3),round(dec,3),details['spatial_patches'],flush=True)
        prep=PBC3.prepare(image,PBC3Config.quality());jpeg_reference.decode(jpeg_reference.encode(prep,75))
        start=time.perf_counter();data=jpeg_reference.encode(prep,75);enc=time.perf_counter()-start
        start=time.perf_counter();decoded=jpeg_reference.decode(data);dec=time.perf_counter()-start
        path=ARTIFACTS/(name+'.jpeg_q75.jpg');path.write_bytes(data)
        rows.append(dict(image=name,preset='jpeg_q75',variant='jpeg',quality=75,bytes=len(data),bpp=8*len(data)/(image.width*image.height),rgb_mse=mse(source,np.asarray(decoded)),encode_seconds_excluding_common_prepare=enc,decode_seconds=dec,stream_sha256=sha(data),pixels_sha256=sha(decoded.tobytes()),file=str(path.relative_to(HERE))))
    comparisons=[];classes={name:'photograph' if name in ('cat.jpg','flower.jpg') else 'art' if name=='monet.jpg' else 'graphics' for name in names}
    for name in names:
        jpeg=next(r for r in rows if r['image']==name and r['variant']=='jpeg')
        for preset in ('quality','high_quality'):
            ordinary=next(r for r in rows if r['image']==name and r['preset']==preset and r['variant']=='ordinary');candidate=next(r for r in rows if r['image']==name and r['preset']==preset and r['variant']=='hybrid');base_only=next(r for r in rows if r['image']==name and r['preset']==preset and r['variant']=='base_only')
            rate=100*(candidate['bytes']/ordinary['bytes']-1);quality=100*(candidate['rgb_mse']/ordinary['rgb_mse']-1)
            comparisons.append(dict(image=name,preset=preset,image_class=classes[name],rate_change_percent=rate,mse_change_percent=quality,encode_change_percent=100*(candidate['encode_seconds']/ordinary['encode_seconds']-1),decode_change_percent=100*(candidate['decode_seconds']/ordinary['decode_seconds']-1),refinement_rate_change_percent=100*(candidate['bytes']/base_only['bytes']-1),refinement_mse_change_percent=100*(candidate['rgb_mse']/base_only['rgb_mse']-1),jpeg_dominates=jpeg['bytes']<=candidate['bytes'] and jpeg['rgb_mse']<=candidate['rgb_mse'],base_only_dominates=base_only['bytes']<=candidate['bytes'] and base_only['rgb_mse']<=candidate['rgb_mse'],useful=(quality<=-5 and rate<=5) or (rate<=-5 and quality<=1)))
    useful=[r for r in comparisons if r['useful'] and not r['jpeg_dominates'] and not r['base_only_dominates']]
    gate=dict(useful_nondominated_classes=sorted(set(r['image_class'] for r in useful)),max_photo_mse_increase=max([r['mse_change_percent'] for r in comparisons if r['image_class']=='photograph'] or [0]),jpeg_dominated_count=sum(r['jpeg_dominates'] for r in comparisons),base_dominated_count=sum(r['base_only_dominates'] for r in comparisons))
    gate['passed']=len(gate['useful_nondominated_classes'])>=2 and gate['max_photo_mse_increase']<=10
    write('extension.json' if args.extend else 'screen.json',dict(provenance=meta(),rows=rows,comparisons=comparisons,gate=gate,extended=args.extend,timing='One deterministic warmed full encode/decode per setting, alternating source/candidate order. Ordinary/hybrid/base-only timers include common prepare, base encode/decode, owned heads/model load, all fresh refinement, serialization/entropy and enabled true-source internal MSE. JPEG encode excludes the separately shared ordinary prepare and is not a full speed comparison. Decoders include final RGB/resize. IO/hash/external scoring excluded.'))
    print('completed gate',gate,flush=True)


if __name__=='__main__':main()
