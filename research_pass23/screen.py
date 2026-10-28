import argparse
import collections
import time
from common import *
from research_pass23.encoder import DecoderAwarePBC,PARAMETERS
from research_pass2.stream_tools import parse


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--extend',action='store_true');args=parser.parse_args()
    setup();ARTIFACTS.mkdir(exist_ok=True)
    feasibility=json.loads((HERE/'feasibility.json').read_text());assert feasibility['chosen']=='conservative' and tuple(feasibility['provenance']['settings']['conservative'])==PARAMETERS
    if args.extend:
        previous=json.loads((HERE/'screen.json').read_text());assert previous['gate']['passed']
        names=('monet.jpg','eiffel.png')
    else:names=IMAGES
    prior=json.loads((ROOT/'research_pass17/baseline.json').read_text())['rows']
    warm=Image.new('RGB',(40,32),(100,140,170))
    for preset in ('quality','high_quality'):
        for codec in (PBC3,DecoderAwarePBC):codec.compress(warm,config=getattr(PBC3Config,preset)(),rng_mode='release')
    rows=[]
    for image_index,name in enumerate(names):
        image=load(name);original=np.asarray(image,dtype=np.uint8)
        for preset_index,preset in enumerate(('quality','high_quality')):
            config=getattr(PBC3Config,preset)()
            baseline_prep=PBC3.prepare(image,config)
            candidate_prep=DecoderAwarePBC.prepare(image,config)
            assert candidate_prep['orig_compare'].tobytes()==original.tobytes()
            assert np.array_equal(candidate_prep['target'],candidate_prep['arr'].astype(np.int32))
            assert all(candidate_prep[k]==baseline_prep[k] for k in baseline_prep if k not in ('target','arr','orig_compare'))
            if not args.extend:
                expected_target=next(r['target_sha256'] for r in feasibility['rows'] if r['image']==name and r['setting']=='conservative')
                assert sha(candidate_prep['target'].tobytes())==expected_target
            variants=(('ordinary',PBC3),('candidate',DecoderAwarePBC))
            if (image_index+preset_index)%2:variants=variants[::-1]
            for label,codec in variants:
                start=time.perf_counter();result=codec.compress(image,config=config,rng_mode='release');encode=time.perf_counter()-start
                start=time.perf_counter();decoded=PBC3.decompress(result.data);decode=time.perf_counter()-start
                output=np.asarray(decoded.image.convert('RGB'),dtype=np.uint8)
                assert output.shape==original.shape and result.data[:6] in (b'PBC3\0\0',b'PBC3\0\2')
                assert result.image.tobytes()==decoded.image.tobytes()
                mse=float(np.mean((original.astype(np.float32)-output.astype(np.float32))**2,dtype=np.float64))
                assert result.mse==mse
                header,patches,ledger=parse(result.data)
                canvas,*_=PBC3._decode_to_canvas(result.data)
                target=(candidate_prep if label=='candidate' else baseline_prep)['target']
                visible=np.clip(canvas,0,255).astype(np.int32)
                path=ARTIFACTS/(name+'.'+preset+'.'+label+'.pbc');path.write_bytes(result.data)
                row=dict(image=name,preset=preset,variant=label,config=config.__dict__,rng_mode='release',source_size=list(image.size),working_size=[result.working_width,result.working_height],bytes=len(result.data),raw_bytes=ledger['raw_bytes'],bpp=8*len(result.data)/(image.width*image.height),rgb_mse=mse,internal_mse=result.mse,working_target_mse=float(np.mean((target.astype(np.float64)-visible)**2)),ordinary_working_target_mse=float(np.mean((baseline_prep['target'].astype(np.float64)-visible)**2)),working_unclipped_target_mse=float(np.mean((target.astype(np.float64)-canvas)**2)),total_patches=len(patches),initial_patches=sum(r['initialization'] for r in ledger['patches']),iterative_patches=sum(not r['initialization'] for r in ledger['patches']),effective_bits_patch_counts=dict(collections.Counter(str(r['bitcount']) for r in ledger['patches'])),iterative_effective_bits_patch_counts=dict(collections.Counter(str(r['bitcount']) for r in ledger['patches'] if not r['initialization'])),effective_bits_grid_cells={str(bits):sum(r['grid_cells'] for r in ledger['patches'] if r['bitcount']==bits) for bits in sorted(set(r['bitcount'] for r in ledger['patches']))},encode_seconds=encode,decode_seconds=decode,input_sha256=sha((ROOT/'hpt_data'/name).read_bytes()),stream_sha256=sha(result.data),pixels_sha256=sha(output.tobytes()),target_sha256=sha(target.tobytes()),file=str(path.relative_to(HERE)))
                if label=='ordinary':
                    expected=next(r for r in prior if r['image']==name and r['preset']==preset)
                    assert row['stream_sha256']==expected['stream_sha256'] and row['pixels_sha256']==expected['pixels_sha256'] and mse==expected['rgb_mse']
                rows.append(row);print(name,preset,label,row['bytes'],round(mse,4),round(encode,3),round(decode,3),flush=True)
    comparisons=[]
    for name in names:
        for preset in ('quality','high_quality'):
            ordinary=next(r for r in rows if r['image']==name and r['preset']==preset and r['variant']=='ordinary')
            candidate=next(r for r in rows if r['image']==name and r['preset']==preset and r['variant']=='candidate')
            comparisons.append(dict(image=name,preset=preset,rate_change_percent=100*(candidate['bytes']/ordinary['bytes']-1),mse_change_percent=100*(candidate['rgb_mse']/ordinary['rgb_mse']-1),encode_change_percent=100*(candidate['encode_seconds']/ordinary['encode_seconds']-1),decode_change_percent=100*(candidate['decode_seconds']/ordinary['decode_seconds']-1),strict_rd_win=candidate['bytes']<ordinary['bytes'] and candidate['rgb_mse']<ordinary['rgb_mse']))
    useful=[r for r in comparisons if r['mse_change_percent']<=-3 and r['rate_change_percent']<=5]
    gate=dict(useful_points=len(useful),useful_images=len(set(r['image'] for r in useful)),strict_rd_wins=sum(r['strict_rd_win'] for r in comparisons),max_mse_increase_percent=max(r['mse_change_percent'] for r in comparisons),mean_rate_increase_percent=float(np.mean([r['rate_change_percent'] for r in comparisons])))
    gate['passed']=gate['useful_points']>=3 and gate['useful_images']>=2 and gate['strict_rd_wins']>=2 and gate['max_mse_increase_percent']<=5 and gate['mean_rate_increase_percent']<=5
    write('extension.json' if args.extend else 'screen.json',dict(provenance=meta(),extended=args.extend,rows=rows,comparisons=comparisons,gate=gate,timing='One warmed deterministic full encode and ordinary v0 decode per configuration, alternating pair order. Full encode includes actual preparation/preconditioning, patches, LZMA and enabled original-source internal MSE. Loading/hash/extra diagnostic prepare/stream parsing/working scores/IO excluded. Ordinary streams exactly reproduce frozen Pass17 controls. Feasibility timings are not repeated performance claims.'))
    print('RD gate',gate,flush=True)


if __name__=='__main__':main()
