import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from research_pass8.worker import THREADS
os.environ.update(THREADS)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--archive',type=Path,required=True,help='Existing research_pass1/streams/current/hpt_data directory')
    args=parser.parse_args()
    import numpy as np
    from PIL import Image,ImageOps
    from PBC3 import PBC3,_canvas_from_bases
    from pbc3_types import PBC3Config,BitWriter,BitReader
    import pbc3_ops as ops
    from research_pass2.stream_tools import parse
    from palette import bounded_palette,local_sse
    images=('cat.jpg','flower.jpg','monet.jpg','mario64.png','eiffel.png','googlewiki.png')
    expected=json.loads((ROOT/'research_pass7/stop_screen.json').read_text())['rows']
    rows=[]; controls=[]
    for image in images:
        for preset in ('quality','high_quality'):
            path=args.archive/(image+'.'+preset+'.pbc3')
            data=path.read_bytes()
            digest=hashlib.sha256(data).hexdigest()
            frozen=next(r for r in expected if (r['image'],r['preset'],r['mode'])==(image,preset,'control'))
            assert digest==frozen['stream_sha256'],path
            header,patches,ledger=parse(data)
            with Image.open(ROOT/'hpt_data'/image) as opened:
                source=ImageOps.exif_transpose(opened).convert('RGB')
            config=getattr(PBC3Config,preset)()
            prep=PBC3.prepare(source,config)
            target=prep['target']
            canvas=_canvas_from_bases(target.shape,header[11])
            assert (prep['w'],prep['h'])==(header[3],header[4]) and not header[12]
            controls.append(dict(image=image,preset=preset,stream_sha256=digest,patches=len(patches),
                input_sha256=hashlib.sha256((ROOT/'hpt_data'/image).read_bytes()).hexdigest()))
            for index,patch in enumerate(patches):
                c,x,y,w,h=(patch[k] for k in ('channel','x','y','w','h'))
                cfg=dataclasses.replace(config,mask_size=len(patch['mask']))
                residual=target[y:y+h,x:x+w,c]-canvas[y:y+h,x:x+w,c]
                small=np.rint(ops.project_residual_to_grid(residual,patch['cell_size'],cfg.residual_projection_mode)).astype(np.int16)
                original,values=ops.make_patch(c,x,y,w,h,patch['cell_size'],residual,cfg,patch['max_bitcount'])
                assert all(original[k]==patch[k] for k in ('mask','neg','pos','bitcount')) and np.array_equal(original['indices'],patch['indices'])
                base_sse,base_delta=local_sse(target,canvas,patch,values)
                base_bits=ops.patch_bits_for(patch,header[7])
                assert base_bits==ledger['patches'][index]['total_bits']
                for percentile in (98,95):
                    candidate,candidate_values=bounded_palette(small,patch,cfg,percentile)
                    sse,delta=local_sse(target,canvas,candidate,candidate_values)
                    bits=ops.patch_bits_for(candidate,header[7])
                    writer=BitWriter(); PBC3._write_patch(writer,candidate,header[7])
                    assert len(writer.data)*8+writer.nbits==bits
                    reader=BitReader(writer.finish())
                    decoded=PBC3._read_patch(reader,header[7],header[8])
                    assert np.array_equal(decoded[6],candidate_values)
                    rows.append(dict(image=image,preset=preset,index=index,initialization=index<3,percentile=percentile,
                        base_sse=base_sse,sse=sse,base_bits=base_bits,bits=bits,
                        bounds_changed=(candidate['neg'],candidate['pos'])!=(patch['neg'],patch['pos']),
                        decoded_changed=not np.array_equal(delta,base_delta)))
                ops.apply_delta(canvas[:,:,c],x,y,w,h,base_delta)
            reconstructed=PBC3._canvas_to_image(canvas,config.color_space,False)
            if header[0]:
                reconstructed=reconstructed.resize(source.size,PBC3.RESAMPLE_FILTER,reducing_gap=PBC3.RESAMPLE_REDUCING_GAP)
            assert np.array_equal(np.asarray(reconstructed),np.asarray(PBC3.decompress(data).image))
            print(image,preset,len(patches),flush=True)
    (HERE/'palette_fixed_raw.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows),newline='\n')
    summary={}
    for percentile in (98,95):
        group=[r for r in rows if r['percentile']==percentile]
        useful=[r for r in group if r['sse']<r['base_sse'] and r['bits']<=r['base_bits']]
        summary[str(percentile)]=dict(patches=len(group),bounds_changed=sum(r['bounds_changed'] for r in group),
            distortion_wins_no_more_bits=len(useful),bits_wins_no_more_distortion=sum(r['bits']<r['base_bits'] and r['sse']<=r['base_sse'] for r in group),
            lower_sse_more_bits=sum(r['sse']<r['base_sse'] and r['bits']>r['base_bits'] for r in group),
            lower_bits_more_sse=sum(r['bits']<r['base_bits'] and r['sse']>r['base_sse'] for r in group),
            useful_images=len({r['image'] for r in useful}),
            useful_local_sse_reduction=sum(r['base_sse']-r['sse'] for r in useful),
            evaluated_baseline_sse=sum(r['base_sse'] for r in group))
    # Freeze this feasibility gate before considering any forward integration.
    for s in summary.values():
        s['gate_pass']=s['distortion_wins_no_more_bits']>=.1*s['patches'] and s['useful_images']>=4 and s['useful_local_sse_reduction']>=.001*s['evaluated_baseline_sse']
    result=dict(controls=controls,summary=summary,percentiles=[98,95],gate='At least 10% useful patches, four images, and 0.1% summed evaluated local SSE opportunity; local opportunity is not final-file RD.',
        searches=0,roundtrip_patch_checks=len(rows),
        raw_sha256=hashlib.sha256((HERE/'palette_fixed_raw.jsonl').read_bytes()).hexdigest(),
        source_lf={p:hashlib.sha256((ROOT/p).read_text(encoding='utf-8').encode()).hexdigest() for p in ('research_pass8/palette.py','research_pass8/palette_feasibility.py','pbc3_ops.py')})
    (HERE/'palette_feasibility.json').write_text(json.dumps(result,indent=2)+'\n',newline='\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
