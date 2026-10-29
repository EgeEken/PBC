import time
import tracemalloc
from common import *
from research_pass25 import base,ordinary,jpeg_reference
from research_pass17.report import interpolate


def main():
    setup();ARTIFACTS.mkdir(exist_ok=True);rows=[];parents=[];config=PBC3Config.quality()
    for name in IMAGES:
        image=load(name);source=np.asarray(image);prep=PBC3.prepare(image,config)
        saved_data,parent=saved(name,'quality');header,patches,accounting=parse(saved_data)
        expected=serialize(header,patches[:3]);hq,_=saved(name,'high_quality');hh,hp,_=parse(hq);assert serialize(hh,hp[:3])==expected
        target_rgb=np.asarray(PBC3._canvas_to_image(prep['target'],prep['color_space'],False))
        parents.append(dict(image=name,ordinary_config=config.__dict__,parent_stream_sha256=sha(saved_data),initialization_stream_sha256=sha(expected),working_size=[prep['w'],prep['h']],source_size=list(image.size),raw_initial_patch_bits=[ops.patch_bits_for(p,header[7]) for p in patches[:3]],full_file_bytes=len(saved_data)))
        variants=[('ordinary',None)]+[('transform',step) for step in STEPS]+[('jpeg',1)]
        def encode(kind,step):return ordinary.encode(prep,config) if kind=='ordinary' else base.encode(prep,step) if kind=='transform' else jpeg_reference.encode(prep,step)
        def decode(kind,data):return PBC3.decompress(data).image if kind=='ordinary' else base.decode(data) if kind=='transform' else jpeg_reference.decode(data)
        for kind,step in variants:
            data=encode(kind,step);decode(kind,data)
            start=time.perf_counter();data=encode(kind,step);enc=time.perf_counter()-start
            start=time.perf_counter();decoded=decode(kind,data);dec=time.perf_counter()-start
            if kind=='ordinary':assert data==expected
            if kind=='jpeg':
                working,_=jpeg_reference.decode_working(data);working_rgb=np.asarray(working);canvas=np.asarray(working.convert(prep['color_space']),dtype=np.int16);n=nnz=None
            else:
                canvas=PBC3._decode_to_canvas(data)[0] if kind=='ordinary' else base.decode_canvas(data)[0]
                working_rgb=np.asarray(PBC3._canvas_to_image(canvas,prep['color_space'],False))
                if kind=='transform':
                    info,payload=base.split(data);n=((prep['h']+7)//8)*((prep['w']+7)//8)*192
                    raw=base.zero_runs.decompress(payload,n);_,_,nnz=base.zero_runs.HEADER.unpack_from(raw)
                else:n=nnz=None
            ordinary_canvas=PBC3._decode_to_canvas(expected)[0]
            tracemalloc.start();traced_output=decode(kind,data);current,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
            assert traced_output.tobytes()==decoded.tobytes()
            path=ARTIFACTS/(name+('.ordinary_init.pbc' if kind=='ordinary' else f'.q{step}.p25i' if kind=='transform' else '.jpeg_q1.jpg'));path.write_bytes(data)
            row=dict(image=name,variant=kind,step=step,bytes=len(data),bpp=8*len(data)/(image.width*image.height),rgb_mse=mse(source,np.asarray(decoded)),working_component_mse=mse(prep['target'],np.clip(canvas,0,255)),working_rgb_mse=mse(target_rgb,working_rgb),working_canvas_difference_from_ordinary_mse=mse(canvas,ordinary_canvas),encode_seconds=enc,decode_seconds=dec,coefficient_count=n,nonzero_count=nnz,canvas_bytes=canvas.nbytes,float_coefficient_buffer_bytes=(n*4 if n else None),full_rgb_bytes=len(decoded.tobytes()),traced_decode_peak=peak,traced_decode_current=current,stream_sha256=sha(data),pixels_sha256=sha(decoded.tobytes()),canvas_sha256=sha(canvas.tobytes()),file=str(path.relative_to(HERE)))
            rows.append(row);print(name,kind,step,row['bytes'],round(row['rgb_mse'],3),round(enc,3),round(dec,3),flush=True)
    comparison=[];classes={'googlewiki.png':'text','monet.jpg':'art','cat.jpg':'photograph','flower.jpg':'photograph'}
    for name in IMAGES:
        control=next(r for r in rows if r['image']==name and r['variant']=='ordinary');points=[dict(r,preset=str(r['step'])) for r in rows if r['image']==name and r['variant']=='transform']
        actual=[p for p in points if (p['bytes']<=control['bytes']*1.05 and p['rgb_mse']<=control['rgb_mse']*.9) or (p['bytes']<=control['bytes']*.9 and p['rgb_mse']<=control['rgb_mse']*1.02)]
        bracket=interpolate(points,control['bytes'],'bytes','rgb_mse')
        advantage=bool(actual) or bool(bracket and bracket['value']<=control['rgb_mse']*.9)
        plausible=[p for p in points if p['decode_seconds']<=control['decode_seconds']*2 or p['decode_seconds']-control['decode_seconds']<=.05]
        comparison.append(dict(image=name,image_class=classes[name],ordinary_bytes=control['bytes'],ordinary_mse=control['rgb_mse'],actual_advantage_steps=[p['step'] for p in actual],bracket_at_ordinary_bytes=bracket,interpolated_mse_change_percent=100*(bracket['value']/control['rgb_mse']-1) if bracket else None,rd_advantage=advantage,plausible_decode_steps=[p['step'] for p in plausible]))
    passed=len(set(r['image_class'] for r in comparison if r['rd_advantage'] and r['plausible_decode_steps']))>=2
    write('feasibility.json',dict(provenance=meta(),parents=parents,rows=rows,comparisons=comparison,passed=passed,frozen_integrated_step=128,timing='Initializer-only complete creation and source-size RGB decode, common original prepare excluded. One actual warmup and one timed encode/decode per fixed setting. Separate traced decode is allocation evidence, not another timing repetition. Tracemalloc includes NumPy/Python tracked buffers, not all native PIL memory. Reused ordinary prefixes validated against fresh selection and Q/HQ agreement. JPEG uses one q1 point and14-byte size comment.'))
    print('initialization gate',passed,comparison,flush=True)


if __name__=='__main__':main()
