import json
import lzma
import struct
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research_pass17.analyze_residuals import PBC3, PBC3Config, Image, load, np, provenance, setup, sha
from research_pass18.proposals import learned_action
import learned_filler as lf
import pbc3_ops as ops
from pbc3_types import BitWriter
from research_pass19 import masks

HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / 'artifacts'
IMAGES = ('googlewiki.png', 'mario64.png', 'eiffel.png', 'cat.jpg')
START = 'c545c56686416f59b1c317658034ad93fdb78a7e'


def write_json(name, value):
    (HERE / name).write_text(json.dumps(value, separators=(',', ':'))+'\n', encoding='utf-8', newline='\n')


def base_stream(name):
    data = (ROOT/'research_pass17/artifacts'/(name+'.dct_q128.p17')).read_bytes()
    header = struct.Struct('<4sIIIIfBBI')
    values = header.unpack_from(data)
    assert values[0] == b'P17D'
    return data[header.size:header.size+values[3]]


def render(canvas, domain, size):
    if domain == 'original': return np.clip(canvas, 0, 255).astype(np.uint8)
    image = PBC3._canvas_to_image(canvas, 'YCbCr', False)
    if image.size != size:
        image = image.resize(size, PBC3.RESAMPLE_FILTER, reducing_gap=PBC3.RESAMPLE_REDUCING_GAP)
    return np.asarray(image)


def sse(original, output):
    difference = original.astype(np.int32) - output.astype(np.int32)
    return int(np.sum(difference.astype(np.int64)**2))


def evaluate(target, canvas, box, config, model, step, original, domain, baseline_sse):
    c,x,y,w,h = box
    record, delta = masks.fit(target, canvas, box)
    payload = masks.pack([record])
    decoded = masks.unpack(payload, canvas.shape[1], canvas.shape[0])
    reconstructed = canvas.copy()
    masks.apply(reconstructed, decoded[0])
    size = (original.shape[1], original.shape[0])
    gain = baseline_sse - sse(original, render(reconstructed, domain, size))
    visible = target[y:y+h,x:x+w,c] - np.clip(canvas[y:y+h,x:x+w,c],0,255)
    after = target[y:y+h,x:x+w,c] - np.clip(canvas[y:y+h,x:x+w,c]+delta,0,255)
    hard = dict(gain=gain, working_gain=int(np.sum(visible.astype(np.int64)**2)-np.sum(after.astype(np.int64)**2)), encoded_bytes=len(payload), estimated_raw_bits=masks.GROUP.size*8+masks.RECORD.size*8+w*h*2, raw_bytes=masks.GROUP.size+len(masks.record_bytes(record)), mask_bytes=(w*h+3)//4, level_bytes=4, geometry_channel_bytes=9, group_bytes=8, levels=[record['lo'],record['hi']], zero_fraction=float(np.mean(record['mask']==0)), gain_per_bit=gain/(8*len(payload)), payload_sha256=sha(payload))
    action = learned_action(model,target,canvas,box,config,step)
    choices = [('model',lf.ACTIONS[action]),('cell1_bits2',(1,2)),('cell2_bits3',(2,3))]
    native = []
    for name,(cell,bits) in choices:
        patch, values, correction, reduction, rawbits = lf._build_action(target,canvas,box,config,2,cell,bits)
        writer = BitWriter()
        PBC3._write_patch(writer,patch,2)
        raw = struct.pack('<4sI',b'V019',1) + writer.finish()
        packed = lzma.compress(raw,format=lzma.FORMAT_XZ,preset=lzma.PRESET_EXTREME)
        changed = canvas.copy()
        ops.apply_delta(changed[:,:,c],x,y,w,h,correction)
        rgb_gain = baseline_sse - sse(original,render(changed,domain,size))
        native.append(dict(kind=name,cell=cell,requested_bits=bits,effective_bits=patch['bitcount'],gain=rgb_gain,working_gain=reduction,raw_bits=int(rawbits)+64,raw_bytes=len(raw),encoded_bytes=len(packed),gain_per_bit=rgb_gain/(8*len(packed))))
    comparable = [r for r in native if gain > 0 and .8*gain <= r['gain'] <= gain/.8]
    strongest = max(comparable,key=lambda r:r['gain_per_bit']) if comparable else None
    win = strongest is not None and hard['gain_per_bit'] >= 1.2*strongest['gain_per_bit']
    return record,dict(box=list(box),hard=hard,native=native,comparable_best=strongest['kind'] if strongest else None,efficiency_ratio=hard['gain_per_bit']/strongest['gain_per_bit'] if strongest and strongest['gain_per_bit']>0 else None,representation_gate=win)


def main():
    setup(); ARTIFACTS.mkdir(exist_ok=True)
    meta = provenance(); meta.update(research_start_commit=START,timing='One-pass offline feasibility; all analysis/fitting/entropy costs measured separately. No completed-codec or runtime claim.',plan_sha256=sha((HERE/'PLAN.md').read_bytes()),research_source_sha256={p.name:sha(p.read_bytes()) for p in HERE.glob('*.py')})
    diagnostics = json.loads((ROOT/'research_pass18/diagnostics.json').read_text())
    extra = json.loads((ROOT/'research_pass17/extra_controls.json').read_text())['rows']
    rows, groups = [], []
    for name in IMAGES:
        original = np.asarray(load(name),dtype=np.uint8)
        states = [s for s in diagnostics['states'] if s['selected']['image']==name and s['selected']['preset']=='quality' and s['selected']['step'] in (3,8)]
        for state in states:
            arrays = np.load(ROOT/'research_pass18'/state['file'])
            target,canvas = arrays['target'],arrays['canvas']
            config = PBC3Config(**state['config'])
            model = lf.LearnedFiller.load(config.learned_filler_model_path,q_override=config.learned_filler_q)
            start=time.perf_counter(); proposed=masks.proposals(target,canvas,2); proposal_time=time.perf_counter()-start
            boxes=list(dict.fromkeys([tuple(state['selected']['box'])]+proposed))
            before=sse(original,render(canvas,'working',(original.shape[1],original.shape[0])))
            selected=lf.build_action(target,canvas,tuple(state['selected']['box']),config,2,state['selected']['action'])
            after=canvas.copy(); patch=selected[0]
            ops.apply_delta(after[:,:,patch['channel']],patch['x'],patch['y'],patch['w'],patch['h'],selected[2])
            selected_gain=before-sse(original,render(after,'working',(original.shape[1],original.shape[0])))
            records=[]
            for index,box in enumerate(boxes):
                start=time.perf_counter(); record,row=evaluate(target,canvas,box,config,model,state['selected']['step'],original,'working',before); cost=time.perf_counter()-start
                row.update(image=name,domain='working',state=state['id'],index=index,baseline_rgb_sse=before,ordinary_selected_rgb_gain=selected_gain,proposal_seconds=proposal_time,evaluation_seconds=cost)
                row['substantial_region']=index==0 or row['hard']['gain']>=.8*selected_gain
                row['gate']=row['representation_gate'] and row['substantial_region']
                rows.append(row); records.append(record)
            payload=masks.pack(records)
            (ARTIFACTS/(state['id']+'.m19')).write_bytes(payload)
            groups.append(dict(image=name,domain='working',state=state['id'],records=len(records),shared_bytes=len(payload),separate_bytes=sum(r['hard']['encoded_bytes'] for r in rows if r['state']==state['id']),file_sha256=sha(payload)))
            print(name,'working',state['selected']['step'],'wins',sum(r['gate'] for r in rows if r['state']==state['id']),flush=True)
        base=base_stream(name)
        base_pixels=np.asarray(PBC3.decompress(base).image.convert('RGB'),dtype=np.int32)
        prior=next(r for r in extra if r['image']==name and r['preset']=='quality_p10_base')
        assert sha(base)==prior['stream_sha256'] and sha(base_pixels.astype(np.uint8).tobytes())==prior['pixels_sha256']
        before=sse(original,base_pixels)
        config=PBC3Config.quality(color_space='RGB',patch_count=10)
        model=lf.LearnedFiller.load(str(ROOT/'patch_policy.npz'),q_override=config.learned_filler_q)
        start=time.perf_counter(); boxes=masks.proposals(original,base_pixels); proposal_time=time.perf_counter()-start
        records=[]
        for index,box in enumerate(boxes):
            start=time.perf_counter(); record,row=evaluate(original.astype(np.int32),base_pixels,box,config,model,10,original,'original',before); cost=time.perf_counter()-start
            row.update(image=name,domain='original',state=name+'.base_p10',index=index,baseline_rgb_sse=before,base_bytes=len(base),base_sha256=sha(base),proposal_seconds=proposal_time,evaluation_seconds=cost)
            row['substantial_region']=row['hard']['gain']>=.005*before
            row['gate']=row['representation_gate'] and row['substantial_region']
            rows.append(row); records.append(record)
        payload=masks.pack(records)
        (ARTIFACTS/(name+'.original.m19')).write_bytes(payload)
        groups.append(dict(image=name,domain='original',state=name+'.base_p10',records=len(records),shared_bytes=len(payload),separate_bytes=sum(r['hard']['encoded_bytes'] for r in rows if r['state']==name+'.base_p10'),file_sha256=sha(payload)))
        print(name,'original','wins',sum(r['gate'] for r in rows if r['state']==name+'.base_p10'),flush=True)
    gates={}
    for domain in ('working','original'):
        winners=[r for r in rows if r['domain']==domain and r['gate']]
        gates[domain]=dict(regions=len(winners),images=sorted({r['image'] for r in winners}),passed=len(winners)>=3 and len({r['image'] for r in winners})>=2)
    write_json('feasibility.json',dict(provenance=meta,rows=rows,groups=groups,gates=gates))
    provenance()
    print(json.dumps(gates),flush=True)


if __name__=='__main__': main()
