import json
import lzma
import struct
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research_pass19.feasibility import ARTIFACTS,HERE,PBC3,PBC3Config,base_stream,lf,load,np,ops,provenance,render,setup,sha,sse,write_json,BitWriter,learned_action
from research_pass19 import masks


def main():
    setup()
    measured=json.loads((HERE/'feasibility.json').read_text())
    captured=json.loads((ROOT/'research_pass18/diagnostics.json').read_text())['states']
    checked=[]
    for group in measured['groups']:
        original=np.asarray(load(group['image']),dtype=np.uint8)
        records_rows=[r for r in measured['rows'] if r['domain']==group['domain'] and r['state']==group['state']]
        if group['domain']=='working':
            state=next(s for s in captured if s['id']==group['state'])
            arrays=np.load(ROOT/'research_pass18'/state['file'])
            target,canvas=arrays['target'],arrays['canvas']
            config=PBC3Config(**state['config']); step=state['selected']['step']
        else:
            target=original.astype(np.int32)
            canvas=np.asarray(PBC3.decompress(base_stream(group['image'])).image.convert('RGB'),dtype=np.int32)
            config=PBC3Config.quality(color_space='RGB',patch_count=10); step=10
        path=ARTIFACTS/(group['state']+'.m19' if group['domain']=='working' else group['image']+'.original.m19')
        payload=path.read_bytes()
        assert sha(payload)==group['file_sha256'] and len(payload)==group['shared_bytes']
        decoded=masks.unpack(payload,canvas.shape[1],canvas.shape[0])
        assert len(decoded)==len(records_rows)==group['records']
        model=lf.LearnedFiller.load(str(ROOT/'patch_policy.npz'),q_override=config.learned_filler_q)
        writers={k:BitWriter() for k in ('model','cell1_bits2','cell2_bits3')}
        for record,row in zip(decoded,records_rows):
            fitted,delta=masks.fit(target,canvas,tuple(row['box']))
            assert record['box']==fitted['box'] and (record['lo'],record['hi'])==(fitted['lo'],fitted['hi'])
            assert np.array_equal(record['mask'],fitted['mask'])
            individual=masks.pack([record])
            assert len(individual)==row['hard']['encoded_bytes'] and sha(individual)==row['hard']['payload_sha256']
            output=canvas.copy(); masks.apply(output,record)
            assert sse(original,render(canvas,group['domain'],(original.shape[1],original.shape[0])))-sse(original,render(output,group['domain'],(original.shape[1],original.shape[0])))==row['hard']['gain']
            action=learned_action(model,target,canvas,record['box'],config,step)
            for kind,(cell,bits) in [('model',lf.ACTIONS[action]),('cell1_bits2',(1,2)),('cell2_bits3',(2,3))]:
                built=lf._build_action(target,canvas,record['box'],config,2,cell,bits)
                PBC3._write_patch(writers[kind],built[0],2)
        native_bytes={k:len(lzma.compress(struct.pack('<4sI',b'V019',len(decoded))+w.finish(),format=lzma.FORMAT_XZ,preset=lzma.PRESET_EXTREME)) for k,w in writers.items()}
        checked.append(dict(state=group['state'],domain=group['domain'],records=len(decoded),hard_shared_bytes=len(payload),native_shared_bytes=native_bytes))
    fixture=dict(box=(1,2,1,13,9),lo=-200,hi=170,mask=np.tile(np.array([0,1,2],dtype=np.uint8),(9,5))[:,:13])
    packet=masks.pack([fixture]); decoded=masks.unpack(packet,17,13)
    canvas=np.full((13,17,3),128,dtype=np.int32)
    masks.apply(canvas,decoded[0])
    expected=np.full((13,17,3),128,dtype=np.int32)
    expected[1:10,2:15,1]+=np.array([0,-200,170],dtype=np.int16)[fixture['mask']]
    assert np.array_equal(canvas,expected)
    assert set(np.unique(np.clip(canvas,0,255)))=={0,128,255}
    assert masks.pack(decoded)==packet
    assert masks.unpack(masks.pack([]),17,13)==[]
    failures=0
    invalid=[packet[:-1],packet+b'junk']
    raw=masks.GROUP.pack(b'M19R',1)+masks.RECORD.pack(0,0,0,1,1,-1,1)
    invalid.extend(lzma.compress(raw+bytes([x]),format=lzma.FORMAT_XZ,preset=lzma.PRESET_EXTREME) for x in (192,1))
    invalid.append(lzma.compress(raw+b'\x00extra',format=lzma.FORMAT_XZ,preset=lzma.PRESET_EXTREME))
    for bad in invalid:
        try: masks.unpack(bad,17,13)
        except (ValueError,lzma.LZMAError,EOFError): failures+=1
        else: raise AssertionError('malformed mask accepted')
    try: masks.unpack(packet,3,3)
    except ValueError: failures+=1
    else: raise AssertionError('out-of-bounds mask accepted')
    meta=provenance(); meta.update(timing='Correctness verification and existing-region shared entropy accounting only; no full-codec timing.',research_source_sha256={p.name:sha(p.read_bytes()) for p in HERE.glob('*.py')})
    write_json('validation.json',dict(provenance=meta,groups=checked,roundtripped_regions=sum(r['records'] for r in checked),fixture='non-aligned signed ternary with zero transparency, sharp boundaries and final clipping',deterministic_fixture=True,empty_group=True,malformed_rejections=failures,production_unchanged=True))
    print('validated',sum(r['records'] for r in checked),'regions;',failures,'malformed inputs rejected')
    for row in checked: print(row)


if __name__=='__main__': main()
