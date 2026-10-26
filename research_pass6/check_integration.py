import csv
import base64
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
ARCHIVE=Path(r'C:\Users\EGE\Desktop\Coding\Personal Projects\Completed Projects\Image Processing\Probabilistic Brush Compression\New Demo\HF Space\PBC')
sys.path.insert(0,str(ROOT))
from worker import THREADS
os.environ.update(THREADS)


def request(process,job):
    process.stdin.write(json.dumps(job)+'\n')
    process.stdin.flush()
    row=json.loads(process.stdout.readline())
    if 'error' in row:
        raise RuntimeError(row['error'])
    return row


def main():
    import numpy as np
    import pbc3_stream as stream
    import pbc3_ops as ops
    from pbc3_types import BitWriter,BitReader,PBC3Config
    checks=0
    for width in range(17):
        for offset in range(8):
            for count in (0,33):
                values=np.arange(count*2,dtype=np.int64)[::2]-17
                a,b=BitWriter(),BitWriter()
                for writer in (a,b):
                    writer.write((1<<offset)-1,offset)
                stream._write_grid_scalar(a,values,width)
                stream.write_grid(b,values,width)
                for writer in (a,b):
                    writer.write(37,6)
                assert a.finish()==b.finish()
                a,b=BitReader(a.finish()),BitReader(b.finish())
                a.read(offset); b.read(offset)
                assert np.array_equal(stream._read_grid_scalar(a,count,width),stream.read_grid(b,count,width))
                assert (a.i,a.acc,a.nbits)==(b.i,b.acc,b.nbits)
                assert a.read(6)==b.read(6)==37
                checks+=1
    for data in (b'\x12\x34\x56\x78',memoryview(b'\x12\x00\x34\x00\x56\x00\x78\x00')[::2]):
        a,b=BitReader(data),BitReader(data)
        a.read(3); b.read(3)
        assert np.array_equal(stream._read_grid_scalar(a,5,5),stream.read_grid(b,5,5))
    for values in (np.array([-2**63,-1,0,2**63-1],dtype=np.int64),np.array([0,2**64-1],dtype=np.uint64)):
        for width in (1,9,16):
            a,b=BitWriter(),BitWriter()
            stream._write_grid_scalar(a,values,width)
            stream.write_grid(b,values,width)
            assert a.finish()==b.finish()
    for count,width in ((1,9),(10**12,9)):
        try:
            stream.read_grid(BitReader(b'\x00'),count,width)
        except EOFError:
            pass
        else:
            raise AssertionError('truncated grid accepted')
    rng=np.random.default_rng(88)
    target=rng.integers(0,256,(39,43,4),dtype=np.int32)
    canvas=rng.integers(-400,700,(39,43,4),dtype=np.int16)
    state=ops.SearchState()
    cfg=PBC3Config.balanced()
    for i in range(20):
        c=i%4
        expected=np.abs(target[:,:,c]-np.clip(canvas[:,:,c],0,255))
        assert np.array_equal(state.error(target,canvas,c),expected)
        assert state.score(target,canvas,c)==float(np.sum(expected))
        a,b=ops.PBC3Rng(2003+i),ops.PBC3Rng(2003+i)
        assert ops.search_boxes(target,canvas,cfg,a,c,40,.5,ordered=True)==ops.search_boxes(target,canvas,cfg,b,c,40,.5,ordered=True,abs_error=state.error(target,canvas,c))
        assert a.state==b.state
        canvas[3:19,5:27,c]+=rng.integers(-200,200,(16,22),dtype=np.int16)
        state.update(target,canvas,c,5,3,22,16)
    replacement=canvas.copy()
    replacement[:]=99
    assert np.array_equal(state.error(target,replacement,0),np.abs(target[:,:,0]-99))
    processes={}
    metadata={}
    rows=[]
    try:
        for source in ('baseline','integrated'):
            processes[source]=subprocess.Popen([sys.executable,str(HERE/'worker.py'),'--source',source],cwd=ROOT,
                env={**os.environ,**THREADS},stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1)
            metadata[source]=request(processes[source],dict(op='metadata'))
        presets=('compression','balanced','quality','high_quality')
        for source in processes:
            for preset in presets:
                request(processes[source],dict(image='googlewiki.png',preset=preset))
        with (ARCHIVE/'research_pass4'/'runtime_full.csv').open(newline='',encoding='utf-8') as f:
            frozen={(r['dataset'],r['image'],r['preset']):r for r in csv.DictReader(f) if r['label']=='baseline'}
        for (dataset,image,preset),old in sorted(frozen.items()):
            result=request(processes['integrated'],dict(dataset=dataset,image=image,preset=preset))
            assert result['stream_sha256']==old['stream_sha256'],(image,preset,'stream')
            assert result['pixels_sha256']==old['pixels_sha256'],(image,preset,'pixels')
            assert result['mse']==float(old['mse']) and result['final_bytes']==int(old['final_bytes'])
            assert result['patches']==int(old['patch_count']) and result['preview_equal']
            rows.append(result)
            print('frozen',len(rows),image,preset,flush=True)
        (HERE/'integration_rows.json').write_text(json.dumps(rows,indent=2)+'\n')
        edge=[]
        for index in range(20):
            spec=dict(shape=[1,1,3]) if index==0 else dict(shape=[71,97,4 if index%2 else 3],opaque=index%4==1,seed=2003+index)
            overrides=dict(patch_count=0 if index<3 else 4,auto_downsample_max_pixels=4096 if index%3==0 else 250000,
                reuse_selected_delta=index%2==0,learned_filler_enabled=index%5!=0,compute_final_mse=index%4!=0)
            if index in (16,17):
                overrides.update(warmup_ratio=.5,warm_downsample_max_pixels=6000,auto_downsample_max_pixels=3000)
            job=dict(synthetic=spec,preset=presets[index%4],overrides=overrides,payload=True)
            a=request(processes['baseline'],job)
            b=request(processes['integrated'],job)
            assert all(a[k]==b[k] for k in ('stream_sha256','pixels_sha256','trajectory_sha256','patches','mse','internal_mse','bpp')),(index,a,b)
            assert request(processes['integrated'],dict(op='decode',payload=a['payload']))['pixels_sha256']==a['pixels_sha256']
            assert request(processes['baseline'],dict(op='decode',payload=b['payload']))['pixels_sha256']==b['pixels_sha256']
            edge.append(dict(case=index,exact=True))
        from PBC3 import PBC3
        data=base64.b64decode(a['payload'])
        _,body=PBC3._open_body(data)
        stored=b'PBC3\x00\x00'+body
        malformed=[stored[:n] for n in (0,5,7,len(stored)//2,len(stored)-1)]
        malformed += [data[:-1],data[:4]+b'\x7f'+data[5:],data[:5]+b'\xff'+data[6:]]
        reader=BitReader(body)
        header=PBC3._read_header(reader)
        flag=reader.i*8-reader.nbits+header[7]+64
        corrupt=bytearray(body)
        corrupt[flag//8] |= 1<<(7-flag%8)
        malformed.append(b'PBC3\x00\x00'+corrupt)
        for data in malformed:
            job=dict(op='decode_outcome',payload=base64.b64encode(data).decode())
            assert request(processes['baseline'],job)==request(processes['integrated'],job)
        concurrent=request(processes['integrated'],dict(op='concurrent',cases=[dict(synthetic=dict(shape=[63,71,c],seed=3000+c),overrides=dict(patch_count=3),preset='quality') for c in (3,4)]))
        result=dict(grid_cases=checks,proposal_rng_checks=20,frozen_cases=len(rows),edges=edge,malformed_cases=len(malformed),concurrent=concurrent,metadata=metadata,all_exact=True)
        (HERE/'integration_checks.json').write_text(json.dumps(result,indent=2)+'\n')
        (HERE/'integration_rows.json').write_text(json.dumps(rows,indent=2)+'\n')
        print(json.dumps(dict(grid_cases=checks,frozen_cases=len(rows),edge_cases=len(edge),all_exact=True)),flush=True)
    finally:
        for process in processes.values():
            if process.poll() is None:
                request(process,dict(op='quit'))
                process.wait(timeout=20)


if __name__=='__main__':
    main()
