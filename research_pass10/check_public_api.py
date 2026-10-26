import argparse
import ast
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.error
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from research_pass5 import run_hf as api
from research_pass8.run_hf import token
from research_pass10.check_codec import THREADS
os.environ.update(THREADS)
api.HERE=HERE


def post(path,image,fields,filename='input.png',part='image'):
    boundary='PBCDefaultPromotion'
    body=b''
    for name,value in fields.items():
        body+=f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode()
    body+=f'--{boundary}\r\nContent-Disposition: form-data; name="{part}"; filename="{filename}"\r\nContent-Type: application/octet-stream\r\n\r\n'.encode()+image+f'\r\n--{boundary}--\r\n'.encode()
    req=urllib.request.Request(api.HOST+path,data=body,headers={'Content-Type':'multipart/form-data; boundary='+boundary})
    return urllib.request.urlopen(req,timeout=180).read()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stream-reference',action='store_true')
    args=parser.parse_args()
    if args.stream_reference:
        sys.path.insert(0,str(ROOT/'research_pass1/release'))
    from PIL import Image,ImageOps
    import numpy as np
    from PBC3 import PBC3
    from pbc3_types import PBC3Config
    fixture=HERE/'hf_runs/fixtures'
    if args.stream_reference:
        image=Image.open(fixture/'native.png').convert('RGB')
        config=PBC3Config.high_quality(auto_downsample_init=False,patch_count=150,max_patch_size=48,min_cell_size=1,max_cell_size=16,
            q_init=.5,q_start=.7,q_end=.6,search_q_start=.6,search_q_end=.4,learned_filler_model_path=str(ROOT/'patch_policy.npz'))
        encoded=PBC3.compress(image,config=config)
        print(json.dumps(dict(config=vars(config),stream_sha256=hashlib.sha256(encoded.data).hexdigest(),pixels_sha256=hashlib.sha256(np.asarray(PBC3.decompress(encoded.data).image).tobytes()).hexdigest())))
        return
    deployed=json.loads((HERE/'hf_deployment.json').read_text())
    meta=api.metadata(); assert meta['sha']==meta['runtime']['sha']==deployed['commit']
    assert not (HERE/'public_checks.json').exists(),'Preserve earlier evidence before making fresh requests'
    frozen=json.loads((ROOT/'research_pass9/comparison_summary.json').read_text())
    with Image.open(ROOT/'hpt_data/googlewiki.png') as original:
        source=ImageOps.exif_transpose(original).convert('RGB')
    buf=io.BytesIO(); source.save(buf,format='PNG'); rgb=buf.getvalue()
    records=[]
    for preset in ('compression','balanced','quality','high_quality'):
        expected=next(p['release'] for p in frozen['equivalence'] if p['image']=='googlewiki.png' and p['preset']==preset)
        output=json.loads(post('/api/compress',rgb,dict(mode='Auto',auto_config=preset,rng_mode='current')))
        stream=base64.b64decode(output['pbc_base64']); decoded=PBC3.decompress(stream).image
        preview=Image.open(io.BytesIO(base64.b64decode(output['reconstructed_image'].split(',')[-1])))
        assert hashlib.sha256(stream).hexdigest()==expected['stream_sha256'],preset
        assert np.array_equal(np.asarray(preview),np.asarray(decoded)) and decoded.size==source.size
        mse=float(np.mean(np.subtract(np.asarray(source),np.asarray(decoded),dtype=np.float64)**2))
        assert output['mse']==round(mse,2) and output['compressed_kb']==round(len(stream)/1024,2)
        records.append(dict(preset=preset,mode='RGB',stream_sha256=expected['stream_sha256'],release_equal=True,mse=mse,displayed_mse=output['mse']))
    rgba=(fixture/'rgba.png').read_bytes()
    output=json.loads(post('/api/compress',rgba,dict(mode='Auto',auto_config='balanced')))
    stream=base64.b64decode(output['pbc_base64']); decoded=PBC3.decompress(stream).image
    expected=next(r for r in json.loads((HERE/'local_checks.json').read_text())['rows'] if r['name']=='rgba')
    assert hashlib.sha256(stream).hexdigest()==expected['stream_sha256']
    preview=Image.open(io.BytesIO(base64.b64decode(output['reconstructed_image'].split(',')[-1])))
    assert np.array_equal(np.asarray(preview),np.asarray(decoded)) and decoded.mode=='RGBA'
    assert output['mse']==round(expected['mse'],2)
    records.append(dict(preset='balanced',mode='RGBA',stream_sha256=expected['stream_sha256'],release_equal=True,mse=expected['mse'],displayed_mse=output['mse']))
    reference=json.loads(subprocess.check_output([sys.executable,str(Path(__file__)),'--stream-reference'],text=True,cwd=ROOT))
    tree=ast.parse((ROOT/'server.py').read_text(encoding='utf-8'))
    fields=next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PBC3_FIELDS' for t in n.targets))
    allowed={k.value for k in fields.keys}
    manual={k:str(v).lower() if isinstance(v,bool) else str(v) for k,v in reference['config'].items() if k in allowed}
    regular=json.loads(post('/api/compress',(fixture/'native.png').read_bytes(),dict(mode='Manual',**manual)))
    assert hashlib.sha256(base64.b64decode(regular['pbc_base64'])).hexdigest()==reference['stream_sha256']
    frames=[json.loads(line) for line in post('/api/stream_compress',(fixture/'native.png').read_bytes(),dict(downsample_initialize='false')).splitlines()]
    assert frames
    last=Image.open(io.BytesIO(base64.b64decode(frames[-1]['image'].split(',')[-1])))
    assert hashlib.sha256(np.asarray(last).tobytes()).hexdigest()==reference['pixels_sha256']
    old=Path(r'C:\Users\EGE\Desktop\Coding\Personal Projects\Completed Projects\Image Processing\Probabilistic Brush Compression\New Demo\HF Space\PBC\research_pass1\space_googlewiki.balanced.pbc3').read_bytes()
    old_result=json.loads(post('/api/decode',old,{},filename='old-v0.pbc3',part='file'))
    old_pixels=Image.open(io.BytesIO(base64.b64decode(old_result['reconstructed_image'].split(',')[-1])))
    assert np.array_equal(np.asarray(old_pixels),np.asarray(PBC3.decompress(old).image))
    for key,status in ((None,403),(token(),400)):
        try:
            api.request('/api/benchmark/start',{'research':{}},key)
        except urllib.error.HTTPError as error:
            assert error.code==status
        else:
            raise AssertionError('Unauthorized/invalid research accepted')
    api.save('public_checks',dict(passed=True,hf_revision=meta['sha'],presets=records,streaming_release_equivalent=True,streaming_frames=len(frames),
        streaming_reference=reference,custom_form_field_cannot_override_default=True,existing_v0_decode=True,unauthorized_rejected=403,invalid_rejected=400,
        health=api.request('/api/health'),http_timings_used_for_benchmark=False))
    print('Four public presets, RGBA, manual/streaming release identity, MSE, old v0 decode and authorization checks passed.',flush=True)


if __name__=='__main__':
    main()
