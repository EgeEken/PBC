import base64
import datetime
import hashlib
import hmac
import io
import json
from pathlib import Path
import secrets
import subprocess
import sys
import time
import urllib.error
import urllib.request

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT))
SPACE='EgeEken/PBC'
HOST='https://egeeken-pbc.hf.space'


def save(name,value):
    (HERE/(name+'.json')).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')


def request(path,payload=None,token=None):
    headers={'User-Agent':'PBC-Pass5-authorized-validation','Cache-Control':'no-cache'}
    if token:
        headers['x-pbc-benchmark-token']=token
    if payload is not None:
        headers['Content-Type']='application/json'
    req=urllib.request.Request(HOST+path,data=None if payload is None else json.dumps(payload).encode(),headers=headers)
    with urllib.request.urlopen(req,timeout=60) as response:
        return json.load(response)


def metadata():
    with urllib.request.urlopen('https://huggingface.co/api/spaces/'+SPACE,timeout=30) as response:
        return json.load(response)


def job(name,spec,token):
    before=request('/api/benchmark/status')
    assert not before['running']
    result=request('/api/benchmark/start',{'research':spec},token)
    save(name+'_start',dict(time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),response=result,spec=spec))
    assert result.get('started'),result
    deadline=time.monotonic()+1250
    polls=[]
    while time.monotonic()<deadline:
        time.sleep(15)
        state=request('/api/benchmark/status')
        polls.append(dict(time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),running=state['running'],error=state['error'],done=state['done']))
        save(name+'_polls',polls)
        print(name,'running' if state['running'] else 'finished',flush=True)
        if not state['running']:
            save(name+'_status',state)
            assert not state['error'],state['error']
            result=state['research_result']
            assert result['exact'] and result['metadata']['spec']==spec
            save(name,result)
            return result
    raise RuntimeError('Research job did not finish within the bounded deadline')


def public_smoke(expected):
    image=(ROOT/'hpt_data'/'googlewiki.png').read_bytes()
    boundary='PBCPass5'+secrets.token_hex(12)
    body=(f'--{boundary}\r\nContent-Disposition: form-data; name="mode"\r\n\r\nAuto\r\n'
        f'--{boundary}\r\nContent-Disposition: form-data; name="auto_config"\r\n\r\nbalanced\r\n'
        f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="googlewiki.png"\r\nContent-Type: image/png\r\n\r\n').encode()+image+f'\r\n--{boundary}--\r\n'.encode()
    if (HERE/'public_api_response.json').exists():
        output=json.loads((HERE/'public_api_response.json').read_text())
    else:
        req=urllib.request.Request(HOST+'/api/compress',data=body,headers={'Content-Type':'multipart/form-data; boundary='+boundary})
        with urllib.request.urlopen(req,timeout=180) as response:
            output=json.load(response)
        save('public_api_response',output)
    stream=base64.b64decode(output['pbc_base64'])
    from PIL import Image
    import numpy as np
    preview=Image.open(io.BytesIO(base64.b64decode(output['reconstructed_image'].split(',')[-1])))
    preview.load()
    source=Image.open(io.BytesIO(image))
    from PBC3 import PBC3
    from pbc3_types import PBC3Config
    from PIL import ImageOps
    source=ImageOps.exif_transpose(source)
    source=source.convert('RGBA') if PBC3._has_alpha(source) else source.convert('RGB')
    decoded=PBC3.decompress(stream).image
    mse=float(np.mean(np.subtract(np.asarray(source),np.asarray(decoded),dtype=np.float32)**2,dtype=np.float64))
    reference=PBC3.compress(source,config=PBC3Config.balanced(learned_filler_model_path=str(ROOT/'patch_policy.npz')))
    assert stream==reference.data
    assert np.array_equal(np.asarray(preview),np.asarray(decoded))
    assert output['mse']==round(mse,2) and output['compressed_kb']==round(len(stream)/1024,2)
    assert preview.size==(output['width'],output['height'])==source.size
    save('public_api_smoke',dict(status=200,input='existing deployed hpt_data/googlewiki.png',preset='balanced',
        final_bytes=len(stream),stream_sha256=hashlib.sha256(stream).hexdigest(),
        pixels_sha256=hashlib.sha256(np.asarray(decoded).tobytes()).hexdigest(),
        preview_equals_decode=True,mse=mse,displayed_mse=output['mse'],image_mode=source.mode,
        public_stream_equals_local_unchanged_rgba_baseline=True,
        benchmark_rgb_stream_compared=False,reason='Public API preserves source alpha; benchmark deliberately converts inputs to RGB.',
        compressed_kb=output['compressed_kb'],width=output['width'],height=output['height'],
        public_config_retains_internal_mse=True,http_latency_used_for_performance=False,
        response_fields=sorted(output),server_stage_timings=output['stage_timings']))
    print('Public compression API: bytes, preview, size and MSE verified',flush=True)


def main():
    deployment=json.loads((HERE/'deployment.json').read_text())
    before=metadata()
    resumed=before['sha']==deployment['commit']
    save('hf_resume' if resumed else 'hf_before',before)
    assert before['sha'] in (deployment['base_revision'],deployment['commit'])
    assert before['runtime']['hardware']['current']=='cpu-basic'
    status={path:request(path) for path in ('/api/health','/api/benchmark/status','/api/quick_rd/status','/api/sweeps/run/status','/api/train/status')}
    save('service_resume' if resumed else 'service_before',status)
    assert not any(v.get('running') for v in status.values())
    credential=subprocess.run(['git','credential','fill'],input='protocol=https\nhost=huggingface.co\n\n',text=True,capture_output=True,check=True)
    fields=dict(line.split('=',1) for line in credential.stdout.splitlines() if '=' in line)
    from huggingface_hub import HfApi
    api=HfApi(token=fields['password'])
    assert api.whoami()['name']=='EgeEken'
    token=hmac.new(fields['password'].encode(),('PBC scoped benchmark token v1 / '+SPACE+' / '+deployment['commit']).encode(),hashlib.sha256).hexdigest()
    ready=False
    if resumed:
        try:
            request('/api/benchmark/start',{'research':{}},token)
        except urllib.error.HTTPError as error:
            ready=error.code==400
    if not ready:
        api.add_space_secret(SPACE,'PBC_RESEARCH_BENCHMARK_TOKEN',token,description='Authenticated bounded runtime benchmark; public codec unchanged. Pass 5.')
        print('Configured scoped private benchmark secret; value is not logged or saved',flush=True)
    if not resumed:
        pushed=subprocess.run(['git','push','origin',deployment['commit']+':refs/heads/main'],capture_output=True,text=True)
        save('push',dict(returncode=pushed.returncode,stdout=pushed.stdout,stderr=pushed.stderr,commit=deployment['commit']))
        assert pushed.returncode==0,'HF push failed; see push.json (benchmark secret can be removed)'
        print('Pushed',deployment['commit'],flush=True)
    deadline=time.monotonic()+1200
    observations=json.loads((HERE/'deployment_observations.json').read_text()) if resumed else []
    while time.monotonic()<deadline:
        observed=metadata()
        runtime=observed.get('runtime',{})
        item=dict(time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),repository_sha=observed.get('sha'),runtime=runtime)
        observations.append(item)
        save('deployment_observations',observations)
        print('Deployment',runtime.get('stage'),runtime.get('sha'),flush=True)
        if runtime.get('sha')==deployment['commit'] and runtime.get('stage')=='RUNNING':
            health=request('/api/health')
            try:
                request('/api/benchmark/start',{'research':{}},token)
            except urllib.error.HTTPError as error:
                if error.code in (403,404):
                    time.sleep(15)
                    continue
                assert error.code==400
            save('hf_deployed',observed)
            save('health_deployed',health)
            break
        if runtime.get('stage') in ('BUILD_ERROR','RUNTIME_ERROR'):
            raise RuntimeError('HF deployment failed: '+runtime['stage'])
        time.sleep(15)
    else:
        raise RuntimeError('HF did not confirm the expected running revision within 20 minutes')
    assert runtime['hardware']['current']=='cpu-basic'
    smoke=dict(images=['googlewiki.png'],presets=['balanced'],variants=['baseline','all_mse','all'],repeats=2)
    for bad,code in ((None,403),('wrong',403)):
        try:
            request('/api/benchmark/start',{'research':smoke},bad)
        except urllib.error.HTTPError as error:
            assert error.code==code
        else:
            raise AssertionError('Unauthenticated research benchmark accepted')
    try:
        request('/api/benchmark/start',{'research':dict(smoke,images=['../patch_policy.npz'])},token)
    except urllib.error.HTTPError as error:
        assert error.code==400
    else:
        raise AssertionError('Unallowlisted image accepted')
    save('remote_auth_checks',dict(missing_token=403,wrong_token=403,path_traversal=400))
    result=json.loads((HERE/'hf_smoke.json').read_text()) if resumed and (HERE/'hf_smoke.json').exists() else job('hf_smoke',smoke,token)
    expected=next(r for r in result['rows'] if r['variant']=='baseline')
    public_smoke(expected)
    spec=dict(images=['cat.jpg','monet.jpg','flower.jpg','cathedral.png','googlewiki.png','stockpoint.jpg','mario64.png','eiffel.png'],
        presets=['compression','balanced','quality','high_quality'],variants=['baseline','all_mse','all'],repeats=3)
    # The bounded representative run is contingent on exact output and no smoke regression.
    from statistics import median
    rows=result['rows']
    base=median(r['encode_seconds'] for r in rows if r['variant']=='baseline')
    for mode in ('all_mse','all'):
        current=median(r['encode_seconds'] for r in rows if r['variant']==mode)
        assert current<base*1.1,(mode,'smoke encode regression',current/base)
    full=job('hf_representative',spec,token)
    after=metadata()
    assert after['runtime']['sha']==deployment['commit']
    save('hf_after',after)
    save('service_after',{path:request(path) for path in ('/api/health','/api/benchmark/status','/api/quick_rd/status','/api/sweeps/run/status','/api/train/status')})
    save('execution_complete',dict(revision=deployment['commit'],hardware='cpu-basic',smoke_rows=len(result['rows']),
        representative_rows=len(full['rows']),public_api_checks=1,training=0,allocator_experiments=0,
        token_saved=False,secret_enabled=True,benchmark_database_modified=False))
    print('HF validation complete',flush=True)


if __name__=='__main__':
    main()
