import hashlib
import hmac
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from research_pass5 import run_hf as api
api.HERE=HERE


def main():
    deployment=json.loads((HERE/'hf_deployment.json').read_text())
    before=api.metadata()
    api.save('hf_before',before)
    assert before['sha']==deployment['base'] and before['runtime']['sha']==deployment['base']
    states={p:api.request(p) for p in ('/api/health','/api/benchmark/status','/api/quick_rd/status','/api/sweeps/run/status','/api/train/status')}
    assert not any(v.get('running') for v in states.values())
    api.save('hf_services_before',states)
    credentials=subprocess.run(['git','credential','fill'],input='protocol=https\nhost=huggingface.co\n\n',capture_output=True,text=True,check=True)
    values=dict(line.split('=',1) for line in credentials.stdout.splitlines() if '=' in line)
    token=hmac.new(values['password'].encode(),('PBC scoped benchmark token v1 / EgeEken/PBC / '+deployment['base']).encode(),hashlib.sha256).hexdigest()
    try:
        api.request('/api/benchmark/start',{'research':{}},token)
    except urllib.error.HTTPError as error:
        assert error.code==400,'Existing private benchmark secret did not authorize the validation client'
    push=subprocess.run(['git','push','origin',deployment['commit']+':refs/heads/main'],capture_output=True,text=True)
    api.save('hf_push',dict(returncode=push.returncode,stdout=push.stdout,stderr=push.stderr))
    assert push.returncode==0
    print('Pushed HF runtime integration',deployment['commit'],flush=True)
    deadline=time.monotonic()+1200
    observations=[]
    while time.monotonic()<deadline:
        meta=api.metadata()
        runtime=meta['runtime']
        observations.append(dict(time_utc=api.datetime.datetime.now(api.datetime.timezone.utc).isoformat(),runtime=runtime,sha=meta['sha']))
        api.save('hf_deployment_observations',observations)
        print(runtime['stage'],runtime.get('sha'),flush=True)
        if runtime['stage']=='RUNNING' and runtime.get('sha')==deployment['commit']:
            api.save('hf_deployed',meta)
            api.save('hf_health',api.request('/api/health'))
            break
        if runtime['stage'] in ('RUNTIME_ERROR','BUILD_ERROR'):
            raise RuntimeError('Deployment failure: '+runtime['stage'])
        time.sleep(15)
    else:
        raise RuntimeError('Expected running revision was not confirmed')
    smoke=dict(images=['googlewiki.png'],presets=['balanced'],variants=['baseline','all_mse','all'],repeats=2)
    first=api.job('hf_smoke',smoke,token)
    api.public_smoke(None)
    from statistics import median
    for metric in ('encode_seconds','decode_seconds'):
        control=median(r[metric] for r in first['rows'] if r['variant']=='baseline')
        assert all(median(r[metric] for r in first['rows'] if r['variant']==v)<control*1.10 for v in ('all_mse','all')),'Substantial smoke regression'
    spec=dict(images=['cat.jpg','flower.jpg','googlewiki.png'],presets=['quality','high_quality'],variants=['baseline','all_mse','all'],repeats=3)
    api.job('hf_representative',spec,token)
    old=Path(r'C:\Users\EGE\Desktop\Coding\Personal Projects\Completed Projects\Image Processing\Probabilistic Brush Compression\New Demo\HF Space\PBC\research_pass1\space_googlewiki.balanced.pbc3').read_bytes()
    boundary='PBCPass6Compatibility'
    body=(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="existing-v0.pbc3"\r\nContent-Type: application/octet-stream\r\n\r\n').encode()+old+f'\r\n--{boundary}--\r\n'.encode()
    req=urllib.request.Request(api.HOST+'/api/decode',data=body,headers={'Content-Type':'multipart/form-data; boundary='+boundary})
    with urllib.request.urlopen(req,timeout=120) as response:
        decoded=json.load(response)
    api.save('hf_old_v0_decode',decoded)
    api.save('hf_after',api.metadata())
    api.save('hf_services_after',{p:api.request(p) for p in ('/api/health','/api/benchmark/status','/api/train/status')})
    print('HF integration validation complete',flush=True)


if __name__=='__main__':
    main()
