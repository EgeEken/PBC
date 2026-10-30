import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request

p=subprocess.run(['git','credential','fill'],input='protocol=https\nhost=github.com\n\n',text=True,capture_output=True,check=True)
credential=dict(line.split('=',1) for line in p.stdout.splitlines() if '=' in line)
token=credential.get('password') or os.environ.get('GH_TOKEN') or os.environ.get('GITHUB_TOKEN')
path=sys.argv[1]
request=urllib.request.Request('https://api.github.com/repos/EgeEken/PBC'+path,headers={'Accept':'application/vnd.github+json','User-Agent':'PBC-Pass28-acceptance','Authorization':'Bearer '+token})
with urllib.request.urlopen(request) as response:
    body=response.read()
if len(sys.argv)>2:
    Path(sys.argv[2]).write_bytes(body)
    print('saved',len(body),'bytes')
else:
    data=json.loads(body)
    if 'workflow_runs' in data:
        data={'total_count':data['total_count'],'runs':[{k:r.get(k) for k in ('id','head_sha','status','conclusion','html_url')} for r in data['workflow_runs']]}
    print(json.dumps(data,indent=2))
