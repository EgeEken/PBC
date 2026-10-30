import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'research_pass27'))
from common import *
CANDIDATE=ROOT/'research_pass28/artifacts/release-source'
def save(name,data): (ROOT/'research_pass28/artifacts/memory.json').write_text(json.dumps(data,indent=2)+'\n',newline='\n')
import subprocess
if len(sys.argv)>1:
    sys.path.insert(0,str(CANDIDATE))
    from PBC3 import PBC3
    from pbc3_types import PBC3Config
    mode,image_name=sys.argv[1:]
    PBC3.compress(Image.new('RGB',(32,32)),config=PBC3Config.quality(),mse_backend=mode)
    image=load(image_name);before=psutil.Process().memory_info().rss
    result=PBC3.compress(image,config=PBC3Config.quality(),mse_backend=mode)
    peak=psutil.Process().memory_info().peak_wset
    print(json.dumps(dict(mode=mode,image=image_name,before_rss=before,peak_wset=peak,stream_sha256=sha(result.data),mse=result.mse)))
else:
    rows=[]
    for image in ('cat.jpg','flower.jpg'):
        for repeat in range(2):
            pair=[]
            for mode in (('reference','native') if repeat%2==0 else ('native','reference')):
                row=json.loads(subprocess.check_output([sys.executable,__file__,mode,image],text=True));row['repeat']=repeat;rows.append(row);pair.append(row)
            assert pair[0]['stream_sha256']==pair[1]['stream_sha256'] and pair[0]['mse']==pair[1]['mse']
    save('native/memory.json',rows);print(rows)
