import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import *
import subprocess
if len(sys.argv)>1:
    mode=sys.argv[1]
    if mode=='native':
        from kernel import install
        install()
    tiny=Image.fromarray(np.zeros((32,32,3),np.uint8))
    PBC3.compress(tiny,config=PBC3Config.quality())
    image=load('cat.jpg')
    before=psutil.Process().memory_info().rss
    result=PBC3.compress(image,config=PBC3Config.quality())
    info=psutil.Process().memory_info()
    print(json.dumps(dict(mode=mode,before_rss=before,peak_wset=info.peak_wset,stream_sha256=sha(result.data),mse=result.mse)))
else:
    rows=[json.loads(subprocess.check_output([sys.executable,__file__,mode],text=True)) for mode in ('reference','native')]
    assert rows[0]['stream_sha256']==rows[1]['stream_sha256'] and rows[0]['mse']==rows[1]['mse']
    save('native/memory.json',rows)
    print(rows)
