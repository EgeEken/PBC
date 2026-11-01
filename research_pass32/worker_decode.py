import hashlib,json,sys
from pathlib import Path
import numpy as np
import psutil
from pbc32_codec import decode,sha

def main(path):
    manifest=json.loads(Path(path).read_text()); base=np.load(manifest['base_npy'],allow_pickle=False); rows=[]
    for item in manifest['streams']:
        data=Path(item['stream']).read_bytes(); image,modes,params=decode(data,base)
        rows.append({'key':item['key'],'pixels_sha256':sha(image.tobytes()),'modes_sha256':sha(modes.tobytes()),
                     'params_sha256':sha(params),'mode_counts':np.bincount(modes,minlength=7).tolist()})
    print(json.dumps({'rows':rows,'process_peak_working_set_bytes':psutil.Process().memory_info().peak_wset}))
if __name__=='__main__':main(sys.argv[1])