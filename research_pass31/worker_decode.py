import hashlib,json,sys
from pathlib import Path
import numpy as np
from pbc31_codec import decode,sha

def main(path):
    manifest=json.loads(Path(path).read_text()); base=np.load(manifest['base_npy'],allow_pickle=False); rows=[]
    for item in manifest['streams']:
        data=Path(item['stream']).read_bytes(); image,modes,params=decode(data,base)
        rows.append({'key':item['key'],'pixels_sha256':sha(image.tobytes()),'modes_sha256':sha(modes.tobytes()),'params_sha256':sha(params)})
    print(json.dumps({'rows':rows}))

if __name__=='__main__':main(sys.argv[1])
