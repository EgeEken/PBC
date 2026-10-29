import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import *
sys.path.insert(0,str(ROOT))
from research_pass2.stream_tools import parse
rows=[]
for name in ('googlewiki.png','mario64.png','monet.jpg','cat.jpg','flower.jpg'):
    image=load(name);cfg=PBC3Config.quality();prep=PBC3.prepare(image,config=cfg)
    direct=PBC3._canvas_to_image(prep['arr'],prep['color_space'],prep['has_alpha'])
    direct=direct.resize(image.size,Image.Resampling.BICUBIC,reducing_gap=None)
    item=dict(image=name,working_target_rgb_mse=mse(image,direct),working_size=[prep['w'],prep['h']],streams=[])
    for preset in ('quality','high_quality'):
        data=(PASS/'artifacts/architecture'/(name+'.'+preset+'.pbc')).read_bytes()
        header,patches,ledger=parse(data)
        metadata=ledger['header_bits']+sum(p['total_bits']-p['indices_bits'] for p in ledger['patches'])
        item['streams'].append(dict(preset=preset,raw_body_bytes=ledger['raw_bytes']-6,final_file_bytes=len(data),raw_metadata_bits=metadata,raw_metadata_fraction=metadata/((ledger['raw_bytes']-6)*8),patches=len(patches)))
    rows.append(item)
save('architecture/diagnostics.json',rows)
print(rows)
