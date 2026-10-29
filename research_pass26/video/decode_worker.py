import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import *
from container import decode
import learned_filler

def forbidden(*a,**k): raise AssertionError('encoder or learned model used during video decoding')
PBC3.compress=forbidden;learned_filler.LearnedFiller.__init__=forbidden
rows=[]
for path in sorted((PASS/'artifacts/video').glob('*.p26v')):
    frames,meta=decode(path.read_bytes())
    rows.append(dict(file=path.name,frame_hashes=[sha(frame.tobytes()) for frame in frames]))
print(json.dumps(rows))
