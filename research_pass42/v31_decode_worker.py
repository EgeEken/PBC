import pathlib,sys
from PIL import Image
HERE=pathlib.Path(__file__).resolve().parent
FROZEN=HERE/'outputs'/'v31_frozen'
sys.path.insert(0,str(FROZEN))
from PBC3 import PBC3
im=PBC3.decompress(pathlib.Path(sys.argv[1]).read_bytes()).image
im.convert('RGB').save(sys.argv[2],format='PNG')
