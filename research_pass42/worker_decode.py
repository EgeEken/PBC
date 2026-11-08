import sys
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pbc4_codec import decode

source = Path(sys.argv[1])
output = Path(sys.argv[2])
image = decode(source.read_bytes())
Image.fromarray(image).save(output, format='PNG')

