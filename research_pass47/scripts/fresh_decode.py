import hashlib
import json
import pathlib
import sys

from PIL import Image

PASS = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PASS / "codec"))
from PBC3 import PBC3

stream_path = pathlib.Path(sys.argv[1])
image_path = pathlib.Path(sys.argv[2])
result = PBC3.decompress(stream_path.read_bytes()).image.convert("RGB")
result.save(image_path, format="PNG")
print(json.dumps({
    "size": list(result.size),
    "rgb_sha256": hashlib.sha256(result.tobytes()).hexdigest(),
    "stream_sha256": hashlib.sha256(stream_path.read_bytes()).hexdigest(),
}))
