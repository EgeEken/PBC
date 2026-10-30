# PBC V3.1

Windows x64, Python 3.11. Install this folder: python -m pip install .

    from PBC3 import PBC3 as PBC
    from pbc3_types import PBC3Config

    result = PBC.compress("input.png", config=PBC3Config.quality())
    result.save("output.pbc")
    PBC.decompress(result.data).image.save("decoded.png")

Presets: compression, balanced, quality, high_quality.
The bundled Windows DLL is selected automatically. PBC3 v0 decoding needs no model.
See LICENSE and THIRD_PARTY_NOTICES.md.
