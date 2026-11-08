# Divergence timeline

## Verified points

| Point | Revision | Evidence and classification |
|---|---|---|
| Pass 26 published ZIP snapshot | Asset SHA-256 `edec3ec73721c3f56913db21f6089a3b6a1637fb3481e61de5b0475da8eeaede`, 130,098 bytes | Pass 26 records this as the published PBC 3.1 ZIP. Its source hashes differ from the currently downloaded public asset. Eight archived Quality/High Quality streams for Cat, Flower, Googlewiki and Mario64 match current release Python exactly. |
| Research baseline promoted | GitHub source `8006f30…`; HF deployed `98ade0c…` | Pass 10 baseline manifest and public equivalence report identify release RNG (`numpy.random.default_rng`, seed 2003), fixed model and four preset configurations. HF current module hashes match this source snapshot. |
| GitHub release tag | `ac7566f8b4fe77020bc27ff720fbe9942040cca4` | Tag points to an October 2026 README update. It does not identify the source bytes bundled in the later downloadable ZIP. |
| Public downloadable release | Release asset `PBC3_1_Release.zip`, published 2026-10-29; SHA-256 `3ee8f68…` | Embedded Python source differs from HF/research `PBC3.py`, `pbc3_types.py`, `pbc3_ops.py`, `pbc3_stream.py`, and `pbc3_features.py`. It adds native MSE/decoder modules and bundled DLLs. This is a semantic-risk source divergence; the Coast Compression output tested here nevertheless matches exactly. |
| Local research packaging candidate | `dist/PBC3_1_Release.zip` | Different archive SHA `dc069a…`; not the published artifact. Do not use as a release identity. |

## Earliest divergence

The exact first source divergence is **not established**. The verified artifact interval is between the Pass 26 130 KB ZIP (`edec3ec…`) and the current 76.9 MB ZIP (`3ee8f68…`); the current ZIP has a different embedded Python tree and Windows executable. It has no source commit manifest linking its Python tree to a Git revision. The earliest commit or repackaging event that changed the downloadable asset, and why, could not be established. Despite source differences, current release Python, HF/research source and the older Pass 26 archive match on the tested overlapping outputs.

## Difference classes

- **Semantic:** Potentially different due to changed encoder/stream source. All 12 Coast/Cat/Googlewiki preset streams and decoded pixels match between current release Python and frozen research. Eight Pass 26 Quality/High Quality archived files also match. Live HF encoding remains unverified.
- **Serialization-only:** No observed evidence of a serialization-only difference; the one tested stream hash matches.
- **Runtime-only:** Native MSE and separate decoder helper modules are present in the ZIP. The release Python decoder took about 0.20–0.21 seconds on Coast in this pass. The historical Pass 44 frozen-source median was 0.116 seconds; a new frozen-source timing attempt did not complete, so the difference is not attributed to a specific code path.
- **Interface-only:** HF Auto/API mode chooses a preset and clamps learned candidates/top-k to one; this is consistent with the preset’s resolved values. Manual mode is a distinct field-driven configuration.
- **Environment-dependent:** Pillow/AVIF codec defaults and platform-native binaries can affect outputs. Exact encoder differences across Windows/Linux were not isolated.

There is no verified first divergent commit per preset. Do not infer a release event from the version string or tag date.
