# PBC 3.2.0rc1 distribution results

Combined Windows directory distribution and Python source/wheel are ready for broader testing, not publication. The decoder-only directory and optional one-file application also work. Published 3.1, its tag/ZIP, main, HF and earlier research/staging are preserved.

Baseline: actual published ZIP edec3ec73721c3f56913db21f6089a3b6a1637fb3481e61de5b0475da8eeaede. prepare.py verifies it and generates a complete candidate from tracked overlays plus the unchanged model. Decode-operation bodies are extracted unchanged. Old root PBC 2.3/tag packaging metadata is not used. Paths, exact bytes and SHA256 are in ../artifacts.json and ../provenance.json.

## Packages and clean reproduction

Ordinary `python -m pip install .` and wheel installation pass in separate fresh Python 3.11 environments, including pip check. From unrelated directories, flat PBC3/types imports and pbc3 re-exports resolve installed site-packages; all four presets match the published reference. Model placement/hash are correct: pbc3/patch_policy.npz is setuptools package-data, not prefix-level data-files. README, INSTALL and retained MIT LICENSE are included. Portable Python wheel contains no Windows DLL or JIT caches. NumPy/Numba/Pillow remain core requirements; LLVM is Numba's dependency. Native C source is retained as data. Demo/training/animation stacks are not mandatory.

Clean sparse checkout 73082d7a4f0943c0b7a3716fb32f0e30f9fa0939 downloaded the pinned asset, reused no existing candidate/model/build directories, rebuilt both applications and a wheel, and stayed clean in Git. Wheel payloads match entry by entry; ZIP timestamps may differ. Both rebuilt apps run from an unrelated directory without Python/compiler on PATH, retaining reference bytes/pixels. This is functional reproducibility, not bit-identical PE builds: paths/timestamps can change executable hashes.

Use 64-bit Python 3.11 with requirements-build.txt. Decoder build uses a separate environment with requirements-decoder-build.txt and no Numba/LLVM:

```
python research_pass27/packaging/prepare.py
python research_pass27/native/build.py
uv venv --no-project research_pass27/artifacts/decoder-env --python PATH_TO_PYTHON311
uv pip install --python research_pass27/artifacts/decoder-env/Scripts/python.exe -r research_pass27/requirements-decoder-build.txt
python research_pass27/packaging/build.py --kind combined
python research_pass27/packaging/build.py --kind decoder
python research_pass27/packaging/build.py --kind decoder --onefile
python research_pass27/packaging/install_test.py
python research_pass27/packaging/install_verify.py
python research_pass27/packaging/validate_distribution.py
python research_pass27/packaging/measure.py
python research_pass27/packaging/memory.py
python research_pass27/packaging/clean_checkout.py
```

Restore hpt_data and preserved Pass 26 streams for validation. Builds need tracked source and the verified published asset. Exact dependency freezes/toolchains accompany provenance. End users need neither compiler nor separate Python. Keep pbc.exe and _internal together. Overwrite requires --force; stdout/errors are UTF-8, including Unicode missing-file paths.

## Distribution and command costs

| Form | Files | Extracted bytes | Download bytes | EXE SHA256 |
| --- | --- | --- | --- | --- |
| combined | 163 | 191210903 | 72582125 | e899942cd3fa334839eb5b26624c7a4532c19ce630ac7739d4b86ce7600c818e |
| decoder | 124 | 65682789 | 27260868 | 83de42afea875fc5a5a90a6981b91eca46897679e69d10e711ee01745678abb4 |
| decoder one-file | 1 | 27232475 | 27232475 | c7c6d5be9a9155ef4182cc4ee797adfaf8666ecbc2d109261a1f6896c0e17bbc |

Fresh-process figures are medians of three calls, not flushed disk-cache startup. Decode command uses 96x95 RGB and includes PNG output; inspect uses the same file. Python includes interpreter/imports. Combined and thin apps share the lazy decode path.

| Runtime | Version s | Inspect s | Decode command s |
| --- | --- | --- | --- |
| pass26_combined | 0.1485 | 0.6555 | 0.6732 |
| combined | 0.1504 | 0.2824 | 0.3044 |
| decoder | 0.1421 | 0.2883 | 0.3128 |
| decoder_onefile | 1.1441 | 1.9715 | 1.9710 |
| published_python | - | 1.2349 | 1.1574 |

First-use measurements use the same small Quality input. Complete command, imports and compress-call time are separate:

| Runtime | Backend | Phase | Command s | Imports s | Compress s |
| --- | --- | --- | --- | --- | --- |
| pass26_combined | None | fresh_cache | 4.896 | 0.441 | 4.131 |
| pass26_combined | None | reused_cache | 1.090 | 0.443 | 0.357 |
| pass26_combined | None | reused_cache_other_cwd | 4.351 | 0.500 | 3.524 |
| combined | reference | fresh_cache | 4.624 | 0.450 | 3.818 |
| combined | reference | reused_cache | 1.090 | 0.450 | 0.356 |
| combined | reference | reused_cache_other_cwd | 1.014 | 0.440 | 0.317 |
| combined | native | fresh_cache | 4.422 | 0.423 | 3.704 |
| combined | native | reused_cache | 0.987 | 0.432 | 0.303 |
| combined | native | reused_cache_other_cwd | 0.973 | 0.422 | 0.311 |

Old frozen Numba ignored the requested NUMBA_CACHE_DIR and used a CWD-derived UserWide cache. Candidate uses a frozen-source locator: shipped source files are found independently of CWD and the requested versioned user cache is populated. Different-CWD commands reuse it. No machine-generated nbc/nbi files are distributed. First-ever compilation still costs seconds. Savings are local observations, not guarantees across processors. Cache validity retains Numba's executable stamp/CPU/code keys. Application relocation/build changes may require compilation. Combined one-file 3.2 was not produced; directory mode is intended. One-file decoder has no JIT.

## Permissions and remaining work

Actual non-admin token: app copies have inheritance removed and user RX plus ACL-management rights for cleanup. Directory creation and existing-file writes fail, while read/execute works with cache/output elsewhere. ACLs return to writable afterward. PATH is System32; PYTHONHOME/PYTHONPATH/VIRTUAL_ENV are cleared. Unicode app/input/output names work. Four defaults and RGBA retain reference bytes. All 32 files decode through thin app; stored/warmup/historical also pass all three binaries. Invalid/truncated/header/resource inputs fail without traceback. Isolated one-file TEMP leaves no _MEI folders after exit.

VirtualBox is installed but no accessible Windows guest exists. No clean VM with Python physically absent, administrator-owned Program Files installation, other CPU/Windows version, installer/signing or formal security audit. Limits remain 32M pixels, 64 MiB input, 128 MiB body, 10k patches; valid files beyond them reject. Keep Numba 0.68 pinned and revalidate locator before cache-API changes. Before official release: clean Windows/CPU coverage, installer/update decision and final licensing review. No release/tag/assets/HF deployment.

## Local artifacts

- C:\Users\EGE\.codex\worktrees\pbc-research-experiments\PBC\research_pass27\artifacts\combined\onedir\pbc\pbc.exe; SHA256 e899942cd3fa334839eb5b26624c7a4532c19ce630ac7739d4b86ce7600c818e
- C:\Users\EGE\.codex\worktrees\pbc-research-experiments\PBC\research_pass27\artifacts\decoder\onedir\pbc-decode\pbc-decode.exe; SHA256 83de42afea875fc5a5a90a6981b91eca46897679e69d10e711ee01745678abb4
- C:\Users\EGE\.codex\worktrees\pbc-research-experiments\PBC\research_pass27\artifacts\decoder\onefile\pbc-decode.exe; SHA256 c7c6d5be9a9155ef4182cc4ee797adfaf8666ecbc2d109261a1f6896c0e17bbc
- C:\Users\EGE\.codex\worktrees\pbc-research-experiments\PBC\research_pass27\artifacts\candidate\pbc3\mse.dll; SHA256 6c4196caffd0ce311825f2f81af6a3a51cddb96aae82ab405ce3bd74fda459a1
- C:\Users\EGE\.codex\worktrees\pbc-research-experiments\PBC\research_pass27\artifacts\packages\pbc_codec-3.2.0rc1-py3-none-any.whl; SHA256 120df9fcbbc0131110d54d417425c318609414d094b43334bbd01e73080bbfa6
- C:\Users\EGE\.codex\worktrees\pbc-research-experiments\PBC\research_pass27\artifacts\packages\pbc_codec-3.2.0rc1.tar.gz; SHA256 9574b7044432cf68fbf45786f8cd6846369de52eea62347f1840bbb20e51206d
