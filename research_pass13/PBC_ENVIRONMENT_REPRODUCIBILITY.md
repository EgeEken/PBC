# Pass 13 environment reproducibility

## Result

The isolated Windows environment loaded the frozen codec and model successfully and reproduced the Pass 12 local stream hashes for all 16 YCbCr cases. It matched only 6/16 saved HF stream hashes, so the environment did not resolve the HF discrepancy.

The runtime was Windows 10 build 26200, Python 3.11.16, NumPy 2.4.6, Numba 0.68.0, Pillow 11.3.0, with Numba and numerical thread variables pinned to one. HF was Linux, Python 3.11.17, NumPy 2.4.6, Numba 0.68.0, Pillow 11.3.0. Python 3.11.17 was unavailable from the local Python installer; 3.11.16 was used. The separate environment is `research_pass13/.venv`; production dependency files were unchanged.

Frozen source hashes, model hash, raw input file hashes, presets, seed 2003, release RNG, and EXIF transpose/RGB normalization matched the saved manifests. The model loaded as `LearnedFiller`; PBC version 0 and YCbCr were confirmed. All decoded images matched their source dimensions and previews.

## Bounded diagnosis

The diagnostic sampled three HF-mismatching cases (Cat/High Quality, Flower/Quality, Monet/Quality) and one matching control (Cat/Quality) in both Windows runtimes: Pass 12 Python 3.13.5 / NumPy 2.2.6 / Numba 0.62.1 and isolated Python 3.11.16 / NumPy 2.4.6 / Numba 0.68.0. All four cases had identical decoded RGB pixel arrays, prepared arrays, prepared integer targets, complete streams, and patch trajectories across those local runtimes. Each local trajectory also matched its Pass 12 stream hash.

This rules out the local Python/NumPy/Numba version change as the cause for these sampled failures. The archived HF comparison contains full-stream hashes and patch counts, but not HF decoded-input hashes, prepared-array hashes, or patch trajectories. Therefore the first HF divergence cannot be located among image decoding, prepared arrays, RNG geometry, first action, or numerical projection. The remaining candidates include Windows-versus-Linux/runtime behavior and the one-patch Python version difference; none is established as the cause. The investigation stops here as requested.

## Recommendation

The isolated Windows environment is repeatable against the Pass 12 Windows run, but it is not a reliable replica of the frozen HF baseline. Do not use it to claim small historical HF RD differences. For future cross-HF reproduction, use a Linux environment pinned to Python 3.11.17, NumPy 2.4.6, Numba 0.68.0, Pillow 11.3.0, one numerical thread and the same CPU-affinity policy. That exact environment has not yet been verified locally.

Create the isolated Windows environment (Python 3.11.16 was the nearest available interpreter):

```powershell
uv venv research_pass13/.venv --python 3.11.16
uv pip install --python research_pass13/.venv/Scripts/python.exe numpy==2.4.6 numba==0.68.0 Pillow==11.3.0
```

Reproduce the 16-case run and four-case local diagnostic. The generated channel analysis and diagnostic files stay local:

```powershell
research_pass13/.venv/Scripts/python.exe research_pass13/run_pass13.py
research_pass13/.venv/Scripts/python.exe research_pass13/diagnose_repro.py > research_pass13/diagnostic_311.json
py -3.13 research_pass13/diagnose_repro.py > research_pass13/diagnostic_313.json
research_pass13/.venv/Scripts/python.exe research_pass13/summarize_pass13.py
```
