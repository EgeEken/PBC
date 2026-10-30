# Consumer Windows smoke test

No Python, compiler, administrator access or network is required. Use Windows
x64 and built-in Windows PowerShell 5.1. Extract this bundle, download the three
candidate files listed in smoke-checksums.json, and open PowerShell here:

```
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\Smoke-Test.ps1 -CombinedArchive "PATH_TO_COMBINED.zip" -DecoderArchive "PATH_TO_DECODER.zip" -OneFile "PATH_TO_DECODER.exe" -Manifest .\smoke-checksums.json -Report .\PBC32-smoke-results.json
```

The execution-policy option applies only to this process. Verify this bundle's
checksum from SHA256SUMS.txt before running it. The script verifies each supplied
application checksum before extraction. It uses an isolated Unicode temporary
folder, clears Python environment variables and restricts PATH to System32 for
runtime checks. System.Drawing checks actual decoded PNG dimensions and raw
RGB/RGBA pixel hashes independently of PBC's hash reporting. Known streams were
produced by the historical 3.0 and actual published 3.1 runtimes. All four encode
presets, repeat determinism, overwrite protection and malformed input are tested.

The JSON records OS, token, results and the temporary working folder. A nonzero
exit code means failure. Work files are retained for diagnosis; after reviewing
the report, delete only its named PBC28 temporary directory. Please report whether
Python/compiler runtimes were physically absent, not just removed from PATH.
The script cannot certify a clean installation by itself. No telemetry is sent.
