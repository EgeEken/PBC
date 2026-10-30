$ErrorActionPreference='Stop'
$root=(Get-Location).Path
$python=(Get-Command python.exe).Source
& $python -m pip install -r research_pass27/requirements-build.txt
if ($LASTEXITCODE) { throw 'Build dependencies failed' }
$gcc=@('C:\msys64\ucrt64\bin\gcc.exe','C:\mingw64\bin\gcc.exe','C:\msys64\mingw64\bin\gcc.exe') | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $gcc) { throw 'No supported preinstalled MinGW GCC' }
$env:PATH=(Split-Path $gcc)+';'+$env:PATH
& $gcc --version | Select-Object -First 1
& $python research_pass28/prepare.py
if ($LASTEXITCODE) { throw 'Source preparation failed' }
& $gcc -O3 -std=c11 -Wall -Wextra -Werror -shared -static-libgcc research_pass28/artifacts/release-source/pbc3/mse.c -o research_pass28/artifacts/release-source/pbc3/mse.dll
if ($LASTEXITCODE) { throw 'Native build failed' }
& $python -m venv research_pass28/artifacts/decoder-env
$decoder=Join-Path $root 'research_pass28/artifacts/decoder-env/Scripts/python.exe'
& $decoder -m pip install -r research_pass27/requirements-decoder-build.txt
if ($LASTEXITCODE) { throw 'Decoder build dependencies failed' }
foreach ($kind in @('combined','decoder')) {
    & $python research_pass28/artifacts/release-source/tools/build_windows.py --kind $kind --decoder-python $decoder --output research_pass28/artifacts/dist
    if ($LASTEXITCODE) { throw "Build failed: $kind" }
}
& $python research_pass28/artifacts/release-source/tools/build_windows.py --kind decoder --onefile --decoder-python $decoder --output research_pass28/artifacts/dist
if ($LASTEXITCODE) { throw 'One-file build failed' }
& $python research_pass28/runtime_check.py
if ($LASTEXITCODE) { throw 'Runtime comparison failed' }
& $python -m build --no-isolation --outdir research_pass28/artifacts/dist research_pass28/artifacts/release-source
if ($LASTEXITCODE) { throw 'Package build failed' }
& $python research_pass28/install_check.py
if ($LASTEXITCODE) { throw 'Package install check failed' }
$dist=Join-Path $root 'research_pass28/artifacts/dist'
$combined=Join-Path $dist 'PBC32rc1-combined-Windows-x64.zip'
$thin=Join-Path $dist 'PBC32rc1-decoder-Windows-x64.zip'
$one=Join-Path $dist 'decoder/onefile/pbc-decode.exe'
$entries=@(@{kind='combined';path=$combined},@{kind='decoder';path=$thin},@{kind='decoder_onefile';path=$one})
foreach ($e in $entries) { $e.sha256=(Get-FileHash -LiteralPath $e.path -Algorithm SHA256).Hash.ToLowerInvariant() }
@{artifacts=$entries} | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath research_pass28/artifacts/ci-manifest.json -Encoding UTF8
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File research_pass28/smoke/Smoke-Test.ps1 -CombinedArchive $combined -DecoderArchive $thin -OneFile $one -Manifest research_pass28/artifacts/ci-manifest.json -Report research_pass28/artifacts/ci-smoke.json
if ($LASTEXITCODE) { throw 'Standalone smoke check failed' }
$smoke=Get-Content research_pass28/artifacts/ci-smoke.json -Raw | ConvertFrom-Json
$implementation=Get-FileHash research_pass28/artifacts/release-source/IMPLEMENTATION.json -Algorithm SHA256
$summary=@{runner=$env:RUNNER_OS;image=$env:ImageOS;image_version=$env:ImageVersion;source_commit=$env:GITHUB_SHA;status=$smoke.status;checks=$smoke.checks.Count;os=$smoke.os;admin=$smoke.admin;python=(& $python --version);implementation_sha256=$implementation.Hash.ToLowerInvariant();runtime_exact=$true;source_wheel_installed=$true;clean_consumer_machine_proven=$false;artifacts_uploaded=$false}
Write-Output ('PBC28_RESULT '+($summary | ConvertTo-Json -Compress))
