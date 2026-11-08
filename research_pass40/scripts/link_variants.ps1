$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$Build = Join-Path $Root 'research_pass35\build\libavif-reference'
$Pass = Join-Path $Root 'research_pass40\outputs'
$Base = Join-Path $Root 'research_pass35\install\profile\lib\libaom.a'
$Runtime = Join-Path $Root 'research_pass35\outputs\binaries\reference'
foreach ($name in @('control', 'experiment', 'experiment-qmap')) {
    $lib = Join-Path $Pass "install\$name\lib\libaom.a"
    $outdir = Join-Path $Pass "binaries\$name"
    New-Item -ItemType Directory -Force $outdir | Out-Null
    Copy-Item (Join-Path $Runtime 'libgcc_s_seh-1.dll') $outdir -Force
    Copy-Item (Join-Path $Runtime 'libwinpthread-1.dll') $outdir -Force
    $cmd = (ninja -C $Build -t commands avifenc.exe | Select-Object -Last 1)
    if ($LASTEXITCODE -ne 0) { throw 'Could not read the pinned avifenc link command.' }
    $cmd = $cmd.Replace($Base.Replace('\', '/'), ('"' + $lib.Replace('\', '/') + '"'))
    $cmd = $cmd.Replace('-o avifenc.exe', ('-o "' + (Join-Path $outdir 'avifenc.exe').Replace('\', '/') + '"'))
    $cmd = $cmd.Replace('-Wl,--out-implib,libavifenc.dll.a', ('-Wl,--out-implib,"' + (Join-Path $outdir 'libavifenc.dll.a').Replace('\', '/') + '"'))
    Push-Location $Build
    try { cmd.exe /C $cmd; if ($LASTEXITCODE -ne 0) { throw "Link failed: $name" } }
    finally { Pop-Location }
    Write-Output "$name $(Get-FileHash (Join-Path $outdir 'avifenc.exe') -Algorithm SHA256 | Select-Object -ExpandProperty Hash)"
}
