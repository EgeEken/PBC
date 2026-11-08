$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$Build = Join-Path $Root 'research_pass35\build\libavif-reference'
$Pass35 = Join-Path $Root 'research_pass35'
$Out = Join-Path $Root 'research_pass41\outputs\binaries'
New-Item -ItemType Directory -Force $Out | Out-Null
$gcc = 'C:\Strawberry\c\bin\gcc.exe'
$gxx = 'C:\Strawberry\c\bin\g++.exe'
$src = Join-Path $PSScriptRoot 'plane_probe.c'
$obj = Join-Path $Out 'plane_probe.obj'
$exe = Join-Path $Out 'plane_probe.exe'
$include = Join-Path $Pass35 'sources\libavif\include'
& $gcc '-O3' '-DNDEBUG' '-std=gnu99' '-Wall' '-Wextra' "-I$include" "-I$(Join-Path $Pass35 'sources\libavif\apps\shared')" '-c' $src '-o' $obj
if ($LASTEXITCODE -ne 0) { throw 'plane_probe compile failed' }
Push-Location $Build
try {
  $aom = Join-Path $Pass35 'install\reference\lib\libaom.a'
  if (-not (Test-Path $aom)) { $aom = Join-Path $Pass35 'install\profile\lib\libaom.a' }
  $args = @('-O3','-DNDEBUG',$obj,'-o',$exe,'libavif_apps.a','libavif_internal.a','_deps/libyuv-build/libyuv.a','_deps/libwebp/libsharpyuv.a','-lm',$aom,'_deps/libpng/libpng16.a','_deps/zlib/libzlibstatic.a','libjpeg/src/libjpeg-build/libjpeg.a','-lkernel32','-luser32','-lgdi32','-lwinspool','-lshell32','-lole32','-loleaut32','-luuid','-lcomdlg32','-ladvapi32')
  & $gxx @args
  if ($LASTEXITCODE -ne 0) { throw 'plane_probe link failed' }
}
finally { Pop-Location }
Get-FileHash $exe -Algorithm SHA256 | Select-Object Hash
